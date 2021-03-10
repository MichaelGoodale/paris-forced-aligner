import io
import os
import tarfile
from zipfile import ZipFile
import random
from typing import List, Optional
from functools import partial 

from torch.multiprocessing import Pool, cpu_count
import buckeye
import youtube_dl
import webvtt
import torchaudio
import tempfile

from paris_forced_aligner.audio_data import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone


class CorpusClass():
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, raise_on_oov: bool = True, return_gold_labels: bool = False):
        self.corpus_path: str = os.path.expanduser(corpus_path)
        self.pronunciation_dictionary: PronunciationDictionary = pronunciation_dictionary
        self.raise_on_oov = raise_on_oov
        self.return_gold_labels = return_gold_labels

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files(self.return_gold_labels))

class YoutubeCorpus(CorpusClass):

    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, language: str = 'en', audio_directory: Optional[str] = None, save_wavs: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary, False, False)
        self.youtube_files = []
        self.save_wavs = save_wavs

        if audio_directory is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.dir = self._temp_dir.name
        else:
            self.dir = audio_directory
        self.language = language

        if os.path.exists(corpus_path):
            with open(corpus_path, 'r') as f:
                for line in f:
                    self.youtube_files.append(line.strip())
        else:
            self.youtube_files.append(corpus_path)

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("Youtube videos do not have gold standard phonemic labels")
        YDL_OPTS = {
            'format': 'bestaudio/best',
            'writeautomaticsub': True,
            'outtmpl': self.dir+'/%(uploader_id)s.%(id)s.%(ext)s',
            'subtitleslangs':[self.language],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }
        with youtube_dl.YoutubeDL(YDL_OPTS) as ydl:
            ydl.download(self.youtube_files)

        sub_file_ending = ".{}.vtt".format(self.language)

        for subtitle_file in filter(lambda x: x.endswith(sub_file_ending), os.listdir(self.dir)):
            subtitle_file = os.path.join(self.dir, subtitle_file)

            audio = subtitle_file.replace(sub_file_ending, ".wav")
            captions = webvtt.read(os.path.join(self.dir, subtitle_file)).captions
            wav, sr = torchaudio.load(os.path.join(self.dir, audio))

            if self.save_wavs:
                idx_starts_ends: List[Tuple[int, int, int]] = []

            for i, (cap_time, cap_string) in enumerate(zip(captions[::2], captions[1::2])):
                cap_name = subtitle_file.replace(sub_file_ending, f"_{i}")
                transcription = cap_string.text.strip().upper()
                if transcription == '':
                    continue
                start = int(cap_time.start_in_seconds * sr)
                end = int(cap_time.end_in_seconds * sr)

                if self.save_wavs:
                    idx_starts_ends.append((i, int(start / sr  * 16000), int(end / sr * 16000)))
                yield AudioFile(cap_name, transcription, self.pronunciation_dictionary, wavobj=(wav[:, start:end], sr), raise_on_oov=self.raise_on_oov)

            if self.save_wavs:
                with open(subtitle_file.replace(sub_file_ending, '.txt'), 'w') as f:
                    for i, start, end in idx_starts_ends:
                        f.write(f"{i} {start} {end}\n")

    def stitch_youtube_utterances(self, audio_file_name: str, utterances: List[Utterance]):
        index_file = audio_file_name + '.txt'
        big_u_data = []
        with open(index_file, 'r') as f:
            for utterance, line in zip(utterances, f):
                i, start, end = line.strip().split(' ')
                utterance.offset(int(start))
                big_u_data += utterance.data
        return Utterance(big_u_data)

    def cleanup_and_recreate(self):
        self.cleanup()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.dir = self._temp_dir.name

    def cleanup(self):
        self._temp_dir.cleanup()


class LibrispeechCorpus(CorpusClass):
    def __init__(self, corpus_path: str, raise_on_oov: bool = False):
        super().__init__(corpus_path, LibrispeechDictionary(), raise_on_oov, False)

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("Librispeech does not have gold standard phonemic labels")

        found_files = False
        for d, s_d, files in os.walk(self.corpus_path):
            for t_file in filter(lambda x: x.endswith('.trans.txt'), files):
                with open(os.path.join(d, t_file)) as f:
                    found_files = True
                    for line in f:
                        filename, transcription = line.strip().split(' ', 1)
                        filename = os.path.join(d, f"{filename}.flac")
                        yield AudioFile(filename, transcription, self.pronunciation_dictionary, raise_on_oov=self.raise_on_oov)

        if not found_files:
            raise IOError(f"{self.corpus_path} has no files!")

class BuckeyeCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False, split_time: int = 10):
        super().__init__(corpus_path, pronunciation_dictionary, True, return_gold_labels)

    def _extract_from_zip(self, zip_dir, track_name, speaker_name):
        with ZipFile(zip_dir.open(f"{speaker_name}/{track_name}.zip")) as sound_dir:
            with sound_dir.open(f"{track_name}.wav") as wav_file:
                with tempfile.NamedTemporaryFile() as f:
                    f.write(wav_file.read())
                    wav, sr = torchaudio.load(f.name)
        return wav, sr

    def convert_to_arpabet(word, word_buckeye, sr=16000):
        '''Handles converting from seconds to frames and also handles any phoneme differences'''
        if isinstance(word, Silence):
            word = Silence(int(word.start * sr), int(word.end *sr))
        else:
            word.label = word.label.strip().replace('-', 'X').replace(' ', 'X').upper() 
            for i, phone in enumerate(word.phones):
                phone.start = int(phone.start * sr)
                phone.end = int(phone.end * sr)
                if phone.label is not None:
                    phone.label = phone.label.upper()
                    if phone.label == "NX":
                        phone.label = "N"
                    elif phone.label == "DX":
                        if i < len(word_buckeye.phonemic):
                            phone.label = word_buckeye.phonemic[i].upper()
                        if phone.label != 'T' or phone.label != 'D':
                            phone.label = 'T' #Good enough!
                    elif phone.label == "EN":
                        phone.label = "N"
                    elif phone.label == "EM":
                        phone.label = "M"
                    elif phone.label == "EL":
                        phone.label = "L"
                    elif phone.label == "TQ": #Hope these aren't epenthetic
                        phone.label = "T"
                    elif len(phone.label) == 3 and phone.label.endswith("N"):
                        phone.label = phone.label[:2] #No nasal phone >:(
                    elif len(phone.label) >= 3:
                        phone.label = PronunciationDictionary.silence
                else:
                    phone.label = PronunciationDictionary.silence
        return word

    def merge_silences(words):
        last = None
        new_words = []
        for word in words:
            if isinstance(word, Silence):
                if last is not None:
                    last.end = word.end
                else:
                    last = word
            else:
                if last is not None:
                    new_words.append(last)
                    last = None
                new_words.append(word)
        if last is not None:
            new_words.append(last)
        return new_words

    def extract_files(self, return_gold_labels):
        for d, s_d, files in os.walk(self.corpus_path):
            for f in filter(lambda x: x.endswith('.zip'), files):
                zip_path = os.path.join(d, f)
                speaker = buckeye.Speaker.from_zip(zip_path)
                with ZipFile(zip_path) as zip_dir:
                    for track in speaker:
                        start = 0.0
                        paris_words = []
                        wav, sr = self._extract_from_zip(zip_dir, track.name, speaker.name)

                        for word in track.words:
                            if isinstance(word, buckeye.containers.Pause):
                                paris_word = Silence(word.beg, word.end)
                            else:
                                paris_word = Word([Phone(p.seg, p.beg, p.end) for p in word.phones], word.orthography)
                                #TODO: Add dynamic dictionary words w/ correct pronunciation (maybe)
                            paris_words.append(BuckeyeCorpus.convert_to_arpabet(paris_word, word))

                            if word.beg - start > 10.0 and isinstance(word, buckeye.containers.Pause):
                                utterance = Utterance(BuckeyeCorpus.merge_silences(paris_words))
                                if utterance.words != []:
                                    self.pronunciation_dictionary.add_words_from_utterance(utterance)
                                    audio = AudioFile(track.name, utterance.transcription, self.pronunciation_dictionary, wavobj=(wav[:, utterance.start:utterance.end], 16000), raise_on_oov=self.raise_on_oov)
                                    if return_gold_labels:
                                        yield audio, utterance
                                    else:
                                        yield audio
                                paris_words = []
                                start = word.beg

