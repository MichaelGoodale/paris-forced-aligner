import io
import os
import re
import tarfile
from zipfile import ZipFile
import random
from typing import List, Optional
from functools import partial 
from itertools import chain

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
    def __init__(self, corpus_path: str, raise_on_oov: bool = True):
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
                        try:
                            yield AudioFile(filename, transcription, self.pronunciation_dictionary, raise_on_oov=self.raise_on_oov)
                        except OutOfVocabularyException:
                            continue

        if not found_files:
            raise IOError(f"{self.corpus_path} has no files!")

class TimitCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False, split: str = "train", stress_labeled: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary, True, return_gold_labels)
        if split not in ["train", "test", "both"]:
            raise NotImplementedError("TIMIT has only a test and train split, please set `split` in ['train', 'test', 'both']")
        self.split = split
        self.stress_labeled = stress_labeled

    def get_utterance(self, phn_f, word_f):
        word_timing = []
        with open(word_f) as f:
            for line in f:
                start, end, label = line.strip().split()
                start = int(start)
                end = int(end)
                word_timing.append((label.upper(), start, end))

        with open(phn_f) as f:
            data = []
            word = []
            for line in f:
                start, end, label = line.strip().split()
                start = int(start)
                end = int(end)
                print(self.pronunciation_dictionary.phonemic_inventory)
                if label in ["pau", "epi", "h#"]:
                    data.append(Silence(start, end))
                else:
                    if not self.stress_labeled:
                        label = re.sub(r'[0-9]+', '', label)

                    if label.endswith('cl'):
                        label = label[0]
                    elif label == "hv":
                        label = "HH"
                    elif label == "ax-h":
                        label = "AX"
                    elif label == "axr":
                        label = "R"
                    elif label == "ix":
                        label = "IH"
                    elif label == "ux":
                        label = "UW"
                    elif label == "el":
                        label = "L"
                    if label == "nx":
                        label = "N"
                    elif label == "dx":
                        try:
                            spelling = self.pronunciation_dictionary.lexicon[word_timing[0][0]]
                            for x in spelling:
                                if x in ["T", "D"]:
                                    label = x
                                    break #Mediocre way of checking but should work ok
                        except KeyError:
                            continue 
                        if label not in ["T", "D"]:
                            label = "T"
                    elif label == "q":
                        label = "T" 
                    elif label == "eng":
                        label = "NG"
                    label = label.upper()

                    if len(word) > 1 and word[-1].label == label:
                        word[-1].end = end
                    else:
                        word.append(Phone(label, start, end))

                    if end >= word_timing[0][2]:
                        data.append(Word(word, word_timing[0][0]))
                        word_timing = word_timing[1:]
                        word = []
        return Utterance(data)

    def extract_files(self, return_gold_labels):
        corpus_paths = [os.path.join(self.corpus_path, x) for x in ["train", "test"]]
        if self.split == "train":
            corpus_paths = [corpus_paths[0]]
        elif self.split == 'test':
            corpus_paths = [corpus_paths[1]]

        for (d, s_d, files) in chain(*[os.walk(x) for x in corpus_paths]):
            for f in filter(lambda x: x.endswith('.wav'), files):
                wav_f = os.path.join(d, f)
                phn_f = wav_f.replace('.wav', '.phn')
                word_f = wav_f.replace('.wav', '.wrd')
                utterance = self.get_utterance(phn_f, word_f)
                self.pronunciation_dictionary.add_words_from_utterance(utterance)
                audio = AudioFile(wav_f, utterance.transcription, self.pronunciation_dictionary, raise_on_oov=self.raise_on_oov)

                if return_gold_labels:
                    yield audio, utterance
                else:
                    yield audio


class BuckeyeCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False):
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
        word.label = word.label.strip().replace('-', 'X').replace(' ', 'X').upper() 
        for i, phone in enumerate(word.phones):
            phone.start = int(phone.start * sr)
            phone.end = int(phone.end * sr)

            if phone.label is not None:
                phone.label = phone.label.upper()
                if phone.label == "NX":
                    phone.label = "N"
                elif phone.label == "DX":
                    if word_buckeye.phonemic is not None and i < len(word_buckeye.phonemic):
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

    def extract_files(self, return_gold_labels):
        for d, s_d, files in os.walk(self.corpus_path):
            speaker_files = list(filter(lambda x: x.endswith('.zip'), files))
            random.shuffle(speaker_files)
            for f in speaker_files:
                zip_path = os.path.join(d, f)
                speaker = buckeye.Speaker.from_zip(zip_path)

                with ZipFile(zip_path) as zip_dir:
                    for track in speaker:
                        paris_words = []
                        wav, sr = self._extract_from_zip(zip_dir, track.name, speaker.name)

                        for word in track.words:
                            if isinstance(word, buckeye.containers.Pause):
                                if len(paris_words) > 1 and isinstance(paris_words[-1], Silence):
                                    paris_words[-1].end = int(word.end * 16000)
                                else:
                                    paris_words.append(Silence(int(word.beg * 16000), int(word.end * 16000)))
                            else:
                                paris_word = Word([Phone(p.seg, p.beg, p.end) for p in word.phones], word.orthography)
                                #TODO: Add dynamic dictionary words w/ correct pronunciation (maybe)
                                paris_words.append(BuckeyeCorpus.convert_to_arpabet(paris_word, word))

                            if len(paris_words) > 2 and isinstance(paris_words[-1], Silence) and paris_words[-1].duration > int(0.150*16000):
                                utterance = Utterance(paris_words[:-1])

                                if utterance.words != []:
                                    self.pronunciation_dictionary.add_words_from_utterance(utterance)
                                    audio = AudioFile(track.name, utterance.transcription, self.pronunciation_dictionary, wavobj=(wav[:, utterance.start:utterance.end], 16000), raise_on_oov=self.raise_on_oov)
                                    if return_gold_labels:
                                        utt_start = utterance.start 
                                        for base in utterance.base_units:
                                            base.start -= utt_start
                                            base.end -= utt_start
                                        yield audio, utterance
                                    else:
                                        yield audio
                                paris_words = []

