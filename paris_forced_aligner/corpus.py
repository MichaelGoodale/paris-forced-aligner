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
from tqdm.autonotebook import tqdm
import webvtt
import torchaudio
import tempfile

from paris_forced_aligner.audio_data import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone


class CorpusClass():
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False):
        self.corpus_path: str = os.path.expanduser(corpus_path)
        self.pronunciation_dictionary: PronunciationDictionary = pronunciation_dictionary
        self.return_gold_labels = return_gold_labels

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files(self.return_gold_labels))

    def __len__(self):
        raise NotImplementedError("CorpusClass has no __len__ implemented by default")

class YoutubeCorpus(CorpusClass):

    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, language: str = 'en', audio_directory: Optional[str] = None, save_wavs: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary)
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
                start = int(cap_time.start_in_seconds * 16000)
                end = int(cap_time.end_in_seconds * 16000)

                if self.save_wavs:
                    idx_starts_ends.append((i, int(start / sr  * 16000), int(end / sr * 16000)))
                try:
                    yield AudioFile(cap_name, transcription, self.pronunciation_dictionary, wavobj=(wav[:, start:end], sr), offset=start)
                except OutOfVocabularyException as e:
                    print(e)
                    continue

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
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary = None):
        if pronunciation_dictionary is None:
            pronunciation_dictionary = LibrispeechDictionary()
        super().__init__(corpus_path, pronunciation_dictionary)

        found_files = False
        self.all_files = []
        for d, s_d, files in os.walk(self.corpus_path):
            for t_file in filter(lambda x: x.endswith('.trans.txt'), files):
                with open(os.path.join(d, t_file)) as f:
                    found_files = True
                    for line in f:
                        filename, transcription = line.strip().split(' ', 1)
                        filename = os.path.join(d, f"{filename}.flac")
                        self.all_files.append((filename, transcription))

        if not found_files:
            raise IOError(f"{self.corpus_path} has no files!")

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("Librispeech does not have gold standard phonemic labels")

        for filename, transcription in self.all_files:
            try:
                yield AudioFile(filename, transcription, self.pronunciation_dictionary)
            except OutOfVocabularyException:
                continue

    def __len__(self):
        return len(self.all_files)

class TIMITCorpus(CorpusClass):

    vowels = ['UH', 'AXR', 'AO', 'IX', 'UW', 'OY', 'AX-H', 'IY',  'AY', 'AA', 'AE', 'AW', 'OW', 'UX', 'AX', 'AH', 'EY', 'IH']

    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False, split: str = "train", val_split=0.75, untranscribed_audio=True, vowel_consonsant_transcription=False):
        super().__init__(corpus_path, pronunciation_dictionary, return_gold_labels)
        if split not in ["train", "test", "val", "both"]:
            raise NotImplementedError("TIMIT has only a test, train and val split, please set `split` in ['train', 'test', 'val' 'both']")
        self.split = split
        self.untranscribed_audio = untranscribed_audio
        self.vowel_consonsant_transcription = vowel_consonsant_transcription

        corpus_paths = [os.path.join(self.corpus_path, x) for x in ["train", "test"]]
        if self.split == "train":
            corpus_paths = [corpus_paths[0]]
        elif self.split == 'test':
            corpus_paths = [corpus_paths[1]]
        elif self.split == 'val':
            corpus_paths = [corpus_paths[1]]

        self.all_files = []
        for (d, s_d, files) in chain(*[os.walk(x) for x in corpus_paths]):
            for f in filter(lambda x: x.endswith('.wav'), files):
                wav_f = os.path.join(d, f)
                phn_f = wav_f.replace('.wav', '.phn')
                word_f = wav_f.replace('.wav', '.wrd')
                self.all_files.append((wav_f, phn_f, word_f))

        if self.split in ['val', 'train']:
            random.Random(1337).shuffle(self.all_files)
            if self.split == 'val':
                self.all_files = self.all_files[int(val_split*len(self.all_files)):]
            else:
                self.all_files = self.all_files[:int(val_split*len(self.all_files))]

    def relabel_word(self, word: List[str], word_label: str) -> List[str]:
        real_spelling = self.pronunciation_dictionary.spelling(word_label)
        if len(word) == len(real_spelling):
            for i, phone in enumerate(word):
                phone.label = real_spelling[i]
        else:
            real_vowels = [w for w in real_spelling if w[-1] in ['1', '2', '0']]
            if len([w for w in word if w.label in TIMITCorpus.vowels]) == len(real_vowels):
                vowel_idx = 0
                for i, phone in enumerate(word):
                    if phone.label in TIMITCorpus.vowels:
                        phone.label = real_vowels[vowel_idx]
                        vowel_idx += 1
        return word

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
                if label in ["pau", "epi", "h#"]:
                    data.append(Silence(start, end))
                else:
                    if label.endswith('cl'):
                        label = label[0]

                    label = label.upper()

                    if len(word) >= 1 and word[-1].label == label and label in ["B", "P", "D", "T", "G", "K"]:
                        word[-1].end = end
                    else:
                        word.append(Phone(label, start, end))

                    if len(word_timing) >= 1 and end >= word_timing[0][2]:
                        word_label = word_timing[0][0]
                        word = self.relabel_word(word, word_label)
                        word = Word(word, word_timing[0][0])
                        data.append(word)
                        word_timing = word_timing[1:]
                        word = []
        return Utterance(data)

    def extract_files(self, return_gold_labels):
        for wav_f, phn_f, word_f in self.all_files:
            utterance = self.get_utterance(phn_f, word_f)
            if self.untranscribed_audio:
                transcription = ""
            else:
                transcription = utterance.transcription

            if self.vowel_consonsant_transcription:
                for phone in utterance.base_units:
                    if phone.label in TIMITCorpus.vowels:
                        phone.label = "V"
                    elif phone.label != '<SIL>':
                        phone.label = "C"

            audio = AudioFile(wav_f, transcription, self.pronunciation_dictionary)

            if return_gold_labels:
                yield audio, utterance
            else:
                yield audio

    def __len__(self):
        return len(self.all_files)

class BuckeyeCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False, relabel: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary, return_gold_labels)
        self.relabel = relabel

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
                                if len(paris_words) >= 1 and isinstance(paris_words[-1], Silence):
                                    paris_words[-1].end = int(word.end * 16000)
                                else:
                                    paris_words.append(Silence(int(word.beg * 16000), int(word.end * 16000)))
                            else:
                                paris_word = Word([Phone(p.seg, p.beg, p.end) for p in word.phones], word.orthography)
                                #TODO: Add dynamic dictionary words w/ correct pronunciation (maybe)
                                if word.phones != []:
                                    paris_words.append(BuckeyeCorpus.convert_to_arpabet(paris_word, word))

                            if len(paris_words) >= 2 and isinstance(paris_words[-1], Silence) and paris_words[-1].duration > int(0.150*16000):
                                utterance = Utterance(paris_words[:-1])

                                if utterance.words != []:
                                    try:
                                        audio = AudioFile(track.name, utterance.transcription, self.pronunciation_dictionary, wavobj=(wav[:, utterance.start:utterance.end], 16000))
                                    except OutOfVocabularyException:
                                        continue 

                                    if return_gold_labels:
                                        utt_start = utterance.start 
                                        for base in utterance.base_units:
                                            base.start -= utt_start
                                            base.end -= utt_start

                                        if self.relabel:
                                            for word in utterance.words:
                                                lexical_word = self.pronunciation_dictionary.lexicon[word.label]
                                                for p, lexical_phone in zip(word.phones, lexical_phone):
                                                    p.label = lexical_phone
                                            for base in utterance.base_units:
                                                if base.label not in self.pronunciation_dictionary.phonemic_inventory:
                                                    continue

                                        yield audio, utterance
                                    else:
                                        yield audio
                                paris_words = []
