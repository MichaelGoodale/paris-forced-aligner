import os
import random
from typing import List, Optional
from itertools import chain

from paris_forced_aligner.pronunciation import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone
from paris_forced_aligner.corpus import CorpusClass

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

