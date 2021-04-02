import os
import re
from typing import Union, BinaryIO, Optional, Mapping, List, Set, Tuple

import torch
import torchaudio

from torchaudio.transforms import Resample
from torch import Tensor
from num2words import num2words

from paris_forced_aligner.utils import data_directory, download_data_file
from paris_forced_aligner.ipa_data import arpabet_to_ipa
from paris_forced_aligner.phonological import Utterance

class OutOfVocabularyException(Exception):
    """Raise for my specific kind of exception"""

class PronunciationDictionary:
    silence = "<SIL>"
    OOV = "<OOV>"

    def __init__(self, use_G2P:bool = False, lang='en'):
        self.lexicon: Mapping[str, List[str]] = {}
        self.phonemic_inventory: Set[str] = set([PronunciationDictionary.silence])
        self.phone_to_phoneme: Mapping[str, str] = {}
        self.load_lexicon()
        self.phonemic_mapping: Mapping[str, int] = {phone: i+1 for i, phone in \
                                            enumerate(sorted(self.phonemic_inventory))}
        self.index_mapping: Mapping[int, str] = {v:k for k, v in self.phonemic_mapping.items()}
        self.use_G2P = use_G2P
        self.lang = lang

    def load_lexicon(self):
        '''Function to load lexicon and phonemic inventory'''
        raise NotImplementedError("Lexicon loading must be defined in a subclass of PronunciationDictionary")

    def vocab_size(self) -> int:
        return len(self.phonemic_inventory) + 1

    def index_to_phone(self, idx: int) -> str:
        return self.index_mapping[idx]

    def add_G2P_spelling(self, word: str):
        raise NotImplementedError("G2P models not yet available")

    def add_words_from_utterance(self, utterance: Utterance):
        for word in utterance.words:
            if word.label not in self.lexicon:
                self.lexicon[word.label] = [p.label for p in word.phones]

    def split_sentence(self, sentence:str) -> List[str]:
        return_sentence = []
        for word in sentence.strip('-').split():
            if word.isdigit():
                word = num2words(int(word), lang=self.lang).strip('-').split(' ')
                return_sentence += word.upper()
            else:
                return_sentence.append(word.upper())
        return return_sentence

    def spell_sentence(self, sentence: str, return_words: bool = True):
        sentence = self.split_sentence(sentence)
        spelling: List[str] = [PronunciationDictionary.silence]

        for word in sentence:
            if word not in self.lexicon:
                if not self.use_G2P:
                    raise OutOfVocabularyException(f"{word} is not present in the lexicon")
                self.add_G2P_spelling(word)

            spelling += self.lexicon[word]
            spelling.append(PronunciationDictionary.silence)

        if return_words:
            return spelling, sentence
        return spelling

class LibrispeechDictionary(PronunciationDictionary):
    LIBRISPEECH_URL = 'https://www.openslr.org/resources/11/librispeech-lexicon.txt'
    def __init__(self, remove_stress=False):
        self.remove_stress = remove_stress
        super().__init__()

    def load_lexicon(self):
        self.phone_to_phoneme: Mapping[str, str] = arpabet_to_ipa

        lexicon_path = os.path.join(data_directory, 'librispeech-lexicon.txt')
        if not os.path.exists(lexicon_path):
            download_data_file(LibrispeechDictionary.LIBRISPEECH_URL, lexicon_path)

        with open(lexicon_path) as f:
            for line in f:
                word, pronunciation = re.split(r'\s+', line.strip(), maxsplit=1)
                if self.remove_stress:
                    pronunciation = re.sub(r'[1-9]+', '', pronunciation)
                    pronunciation = re.sub(r' \w+0', ' AX', pronunciation)

                self.lexicon[word] = pronunciation.split(' ')
                for phone in self.lexicon[word]:
                    self.phonemic_inventory.add(phone)

class TSVDictionary(PronunciationDictionary):
    def __init__(self, lexicon_path: str, seperator: str='\t', \
            phone_to_phoneme: Optional[Mapping[str, str]] = None):
        super().__init__()
        self.lexicon_path = lexicon_path
        self.seperator = seperator
        if phone_to_phoneme is not None:
            self.phone_to_phoneme = phone_to_phoneme
            self.already_ipa = False
        else:
            self.phone_to_phoneme: Mapping[str, str] = {}
            self.already_ipa = True


    def load_lexicon(self):
        with open(self.lexicon_path) as f:
            for line in f:
                word, pronunciation = line.strip().split(self.seperator, 1)
                self.lexicon[word] = pronunciation.split(' ')
                for phone in self.lexicon[word]:
                    self.phonemic_inventory.add(phone)
                    if self.already_ipa:
                        self.phone_to_phoneme[phone] = phone

class AudioFile:
    def __init__(self, filename: str, transcription: str,
            pronunciation_dictionary: PronunciationDictionary,
            fileobj: Optional[BinaryIO] = None,
            wavobj: Optional[Tuple[Tensor, int]] = None,
            offset: int = 0):

        self.filename = filename
        self.pronunciation_dictionary = pronunciation_dictionary
        self.load_audio(fileobj, wavobj)

        self.transcription, self.words = pronunciation_dictionary.spell_sentence(transcription, return_words=True)
        self.tensor_transcription = torch.tensor([self.pronunciation_dictionary.phonemic_mapping[x] \
                                                    for x in self.transcription])

        self.offset = offset

    def load_audio(self, fileobj: Optional[BinaryIO] = None, wavobj = None):
        if fileobj is not None:
            self.wav, sr = torchaudio.load(fileobj)
        elif wavobj is not None:
            self.wav, sr = wavobj
        else:
            self.wav, sr = torchaudio.load(self.filename)

        if self.wav.shape[0] != 1:
            self.wav = torch.mean(self.wav, dim=0).unsqueeze(0)

        if sr != 16000:
            self.wav = Resample(sr, 16000)(self.wav)

    def move_to_device(self, device:str):
        self.wav = self.wav.to(device)
        self.tensor_transcription = self.tensor_transcription.to(device)
