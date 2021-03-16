import torch
import torchaudio

from torchaudio.transforms import MFCC, Resample
from torch import Tensor

import os
import re
import urllib.request
from typing import Callable, Union, BinaryIO, Optional, Mapping, List, Set, Tuple

from paris_forced_aligner.utils import data_directory, download_data_file
from paris_forced_aligner.ipa_data import arpabet_to_ipa
from paris_forced_aligner.phonological import Utterance

class OutOfVocabularyException(Exception):
    """Raise for my specific kind of exception"""

class PronunciationDictionary:
    silence = "<SIL>"
    OOV = "<OOV>"

    def __init__(self):
        self.lexicon: Mapping[str, List[str]] = {}
        self.phonemic_inventory: Set[str] = set([PronunciationDictionary.silence, PronunciationDictionary.OOV])
        self.phone_to_phoneme: Mapping[str, str] = {}
        self.load_lexicon()
        self.phonemic_mapping: Mapping[str, int] = {phone: i+1 for i, phone in enumerate(sorted(self.phonemic_inventory))}
        self.index_mapping: Mapping[int, str] = {v:k for k, v in self.phonemic_mapping.items()}

    def load_lexicon(self):
        '''Function to load lexicon and phonemic inventory'''
        raise NotImplementedError("Lexicon loading must be defined in a subclass of PronunciationDictionary")

    def vocab_size(self) -> int:
        return len(self.phonemic_inventory) + 1

    def index_to_phone(self, idx: int) -> str:
        return self.index_mapping[idx]

    def add_words_from_utterance(self, utterance: Utterance):
        for word in utterance.words:
            if word.label not in self.lexicon:
                self.lexicon[word.label] = [p.label for p in word.phones]


class LibrispeechDictionary(PronunciationDictionary):

    def __init__(self, remove_stress=True):
        self.remove_stress = remove_stress
        super().__init__()

    def load_lexicon(self):
        self.phone_to_phoneme: Mapping[str, str] = arpabet_to_ipa

        lexicon_path = os.path.join(data_directory, 'librispeech-lexicon.txt')
        if not os.path.exists(lexicon_path):
            download_data_file('https://www.openslr.org/resources/11/librispeech-lexicon.txt', lexicon_path)

        with open(lexicon_path) as f:
            for line in f:
                word, pronunciation = re.split(r'\s+', line.strip(), maxsplit=1)
                if self.remove_stress:
                    pronunciation = re.sub(r'[0-9]+', '', pronunciation)

                self.lexicon[word] = pronunciation.split(' ')
                for phone in self.lexicon[word]:
                    self.phonemic_inventory.add(phone)

class TSVDictionary(PronunciationDictionary):
    def __init__(self, lexicon_path: str, seperator: str='\t', phone_to_phoneme: Optional[Mapping[str, str]] = None):
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
    def __init__(self, filename: str, transcription: str, pronunciation_dictionary: PronunciationDictionary,
            fileobj: Optional[BinaryIO] = None,
            wavobj: Optional[Tuple[Tensor, int]] = None,
            raise_on_oov: bool=True):

        self.filename = filename
        self.pronunciation_dictionary = pronunciation_dictionary
        self.load_audio(fileobj, wavobj)

        #Treat words with hyphens as two words.
        transcription = transcription.replace('-', ' ')

        self.transcription, self.tensor_transcription = self.get_phone_transcription(transcription, raise_on_oov)
        self.words = self.get_word_transcription(transcription)

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

    def get_phone_transcription(self, transcription: str, raise_on_oov: bool=True) -> Tuple[List[str], Tensor]:
        new_transcription: List[str] = [PronunciationDictionary.silence]

        for word in transcription.split(' '):
            try:
                new_transcription += self.pronunciation_dictionary.lexicon[word] 
            except KeyError:
                if raise_on_oov:
                    raise OutOfVocabularyException(f"{word} is not present in the librispeech lexicon")
                else:
                    new_transcription.append(self.pronunciation_dictionary.OOV)
            new_transcription.append(PronunciationDictionary.silence)

        new_transcription_tensor = torch.tensor([self.pronunciation_dictionary.phonemic_mapping[x] for x in new_transcription])
        return new_transcription, new_transcription_tensor

    def get_word_transcription(self, transcription: str) -> List[str]:
        return transcription.split(' ')

    def move_to_device(self, device:str):
        self.wav = self.wav.to(device)
        self.tensor_transcription = self.tensor_transcription.to(device)
