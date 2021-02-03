import torch
import torchaudio
torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
torchaudio.set_audio_backend("soundfile")

from torchaudio.transforms import MFCC, Resample
from torch import Tensor

import os
import re
import urllib.request
from typing import Callable, Union, BinaryIO, Optional, Mapping, List, Set, Tuple

from paris_forced_aligner.utils import arpabet_to_ipa, data_directory, download_data_file

class OutOfVocabularyException(Exception):
    """Raise for my specific kind of exception"""

class AudioFile:
    def __init__(self, filename:str, transcription: str,
            fileobj: Optional[BinaryIO] = None,
            wavobj: Optional[Tuple[Tensor, int]] = None,
            extract_features: bool = False,
            feature_extractor: Callable[[Tensor], Tensor] = MFCC()):

        self.filename = filename
        self.load_audio(fileobj, wavobj)

        if extract_features:
            self.features = feature_extractor(self.wav)
        else:
            self.features = self.wav

        self.transcription, self.tensor_transcription = self.get_phone_transcription(transcription)
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

    def get_phone_transcription(self, transcription: str) -> Tuple[List[str], Tensor]:
        raise NotImplementedError("get_phone_transcription is not implemented in base class")

    def get_word_transcription(self, transcription: str) -> List[str]:
        raise NotImplementedError("get_word_transcription is not implemented in base class")

    def index_to_phone(self, idx: int) -> str:
        raise NotImplementedError("index_to_phone is not implemented in base class")

    def vocab_size() -> int:
        raise NotImplementedError("vocab_size is not implemented in base class")

class LibrispeechFile(AudioFile):
    lexicon: Mapping[str, List[str]] = {}
    phonemic_inventory: Set[str] = set(['<SIL>'])
    phone_to_phoneme: Mapping[str, str] = arpabet_to_ipa

    lexicon_path = os.path.join(data_directory, 'librispeech-lexicon.txt')
    if not os.path.exists(lexicon_path):
        download_data_file('https://www.openslr.org/resources/11/librispeech-lexicon.txt', lexicon_path)

    with open(lexicon_path) as f:
        for line in f:
            word, pronunciation = re.split(r'\s+', line.strip(), maxsplit=1)
            lexicon[word] = pronunciation.split(' ')
            for phone in lexicon[word]:
                phonemic_inventory.add(phone)

    phonemic_mapping: Mapping[str, int] = {phone: i+1 for i, phone in enumerate(sorted(phonemic_inventory))}
    index_mapping: Mapping[int, str] = {v:k for k, v in phonemic_mapping.items()}

    def vocab_size() -> int:
        return len(LibrispeechFile.phonemic_inventory) + 1

    def get_word_transcription(self, transcription: str) -> List[str]:
        return transcription.split(' ')

    def get_phone_transcription(self, transcription: str) -> Tuple[List[str], Tensor]:
        new_transcription: List[str] = ['<SIL>']

        for word in transcription.split(' '):
            try:
                new_transcription += LibrispeechFile.lexicon[word] 
            except KeyError:
                raise OutOfVocabularyException(f"{word} is not present in the librispeech lexicon")
            new_transcription.append('<SIL>')

        new_transcription_tensor = torch.tensor([LibrispeechFile.phonemic_mapping[x] for x in new_transcription])
        return new_transcription, new_transcription_tensor

    def index_to_phone(self, idx: int) -> str:
        return LibrispeechFile.index_mapping[idx]
