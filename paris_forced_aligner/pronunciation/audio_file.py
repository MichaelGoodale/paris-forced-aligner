from typing import Union, BinaryIO, Optional, Mapping, List, Set, Tuple

import torch
from torch import Tensor
import torchaudio 
from torchaudio.transforms import Resample, Vad

from paris_forced_aligner.pronunciation import PronunciationDictionary

class AudioFile:
    def __init__(self, filename: str, transcription: str,
            pronunciation_dictionary: PronunciationDictionary,
            fileobj: Optional[BinaryIO] = None,
            wavobj: Optional[Tuple[Tensor, int]] = None,
            offset: int = 0):

        self.filename = filename
        self.pronunciation_dictionary = pronunciation_dictionary
        self.offset = offset
        self.load_audio(fileobj, wavobj)

        self.transcription, self.words = pronunciation_dictionary.spell_sentence(transcription, return_words=True)
        self.tensor_transcription = torch.tensor([self.pronunciation_dictionary.phonemic_mapping[x] \
                                                    for x in self.transcription])


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
        old_length = self.wav.shape[-1]
        self.wav = Vad(16000)(self.wav)
        self.offset += old_length - self.wav.shape[-1] 

    def move_to_device(self, device:str):
        self.wav = self.wav.to(device)
        self.tensor_transcription = self.tensor_transcription.to(device)
