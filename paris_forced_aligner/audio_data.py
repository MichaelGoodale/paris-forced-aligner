import os
import random
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
from paris_forced_aligner.model import G2PModel

class OutOfVocabularyException(Exception):
    """Raise for my specific kind of exception"""

class PronunciationDictionary:
    silence = "<SIL>"
    OOV = "<OOV>"

    def __init__(self, use_G2P:bool = True, lang='en', 
            G2P_model_path: str = "en_G2P_model.pt",
            train_G2P:bool = False,
            device:str = 'cpu',
            train_params = {"n_epochs": 10, "lr": 3e-4, "batch_size":64,}):
        self.lexicon: Mapping[str, List[str]] = {}
        self.phonemic_inventory: Set[str] = set([PronunciationDictionary.silence])
        self.graphemic_inventory: Set[str] = set()
        self.phone_to_phoneme: Mapping[str, str] = {}
        self.load_lexicon()
        self.phonemic_mapping: Mapping[str, int] = {phone: i+1 for i, phone in \
                                            enumerate(sorted(self.phonemic_inventory))}

        self.graphemic_mapping: Mapping[str, int] = {grapheme: i for i, grapheme in \
                                            enumerate(sorted(self.graphemic_inventory))}

        self.index_mapping: Mapping[int, str] = {v:k for k, v in self.phonemic_mapping.items()}
        self.use_G2P = use_G2P
        if use_G2P:
            self.device = device
            self.grapheme_pad_idx = len(self.graphemic_inventory) 
            self.grapheme_oov_idx = len(self.graphemic_inventory) + 1
            self.phoneme_pad_idx = len(self.phonemic_inventory)
            self.phoneme_start_idx = len(self.phonemic_inventory) + 1
            self.phoneme_end_idx = len(self.phonemic_inventory) + 2
            self.G2P_model = G2PModel(len(self.graphemic_inventory) + 2, len(self.phonemic_inventory) + 3,
                    self.grapheme_pad_idx, self.phoneme_pad_idx).to(device)
            if train_G2P:
                self.train_params = train_params
                self.cross_loss = torch.nn.NLLLoss(ignore_index=self.phoneme_pad_idx)
                self.optimizer = torch.optim.Adam(self.G2P_model.parameters(), lr=0.0003)
                self.train_G2P_model(G2P_model_path)
            else:
                self.G2P_model.load_state_dict(torch.load(G2P_model_path))

        self.lang = lang

    def load_lexicon(self):
        '''Function to load lexicon and phonemic inventory'''
        raise NotImplementedError("Lexicon loading must be defined in a subclass of PronunciationDictionary")

    def vocab_size(self) -> int:
        return len(self.phonemic_inventory) + 1

    def index_to_phone(self, idx: int) -> str:
        return self.index_mapping[idx]

    def teach_model(self, word_batch, pron_batch):
        word_batch_length = max(len(x) for x in word_batch)
        pron_batch_length = max(len(x) for x in pron_batch)
        word_batch = torch.LongTensor([w + [self.grapheme_pad_idx]*(word_batch_length - len(w)) \
                for w in word_batch]).T.to(self.device)
        pron_batch = torch.LongTensor([[self.phoneme_start_idx] + p + [self.phoneme_end_idx] + [self.phoneme_pad_idx]*(pron_batch_length - len(p)) \
                for p in pron_batch]).T.to(self.device)
        y = self.G2P_model(word_batch, pron_batch[:-1])
        loss = self.cross_loss(y.transpose(0, 1).transpose(1,2), pron_batch[1:].T)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train_G2P_model(self, model_path):
        n_epochs = self.train_params['n_epochs']
        batch_size = self.train_params['batch_size']
        epoch = 0
        words = list(self.lexicon.keys())
        random.Random(1337).shuffle(words)

        while epoch < n_epochs:
            word_batch = []
            pron_batch = []
            for i, word in enumerate(words):
                word_batch.append([self.graphemic_mapping[g] for g in word])
                pron_batch.append([self.phonemic_mapping[p] for p in self.lexicon[word]])
                if i % batch_size == 0:
                    loss = self.teach_model(word_batch, pron_batch)
                    word_batch = []
                    pron_batch = []
                    if (i // batch_size) % 10 == 0:
                        print(f"Loss {loss}, Epoch {epoch+1} / {n_epochs}, Step {i}")
            epoch += 1
            if len(word_batch) > 0:
                self.teach_model(word_batch, pron_batch)
        torch.save(self.G2P_model.state_dict(), model_path)

    def add_G2P_spelling(self, word: str):
        word = torch.LongTensor([[self.graphemic_mapping[w] for w in word]]).T.to(self.device)
        pronunciation = torch.LongTensor([[self.phoneme_start_idx]]).T.to(self.device)
        beams = [(0.0, pronunciation)]
        i = 0
        while i < int(len(word)*1.5):
            new_beams = []
            for probability, pronunciation in beams:
                y = self.G2P_model(word, pronunciation)
                probabilities, indices = torch.topk(y[-1, :, :], 5, dim=-1)
                for p, idx in zip(probabilities[0], indices[0]):
                    pronunciation = torch.cat((pronunciation, idx.unsqueeze(0).unsqueeze(0)))
                    new_beams.append((probability + p, pronunciation))
            i += 1
            beams = sorted(new_beams, reverse=True)[:50]


        pronunciation = []
        for phone in beams[0][1][:, 0]:
            if phone == self.phoneme_start_idx:
                continue
            elif phone == self.phoneme_end_idx:
                break
            pronunciation.append(self.index_mapping[phone.item()])

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
    def __init__(self, remove_stress=False, **kwargs):
        self.remove_stress = remove_stress
        self.get_lexicon_file()
        super().__init__(**kwargs)

    def get_lexicon_file(self):
        self.lexicon_path = os.path.join(data_directory, 'librispeech-lexicon.txt')
        if not os.path.exists(self.lexicon_path):
            download_data_file(LibrispeechDictionary.LIBRISPEECH_URL, self.lexicon_path)

    def get_word_and_pronunciation(self, line):
        word, pronunciation = re.split(r'\s+', line.strip(), maxsplit=1)
        if self.remove_stress:
            pronunciation = re.sub(r'[1-9]+', '', pronunciation)
            pronunciation = re.sub(r' \w+0', ' AX', pronunciation)
        return word, pronunciation.split(' ')


    def load_lexicon(self):
        self.phone_to_phoneme: Mapping[str, str] = arpabet_to_ipa

        with open(self.lexicon_path) as f:
            for line in f:
                word, pronunciation = self.get_word_and_pronunciation(line)
                self.lexicon[word] = pronunciation
                for phone in self.lexicon[word]:
                    self.phonemic_inventory.add(phone)

                for letter in word:
                    self.graphemic_inventory.add(letter)

class TSVDictionary(PronunciationDictionary):
    def __init__(self, lexicon_path: str, seperator: str='\t', \
            phone_to_phoneme: Optional[Mapping[str, str]] = None, **kwargs):
        super().__init__(**kwargs)
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

                for letter in word:
                    self.graphemic_inventory.add(letter)

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
