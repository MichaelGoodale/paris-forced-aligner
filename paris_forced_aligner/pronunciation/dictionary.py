import os
import random
import re
from typing import Union, BinaryIO, Optional, Mapping, List, Set, Tuple

import torch
import torchaudio

from torch import Tensor

from num2words import num2words

from jiwer import wer 

from tqdm.autonotebook import tqdm

from paris_forced_aligner.ipa_data import arpabet_to_ipa
from paris_forced_aligner.phonological import Utterance
from paris_forced_aligner.model import G2PModel

class OutOfVocabularyException(Exception):
    """Raise for my specific kind of exception"""

class PronunciationDictionary:
    silence = "<SIL>"
    blank = "<BLANK>"
    OOV = "<OOV>"

    def __init__(self, use_G2P:bool = False, lang='en', 
            G2P_model_path: str = "en_G2P_model.pt",
            train_G2P:bool = False,
            continue_training=False,
            device:str = 'cpu',
            train_params = {"n_epochs": 10, "lr": 3e-4, "batch_size":64,}):

        self.lexicon: Mapping[str, List[str]] = {}
        self.phonemic_inventory: Set[str] = set()
        self.graphemic_inventory: Set[str] = set()
        self.phone_to_phoneme: Mapping[str, str] = {}
        self.load_lexicon()
        self.phonemic_mapping: Mapping[str, int] = {phone: i+1 for i, phone in \
                                            enumerate(sorted(self.phonemic_inventory))}

        self.phonemic_inventory.add(PronunciationDictionary.blank)
        self.phonemic_mapping[PronunciationDictionary.blank] = 0
        self.graphemic_mapping: Mapping[str, int] = {grapheme: i for i, grapheme in \
                                            enumerate(sorted(self.graphemic_inventory))}

        self.index_mapping: Mapping[int, str] = {v:k for k, v in self.phonemic_mapping.items()}
        self.use_G2P = use_G2P

        if use_G2P:
            self.device = device
            self.grapheme_pad_idx = len(self.graphemic_inventory)
            self.grapheme_oov_idx = len(self.graphemic_inventory) + 1
            self.grapheme_end_idx = len(self.phonemic_inventory) + 2

            self.phoneme_pad_idx = 0 #Use <SIL> as PAD
            self.phoneme_start_idx = len(self.phonemic_inventory)
            self.phoneme_end_idx = len(self.phonemic_inventory) + 1
            self.G2P_model = G2PModel(len(self.graphemic_inventory) + 2, len(self.phonemic_inventory) + 2,
                    self.grapheme_pad_idx, self.phoneme_pad_idx).to(device)
            if train_G2P:
                self.train_params = train_params
                self.cross_loss = torch.nn.NLLLoss()
                self.optimizer = torch.optim.Adam(self.G2P_model.parameters(), lr=train_params["lr"])
                starting_epoch = 0
                if continue_training:
                    checkpoint = torch.load(G2P_model_path, map_location=self.device)
                    self.G2P_model.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    starting_epoch = checkpoint["epoch"]
                    self.train_params = checkpoint["train_params"]
                self.train_G2P_model(G2P_model_path, starting_epoch=starting_epoch)
            else:
                self.G2P_model.load_state_dict(torch.load(G2P_model_path, map_location=self.device)['model_state_dict'])
                self.G2P_model.eval()

        self.lang = lang

    def load_lexicon(self):
        '''Function to load lexicon and phonemic inventory'''
        raise NotImplementedError("Lexicon loading must be defined in a subclass of PronunciationDictionary")

    def vocab_size(self) -> int:
        return len(self.phonemic_inventory)

    def index_to_phone(self, idx: int) -> str:
        return self.index_mapping[idx]

    def prepare_batches(self, word_batch, pron_batch):
        word_batch_length = max(len(x) for x in word_batch)
        pron_batch_length = max(len(x) for x in pron_batch)
        word_batch = torch.LongTensor([w + [self.grapheme_pad_idx]*(word_batch_length - len(w)) \
                for w in word_batch]).T.to(self.device)
        pron_batch = torch.LongTensor([[self.phoneme_start_idx] + p + [self.phoneme_end_idx] + [self.phoneme_pad_idx]*(pron_batch_length - len(p)) \
                for p in pron_batch]).T.to(self.device)
        return word_batch, pron_batch

    def teach_model(self, word_batch, pron_batch):
        word_batch, pron_batch = self.prepare_batches(word_batch, pron_batch)
        y = self.G2P_model(word_batch, pron_batch[:-1], device=self.device)
        loss = self.cross_loss(y.transpose(0, 1).transpose(1,2), pron_batch[1:].T)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def get_G2P_accuracy(self, words, batch_size):
        real_word_batch = []
        word_batch = []
        pron_batch = []
        #Extremely confusing terminology G2P transformer paper where phoneme error rate is WER on phonemes whereas "WER" is just percentage of incorrect words
        avg_per = 0
        avg_wer = 0
        n = 0 
        def run_batch(real_word_batch, word_batch, pron_batch, avg_per, avg_wer, n):
            word_batch, pron_batch = self.prepare_batches(word_batch, pron_batch)
            max_pron_len = int(pron_batch.shape[0] * 1.2)
            pron_batch = torch.LongTensor([[self.phoneme_start_idx]*word_batch.shape[1]]).to(self.device)
            for _ in range(max_pron_len):
                y = self.G2P_model(word_batch, pron_batch, device=self.device)
                pron_batch = torch.cat((pron_batch, torch.argmax(y[-1, :, :], dim=-1).unsqueeze(0)), dim=0)

            for i, word in enumerate(real_word_batch):
                real_pronunciation = self.lexicon[word]
                pronunciation = []
                for x in pron_batch[1:, i]:
                    x = x.item()
                    if x not in self.index_mapping or x == self.phoneme_pad_idx:
                        break #Since pad_idx is <SIL> it will be in index_mapping
                    pronunciation.append(self.index_mapping[x])
                avg_per += (wer(real_pronunciation, pronunciation) - avg_per) / (n+1)
                avg_wer += (int(real_pronunciation != pronunciation) - avg_wer) / (n+1)
                n += 1
            return avg_per, avg_wer, n

        for i, word in enumerate(words):
            if i % batch_size == 0 and i > 0:
                avg_per, avg_wer, n = run_batch(real_word_batch, word_batch, pron_batch, avg_per, avg_wer, n)
                real_word_batch = []
                word_batch = []
                pron_batch = []

            real_word_batch.append(word)
            word_batch.append([self.graphemic_mapping[g] for g in word])
            pron_batch.append([self.phonemic_mapping[p] for p in self.lexicon[word]])

        if real_word_batch != []:
            avg_per, avg_wer, n = run_batch(real_word_batch, word_batch, pron_batch, avg_per, avg_wer, n)
        return avg_per, avg_wer


    def train_G2P_model(self, model_path, train_test_split=0.98, output_model_every=20, starting_epoch=0):
        n_epochs = self.train_params['n_epochs']
        batch_size = self.train_params['batch_size']
        epoch = starting_epoch
        words = list(self.lexicon.keys())
        random.Random(1337).shuffle(words)

        test_words = words[int(len(words)*train_test_split):]
        train_words = words[:int(len(words)*train_test_split)]

        while epoch < n_epochs:
            word_batch = []
            pron_batch = []
            losses = []
            for i, word in enumerate(tqdm(train_words, desc=f"Epoch {epoch+1}/{n_epochs}")):
                word_batch.append([self.graphemic_mapping[g] for g in word])
                pron_batch.append([self.phonemic_mapping[p] for p in self.lexicon[word]])
                if i % batch_size == 0:
                    loss = self.teach_model(word_batch, pron_batch)
                    word_batch = []
                    pron_batch = []
                    losses.append(loss)

            if len(word_batch) > 0:
                loss = self.teach_model(word_batch, pron_batch)
                losses.append(loss)

            with torch.no_grad():
                per, wer = self.get_G2P_accuracy(test_words, batch_size)
            print(f"Average Loss={sum(losses)/len(losses):.4f}, PER {per:.4f}, WER {wer:.4f}")

            epoch += 1
            if epoch % output_model_every == 0:
                torch.save({"model_state_dict": self.G2P_model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "loss": sum(losses)/len(losses),
                            "per": per,
                            "wer": wer,
                            "epoch": epoch,
                            "train_params": self.train_params},
                        model_path)

        torch.save({"model_state_dict": self.G2P_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "loss": sum(losses)/len(losses),
                    "per": per,
                    "wer": wer,
                    "epoch": epoch,
                    "train_params": self.train_params},
                model_path)

    def add_G2P_spelling(self, word: str):
        word_tensor = torch.LongTensor([[self.graphemic_mapping[w] if w in self.graphemic_mapping else self.grapheme_oov_idx for w in word] + [self.grapheme_pad_idx]]).T.to(self.device)
        pronunciation = torch.LongTensor([[self.phoneme_start_idx]]).T.to(self.device)
        beams = [(0.0, pronunciation.clone())]
        i = 0
        while i < int(len(word_tensor)*2):
            new_beams = []
            for probability, pronunciation in beams:
                y = self.G2P_model(word_tensor, pronunciation, device=self.device)
                probabilities, indices = torch.topk(y[-1, :, :], 5, dim=-1)
                for p, idx in zip(probabilities[0], indices[0]):
                    pronunciation = torch.cat((pronunciation, idx.unsqueeze(0).unsqueeze(0)))
                    new_beams.append((probability + p, pronunciation.clone()))
            i += 1
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:10]

        pronunciation = []
        for phone in beams[0][1][:, 0]:
            if phone == self.phoneme_start_idx:
                continue
            elif phone == self.phoneme_end_idx:
                break
            pronunciation.append(self.index_mapping[phone.item()])
        self.lexicon[word] = pronunciation

    def add_words_from_utterance(self, utterance: Utterance):
        for word in utterance.words:
            if word.label not in self.lexicon:
                self.lexicon[word.label] = [p.label for p in word.phones]

    def split_sentence(self, sentence:str) -> List[str]:
        return_sentence = []
        sentence = sentence.replace('-', ' ')
        for word in sentence.split():
            if word in self.lexicon:
                return_sentence.append(word.upper())
            elif word.isdigit():
                word = num2words(int(word), lang=self.lang).replace('-', ' ').upper().split(' ')
                return_sentence += word
            elif '\'' in word:
                x = word.split('\'')
                if len(x) == 2:
                    prefix, suffix = (x[0], x[1])
                    if prefix + '\'' in self.lexicon:
                        return_sentence.append(prefix.upper() + '\'')
                        return_sentence.append(suffix.upper())
                    elif '\'' + suffix in self.lexicon:
                        return_sentence.append(prefix.upper())
                        return_sentence.append('\'' + suffix.upper())
                    else:
                        return_sentence.append(word.upper())
                else:
                    return_sentence.append(word.upper())
            else:
                return_sentence.append(word.upper())
                        
        return return_sentence

    def spell_sentence(self, sentence: str, return_words: bool = True):
        sentence = self.split_sentence(sentence)
        spelling: List[str] = []

        for word in sentence:
            spelling += self.spelling(word)

        if return_words:
            return spelling, sentence
        return spelling

    def spelling(self, word: str) -> List[str]:
        if word not in self.lexicon:
            if not self.use_G2P:
                raise OutOfVocabularyException(f"{word} is not present in the lexicon")
            self.add_G2P_spelling(word)
        return self.lexicon[word]

class MultiLanguagePronunciationDictionary:
    def __init__(self, pronunciation_dictionaries: List[PronunciationDictionary]):
        self.dictionaries = {x.lang: x for x in pronunciation_dictionaries}

    def spell_sentence(self, language:str, sentence: str, return_words: bool = True):
        return self.pronunciation_dictionaries[language].spell_sentence(sentence, return_words=return_words)

    def spelling(self, language:str, word: str) -> List[str]:
        return self.pronunciation_dictionaries[language].spelling(word)

