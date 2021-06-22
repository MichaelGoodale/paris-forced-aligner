import re
import csv
import random
from typing import Optional, Mapping, List

from jiwer import wer 
import torch
from tqdm import tqdm

from paris_forced_aligner import ipa_data
from paris_forced_aligner.pronunciation import PronunciationDictionary, OutOfVocabularyException
from paris_forced_aligner.model import G2PModel


class KabyleDictionary(PronunciationDictionary):

    #(first letter, second letter) -> (replacement consonant(s), only when it's a particle)
    assimilations = {('n', 't'): ('T', False),
                     ('n', 'w'): ('W', False),
                     ('n', 'y'): ('G', False),
                     ('g', 'w'): ('G', False),#Probably labialised
                     ('g', 'y'): ('G', False),
                     ('g', 'u'): ('g', False), #Probably labialised
                     ('f', 'u'): ('Fu', False),
                     ('f', 'w'): ('F', False),
                     ('m', 'w'): ('M', False),
                     ('d', 't'): ('T', False),
                     ('i', 'i'): ('ig', False),
                     ('i', 'y'): ('ig', False),
                     ('ay', 'y'): ('ag', False),
                     ('ḍ', 't'): ('ṭ', False),
                     ('n', 'ṛ'): ('Ṛ', True),
                     ('n', 'f'): ('F', True),
                     ('n', 'l'): ('L', True),
                     ('n', 'm'): ('M', True),
                     ('n', 'm'): ('mb', True)}

    #Doesn't include stop/fricative phonemes
    ipa_mapping = {
            'a': 'æ',
            'i': 'ɪ',
            'u': 'ʊ',
            'e': 'ə',
            'f': 'f',
            'j': 'ʒ',
            'y': 'j',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'z': 'z',
            's': 's',
            'c': 'ʃ',
            'r': 'r',
            'h': 'h',
            'q': 'q',
            'w': 'w',
            'x': 'χ',
            'ɣ': 'ʁ',
            'γ': 'ʁ',
            'ḥ': 'ħ',
            'č': 'tʃ',
            'ǧ': 'dʒ',
            'ğ': 'dʒ',
            'ṭ': 'tˤ',
            'ḍ': 'ðˤ',
            'ṛ': 'rˤ',
            'ḷ': 'lˤ',
            'ṣ': 'sˤ',
            'ẓ': 'zˤ',
            'ɛ': 'ʕ', #<- Unicode shenanigans
            'ε': 'ʕ'}


    def __init__(self, multilingual=False, text_corpus_path: str=None, use_P2G=False,
            train_P2G=False, 
            continue_training=False,
            P2G_model_path="kab_P2G_model.pt",
            device: str = 'cpu',
            train_params = {"n_epochs": 10, "lr": 3e-4, "batch_size":64,}):

        self.multilingual = multilingual
        self.text_corpus_path = text_corpus_path
        super().__init__(use_G2P=False, lang="kab")

        if use_P2G:
            self.device = device
            #Confusing terminology in below lets us use G2P helper methods
            #Should be refactored...
            self.grapheme_pad_idx = 0
            self.grapheme_oov_idx = len(self.phonemic_inventory)
            self.grapheme_end_idx = len(self.phonemic_inventory) + 1

            self.phoneme_pad_idx = len(self.graphemic_inventory) 
            self.phoneme_start_idx = len(self.graphemic_inventory) + 1
            self.phoneme_end_idx = len(self.graphemic_inventory) + 2

            self.G2P_model = G2PModel(len(self.phonemic_inventory) + 2, len(self.graphemic_inventory) + 3,
                    self.grapheme_pad_idx, self.phoneme_pad_idx).to(device)

            if train_P2G:
                self.train_params = train_params
                self.cross_loss = torch.nn.NLLLoss()
                self.optimizer = torch.optim.Adam(self.G2P_model.parameters(), lr=train_params["lr"])
                starting_epoch = 0
                if continue_training:
                    checkpoint = torch.load(P2G_model_path, map_location=self.device)
                    self.G2P_model.load_state_dict(checkpoint["model_state_dict"])
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    starting_epoch = checkpoint["epoch"]
                    self.train_params = checkpoint["train_params"]
                self.train_P2G_model(P2G_model_path, text_corpus_path, starting_epoch=starting_epoch)
            else:
                self.G2P_model.load_state_dict(torch.load(P2G_model_path, map_location=self.device)['model_state_dict'])
                self.G2P_model.eval()


    def get_P2G_accuracy(self, sentences, batch_size):
        sentence_batch = []
        pron_batch = []
        orth_batch = []
        #Extremely confusing terminology G2P transformer paper where phoneme error rate is WER on phonemes whereas "WER" is just percentage of incorrect words
        avg_wer = 0
        n = 0 
        def run_batch(sentence_batch, pron_batch, orth_batch, avg_wer, n):
            pron_batch, orth_batch = self.prepare_batches(pron_batch, orth_batch)
            max_orth_len = int(orth_batch.shape[0] * 1.2)
            orth_batch = torch.LongTensor([[self.phoneme_start_idx]*len(sentence_batch)]).to(self.device)
            for _ in range(max_orth_len):
                y = self.G2P_model(pron_batch, orth_batch, device=self.device)
                orth_batch = torch.cat((orth_batch, torch.argmax(y[-1, :, :], dim=-1).unsqueeze(0)), dim=0)

            for i, sentence in enumerate(sentence_batch):
                spelling = []
                for x in orth_batch[1:, i]:
                    x = x.item()
                    if x not in self.grapheme_index_mapping or x == self.phoneme_pad_idx:
                        break #Since pad_idx is <SIL> it will be in index_mapping
                    spelling.append(self.grapheme_index_mapping[x])
                avg_wer += (wer(sentence, spelling) - avg_wer) / (n+1)
                n += 1
            return avg_wer, n

        for sentence in tqdm(sentences, desc="Validating..."):
            if len(sentence_batch) % batch_size == 0 and len(pron_batch) > 0:
                avg_wer, n = run_batch(sentence_batch, pron_batch, orth_batch, avg_wer, n)
                sentence_batch = []
                pron_batch = []
                orth_batch = []
            try:
                sentence = self.clean_sentence(sentence)
            except OutOfVocabularyException:
                continue
            sentence_batch.append(sentence)
            pron_batch.append([self.phonemic_mapping[p] for p in self.spell_sentence(sentence, return_words=False)])
            orth_batch.append([self.graphemic_mapping[g] for g in sentence])

        if sentence_batch != []:
            avg_wer, n = run_batch(sentence_batch, pron_batch, orth_batch, avg_wer, n)
        return avg_wer

    def train_P2G_model(self, model_path, text_corpus_path, train_test_split=0.98, output_model_every=20, starting_epoch=0):
        if text_corpus_path is None:
            raise ValueError("text_corpus_path is none, please provide a corpus path")

        n_epochs = self.train_params['n_epochs']
        batch_size = self.train_params['batch_size']
        epoch = starting_epoch
        sentences = []
        with open(text_corpus_path) as f:
            csv_file = csv.DictReader(f, delimiter='\t')
            for row in csv_file:
                sentences.append(row['sentence'])
        rand = random.Random(1337)
        rand.shuffle(sentences)
        test_sentences = sentences[int(len(sentences)*train_test_split):]
        train_sentences = sentences[:int(len(sentences)*train_test_split)]

        while epoch < n_epochs:
            rand.shuffle(sentences)
            pron_batch = []
            orth_batch = []
            losses = []
            with tqdm(train_sentences, desc=f"Epoch {epoch+1}/{n_epochs}") as sentences_iterator:
                for sentence in sentences_iterator:
                    try:
                        sentence = self.clean_sentence(sentence)
                    except OutOfVocabularyException:
                        continue
                    pron_batch.append([self.phonemic_mapping[p] for p in self.spell_sentence(sentence, return_words=False)])
                    orth_batch.append([self.graphemic_mapping[g] for g in sentence])
                    if len(pron_batch) % batch_size == 0 and len(pron_batch) > 0:
                        loss = self.teach_model(pron_batch, orth_batch)
                        orth_batch = []
                        pron_batch = []
                        losses.append(loss)
                        sentences_iterator.set_postfix({"Previous loss": loss})

            if len(pron_batch) > 0:
                loss = self.teach_model(pron_batch, orth_batch)
                losses.append(loss)

            with torch.no_grad():
                wer = self.get_P2G_accuracy(test_sentences, batch_size)

            print(f"Average Loss={sum(losses)/len(losses):.4f}, WER {wer:.4f}")

            epoch += 1
            if epoch % output_model_every == 0:
                torch.save({"model_state_dict": self.G2P_model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "loss": sum(losses)/len(losses),
                            "wer": wer,
                            "epoch": epoch,
                            "train_params": self.train_params},
                        model_path)

        torch.save({"model_state_dict": self.G2P_model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "loss": sum(losses)/len(losses),
                    "wer": wer,
                    "epoch": epoch,
                    "train_params": self.train_params},
                model_path)

    def load_lexicon(self):
        '''Since orthography is entirely transparent; we're not loading a lexiocon,
        just the phonemic/graphemic inventories'''
        if self.multilingual:
            self.phonemic_inventory = set(ipa_data.CHAR2IDX.keys())
        else:
            self.phonemic_inventory = set(x for _, x in KabyleDictionary.ipa_mapping.items())
            self.phonemic_inventory.update(['b', 'g', 'k', 'ts', 'dz', 'β', 'd̪', 'ð', 'ʝ', 'ç', 't̪', 'θ'])
            for phone in list(self.phonemic_inventory):
                if phone not in ['æ', 'ɪ', 'ʊ', 'ə', 'ts', 'dz', 'ts', 'dz', 'β', 'd̪', 'ð', 'ʝ', 'ç', 't̪', 'θ']:
                    self.phonemic_inventory.add(f'{phone}ː')
        self.letter_inventory = set("abcčdḍeɛfgǧɣhḥijklmnqrṛsṣtṭuwxyzẓε")
        self.graphemic_inventory = set(" -").union(self.letter_inventory)

    def mark_assimilation(self, string: str) -> str:
        ''' Takes a string and returns a string where all
        possible Kabyle assimilations are marked with by the letters together with a +

        from: Sur la notation usuelle du berbère – Eléments d’orthographe[note élaborée par K. Naït-Zerrad, 1998 – révision 2002 par S. Chaker]
        '''
        for (a, b), (particle_only, _) in KabyleDictionary.assimilations.items():
            if particle_only:
                pattern = f'(\s{a})\s({b})\w'
            else:
                pattern = f'({a})\s({b})\w'
            replace_fn = lambda x: x.group(1)+'+'+x.group(2)
            string = re.sub(pattern, replace_fn, string)
        return string

    def assimilate(self, string: str) -> str:
        def replace_func(x):
            a = x.group(1)
            b = x.group(2)
            if a == 'y' == b:
                a = 'ay'
            return KabyleDictionary.assimilations[(a, b)][0]
        return re.sub(r'(\w)\+(\w)', replace_func, string)

    def geminate(self, string: str) -> str:
        '''Replace repeated consonants as upper case to mark as a geminate''' 
        ##ADD GEMINATES FROM -
        string = re.sub(r'([^\Waeiu_])-\1', lambda x: x.group(1).upper()+'-', string)
        return re.sub(r'([^\Waeiu_])\1', lambda x: x.group(1).upper(), string)

    def degeminate(self, string: str) -> str:
        '''Replace repeated consonants as upper case to mark as a geminate'''
        all_letters = "".join(self.letter_inventory).upper()
        string = re.sub(f'([{all_letters}]-)', lambda x: (x.group(1).lower()*2)[:-1], string)
        return re.sub(f'([{all_letters}])', lambda x: x.group(1).lower()*2, string)

    def split_assimilate(self, string: str) -> List[str]:
        '''Returns a string split on whitespace or on +
        where the character to the left of + is a part of the word'''
        string = string.split(' ')
        ret_string = []
        for word in string:
            word = word.split('+')
            if len(word) > 1:
                last_letter = None
                for i in range(len(word)):
                    if last_letter is not None:
                        word[i] = f'{last_letter}+{word[i]}'
                    last_letter = word[i][-1]
                    if i < len(word) - 1:
                        word[i] = word[i][:-1]

                word = [w for w in word if w != '']
            ret_string += word
        return ret_string

    def make_occlusive(self, char: str, prev_char: str) -> str:
        '''Give a char and the one preceeding, figure out whether to make into stop
        NOTE: <g> is /g/ in ngeb, ngeḥ, ngeẓwer, angaẓ, ngedwi, nages, ngedwal but I don't
        have an easy way to check this, since the morphology is a bit complex :(
        '''
        if char == 'b':
            if prev_char == 'm':
                return 'b'
            return 'β'
        elif char == 'd':
            if prev_char in ['n', 'l']:
                return 'd̪'
            return 'ð'
        elif char == 'g':
            if prev_char in ['b', 'j', 'r', 'z', 'ɛ']:
                return 'g'
            return 'ʝ'
        elif char == 'k':
            if prev_char in ['f', 'b', 's', 'l', 'r', 'n', 'ḥ', 'c', 'ɛ']:
                return 'k'
            return 'ç'
        elif char == 't':
            if prev_char in ['n', 'l']:
                return 't̪'
            return 'θ'
        return char

    def turn_into_ipa(self, string: str) -> List[str]:
        string = string.replace('-', '')
        pronunciation = []
        prev_char = None
        for char in string:
            if char in ['b', 'd', 'g', 'k', 't']:
                ipa_char = self.make_occlusive(char, prev_char)
            elif char.isupper():
                non_geminated_char = char.lower()
                if non_geminated_char == 'ḍ':
                    ipa_char = 'tˤː'
                elif non_geminated_char == 'y':
                    ipa_char = 'gː'
                elif non_geminated_char in 'btdgk':
                    ipa_char = non_geminated_char + 'ː'
                else:
                    ipa_char = KabyleDictionary.ipa_mapping[non_geminated_char] + 'ː'
            else:
                ipa_char = KabyleDictionary.ipa_mapping[char]
            pronunciation.append(ipa_char)
            prev_char = char
        return pronunciation 

    def clean_sentence(self, sentence: str) -> str:
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[\.\?\!,:;\'\"«»]', '', sentence)
        for char in sentence:
            if char not in ' -' and char not in self.letter_inventory:
                raise OutOfVocabularyException("Missing symbol {} due to probable loan word in sentence {}".format(char, sentence))
        return sentence
        
    def spell_sentence(self, sentence: str, return_words: bool = True):
        sentence = self.clean_sentence(sentence)
        spelling: List[str] = []
        sentence = self.geminate(sentence)
        sentence = self.mark_assimilation(sentence)
        sentence = self.split_assimilate(sentence)
        for word in sentence:
            word = self.assimilate(word)
            word = self.turn_into_ipa(word)
            spelling.append(word)
        sentence = [(self.degeminate(word), word_spelling) for word, word_spelling in zip(sentence, spelling)]
        spelling = [phone for word in spelling for phone in word]
        if return_words:
            return spelling, sentence
        return spelling
