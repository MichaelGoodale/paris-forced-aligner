import re
from typing import Optional, Mapping, List

from paris_forced_aligner import ipa_data
from paris_forced_aligner.pronunciation import PronunciationDictionary, OutOfVocabularyException


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


    def __init__(self, multilingual=False):
        self.multilingual = multilingual
        super().__init__(use_G2P=False)

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
        self.graphemic_inventory = set("abcčdḍeɛfgǧɣhḥijklmnqrṛsṣtṭuwxyzẓε")

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
        all_letters = "".join(self.graphemic_inventory).upper()
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
        
    def spell_sentence(self, sentence: str, return_words: bool = True):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[\.\?\!,:;\'\"«»]', '', sentence)
        for char in sentence:
            if char not in ' -' and char not in self.graphemic_inventory:
                raise OutOfVocabularyException("Missing symbol {} due to probable loan word in sentence {}".format(char, sentence))
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
