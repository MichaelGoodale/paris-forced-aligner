from typing import Optional, Mapping
from paris_forced_aligner.pronunciation import PronunciationDictionary

class TSVDictionary(PronunciationDictionary):
    def __init__(self, lexicon_path: str, seperator: str='\t', \
            phone_to_phoneme: Optional[Mapping[str, str]] = None, **kwargs):
        self.lexicon_path = lexicon_path
        self.seperator = seperator
        if phone_to_phoneme is not None:
            self.phone_to_phoneme = phone_to_phoneme
            self.already_ipa = False
        else:
            self.phone_to_phoneme: Mapping[str, str] = {}
            self.already_ipa = True
        super().__init__(**kwargs)

    def load_lexicon(self):
        with open(self.lexicon_path) as f:
            for line in f:
                word, pronunciation = line.strip().split(self.seperator, 1)
                word = word.upper()
                if word not in self.lexicon:
                    self.lexicon[word] = pronunciation.split(' ')
                for phone in self.lexicon[word]:
                    self.phonemic_inventory.add(phone)
                    if self.already_ipa:
                        self.phone_to_phoneme[phone] = phone

                for letter in word:
                    self.graphemic_inventory.add(letter)
