import os
import re

from paris_forced_aligner.utils import data_directory, download_data_file
from paris_forced_aligner.pronunciation import PronunciationDictionary

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
