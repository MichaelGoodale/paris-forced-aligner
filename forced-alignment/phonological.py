from dataclasses import dataclass
from typing import List

@dataclass
class Phoneme:
    phoneme_label: str
    phoneme_ipa: str

@dataclass
class Phone:
    phoneme: Phoneme
    start: int
    end: int

@dataclass
class Word:
    phones: List[Phone]
    word_label: str

    @property
    def start(self):
        return self.phones[0].start

    @property
    def end(self):
        return self.phones[-1].end

class Silence:
    start: int
    end: int


class Utterance:
    def __init__(data: List[Union[Word, Silence]]):
        self.data = data

    @property
    def words:
        return list(filter(lambda x: isinstance(x, Word), self.data))

    @property
    def start:
        return self.data[0].start

    @property
    def end:
        return self.data[-1].end
