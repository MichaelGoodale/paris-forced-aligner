from dataclasses import dataclass
from typing import List, Union

@dataclass
class Phone:
    label: str
    start: int
    end: int

@dataclass
class Word:
    phones: List[Phone]
    label: str

    @property
    def start(self):
        return self.phones[0].start

    @property
    def end(self):
        return self.phones[-1].end

@dataclass
class Silence:
    start: int
    end: int


@dataclass
class Utterance:
    data: List[Union[Word, Silence]]

    @property
    def words(self):
        return list(filter(lambda x: isinstance(x, Word), self.data))

    @property
    def start(self):
        return self.data[0].start

    @property
    def end(self):
        return self.data[-1].end
