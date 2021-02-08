from dataclasses import dataclass
from typing import List, Union

import textgrid
import csv

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

    @property
    def base_units(self):
        base = []
        for d in self.data:
            if isinstance(d, Silence):
                base.append(d)
            else:
                base += d.phones
        return base

    def offset(self, offset):
        for unit in self.base_units:
            unit.start += offset
            unit.end += offset

    def save_csv(self, output_file: str, in_seconds: bool = False):
        '''WARNING: If your wav file's sample rate is not 16Hz, the sample labels here will be offset'''

        with open(output_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames = ['phone', 'word', 'start', 'end'])
            for word in self.words:
                for phone in word.phones:
                    if in_seconds:
                        writer.writerow({'phone': phone.label, 'word': word.label,
                            'start': phone.start / 16000, 'end': phone.end / 16000})
                    else:
                        writer.writerow({'phone': phone.label, 'word': word.label,
                            'start': phone.start, 'end': phone.end})

    def save_textgrid(self, output_file: str):
        tg = textgrid.TextGrid()
        words = textgrid.IntervalTier('words')
        phones = textgrid.IntervalTier('phones')
        for word in self.words:
            words.add(word.start / 16000, word.end / 16000, word.label)
            for phone in word.phones:
                phones.add(phone.start / 16000, phone.end/ 16000, phone.label)

        tg.append(words)
        tg.append(phones)

        with open(output_file, 'w') as f:
            tg.write(f)
