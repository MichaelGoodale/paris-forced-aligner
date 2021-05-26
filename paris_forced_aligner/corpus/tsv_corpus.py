import os
import csv

from paris_forced_aligner.pronunciation import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone
from paris_forced_aligner.corpus import CorpusClass

class TSVCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary = None, delimiter='\t', path_prefix=''):
        super().__init__(corpus_path, pronunciation_dictionary)
        self.all_files = []
        with open(corpus) as f:
            csv_file = csv.DictReader(f, delimiter=delimiter)
            for row in csv_file:
                self.all_files.append((os.path.join(path_prefix, row['path']), row['sentence']))

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("TSVCorpus does not support gold standard phonemic labels")

        for filename, transcription in self.all_files:
            try:
                yield AudioFile(filename, transcription, self.pronunciation_dictionary)
            except OutOfVocabularyException:
                continue

    def __len__(self):
        return len(self.all_files)

