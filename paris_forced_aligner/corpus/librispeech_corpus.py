import os

from paris_forced_aligner.pronunciation import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone
from paris_forced_aligner.corpus import CorpusClass

class LibrispeechCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary = None):
        if pronunciation_dictionary is None:
            pronunciation_dictionary = LibrispeechDictionary()
        super().__init__(corpus_path, pronunciation_dictionary)

        found_files = False
        self.all_files = []
        for d, s_d, files in os.walk(self.corpus_path):
            for t_file in filter(lambda x: x.endswith('.trans.txt'), files):
                with open(os.path.join(d, t_file)) as f:
                    found_files = True
                    for line in f:
                        filename, transcription = line.strip().split(' ', 1)
                        filename = os.path.join(d, f"{filename}.flac")
                        self.all_files.append((filename, transcription))

        if not found_files:
            raise IOError(f"{self.corpus_path} has no files!")

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("Librispeech does not have gold standard phonemic labels")

        for filename, transcription in self.all_files:
            try:
                yield AudioFile(filename, transcription, self.pronunciation_dictionary)
            except OutOfVocabularyException:
                continue

    def __len__(self):
        return len(self.all_files)

