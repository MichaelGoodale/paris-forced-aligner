import io
import os
import re
import tarfile
from zipfile import ZipFile
import random
from typing import List, Optional
from functools import partial 
from itertools import chain

from torch.multiprocessing import Pool, cpu_count
import buckeye
import youtube_dl
from tqdm.autonotebook import tqdm
import webvtt
import torchaudio
import tempfile

from paris_forced_aligner.pronunciation import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone

class CorpusClass():
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False):
        self.corpus_path: str = os.path.expanduser(corpus_path)
        self.pronunciation_dictionary: PronunciationDictionary = pronunciation_dictionary
        self.return_gold_labels = return_gold_labels

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files(self.return_gold_labels))

    def __len__(self):
        raise NotImplementedError("CorpusClass has no __len__ implemented by default")

