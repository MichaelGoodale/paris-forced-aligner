import tarfile
import random
from typing import List
from functools import partial 

from torch.multiprocessing import Pool, cpu_count

from paris_forced_aligner.audio_data import LibrispeechFile, AudioFile, OutOfVocabularyException

class CorpusClass():
    def __init__(self, corpus_path: str):
        self.corpus_path: str = corpus_path

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files())

class LibrispeechCorpus(CorpusClass):
    def __init__(self, corpus_path: str, n_proc: int = cpu_count()):
        super().__init__(corpus_path)
        self.n_proc = n_proc

    def extract_files(self):
        with tarfile.open(self.corpus_path, 'r:gz', encoding='utf-8') as f:
            text_files = list(filter(lambda x: x.endswith('.trans.txt'), f.getnames()))
            directory_path = '/'.join(text_files[0].split('/')[:2])

        random.shuffle(text_files)

        if self.n_proc == 1:
            with tarfile.open(self.corpus_path, 'r:gz', encoding='utf-8') as tar_file:
                for text_path in text_files:
                    with tar_file.extractfile(text_path) as f:
                        for line in f:
                            line = line.decode('utf-8')
                            filename, transcription = line.strip().split(' ', 1)
                            filename = LibrispeechCorpus._get_flac_filepath(directory_path, filename)
                            try:
                                yield LibrispeechFile(filename, transcription, fileobj=tar_file.extractfile(filename))
                            except OutOfVocabularyException:
                                pass

        else:
            BATCH_SIZE = self.n_proc
            with Pool(self.n_proc) as p:
                for i in range(len(text_files)//BATCH_SIZE):
                    text_file_batch = text_files[BATCH_SIZE*i:BATCH_SIZE*i+BATCH_SIZE]
                    for directory in p.imap_unordered(partial(LibrispeechCorpus._extract_directory, self.corpus_path, directory_path), text_file_batch):
                        for audio in directory:
                            yield audio

    def _get_flac_filepath(directory_path, file_name):
        top_dir, mid_dir, _ = file_name.split('-')
        return '{}/{}/{}/{}.flac'.format(directory_path, top_dir, mid_dir, file_name)

    def _extract_directory(corpus_path, directory_path, text_path):
        returns = []
        with tarfile.open(corpus_path, 'r:gz', encoding='utf-8') as tar_file:
            with tar_file.extractfile(text_path) as f:
                for line in f:
                    line = line.decode('utf-8')
                    filename, transcription = line.strip().split(' ', 1)
                    filename = LibrispeechCorpus._get_flac_filepath(directory_path, filename)
                    try:
                        audio = LibrispeechFile(filename, transcription, fileobj=tar_file.extractfile(filename))
                        returns.append(audio)
                    except OutOfVocabularyException:
                        pass
        return returns
