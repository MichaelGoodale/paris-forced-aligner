import tarfile
from typing import List

from audio_data import LibrispeechFile, AudioFile, OutOfVocabularyException

class CorpusClass():
    def __init__(self, corpus_path: str):
        self.corpus_path: str = corpus_path

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files())


class LibrispeechCorpus(CorpusClass):
    def extract_files(self):
        self._directory_path: str = ''
        with tarfile.open(self.corpus_path, 'r:gz', encoding='utf-8') as f:
            for text_file in filter(lambda x: x.endswith('.trans.txt'), f.getnames()):
                if self._directory_path == '':
                    self._directory_path = '/'.join(text_file.split('/')[:2])
                for audio in self._extract_directory(text_file, f):
                    yield audio

    def _get_flac_filepath(self, file_name):
        top_dir, mid_dir, _ = file_name.split('-')
        return '{}/{}/{}/{}.flac'.format(self._directory_path, top_dir, mid_dir, file_name)

    def _extract_directory(self, text_path, tar_file):
        with tar_file.extractfile(text_path) as f:
            for line in f:
                line = line.decode('utf-8')
                filename, transcription = line.strip().split(' ', 1)
                filename = self._get_flac_filepath(filename)
                try:
                    yield LibrispeechFile(filename, transcription, fileobj=tar_file.extractfile(filename))
                except OutOfVocabularyException:
                    pass
