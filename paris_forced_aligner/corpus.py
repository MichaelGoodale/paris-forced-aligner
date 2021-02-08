import os
import tarfile
import random
from typing import List, Optional
from functools import partial 

from torch.multiprocessing import Pool, cpu_count
import youtube_dl
import webvtt
import torchaudio
import tempfile

from paris_forced_aligner.audio_data import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance


class CorpusClass():
    def __init__(self, corpus_path: str):
        self.corpus_path: str = corpus_path
        self.pronunciation_dictionary = pronunciation_dictionary

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files())

class YoutubeCorpus(CorpusClass):

    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, language: str = 'en', audio_directory: Optional[str] = None, save_wavs: bool = False, clear_data_every=100):
        super().__init__(corpus_path, pronunciation_dictionary)
        self.youtube_files = []
        self.save_wavs = save_wavs
        self.clear_data_every = clear_data_every

        if audio_directory is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.dir = self._temp_dir.name
        else:
            self.dir = audio_directory
        self.language = language

        if os.path.exists(corpus_path):
            with open(corpus_path, 'r') as f:
                for line in f:
                    self.youtube_files.append(line.strip())
        else:
            self.youtube_files.append(corpus_path)

    def extract_files(self):
        YDL_OPTS = {
            'format': 'bestaudio/best',
            'writeautomaticsub': True,
            'outtmpl': self.dir+'/%(uploader_id)s.%(id)s.%(ext)s',
            'subtitleslangs':[self.language],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }
        with youtube_dl.YoutubeDL(YDL_OPTS) as ydl:
            ydl.download(self.youtube_files)

        sub_file_ending = ".{}.vtt".format(self.language)

        for subtitle_file in filter(lambda x: x.endswith(sub_file_ending), os.listdir(self.dir)):
            audio = subtitle_file.replace(sub_file_ending, ".wav")
            captions = webvtt.read(os.path.join(self.dir, subtitle_file)).captions
            wav, sr = torchaudio.load(os.path.join(self.dir, audio))

            if self.save_wavs:
                idx_starts_ends: List[Tuple[int, int, int]] = []

            for i, cap_time, cap_string in enumerate(zip(captions[::2], captions[1::2])):
                cap_name = subtitle_file.replace(sub_file_ending, f"_{i}")
                transcription = cap_string.text.strip().upper()
                start = int(cap_time.start_in_seconds * sr)
                end = int(cap_time.end_in_seconds * sr)

                if self.save_wavs:
                    idx_starts_ends.append(i, start, end)

                yield AudioFile(cap_name, transcription, self.pronunciation_dictionary, wavobj=(wav[:, start:end], sr))

            if self.save_wavs:
                with open(subtitle_file.replace(sub_file_ending, '.txt'), 'w') as f:
                    for i, start, end in idx_starts_ends:
                        f.write(f"{i} {start} {end}\n")

    def stitch_youtube_utterances(audio_file_name: str, utterances: List[Utterance]):
        index_file = audio_file_name.rsplit("_", 1)[0] + '.txt' 
        with open(index_file, 'r') as f:
            for utterance, line in zip(utterances, f):
                i, start, end = i.strip().split(' ')
                utterance.offset(start)
        return Utterance(sum(u.data for u in utterances))

    def cleanup_and_recreate(self):
        self.cleanup()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.dir = self._temp_dir.name

    def cleanup(self):
        self._temp_dir.cleanup()


class LibrispeechCorpus(CorpusClass):
    def __init__(self, corpus_path: str, n_proc: int = cpu_count()):
        super().__init__(corpus_path, LibrispeechDictionary())
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
                                yield AudioFile(filename, transcription, self.pronunciation_dictionary, fileobj=tar_file.extractfile(filename))
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
                        audio = AudioFile(filename, transcription, self.pronunciation_dictionary, fileobj=tar_file.extractfile(filename))
                        returns.append(audio)
                    except OutOfVocabularyException:
                        pass
        return returns
