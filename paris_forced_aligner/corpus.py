import os
import tarfile
from zipfile import ZipFile
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
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, skip_oov: bool = True, return_gold_labels: bool = False):
        self.corpus_path: str = corpus_path
        self.pronunciation_dictionary: PronunciationDictionary = pronunciation_dictionary
        self.skip_oov = skip_oov
        self.return_gold_labels = return_gold_labels

    def extract_files(self):
        raise NotImplementedError("extract_files must be implemented in a base class")

    def __iter__(self):
        return iter(self.extract_files(self.return_gold_labels))

class YoutubeCorpus(CorpusClass):

    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, language: str = 'en', audio_directory: Optional[str] = None, save_wavs: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary, False, False)
        self.youtube_files = []
        self.save_wavs = save_wavs

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

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("Youtube videos do not have gold standard phonemic labels")
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
            subtitle_file = os.path.join(self.dir, subtitle_file)

            audio = subtitle_file.replace(sub_file_ending, ".wav")
            captions = webvtt.read(os.path.join(self.dir, subtitle_file)).captions
            wav, sr = torchaudio.load(os.path.join(self.dir, audio))

            if self.save_wavs:
                idx_starts_ends: List[Tuple[int, int, int]] = []

            for i, (cap_time, cap_string) in enumerate(zip(captions[::2], captions[1::2])):
                cap_name = subtitle_file.replace(sub_file_ending, f"_{i}")
                transcription = cap_string.text.strip().upper()
                if transcription == '':
                    continue
                start = int(cap_time.start_in_seconds * sr)
                end = int(cap_time.end_in_seconds * sr)

                if self.save_wavs:
                    idx_starts_ends.append((i, int(start / sr  * 16000), int(end / sr * 16000)))
                yield AudioFile(cap_name, transcription, self.pronunciation_dictionary, wavobj=(wav[:, start:end], sr), raise_on_oov=self.skip_oov)

            if self.save_wavs:
                with open(subtitle_file.replace(sub_file_ending, '.txt'), 'w') as f:
                    for i, start, end in idx_starts_ends:
                        f.write(f"{i} {start} {end}\n")

    def stitch_youtube_utterances(self, audio_file_name: str, utterances: List[Utterance]):
        index_file = audio_file_name + '.txt'
        big_u_data = []
        with open(index_file, 'r') as f:
            for utterance, line in zip(utterances, f):
                i, start, end = line.strip().split(' ')
                utterance.offset(int(start))
                big_u_data += utterance.data
        return Utterance(big_u_data)

    def cleanup_and_recreate(self):
        self.cleanup()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.dir = self._temp_dir.name

    def cleanup(self):
        self._temp_dir.cleanup()


class LibrispeechCorpus(CorpusClass):
    def __init__(self, corpus_path: str, n_proc: int = cpu_count(), skip_oov: bool = True):
        super().__init__(corpus_path, LibrispeechDictionary(), skip_oov, False)
        self.n_proc = n_proc

    def extract_files(self, return_gold_labels):
        if return_gold_labels:
            raise NotImplementedError("Librispeech does not have gold standard phonemic labels")

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
                                with tar_file.extractfile(filename) as wav_obj:
                                    audio = AudioFile(filename, transcription, self.pronunciation_dictionary, fileobj=wav_obj, raise_on_oov=self.skip_oov)
                                yield audio
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
                        with tar_file.extractfile(filename) as wav_obj:
                            audio = AudioFile(filename, transcription, self.pronunciation_dictionary, fileobj=wav_obj, raise_on_oov=self.skip_oov)
                        returns.append(audio)
                    except OutOfVocabularyException:
                        pass
        return returns

class BuckeyeCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary, False, return_gold_labels)

    def _extract_from_zip(zip_dir, sound_zip):
        file_id = sound_zip.replace(".zip", "")
        with ZipFile(zip_dir.open(sound_zip)) as sound_dir:
            with sound_dir.open(file_id + ".wav") as wav_file:
                pass

    def extract_files(self, return_gold_labels):
        for d, s_d, files in os.walk(self.corpus_path):
            for f in filter(lambda x: x.endswith('.zip'), files):
                zip_path = os.path.join(d, f)
                with ZipFile(zip_path) as zip_dir:
                    for sound_zip in zip_dir.namelist():
                        audio, utterance = _extract_from_zip(zip_dir, sound_zip)
                        if return_gold_labels:
                            return audio, utterance
                        else:
                            return audio
