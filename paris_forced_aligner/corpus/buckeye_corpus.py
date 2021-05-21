import os
from zipfile import ZipFile
import random

import buckeye
import torchaudio
import tempfile

from paris_forced_aligner.pronunciation import LibrispeechDictionary, AudioFile, OutOfVocabularyException, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Silence, Word, Phone
from paris_forced_aligner.corpus import CorpusClass

class BuckeyeCorpus(CorpusClass):
    def __init__(self, corpus_path: str, pronunciation_dictionary: PronunciationDictionary, return_gold_labels: bool = False, relabel: bool = False):
        super().__init__(corpus_path, pronunciation_dictionary, return_gold_labels)
        self.relabel = relabel
        self.random = random.Random(1337)

    def _extract_from_zip(self, zip_dir, track_name, speaker_name):
        with ZipFile(zip_dir.open(f"{speaker_name}/{track_name}.zip")) as sound_dir:
            with sound_dir.open(f"{track_name}.wav") as wav_file:
                with tempfile.NamedTemporaryFile() as f:
                    f.write(wav_file.read())
                    wav, sr = torchaudio.load(f.name)
        return wav, sr

    def convert_to_arpabet(word, word_buckeye, sr=16000):
        '''Handles converting from seconds to frames and also handles any phoneme differences'''
        word.label = word.label.strip().replace('-', 'X').replace(' ', 'X').upper() 
        for i, phone in enumerate(word.phones):
            phone.start = int(phone.start * sr)
            phone.end = int(phone.end * sr)

            if phone.label is not None:
                phone.label = phone.label.upper()
            else:
                phone.label = PronunciationDictionary.silence
        return word

    def extract_files(self, return_gold_labels):
        for d, s_d, files in os.walk(self.corpus_path):
            speaker_files = list(filter(lambda x: x.endswith('.zip'), files))
            self.random.shuffle(speaker_files)
            for f in speaker_files:
                zip_path = os.path.join(d, f)
                speaker = buckeye.Speaker.from_zip(zip_path)
                with ZipFile(zip_path) as zip_dir:
                    for track in speaker:
                        paris_words = []
                        wav, sr = self._extract_from_zip(zip_dir, track.name, speaker.name)
                        print(track.name, speaker.name)
                        for word in track.words:
                            if isinstance(word, buckeye.containers.Pause):
                                if len(paris_words) >= 1 and isinstance(paris_words[-1], Silence):
                                    paris_words[-1].end = int(word.end * 16000)
                                else:
                                    paris_words.append(Silence(int(word.beg * 16000), int(word.end * 16000)))
                            else:
                                paris_word = Word([Phone(p.seg, p.beg, p.end) for p in word.phones], word.orthography)
                                #TODO: Add dynamic dictionary words w/ correct pronunciation (maybe)
                                if word.phones != []:
                                    paris_words.append(BuckeyeCorpus.convert_to_arpabet(paris_word, word))

                            if len(paris_words) >= 2 and isinstance(paris_words[-1], Silence) and paris_words[-1].duration > int(0.150*16000):
                                if paris_words[0].label == "<SIL>":
                                    paris_words = paris_words[:1]
                                utterance = Utterance(paris_words[:-1])

                                if utterance.words != []:
                                    try:
                                        audio = AudioFile(track.name, utterance.transcription, self.pronunciation_dictionary, wavobj=(wav[:, utterance.start:utterance.end], 16000))
                                    except OutOfVocabularyException:
                                        continue 

                                    if return_gold_labels:
                                        utt_start = utterance.start 
                                        for base in utterance.base_units:
                                            base.start -= utt_start
                                            base.end -= utt_start

                                        if self.relabel:
                                            for word in utterance.words:
                                                lexical_word = self.pronunciation_dictionary.lexicon[word.label]
                                                for p, lexical_phone in zip(word.phones, lexical_phone):
                                                    p.label = lexical_phone
                                            for base in utterance.base_units:
                                                if base.label not in self.pronunciation_dictionary.phonemic_inventory:
                                                    continue

                                        yield audio, utterance
                                    else:
                                        yield audio
                                paris_words = []
