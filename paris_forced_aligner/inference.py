from typing import List, Tuple

import torch

from paris_forced_aligner.model import PhonemeDetector
from paris_forced_aligner.audio_data import AudioFile, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Phone, Word, Silence


class ForcedAligner:

    def __init__(self, model: PhonemeDetector, n_beams: int = 50):
        self.model = model
        self.BEAMS = n_beams

    def align_tensors(self, X, y, pron_dict, wav_length, offset=0):    
        vocab_size = pron_dict.vocab_size()
        y = y.repeat_interleave(2)
        y[1::2] += vocab_size - 1

        beams = [(0, y, [], None)]
        for t in range(X.shape[0]):
            #Kinda funky candidates dict to prevent repeat paths based on prob
            #Shouldn't technically be based only on prob but likelihood is small of collisions
            candidates = {}
            for score, transcription, states, prev_non_blank in beams:
                current_state = transcription[0].item()
                p_current_state = X[t, 0, current_state].item()
                candidates[score + p_current_state] = (transcription, states + [current_state], current_state)

                if len(transcription) > 1 and prev_non_blank is not None:
                    next_state = transcription[1].item()
                    p_next_state = X[t, 0, next_state].item()
                    candidates[score + p_next_state] = (transcription[1:], states + [next_state], next_state)

                p_blank = X[t, 0, 0].item()

                candidates[score + p_blank] = (transcription, states + [0], prev_non_blank)

            beams = [(p, *candidates[p]) for p in sorted(candidates.keys(), reverse=True)[:self.BEAMS]]

        _, _, states, _ = beams[0]

        inference = []
        old_x = None

        for t, x in enumerate(states):
            if x == 0:#Skip blanks
                continue 

            if old_x != x:
                if x < vocab_size:#This is an openining of a phone
                    phone = pron_dict.index_to_phone(x)
                    start_time_16khz = int((t / X.shape[0]) * wav_length) + offset
                else:#this is the closing of a phone
                    end_time_16khz = int((t / X.shape[0]) * wav_length) + offset
                    inference.append((phone, start_time_16khz, end_time_16khz))
            old_x = x

        return inference

    def to_utterance(self, inference: List[Tuple[str, int]], words: List[str], wav_length:int, pron_dict: PronunciationDictionary) -> Utterance:
        word_idx = 0
        utterance = []
        current_word = []
        for i, (phone, start, end) in enumerate(inference):
            current_word.append(Phone(phone, start, end))
            if pron_dict.spelling(words[word_idx]) == [x.label for x in current_word]:
                utterance.append(Word(current_word, words[word_idx]))
                word_idx += 1
                current_word = []

        if current_word != []:
            utterance.append(Word(current_word, words[word_idx]))

        return Utterance(utterance)

    def align_file(self, audio: AudioFile):
        X = self.model(audio.wav)
        y = audio.tensor_transcription
        inference = self.align_tensors(X, y, audio.pronunciation_dictionary, audio.wav.shape[1], audio.offset)
        return self.to_utterance(inference, audio.words, audio.pronunciation_dictionary)

