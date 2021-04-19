from typing import List, Tuple

import torch

from paris_forced_aligner.model import PhonemeDetector
from paris_forced_aligner.audio_data import AudioFile, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Phone, Word, Silence


def join_word_phones(word: List[Phone]) -> List[Phone]:
    for i in range(len(word) - 1):
        mid_point = (word[i].end + word[i+1].start) / 2
        word[i].end = mid_point
        word[i+1].start = mid_point
    return word


def words_to_indices(words: List[str], pron_dict: PronunciationDictionary) -> List[List[int]]:
    indices = []
    for word in words:
        indices.append([pron_dict.phonemic_mapping[x] for x in pron_dict.spelling(word)])
    return indices

class ForcedAligner:

    def __init__(self, model: PhonemeDetector, n_beams: int = 50):
        self.model = model
        self.BEAMS = n_beams

    def align_tensors(self, words, X, y, pron_dict, wav_length, offset=0):    
        vocab_size = pron_dict.vocab_size()
        indices = words_to_indices(words, pron_dict)
        #(probs, transcription_left, states_per_timestep, in_word, word_idx, char_idx)
        beams = [(0, y, [], False, -1, 0)]
        for t in range(X.shape[0]):
            #Kinda funky candidates dict to prevent repeat paths based on prob
            #Shouldn't technically be based only on prob but likelihood is small of collisions
            candidates = {}
            for score, transcription, states, in_word, word_idx, char_idx in beams:

                if not (not in_word and word_idx >= len(indices)): #If we haven't had a blank after the very last char
                    current_state = transcription[0].item()
                    p_current_state = X[t, 0, current_state].item()
                    candidates[score + p_current_state] = (transcription, states + [current_state], True, word_idx, char_idx)

                if len(transcription) > 1 and in_word and word_idx < len(indices):
                    next_state = transcription[1].item()
                    p_next_state = X[t, 0, next_state].item()
                    if char_idx == len(indices[word_idx]) - 1: #We're at the end of a word
                        candidates[score + p_next_state] = (transcription[1:], states + [next_state], True, word_idx + 1, 0)
                    else:
                        candidates[score + p_next_state] = (transcription[1:], states + [next_state], True, word_idx, char_idx+1)

                if not in_word or word_idx >= len(indices) or char_idx == len(indices[word_idx]) - 1:
                    #We can only have a blank in between words.
                    p_blank = X[t, 0, 0].item() 
                    candidates[score + p_blank] = (transcription, states + [0], False, word_idx + int(in_word), 0)

            beams = [(p, *candidates[p]) for p in sorted(candidates.keys(), reverse=True)[:self.BEAMS]]
        _, _, states, _, _, _ = beams[0]

        inference = []
        old_x = None

        for t, x in enumerate(states):
            if old_x != x:
                if x == 0:
                    phone = None
                else:
                    phone = pron_dict.index_to_phone(x)
                time_16khz = int((t / X.shape[0]) * wav_length) + offset
                inference.append((phone, time_16khz))
            old_x = x

        for i in range(len(inference)):
            if i < len(inference) - 1:
                inference[i] = (inference[i][0], inference[i][1], inference[i+1][1])
            else:
                inference[i] = (inference[i][0], inference[i][1], wav_length)

        infernce = list(filter(lambda x: x[0] is not None, inference))
        return inference

    def to_utterance(self, inference: List[Tuple[str, int]], words: List[str], wav_length:int, pron_dict: PronunciationDictionary) -> Utterance:
        word_idx = 0
        utterance = []
        current_word = []
        for i, (phone, start, end) in enumerate(inference):
            current_word.append(Phone(phone, start, end))
            if pron_dict.spelling(words[word_idx]) == [x.label for x in current_word]:
                current_word = join_word_phones(current_word) 
                utterance.append(Word(current_word, words[word_idx]))
                word_idx += 1
                current_word = []

        if current_word != []:
            current_word = join_word_phones(current_word) 
            utterance.append(Word(current_word, words[word_idx]))

        return Utterance(utterance)

    def align_file(self, audio: AudioFile):
        X = self.model(audio.wav)
        y = audio.tensor_transcription
        inference = self.align_tensors(audio.words, X, y, audio.pronunciation_dictionary, audio.wav.shape[1], audio.offset)
        return self.to_utterance(inference, audio.words, audio.wav.shape[1], audio.pronunciation_dictionary)

