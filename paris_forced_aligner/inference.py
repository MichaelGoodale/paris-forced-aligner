from typing import List, Tuple

import torch

from paris_forced_aligner.model import PhonemeDetector
from paris_forced_aligner.pronunciation import AudioFile, PronunciationDictionary
from paris_forced_aligner.phonological import Utterance, Phone, Word, Silence
import paris_forced_aligner.ipa_data as ipa_data


def join_word_phones(word: List[Phone]) -> List[Phone]:
    for i in range(len(word) - 1):
        mid_point = (word[i].end + word[i+1].start) / 2
        word[i].end = mid_point
        word[i+1].start = mid_point
    return word


def words_to_indices(words: List[str], pron_dict: PronunciationDictionary) -> List[List[int]]:
    indices = []
    for word, word_spelling in words:
        indices.append([pron_dict.phonemic_mapping[x] for x in word_spelling])
    return indices

class ForcedAligner:

    def __init__(self, model: PhonemeDetector, n_beams: int = 50):
        self.model = model
        self.BEAMS = n_beams

    def align_tensors(self, words, X, y, pron_dict, wav_length, offset=0):    
        vocab_size = pron_dict.vocab_size()
        indices = words_to_indices(words, pron_dict)
        #(probs, transcription_left, states_per_timestep, in_word, word_idx, char_idx)
        beams = [(0, y, [], False, 0, 0)]
        for t in range(X.shape[0]):
            #Kinda funky candidates dict to prevent repeat paths based on prob
            #Shouldn't technically be based only on prob but likelihood is small of collisions
            candidates = {}
            for score, transcription, states, in_word, word_idx, char_idx in beams:
                p_blank = X[t, 0, 0].item() * 100
                if not (not in_word and word_idx >= len(indices)): #If we haven't had a blank after the very last char
                    current_state = transcription[0].item()
                    p_current_state = X[t, 0, current_state].item()
                    candidates[score + p_current_state] = (transcription, states + [current_state], True, word_idx, char_idx)

                if len(transcription) > 1 and in_word and word_idx < len(indices):
                    next_state = transcription[1].item()
                    p_next_state = X[t, 0, next_state].item()
                    if next_state != current_state:
                        if char_idx == len(indices[word_idx]) - 1: #We're at the end of a word
                            candidates[score + p_next_state] = (transcription[1:], states + [next_state], True, word_idx + 1, 0)
                        else:
                            candidates[score + p_next_state] = (transcription[1:], states + [next_state], True, word_idx, char_idx+1)
                    else:
                        #If we have two consecutive characters we need to force a blank
                        if char_idx == len(indices[word_idx]) - 1: #We're at the end of a word
                            candidates[score + p_blank] = (transcription[1:], states + [0], True, word_idx + 1, 0)
                        else:
                            candidates[score + p_blank] = (transcription[1:], states + [0], True, word_idx, char_idx+1)


                if not in_word or word_idx >= len(indices) or char_idx == len(indices[word_idx]) - 1:
                    #We can only have a blank in between words.
                    if in_word:
                        candidates[score + p_blank] = (transcription[1:], states + [0], False, word_idx + 1, 0)
                    else:
                        candidates[score + p_blank] = (transcription, states + [0], False, word_idx, char_idx)

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
                inference[i] = (inference[i][0], inference[i][1], wav_length + offset)

        inference = list(filter(lambda x: x[0] is not None, inference))

        return inference

    def to_utterance(self, inference: List[Tuple[str, int]], words: List[str], wav_length:int, pron_dict: PronunciationDictionary) -> Utterance:
        word_idx = 0
        utterance = []
        current_word = []
        for i, (phone, start, end) in enumerate(inference):
            current_word.append(Phone(phone, start, end))
            if words[word_idx][1] == [x.label for x in current_word]:
                current_word = join_word_phones(current_word) 
                utterance.append(Word(current_word, words[word_idx][0]))
                word_idx += 1
                current_word = []

        if current_word != []:
            current_word = join_word_phones(current_word) 
            utterance.append(Word(current_word, words[word_idx][0]))

        return Utterance(utterance)

    def ensemble_inference(self, audio: AudioFile, n: int):
        if n == 1:
            return self.model(audio.wav)
        if 320 % n != 0:
            raise ValueError(f"Invalid value of n={n}, 50%n must be 0")
        offset = 320 // n

        batched_wavs = torch.ones(n, audio.wav.shape[-1])
        padding_mask = torch.zeros(n, audio.wav.shape[-1])

        for i in range(n):
            start_at = i*offset
            length = audio.wav.shape[-1] - start_at
            batched_wavs[i, :length] = audio.wav[0, start_at:]
            padding_mask[i, :length] = 1

        X, lengths = self.model(batched_wavs, padding_mask=padding_mask)
        X = X.repeat_interleave(n, dim=0)
        for i in range(n):
            if i == 0:
                continue
            X[-i:, i, :] = -9999
            X[:, i, :] = X[:, i, :].roll(i, dims=0)
        X = X.sum(dim=1).unsqueeze(1)
        return X

    def align_file(self, audio: AudioFile):
        X = self.ensemble_inference(audio, 1)
        y = audio.tensor_transcription
        if self.model.multilingual:
            X = ipa_data.multilabel_ctc_log_prob(X)
        inference = self.align_tensors(audio.words, X, y, audio.pronunciation_dictionary, audio.wav.shape[1], audio.offset)
        return self.to_utterance(inference, audio.words, audio.wav.shape[1], audio.pronunciation_dictionary)

