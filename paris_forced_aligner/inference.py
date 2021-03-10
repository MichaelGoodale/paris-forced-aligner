import torch

from paris_forced_aligner.model import PhonemeDetector
from paris_forced_aligner.audio_data import AudioFile
from paris_forced_aligner.phonological import Utterance, Phone, Word, Silence

class ForcedAligner:

    def __init__(self, filepath: str, wav2vec_file: str, vocab_size: int, n_beams: int = 50):
        model = PhonemeDetector(wav2vec_file, vocab_size)
        model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        self.model = model
        self.BEAMS = n_beams


    def align_file(self, audio: AudioFile):
        X = self.model(audio.wav)
        y = audio.tensor_transcription

        beams = [(0, y, [])]

        for t in range(X.shape[0]):
            #Kinda funky candidates dict to prevent repeat paths based on prob
            #Shouldn't technically be based only on prob but likelihood is small of collisions
            candidates = {}
            for score, transcription, states in beams:
                p_current_state = X[t, 0, transcription[0]].item()
                candidates[score + p_current_state] = (transcription, states + [transcription[0].item()])

                if len(transcription) > 1:
                    p_next_state = X[t, 0, transcription[1]].item()
                    candidates[score + p_next_state] = (transcription[1:], states + [transcription[1].item()])

            beams = [(p, *candidates[p]) for p in sorted(candidates.keys(), reverse=True)[:self.BEAMS]]

        _, _, states = beams[0]

        inference = []
        old_x = None
        for t, x in enumerate(states):
            if old_x != x:
                inference.append((audio.pronunciation_dictionary.index_to_phone(x), self.model.get_idx_in_sample(t)))
            old_x = x

        word_idx = 0
        utterance = []
        current_word = []

        for i, (phone, start) in enumerate(inference):
            if i < len(inference) - 1:
                end = inference[i+1][1]
            else:
                end = audio.wav.shape[-1]

            if phone == "<SIL>":
                utterance.append(Silence(start, end))
                if current_word != []:
                    utterance.append(Word(current_word, audio.words[word_idx]))
                    word_idx += 1
                    current_word = []
            else:
                current_word.append(Phone(phone, start, end))

        return Utterance(utterance)

