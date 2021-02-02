import torch
import textgrid

from model import PhonemeDetector
from audio_data import AudioFile

class ForceAligner:

    def __init__(self, filepath: str, wav2vec_file: str, vocab_size: int, n_beams: int = 50, allow_deleted_phones: bool = False):
        model = PhonemeDetector(wav2vec_file, vocab_size)
        model.load_state_dict(torch.load(filepath))
        self.model = model
        self.BEAMS = n_beams
        self.allow_deleted_phones = allow_deleted_phones


    def align_file(self, audio: AudioFile):
        X = self.model(audio.wav)
        y = audio.tensor_transcription

        beams = [(0, y, []) for _ in range(self.BEAMS)]
        for t in range(X.shape[0]):
            candidates = []
            for score, transcription, states in beams:
                p_current_state = X[t, 0, transcription[0]].item()
                candidates.append((score + p_current_state, transcription, states + [transcription[0].item()]))

                if len(transcription) > 1:
                    p_next_state = X[t, 0, transcription[1]].item()
                    candidates.append((score + p_next_state, transcription[1:], states + [transcription[1].item()]))

                if self.allow_deleted_phones and len(transcription) > 2:
                    p_next_state = X[t, 0, transcription[2]].item()
                    candidates.append((score + p_next_state, transcription[2:], states + [transcription[2].item()]))

            beams = sorted(candidates, reverse=True, key=lambda x: x[0])[:5]

        _, _, states = beams[0]

        inference = []
        old_x = None
        for t, x in enumerate(states):
            if old_x != x:
                inference.append((audio.index_to_phone(x), self.model.get_idx_in_sample(t)))
            old_x = x

        inference_return = []
        for i, (phone, time) in enumerate(inference):
            if i < len(inference) - 1:
                end_time = inference[i+1][1]
            else:
                end_time = audio.wav.shape[-1]
            inference_return.append((phone, time, end_time))

        return inference_return

    def output_textgrid(self, inference, output_file):
        tg = textgrid.TextGrid()
        phones = textgrid.IntervalTier()
        for phone, start_time, end_time in inference:
            phones.add(start_time / 16000, end_time / 16000, phone)

        tg.append(phones)
        with open(output_file, 'w') as f:
            tg.write(f)
