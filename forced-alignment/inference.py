import torch
import torchaudio
import textgrid
from torchaudio.transforms import Spectrogram

import matplotlib.pyplot as plt

from model import PhonemeDetector
from audio_data import AudioFile, LibrispeechFile
from corpus import LibrispeechCorpus

class ForceAligner:

    def __init__(self, filepath: str, wav2vec_file: str, vocab_size: int):
        model = PhonemeDetector(wav2vec_file, vocab_size)
        model.load_state_dict(torch.load(filepath))
        self.model = model


    def align_file(self, audio: AudioFile):
        X = self.model(audio.wav)
        y = audio.tensor_transcription
        inference = [('<SIL>', 0)]

        for t in range(X.shape[0]):
            if len(y) == 1:
                break

            p_current_state = X[t, 0, y[0]]
            p_next_state = X[t, 0, y[1]]
            if p_next_state > p_current_state:
                inference.append((audio.index_to_phone(y[1].item()), self.model.get_idx_in_sample(t)))
                y = y[1:]

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
            print(start_time, end_time)
            phones.add(start_time / 16000, end_time / 16000, phone)

        tg.append(phones)
        with open(output_file, 'w') as f:
            tg.write(f)

    def plot_specgram(inference):
        spec = Spectrogram()
        specgram = spec(audio.wav)
        plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')
        for phone, start_time, end_time in inference:
            plt.axvline(x=start_time / 200)
            plt.text(((end_time - start_time) / 2 + start_time) / 200, 0.5, phone)
        plt.show()



f = ForceAligner('models/final_output.pt', '../wav2vec2_models/wav2vec_small.pt', LibrispeechFile.vocab_size())
for audio_file in LibrispeechCorpus('../data/librispeech-clean-100.tar.gz', 1):
    f.output_textgrid(f.align_file(audio_file), 'file.textgrid')
    torchaudio.save("file.wav", audio_file.wav, 16000)
    break

