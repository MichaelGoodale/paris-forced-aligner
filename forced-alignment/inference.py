import torch

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
        inference = []
        for t in range(X.shape[0]):
            print(torch.argmax(X[t, 0, :]))
            p_current_state = X[t, 0, y[0]]
            p_split = X[t, 0, 0]
            if p_split > p_current_state:
                inference.append(0)
                y = y[1:]
            else:
                inference.append(y[0])
        print(inference)



f = ForceAligner('models/5500_output.pt', '../wav2vec2_models/wav2vec_small.pt', LibrispeechFile.vocab_size())
for audio_file in LibrispeechCorpus('../data/librispeech-clean-100.tar.gz'):
    f.align_file(audio_file)
    break

