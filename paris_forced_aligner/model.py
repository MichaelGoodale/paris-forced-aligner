import torch

from torch import nn
import torch.nn.functional as F
import fairseq

def load_wav2vec_model(filepath):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([filepath], task=None)
    model = model[0]
    return model


class PhonemeDetector(nn.Module):
    wav2vec_to_16khz = 320

    def __init__(self, filepath, vocab_size):
        super().__init__()
        self.wav2vec = load_wav2vec_model(filepath)
        self.fc = nn.Linear(768, vocab_size)#TODO: Find dynamic way to get this.

    def forward(self, wav_input_16khz, padding_mask=None):
        c = self.wav2vec.forward(wav_input_16khz, mask=False, features_only=True, padding_mask=padding_mask)
        x = c['x'].transpose(0,1)
        x = self.fc(x)
        if padding_mask is not None:
            x_lengths = (1 - c['padding_mask'].long()).sum(-1)
            return F.log_softmax(x, dim=-1), x_lengths
        return F.log_softmax(x, dim=-1)

    def get_idx_in_sample(self, idx: int) -> int:
        return PhonemeDetector.wav2vec_to_16khz * idx 
    
    def freeze_encoder(self):
        for name, param in self.wav2vec.named_parameters():
            if name.startswith('feature_extractor'):
                param.requires_grad = False

    def freeze_wav2vec(self):
        for name, param in self.wav2vec.named_parameters():
            param.requires_grad = False

    def unfreeze_wav2vec(self):
        for name, param in self.wav2vec.named_parameters():
            param.requires_grad = True
