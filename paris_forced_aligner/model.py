import torch

from torch import nn
import torch.nn.functional as F
import fairseq

def load_wav2vec_model(filepath):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([filepath], task=None)
    model = model[0]
    return model


class Upscaler(nn.Module):
    def __init__(self, input_dim, internal_dim):
        super().__init__()
        self.time_transform = nn.Upsample(scale_factor=2, mode='linear')
        self.conv1 = nn.Conv1d(input_dim, internal_dim, 4)
        self.batch_norm1 = nn.GroupNorm(internal_dim, internal_dim)
        self.conv2 = nn.Conv1d(internal_dim, internal_dim, 2)
        self.batch_norm2 = nn.GroupNorm(internal_dim, internal_dim)

    def forward(self, x):
        x = self.time_transform(x)
        x = F.gelu(self.batch_norm1(self.conv1(x)))
        x = self.time_transform(x)
        x = F.gelu(self.batch_norm2(self.conv2(x)))
        return x

    def get_upscaled_length(self, length: int) -> int:
        return 2*(2*length - 4 +1) - 2 + 1

    def invert_upscale_length(self, length: int) -> int:
        return ((idx + 7) / 4)

class PhonemeDetector(nn.Module):
    wav2vec_to_16khz = 320

    def __init__(self, filepath, vocab_size, internal_vector_size=256):
        super().__init__()
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.wav2vec = load_wav2vec_model(filepath)
        self.upscaler = Upscaler(768, internal_vector_size)
        self.fc = nn.Linear(internal_vector_size, vocab_size)

    def forward(self, wav_input_16khz, padding_mask=None):
        c = self.wav2vec.forward(wav_input_16khz, mask=False, features_only=True, padding_mask=padding_mask)
        #c['x'] = (N, L, C)
        x = self.upscaler(c['x'].transpose(1,2))
        x = x.transpose(1,2).transpose(0,1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        if padding_mask is not None:
            x_lengths = self.upscaler.get_upscaled_length((1 - c['padding_mask'].long()).sum(-1))
            return x, x_lengths
        return x

    def get_idx_in_sample(self, idx: int) -> int:
        return self.upscaler.invert_upscale_length(idx) * PhonemeDetector.wav2vec_to_16khz 

    def get_sample_in_idx(self, sample_idx: int) -> int:
        return int(self.upscaler.get_upscaled_length(sample_idx / PhonemeDetector.wav2vec_to_16khz))
    
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

def AlignmentPretrainingModel(PhonemeDetector):
    def __init__(self, filepath, internal_dim=256):
        super().__init__(filepath, 3, internal_dim)
