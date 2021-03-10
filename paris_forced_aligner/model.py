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

    def __init__(self, filepath, vocab_size, upscale=2, internal_vector_size=256, kernel_size=4):
        super().__init__()
        self.wav2vec = load_wav2vec_model(filepath)
        #TODO: Find dynamic way to get this.
        self.time_transform = nn.Upsample(scale_factor=upscale, mode='linear')
        #Kernel_size * 25 ms = receptive field of conv
        self.conv1 = nn.Conv1d(768, internal_vector_size, kernel_size*upscale)
        self.batch_norm1 = nn.BatchNorm1d(internal_vector_size)

        self.upscale = upscale 
        self.kernel_size = kernel_size
        self.fc = nn.Linear(internal_vector_size, vocab_size)

    def get_upscaled_length(self, length: int) -> int:
        return self.upscale*length - self.kernel_size*self.upscale + 1

    def forward(self, wav_input_16khz, padding_mask=None):
        c = self.wav2vec.forward(wav_input_16khz, mask=False, features_only=True, padding_mask=padding_mask)
        #c['x'] = (N, L, C)
        x = self.time_transform(c['x'].transpose(1,2))
        x = F.gelu(self.batch_norm1(self.conv1(x)))
        x = x.transpose(1,2).transpose(0,1)
        x = self.fc(x)

        if padding_mask is not None:
            x_lengths = self.get_upscaled_length((1 - c['padding_mask'].long()).sum(-1))
            return F.log_softmax(x, dim=-1).to(torch.float64), x_lengths
        return F.log_softmax(x, dim=-1)

    def get_idx_in_sample(self, idx: int) -> int:
        return PhonemeDetector.wav2vec_to_16khz * self.get_upscaled_length(idx)

    def get_sample_in_idx(self, sample_idx: int) -> int:
        return self.get_upscaled_length(sample_idx // PhonemeDetector.wav2vec_to_16khz)
    
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

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)
