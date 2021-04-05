import torch

from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class Upscaler(nn.Module):
    def __init__(self, input_dim, internal_dim):
        super().__init__()
        self.time_transform = nn.Upsample(scale_factor=2, mode='linear')
        self.conv1 = nn.Conv1d(input_dim, internal_dim, 4)
        self.batch_norm1 = nn.InstanceNorm1d(internal_dim)
        self.conv2 = nn.Conv1d(internal_dim, internal_dim, 2)
        self.batch_norm2 = nn.InstanceNorm1d(internal_dim)

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
    def __init__(self, filepath, vocab_size, internal_vector_size=256):
        super().__init__()
        self.filepath = filepath
        self.vocab_size = vocab_size
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")#, apply_spec_augment=False)
        self.upscaler = Upscaler(self.wav2vec.config.hidden_size, internal_vector_size)
        self.fc = nn.Linear(internal_vector_size, vocab_size)

    def forward(self, wav_input_16khz, padding_mask=None):

        if self.wav2vec.config.feat_extract_norm == "layer":
            c = self.wav2vec(wav_input_16khz, attention_mask=padding_mask)
        else:
            if padding_mask is not None:
                wav_input_16khz = wav_input_16khz * padding_mask
            c = self.wav2vec(wav_input_16khz)

        x = self.upscaler(c['last_hidden_state'].transpose(1,2))
        x = x.transpose(1,2).transpose(0,1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        if padding_mask is not None:
            x_lengths = self.upscaler.get_upscaled_length(self.wav2vec._get_feat_extract_output_lengths((padding_mask).sum(-1)))
            return x, x_lengths
        return x

    def get_idx_in_sample(self, idx: int) -> int:
        return self._invert_feat_extract_output_lengths(self.upscaler.invert_upscale_length(idx))

    def _invert_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the input length after having applied convolutional layers (adapted from HuggingFace code)
        """

        def _conv_out_length(output_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            # could be ameliorated to account for the // in the stride part.
            return stride*(output_length - 1 ) + kernel_size 

        for kernel_size, stride in zip(self.wav2vec.config.conv_kernel[::-1], self.wav2vec.config.conv_stride[::-1]):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


    def get_sample_in_idx(self, sample_idx: int) -> int:
        return self.upscaler.get_upscaled_length(self.wav2vec._get_feat_extract_output_lengths(sample_idx))
    
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
