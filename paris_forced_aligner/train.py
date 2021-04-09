import os
import time

from paris_forced_aligner.corpus import LibrispeechCorpus, CorpusClass
from paris_forced_aligner.model import PhonemeDetector, AlignmentPretrainingModel

import torch
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.nn.functional as F

class Trainer:

    def __init__(self,
            model: PhonemeDetector,
            corpus: CorpusClass,
            output_directory:str = "models",
            batch_size:int = 20,
            lr:float = 3e-5,
            accumulate_steps: int = 1,
            total_steps:int = 30000,
            thaw_after:int = 10000,
            output_model_every:int = 1000,
            checkpoint=None,
            device:str = 'cpu'):

        self.device = device
        self.total_steps = total_steps
        self.output_model_every = output_model_every
        self.output_directory = output_directory
        self.accumulate_steps = accumulate_steps

        self.batch_size = batch_size
        self.lr = lr
        self.accumulate_steps = accumulate_steps
        self.thaw_after = thaw_after

        os.makedirs(output_directory, exist_ok=True)
        self.model = model
        self.model.train()
        self.freeze()

        self.corpus = corpus

        self.loss_fn = CTCLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.memory_max_length = 260000

    def freeze(self):
        self.frozen = True
        self.model.freeze_wav2vec()

    def thaw(self):
        self.frozen = False
        self.model.unfreeze_wav2vec()
        self.model.freeze_encoder()

    def prepare_audio_batch(self, batch):
        wav_lengths = torch.tensor([a.wav.shape[1] for a in batch])
        input_wavs = torch.ones((len(batch), wav_lengths.max()))
        padding_mask = torch.zeros((len(batch), wav_lengths.max()), dtype=torch.long)

        for i, (length, a) in enumerate(zip(wav_lengths, batch)):
            input_wavs[i, :length] = a.wav
            padding_mask[i, :length] = 1

        input_wavs = input_wavs.to(self.device)
        padding_mask = padding_mask.to(self.device)
        return input_wavs, padding_mask

    def prepare_ctc_batch(self, batch):
        input_wavs, padding_mask = self.prepare_audio_batch(batch)
        transcription_lengths = torch.tensor([a.tensor_transcription.shape[0] for a in batch])
        transcriptions = torch.zeros((self.batch_size, transcription_lengths.max()))
        for i, (length, a) in enumerate(zip(transcription_lengths, batch)):
            transcriptions[i, :length] = a.tensor_transcription
        transcriptions = transcriptions.to(self.device)
        transcription_lengths = transcription_lengths.to(self.device)
        return input_wavs, padding_mask, transcriptions, transcription_lengths

    def batched_audio_files(self):
        batch = []
        if self.corpus.return_gold_labels:
            utt_batch = []
            for audio_file, utterance in self.corpus:
                if audio_file.wav.shape[1] < self.memory_max_length:
                    batch.append(audio_file)
                    utt_batch.append(utterance)

                if len(batch) == self.batch_size:
                    input_wavs, padding_mask = prepare_audio_batch(batch)
                    yield input_wavs, padding_mask, utt_batch, None
                    utt_batch = []
                    batch = []

            if len(batch) != 0:
                input_wavs, padding_mask = prepare_audio_batch(batch)
                yield input_wavs, padding_mask, utt_batch, None
                utt_batch = []
                batch = []
        else:
            for audio_file in self.corpus:
                if audio_file.wav.shape[1] < self.memory_max_length:
                    batch.append(audio_file)

                if len(batch) == self.batch_size:
                    yield self.prepare_ctc_batch(batch)
                    batch = []

            if len(batch) != 0:
                yield self.prepare_ctc_batch(batch)
                batch = []

    def calculate_loss(self, X, X_lengths, transcriptions, transcription_lengths):
        if self.corpus.return_gold_labels:
            transcriptions_mat = -100*torch.ones((len(X_lengths), X_lengths.max()), dtype=torch.long)
            #Ones are ignored indices, this makes it ignored by pytorch crossentropy

            for j, utterance in enumerate(transcriptions):
                for phone in utterance.base_units:
                    phone_idx = self.corpus.pronunciation_dictionary.phonemic_mapping[phone.label]
                    start = model.get_sample_in_idx(phone.start)
                    end = model.get_sample_in_idx(phone.end)
                    transcriptions_mat[j, start:end] = phone_idx

            transcriptions_mat = transcriptions_mat.to(self.device)
            X = X.transpose(0, 1).transpose(1, 2)
            return self.loss_fn(X, transcriptions_mat) / self.accumulate_steps
        # CTC Loss 
        return self.loss_fn(X, transcriptions, X_lengths, transcription_lengths) / self.accumulate_steps

    def save_checkpoint(self, step):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step
           },
        f"{self.output_directory}/{step}_model.pt")

    def train(self):
        step = 0

        while step < self.total_steps:
            accumulate_step = 0
            losses = []
            for input_wavs, padding_mask, transcriptions, transcription_lengths in self.batched_audio_files():
                if self.frozen and step > self.thaw_after:
                    self.thaw()

                X, X_lengths = self.model(input_wavs, padding_mask=padding_mask)
                loss = self.calculate_loss(X, X_lengths, transcriptions, transcription_lengths)
                accumulate_step += 1
                loss.backward()
                losses.append(loss.item())

                if accumulate_step == self.accumulate_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accumulate_step = 0
                    step += 1
                    if step % self.output_model_every == 0:
                        self.save_checkpoint(step)

                if step % 1000 == 0:
                    mean_loss = sum(losses)/len(losses)
                    print(f"After {step} steps, the mean loss is {mean_loss:.4f}")

                if step >= self.total_steps: 
                    break

        self.save_checkpoint(step)



