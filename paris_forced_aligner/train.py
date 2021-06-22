import os
import time
from functools import partial
from typing import Mapping

from tqdm.autonotebook import tqdm

import torch
from torch.nn import CTCLoss, NLLLoss
import torch.nn.functional as F

from paris_forced_aligner.corpus import LibrispeechCorpus, CorpusClass
from paris_forced_aligner.model import PhonemeDetector
from paris_forced_aligner.inference import ForcedAligner
from paris_forced_aligner.ipa_data import multilabel_ctc_log_prob

class Trainer:

    def __init__(self,
            model: PhonemeDetector,
            corpus: CorpusClass,
            val_corpus: CorpusClass = None,
            pretraining = False,
            kl_ratio = 0.10,
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
        self.pretraining = pretraining
        self.output_model_every = output_model_every
        self.output_directory = output_directory
        self.accumulate_steps = accumulate_steps

        self.batch_size = batch_size
        self.lr = lr
        self.kl_ratio = kl_ratio
        self.accumulate_steps = accumulate_steps
        self.thaw_after = thaw_after

        os.makedirs(output_directory, exist_ok=True)
        self.model = model
        self.model.train()
        self.freeze()
        self.forced_aligner = ForcedAligner(self.model, n_beams=10)

        self.corpus = corpus
        self.val_corpus = val_corpus

        self.loss_fn = CTCLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.memory_max_length = 300000
        self.epoch = 0

        if checkpoint is not None:
            self.epoch = checkpoint["epoch"]
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
        transcriptions = torch.zeros((len(batch), transcription_lengths.max()))
        for i, (length, a) in enumerate(zip(transcription_lengths, batch)):
            transcriptions[i, :length] = a.tensor_transcription
        transcriptions = transcriptions.to(self.device)
        transcription_lengths = transcription_lengths.to(self.device)
        return input_wavs, padding_mask, transcriptions, transcription_lengths

    def batched_audio_files(self, corpus):
        batch = []
        self.corpus_iterator = tqdm(corpus)
        for audio_file in self.corpus_iterator:
            if audio_file.wav.shape[1] < self.memory_max_length:
                batch.append(audio_file)

            if len(batch) == self.batch_size:
                yield self.prepare_ctc_batch(batch)
                batch = []

        if len(batch) != 0:
            yield self.prepare_ctc_batch(batch)
            batch = []

    def calculate_loss(self, X, X_lengths, transcriptions, transcription_lengths):
        if self.model.multilingual:
            X = multilabel_ctc_log_prob(X, device=self.device)
        uniform_distribution = torch.full_like(X, 1. / X.shape[-1], device=self.device)
        kl_loss = F.kl_div(X, uniform_distribution, reduction='batchmean')
        return ((1-self.kl_ratio) * self.loss_fn(X, transcriptions, X_lengths, transcription_lengths)
               + self.kl_ratio * kl_loss) / self.accumulate_steps

    def update_progress_bar(self, prefix_string, postfix_stats):
        '''Adds string to current progress bar, and allow it to fail'''
        self.corpus_iterator.set_description(prefix_string)
        self.corpus_iterator.set_postfix(postfix_stats)

    def save_checkpoint(self, step):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'epoch': self.epoch
           },
        f"{self.output_directory}/{step}_model.pt")


    def validate(self) -> Mapping[str, float]:
        for input_wavs, padding_mask, transcriptions, transcription_lengths in self.batched_audio_files(self.val_corpus):
            pass

    def train(self, step=0):
        losses = []
        while step < self.total_steps:
            accumulate_step = 0

            if self.val_corpus is not None:
                with torch.no_grad():
                    metrics = self.validate()
                s = ""
                for metric, val in metrics.items():
                    s += f", {metric} = {val:.4g}"
                print(f"Epoch: {self.epoch} Step: {step}/{self.total_steps}, the mean loss is {sum(losses)/max(1, len(losses)):.4f}"+s)
            else:
                print(f"Epoch: {self.epoch} Step: {step}/{self.total_steps}, the mean loss is {sum(losses)/max(1, len(losses)):.4f}")

            losses = []
            for input_wavs, padding_mask, transcriptions, transcription_lengths in self.batched_audio_files(self.corpus):
                if self.frozen and step > self.thaw_after:
                    self.thaw()

                X, X_lengths = self.model(input_wavs, padding_mask=padding_mask, device=self.device)
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
                    self.update_progress_bar(f"Epoch {self.epoch+1}", {"Previous loss": sum(losses[-self.accumulate_steps:])/len(losses[-self.accumulate_steps:]), "Step": step})

                if step >= self.total_steps: 
                    break
            self.epoch += 1

        self.save_checkpoint(step)
