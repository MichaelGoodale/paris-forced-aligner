import os
import time
from functools import partial

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

        if self.corpus.return_gold_labels:
            self.loss_fn = NLLLoss()
        else:
            self.loss_fn = CTCLoss(zero_infinity=True)
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

    def prepare_for_cross_entropy(self, transcriptions, X, X_lengths):
        transcriptions_mat = -100*torch.ones((len(X_lengths), X_lengths.max()), dtype=torch.long)
        #Ones are ignored indices, this makes it ignored by pytorch crossentropy

        for j, utterance in enumerate(transcriptions):
            for phone in utterance.base_units:
                if self.pretraining:
                    if phone.label == '<SIL>':
                        phone_idx = 1
                    elif phone.label == 'V':
                        phone_idx = 2
                    elif phone.label == 'C':
                        phone_idx = 3
                else:
                    phone_idx = self.corpus.pronunciation_dictionary.phonemic_mapping[phone.label]

                start = max(self.model.get_sample_in_idx(torch.tensor(phone.start)), 0)
                end = self.model.get_sample_in_idx(torch.tensor(phone.end))
                transcriptions_mat[j, start:end] = phone_idx
                if self.pretraining:
                    transcriptions_mat[j, start] = 0

        transcriptions_mat = transcriptions_mat.to(self.device)
        X = X.transpose(0, 1).transpose(1, 2)
        return X, transcriptions_mat

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
        if corpus.return_gold_labels:
            utt_batch = []
            for audio_file, utterance in self.corpus_iterator:
                if audio_file.wav.shape[1] < self.memory_max_length:
                    batch.append(audio_file)
                    utt_batch.append(utterance)

                if len(batch) == self.batch_size:
                    input_wavs, padding_mask = self.prepare_audio_batch(batch)
                    yield input_wavs, padding_mask, utt_batch, None
                    utt_batch = []
                    batch = []

            if len(batch) != 0:
                input_wavs, padding_mask = self.prepare_audio_batch(batch)
                yield input_wavs, padding_mask, utt_batch, None
                utt_batch = []
                batch = []
        else:
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
        if self.corpus.return_gold_labels:
            X, transcriptions_mat = self.prepare_for_cross_entropy(transcriptions, X, X_lengths)
            return self.loss_fn(X, transcriptions_mat) / self.accumulate_steps
        # CTC Loss 
        if self.model.multilingual:
            X = multilabel_ctc_log_prob(X, device=self.device)

        uniform_distribution = torch.ones(X.shape[0], X.shape[2], device=self.device) / X.shape[-1]
        kl_loss = torch.tensor(0.0, device=self.device)
        for i in range(X.shape[1]):
            kl_loss += (F.kl_div(X[:X_lengths[i], i, :], uniform_distribution[:X_lengths[i], :], reduction='batchmean'))
        kl_loss /= X.shape[1]
        return ((1-self.kl_ratio) * self.loss_fn(X, transcriptions, X_lengths, transcription_lengths) \
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

    def pretraining_validate(self):
        sums = 0
        n = 0
        for input_wavs, padding_mask, transcriptions, transcription_lengths in self.batched_audio_files(self.val_corpus):
            X, X_lengths = self.model(input_wavs, padding_mask=padding_mask, device=self.device)
            X, transcriptions_mat = self.prepare_for_cross_entropy(transcriptions, X, X_lengths)
            model_transcription = torch.argmax(X, dim=1)
            sums += torch.sum(model_transcription == transcriptions_mat)
            n += model_transcription.shape[0]*model_transcription.shape[-1]
        return sums / n

    def validate(self):
        if self.pretraining:
            return self.pretraining_validate()

        offsets = {"word_start":[],
                   "word_end":[],
                   "phone_start":[],
                   "phone_end":[]}

        for input_wavs, padding_mask, transcriptions, _ in self.batched_audio_files(self.val_corpus):
            X, X_lengths = self.model(input_wavs, padding_mask=padding_mask, device=self.device)
            for i, utterance in enumerate(transcriptions):
                wav_length = torch.sum(padding_mask[i, :])
                probs = X[:wav_length, i, :].unsqueeze(1)
                lexical_phones, words = self.corpus.pronunciation_dictionary.spell_sentence(utterance.transcription)
                y = torch.tensor([self.corpus.pronunciation_dictionary.phonemic_mapping[x] \
                                                    for x in lexical_phones], device=self.device)
                inference = self.forced_aligner.align_tensors(words, probs, y, self.corpus.pronunciation_dictionary, wav_length)
                aligned_utterance = self.forced_aligner.to_utterance(inference, words, wav_length, self.corpus.pronunciation_dictionary)


                for aligned_word, real_word in zip(aligned_utterance.words, utterance.words):
                    offsets["word_start"].append(abs(aligned_word.start - real_word.start))
                    offsets["word_end"].append(abs(aligned_word.end - real_word.end))
                    if len(aligned_word.phones) == len(real_word.phones):
                        for a, r in zip(aligned_word.phones, real_word.phones):
                            offsets["phone_start"].append(abs(a.start - r.start))
                            offsets["phone_end"].append(abs(a.end - r.end))
        for key, item in offsets.items():
            if len(item) > 0:
                offsets[key] = (sum(item)/len(item)) / 16 #milliseconds
            else:
                offsets[key] = 9999
        return offsets 

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



