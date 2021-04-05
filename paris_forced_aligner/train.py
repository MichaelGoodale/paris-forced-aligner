import os
import time

from paris_forced_aligner.corpus import LibrispeechCorpus, CorpusClass
from paris_forced_aligner.model import PhonemeDetector, AlignmentPretrainingModel

import torch
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.nn.functional as F

def prepare_audio_batch(batch, device):
    wav_lengths = torch.tensor([a.wav.shape[1] for a in batch])
    input_wavs = torch.ones((len(batch), wav_lengths.max()))
    padding_mask = torch.zeros((len(batch), wav_lengths.max()), dtype=torch.long)

    for i, (length, a) in enumerate(zip(wav_lengths, batch)):
        input_wavs[i, :length] = a.wav
        padding_mask[i, :length] = 1

    input_wavs = input_wavs.to(device)
    padding_mask = padding_mask.to(device)
    return input_wavs, padding_mask

def batched_audio_files(corpus, batch_size=1, device='cpu', memory_max_length=260000):
    batch = []
    if corpus.return_gold_labels:
        utt_batch = []
        for audio_file, utterance in corpus:
            if audio_file.wav.shape[1] < memory_max_length:
                batch.append(audio_file)
                utt_batch.append(utterance)

            if len(batch) == batch_size:
                with torch.no_grad():
                    input_wavs, padding_mask = prepare_audio_batch(batch, device)
                yield input_wavs, padding_mask, utt_batch, None
                utt_batch = []
                batch = []
    else:
        for audio_file in corpus:
            if audio_file.wav.shape[1] < memory_max_length:
                batch.append(audio_file)

            if len(batch) == batch_size:
                with torch.no_grad():
                    input_wavs, padding_mask = prepare_audio_batch(batch, device)

                    transcription_lengths = torch.tensor([a.tensor_transcription.shape[0] for a in batch])
                    transcriptions = torch.zeros((batch_size, transcription_lengths.max()))
                    for i, (length, a) in enumerate(zip(transcription_lengths, batch)):
                        transcriptions[i, :length] = a.tensor_transcription

                batch = []

                transcriptions = transcriptions.to(device)
                transcription_lengths = transcription_lengths.to(device)
                yield input_wavs, padding_mask, transcriptions, transcription_lengths

def train(model: PhonemeDetector, 
        corpus: CorpusClass,
        output_directory:str = "models",
        batch_size:int = 20,
        lr:float = 3e-5,
        accumulate_steps: int = 1,
        n_steps:int = 30000,
        unfreeze_after:int = 10000,
        output_model_every:int = 1000,
        checkpoint=None,
        device:str = 'cpu'):
    ''' Example usage:

        model = PhonemeDetector('../wav2vec2_models/wav2vec_small.pt', pronunciation_dictionary.vocab_size())
        model.load_state_dict(torch.load('models/final_output.pt'))
        train(model)
    '''
    os.makedirs(output_directory, exist_ok=True)
    model.train()
    model.freeze_wav2vec()
    ctc_loss_fn = CTCLoss()
    cross_entropy = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    unfrozen = False
    with open(f"{output_directory}/log.txt", 'a') as f:
        i = 0

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            i = checkpoint['steps']

        while i // accumulate_steps < n_steps:
            for input_wavs, padding_mask, transcriptions, transcription_lengths in batched_audio_files(corpus, batch_size=batch_size, device=device):
                if not unfrozen and (i//accumulate_steps) >= unfreeze_after:
                    model.unfreeze_wav2vec()
                    model.freeze_encoder()
                    unfrozen = True


                X, X_lengths = model(input_wavs, padding_mask=padding_mask)

                if corpus.return_gold_labels:
                    transcriptions_mat = -100*torch.ones((len(X_lengths), X_lengths.max()), dtype=torch.long)
                    #Ones are ignored indices, this makes it ignored by pytorch crossentropy

                    for j, utterance in enumerate(transcriptions):
                        for phone in utterance.base_units:
                            phone_idx = corpus.pronunciation_dictionary.phonemic_mapping[phone.label]
                            start = model.get_sample_in_idx(phone.start)
                            end = model.get_sample_in_idx(phone.end)
                            transcriptions_mat[j, start:end] = phone_idx

                    transcriptions_mat = transcriptions_mat.to(device)
                    X = X.transpose(0, 1).transpose(1, 2)
                    loss = cross_entropy(X, transcriptions_mat) / accumulate_steps
                else:
                    loss = ctc_loss_fn(X, transcriptions, X_lengths, transcription_lengths) / accumulate_steps

                loss.backward()
                losses.append(loss.item())

                if i % accumulate_steps == 0 and i != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Step {i/accumulate_steps}: {sum(losses)}")
                    f.write(f'{i/accumulate_steps} {sum(losses)}\n')
                    losses = []

                if i > 0 and (i//accumulate_steps) % output_model_every == 0:
                    torch.save({
                        'steps': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        },
                    f"{output_directory}/{i}_model.pt")

                i += 1

                if i > n_steps: 
                    break

    torch.save(model.state_dict(), f'{output_directory}/final_output.pt')
