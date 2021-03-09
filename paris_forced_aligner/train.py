import os

from paris_forced_aligner.corpus import LibrispeechCorpus, CorpusClass
from paris_forced_aligner.model import PhonemeDetector

import torch
from torch.nn import CTCLoss, CrossEntropyLoss

def get_cross_entropy_label(X):
    '''Ensure to pass detached X'''
    with torch.no_grad():
        y_zero = torch.argmax(X, dim=-1)
        y_no_zero = torch.argmax(X[:, 1:], dim=-1) + 1
        
        temp_y_zero = torch.clone(y_zero)
        for idx in torch.nonzero(y_zero == 0):#Iterate over zeroes
            if idx - 1 >= 0 and y_zero[idx - 1] == 0:
                temp_y_zero[idx] = y_no_zero[idx]

        y_zero = temp_y_zero

        for idx in torch.nonzero(y_zero == 0):
            if idx - 1 >= 0 and idx + 1 < y_zero.shape[0]:
                #delete blanks that aren't surrounded by the same number
                if y_zero[idx-1] != y_zero[idx+1]:
                    y_zero[idx] = y_no_zero[idx]
            else:
                #delete edge blanks
                y_zero[idx] = y_no_zero[idx]
    return y_zero

def self_framewise_loss(X):
    return cross_entropy_fn(X.squeeze(), get_cross_entropy_label(X.detach().squeeze()))

def batched_audio_files(corpus, batch_size=1, device='cpu'):
    batch = []
    for audio_file in corpus:
        batch.append(audio_file)
        if len(batch) == batch_size:
            wav_lengths = torch.tensor([a.wav.shape[1] for a in batch])
            input_wavs = torch.zeros((batch_size, wav_lengths.max()))
            padding_mask = torch.ones((batch_size, wav_lengths.max()))

            for i, (length, a) in enumerate(zip(wav_lengths, batch)):
                input_wavs[i, :length] = a.wav
                padding_mask[i, :length] = 0

            transcription_lengths = torch.tensor([a.tensor_transcription.shape[0] for a in batch])
            transcriptions = torch.zeros((batch_size, transcription_lengths.max()))
            for i, (length, a) in enumerate(zip(transcription_lengths, batch)):
                transcriptions[i, :length] = a.tensor_transcription
            input_wavs = input_wavs.to(device)
            padding_mask = padding_mask.to(device)
            transcriptions = transcriptions.to(device)
            transcription_lengths = transcription_lengths.to(device)
            yield input_wavs, padding_mask, transcriptions, transcription_lengths
            batch = []

def train(model: PhonemeDetector, 
        corpus: CorpusClass,
        output_directory:str = "models",
        batch_size:int = 20,
        lr:float = 3e-5,
        accumulate_steps: int = 1,
        n_steps:int = 30000,
        unfreeze_after:int = 10000,
        zero_lambda_until:int = 10000,
        lambda_param:float = 0.1,
        output_model_every:int = 1000,
        device:str = 'cpu'):
    ''' Example usage:

        model = PhonemeDetector('../wav2vec2_models/wav2vec_small.pt', pronunciation_dictionary.vocab_size())
        model.load_state_dict(torch.load('models/final_output.pt'))
        train(model)
    '''
    os.makedirs(output_directory, exist_ok=True)
    model.to(device)
    if device != 'cpu':
        model.wav2vec.cuda()

    model.train()
    model.freeze_wav2vec()

    ctc_loss_fn = CTCLoss()
    cross_entropy_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    unfrozen = False
    with open(f"{output_directory}/log.txt", 'w') as f:
        i = 0
        while i < n_steps:
            for input_wavs, wav_lengths, transcriptions, transcription_lengths in batched_audio_files(corpus, batch_size=batch_size, device=device):
                X, X_lengths = model(input_wavs, padding_mask=wav_lengths)
                ctc_loss = ctc_loss_fn(X, transcriptions, X_lengths, transcription_lengths)

                if i > zero_lambda_until:
                    cross_loss = self_framewise_loss(X)
                    loss = (1-lambda_param)*ctc_loss + lambda_param*cross_loss
                else:
                    loss = ctc_loss

                loss.backward()
                losses.append(loss.item())

                if i % accumulate_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(i, sum(losses)/len(losses))
                    f.write(f'{i} {sum(losses)/len(losses)}\n')
                    losses = []

                if not unfrozen and i >= unfreeze_after:
                    model.unfreeze_wav2vec()
                    model.freeze_encoder()
                    unfrozen = True

                if i > 0 and i % output_model_every == 0:
                    torch.save(model.state_dict(), f"{output_directory}/{i}_model.pt")

                i += 1
                if i > n_steps: 
                    break

    torch.save(model.state_dict(), f'{output_directory}/final_output.pt')
