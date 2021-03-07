import os

from paris_forced_aligner.corpus import LibrispeechCorpus, CorpusClass
from paris_forced_aligner.model import PhonemeDetector

import torch
from torch.nn import CTCLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

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

def train(model: PhonemeDetector, 
        corpus: CorpusClass,
        output_directory:str = "models",
        accumulate_steps: int = 20,
        n_steps:int = 30000,
        unfreeze_after:int = 10000,
        zero_lambda_until:int = 20000,
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
    model.freeze_encoder()

    ctc_loss_fn = CTCLoss()
    cross_entropy_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = StepLR(optimizer, 8000, 0.1)

    losses = []

    unfrozen = False
    with open(f"{output_directory}/log.txt", 'w') as f:
        i = 0
        while i < n_steps:
            for audio_file in corpus:
                audio_file.wav.to(device)
                audio_file.tensor_transcription.to(device)

                X = model(audio_file.wav)
                ctc_loss = ctc_loss_fn(X, audio_file.tensor_transcription.unsqueeze(0), (X.shape[0],), (audio_file.tensor_transcription.shape[0], )) 

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
                lr_scheduler.step()
                if i > n_steps: 
                    break

    torch.save(model.state_dict(), f'{output_directory}/final_output.pt')
