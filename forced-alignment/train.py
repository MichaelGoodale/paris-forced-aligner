import time

from corpus import LibrispeechCorpus
from audio_data import LibrispeechFile
from model import PhonemeDetector

import torch
from torch.nn import CTCLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

model = PhonemeDetector('../wav2vec2_models/wav2vec_small.pt', LibrispeechFile.vocab_size())
model.load_state_dict(torch.load('models/final_output.pt'))
model.train()
model.freeze_encoder()
ctc_loss_fn = CTCLoss()
cross_entropy_fn = CrossEntropyLoss()

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

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

LAMBDA = 0.1
losses = []
unfrozen = True#False 

start = time.time()
with open("log.txt", 'w') as f:
    i = 0
    while i < 2000:
        for audio_file in LibrispeechCorpus('../data/librispeech-clean-100.tar.gz'):
            X = model(audio_file.features)
            ctc_loss = ctc_loss_fn(X, audio_file.tensor_transcription.unsqueeze(0), (X.shape[0],), (audio_file.tensor_transcription.shape[0], )) 
            cross_loss = self_framewise_loss(X)

            loss = (1-LAMBDA)*ctc_loss + LAMBDA*cross_loss
            loss.backward()
            losses.append(loss.item())

            if i % 20 == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(torch.argmax(X, dim=-1).squeeze().tolist())
                print(i, sum(losses)/len(losses))
                f.write(f'{i} {sum(losses)/len(losses)}\n')
                losses = []

            if i >= 1000 and not unfrozen:
                model.unfreeze_wav2vec()
                model.freeze_encoder()
                unfrozen = True

            if i > 0 and i % 500 == 0:
                torch.save(model.state_dict(), f'models/multi_loss{2000+i}_output.pt')

            i += 1

            if i > 2000: 
                break

torch.save(model.state_dict(), 'models/final_output.pt')
