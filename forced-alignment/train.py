import time

from corpus import LibrispeechCorpus
from audio_data import LibrispeechFile
from model import PhonemeDetector

import torch
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import StepLR

model = PhonemeDetector('../wav2vec2_models/wav2vec_small.pt', LibrispeechFile.vocab_size())
model.train()
model.freeze_wav2vec()
ctc_loss_fn = CTCLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
unfrozen = True#False 

start = time.time()
with open("log.txt", 'w') as f:
    i = 0
    while i < 2000:
        for audio_file in LibrispeechCorpus('../data/librispeech-clean-100.tar.gz'):
            X = model(audio_file.features)
            loss = ctc_loss_fn(X, audio_file.tensor_transcription.unsqueeze(0), (X.shape[0],), (audio_file.tensor_transcription.shape[0], ))
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
                torch.save(model.state_dict(), f'models/{i}_output.pt')

            i += 1

            if i > 2000: 
                break

torch.save(model.state_dict(), 'models/final_output.pt')
