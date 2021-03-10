#!/bin/bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar xvzf train-clean-100.tar.gz
rm train-clean-100.tar.gz
paris_forced_trainer --corpus_path LibriSpeech/ --wav2vec_model small --download-wav2vec --learning_rate 0.0001 --batch_size 4 --unfreeze_after 1200 --output_model_every 400 --zero_lambda_until 3200 --n_steps 4000 --accumulate_steps=2 --gpu

