#!/bin/bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar xvzf train-clean-100.tar.gz
rm train-clean-100.tar.gz
paris_forced_trainer --corpus_path LibriSpeech/ --wav2vec_model small --download-wav2vec --learning_rate 0.00003 --batch_size 4 --unfreeze_after 10000 --output_model_every 1000 --n_steps 20000 --accumulate_steps=8 --gpu  --upscale 5

