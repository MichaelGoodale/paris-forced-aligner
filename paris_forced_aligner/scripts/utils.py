import argparse
import os 
from typing import Tuple

import torch

from paris_forced_aligner.audio_data import PronunciationDictionary, TSVDictionary, LibrispeechDictionary
from paris_forced_aligner.utils import data_directory, download_data_file
from paris_forced_aligner.model import PhonemeDetector

wav2vec_names = {
        'small': "facebook/wav2vec2-base",
        'large': "facebook/wav2vec2-large",
        'multilingual': "wav2vec2-large-xlsr-53"
        }

def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument('--wav2vec_model', type=str, help="Options are small, large, multilingual or anything listed by HuggingFace")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--internal_vector_dim', type=int, default=256)

def add_dictionary_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dictionary", default='librispeech', choices=['librispeech', 'librispeech_unstressed', 'tsv'])
    parser.add_argument("--dictionary_path", type=str)
    parser.add_argument("--G2P_model", type=str)

def process_dictionary_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Tuple[PronunciationDictionary, int]:
    use_G2P = args.G2P_model is not None
    if args.dictionary == "librispeech":
        dictionary = LibrispeechDictionary(use_G2P=use_G2P, G2P_model_path=args.G2P_model)
    elif args.dictionary == "librispeech_unstressed":
        dictionary = LibrispeechDictionary(use_G2P=use_G2P, G2P_model_path=args.G2P_model, remove_stress=True)
    elif args.dictionary == "tsv":
        if not args.dictionary_path:
            parser.error("Please supply --dictionary_path when using TSV dictionaries")
        dictionary = TSVDictionary(args.dictionary_path, use_G2P=use_G2P, G2P_model_path=args.G2P_model)
    return dictionary, dictionary.vocab_size()

def process_model_args(parser: argparse.ArgumentParser, args: argparse.Namespace, vocab_size: int, pretraining: bool = False, device:str = 'cpu') -> str:
    wav2vec_model_name = wav2vec_names[args.wav2vec_model]

    if args.wav2vec_model in wav2vec_names:
        wav2vec_model_name = wav2vec_names[args.wav2vec_model]

    if pretraining:
        model = PhonemeDetector(wav2vec_model_name, 4, args.internal_vector_dim)
    else:
        model = PhonemeDetector(wav2vec_model_name, vocab_size, args.internal_vector_dim)

    if device != "cpu":
        model.wav2vec.cuda()
        model = model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        checkpoint = None

    return model, checkpoint
