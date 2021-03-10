import argparse
import os 
from typing import Tuple
from paris_forced_aligner.audio_data import PronunciationDictionary, TSVDictionary, LibrispeechDictionary
from paris_forced_aligner.utils import data_directory, download_data_file
from paris_forced_aligner.model import PhonemeDetector

wav2vec_urls = {
        'small': ('wav2vec2_small.pt', 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt'),
        'large': ('wav2vec2_large.pt', 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'),
        'multi-lingual': ('wav2vec_multiling.pt', 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt')
        }

def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument('--wav2vec_model',  choices=list(wav2vec_urls.keys()))
    parser.add_argument('--download-wav2vec', action='store_true')
    parser.add_argument('--wav2vec_model_path', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--internal_vector_dim', type=int, default=128)

def add_dictionary_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dictionary", default='librispeech', choices=['librispeech', 'librispeech_stressed', 'tsv'])
    parser.add_argument("--dictionary_path", type=str)

def process_dictionary_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Tuple[PronunciationDictionary, int]:
    if args.dictionary == "librispeech":
        dictionary = LibrispeechDictionary()
    elif args.dictionary == "librispeech_stressed":
        dictionary = LibrispeechDictionary(remove_stress=False)
    elif args.dictionary == "tsv":
        if not args.dictionary_path:
            parser.error("Please supply --dictionary_path when using TSV dictionaries")
        dictionary = TSVDictionary()
    return dictionary, dictionary.vocab_size()

def process_model_args(parser: argparse.ArgumentParser, args: argparse.Namespace, vocab_size: int) -> str:
    if args.wav2vec_model and args.wav2vec_model_path:
        parser.error("Please provide a wav2vec_model or wav2vec_model_path, not both.")

    if args.wav2vec_model:
        wav2vec_model_name, url = wav2vec_urls[args.wav2vec_model]
        wav2vec_model_path = os.path.join(data_directory, wav2vec_model_name)

        if not os.path.exists(wav2vec_model_path) and not args.download_wav2vec:
            parser.error(f"Wav2Vec2.0 model {wav2vec_model_name} has not been downloaded, you can download it by passing the --download_wav2vec flag")

        if args.download_wav2vec:
            download_data_file(url, wav2vec_model_path)

    elif args.wav2vec_model_path:
        wav2vec_model_path = args.wav2vec_model_path
    else:
        parser.error("You must pass either a wav2vec_model or wav2vec_model_path")

    model = PhonemeDetector(wav2vec_model_path, vocab_size, args.upscale, args.internal_vector_dim)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    return model
