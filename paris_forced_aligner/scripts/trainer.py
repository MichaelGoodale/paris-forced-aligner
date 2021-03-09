import argparse

import torch

from paris_forced_aligner.utils import download_data_file, data_directory
from paris_forced_aligner.scripts.utils import process_download_args, add_download_args, add_dictionary_args, process_dictionary_args
from paris_forced_aligner.train import train
from paris_forced_aligner.corpus import LibrispeechCorpus, YoutubeCorpus
from paris_forced_aligner.model import PhonemeDetector

def train_model():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_download_args(parser)
    add_dictionary_args(parser)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--corpus_type", default="librispeech", choices=['librispeech', 'youtube'])
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--n_proc", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=30000)
    parser.add_argument("--unfreeze_after", type=int, default=10000)
    parser.add_argument("--zero_lambda_until", type=int, default=20000)
    parser.add_argument("--lambda_param", type=float, default=0.1)
    parser.add_argument("--output_model_every", type=int, default=1000)

    args = parser.parse_args()
    wav2vec_model_path = process_download_args(parser, args)
    pronunciation_dictionary, vocab_size = process_dictionary_args(parser, args)

    if args.corpus_type == 'librispeech':
        if args.dictionary != "librispeech":
            parser.error("You must use --dictionary librispeech when using --corpus librispeech")
        corpus = LibrispeechCorpus(args.corpus_path, n_proc=args.n_proc)
    elif arg.corpus_type == 'youtube':
        corpus = YoutubeCorpus(args.corpus_path)

    model = PhonemeDetector(wav2vec_model_path, vocab_size)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    train(model, corpus, 
        output_directory=args.output_dir,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        accumulate_steps=args.accumulate_steps,
        n_steps=args.n_steps,
        unfreeze_after=args.unfreeze_after,
        zero_lambda_until=args.zero_lambda_until,
        lambda_param=args.lambda_param,
        output_model_every=args.output_model_every,
        device=device)

if __name__ == "__main__":
    train_model()
