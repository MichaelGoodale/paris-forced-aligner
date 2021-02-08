import argparse

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
    parser.add_argument("--corpus_type", default="librispeech", choices=['librispeech'])
    parser.add_argument("--n_proc", type=int, default=1)

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

    train(model, corpus, 
        output_directory=args.output_dir)

if __name__ == "__main__":
    train_model()
