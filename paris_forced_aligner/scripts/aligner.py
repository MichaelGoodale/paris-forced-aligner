import argparse
from paris_forced_aligner.utils import download_data_file, data_directory, process_download_args, add_download_args

from paris_forced_aligner.inference import ForceAligner
from paris_forced_aligner.corpus import LibrispeechCorpus
from paris_forced_aligner.audio_data import LibrispeechFile

def align():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_download_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", default='librispeech', choices=['librispeech'])

    args = parser.parse_args()
    wav2vec_model_path = process_download_args(args)

    if args.vocab == 'librispeech':
        vocab_size = LibrispeechFile.vocab_size()

    f = ForceAligner(args.checkpoint, wav2vec_model_path, vocab_size)

if __name__ == "__main__":
    align()
