import os
import argparse
import warnings
from paris_forced_aligner.utils import download_data_file, data_directory
from paris_forced_aligner.scripts.utils import process_download_args, add_download_args, add_dictionary_args, process_dictionary_args

from paris_forced_aligner.inference import ForcedAligner
from paris_forced_aligner.corpus import BuckeyeCorpus
from paris_forced_aligner.audio_data import AudioFile, LibrispeechDictionary

def evaluate():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_download_args(parser)
    add_dictionary_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--buckeye_dir", type=str, default="~/Documents/Buckeye")

    args = parser.parse_args()
    wav2vec_model_path = process_download_args(parser, args)
    pronunciation_dictionary, vocab_size = process_dictionary_args(parser, args)

    forced_aligner = ForcedAligner(args.checkpoint, wav2vec_model_path, vocab_size)

    word_mean_start_difference = 0.0
    word_mean_end_difference = 0.0
    phone_mean_start_difference = 0.0
    phone_mean_end_difference = 0.0
    n = 0

    for audio_file, gold_utterance in BuckeyeCorpus(args.buckeye_dir, pronunciation_dictionary, return_gold_labels=True):
        aligned_utterance = forced_aligner.align_file(audio_file)
        offset = gold_utterance.start
        for aligned_word, gold_word in zip(aligned_utterance.words, gold_utterance.words):
            word_mean_start_difference = (abs(aligned_word.start + offset - gold_word.start)/16000 + n*mean_start_difference) / (n+1)
            word_mean_end_difference = (abs(aligned_word.end + offset - gold_word.end)/16000 + n*mean_end_difference) / (n+1)

        for aligned_phone, gold_phone in zip(aligned_utterance.words, gold_utterance.words):
            phone_mean_start_difference = (abs(aligned_phone.start + offset - gold_phone.start)/16000 + n*mean_start_difference) / (n+1)
            phone_mean_end_difference = (abs(aligned_phone.end + offset - gold_phone.end)/16000 + n*mean_end_difference) / (n+1)
    

if __name__ == "__main__":
    evaluate()
