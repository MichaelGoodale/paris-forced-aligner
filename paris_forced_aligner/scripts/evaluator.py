import os
import argparse
import warnings
from paris_forced_aligner.utils import download_data_file, data_directory
from paris_forced_aligner.scripts.utils import process_model_args, add_model_args, add_dictionary_args, process_dictionary_args

from paris_forced_aligner.inference import ForcedAligner
from paris_forced_aligner.corpus import BuckeyeCorpus
from paris_forced_aligner.audio_data import AudioFile, LibrispeechDictionary

def evaluate():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_model_args(parser)
    add_dictionary_args(parser)
    parser.add_argument("--buckeye_dir", type=str, default="~/Documents/Buckeye")

    args = parser.parse_args()
    pronunciation_dictionary, vocab_size = process_dictionary_args(parser, args)
    model = process_model_args(parser, args, vocab_size)

    forced_aligner = ForcedAligner(model)

    word_mean_start_difference = 0.0
    word_mean_end_difference = 0.0
    phone_mean_start_difference = 0.0
    phone_mean_end_difference = 0.0
    n_word = 0
    n_phone = 0

    for audio_file, gold_utterance in BuckeyeCorpus(args.buckeye_dir, pronunciation_dictionary, return_gold_labels=True):
        aligned_utterance = forced_aligner.align_file(audio_file)
        for aligned_word, gold_word in zip(aligned_utterance.words, gold_utterance.words):
            word_mean_start_difference = (abs(aligned_word.start - gold_word.start)/16000 + n_word*word_mean_start_difference) / (n_word+1)
            word_mean_end_difference = (abs(aligned_word.end - gold_word.end)/16000 + n_word*word_mean_end_difference) / (n_word+1)
            n_word += 1
            for aligned_phone, gold_phone in zip(aligned_word.phones, gold_word.phones):
                phone_mean_start_difference = (abs(aligned_phone.start - gold_phone.start)/16000 + n_phone*phone_mean_start_difference) / (n_phone+1)
                phone_mean_end_difference = (abs(aligned_phone.end - gold_phone.end)/16000 + n_phone*phone_mean_end_difference) / (n_phone+1)
                n_phone += 1

        print(f"Phone start {phone_mean_start_difference}, Phone end {phone_mean_end_difference}")
        print(f"Word start {word_mean_start_difference}, Word end {word_mean_end_difference}")
    

if __name__ == "__main__":
    evaluate()
