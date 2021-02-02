import argparse
from paris_forced_aligner.utils import download_data_file, data_directory, process_download_args, add_download_args

def train_model():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_download_args(parser)
    args = parser.parse_args()
    wav_2vec_model_path = process_download_args(args)
    
if __name__ == "__main__":
    train_model()
