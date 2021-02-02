import argparse
from paris_forced_aligner.utils import download_data_file, data_directory, process_download_args, add_download_args

from paris_forced_aligner.inference import ForcedAligner
from paris_forced_aligner.corpus import LibrispeechCorpus
from paris_forced_aligner.audio_data import LibrispeechFile

def align():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_download_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", default='librispeech', choices=['librispeech'])

    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--transcription", type=str, required=True)
    parser.add_argument("--save_as", default='textgrid', choices=['textgrid', 'csv'])
    parser.add_argument("--in_seconds", action='store_true')

    args = parser.parse_args()
    wav2vec_model_path = process_download_args(args)

    if args.vocab == 'librispeech':
        vocab_size = LibrispeechFile.vocab_size()
        audio_file = LibrispeechFile(args.audio_file, args.transcription)

    forced_aligner = ForcedAligner(args.checkpoint, wav2vec_model_path, vocab_size)
    
    utterance = forced_aligner.align_file(audio_file)

    output_file = f'{audio_file.rsplit('.', 1)[0]].{args.save_as}'
    if args.save_as == 'textgrid':
        utterance.save_textgrid(output_file)
    elif args.save_as == 'csv':
        utterance.save_csv(output_file, in_seconds=args.in_seconds)

    
    

if __name__ == "__main__":
    align()
