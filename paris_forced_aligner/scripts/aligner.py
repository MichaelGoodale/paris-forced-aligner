import argparse
import warnings
from paris_forced_aligner.utils import download_data_file, data_directory, process_download_args, add_download_args

from paris_forced_aligner.inference import ForcedAligner
from paris_forced_aligner.corpus import LibrispeechCorpus
from paris_forced_aligner.audio_data import AudioFile, LibrispeechDictionary

def read_file_list(path):
    files = []
    with open(path, 'r') as f:
        for line in f:
            files.append(line.strip())
    return files

def get_audio_file(vocab_type, input_path, transcription):
    if vocab_type == 'librispeech':
        audio_file = LibrispeechFile(input_path, transcription)
    return audio_file

def align():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_download_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", default='librispeech', choices=['librispeech'])

    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--transcription", type=str)
    parser.add_argument("--save_as", default='textgrid', choices=['textgrid', 'csv'])

    parser.add_argument("--input_files", type=str)
    parser.add_argument("--transcripts", type=str)
    parser.add_argument("--output_files", type=str)

    parser.add_argument("--overwrite", action='store_true')

    parser.add_argument("--in_seconds", action='store_true')

    args = parser.parse_args()
    wav2vec_model_path = process_download_args(args)

    if not args.audio_file and not args.input_files:
        raise RuntimeError("You must provide either --audio_file or --input_files or both.")

    if args.audio_file and not args.transcription:
        raise RuntimeError("You must provide a transcription with --transcription for --audio_file")

    if args.input_files and not args.transcripts:
        raise RuntimeError("You must provide a file with transcriptions with --transcripts for --input_files")

    if args.vocab == 'librispeech':
        dictionary = LibrispeechDictionary()
        vocab_size = dictionary.vocab_size()

    forced_aligner = ForcedAligner(args.checkpoint, wav2vec_model_path, vocab_size)
    

    if args.audio_file:
        audio_file = AudioFile(args.audio_file, args.transcription, dictionary)
        utterance = forced_aligner.align_file(audio_file)

        output_file = f'{args.audio_file.rsplit(".", 1)[0]}.{args.save_as}'
        if not overwrite and os.path.exists(output_file):
            print(f"{output_file} already exists, pass --overwrite to overwrite")
        else:
            if args.save_as == 'textgrid':
                utterance.save_textgrid(output_file, overwrite=args.overwrite)
            elif args.save_as == 'csv':
                utterance.save_csv(output_file, in_seconds=args.in_seconds, overwrite=args.overwrite)

    if args.input_files:
        input_files = read_file_list(args.input_files)
        transcripts = read_file_list(args.transcripts)
        if args.output_files:
            output_files = read_file_list(args.output_files)
        else:
            output_files = [f'{x.rsplit(".", 1)[0]}.{args.save_as}' for x in input_files]
        
        if len(input_files) != len(transcripts) or len(input_files) != len(output_files):
            raise RuntimeError("--input_files, --transcripts, --output_files must all be same length")

        for input_file, transcription, output_file in zip(input_file, transcripts, output_files):
            audio_file = AudioFile(input_file, transcription, dictionary)
            utterance = forced_aligner.align_file(audio_file)

            if not overwrite and os.path.exists(output_file):
                print(f"{output_file} already exists, pass --overwrite to overwrite")
            else:
                terminator = output_file.rsplit('.', 1)[1]
                if terminator == 'textgrid':
                    utterance.save_textgrid(output_file)
                elif terminator == 'csv':
                    utterance.save_csv(output_file, in_seconds=args.in_seconds)
                else:
                    warnings.warn(f"Warning: Unknown file type: {terminator}, saving as TextGrid")
                    utterance.save_textgrid(output_file)

if __name__ == "__main__":
    align()
