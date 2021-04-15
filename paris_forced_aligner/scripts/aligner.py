import os
import argparse
import warnings
from paris_forced_aligner.utils import download_data_file, data_directory
from paris_forced_aligner.scripts.utils import process_model_args, add_model_args, add_dictionary_args, process_dictionary_args

from paris_forced_aligner.inference import ForcedAligner
from paris_forced_aligner.phonological import Utterance
from paris_forced_aligner.corpus import LibrispeechCorpus, YoutubeCorpus
from paris_forced_aligner.audio_data import AudioFile, LibrispeechDictionary

def read_file_list(path):
    files = []
    with open(path, 'r') as f:
        for line in f:
            files.append(line.strip())
    return files

def align():
    parser = argparse.ArgumentParser(description='Train forced aligner')
    add_model_args(parser)
    add_dictionary_args(parser)

    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--transcription", type=str)
    parser.add_argument("--save_as", default='textgrid', choices=['textgrid', 'csv'])

    parser.add_argument("--input_files", type=str)
    parser.add_argument("--transcripts", type=str)
    parser.add_argument("--output_files", type=str)

    parser.add_argument("--youtube", type=str, nargs="*")
    parser.add_argument("--youtube_lang", type=str, default='en')
    parser.add_argument("--youtube_output_dir", type=str, default='./youtube_audio')

    parser.add_argument("--overwrite", action='store_true')

    parser.add_argument("--in_seconds", action='store_true')

    args = parser.parse_args()
    pronunciation_dictionary, vocab_size = process_dictionary_args(parser, args)
    model, _ = process_model_args(parser, args, vocab_size)

    if not args.audio_file and not args.input_files and not args.youtube:
        parser.error("You must provide either --audio_file or --input_files or --youtube or any combination thereof.")

    if args.audio_file and not args.transcription:
        parser.error("You must provide a transcription with --transcription for --audio_file")

    if args.input_files and not args.transcripts:
        parser.error("You must provide a file with transcriptions with --transcripts for --input_files")

    forced_aligner = ForcedAligner(model)
    
    if args.audio_file:
        audio_file = AudioFile(args.audio_file, args.transcription, pronunciation_dictionary)
        utterance = forced_aligner.align_file(audio_file)

        output_file = f'{args.audio_file.rsplit(".", 1)[0]}.{args.save_as}'
        if not args.overwrite and os.path.exists(output_file):
            print(f"{output_file} already exists, pass --overwrite to overwrite")
        else:
            if args.save_as == 'textgrid':
                utterance.save_textgrid(output_file)
            elif args.save_as == 'csv':
                utterance.save_csv(output_file, in_seconds=args.in_seconds)

    if args.input_files:
        input_files = read_file_list(args.input_files)
        transcripts = read_file_list(args.transcripts)
        if args.output_files:
            output_files = read_file_list(args.output_files)
        else:
            output_files = [f'{x.rsplit(".", 1)[0]}.{args.save_as}' for x in input_files]
        
        if len(input_files) != len(transcripts) or len(input_files) != len(output_files):
            parser.error("--input_files, --transcripts, --output_files must all be same length")

        for input_file, transcription, output_file in zip(input_file, transcripts, output_files):
            audio_file = AudioFile(input_file, transcription, pronunciation_dictionary)
            utterance = forced_aligner.align_file(audio_file)

            if not args.overwrite and os.path.exists(output_file):
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
    if args.youtube:
        if not args.youtube_output_dir:
            parser.error("--youtube_output_dir must be supplied when using --youtube")

        for youtube in args.youtube:
            corpus = YoutubeCorpus(youtube, pronunciation_dictionary, language=args.youtube_lang,
                    audio_directory=os.path.abspath(args.youtube_output_dir),
                    save_wavs=True)

            name_utt_dict = {}

            for audio_file in corpus:
                utterance = forced_aligner.align_file(audio_file)
                name = audio_file.filename.rsplit('_', 1)[0]
                if name not in name_utt_dict:
                    name_utt_dict[name] = [utterance]
                else:
                    name_utt_dict[name].append(utterance)

            for name, utterances in name_utt_dict.items():
                big_utterance = Utterance([d for u in sorted(utterances, key=lambda u: u.start) for d in u.data])
                #big_utterance = corpus.stitch_youtube_utterances(name, utterances)
                output_file = name + '.TextGrid' if args.save_as == 'textgrid' else '.csv'
                if not args.overwrite and os.path.exists(output_file):
                    print(f"{output_file} already exists, pass --overwrite to overwrite")
                else:
                    if args.save_as == 'textgrid':
                        big_utterance.save_textgrid(output_file)
                    elif args.save_as == 'csv':
                        big_utterance.save_csv(output_file, in_seconds=args.in_seconds)

if __name__ == "__main__":
    align()
