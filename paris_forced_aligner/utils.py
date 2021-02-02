import os
from appdirs import user_data_dir
from tqdm import tqdm
import urllib.request
import argparse
import click

data_directory = user_data_dir('paris-forced-aligner', 'mgoodale')
os.makedirs(data_directory, exist_ok=True)

arpabet_to_ipa = {
"AA": "ɑ",
"AE": "æ",
"AH": "ʌ",
"AO": "ɔ",
"AW": "aʊ",
"AX": "ə",
"AXR": "ɚ",
"AY": "aɪ",
"EH": "ɛ",
"ER": "ɝ",
"EY": "eɪ",
"IH": "ɪ",
"IX": "ɨ",
"IY": "i",
"OW": "oʊ",
"OY": "ɔɪ",
"UH": "ʊ",
"UW": "u",
"UX": "ʉ",
"B": "b ",
"CH": "tʃ",
"D": "d ",
"DH": "ð",
"DX": "ɾ",
"EL": "l̩",
"EM": "m̩",
"EN": "n̩",
"F": "f",
"G": "ɡ",
"HH": "h",
"H": "h",
"JH": "dʒ",
"K": "k",
"L": "l",
"M": "m",
"N": "n",
"NG": "ŋ",
"NX": "ɾ̃",
"P": "p",
"Q": "ʔ",
"R": "ɹ",
"S": "s",
"SH": "ʃ",
"T": "t",
"TH": "θ",
"V": "v",
"W": "w",
"WH ": "ʍ",
"Y": "j",
"Z": "z",
"ZH": "ʒ"}

wav2vec_urls = {
        'small': ('wav2vec2_small.pt', 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt'),
        'large': ('wav2vec2_large.pt', 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'),
        'multi-lingual': ('wav2vec_multiling.pt', 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt')
        }

class DownloadBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_data_file(url, output_file):
    print(f"Downloading file to {output_file}")
    if os.path.exists(output_file):
        if not click.confirm(f"{output_file} exists, overwrite?"):
            return

    with DownloadBar(unit='B', unit_scale=True, unit_divisor=1024,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_file,
                   reporthook=t.update_to, data=None)
        t.total = t.n

def add_download_args(parser: argparse.ArgumentParser):
    parser.add_argument('--wav2vec_model',  choices=list(wav2vec_urls.keys()))
    parser.add_argument('--download-wav2vec', action='store_true')
    parser.add_argument('--wav2vec_model_path', type=str)

def process_download_args(args: argparse.Namespace) -> str:
    if args.wav2vec_model and args.wav2vec_model_path:
        raise RuntimeError("Please provide a wav2vec_model or wav2vec_model_path, not both.")

    if args.wav2vec_model:
        wav2vec_model_name, url = wav2vec_urls[args.wav2vec_model]
        wav2vec_model_path = os.path.join(data_directory, wav2vec_model_name)

        if not os.path.exists(wav2vec_model_path) and not args.download_wav2vec:
            raise RuntimeError(f"Wav2Vec2.0 model {wav2vec_model_name} has not been downloaded, you can download it by passing the --download_wav2vec flag")

        if args.download_wav2vec:
            download_data_file(url, wav2vec_model_path)

    elif args.wav2vec_model_path:
        wav2vec_model_path = args.wav2vec_model_path
    else:
        raise RuntimeError("You must pass either a wav2vec_model or wav2vec_model_path")

    return wav2vec_model_path
