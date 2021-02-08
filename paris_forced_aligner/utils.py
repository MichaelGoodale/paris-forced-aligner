import os
from appdirs import user_data_dir
from tqdm import tqdm
import urllib.request
import click

data_directory = user_data_dir('paris-forced-aligner', 'mgoodale')
os.makedirs(data_directory, exist_ok=True)

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

