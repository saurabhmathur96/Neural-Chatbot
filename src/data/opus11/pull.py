import os
from os import path
import sys
import tarfile

sys.path.append('src/utils')
from data_utils import download

if __name__ == '__main__':
    url = 'http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz'
    save_path = 'data/raw/opus11.tar.gz'
    # download(url, save_path)

    extract_dir = 'data/raw/'
    with tarfile.open(save_path, 'r:gz') as f:
        f.extractall(extract_dir)

    # os.remove(save_path)

