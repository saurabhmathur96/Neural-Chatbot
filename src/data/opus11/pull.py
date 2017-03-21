import os
from os import path
import sys
import tarfile

sys.path.append('src/utils')
from data_utils import download
from config_utils import settings

if __name__ == '__main__':
    
    url = 'http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz'
    save_path = setting.data.save_path
    download(url, save_path)

    extract_dir = settings.data.extract_dir
    with tarfile.open(save_path, 'r:gz') as f:
        f.extractall(extract_dir)

    os.remove(save_path)

