import os
from os import path
import sys
import tarfile

sys.path.append('src/utils')
from data_utils import download
from config_utils import settings

if __name__ == '__main__':
    
    url = 'http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz'
    save_path = settings.data.save_path
    download(url, save_path)

    extract_dir = settings.data.extract_dir
    with tarfile.open(save_path, 'r:gz') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, extract_dir)

    os.remove(save_path)

