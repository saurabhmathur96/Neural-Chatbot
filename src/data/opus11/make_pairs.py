import os
from os import path
import gzip
from subprocess import PIPE, Popen
from tqdm import tqdm
import csv
import re

import sys
sys.path.append('src/utils')
from data_utils import augment
from config_utils import settings


def all_filenames(root):
    for each in os.listdir(root):
        each = path.join(root, each)
        if path.isfile(each):
            yield each
        elif path.isdir(each):
            for subpath in all_filenames(each):
                yield subpath

if __name__ == '__main__':

    pairs_path = settings.data.pairs_path

    with open(pairs_path, 'w') as pairs_handle:
        writer = csv.writer(pairs_handle, quoting=csv.QUOTE_ALL)
        
        base_path = path.join(settings.data.extract_dir, 'OpenSubtitles', 'en')
        names = list(all_filenames(base_path))
        for filepath in tqdm(names):
            try:
                with gzip.open(filepath) as handle:
                    pipe = Popen(['perl', 'lib/wikifil.pl'], stdin=PIPE, stdout=PIPE)
                    text, _ = pipe.communicate(handle.read())
                    lines = re.sub(r'([\.\?\!])[\.\?\! ]+', r'\1 ', text).strip().split('\n')
                    

                    lines = [line.strip() for line in lines]

                    
                    for question, answer in zip(lines[0::2], lines[1::2]):
                        for q, a in augment([question, answer]):
                            writer.writerow([q, a])


                    for question, answer in zip(lines[1::2], lines[2::2]):
                        for q, a in augment([question, answer]):
                            writer.writerow([q, a])
            except IOError:
                pass
                # skip files that cause an error



        