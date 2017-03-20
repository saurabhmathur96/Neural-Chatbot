import os
import sys
import zipfile
import csv

sys.path.append('src/utils')
from data_utils import read_lines, make_pairs

if __name__ == '__main__':
    movie_lines = 'data/raw/movie_lines.txt'
    lines = read_lines(movie_lines)

    movie_conversation = 'data/raw/movie_conversation.txt'
    pairs = make_pairs(movie_conversation, lines)

    data_file = 'data/processed/pairs.txt'
    with open(data_file, 'w') as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerows(pairs)
