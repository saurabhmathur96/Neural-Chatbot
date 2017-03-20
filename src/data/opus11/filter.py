import os
import sys
import zipfile
import csv
import json
from tqdm import tqdm
from itertools import chain
from nltk import FreqDist

sys.path.append('src/utils')
from data_utils import PAD, START, END, UNK

if __name__ == '__main__':
    data_file = 'data/processed/opus11/pairs.txt'
    print ('Reading {0}'.format(data_file))
    with open(data_file) as handle:
        reader = csv.reader(handle)
        pairs = [(question, answer) for question, answer in reader]

    print ('Building Frequency Distribution')
    vocabulary_size = 2000 - 4 # pad, start, end, unk
    words = ' '.join(chain.from_iterable(pairs)).split()
    print ('Total {0} words'.format(len(words)))
    freq_dist = FreqDist(words)
    print ('Total {0} unique words'.format(len(freq_dist)))
    word_counts = freq_dist.most_common(vocabulary_size)
    vocabulary = [word for word, count in word_counts]

    length = 10
    vocabulary_set = set(vocabulary)
    def remove_unknown(line):
        return ' '.join(word if word in vocabulary_set else UNK for word in line.split())

    def is_valid(line):
        words = line.split()
        return len(words) <= length and (words.count(UNK) / float(len(words))) < .2
    
    def mark_ends(line):
        return START + ' ' + line + ' ' + END
    
    pairs = [map(remove_unknown, pair) for pair in tqdm(pairs, desc='removing rare words')]
    pairs = [map(mark_ends, (question, answer)) for question, answer in tqdm(pairs, desc='filtering lines') if is_valid(question) and is_valid(answer)]
    
    vocabulary = [PAD, UNK, START, END] + vocabulary
    
    vocabulary_file = 'data/processed/opus11/vocabulary.txt'
    print ('Writing vocabulary to {0}'.format(vocabulary_file))
    with open(vocabulary_file, 'w') as handle:
        json.dump(vocabulary, handle)

    filtered_file = 'data/processed/opus11/filtered_pairs.txt'
    print ('Writing filtered pairs to {0}'.format(filtered_file))
    with open(filtered_file, 'w') as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerows(pairs)
