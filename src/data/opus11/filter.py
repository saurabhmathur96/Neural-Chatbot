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
from config_utils import settings

if __name__ == '__main__':
    

    with open(settings.data.blacklist_path) as handle:
        blacklist = set(handle.read().split('\n'))

    data_file = settings.data.pairs_path
    print ('Reading {0}'.format(data_file))
    with open(data_file) as handle:
        reader = csv.reader(handle)
        pairs = ((question, answer) for question, answer in reader if not any(w in question for w in blacklist) and not any(w in answer for w in blacklist))

        print ('Building Frequency Distribution')
        vocabulary_size = settings.model.vocabulary_size - 4 # pad, start, end, unk
        freq_dist = FreqDist(chain.from_iterable(q.split() + a.split() for q, a in pairs))

        print ('Total {0} unique words'.format(len(freq_dist)))
        word_counts = freq_dist.most_common(vocabulary_size)
        vocabulary = [word for word, count in word_counts]

        length = settings.model.sequence_length - 2 # start, end
        vocabulary_set = set(vocabulary)
        def remove_unknown(line):
            return ' '.join(word if word in vocabulary_set else UNK for word in line.split())

        unk_ratio = settings.data.unk_ratio
        def is_valid(line):
            words = line.split()
            return len(words) <= length and (words.count(UNK) / float(len(words))) < unk_ratio
        
        def mark_ends(line):
            return START + ' ' + line + ' ' + END

    with open(data_file) as handle:
        reader = csv.reader(handle)
        pairs = ((question, answer) for question, answer in reader)
        
        pairs = (map(remove_unknown, pair) for pair in tqdm(pairs, desc='removing rare words'))
        pairs = (map(mark_ends, (question, answer)) for question, answer in tqdm(pairs, desc='filtering lines') if is_valid(question) and is_valid(answer))
    
        vocabulary = [PAD, UNK, START, END] + vocabulary
        
        vocabulary_file = settings.data.vocabulary_path
        print ('Writing vocabulary to {0}'.format(vocabulary_file))
        with open(vocabulary_file, 'w') as handle:
            json.dump(vocabulary, handle)

        filtered_file = settings.data.filtered_path
        print ('Writing filtered pairs to {0}'.format(filtered_file))
        with open(filtered_file, 'w') as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerows(pairs)
