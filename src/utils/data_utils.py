import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# fix encoding
from io import open

from tqdm import tqdm
import requests
from itertools import chain
import json
from nltk import sent_tokenize
import string
import re
import unicodedata

UNK = 'unk'
START = '^'
END = '$'
PAD = '_'

def download(url, save_path):
    response = requests.get(url, stream=True)
    length = int(response.headers.get('content-length'))
    with open(save_path, 'wb') as handle:
        for data in tqdm(response.iter_content(), total=length):
            handle.write(data)

def read_lines(file_path):
    def process(line):
        tokens = line.strip().split(' +++$+++ ')
        return (tokens[0], clean(tokens[-1]) if len(tokens) == 5 else '')

    with open(file_path, encoding='latin-1') as handle:
         lines = dict(process(line) for line in tqdm(handle,total=304713) if line)
         return lines


def normalize_unicode(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.printable
    )

def clean(line):
    line = unicode(line)
    text = normalize_unicode(line)

    # remove html tags
    text = re.sub(r'</?\s?[a-z]\s?>', ' ', text)
    # remove duplicates
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
    punctuation = ".?!' "
    allowed_chars = string.ascii_lowercase + string.ascii_uppercase + punctuation
    text = ''.join(c for c in text if c in allowed_chars)

    for p in punctuation:
        text = text.replace(p, ' ' + p + ' ')
    text = ' '.join(text.split())
    return text

def augment(pair):
    # convert single pair into multiple pairs
    question, answer = map(sent_tokenize, pair)
    q_sents = list(reversed(question))
    for _ in range(len(q_sents)):
        a_sents = answer[:]
        for _ in range(len(a_sents)):
            yield (' '.join(reversed(q_sents)), ' '.join(a_sents))
            a_sents.pop()
        q_sents.pop()

def make_pairs(file_path, lines):
    def process(line, lines):
        tokens = line.strip().split(' +++$+++ ')
        text = tokens[3].replace("'", '"')
        convsersation = json.loads(text)

        # normal pairs
        pairs_1 = [(lines[question], lines[answer]) for question, answer in zip(convsersation[0::2], convsersation[1::2])]
        
        # pairs shifted by one
        pairs_2 = [(lines[question], lines[answer]) for question, answer in zip(convsersation[1::2], convsersation[2::2])]
        return pairs_1 + pairs_2

    with open(file_path, encoding='latin-1') as handle:
        pairs = chain.from_iterable(process(line, lines) for line in handle)
        augmented = chain.from_iterable(augment(pair) for pair in pairs)
        return list(augmented)


        
