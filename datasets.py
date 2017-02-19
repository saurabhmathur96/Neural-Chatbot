from __future__ import print_function, unicode_literals
import os
import itertools
from collections import Counter
from io import open
import spacy
import tqdm
import string
import re


# nlp = spacy.load("en")

class CornellMovieDialogs(object):
    def __init__(self, data_directory, vocabulary_size=20000):
        self.data_directory = data_directory
        self.vocabulary_size = vocabulary_size
        self.PAD = "PAD"
        self.START = "START"
        self.UNK = "UNK"
        self.END = "END"
        self.cleaned_directory = os.path.join(data_directory, "cleaned")
        self.vocabulary_path = os.path.join(self.cleaned_directory, "vocabulary.txt")
        self.statements_path = os.path.join(self.cleaned_directory, "statements.txt")
        self.responses_path = os.path.join(self.cleaned_directory, "responses.txt")
    
    def make_conversations(self):

        if not os.path.exists(self.cleaned_directory):
            os.makedirs(self.cleaned_directory)

        #
        # Load lines

        movie_lines_path = os.path.join(self.data_directory, "movie_lines.txt")
        lines = dict()
        with open(movie_lines_path, "r", encoding="iso-8859-1") as f:
            for line in tqdm.tqdm(f, total=304713):
                id, _, _, _, text = line.split(" +++$+++ ")
                lines[id] = self.clean(text.encode("ascii", "ignore"))
        
        #
        # Load conversations

        conversations = list()
        movie_conversations_path = os.path.join(self.data_directory, "movie_conversations.txt")
        with open(movie_conversations_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                _, _, _, ids = line.split(" +++$+++ ")
                ids = ids[2:-3].split("', '")
                conversations += [lines[id] for id in ids]
        
        tokens = list(itertools.chain.from_iterable(map(str, nlp(unicode(line))) for line in tqdm.tqdm(conversations, desc="building vocabulary") ))
        
        print ("{0} tokens, {1} unique".format(len(tokens), len(set(tokens))))
        
        init_vocabulary = [self.PAD, self.START, self.UNK, self.END]
        vocabulary = init_vocabulary + [word for word, count in Counter(tokens).most_common(self.vocabulary_size - len(init_vocabulary))]
        self.vocabulary = vocabulary
        self.inverse_vocabulary = dict((word, i) for i, word in enumerate(self.vocabulary))

        with open(self.vocabulary_path, "w") as vocabulary_file:
            for word in self.vocabulary:
                print (unicode(word), file=vocabulary_file)

        
        # conversations = [self.encode_sequence(sentence) for sentence in tqdm.tqdm(conversations, desc="encoding sequences")]
        even, odd = conversations[0::2], conversations[1::2]
        
        statements = even + odd 
        responses = odd + even

        #
        # Save conversations

        
        with open(self.statements_path, "w") as statements_file, open(self.responses_path, "w") as responses_file:
            for s, r in zip(statements, responses):
                if len(s) < 2 or len(r) < 2:
                    continue
                print (unicode(s), file=statements_file)
                print (unicode(r), file=responses_file) 
    
    def encode_sequence(self, sentence):
        unk_id = self.inverse_vocabulary[self.UNK]
        start_id = self.inverse_vocabulary[self.START]
        end_id = self.inverse_vocabulary[self.END]
        return tuple([start_id]) + tuple([self.inverse_vocabulary.get(word, unk_id) for word in sentence.split(" ")]) + tuple([end_id])
    
    def decode_sequence(self, ids):
        return [self.vocabulary[id] for id in ids]

    def clean(self, s):
        MATCH_MULTIPLE_SPACES = re.compile(r"\s{2,}")
        MATCH_TAGS = re.compile(r"</?\s?[a-z]\s?>")

        s = s.lower().replace("\n", "")
        s = MATCH_TAGS.sub(" ", s)

        for i in range(10):
            s = s.replace(unicode(i), " " + unicode(i) + " ")

        s = MATCH_MULTIPLE_SPACES.sub(" ", s).strip()
        # take first line
        # sents = nltk.sent_tokenize(s)
        try:
            sents = list(each for each in nlp(unicode(s)).sents)
            s = str(sents[0]).strip() if len(sents) > 0 else s
            s = " ".join(map(str, nlp(unicode(s))))
        except IndexError:
            # print (s)
            pass
        return s



    def load(self):

        #
        # Load vocabulary

        with open(self.vocabulary_path, "r") as vocabulary_file:
            self.vocabulary = vocabulary_file.read().strip().split("\n")

        self.inverse_vocabulary = dict((word, i) for i, word in enumerate(self.vocabulary))
    
    def data_generator(self):
        while True:
            with open(self.statements_path, "r") as statements_file, open(self.responses_path, "r")  as responses_file:
                for s, r in zip(statements_file, responses_file):
                    yield (s, r)
