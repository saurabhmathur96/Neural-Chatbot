from sampling import Sampler
from models import seq2seq, seq2seq_attention
import json

if __name__ == '__main__':
    sequence_length = 16
    vocabulary_size = 2000
    hidden_size = 256
    model = seq2seq_attention(sequence_length, vocabulary_size, hidden_size)
    model.load_weights('models/seq2seq_weights.h5')

    vocabulary_file = 'data/processed/opus11/vocabulary.txt'
    with open(vocabulary_file, 'r') as handle:
        vocabulary = json.load(handle)
    
    sampler = Sampler(model, vocabulary, sequence_length)
    
    while True:
        question = raw_input('>>')
        response = sampler.respond(question, greedy=True)
        print (response)
        for t in (.7, .8, .9):
            response = sampler.respond(question, temperature=t)
            print (response)

