from sampling import Sampler
from models import seq2seq, seq2seq_attention
import json


import sys
sys.path.append('src/utils')
from config_utils import settings

if __name__ == '__main__':
    sequence_length = settings.model.sequence_length
    vocabulary_size = settings.model.vocabulary_size
    hidden_size = settings.model.hidden_size
    print ('Creating model with configuration: {0}'.format(settings.model))

    model = seq2seq_attention(sequence_length, vocabulary_size, hidden_size)
    print ('Loading model weights from {0}'.format(settings.model.weights_path))
    model.load_weights('models/seq2seq_weights.h5')

    vocabulary_file = settings.data.vocabulary_path
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

