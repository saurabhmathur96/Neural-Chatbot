from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from numpy import random



class Sampler(object):
    def __init__(self, model, vocabulary, sequence_length):
        self.model = model
        self.vocabulary = vocabulary
        self.sequence_length = sequence_length
        self.inverse_vocabulary = { word: i for i, word in enumerate(vocabulary) }

    def respond(self, input, temperature=1.0, greedy=False):
        input = pad_sequences([self._encode(input)], maxlen=self.sequence_length)
        print (input)
        output = self.model.predict(input)[0]
        print (output.shape)
        output[:, 1] = 0
        indices = [probability.argmax(axis=-1) for probability in output] if greedy \
        else [self.sample(probability, temperature) for probability in output]

        return self._decode(indices)
    
    def sample(self, probabilities, temperature=1.0):
        probabilities = np.asarray(probabilities).astype("float64")
        probabilities = np.log(probabilities + 1e-8) / temperature
        e_probabilities = np.exp(probabilities)
        probabilities = e_probabilities / np.sum(e_probabilities)
        p = random.multinomial(1, probabilities, 1)
        return np.argmax(p)

    def _encode(self, statement):
        statement = '^ ' + statement.strip() + ' $'
        unk_id = self.inverse_vocabulary['unk']
        return [self.inverse_vocabulary.get(word, unk_id) for word in word_tokenize(statement)]
    
    def _decode(self, indices):
        return ' '.join(self.vocabulary[i] for i in indices)