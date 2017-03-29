from keras.preprocessing.sequence import pad_sequences
from numpy import random
from numpy import zeros

random.seed(0)

class BatchIterator(object):
    def __init__(self, questions, answers, vocabulary, batch_size, sequence_length, one_hot_target):
        self.sequence_length = sequence_length
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.one_hot_target = one_hot_target

        inverse_vocabulary = dict((word, i) for i, word in enumerate(self.vocabulary))
        q = ([inverse_vocabulary[word] for word in question.split()] for question in questions)
        a = ([inverse_vocabulary[word] for word in answer.split()] for answer in answers)
        
        self.X = pad_sequences(q, maxlen=self.sequence_length)
        self.y = pad_sequences(a, maxlen=self.sequence_length)

    
    def to_one_hot(self, y):
        out = zeros(shape=(self.batch_size, self.sequence_length, len(self.vocabulary)), dtype=bool)
        for batch in range(self.batch_size):
            for index, word in enumerate(y[batch]):
                out[batch, index, word] = True
        return out

    def next_batch(self):
        n_example = self.X.shape[0]
        indices = random.randint(0, n_example, size=(self.batch_size))
        if self.one_hot_target:
            return (self.X[indices], self.to_one_hot(self.y[indices]))
        else:
            return (self.X[indices], self.y[indices])

    
    
    

    