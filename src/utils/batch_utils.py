from keras.preprocessing.sequence import pad_sequences
from numpy import random
from numpy import zeros


class BatchIterator(object):
    def __init__(self, questions, answers, vocabulary, batch_size, sequence_length, one_hot_target, stream=False):
        random.seed(0)
        self.sequence_length = sequence_length
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.one_hot_target = one_hot_target
        self.stream = stream

        self.questions = questions
        self.answers = answers
        self.inverse_vocabulary = dict((word, i) for i, word in enumerate(self.vocabulary))
    def to_one_hot(self, y):
        out = zeros(shape=(self.batch_size, self.sequence_length, len(self.vocabulary)), dtype=bool)
        for batch in range(self.batch_size):
            for index, word in enumerate(y[batch]):
                out[batch, index, word] = True
        return out

    def next_batch(self):
        inverse_vocabulary = self.inverse_vocabulary
        if self.stream:
            q = [[inverse_vocabulary[word] for word in next(self.questions).strip().split() ] for i in range(len(batch_size))]
            a = [[inverse_vocabulary[word] for word in next(self.answers).strip().split() ] for i in range(len(batch_size))]
        else:
            n_example = len(self.answers)
            indices = random.randint(0, n_example, size=(self.batch_size))
            q = [[inverse_vocabulary[word] for word in self.questions[i].split()] for i in indices]
            a = [[inverse_vocabulary[word] for word in self.answers[i].split()] for i in indices]

        X = pad_sequences(q, maxlen=self.sequence_length)
        y = pad_sequences(a, maxlen=self.sequence_length)

        if self.one_hot_target:
            return (X, self.to_one_hot(y))
        else:
            return (X, y)

    
    
    

    