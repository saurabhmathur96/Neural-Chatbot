from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, GRU, Bidirectional, RepeatVector, TimeDistributed, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np

class Chatbot(object):
    def __init__(self, sequence_length=20, vocabulary_size=20000, hidden_size=300):
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        model = Sequential()
        model.add(Embedding(vocabulary_size, hidden_size, mask_zero=True, input_length=sequence_length, name="input_embedding"))
        model.add(Bidirectional(GRU(hidden_size, return_sequences=False, unroll=True, name="encoder_1")))
        model.add(RepeatVector(sequence_length, name="repeat_vector"))
        model.add(GRU(hidden_size, return_sequences=True, unroll=True, name="decoder_1"))
        model.add(TimeDistributed(Dense(vocabulary_size, activation="softmax", name="output_dense")))


        # SGD(lr=0.0001, momentum=0.9, clipvalue=5.)
        model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model
    
    def fit_generator(self, generator, samples_per_epoch, nb_epoch=1):
        min_lr = 0.00001
        lr = 0.001
        saturate_epoch = 20
        decay_factor = (min_lr - lr) / float(saturate_epoch)
        def decay_lr(epoch):
            return max(lr + epoch * decay_factor, min_lr)
        
        lr_scheduler = LearningRateScheduler(decay_lr)
        checkpointer = ModelCheckpoint("models/checkpoints/checkpoint.h5", monitor="val_loss")
        self.model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, callbacks=[checkpointer])
    
    def respond(self, input_sequence, temperature=1.0):
        def sample(probabilities, temperature=1.0):
            probabilities = np.asarray(probabilities).astype("float64")
            probabilities = np.log(probabilities + 1e-8) / temperature
            e_probabilities = np.exp(probabilities)
            probabilities = e_probabilities / np.sum(e_probabilities)
            p = np.random.multinomial(1, probabilities, 1)
            return np.argmax(p)
        
        y = self.model.predict(input_sequence)[0]
        y[:, 2] = 0.
        y[:, 0] = 0.
        return [sample(p, temperature) for p in y]
        


    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path, by_name=True)

"""
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_dimension, input_length=sequence_length, mask_zero=True, name="input_embedding"))
model.add(LSTM(hidden_size, return_sequences=False, go_backwards=True, name="encoder_lstm_1"))
model.add(RepeatVector(sequence_length, name="repeat_vector"))
model.add(LSTM(hidden_size, return_sequences=True, name="decoder_lstm"))
model.add(Dense(hidden_size, name="hidden_dense"))
model.add(BatchNormalization(name="hidden_batchnorm"))
model.add(Activation("relu", name="output_relu"))
model.add(TimeDistributed(Dense(vocabulary_size, name="output_dense")))
model.add(BatchNormalization(name="output_batchnorm"))
model.add(Activation("softmax", name="output_softmax"))
"""
