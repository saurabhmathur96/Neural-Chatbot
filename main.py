from __future__ import print_function
from keras.preprocessing import sequence
from datasets import CornellMovieDialogs
from chatbot import Chatbot
import os
import numpy as np

def batch_generator(dataset, sequence_length, batch_size=32):
    source = dataset.data_generator()
    vocabulary_size = dataset.vocabulary_size
    while True:
        statements = []
        response_matrix = np.zeros([batch_size, sequence_length, vocabulary_size], dtype=np.bool)
        for b in range(batch_size):
            while True:
                statement, response = next(source)
                statement = dataset.encode_sequence(statement)
                response = dataset.encode_sequence(response)
                if len(statement) < sequence_length and len(response) < sequence_length:
                    break
            
            statements.append(statement)
            
            for i, r in enumerate(response):
                response_matrix[b, i, r] = 1
        
        statement_matrix = sequence.pad_sequences(statements, maxlen=sequence_length)
        yield statement_matrix, response_matrix
            


def gen(dataset, sequence_length, batch_size=32, max_samples=50000):
    while True:
        g = batch_generator(dataset, sequence_length, batch_size=batch_size)
        for i in range(max_samples/batch_size):
            yield next(g)

def main():
    dataset = dataset = CornellMovieDialogs(data_directory="data/", vocabulary_size=20000)
    if not os.path.exists("data/cleaned"):
        dataset.make_conversations()
    else:
        dataset.load()

    bot = Chatbot(sequence_length=10, hidden_size=128, vocabulary_size=dataset.vocabulary_size)
    # bot.load("models/checkpoints/checkpoint.h5")
    conversation_generator = gen(dataset, bot.sequence_length, batch_size=10, max_samples=5000) #gen(dataset, bot.sequence_length, batch_size=10) # 
    bot.fit_generator(conversation_generator, samples_per_epoch=10000, nb_epoch=100)
    bot.save("models/chatbot.h5")

def test():
    dataset = CornellMovieDialogs(data_directory="data/", vocabulary_size=20000)
    if not os.path.exists("data/cleaned"):
        dataset.make_conversations()
    else:
        dataset.load()

    bot = Chatbot(sequence_length=10, hidden_size=128, vocabulary_size=dataset.vocabulary_size)
    bot.load("models/checkpoints/checkpoint.h5")
    while True:
        print (">>>", end="")
        sentence = raw_input()
        # sentence = dataset.clean(sentence)
        sentence = dataset.encode_sequence(sentence)
        sentence = sequence.pad_sequences([sentence], maxlen=bot.sequence_length)
        print (sentence)
        for t in [1]:
            response = bot.respond(sentence, temperature=t)
            # print(response)
            response = dataset.decode_sequence(response)
            start_index = response.index("START")+1 if "START" in response else 0
            end_index = response.index("END") if "END" in response else len(response)
            response = response[start_index:end_index]
            print (" ".join(response))
            
if __name__ == "__main__":
    # loss: 2.1947 - acc: 0.2345
    test()
    main()
    
