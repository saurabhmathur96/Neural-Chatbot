import sys
import csv
import json
from itertools import count
from tqdm import tqdm
from models import seq2seq, seq2seq_attention
from keras.optimizers import SGD, Adagrad, Adam

sys.path.append('src/utils')
from batch_utils import BatchIterator
from config_utils import settings

if __name__ == '__main__':
    data_file = settings.data.filtered_path
    with open(data_file) as handle:
        reader = csv.reader(handle)
        questions, answers = zip(*reader)

    vocabulary_file = settings.data.vocabulary_path
    with open(vocabulary_file) as handle:
        vocabulary = json.load(handle)

    batch_size = settings.train.batch_size
    n_iter = settings.train.n_iter # 16384
    n_epoch = settings.train.n_epoch

    for experiment_no in [0, 1, 2, 3]:    
        sequence_length = settings.model.sequence_length
        vocabulary_size = settings.model.vocabulary_size
        hidden_size = settings.model.hidden_size
        print ('Creating model with configuration: {0}'.format(settings.model))

        if experiment_no == 0:
            model = seq2seq_attention(sequence_length, vocabulary_size, hidden_size, use_gru=False, bidirectional_decoder=False, use_elu=False)
        elif experiment_no == 1:
            model = seq2seq_attention(sequence_length, vocabulary_size, hidden_size, use_gru=True, bidirectional_decoder=False, use_elu=False)
        elif experiment_no == 2:
            model = seq2seq_attention(sequence_length, vocabulary_size, hidden_size, use_gru=True, bidirectional_decoder=True, use_elu=False)
        elif experiment_no == 3:
            model = seq2seq_attention(sequence_length, vocabulary_size, hidden_size, use_gru=True, bidirectional_decoder=True, use_elu=True)
        else :
            print ('unknown experiment_no')
        
        print (model.summary())
        
        print ('Initializing training with configuration: {0}'.format(settings.train))
        iterator = BatchIterator(questions, answers, vocabulary, batch_size, sequence_length, one_hot_target=True)
        # generator = (iterator.next_batch() for _ in count(start=0, step=1)) # infinite generator
        # model.fit_generator(generator, epochs=2, steps_per_epoch=n_iter * batch_size)
        # 
        bar_format = '{n_fmt}/{total_fmt}|{bar}|ETA: {remaining} - {desc}'
        for epoch in range(n_epoch):
            print ('-' * 80)
            print ('Epoch {0}'.format(epoch))
            print ('-' * 80)
            bar = tqdm(range(1, n_iter+1), total=n_iter, bar_format=bar_format, ncols=80)
            loss = 0.0
            losses = []
            for i in bar:
                batch = iterator.next_batch()
                losses.append(float(model.train_on_batch(*batch)))
                loss += losses[-1]
                bar.set_description('loss: {0:.2f}'.format( float(loss)/i ))
                bar.refresh()

        losses_path = 'models/experiment_{0}_loss.txt'.format(experiment_no)
        print ('Saving training loss to {0}'.format(losses_path))
        json.dump(losses, open(losses_path, 'w'))

        model_weights_path = 'models/experiment_{0}.h5'.format(experiment_no)
        print ('Saving model weights to {0}'.format(model_weights_path))
        model.save_weights(model_weights_path)

        model_architecture_path = 'models/experiment_{0}.json'.format(experiment_no)
        print ('Saving model architecture to {0}'.format(model_architecture_path))
        open(model_architecture_path, 'w').write(model.to_json())
