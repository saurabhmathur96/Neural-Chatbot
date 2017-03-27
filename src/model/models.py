from keras.models import Model
from keras import backend as K
from keras.layers import *
from sequence_blocks import *
from keras.optimizers import *

def seq2seq(sequence_length, vocabulary_size, hidden_size, use_gru=True):

    # Input Block
    i = Input(shape=(sequence_length,))
    x = Embedding(vocabulary_size, 128, mask_zero=True)(i)
    
    # Encoder Block
    x = Encoder(hidden_size, return_sequences=False, use_gru=use_gru)(x)
    x = Dropout(.5)(x)
    x = Encoder(hidden_size, return_sequences=False, use_gru=use_gru)(x)
    x = Dropout(.5)(x)
    
    x = Dense(hidden_size, activation='linear')(x)
    x = ELU()(x)
    x = RepeatVector(sequence_length)(x)

    # Decoder Block
    x = Decoder(hidden_size, return_sequences=True, use_gru=use_gru)(x)
    x = Dropout(.5)(x)
    x = Decoder(hidden_size, return_sequences=True, use_gru=use_gru)(x)
    x = Dropout(.5)(x)
    
    x = TimeDistributed(Dense(vocabulary_size, activation='softmax'))(x)
    
    model = Model(inputs=i, outputs=x)

    opt = Adam(lr=0.0001, clipvalue=1.)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    
    return model



def seq2seq_attention(sequence_length, vocabulary_size, hidden_size, use_gru=True):

    # Input Block
    i = Input(shape=(sequence_length,))
    x = Embedding(vocabulary_size, 128, mask_zero=True)(i)

    # Encoder Block
    x = Encoder(hidden_size, return_sequences=True, bidirectional=True, use_gru=use_gru)(x)
    x = Dropout(.5)(x)
    x = Encoder(hidden_size, return_sequences=True, bidirectional=True, use_gru=use_gru)(x)
    x = Dropout(.5)(x)

    x = TimeDistributed(Dense(hidden_size, activation='linear'))(x)
    x = ELU()(x)
    attention = Maxpool(x)
    
    # Decoder Block
    x = AttentionDecoder(hidden_size, return_sequences=True, bidirectional=True, use_gru=use_gru)(x, attention)
    x = Dropout(.5)(x)
    x = AttentionDecoder(hidden_size, return_sequences=True, bidirectional=True, use_gru=use_gru)(x, attention)
    x = Dropout(.5)(x)


    x = TimeDistributed(Dense(vocabulary_size, activation='softmax'))(x)

    model = Model(inputs=i, outputs=x)
    
    opt = Adam(lr=0.0001, clipvalue=1.)
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    
    return model

    