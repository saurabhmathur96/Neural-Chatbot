from __future__ import absolute_import

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import LSTM, activations, Wrapper
from keras.layers import Lambda, merge, GRU
from keras.layers import ELU
from keras.initializers import Zeros
from keras.layers.merge import concatenate


class AttentionWrapper(Wrapper):
    def __init__(self, layer, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM) or isinstance(layer, GRU)
        super(AttentionWrapper, self).__init__(layer, **kwargs)
        self.supports_masking = True
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionWrapper, self).build()

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception(
                'Layer could not be build: No information about expected input shape.')

        kernel_initializer = self.layer.kernel_initializer
        self.U_a = self.layer.add_weight((self.layer.units, self.layer.units), name='{}_U_a'.format(
            self.name), initializer=kernel_initializer)
        self.b_a = self.layer.add_weight(
            (self.layer.units,), name='{}_b_a'.format(self.name), initializer=Zeros())

        self.U_m = self.layer.add_weight((attention_dim, self.layer.units), name='{}_U_m'.format(
            self.name), initializer=kernel_initializer)
        self.b_m = self.layer.add_weight(
            (self.layer.units,), name='{}_b_m'.format(self.name), initializer=Zeros())

        if self.single_attention_param:
            self.U_s = self.layer.add_weight((self.layer.units, 1), name='{}_U_s'.format(
                self.name), initializer=kernel_initializer)
            self.b_s = self.layer.add_weight(
                (1,), name='{}_b_s'.format(self.name), initializer=Zeros())
        else:
            self.U_s = self.layer.add_weight((self.layer.units, self.layer.units), name='{}_U_s'.format(
                self.name), initializer=kernel_initializer)
            self.b_s = self.layer.add_weight(
                (self.layer.units,), name='{}_b_s'.format(self.name), initializer=Zeros())

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def step(self, x, states):
        h, params = self.layer.step(x, states)
        attention = states[-1]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.units, axis=1)
        else:
            h = h * s

        return h, params

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output


Maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False),
                 output_shape=lambda x: (x[0], x[2]))
Maxpool.supports_masking = True


def Encoder(hidden_size, activation=None, return_sequences=True, bidirectional=False, use_gru=True):
    if activation is None:
        activation = ELU()
    if use_gru:
        def _encoder(x):
            if bidirectional:
                branch_1 = GRU(hidden_size, activation='linear',
                               return_sequences=return_sequences, go_backwards=False)(x)
                branch_2 = GRU(hidden_size, activation='linear',
                               return_sequences=return_sequences, go_backwards=True)(x)
                x = concatenate([branch_1, branch_2])
                x = activation(x)
                return x
            else:
                x = GRU(hidden_size, activation='linear',
                        return_sequences=return_sequences)(x)
                x = activation(x)
                return x
    else:
        def _encoder(x):
            if bidirectional:
                branch_1 = LSTM(hidden_size, activation='linear',
                                return_sequences=return_sequences, go_backwards=False)(x)
                branch_2 = LSTM(hidden_size, activation='linear',
                                return_sequences=return_sequences, go_backwards=True)(x)
                x = concatenate([branch_1, branch_2])
                x = activation(x)
            else:
                x = LSTM(hidden_size, activation='linear',
                         return_sequences=return_sequences)(x)
                x = activation(x)
                return x
    return _encoder


def AttentionDecoder(hidden_size, activation=None, return_sequences=True, bidirectional=False, use_gru=True):
    if activation is None:
        activation = ELU()
    if use_gru:
        def _decoder(x, attention):
            if bidirectional:
                branch_1 = AttentionWrapper(GRU(hidden_size, activation='linear', return_sequences=return_sequences,
                                                go_backwards=False), attention, single_attention_param=True)(x)
                branch_2 = AttentionWrapper(GRU(hidden_size, activation='linear', return_sequences=return_sequences,
                                                go_backwards=True), attention, single_attention_param=True)(x)
                x = concatenate([branch_1, branch_2])
                return activation(x)
            else:
                x = AttentionWrapper(GRU(hidden_size, activation='linear',
                                         return_sequences=return_sequences), attention, single_attention_param=True)(x)
                return activation(x)
    else:
        def _decoder(x, attention):
            if bidirectional:
                branch_1 = AttentionWrapper(LSTM(hidden_size, activation='linear', return_sequences=return_sequences,
                                                 go_backwards=False), attention, single_attention_param=True)(x)
                branch_2 = AttentionWrapper(GRU(hidden_size, activation='linear', return_sequences=return_sequences,
                                                go_backwards=True), attention, single_attention_param=True)(x)
                x = concatenate([branch_1, branch_2])
                x = activation(x)
                return x
            else:
                x = AttentionWrapper(LSTM(hidden_size, activation='linear', return_sequences=return_sequences),
                                     attention, single_attention_param=True)(x)
                x = activation(x)
                return x

    return _decoder


def Decoder(hidden_size, activation=None, return_sequences=True, bidirectional=False, use_gru=True):
    if activation is None:
        activation = ELU()
    if use_gru:
        def _decoder(x):
            if bidirectional:
                x = Bidirectional(
                    GRU(hidden_size, activation='linear', return_sequences=True))(x)
                x = activation(x)
                return x
            else:
                x = GRU(hidden_size, activation='linear',
                        return_sequences=True)(x)
                x = activation(x)
                return x
    else:
        def _decoder(x):
            if bidirectional:
                x = Bidirectional(
                    LSTM(hidden_size, activation='linear', return_sequences=True))(x)
                x = activation(x)
                return x
            else:
                x = LSTM(hidden_size, activation='linear',
                         return_sequences=True)(x)
                x = activation(x)
                return x
    return _decoder
