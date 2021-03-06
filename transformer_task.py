# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
assert len(np.unique(y_train)) == len(np.unique(y_test))  # 46

NUM_CATEGORY = len(np.unique(y_train))
word_index = keras.datasets.reuters.get_word_index()

MAX_FEATURE = max(word_index.values()) + 100  # additional index to fix InvalidArgument Error
len(word_index.values())

np.random.seed(42)
NUM_WORDS = max([len(sent) for sent in x_train])  # 2376


def idx2word(idx):
    return [w for w in word_index if word_index[w] == idx][0]


def _tokenize():
    x_example_idx = np.random.choice(range(len(x_train)), 1)[0]
    x_example = ' '.join([idx2word(idx) for idx in x_train[x_example_idx]])
    print(x_example)
    NUM_WORDS = max([len(sent) for sent in x_train])  # 2376
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts([x_example])
    print(tokenizer.texts_to_sequences([x_example]))


def _pad_x(x):
    return pad_sequences(x, maxlen=NUM_WORDS)


class ScaledDotProductAttention:
    def __init__(self, dropout_rate=0.1):
        """
        dropout_rate: A floating point number of [0, 1].
        """
        self.dropout = keras.layers.Dropout(dropout_rate)

    def __call__(self, q, k, v, mask):
        """
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        """
        scale = K.sqrt(K.shape(k)[-1], dtype='float32')
        attention = keras.layers.Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / scale)([q, k])  # why axes?
        if mask is not None:
            """This masking, combined with fact that the output embeddings are offset by one position, ensures that the
            predictions for position i can depend only on the known outputs at positions less than i. Set to -inf"""
            masked = keras.layers.Lambda(lambda x: (-1e10) * (1 - K.cast(x, 'float32')))(mask)
            attention = keras.layers.Add([attention, masked])
        attention = keras.layers.Activation('softmax')(attention)
        attention = self.dropout(attention)
        head = keras.layers.Lambda(lambda x: K.batch_dot(x[0], x[1]))([attention, v])
        return head, attention


class MultiHeadAttention:
    def __init__(self, n_head, d_model, dropout, **kwargs):
        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head  # 3.2.2
        self.dropout = dropout
        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []
        for _ in range(n_head):
            self.qs_layers.append(keras.layers.TimeDistributed(keras.layers.Dense(self.d_k, use_bias=False)))
            self.ks_layers.append(keras.layers.TimeDistributed(keras.layers.Dense(self.d_k, use_bias=False)))
            self.vs_layers.append(keras.layers.TimeDistributed(keras.layers.Dense(self.d_v, use_bias=False)))

        self.attention = ScaledDotProductAttention()
        self.w_o = keras.layers.TimeDistributed(keras.layers.Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        multihead = []
        multiattention = []
        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](k)
            head, attention = self.attention(qs, ks, vs)
            multihead.append(head)
            multiattention.append(attention)
        multihead = keras.layers.Concatenate()(multihead) if self.n_head > 1 else multihead[0]
        multiattention = keras.layers.Concatenate()(multiattention) if self.n_head > 1 else multiattention[0]

        multihead = self.w_o(multihead)
        multihead = keras.layers.Dropout(self.dropout)(multihead)
        return multihead, multiattention

    def compute_output_shape(self, input_shape):
        pass


class ADDNORM(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(ADDNORM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=keras.initializers.Ones())
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=keras.initializers.Zeros())
        super(ADDNORM, self).build(input_shape)

    def __call__(self, x, output):
        """
        :param x: residual input
        :param output: sublayer output for input
        :return:
        """
        mean = K.mean(output, axis=-1, keepdims=True)
        std = K.std(output, axis=-1, keepdims=True)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta  # normalized
        output = keras.layers.Add()([output, x])  # Add, residual
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionwiseFFN:
    def __init__(self, d_model=512, d_ff=2048):
        self.w_1 = keras.layers.Conv1D(d_ff, 1, activation='relu')
        self.w_2 = keras.layers.Conv1D(d_model, 1)  # linear

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        return output


class EncoderLayer:
    def __init__(self, d_model, d_ff, n_head, dropout_rate=0.1):
        self.multi_head_attention = MultiHeadAttention(n_head, d_model, dropout_rate)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.addnorm1 = ADDNORM()
        self.pos_ffn = PositionwiseFFN(d_model, d_ff)
        self.addnorm2 = ADDNORM()

    def __call__(self, enc_inputs, mask):
        """enc_inputs: input embedding after Positional Encoding."""
        multihead, multiattention = self.multi_head_attention(enc_inputs, enc_inputs, enc_inputs, mask=mask)
        multihead = self.dropout(multihead)
        multihead_addnorm = self.addnorm1(enc_inputs, multihead)

        pos_ffn_output = self.pos_ffn(multihead_addnorm)
        pos_ffn_output = self.dropout(pos_ffn_output)
        pos_ffn_output_addnorm = self.addnorm2(multihead_addnorm, pos_ffn_output)

        return pos_ffn_output_addnorm, multiattention


class DecoderLayer:
    def __init__(self, d_model, d_ff, n_head, dropout_rate=0.1):
        self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model, dropout_rate)
        self.addnorm1 = ADDNORM()

        self.multi_head_attention = MultiHeadAttention(n_head, d_model, dropout_rate)
        self.addnorm2 = ADDNORM()

        self.pos_ffn = PositionwiseFFN(d_model, d_ff)
        self.addnorm3 = ADDNORM()

    def __call__(self, dec_inputs, enc_outputs, dec_mask=None, enc_mask=None, dec_last_state=None):
        dec_last_state = dec_inputs if dec_last_state is None else dec_last_state
        m_multihead, dec_multiattention = self.masked_multi_head_attention(dec_inputs, dec_last_state, dec_last_state,
                                                                           mask=enc_mask)
        m_multihead_addnorm = self.addnorm1(dec_inputs, m_multihead)

        multihead, enc_multiattention = self.multi_head_attention(m_multihead_addnorm, enc_outputs, enc_outputs,
                                                                  mask=enc_mask)
        multihead_addnorm = self.addnorm2(m_multihead, multihead)

        pos_ffn_output = self.pos_ffn(multihead_addnorm)
        pos_ffn_output_addnorm = self.addnorm2(multihead_addnorm, pos_ffn_output)
        return pos_ffn_output_addnorm, dec_multiattention, enc_multiattention


class Encoder:
    def __init__(self, d_model, d_ff, n_head=8, n_layers=6, dropout_rate=0.1):
        self.layers = [EncoderLayer(d_model, d_ff, n_head, dropout_rate) for _ in range(n_layers)]

    def __call__(self, inputs_embedding, input_sequence, return_attentions=False):
        if return_attentions:
            attentions = []
        x = inputs_embedding
        for enc_layer in self.layers:
            x, att = enc_layer(x)
            if return_attentions: attentions.append(att)
        return (x, attentions) if return_attentions else x


def get_PAD_mask(q, k):  # todo check it latter
    ones = K.expand_dims(K.ones_like(q, dtype='float32'), -1)
    mask = K.cast(K.expand_dims(K.not_qual(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def get_sub_mask(s):  # todo check it latter
    len_s = K.shape(s)[1]
    mask = K.cumsum(K.eye(len_s), 1)
    return mask


class Deocoder:
    def __init__(self, d_model, d_ff, n_head=8, n_layers=6, dropout_rate=0.1):
        self.layers = [DecoderLayer(d_model, d_ff, n_head, dropout_rate) for _ in range(n_layers)]

    def __call__(self, outputs_embedding, input_seq, output_seq, encoder_output, return_attentions=False):
        if return_attentions:
            encoder_attentions = []
            decoder_attentions = []

        dec_pad_mask = keras.layers.Lambda(lambda x: get_PAD_mask(x, x))(output_seq)
        dec_sub_mask = keras.layers.Lambda(get_sub_mask)(output_seq)
        dec_mask = keras.layers.Lambda(lambda x: K.minimum(x[0], x[1]))([dec_pad_mask, dec_sub_mask])
        enc_mask = keras.layers.Lambda(lambda x: get_PAD_mask(x[0], x[1]))([output_seq, input_seq])

        x = outputs_embedding
        for dec_layer in self.layers:
            x, dec_multiattention, enc_multiattention = dec_layer(x, encoder_output, dec_mask, enc_mask)
            if return_attentions:
                decoder_attentions.append(dec_multiattention)
                encoder_attentions.append(enc_multiattention)
        return (x, decoder_attentions, encoder_attentions) if return_attentions else x



class Transformer:
    def __init__(self, maxlen, d_model=512, d_ff=2048, n_head=8, d_v=64, layers=2, dropout=0.1):
        self.input_embedding = None  # dot it here
        self.positional_encoding = None  # do it here
        self.output_embedding = None
        self.encoder = None
        self.decoder = None
        self.linear = None
        self.softmax = None

class Transformer(keras.Model):
    def __init__(self,in_dim=784, out_dim=10,d_model=512,n_layers=10,dp_rate=0.5):
        self.n_model = d_model
        self.n_layers = n_layers
        self.dp_rate = dp_rate
        self.out_dim = out_dim

        super(Transformer,self).__init__(name='Transformer')

        self.input_layer = keras.layers.Input(shape=(784,))
        for i in range(self.n_layers):
            exec(f"self.dense_{i+1} = keras.layers.Dense(d_model,activation='relu',name='dense_{i+1}')")
            if self.dp_rate > 0:
                exec(f"self.dp_{i + 1} = keras.layers.Dropout(dp_rate,name='dp_{i + 1}')")

        self.output_layer = keras.layers.Dense(self.out_dim,activation='softmax')