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
    def __init__(self, dropout=0.1):
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, q, k, v, mask):
        x = K.dot(q, k)
        x /= K.sqrt(K.shape(k)[-1])  # sclaed
        if mask is not None:
            """This masking, combined with fact that the output embeddings are offset by one position, ensures that the
            predictions for position i can depend only on the known outputs at positions less than i. Set to -inf"""
            masked = keras.layers.Lambda(lambda x: (-1e10)*(1-K.cast(x,'float32')))(mask)
            masked = keras.layers.Add([x,masked])

            pass  # todo mask
        x = keras.layers.Activation('softmax')(x)
        y = K.dot(x, v)
        return y


class MultiHeadAttention:
    def __init__(self, n_head, size_per_head, mask_right=False, **kwargs):
        self.n_head = n_head
        self.size_per_head = size_per_head
        self.output_dim = self.n_head * self.size_per_head
        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []
        for _ in range(n_head):
            self.qs_layers.append(keras.layers.TimeDistributed(keras.layers.Dense(size_per_head, use_bias=False)))
            self.ks_layers.append(keras.layers.TimeDistributed(keras.layers.Dense(size_per_head, use_bias=False)))
            self.vs_layers.append(keras.layers.TimeDistributed(keras.layers.Dense(size_per_head, use_bias=False)))

        self.attention = ScaledDotProductAttention()
        self.w_o = keras.layers.TimeDistributed(keras.layers.Dense(n_head * size_per_head))

    def build(self, input_shape):
        # query, key, value
        self.WQ = self.add_weight(name='WQ', shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_initializer', trainable=True)
        self.WK = self.add_weight(name='WK', shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_initializer', trainable=True)
        self.WV = self.add_weight(name='WV', shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_initializer', trainable=True)
        super(MultiHeadAttention, self).build(input_shape)

    def __call__(self, x):
        # todo implement mode 1
        pass

    def compute_output_shape(self, input_shape):
        pass

# test here