# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
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


class MultiHeadAttention(Layer):
    def __init__(self,n_head,size_per_head,mask_right=False,**kwargs):
        self.n_head = n_head
        self.size_per_head = size_per_head
        self.output_dim = self.n_head * self.size_per_head
        self.mask_right = mask_right
        super(MultiHeadAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        # query, key, value
        self.WQ = self.add_weight(name='WQ', shape=(input_shape[0][-1], self.output_dim), initializer='glorot_initializer',trainable=True)
        self.WK = self.add_weight(name='WK', shape=(input_shape[1][-1], self.output_dim), initializer='glorot_initializer',trainable=True)
        self.WV = self.add_weight(name='WV', shape=(input_shape[2][-1], self.output_dim), initializer='glorot_initializer',trainable=True)
        super(MultiHeadAttention,self).build(input_shape)

    #todo
    def mask(self,inputs, seq_len, mode='mul'):
        pass

    def __call__(self):
        pass
    def compute_output_shape(self,input_shape):
        pass
