# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Bidirectional, \
    GlobalMaxPooling1D, Flatten, GlobalAveragePooling1D

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
assert len(np.unique(y_train)) == len(np.unique(y_test)) #46

NUM_CATEGORY = len(np.unique(y_train))
word_index = keras.datasets.reuters.get_word_index()

MAX_FEATURE = max(word_index.values())

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


x_train, x_val, x_test = [_pad_x(x) for x in [x_train, x_val, x_test]]

model = keras.models.Sequential(
    [keras.layers.Embedding(MAX_FEATURE, 1000),
     keras.layers.Dense(1000, activation='relu'),
     keras.layers.Dense(NUM_CATEGORY,activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(x_train,y_train)
print(x_train.shape)
print(y_train.shape)