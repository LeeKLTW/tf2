# -*- coding: utf-8 -*-

# imdb case
from __future__ import print_function
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

max_features = 2000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding,GlobalAveragePooling1D,Dropout, Dense
from attention_keras import Attention, Position_Embedding

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
# embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print('Train...')
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
# no embedding 0.82624
# w/ embedding 0.8212
model.evaluate(x_test, y_test)



# reuters case

from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data()
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


import numpy as np
len(np.unique(y_test))
n_category = 46
n_words = np.max(x_train)+10


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding,GlobalAveragePooling1D,Dropout, Dense
from attention_keras import Attention, Position_Embedding

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(n_words, 128)(S_inputs)
# embeddings = Position_Embedding()(embeddings)
O_seq = Attention(16,64)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(n_category, activation='softmax')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print('Train...')
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_data=(x_test, y_test))

# with pe loss: 0.7477 - accuracy: 0.8153 - val_loss: 1.1340 - val_accuracy: 0.7302
# w/out pe loss: 0.6685 - accuracy: 0.8336 - val_loss: 1.1710 - val_accuracy: 0.7413
#
model.evaluate(x_test, y_test)
# 8982/8982 [==============================] - 27s 3ms/sample - loss: 0.1411 - accuracy: 0.9536 - val_loss: 1.6560 - val_accuracy: 0.7324
