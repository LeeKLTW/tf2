# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(loss='mean_squared_error',optimizer='sgd')
xs = np.arange(0,7)
ys = xs*50+50
model.fit(xs, ys, epochs =1000)
print(model.predict([7.0]))