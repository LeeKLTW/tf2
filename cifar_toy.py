# -*- coding: utf-8 -*-
from random import sample
import numpy as np
from tensorflow import keras

batch_size = 16


def main(epochs):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    NUM_CATEGORY = len(np.unique(np.concatenate([y_train, y_test])))
    y_train = keras.utils.to_categorical(y_train, NUM_CATEGORY)
    y_test = keras.utils.to_categorical(y_test, NUM_CATEGORY)

    def train_test_split(*arrays, **options):
        n_arrays = len(arrays)
        if n_arrays == 0:
            raise ValueError("At least one array required as input")

        test_size = options.pop('test_size', None)
        train_size = options.pop('train_size', None)
        shuffle = options.pop('shuffle', True)

        if options:
            raise TypeError("Invalid parameters passed: %s" % str(options))

        if len(arrays[0]) != len(arrays[1]):
            raise Exception("Shape of arrays are not equal.")

        if (test_size != None) and (train_size != None) and (test_size + train_size != 1.0):
            raise Exception("Please check train & test size.")

        idx_li = list(range(len(arrays[0])))

        if test_size:
            if shuffle:
                test_size = int(float(len(idx_li)) * test_size)
                test_idx = sample(idx_li, test_size)
                train_idx = list(set(idx_li) - set(test_idx))
            else:
                test_size = int(float(len(idx_li)) * test_size)
                test_idx = idx_li[:test_size]
                train_idx = idx_li[test_size:]
        else:
            pass
        x_train, y_train, x_val, y_val = arrays[0][train_idx], arrays[1][train_idx], arrays[0][test_idx], arrays[1][
            test_idx]
        return (x_train, y_train), (x_val, y_val)

    (x_train, y_train), (x_val, y_val) = train_test_split(x_train, y_train, test_size=0.1)

    cb_ckpt = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    cb_tboard = keras.callbacks.TensorBoard()
    cb_csv = keras.callbacks.CSVLogger('cifar', separator=',', append=False)
    cb_estop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(NUM_CATEGORY, activation='softmax'))
    model.summary()
    loss = 'categorical_crossentropy'
    optimizer = keras.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
              callbacks=[cb_ckpt, cb_tboard, cb_csv,cb_estop])

    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1)
    args, unparsed = parser.parse_known_args()
    main(args.epochs)
