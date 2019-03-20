# encoding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import Model
from tensorflow.keras import layers

EPOCHS = 1000
BUFFER_SIZE = 10000
BATCH_SIZE = 128
noise_dim = 100
next_example_to_generate = 16
seed = tf.random.normal([next_example_to_generate, noise_dim])
chekpoint_dir = './train_checkpoints'
checkpoint_prefix = os.path.join(chekpoint_dir, 'ckpt')


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)  # 64 = model.output_shape[[1 or 2]] * strides[[1 or 2]]  up
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, [5, 5], strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    assert model.output_shape == (None, 14, 14, 128)

    model.add(layers.Conv2D(64, [5, 5], strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 7, 7, 64)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    assert model.output_shape == (None, 1)

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminaor = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimzer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminaor, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def train(dataset, epochs):
    for epoch in range(epochs):
        print(f'start {epoch}')
        for image_batch in dataset:
            train_step(image_batch)
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            checkpoint.save(checkpoint_prefix)
            print(f'Save at  {checkpoint_prefix}')


def main():

    global discriminator
    global generator
    global checkpoint
    global discriminator_optimizer
    global generator_optimzer
    global cross_entropy

    discriminator = make_discriminator()
    generator = make_generator_model()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimzer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(generator_optimzer=generator_optimzer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)
    x_train = x_train / 255
    x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    x_train_dataset = x_train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    train(x_train_dataset, EPOCHS)


if __name__ == '__main__':
    main()
