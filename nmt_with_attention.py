# encoding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import re
import numpy as np
import os
import unicodedata
import io


def download_dataset():
    global path_to_file
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
    return path_to_file


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'([?.!,¿])', r' \1 ', w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!¿]+", " ", w)
    w = w.rstrip().strip()
    w = "<start> " + w + " <end>"
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    inp_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    targ_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return inp_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang_tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print(f"{t} ----> {lang_tokenizer.index_word[t]}")


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention.call(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # bool
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    BATCH_SIZE = 64
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden, = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang_tokenizer.index_word[predicted_id] + ' '

        if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    plt.show()


def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    print('Input: %s' % sentence.encode('utf-8'))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot,sentence.split(' '),result.split(' '))

translate('vi')

translate(u'en casa')



def main():
    global encoder
    global decoder
    global inp_lang_tokenizer
    global targ_lang_tokenizer
    global optimizer
    global max_length_inp
    global max_length_targ
    global UNITS
    NUM_EXAMPLES = 10000
    EPOCHS = 10

    inp_tensor, targ_tensor, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(path_to_file, NUM_EXAMPLES)

    max_length_targ, max_length_inp = max_length(targ_tensor), max_length(inp_tensor)
    inp_tensor_train, inp_tensor_test, targ_tensor_train, targ_tensor_test = train_test_split(inp_tensor, targ_tensor,
                                                                                              test_size=0.2)

    BUFFER_SIZE = len(inp_tensor_train)
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = len(inp_tensor_train) // BATCH_SIZE
    EMBEDDING_DIM = 256
    UNITS = 1024
    vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
    vocab_targ_size = len(targ_lang_tokenizer.word_index) + 1

    dataset = tf.data.Dataset.from_tensor_slices((inp_tensor_train, targ_tensor_train)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    # inp_example, targ_example = next(iter(dataset))
    # sample_output, sample_hidden = encoder.call(inp_example, sample_hidden)
    # attention_layer = BahdanauAttention(10)
    # attention_result, attention_weights = attention_layer.call(query=sample_hidden, values=sample_output)
    # attention_result
    # attention_weights.shape
    #
    decoder = Decoder(vocab_targ_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    sample_decoder_output, _, _ = decoder.call(tf.random.uniform((64, 1)), sample_hidden, sample_output)
    #
    #
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    for epoch in range(EPOCHS):
        start = time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for batch, (inp, targ) in enumerate(dataset.take(STEPS_PER_EPOCH)):
            batch_loss = train_step(inp, targ,enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch + 1, batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss{:.4f}'.format(epoch + 1, total_loss / STEPS_PER_EPOCH))
        print('Time take for 1 epoch {} sec\n'.format(time() - start))
