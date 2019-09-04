import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

WORD_EMBEDDING_DIM = 50
NUMBER_OF_ENCODER_UNITS = 50
NUMBER_OF_DECODER_UNITS = 50
RANDOM_SEED = 101
SEQUENCE_MAX_LENGTH = 50
EPOCH = 100
LEARNING_RATE = 0.01
CORPUS_LOCATION = '/home/hatef/courses/Term-2/NLP/HWs/CA4/xx.txt'


# this function returns all sentences and their corresponding tags, vocabulary and tag list
def fetch_data():
    from util import Bijenkhan
    sentences, tags = [], []
    bijenkhan = Bijenkhan(CORPUS_LOCATION)

    data_gen = bijenkhan.sent_tag_gen(100)
    for lsents, ltags in data_gen:
        sentences.extend(lsents)
        tags.extend(ltags)

    return sentences, tags


# this method receives a list of tokenized list and
# returns word2ind
def token2ind(tokenized_list):
    """
     this method receives a list of tokenized list and
      returns word2ind
    :param tokenized_list: [[a,b,c],[c],[a,b]]
    :return: {a:1, b:2, c:3}
    """
    token2ind = {}
    vocabset = set()
    index = 1
    [[vocabset.add(w) for w in s] for s in tokenized_list]
    for word in vocabset:
        token2ind[word] = index
        index += 1
    return token2ind


def tosequence(tokenized_list, token2id):
    sequences = []
    [sequences.append(list(map(lambda w: token2id[w], s))) for s in tokenized_list]
    return sequences


sentences, tags = fetch_data()

tag2ind = token2ind(tags)
word2ind = token2ind(sentences)

sentences_sequence = tosequence(sentences, word2ind)
tags_sequence = tosequence(tags, tag2ind)

sequence_max_length = max(len(s) for s in sentences_sequence)

encoder_inputs = pad_sequences(sentences_sequence, maxlen=sequence_max_length)
encoder_targets = pad_sequences(sentences_sequence, maxlen=sequence_max_length)

# shuffle data
np.random.seed(RANDOM_SEED)
shuffle_indexes = np.random.permutation(np.arange(len(encoder_inputs)))
encoder_inputs = encoder_inputs[shuffle_indexes]
encoder_targets = encoder_targets[shuffle_indexes]

# real sentence and target
x = tf.placeholder(dtype=tf.int32, shape=(None, min([sequence_max_length, SEQUENCE_MAX_LENGTH])))
y = tf.placeholder(dtype=tf.int32, shape=(None, min([sequence_max_length, SEQUENCE_MAX_LENGTH])))

embedding_matrix = tf.Variable(tf.random_uniform([len(word2ind) + 1, WORD_EMBEDDING_DIM], minval=-1, maxval=1))
embedding = tf.nn.embedding_lookup(embedding_matrix, x)

print(f'shape of embedding: {embedding.get_shape()}')
# shape of embedding: (?, 34, 50)

cell = tf.keras.layers.GRU(NUMBER_OF_ENCODER_UNITS, return_sequences=True, return_state=True,
                           recurrent_initializer='glorot_uniform')
encoder = tf.keras.layers.Bidirectional(cell)(embedding)
