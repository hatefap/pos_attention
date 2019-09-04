from keras.preprocessing.sequence import pad_sequences

WORD_EMBEDDING_DIM = 50
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

encoder_inputs = pad_sequences(sentences_sequence, maxlen=sequence_max_length, padding='post')
encoder_targets = pad_sequences(sentences_sequence, maxlen=sequence_max_length, padding='post')
