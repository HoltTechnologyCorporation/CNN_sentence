
import numpy as np
from collections import defaultdict


def load_sentences(train_file, tagField=1, textField=2):
    """
    Loads sentences.
    :param train_file: filename containing labeled sentences in TSV format.
    :return: sents (paired with labels), word doc freq, list of labels.
    """
    sents = []
    tags = {}
    word_df = defaultdict(int)
    with open(train_file, "rb") as f:
        for line in f:       
            fields = line.strip().split("\t")
            text = fields[textField]
            tag = fields[tagField]
            if tag not in tags:
                tags[tag] = len(tags)
            clean_text = text.lower()
            words = clean_text.split()
            for word in set(words):
                word_df[word] += 1
            pair = (words, tags[tag])
            sents.append(pair)
    labels = [0] * len(tags)
    for tag,i in tags.iteritems():
        labels[i] = tag
    return sents, word_df, labels


def load_vectors(fname, binary=True):
    """
    Loads word vectors from file in word2vec format.
    :param fname: name of file in word2vec format.
    :return: vectors and word list.
    """
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, embeddings_size = map(int, header.split())
        words = [''] * vocab_size
        vectors = np.empty((vocab_size, embeddings_size), dtype='float32')
        if binary:
            binary_len = np.dtype('float32').itemsize * embeddings_size
            for i in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                words[i] = word
                vectors[i,:] = np.fromstring(f.read(binary_len), dtype='float32')  
        else:                   # text
            for i,line in enumerate(f):
                items = line.split()
                words[i] = unicode(items[0], 'utf-8')
                vectors[i,:] = np.array(map(float, items[1:]))
    return vectors, words


def load_word_vectors(fname, word_index, binary=True):
    """
    Loads word vectors from file in word2vec format.
    :param fname: name of file in word2vec format.
    :return: vectors and word list.
    """
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, embeddings_size = map(int, header.split())
        vectors = np.zeros((len(word_index), embeddings_size), dtype='float32')
        if binary:
            binary_len = np.dtype('float32').itemsize * embeddings_size
            for i in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in word_index:
                    vectors[word_index[word],:] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        else:                   # text
            for i,line in enumerate(f):
                items = line.split()
                word = unicode(items[0], 'utf-8')
                if word in word_index:
                    vectors[word_index[word],:] = np.array(map(float, items[1:]))
    high = 2.38 / np.sqrt(len(vectors) + embeddings_size) # see (Bottou '88)
    for i,v in enumerate(vectors):
        if np.count_nonzero(v) == 0:
            vectors[i:] = np.random.uniform(-high, high, embeddings_size)
    return np.asarray(vectors, dtype="float32")

def add_unknown_words(vectors, words, word_df, k, min_df=1):
    """
    Create word vector for words in :param word_df: that occur in at least :param min_df: documents.
    :param word_df: dictionary of word document frequencies.
    :param k: size of embedding vectors.
    """
    wordset = set(words)
    high = 2.38 / np.sqrt(len(vectors)) # see (Bottou '88)
    start = len(vectors)
    end = start
    for word in word_df:
        if word not in wordset and word_df[word] >= min_df:
            end += 1
            words.append(word)
    vectors.resize((end, k), refcheck=False)
    for i in range(start, end):
        vectors[i:] = np.random.uniform(-high, high, k)
