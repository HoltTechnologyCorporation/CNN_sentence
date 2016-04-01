#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from collections import defaultdict
import sys, re


def build_data(train_file, clean_string=True, tagField=1, textField=2):
    """
    Loads data.
    :param train_file: filename containing labeled sentences in TSV format.
    :return: sents (paired with labels), word doc freq, list of labels.
    """
    sents = []
    tags = {}
    vocab = defaultdict(int)
    with open(train_file, "rb") as f:
        for line in f:       
            fields = line.strip().split("\t")
            text = fields[textField]
            tag = fields[tagField]
            if tag not in tags:
                tags[tag] = len(tags)
            if clean_string:
                clean_text = clean_str(text)
            else:
                clean_text = text.lower()
            words = clean_text.split()
            for word in set(words):
                vocab[word] += 1
            pair = (words, tags[tag])
            sents.append(pair)
    labels = [0] * len(tags)
    for tag,i in tags.iteritems():
        labels[i] = tag
    return sents, vocab, labels


def get_W(word_vecs):
    """
    Get word matrix and word index dict. W[i] is the vector for word of index i.
    """
    vocab_size = len(word_vecs)
    word_idx_map = {}
    k = len(word_vecs.itervalues().next())
    # CHECKME: why +1?
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    #W[0] = np.zeros(k, dtype='float32') # overridden below
    #W = np.zeros(shape=(vocab_size, k), dtype='float32')            
    for i, word in enumerate(word_vecs):
        W[i] = word_vecs[word]
        word_idx_map[word] = i
    return W, word_idx_map


def load_word2vec(fname, vocab, binary=True):
    """
    Loads word vectors from file in word2vec format
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        if binary:
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        else:                   # text
            for line in f:
                items = line.split()
                word = unicode(items[0], 'utf-8')
                word_vecs[word] = np.array(map(float, items[1:]))
    return word_vecs


def add_unknown_words(word_vecs, vocab, k, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    :param k: size of embedding vectors.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)  


def process_data(train_file, clean, w2v_file=None,
                 tagField=1, textField=2, k=300):
    """
    :param k: embeddigs size (300 for GoogleNews)
    :param cv: cross-validation folds, -1 for none
    :return: sents (paired with labels), vectors, word dictionary, vocabulary, list of labels.
    """
    np.random.seed(345)         # for replicability
    print "loading data...",
    sents, vocab, labels = build_data(train_file,
                                      clean_string=clean,
                                      tagField=tagField, textField=textField)
    max_l = max(len(words) for words,l in sents)
    print "data loaded!"
    print "number of sentences: " + str(len(sents))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    if w2v_file:
        print "loading word2vec vectors...",
        w2v = load_word2vec(w2v_file, vocab, w2v_file.endswith('.bin'))
        # get embeddings size:
        k = len(w2v.itervalues().next())
        print "word2vec loaded (%d, %d)" % (len(w2v), k)
        add_unknown_words(w2v, vocab, k)
        W, word_idx_map = get_W(w2v)
    else:
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab, k)
        W, word_idx_map = get_W(rand_vecs)
    return sents, W, word_idx_map, vocab, labels


