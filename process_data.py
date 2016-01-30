#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from collections import defaultdict
import sys, re

tagField = 2
textField = 3
label = {'negative': 0,
         'positive': 1,
         'neutral': 2
         }

def build_data_cv(train_file, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    :return: sents (with class and split properties), word doc freq.
    """
    revs = []
    vocab = defaultdict(int)
    with open(train_file, "rb") as f:
        for line in f:       
            fields = line.strip().split("\t")
            text = fields[textField]
            tag = fields[tagField]
            if clean_string:
                clean_text = clean_str(text)
            else:
                clean_text = text.lower()
            words = clean_text.split()
            for word in set(words):
                vocab[word] += 1
            datum = {"y": label[tag],
                     "text": clean_text,
                     "num_words": len(words),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix and word index dict. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
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
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def tokenize(string, no_lower=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Lower case except when no_lower is Ytur
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if no_lower else string.strip().lower()

def tokenize_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]     
    train_file = sys.argv[2]    # Attardi
    clean = int(sys.argv[3])    # Attardi
    np.random.seed(345)         # for replicability
    print "loading data...",        
    revs, vocab = build_data_cv(train_file, cv=10, clean_string=clean)
    max_l = max(x["num_words"] for x in revs)
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    #pickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    file_out = sys.argv[4]      # Attardi
    pickle.dump([revs, W, W2, word_idx_map, vocab], open(file_out, "wb"))
    print "dataset created!"

