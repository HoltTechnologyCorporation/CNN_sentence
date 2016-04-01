#!/usr/bin/env python

"""
Training a convolutional network for sentence classification,
as described in paper:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
"""
import cPickle as pickle
import numpy as np
import theano
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")   

# run from everywhere without installing
sys.path.append(".")
from conv_net_classes import *
from process_data import process_data


def sent2indices(sent, word_index, max_l, pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    Drop words non in word_index.
    :param sent: list of words.
    :param word_index: associates an index to each word
    :param max_l: max sentence length
    :param pad: pad size
    """
    x = [0] * pad                # left padding
    for word in sent:
        if word in word_index: # FIXME: skips unknown words
            if len(x) < max_l: # truncate long sent
                x.append(word_index[word])
            else:
                break
    # len(x) includes pad
    rpad = [0] * max(0, max_l + 2 * pad - len(x)) # right padding
    return x + rpad


def read_corpus(filename, word_index, max_l, pad=2, clean_string=False,
                textField=3):
    """
    Load test corpus, in TSV format.
    :param filename: file with sentences.
    :param word_index: word IDs.
    :param max_l: max sentence length.
    :param pad: padding size.
    :param textField: index of field containing text.
    :return: an array, each row consists of sentence word indices
    """
    corpus = []
    with open(filename) as f:
        for line in f:
            fields = line.strip().split("\t")
            text = fields[textField]
            if clean_string:
                text_clean = clean_str(text)
            else:
                text_clean = text.lower()
            # turn sentences into lists of indices
            sent = sent2indices(text_clean.split(), word_index, max_l, pad)
            corpus.append(sent)
    return np.array(corpus, dtype="int32")


def predict(cnn, test_set_x):

    test_set_y_pred = cnn.predict(test_set_x)
    # compile expression
    test_function = theano.function([cnn.x], test_set_y_pred, allow_input_downcast=True)
    return test_function(test_set_x)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CNN sentence classifier.")
    
    parser.add_argument('model', type=str, default='mr',
                        help='model file (default %(default)s)')
    parser.add_argument('input', type=str,
                        help='train/test file in SemEval twitter format')
    parser.add_argument('-train', help='train model',
                        action='store_true')
    parser.add_argument('-clean', help='tokenize text',
                        action='store_true')
    parser.add_argument('-filters', type=str, default='3,4,5',
                        help='n[,n]* (default %(default)s)')
    parser.add_argument('-vectors', type=str,
                        help='word2vec embeddings file (random values if missing)')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability (default %(default)s)')
    parser.add_argument('-hidden', type=int, default=100,
                        help='hidden units in feature map (default %(default)s)')
    parser.add_argument('-epochs', type=int, default=100,
                        help='training iterations (default %(default)s)')
    parser.add_argument('-tagField', type=int, default=1,
                        help='label field in files (default %(default)s)')
    parser.add_argument('-textField', type=int, default=2,
                        help='text field in files (default %(default)s)')

    args = parser.parse_args()

    # theano.config
    theano.config.floatX = 'float32'

    if not args.train:
        # predict
        with open(args.model) as mfile:
            cnn = ConvNet.load(mfile)
            word_index, max_l, pad, labels = pickle.load(mfile)
        test_set_x = read_corpus(args.input, word_index, max_l, pad, textField=args.textField)
        results = predict(cnn, test_set_x)
        # convert indices to labels
        for line, y in zip(open(args.input), results):
            tokens = line.split("\t")
            tokens[args.tagField] = labels[y]
            print "\t".join(tokens),
        sys.exit()

    # training
    sents, U, word_index, vocab, labels = process_data(args.input, args.clean,
                                                       args.vectors,
                                                       args.tagField,
                                                       args.textField)

    # sents is a list of pairs: (list of words, label)
    # vocab: dict of word doc freq
    filter_hs = [int(x) for x in args.filters.split(',')]
    model = args.model
    if args.vectors:
        print "using: word2vec vectors"
    else:
        print "using: random vectors"

    # filter_h determines padding, hence it depends on largest filter size.
    pad = max(filter_hs) - 1
    max_l = max(len(x_y[0]) for x_y in sents)
    height = max_l + 2 * pad    # padding on both sides
    width = U.shape[1]
    feature_maps = args.hidden
    output_units = len(labels)
    batch_size = 50
    conv_activation = "relu"
    activation = Iden #T.tanh         # Iden
    dropout_rate = args.dropout
    sqr_norm_lim = 9
    shuffle_batch = True
    lr_decay = 0.95
    parameters = (("image shape", height, width),
                  ("filters", args.filters),
                  ("feature maps", feature_maps),
                  ("output units", output_units),
                  ("dropout rate", dropout_rate),
                  ("batch size", batch_size),
                  ("adadelta decay", lr_decay),
                  ("conv_activation", conv_activation),
                  ("activation", activation),
                  ("sqr_norm_lim", sqr_norm_lim),
                  ("shuffle batch", shuffle_batch))
    for param in parameters:
        print "%s: %s" % (param[0], ",".join(str(x) for x in param[1:]))

    cnn = ConvNet(U, height, width,
                  filter_hs=filter_hs,
                  conv_activation=conv_activation,
                  feature_maps=feature_maps,
                  output_units=output_units,
                  batch_size=batch_size,
                  dropout_rates=[dropout_rate],
                  activations=[activation])
    # each item in train is a list of indices for each sentencs plus the id of the label
    train = [sent2indices(words, word_index, max_l, pad) + [y]
             for words,y in sents]
    train_set = np.array(train, dtype="int32")

    # model saver
    def save():
        with open(model, "wb") as mfile:
            cnn.save(mfile)
            pickle.dump((word_index, max_l, pad, labels), mfile)

    cnn.train(train_set, epochs=args.epochs,
              lr_decay=lr_decay,
              shuffle_batch=shuffle_batch, 
              sqr_norm_lim=sqr_norm_lim,
              save=save)
