#!/usr/bin/env python

"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle as pickle
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
import argparse
from itertools import izip
warnings.filterwarnings("ignore")   

sys.path.append(".")
from conv_net_classes import *

class ConvNet(MLPDropout):
    """
    Adds convolution layers in front of a MLPDropout.
    """


    def __init__(self, U, height, width, filter_hs, conv_non_linear,
                 hidden_units, batch_size, non_static, dropout_rates,
                 activations=[Iden]):
        """
        height = sentence length (padded where necessary)
        width = word vector length (300 for word2vec)
        filter_hs = filter window sizes    
        hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
        """
        rng = np.random.RandomState(3435)
        feature_maps = hidden_units[0]
        self.batch_size = batch_size

        # define model architecture
        self.index = T.lscalar()
        self.x = T.matrix('x')   
        self.y = T.ivector('y')
        self.Words = theano.shared(value=U, name="Words")
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(width)
        # reset Words to 0?
        self.set_zero = theano.function([zero_vec_tensor],
                                        updates=[(self.Words, T.set_subtensor(self.Words[0,:],
                                                                         zero_vec_tensor))],
                                        allow_input_downcast=True)
        # inputs to the ConvNet go to all convolutional filters:
        layer0_input = self.Words[T.cast(self.x.flatten(), dtype="int32")].reshape(
            (self.x.shape[0], 1, self.x.shape[1], self.Words.shape[1]))
        self.conv_layers = []
        # outputs of convolutional filters
        layer1_inputs = []
        image_shape = (batch_size, 1, height, width)
        filter_w = width    
        for filter_h in filter_hs:
            filter_shape = (feature_maps, 1, filter_h, filter_w)
            pool_size = (height-filter_h+1, width-filter_w+1)
            conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                            image_shape=image_shape,
                                            filter_shape=filter_shape,
                                            poolsize=pool_size,
                                            non_linear=conv_non_linear)
            layer1_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        # inputs to the MLP
        layer1_input = T.concatenate(layer1_inputs, 1)
        layer_sizes = [feature_maps*len(filter_hs)] + hidden_units[1:]
        super(ConvNet, self).__init__(rng, input=layer1_input,
                                      layer_sizes=layer_sizes,
                                      activations=activations,
                                      dropout_rates=dropout_rates)

        # add parameters from convolutional layers
        for conv_layer in self.conv_layers:
            self.params += conv_layer.params
        if non_static:
            # if word vectors are allowed to change, add them as model parameters
            self.params += [self.Words]


    def train(self, train_set, test_set, shuffle_batch=True,
              epochs=25, lr_decay=0.95, sqr_norm_lim=9):
        """
        Train a simple conv net
        sqr_norm_lim = s^2 in the paper
        lr_decay = adadelta decay parameter
        """    

        cost = self.negative_log_likelihood(self.y) 
        dropout_cost = self.dropout_negative_log_likelihood(self.y)           
        # adadelta upgrades: dict of variable:delta
        grad_updates = sgd_updates_adadelta(self.params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

        # shuffle dataset and assign to mini batches.
        # if dataset size is not a multiple of batch size, replicate 
        # extra data (at random)
        np.random.seed(3435)
        batch_size = self.batch_size
        if train_set.shape[0] % batch_size > 0:
            extra_data_num = batch_size - train_set.shape[0] % batch_size
            #extra_data = train_set[np.random.choice(train_set.shape[0], extra_data_num)]
            perm_set = np.random.permutation(train_set)   
            extra_data = perm_set[:extra_data_num]
            new_data = np.append(train_set, extra_data, axis=0)
        else:
            new_data = train_set
        shuffled_data = np.random.permutation(new_data) # Attardi
        n_batches = shuffled_data.shape[0]/batch_size

        # divide train set into 90% train, 10% validation sets
        n_train_batches = int(np.round(n_batches*0.9))
        n_val_batches = n_batches - n_train_batches
        train_set = shuffled_data[:n_train_batches*batch_size,:]
        val_set = shuffled_data[n_train_batches*batch_size:,:]     

        train_set_x, train_set_y = shared_dataset(train_set[:,:-1], train_set[:,-1])
        val_set_x, val_set_y = shared_dataset(val_set[:,:-1], val_set[:,-1])

        batch_start = self.index * batch_size
        batch_end = batch_start + batch_size
        # compile Theano functions to get train/val/test errors
        train_model = theano.function([self.index], cost, updates=grad_updates,
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end]},
                                      allow_input_downcast = True)

        val_model = theano.function([self.index], self.errors(self.y),
                                    givens={
                                        self.x: val_set_x[batch_start:batch_end],
                                        self.y: val_set_y[batch_start:batch_end]},
                                    allow_input_downcast=True)

        # errors on train set
        train_error = theano.function([self.index], self.errors(self.y),
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end]},
                                      allow_input_downcast=True)

        test_set_x = test_set[:,:-1]
        test_set_y = test_set[:,-1]
        test_y_pred = self.predict(test_set_x)

        test_error = T.mean(T.neq(test_y_pred, self.y))
        test_model = theano.function([self.x, self.y], test_error, allow_input_downcast=True)

        # start training over mini-batches
        print 'training...'
        best_val_perf = 0
        test_perf = 0       
        for epoch in xrange(epochs):
            start_time = time.time()
            # FIXME: should permute whole set rather than minibatch indexes
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    self.set_zero(self.zero_vec) # CHECKME: Why?
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_model(minibatch_index)  
                    self.set_zero(self.zero_vec)
            train_losses = [train_error(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)                        
            print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (
                epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                test_loss = test_model(test_set_x, test_set_y)        
                test_perf = 1 - test_loss         
        return test_perf


    def predict(self, test_set_x):
        test_size = test_set_x.shape[0]
        height = test_set_x.shape[1]
        layer0_input = self.Words[T.cast(self.x.flatten(), dtype="int32")].reshape(
            (test_size, 1, height, self.Words.shape[1]))
        layer0_outputs = []
        for conv_layer in self.conv_layers:
            layer0_output = conv_layer.predict(layer0_input, test_size)
            layer0_outputs.append(layer0_output.flatten(2))
        layer1_input = T.concatenate(layer0_outputs, 1)
        return super(ConvNet, self).predict(layer1_input)


    def save(self, mfile):
        """
        Save network params to file.
        """
        pickle.dump((self.params, self.layers, self.conv_layers),
                    mfile, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, mfile):
        cnn = cls.__new__(cls)
        cnn.params, cnn.layers, cnn.conv_layers = pickle.load(mfile)
        cnn.Words = cnn.params[-1]
        cnn.index = T.lscalar()
        cnn.x = T.matrix('x')   
        cnn.y = T.ivector('y')
        return cnn


def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32),
                                 borrow=borrow)
        return shared_x, shared_y


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9,
                         word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    :retuns: a dictionary of variable:delta
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),
                                             name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty),
                                           name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    

def get_idx_from_sent(sent, word_index, max_l, pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    Drop words non in word_index. Attardi.
    :param mak_l: max sentence length
    :param pad: pad length
    """
    x = [0] * pad               # left padding
    words = sent.split()
    for word in words:
        if word in word_index: # FIXME: skips unknown words
            x.append(word_index[word])
    while len(x) < max_l + 2 * pad: # right padding
        x.append(0)
    return x


def make_idx_data_cv(revs, word_index, cv, max_l, pad):
    """
    Transforms sentences into a 2-d matrix and splits them into
    train and test according to cv.
    :param cv: cross-validation step
    :param max_l: max sentence length
    :param pad: pad length
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_index, max_l, pad)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train, dtype="int32")
    test = np.array(test, dtype="int32")
    return train, test
  

def read_corpus(filename, word_index, max_l, pad=2, clean_string=False):
    test = []
    with open(filename, "rb") as f:
        for line in f:
            tokens = line.strip().split()
            if clean_string:
                orig_clean = clean_str(" ".join(tokens))
            else:
                orig_clean = " ".join(tokens).lower()
            sent = get_idx_from_sent(orig_clean, word_index, max_l, pad)
            #sent.append(0)      # unknown y
            test.append(sent)
    return np.array(test, dtype="int32")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CNN sentence classifier.")
    
    parser.add_argument('model', type=str, default='mr',
                        help='model file (default %(default)s)')
    parser.add_argument('input', type=str, nargs='?',
                        help='input test file')
    parser.add_argument('-train', help='train model',
                        action='store_true')
    parser.add_argument('-static', help='static or nonstatic',
                        action='store_true')
    parser.add_argument('-rand', help='random vector initializatino',
                        action='store_true')
    parser.add_argument('-filters', type=str, default='3,4,5',
                        help='n[,n]* (default %(default)s)')
    parser.add_argument('-data', type=str, default='mr.data',
                        help='data file (default %(default)s)')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability (default %(default)s)')
    parser.add_argument('-epochs', type=int, default=25,
                        help='training iterations (default %(default)s)')

    args = parser.parse_args()

    if not args.train:
        # test
        with open(args.model) as mfile:
            cnn = ConvNet.load(mfile)
            word_index, max_l, pad = pickle.load(mfile)

        cnn.activations = [Iden] #TODO: save it in the model

        test_set_x = read_corpus(args.input, word_index, max_l, pad)
        test_y_pred = cnn.predict(test_set_x)
        test_model = theano.function([cnn.x], test_y_pred, allow_input_downcast=True)
        results = test_model(test_set_x)
        print results.shape
        sys.exit()

    # training
    print "loading data...",
    revs, W, W2, word_index, vocab = pickle.load(open(args.data,"rb"))

    # revs is a list of entries, where each entry is a dict:
    # {"y": 0/1, "text": , "num_words": , "split": cv fold}
    # W2: random vectors. Attardi
    # vocab: dict of word doc freq
    print "data loaded!"
    filter_hs = [int(x) for x in args.filters.split(',')]
    model = args.model
    if args.static:
        print "model architecture: CNN-static"
        non_static = False
    else:
        print "model architecture: CNN-non-static"
        non_static = True
    if args.rand:
        print "using: random vectors"
        U = W2
    else:
        print "using: word2vec vectors"
        U = W

    # filter_h determines padding, hence it depends on largest filter size.
    pad = max(filter_hs) - 1
    max_l = 56 # DEBUG: max(x["num_words"] for x in revs)
    height = max_l + 2 * pad # padding on both sides

    classes = set(x["y"] for x in revs)
    width = U.shape[1]
    conv_non_linear = "relu"
    hidden_units = [100, len(classes)]
    batch_size = 50
    dropout_rate = args.dropout
    sqr_norm_lim = 9
    shuffle_batch = True
    lr_decay = 0.95
    layer_sizes = [hidden_units[0]*len(filter_hs)] + hidden_units[1:]
    parameters = (("image shape", height, width),
                  ("filters", args.filters),
                  ("hidden units", hidden_units),
                  ("layer sizes", layer_sizes),
                  ("dropout rate", dropout_rate),
                  ("batch size", batch_size),
                  ("adadelta decay", lr_decay),
                  ("conv_non_linear", conv_non_linear),
                  ("non static", non_static),
                  ("sqr_norm_lim", sqr_norm_lim),
                  ("shuffle batch", shuffle_batch))
    print parameters    
    results = []
    # ensure replicability
    np.random.seed(3435)
    # perform 10-fold cross-validation
    for i in range(0, 10):
        # test = [padded(x) for x in revs if x[split] == i]
        # train is rest
        train_set, test_set = make_idx_data_cv(revs, word_index, i, max_l, pad)
        cnn = ConvNet(U, height, width,
                      filter_hs=filter_hs,
                      conv_non_linear=conv_non_linear,
                      hidden_units=hidden_units,
                      batch_size=batch_size,
                      non_static=non_static,
                      dropout_rates=[dropout_rate])
        perf = cnn.train(train_set, test_set,
                         lr_decay=lr_decay,
                         shuffle_batch=shuffle_batch, 
                         epochs=args.epochs,
                         sqr_norm_lim=sqr_norm_lim)
        print "cv: %d, perf: %f" % (i, perf)
        results.append(perf)  
    print "Avg. accuracy: %.4f" % np.mean(results)
    # save model
    with open(model, "wb") as mfile:
        cnn.save(mfile)
        pickle.dump((word_index, max_l, pad), mfile)
