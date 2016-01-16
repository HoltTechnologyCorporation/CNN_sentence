## Convolutional Neural Networks for Sentence Classification

Fork from https://github.com/yoonkim/CNN_sentence.git

Code for the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).

Runs the model on Pang and Lee's movie review dataset (MR in the paper).
Please cite the original paper when using the data.

### Requirements
Code is written in Python (2.7) and requires Theano (0.7).

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/


### Data Preprocessing
To process the raw data, run

```
./process_data.py path clean pos_file neg_file data
```

where path points to the word2vec binary file (e.g. `GoogleNews-vectors-negative300.bin` file),
clean is 0/1 depending on whether text should be lower-cased and tokenized,
pos_file, neg_file contain the positive/negative examples from the corpus.
This will create a pickle object called `data` in the same folder, which contains the dataset
in the right format.

Note: This will create the dataset with different fold-assignments than was used in the paper.
You should still be getting a CV score of >81% with CNN-nonstatic model, though.

### Invocation
```
usage: conv_net_sentence.py [-h] [-train] [-static] [-rand] [-filters FILTERS]
                            [-data DATA] [-dropout DROPOUT] [-epochs EPOCHS]
                            model

CNN sentence classifier.

positional arguments:
  model             model file (default mr)

optional arguments:
  -h, --help        show this help message and exit
  -train            train model
  -static           static or nonstatic
  -rand             random vector initializatino
  -filters FILTERS  n[,n]* (default 3,4,5)
  -data DATA        data file (default mr.data)
  -dropout DROPOUT  dropout probability (default 0.5)
  -epochs EPOCHS    training iterations (default 25)
```

### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py -rand
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py -word2vec -static
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py
```

This will run the CNN-rand, CNN-static, and CNN-nonstatic models respectively in the paper.

### Using the GPU
GPU will result in a good 10x to 20x speed-up, so it is highly recommended. 
To use the GPU, simply change `device=cpu` to `device=gpu` (or whichever gpu you are using).
For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

### Example output
CPU output:
```
epoch: 1, training time: 219.72 secs, train perf: 81.79 %, val perf: 79.26 %
epoch: 2, training time: 219.55 secs, train perf: 82.64 %, val perf: 76.84 %
epoch: 3, training time: 219.54 secs, train perf: 92.06 %, val perf: 80.95 %
```
GPU output:
```
epoch: 1, training time: 16.49 secs, train perf: 81.80 %, val perf: 78.32 %
epoch: 2, training time: 16.12 secs, train perf: 82.53 %, val perf: 76.74 %
epoch: 3, training time: 16.16 secs, train perf: 91.87 %, val perf: 81.37 %
```

### Other Implementations
#### TensorFlow
[Denny Britz](http://www.wildml.com) has an implementation of the model in TensorFlow:

https://github.com/dennybritz/cnn-text-classification-tf

He also wrote a [nice tutorial](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow) on it, as well as a general tutorial on [CNNs for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp).

#### Torch
Coming soon.

### Hyperparameters
At the time of my original experiments I did not have access to a GPU so I could not run a lot of different experiments.
Hence the paper is missing a lot of things like ablation studies and variance in performance.

Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.
