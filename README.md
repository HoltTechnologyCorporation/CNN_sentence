## Convolutional Neural Networks for Sentence Classification

Fork by Giuseppe Attardi from https://github.com/yoonkim/CNN_sentence.git.
Differences with the original:
- multiclass classification
- model saving and loading for classifying files, e.g. tweets
- configurable parameters
- integrated program, without a separate preprocessing stage
- input in a single file, annotated in the format used in the SemEval Twitter Sentiment Analysis tasks.

The original code by Kim Yoon implements the technique described in the paper:
[Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).
Please cite both this page and the original paper when using the program.

### Requirements
Code is written in Python (2.7) and requires Theano (0.7).

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/


### Invocation
```
usage: conv_net_sentence.py [-h] [-train] [-static] [-vectors VECTORS] [-filters FILTERS]
                            [-clean] [-dropout DROPOUT] [-epochs EPOCHS]
                            [-tagField TAGFIELD] [-textField TEXTFIELD]
                            model input

CNN sentence classifier.

positional arguments:
  model                 model file (default mr)
  input                 train/test file in SemEval twitter format

optional arguments:
  -h, --help            show this help message and exit
  -train                train model
  -static               static or nonstatic
  -clean                tokenize text
  -filters FILTERS      n[,n]* (default 3,4,5)
  -vectors VECTORS      word2vec embeddings file (random values if missing)
  -dropout DROPOUT      dropout probability (default 0.5)
  -epochs EPOCHS        training iterations (default 25)
  -tagField TAGFIELD    label field in files (default 1)
  -textField TEXTFIELD  text field in files (default 2)
```

VECTORS is the word2vec binary file (e.g. `GoogleNews-vectors-negative300.bin` file),
clean, if present, text is lower-cased and tokenized,

input contain sentences from the corpus in SemEval format, i.e. one sentence
per line, tab separated values:

ID	UID	text

Note: This will create the dataset with different fold-assignments than was used in the paper.
You should still be getting a CV score of >81% with CNN-nonstatic model, though.

### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py mr.rand mr.tsv
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py -vectors w2v_file -static mr.w2v-static mr.tsv
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 ./conv_net_sentence.py -vectors w2v_file mr.w2v mr.tsv
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

### Hyperparameters
At the time of my original experiments I did not have access to a GPU so I could not run a lot of different experiments.
Hence the paper is missing a lot of things like ablation studies and variance in performance.

Ye Zhang has written a [very nice paper](http://arxiv.org/abs/1510.03820) doing an extensive analysis of model variants (e.g. filter widths, k-max pooling, word2vec vs Glove, etc.) and their effect on performance.

### SemEval-2015 Task 10: Sentiment Analysis in Twitter

Experiments using the data from SemEval-15, using option --filters 7,7,7, achieves an accuracy
of 67.28%, better than the top official score (64.84%), as reported in:
Sara Rosenthal et al. [SemEval-2015 Task 10: Sentiment Analysis in Twitter](http://alt.qcri.org/semeval2015/cdrom/pdf/SemEval078.pdf).

