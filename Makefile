
GOOGLENEWS=/project/piqasso/Collection/WordEmbeddings/GoogleNews-vectors-negative300.bin
TWEETS=../Convolutional/uvectors.300.5.50.w2v 
SENTWEETS=../Convolutional/senti-vectors.100.w2v

CORPUS=semeval16

DATA=data/semeval16-A-train.tsv
TEST=-A-test
CLEAN = 0
TAG=1

# Sentiment specific Word Embeddings
#SWE=-swe

# ----------------------------------------------------------------------
# Scoring
EVAL = ./pwaeval.py -t $(TAG)
SCORER15 = /project/piqasso/Collection/SemEval/2015/task-10/scoring/score-semeval2015-task10-subtaskB.pl
SCORER16 = ./score-semeval2016-task4-subtaskA.pl

# ----------------------------------------------------------------------
# Targets

all: $(CORPUS)$(MODE)$(SWE)-$(FILTERS)

# best configuration according to
# http://arxiv.org/pdf/1510.03820v2.pdf
FILTERS=7,7,7
# MODE=
# L2= 4
DROPOUT = 0.5
EPOCHS = 25

ifeq ($(SWE), -swe)
EMBEDDINGS = $(SENTWEETS)
else
#EMBEDDINGS = $(GOOGLENEWS)
EMBEDDINGS = $(TWEETS)
endif

$(CORPUS)$(MODE)$(SWE)-$(FILTERS): $(DATA) $(EMBEDDINGS)
	THEANO_FLAGS=device=gpu,floatX=float32 ./conv_net_sentence.py \
	-vectors $(EMBEDDINGS) $(MODE) -filters $(FILTERS) -dropout $(DROPOUT) \
	-epochs $(EPOCHS) -train $@ $< > $@.out 2>&1

$(CORPUS)$(TEST)$(MODE)$(SWE)-$(FILTERS).tsv: $(CORPUS)$(MODE)$(SWE)-$(FILTERS) data/$(CORPUS)$(TEST).tsv
	THEANO_FLAGS=device=gpu,floatX=float32 ./conv_net_sentence.py $^ > $@

$(CORPUS)$(TEST)$(MODE)$(SWE)-$(FILTERS).eval: data/$(CORPUS)$(TEST).tsv $(CORPUS)$(TEST)$(MODE)$(SWE)-$(FILTERS).tsv
	$(EVAL) $^ > $@

$(CORPUS)$(TEST)$(MODE)$(SWE)-$(FILTERS).scored: SemEval2016_task4_subtaskA_test_gold.txt $(CORPUS)$(TEST)$(MODE)$(SWE)-$(FILTERS).tsv
	@cut -f1,3 $< > $<.tmp
	f=$(basename $(word 2,$^)); \
	paste $<.tmp $(word 2,$^) | cut -f1,4 > $$f; \
	$(SCORER16) $<.tmp $$f; \
	rm $<.tmp $$f
