
GOOGLENEWS=/project/piqasso/Collection/WordEmbeddings/GoogleNews-vectors-negative300.bin

#CORPUS=semeval
CORPUS=mr

ifeq ($(CORPUS), semeval)
DATA=data/semeval.pos data/semeval.neg
CLEAN = 0
else
DATA=rt-polarity.pos rt-polarity.neg
CLEAN = 1
endif

all: $(CORPUS)$(MODE)-$(FILTERS)

$(CORPUS).data: $(GOOGLENEWS) $(DATA)
	./process_data.py $^ $(CLEAN) $@

# original settings
# FILTERS=3,4,5
# MODE=-static

# best configuration according to
# http://arxiv.org/pdf/1510.03820v2.pdf
FILTERS=7,7,7
# MODE=
# L2= 4
DROPOUT = 0.5

$(CORPUS)$(MODE)-$(FILTERS): $(CORPUS).data
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,openmp=True,floatX=float32 conv_net_sentence.py -data $< $(MODE) -filters $(FILTERS) -dropout $(DROPOUT) -train $@ > $@.out 2>&1
