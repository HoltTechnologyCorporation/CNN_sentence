
GOOGLENEWS=/project/piqasso/Collection/WordEmbeddings/GoogleNews-vectors-negative300.bin

#CORPUS=semeval15
#CORPUS=semeval
CORPUS=mr

ifeq ($(CORPUS), semeval15)
DATA=data/semeval15-B-train.tsv
CLEAN = 0
else
ifeq ($(CORPUS), semeval)	# no neutral tweets
DATA=data/semeval.tsv
CLEAN = 0
else
DATA=data/rt-polarity.tsv
CLEAN = 1
endif
endif

all: $(CORPUS)$(MODE)-$(FILTERS)

data/semeval.tsv: data/semeval.pos data/semeval.neg
	awk '{ printf "0\t0\tpositive\t%s\n", $$0;}' data/semeval.pos > $@
	awk '{ printf "0\t0\tnegative\t%s\n", $$0;}' data/semeval.neg >> $@

data/rt-polarity.tsv: data/rt-polarity.pos data/rt-polarity.neg
	awk '{ printf "0\t0\tpositive\t%s\n", $$0;}' data/rt-polarity.pos > $@
	awk '{ printf "0\t0\tnegative\t%s\n", $$0;}' data/rt-polarity.neg >> $@

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
EPOCHS = 25

$(CORPUS)$(MODE)-$(FILTERS): $(CORPUS).data
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,openmp=True,floatX=float32 ./conv_net_sentence.py -data $< $(MODE) -filters $(FILTERS) -dropout $(DROPOUT) \
	-epochs $(EPOCHS) -train $@ > $@.out 2>&1
