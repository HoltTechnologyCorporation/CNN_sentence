
GOOGLENEWS=/project/piqasso/Collection/WordEmbeddings/GoogleNews-vectors-negative300.bin

CORPUS=semeval16
#CORPUS=semeval15
#CORPUS=semeval
#CORPUS=mr

ifeq ($(CORPUS), semeval16)
DATA=data/semeval16-A-train.tsv
TEST=-A-test
CLEAN = 0
TAG=1
else ifeq ($(CORPUS), semeval15)
DATA=data/semeval15-B-train.tsv
TEST=-B-test
CLEAN = 0
else ifeq ($(CORPUS), semeval)	# no neutral tweets
DATA=data/semeval.tsv
CLEAN = 0
TAG=2
else
DATA=data/rt-polarity.tsv
CLEAN = 1
TAG=2
endif

EVAL = /project/piqasso/QA/Tanl/src/tag/pwaeval.py -t $(TAG)
SCORER15 = /project/piqasso/Collection/SemEval/2015/task-10/scoring/score-semeval2015-task10-subtaskB.pl
SCORER16 = /project/piqasso/Collection/SemEval/2016/task-4/SemEval2016-task4-scorers-v2.2/score-semeval2016-task4-subtaskA.pl

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

$(CORPUS)$(MODE)-$(FILTERS): $(DATA) $(GOOGLENEWS)
	THEANO_FLAGS=mode=FAST_RUN,device=cpu,openmp=True,floatX=float32 ./conv_net_sentence.py \
	-vectors $(GOOGLENEWS) $(MODE) -filters $(FILTERS) -dropout $(DROPOUT) \
	-epochs $(EPOCHS) -train $@ $< > $@.out 2>&1

$(CORPUS)$(TEST)$(MODE)-$(FILTERS).tsv: $(CORPUS)$(MODE)-$(FILTERS) data/$(CORPUS)$(TEST).tsv
	./conv_net_sentence.py $^ > $@

$(CORPUS)$(TEST)$(MODE)-$(FILTERS).eval: data/$(CORPUS)$(TEST).tsv $(CORPUS)$(TEST)$(MODE)-$(FILTERS).tsv
	$(EVAL) $^ > $@

ifeq ($(CORPUS), semeval15)
$(CORPUS)$(TEST)$(MODE)-$(FILTERS).scored: $(CORPUS)$(TEST)$(MODE)-$(FILTERS).tsv SemEval2015-task10-test-B-gold.txt
	awk -F\\t '{printf "NA\t%s\t%s\n", $$2, $$3}' $< > $<.tmp
	$(SCORER15) $<.tmp
	mv $<.tmp.scored  $@
	rm $<.tmp
else
# 2016 scorer
$(CORPUS)$(TEST)$(MODE)-$(FILTERS).scored: SemEval2016_task4_subtaskA_test_gold.txt $(CORPUS)$(TEST)$(MODE)-$(FILTERS).tsv
	@cut -f1,3 $< > $<.tmp
	$(SCORER16) $<.tmp $(word 2,$^) $@p
	@rm $<.tmp
endif
