all: spec

OBJS= main.o

KALDI_ROOT= $(HOME)/vox/src/kaldi/src

LIBS= -lkaldi-online2 -lkaldi-decoder -lkaldi-ivector -lkaldi-gmm -lkaldi-nnet3 -lkaldi-tree -lkaldi-lat
LIBS+= -lkaldi-lm -lkaldi-hmm -lkaldi-transform -lkaldi-matrix -lkaldi-cudamatrix -lkaldi-fstext -lkaldi-util -lkaldi-base
LIBS+= -lkaldi-feat
LIBS+= -lfst -lfstngram

LDFLAGS= -L$(KALDI_ROOT)/src/lib
LDFLAGS+= -L$(KALDI_ROOT)/tools/openfst/lib
LDFLAGS+= -Wl,-rpath,$(KALDI_ROOT)/src/lib

CXXFLAGS=
CXXFLAGS+= -I$(KALDI_ROOT)/src -I$(KALDI_ROOT)/tools/openfst/include
CXXFLAGS+= -std=c++11

spec: $(OBJS)
	c++ -o $@ $(LDFLAGS) $(LIBS) $(OBJS)

%.o: %.cc
	c++ $(CXXFLAGS) -o $@ -c $<

.PHONY: test
test:
	./spec --min-active=200 --max-active=300 --acoustic-scale=1.0 --frame-subsampling-factor=3 --endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10 model scp:wav.scp
