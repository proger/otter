all: spec

OBJS= kaldi_recognizer.o main.o model.o spk_model.o

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
