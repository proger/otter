#!/usr/bin/env bash
# Copyright 2010-2012 Microsoft Corporation
#           2012-2013 Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script creates a fully expanded decoding graph (HCLG) that represents
# all the language-model, pronunciation dictionary (lexicon), context-dependency,
# and HMM structure in our model.  The output is a Finite State Transducer
# that has word-ids on the output, and pdf-ids on the input (these are indexes
# that resolve to Gaussian Mixture Models).
# See
#  http://kaldi-asr.org/doc/graph_recipe_test.html
# (this is compiled from this repository using Doxygen,
# the source for this part is in src/doc/graph_recipe_test.dox)

set -e
set -o pipefail

tscale=1.0
loopscale=1.0

remove_oov=false

for x in `seq 4`; do
  [ "$1" == "--mono" -o "$1" == "--left-biphone" -o "$1" == "--quinphone" ] && shift && \
    echo "WARNING: the --mono, --left-biphone and --quinphone options are now deprecated and ignored."
  [ "$1" == "--remove-oov" ] && remove_oov=true && shift;
  [ "$1" == "--transition-scale" ] && tscale=$2 && shift 2;
  [ "$1" == "--self-loop-scale" ] && loopscale=$2 && shift 2;
done

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <lang-dir> <model-dir> <graphdir>"
   echo "e.g.: $0 data/lang_test exp/tri1/ exp/tri1/graph"
   echo " Options:"
   echo " --remove-oov       #  If true, any paths containing the OOV symbol (obtained from oov.int"
   echo "                    #  in the lang directory) are removed from the G.fst during compilation."
   echo " --transition-scale #  Scaling factor on transition probabilities."
   echo " --self-loop-scale  #  Please see: http://kaldi-asr.org/doc/hmm.html#hmm_scale."
   echo "Note: the --mono, --left-biphone and --quinphone options are now deprecated"
   echo "and will be ignored."
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

lang=$1
tree=$2/tree
model=$2/final.mdl
dir=$3

mkdir -p $dir

required="$lang/phones.txt $lang/phones/silence.csl $lang/phones/disambig.int $model $tree"
for f in $required; do
  [ ! -f $f ] && echo "mkphonegraph.sh: expected $f to exist" && exit 1;
done

N=$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
P=$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }

[[ -f $2/frame_subsampling_factor && "$loopscale" == "0.1" ]] && \
  echo "$0: WARNING: chain models need '--self-loop-scale 1.0'";


# [nix-shell:/tank/asr]$ /home/proger/soundflow/kaldi/s6/utils/mkphonegraph.sh --self-loop-scale 1.0 data/phonelang exp/chain/tdnn_7b ph

awk 'NR>1 && !/^#/ {print $1, "1", $1}' $lang/phones.txt > $dir/lexicon.txt
awk 'NR {print $1, NR}' $lang/phones.txt > $dir/lexicon1.txt

# https://kaldi-asr.org/doc/graph_recipe_test.html

#
# G
#

w=-2.25
echo '\data\' > $dir/lexicon.arpa
echo 'ngram 1='$(wc -l $dir/lexicon.txt | awk '{print 2+$1}') >> $dir/lexicon.arpa
echo '' >> $dir/lexicon.arpa
echo '\1-grams:' >> $dir/lexicon.arpa
#printf "%f\t%s\n" -0.1 '<unk>' >> $dir/lexicon.arpa
printf "%f\t%s\n" $w '<s>' >> $dir/lexicon.arpa
printf "%f\t%s\n" $w '</s>' >> $dir/lexicon.arpa
awk -vOFS='\t' -v w=$w '{print w, $1}' $dir/lexicon.txt >> $dir/lexicon.arpa
echo '' >> $dir/lexicon.arpa
echo '\end\' >> $dir/lexicon.arpa

arpa2fst --disambig-symbol=#0 --read-symbol-table=$dir/lexicon1.txt $dir/lexicon.arpa $dir/G.fst

fstisstochastic $dir/G.fst

#
# L
#

phone_disambig_symbol=`grep \#0 $lang/phones.txt | awk '{print $2}'`

s5=$KALDI_ROOT/egs/wsj/s5
$s5/utils/lang/make_lexicon_fst.py $dir/lexicon.txt  \
    | fstcompile --isymbols=$lang/phones.txt --osymbols=$lang/phones.txt \
        --keep_isymbols=false --keep_osymbols=false \
    | fstaddselfloops "echo $phone_disambig_symbol |" "echo $phone_disambig_symbol |" \
    | fstarcsort --sort_type=olabel > $dir/L_disambig.fst

#
# LG
#

fsttablecompose $dir/L_disambig.fst $dir/G.fst | \
    fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstpushspecial | \
     fstarcsort --sort_type=ilabel > $dir/LG.fst

#
# C
#

grep '#' $lang/phones.txt | awk '{print $2}' > $dir/disambig_phones.list
subseq_sym=`tail -1 $dir/disambig_phones.list | awk '{print $1+1;}'`
echo "subseq_sym: $subseq_sym"

fstmakecontextfst \
    --read-disambig-syms=$dir/disambig_phones.list \
    --write-disambig-syms=$dir/disambig_ilabels \
    <(grep -v '#' $lang/phones.txt) $subseq_sym $dir/ilabels | \
    fstarcsort --sort_type=olabel > $dir/C.fst

fstmakecontextsyms $lang/phones.txt $dir/ilabels > $dir/context_syms.txt

cat $lang/phones.txt > $dir/C_output_syms.txt
printf "SUBSEQ\t%s\n" "$subseq_sym" >> $dir/C_output_syms.txt

fstrandgen --select=log_prob $dir/C.fst | \
   fstprint --isymbols=$dir/context_syms.txt -osymbols=$dir/C_output_syms.txt -

fstdeterminizestar --use-log=true $dir/C.fst >/dev/null

# The transducer must be functional. The weights must be (weakly) left divisible (valid for TropicalWeight and LogWeight for instance) and zero-sum-free.
# XXX: something's wrong with C?

#
# H (needs ilabels from C)
#

make-h-transducer --disambig-syms-out=$dir/disambig_tid.int \
    --transition-scale=$tscale $dir/ilabels $tree $model \
    > $dir/Ha.fst

#
# CLG manually
#

fstaddsubsequentialloop $subseq_sym $dir/LG.fst | \
 fsttablecompose $dir/C.fst - > $dir/CLG.fst

# test
fstrandgen --select=log_prob $dir/CLG.fst | \
   fstprint --isymbols=$dir/context_syms.txt --osymbols=$lang/phones.txt -

#
# CLG dynamically
#

#fstcomposecontext --context-size=$N --central-position=$P \
#    --read-disambig-syms=$lang/phones/disambig.int \
#    --write-disambig-syms=$lang/tmp/disambig_ilabels_${N}_${P}.int \
#    $dir/ilabels $dir/LG.fst | \
#    fstarcsort --sort_type=ilabel > $dir/CLG.fst


# works out from the decision tree and the HMM topology information, which subsets of context-dependent phones would correspond to the same compiled graph and can therefore be merged (we pick an arbitrary element of each subset and convert all context windows to that context window).
#make-ilabel-transducer --write-disambig-syms=$dir/disambig_ilabels_remapped.list \
#  $dir/ilabels $tree $model $dir/ilabels.remapped > $dir/ilabel_map.fst
#
#fstcompose $dir/ilabel_map.fst $dir/CLG.fst  | \
#   fstdeterminizestar --use-log=true | \
#   fstminimizeencoded > $dir/CLG2.fst

#
# HCLG
#

fsttablecompose $dir/Ha.fst $dir/CLG.fst > $dir/HCLGpre.fst

fstdeterminizestar --use-log=true $dir/HCLGpre.fst $dir/HCLGadet.fst

fstrmsymbols $dir/disambig_tid.int $dir/HCLGadet.fst > $dir/HCLGanodisambig.fst

< $dir/HCLGanodisambig.fst fstrmepslocal  | fstminimizeencoded > $dir/HCLGa.fst

add-self-loops --self-loop-scale=$loopscale \
    --reorder=true $model < $dir/HCLGa.fst > $dir/HCLG.fst




exit 0 




fstcompose $dir/Ha.fst $dir/C.fst $dir/HCbig.fst
#| fstdeterminizestar --use-log=true \
#| fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
# fstminimizeencoded | add-self-loops --self-loop-scale=$loopscale --reorder=true $model > $dir/HC.fst #| fstconvert --fst_type=const > $dir/HC.fst

