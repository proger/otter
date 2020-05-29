#include <sys/stat.h>

#include <fst/fst.h>
#include <fst/register.h>
#include <fst/matcher-fst.h>
#include <fst/extensions/ngram/ngram-fst.h>

#include "base/kaldi-common.h"
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "fstext/fstext-utils.h"
#include "fstext/fstext-utils.h"
#include "lat/lattice-functions.h"
#include "lat/sausages.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice.h"
#include "lm/const-arpa-lm.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-timing.h"
#include "online2/onlinebin-util.h"
#include "rnnlm/rnnlm-utils.h"
#include "util/parse-options.h"

#include <string.h>

using namespace kaldi;
using namespace kaldi::nnet3;
using namespace fst;

namespace fst {

static FstRegisterer<StdOLabelLookAheadFst> OLabelLookAheadFst_StdArc_registerer;
static FstRegisterer<NGramFst<StdArc>> NGramFst_StdArc_registerer;

}

struct Model {
    OnlineEndpointConfig endpoint_config_;
    LatticeFasterDecoderConfig nnet3_decoding_config_;
    NnetSimpleLoopedComputationOptions decodable_opts_;

    OnlineNnet2FeaturePipelineInfo feature_info_;
    OnlineNnet2FeaturePipelineConfig feature_config_;

    DecodableNnetSimpleLoopedInfo *decodable_info_;
    TransitionModel *trans_model_;
    AmNnetSimple *nnet_;
    const fst::SymbolTable *word_syms_;
    WordBoundaryInfo *winfo_;
    vector<int32> disambig_;

    fst::Fst<fst::StdArc> *hclg_fst_;
    fst::Fst<fst::StdArc> *hcl_fst_;
    fst::Fst<fst::StdArc> *g_fst_;
    
    void Configure(string model_path_str_) {
        feature_info_.feature_type = "mfcc"; 
        ReadConfigFromFile(model_path_str_ + "/mfcc.conf", &feature_info_.mfcc_opts);
        feature_info_.mfcc_opts.frame_opts.allow_downsample = true; // It is safe to downsample

        feature_info_.silence_weighting_config.silence_weight = 1e-3;
        feature_info_.silence_weighting_config.silence_phones_str = endpoint_config_.silence_phones;

        OnlineIvectorExtractionConfig ivector_extraction_opts;
        ivector_extraction_opts.splice_config_rxfilename = model_path_str_ + "/ivector/splice.conf";
        ivector_extraction_opts.cmvn_config_rxfilename = model_path_str_ + "/ivector/online_cmvn.conf";
        ivector_extraction_opts.lda_mat_rxfilename = model_path_str_ + "/ivector/final.mat";
        ivector_extraction_opts.global_cmvn_stats_rxfilename = model_path_str_ + "/ivector/global_cmvn.stats";
        ivector_extraction_opts.diag_ubm_rxfilename = model_path_str_ + "/ivector/final.dubm";
        ivector_extraction_opts.ivector_extractor_rxfilename = model_path_str_ + "/ivector/final.ie";
        feature_info_.use_ivectors = true;
        feature_info_.ivector_extractor_info.Init(ivector_extraction_opts);
    }

    void Read(string model_path_str_) {
        struct stat buffer; 

        trans_model_ = new kaldi::TransitionModel();
        nnet_ = new kaldi::nnet3::AmNnetSimple();
        {
            bool binary;
            kaldi::Input ki(model_path_str_ + "/final.mdl", &binary);
            trans_model_->Read(ki.Stream(), binary);
            nnet_->Read(ki.Stream(), binary);
            SetBatchnormTestMode(true, &(nnet_->GetNnet()));
            SetDropoutTestMode(true, &(nnet_->GetNnet()));
            nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(nnet_->GetNnet()));
        }
        decodable_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts_,
                                                                   nnet_);


        if (stat((model_path_str_ + "/HCLG.fst").c_str(), &buffer) == 0) {
            KALDI_LOG << "HCLG";
            hclg_fst_ = fst::ReadFstKaldiGeneric(model_path_str_ + "/HCLG.fst");
            hcl_fst_ = NULL;
            g_fst_ = NULL;
        } else {
            KALDI_LOG << "HCLr + Gr";
            hclg_fst_ = NULL;
            hcl_fst_ = fst::StdFst::Read(model_path_str_ + "/HCLr.fst");
            g_fst_ = fst::StdFst::Read(model_path_str_ + "/Gr.fst");
            ReadIntegerVectorSimple(model_path_str_ + "/disambig_tid.int", &disambig_);
        }

#if 0
        g_fst_ = new StdVectorFst();
        g_fst_->AddState();
        g_fst_->SetStart(0);
        g_fst_->AddState();
        g_fst_->SetFinal(1, fst::TropicalWeight::One());
        g_fst_->AddArc(1, StdArc(0, 0, fst::TropicalWeight::One(), 0));
        g_fst_->AddState();

        int32 w_unk = model_->word_syms_->Find("[unk]");
        g_fst_->AddArc(0, StdArc(w_unk, w_unk, 1, 1));

        int32 w_max = model_->word_syms_->Find("max");
        g_fst_->AddArc(0, StdArc(w_max, w_max, 5, 2));
        int32 w_decoding = model_->word_syms_->Find("decoding");
        //g_fst_->AddArc(2, StdArc(0, 0, 1, 0));
        g_fst_->AddArc(2, StdArc(w_decoding, w_decoding, 2, 1));

        ArcSort(g_fst_, ILabelCompare<StdArc>());

        decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *g_fst_, model_->disambig_);
#endif

        word_syms_ = NULL;
        if (hclg_fst_ && hclg_fst_->OutputSymbols()) {
            word_syms_ = hclg_fst_->OutputSymbols();
        } else if (g_fst_ && g_fst_->OutputSymbols()) {
            word_syms_ = g_fst_->OutputSymbols();
        }
        //if (!word_syms_) {
        //    KALDI_LOG << "Loading words from " << word_syms_rxfilename_;
        //    if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_)))
        //        KALDI_ERR << "Could not read symbol table from file "
        //                  << word_syms_rxfilename_;
        //}
        KALDI_ASSERT(word_syms_);

        string winfo_rxfilename_ = model_path_str_ + "/word_boundary.int";
        if (stat(winfo_rxfilename_.c_str(), &buffer) == 0) {
            KALDI_LOG << "Loading winfo " << winfo_rxfilename_;
            kaldi::WordBoundaryInfoNewOpts opts;
            winfo_ = new kaldi::WordBoundaryInfo(opts, winfo_rxfilename_);
        } else {
            winfo_ = NULL;
        }
    }

    void Register(OptionsItf *po) {
        feature_config_.Register(po);
        nnet3_decoding_config_.Register(po);
        endpoint_config_.Register(po);
        decodable_opts_.Register(po);
    }
};

int
main(int argc, char **argv)
{
    auto model_ = new Model();

    ParseOptions po("spec <model-dir> <wav-rspecifier>");

    model_->Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
        po.PrintUsage();
        exit(1);
    }
    std::string model_path = po.GetArg(1);
    std::string wav_rspecifier = po.GetArg(2);

    model_->Configure(model_path);
    model_->Read(model_path);

    auto feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline(model_->feature_info_);
    auto silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    auto frame_offset_ = 0;

    SingleUtteranceNnet3Decoder *decoder_;
    if (model_->hcl_fst_ && model_->g_fst_) {
        auto decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *model_->g_fst_, model_->disambig_);
        decoder_ = new SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
                *model_->trans_model_,
                *model_->decodable_info_,
                *decode_fst_,
                feature_pipeline_);
    } else {
        decoder_ = new SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
                *model_->trans_model_,
                *model_->decodable_info_,
                *model_->hclg_fst_,
                feature_pipeline_);
    }

    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    for (; !wav_reader.Done(); wav_reader.Next()) {
        std::string key = wav_reader.Key();
        const WaveData &wave_data = wav_reader.Value();
        BaseFloat duration = wave_data.Duration();

        KALDI_LOG << key << ": " << duration;

        const int step = 4000;
        for (int i = 0; i < wave_data.Data().NumCols(); i+= step) {

            int size = std::min(step, wave_data.Data().NumCols()-i);

            auto wdata = wave_data.Data().Row(0).Range(i, size);
            feature_pipeline_->AcceptWaveform(16000, wdata);

            if (silence_weighting_->Active() && feature_pipeline_->NumFramesReady() > 0 &&
                    feature_pipeline_->IvectorFeature() != NULL) {
                vector<pair<int32, BaseFloat> > delta_weights;
                silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
                silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                        frame_offset_ * 3,
                        &delta_weights);
                feature_pipeline_->UpdateFrameWeights(delta_weights);
            }

            decoder_->AdvanceDecoding();

            auto done = decoder_->EndpointDetected(model_->endpoint_config_);
            //KALDI_LOG << i << ": "<< done << " size: " << size;
        }
        decoder_->FinalizeDecoding();
        KALDI_LOG << "NumFramesDecoded: " << decoder_->NumFramesDecoded();

        delete silence_weighting_;
        silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

        // kaldi::Lattice lat;
        // decoder_->GetBestPath(false, &lat);
        // vector<kaldi::int32> alignment, words;
        // LatticeWeight weight;
        // GetLinearSymbolSequence(lat, &alignment, &words, &weight);

        kaldi::CompactLattice clat;
        decoder_->GetLattice(true, &clat);

        fst::ScaleLattice(fst::LatticeScale(9.0, 10.0), &clat);
        CompactLattice aligned_lat;
        if (model_->winfo_) {
            WordAlignLattice(clat, *model_->trans_model_, *model_->winfo_, 0, &aligned_lat);
        } else {
            aligned_lat = clat;
        }

        MinimumBayesRisk mbr(aligned_lat);
        const vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
        const vector<int32> &words = mbr.GetOneBest();
        const vector<pair<BaseFloat, BaseFloat> > &times = mbr.GetOneBestTimes();

        for (int i = 0; i < words.size(); i++) {
            KALDI_LOG << model_->word_syms_->Find(words[i])
                << " " << (frame_offset_ + times[i].first) * 0.03
                << " " << (frame_offset_ + times[i].second) * 0.03
                << " " << conf[i];
        }

        frame_offset_ += decoder_->NumFramesDecoded();
        decoder_->InitDecoding(frame_offset_);
    }

    return 0;
}

// vim: ts=8 sts=4 sw=4 et
