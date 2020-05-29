#include "kaldi_recognizer.h"
#include "model.h"
#include "spk_model.h"
#include "feat/wave-reader.h"

#include <string.h>

using namespace kaldi;

int
main(int argc, char **argv)
{
    ParseOptions po("this is spec");
    po.Read(argc, argv);
    std::string wav_rspecifier = po.GetArg(1);

    Model *model = new Model("vosk-model-small-en-us-0.3/");
    KaldiRecognizer *r = new KaldiRecognizer(model, 16000);

    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    for (; !wav_reader.Done(); wav_reader.Next()) {
         std::string key = wav_reader.Key();
         const WaveData &wave_data = wav_reader.Value();
         BaseFloat duration = wave_data.Duration();

         KALDI_LOG << key << ": " << duration;

         const int step = 4000;
         for (int i = 0; i < wave_data.Data().NumCols(); i+= step) {

            Vector<BaseFloat> wave;
            int size = std::min(step, wave_data.Data().NumCols()-i);
            KALDI_LOG << "size" << ": "<< size << " i: " << i;
            wave.Resize(size, kUndefined);
            for (int k = 0; k < size; k++)
                wave(k) = wave_data.Data().Row(0)(i+k);

            bool done = r->AcceptWaveform(wave);
            KALDI_LOG << i << ": "<< done;
            if (done) {
                KALDI_LOG << r->Result();
            }
         }
         KALDI_LOG << r->Result();
    }

    return 0;
}

// vim: ts=8 sts=4 sw=4 et
