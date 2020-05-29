#include "vosk_api.h"
#include "kaldi_recognizer.h"
#include "model.h"
#include "spk_model.h"

#include <string.h>

using namespace kaldi;

int
main(int argc, char **argv)
{

    Model *model = new Model("vosk-model-small-en-us-0.3/");
    KaldiRecognizer *r = new KaldiRecognizer(model, 16000);
    return 0;
}

// vim: ts=8 sts=4 sw=4 et
