# Quantitative Evaluation

```
python -m tools.evaluate --model_id openai/whisper-tiny.en --dataset librispeech --split test.clean
```

Tried datasets: 
```
# english short form
hf-audio/esb-datasets-test-only-sorted ami test
hf-audio/esb-datasets-test-only-sorted common_voice test
hf-audio/esb-datasets-test-only-sorted earnings22 test
hf-audio/esb-datasets-test-only-sorted gigaspeech test
hf-audio/esb-datasets-test-only-sorted librispeech test.clean
hf-audio/esb-datasets-test-only-sorted librispeech test.other
hf-audio/esb-datasets-test-only-sorted spgispeech test
hf-audio/esb-datasets-test-only-sorted tedlium test
hf-audio/esb-datasets-test-only-sorted voxpopuli test

TwinkStart/peoples_speech default test
TwinkStart/audio-MNIST default test
TwinkStart/librispeech default dev_clean
TwinkStart/librispeech default dev_other
TwinkStart/librispeech default test_clean
TwinkStart/librispeech default test_other
TwinkStart/tedlium default test
TwinkStart/tedlium release1 test
TwinkStart/tedlium release2 test
TwinkStart/tedlium release3 test

adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en default train
adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en default test

TwinkStart/llama-questions default test
TwinkStart/speech-chatbot-alpaca-eval default test
TwinkStart/speech-web-questions default test
TwinkStart/speech-triavia-qa default test
TwinkStart/air-chat default test



TwinkStart/AISHELL-1 default test
# this one fails in the hpc server
TwinkStart/kespeech default test
TwinkStart/WenetSpeech default test_meeting
TwinkStart/WenetSpeech default test_net

TwinkStart/speech-CMMLU default train
TwinkStart/speech-HSK default hsk1
TwinkStart/speech-HSK default hsk2
TwinkStart/speech-HSK default hsk3
TwinkStart/speech-HSK default hsk4
TwinkStart/speech-HSK default hsk5
TwinkStart/speech-HSK default hsk6


JacobLinCool/common_voice_19_0_zh-TW default validated_without_test
JacobLinCool/common_voice_19_0_zh-TW default test

adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw default train
adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw default test

adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien default train
adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien default test
adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka default train
adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka default test

TwinkStart/CommonVoice_15 default yue
TwinkStart/CommonVoice_15 default zh



nithinraok/asr-leaderboard-datasets fleurs_bg test
nithinraok/asr-leaderboard-datasets fleurs_cs test
nithinraok/asr-leaderboard-datasets fleurs_da test
nithinraok/asr-leaderboard-datasets fleurs_de test
nithinraok/asr-leaderboard-datasets fleurs_el test
nithinraok/asr-leaderboard-datasets fleurs_en test
nithinraok/asr-leaderboard-datasets fleurs_es test
nithinraok/asr-leaderboard-datasets fleurs_et test
nithinraok/asr-leaderboard-datasets fleurs_fi test
nithinraok/asr-leaderboard-datasets fleurs_fr test
nithinraok/asr-leaderboard-datasets fleurs_hr test
nithinraok/asr-leaderboard-datasets fleurs_hu test
nithinraok/asr-leaderboard-datasets fleurs_it test
nithinraok/asr-leaderboard-datasets fleurs_lt test
nithinraok/asr-leaderboard-datasets fleurs_lv test
nithinraok/asr-leaderboard-datasets fleurs_mt test
nithinraok/asr-leaderboard-datasets fleurs_nl test
nithinraok/asr-leaderboard-datasets fleurs_pl test
nithinraok/asr-leaderboard-datasets fleurs_pt test
nithinraok/asr-leaderboard-datasets fleurs_ro test
nithinraok/asr-leaderboard-datasets fleurs_ru test
nithinraok/asr-leaderboard-datasets fleurs_sk test
nithinraok/asr-leaderboard-datasets fleurs_sl test
nithinraok/asr-leaderboard-datasets fleurs_sv test
nithinraok/asr-leaderboard-datasets fleurs_uk test

nithinraok/asr-leaderboard-datasets mcv_de test
nithinraok/asr-leaderboard-datasets mcv_en test
nithinraok/asr-leaderboard-datasets mcv_es test
nithinraok/asr-leaderboard-datasets mcv_et test
nithinraok/asr-leaderboard-datasets mcv_fr test
nithinraok/asr-leaderboard-datasets mcv_it test
nithinraok/asr-leaderboard-datasets mcv_lv test
nithinraok/asr-leaderboard-datasets mcv_nl test
nithinraok/asr-leaderboard-datasets mcv_pt test
nithinraok/asr-leaderboard-datasets mcv_ru test
nithinraok/asr-leaderboard-datasets mcv_sl test
nithinraok/asr-leaderboard-datasets mcv_sv test
nithinraok/asr-leaderboard-datasets mcv_uk test

nithinraok/asr-leaderboard-datasets mls_es test
nithinraok/asr-leaderboard-datasets mls_fr test
nithinraok/asr-leaderboard-datasets mls_it test
nithinraok/asr-leaderboard-datasets mls_nl test
nithinraok/asr-leaderboard-datasets mls_pl test
nithinraok/asr-leaderboard-datasets mls_pt test

TwinkStart/CommonVoice_15 default en
TwinkStart/CommonVoice_15 default fr

TwinkStart/facebook_multilingual_librispeech default mls_dutch
TwinkStart/facebook_multilingual_librispeech default mls_french
TwinkStart/facebook_multilingual_librispeech default mls_german
TwinkStart/facebook_multilingual_librispeech default mls_italian
TwinkStart/facebook_multilingual_librispeech default mls_polish
TwinkStart/facebook_multilingual_librispeech default mls_portuguese
TwinkStart/facebook_multilingual_librispeech default mls_spanish


espnet/floras monolingual train
espnet/floras monolingual dev
espnet/floras monolingual test
espnet/floras multilingual dev
espnet/floras multilingual test

hf-audio/asr-leaderboard-longform earnings21 test
hf-audio/asr-leaderboard-longform earnings22 test
hf-audio/asr-leaderboard-longform tedlium test

distil-whisper/meanwhile default test
distil-whisper/rev16 full test
distil-whisper/rev16 whisper_subset test
distil_whisper/tedlium-long-form default validation


OmniAICreator/ASMR-Archive-Processed default train

# not suitable for ASR
TwinkStart/MMAU default v05.15.25

# needs additional preprocessing, also not compatible with streaming
speechcolab/gigaspeech2 default train

```

Tried models:
```
openai/whisper-tiny.en
openai/whisper-small.en
openai/whisper-base.en
openai/whisper-medium.en
openai/whisper-large
openai/whisper-large-v2
openai/whisper-large-v3
distil-whisper/distil-medium.en
distil-whisper/distil-large-v2
distil-whisper/distil-large-v3
mistralai/Voxtral-Mini-3B-2507
mistralai/Voxtral-Small-24B-2507

# update 01/23
ibm-granite/granite-speech-3.3-2b
ibm-granite/granite-speech-3.3-8b
nyrahealth/CrisperWhisper

# update 01/26
# need to set language and task in advance for lite-whisper (manual)
# need to set transcription length masks for moonshine (automatic)
efficient-speech/lite-whisper-large-v3-acc
efficient-speech/lite-whisper-large-v3
efficient-speech/lite-whisper-large-v3-fast
efficient-speech/lite-whisper-large-v3-turbo-acc
efficient-speech/lite-whisper-large-v3-turbo
efficient-speech/lite-whisper-large-v3-turbo-fast
usefulsensors/moonshine-base
usefulsensors/moonshine-tiny

# update 01/27
Qwen/Qwen2.5-Omni-7B 

# unsure about the issue with qwen2-audio but the results are okay-ish (in English)
# in Chinese match papers performance
Qwen/Qwen2-Audio-7B
Qwen/Qwen2-Audio-7B-Instruct

# partial (does not match reported results)
# also needs an updated transformers (v5 transformers)
# nvidia is probably due to output format: includes prefixes such as the audio is
nvidia/audio-flamingo-3-hf

```

To add:
```
microsoft/Phi-4-multimodal-instruct
canary family
parakeet family
stepaudio-r1
ultravox
```

Autoregressive LALMs such as Voxtral or Granite require adjusting max_new_tokens to at least 200 (number used by Granite in paper)

# Qualitative Evaluation

To compare two result transcripts for discrepancies / changes between ASR models

```
python compare_transcriptions.py --file_a results/MODEL_mistralai-Voxtral-Mini-3B-2507_DATASET_hf-audio-esb-datasets-test-only-sorted_gigaspeech_test.jsonl --file_b results_2048tokens/MODEL_mistralai-Voxtral-Mini-3B-2507_DATASET_hf-audio-esb-datasets-test-only-sorted_gigaspeech_test.jsonl --label_a 548t --label_b 2048t
```

To sort results based on WER when looking for outliers or particularly problematic results:

```
python sort_jsonl_wer.py results/MODEL_mistralai-Voxtral-Mini-3B-2507_DATASET_hf-audio-esb-datasets-test-only-sorted_gigaspeech_test.jsonl --filter_wer 200
```

To download .wav audio files for checking the audio to transcribe

```
python download_audio.py --dataset_path="hf-audio/esb-datasets-test-only-sorted" --dataset="gigaspeech" --split="test" --jsonl_file results_sorted/MODEL_mistralai-Voxtral-Mini-3B-2507_DATASET_hf-audio-esb-datasets-test-only-sorted_gigaspeech_test_sorted.jsonl
```
