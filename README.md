# Quantitative Evaluation

```
python -m tools.evaluate --model_id openai/whisper-tiny.en --dataset librispeech --split test.clean
```

Tried datasets (all `--split test` except except librispeech): 
```
ami
earnings22
gigaspeech
librispeech test.clean
librispeech test.other
spgispeech
tedlium
voxpopuli
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
# need to set language and task in advance for lite-whisper
# need to set transcription length masks for moonshine
efficient-speech/lite-whisper-large-v3-acc
efficient-speech/lite-whisper-large-v3
efficient-speech/lite-whisper-large-v3-fast
efficient-speech/lite-whisper-large-v3-turbo-acc
efficient-speech/lite-whisper-large-v3-turbo
efficient-speech/lite-whisper-large-v3-turbo-fast
usefulsensors/moonshine-base
usefulsensors/moonshine-tiny

# partial reproduction (acc lower)



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
