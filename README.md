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

hf-audio/esb-datasets-test-only-sorted ami test
hf-audio/esb-datasets-test-only-sorted common_voice test
hf-audio/esb-datasets-test-only-sorted earnings22 test
hf-audio/esb-datasets-test-only-sorted gigaspeech test
hf-audio/esb-datasets-test-only-sorted librispeech test.clean
hf-audio/esb-datasets-test-only-sorted librispeech test.other
hf-audio/esb-datasets-test-only-sorted spgispeech test
hf-audio/esb-datasets-test-only-sorted tedlium test
hf-audio/esb-datasets-test-only-sorted voxpopuli test
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
hf-audio/asr-leaderboard-longform earnings21 test
hf-audio/asr-leaderboard-longform earnings22 test
hf-audio/asr-leaderboard-longform tedlium test
distil-whisper/meanwhile default test
distil-whisper/rev16 whisper_subset test

distil-whisper/earnings22 chunked test



from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
ds_repos = ['hf-audio/esb-datasets-test-only-sorted', 'nithinraok/asr-leaderboard-datasets', 'hf-audio/asr-leaderboard-longform', 'distil-whisper/meanwhile', 'distil-whisper/rev16']
ds_repos = ['AISHELL/AISHELL-1', 'AISHELL/AISHELL-3', 'AISHELL/AISHELL-4', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka']
combs = []
for ds in ds_repos:
    cfgs = get_dataset_config_names(ds)
    for cfg in cfgs:
        splits = get_dataset_split_names(ds, cfg)
        for split in splits:
            print(ds, cfg, split)
            combs.append({'ds': ds, 'cfg': cfg, 'split': split})
            # dataset = load_dataset(ds, cfg, split=split, streaming=True, token=True)
            # print(next(iter(dataset)))


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

# update 01/27
Qwen/Qwen2.5-Omni-7B 

# partial (does not match reported results)
# nvidia is probably due to output format: includes prefixes such as the audio is
nvidia/audio-flamingo-3-hf
# unsure about the issue with qwen2-audio but the results are okay-ish
Qwen/Qwen2-Audio-7B
Qwen/Qwen2-Audio-7B-Instruct
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
