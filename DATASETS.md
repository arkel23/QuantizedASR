
# Datasets

For exploring datasets

```
from datasets import load_dataset, get_dataset_split_names, get_dataset_config_names
ds_repos_en_short = ['hf-audio/esb-datasets-test-only-sorted', 'TwinkStart/peoples_speech', 'TwinkStart/audio-MNIST', 'TwinkStart/librispeech', 'TwinkStart/tedlium', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en']
ds_repos_zh_short = ['TwinkStart/AISHELL-1', 'TwinkStart/kespeech', 'TwinkStart/WenetSpeech', 'JacobLinCool/common_voice_19_0_zh-TW', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien', 'adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka',]
ds_repos_nlp = ['TwinkStart/llama-questions', 'TwinkStart/speech-chatbot-alpaca-eval', 'TwinkStart/speech-web-questions', 'TwinkStart/speech-triavia-qa', 'TwinkStart/air-chat', 'TwinkStart/speech-CMMLU', 'TwinkStart/speech-HSK']
ds_repos_ml = ['nithinraok/asr-leaderboard-datasets', 'TwinkStart/CommonVoice_15', 'TwinkStart/facebook_multilingual_librispeech', 'OmniAICreator/ASMR-Archive-Processed']
ds_repos_ml_long = ['espnet/floras']
ds_repos_long = ['hf-audio/asr-leaderboard-longform', 'distil-whisper/meanwhile', 'distil-whisper/rev16', 'distil_whisper/tedlium-long-form]
ds_repos_train = ['allenai/OLMoASR-Pool', 'fixie-ai/common_voice_17_0']
ds_repos = ds_repos_en_short + ds_repos_zh_short + ds_repos_nlp + ds_repos_ml + ds_repos_ml_long + ds_repos_long


combs = []
for ds in ds_repos:
    cfgs = get_dataset_config_names(ds)
    for cfg in cfgs:
        splits = get_dataset_split_names(ds, cfg)
        for split in splits:
            print(ds, cfg, split)
            combs.append({'ds': ds, 'cfg': cfg, 'split': split})

for i in combs:
    print(i['ds'], i['cfg'], i['split'])
    try:
        dataset = load_dataset(i['ds'], i['cfg'], split=i['split'], streaming=True, token=True)
        print(next(iter(dataset)))
    except:
        print('cannot load data')
```


## Datasets with issues

```
# datasets with issues
# pip install datasets<4.0.0 
# 'wenet-e2e/wenetspeech', 'fsicoli/common_voice_15_0', 'fsicoli/common_voice_22_0',
# issue parsing json content (different types)
# 'ASLP-lab/WenetSpeech-Yue', 
# column names dont match
# 'ASLP-lab/WSYue-ASR-eval'
# no audio (need to extract from some videos but no code for that) or transcription data 
# 'ASLP-lab/WSC-Train', 'ASLP-lab/WSC-Eval',
# missing transcriptions or audio
# 'AISHELL/AISHELL-1', 'AISHELL/AISHELL-3', 'AISHELL/AISHELL-4', 

# not suitable for ASR
# 'TwinkStart/MMAU', 
# TwinkStart/MMAU default v05.15.25

# needs additional preprocessing, also not compatible with streaming
# speechcolab/gigaspeech2 default train
```

