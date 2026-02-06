#!/bin/bash

dataset_paths=("TwinkStart/AISHELL-1" "TwinkStart/kespeech" "TwinkStart/WenetSpeech" "TwinkStart/WenetSpeech" "TwinkStart/speech-CMMLU" "TwinkStart/speech-HSK" "TwinkStart/speech-HSK" "TwinkStart/speech-HSK" "TwinkStart/speech-HSK" "TwinkStart/speech-HSK" "TwinkStart/speech-HSK" "TwinkStart/CommonVoice_15" "JacobLinCool/common_voice_19_0_zh-TW" "JacobLinCool/common_voice_19_0_zh-TW" "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw" "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien" "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka" "TwinkStart/CommonVoice_15")

configs=("default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default" "default")

splits=("test" "test" "test_meeting" "test_net" "train" "hsk1" "hsk2" "hsk3" "hsk4" "hsk5" "hsk6" "zh" "validated_without_test" "test" "test" "test" "test" "yue")

base_cmd="python -m tools.evaluate --serial 999 --warmup_steps 2 --max_eval_samples 4 --chinese --eval_metrics cer bert --model_id openai/whisper-tiny --force_asr_language zh"

# Iterate through all combinations
for i in "${!dataset_paths[@]}"; do
    # Execute the command
    cmd="$base_cmd --dataset_path ${dataset_paths[$i]} --dataset ${configs[$i]} --split ${splits[$i]}"
    echo "$cmd"
    $cmd
done
