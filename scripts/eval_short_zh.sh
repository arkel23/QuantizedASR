#!/bin/bash

CONFIG_FILES=(
    "uea_aishell1.yaml"
    "uea_kespeech.yaml"
    "uea_wenetspeech_meeting.yaml"
    "uea_wenetspeech_net.yaml"
    "uea_cmmlu.yaml"
    "uea_hsk1.yaml"
    "uea_hsk2.yaml"
    "uea_hsk3.yaml"
    "uea_hsk4.yaml"
    "uea_hsk5.yaml"
    "uea_hsk6.yaml"
    "uea_cv15_zh.yaml"
    "mcv19_zhtw_validated.yaml"
    "mcv19_zhtw_test.yaml"
    "taiwan_tongues_zhtw.yaml"
    "taiwan_tongues_hokkien.yaml"
    "taiwan_tongues_hakka.yaml"
    "uea_cv15_yue.yaml"
)

base_cmd="python -m tools.evaluate --serial 999"

# Iterate through all combinations
for cfg in "${CONFIG_FILES[@]}"; do
    # Execute the command
    cmd="$base_cmd --config configs/models/qwen_2_audio_7b.yaml configs/datasets/short_zh/$cfg"
    echo "$cmd"
    $cmd
done
