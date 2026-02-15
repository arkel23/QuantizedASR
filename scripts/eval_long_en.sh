#!/bin/bash

DATASET_CONFIGS=(
    "openasr_longform_earnings21.yaml"
)
    # "openasr_longform_earnings22.yaml"
    # "openasr_longform_tedlium.yaml"
    # "distil_meanwhile.yaml"
    # "distil_rev16_whisper.yaml"
    # "distil_tedlium_longform.yaml"
    # "floras_dev.yaml"
    # "floras_test.yaml"

MODEL_CONFIGS=(
    "whisper_large_v3.yaml"

    "voxtral_small_24b.yaml"
    "qwen_25_omni_7b.yaml"
    "qwen_2_audio_7b.yaml"
)

    # "whisper_tiny.yaml"
    # "whisper_small.yaml"
    # "whisper_base.yaml"
    # "whisper_medium.yaml"
    # "whisper_large.yaml"
    # "whisper_large_v2.yaml"
    # "whisper_large_v3.yaml"
    # "whisper_large_v3_turbo.yaml"

    # "lite_whisper_large_v3_acc.yaml"
    # "lite_whisper_large_v3.yaml"
    # "lite_whisper_large_v3_fast.yaml"
    # "lite_whisper_large_v3_turbo_acc.yaml"
    # "lite_whisper_large_v3_turbo.yaml"
    # "lite_whisper_large_v3_turbo_fast.yaml"

    # "voxtral_mini_3b.yaml"
    # "voxtral_small_24b.yaml"
    # "qwen_25_omni_7b.yaml"
    # "qwen_2_audio_7b.yaml"
    # "qwen_2_audio_7b_instruct.yaml"



base_cmd="python -m tools.evaluate --serial 400 --batch_size 128"

# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/long_en/$dataset_cfg --wandb_save_figs"
        echo "$cmd"
        $cmd
    done
done
