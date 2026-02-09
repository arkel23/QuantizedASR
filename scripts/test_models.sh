#!/bin/bash

MODEL_CONFIGS=(
    "whisper_tiny_en.yaml"
    "whisper_small_en.yaml"
    "whisper_base_en.yaml"
    "whisper_medium_en.yaml"
    "whisper_large.yaml"
    "whisper_large_v2.yaml"
    "whisper_large_v3.yaml"
    "distil_whisper_medium_en.yaml"
    "distil_whisper_large_v2.yaml"
    "distil_whisper_large_v3.yaml"
    "nyrahealth_crisper_whisper.yaml"
    "lite_whisper_large_v3_acc.yaml"
    "lite_whisper_large_v3.yaml"
    "lite_whisper_large_v3_fast.yaml"
    "lite_whisper_large_v3_turbo_acc.yaml"
    "lite_whisper_large_v3_turbo.yaml"
    "lite_whisper_large_v3_turbo_fast.yaml"
    "moonshine_tiny.yaml"
    "moonshine_base.yaml"
    "voxtral_mini_3b.yaml"
    "voxtral_small_24b.yaml"
    "granite_speech_2b.yaml"
    "granite_speech_8b.yaml"
    "qwen_25_omni_7b.yaml"
    "qwen_2_audio_7b.yaml"
    "qwen_2_audio_7b_instruct.yaml"
)

base_cmd="python -m tools.evaluate --serial 997 --warmup_steps 2 --max_eval_samples 4 --batch_size 2"

# Iterate through all combinations
for model in "${MODEL_CONFIGS[@]}"; do
    # Execute the command
    cmd="$base_cmd --config configs/models/$model"
    echo "$cmd"
    $cmd
done
