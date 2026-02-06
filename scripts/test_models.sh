#!/bin/bash

models=("openai/whisper-tiny.en" "openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-medium.en" "openai/whisper-large" "openai/whisper-large-v2" "openai/whisper-large-v3" "distil-whisper/distil-medium.en" "distil-whisper/distil-large-v2" "distil-whisper/distil-large-v3" "mistralai/Voxtral-Mini-3B-2507" "mistralai/Voxtral-Small-24B-2507" "ibm-granite/granite-speech-3.3-2b" "ibm-granite/granite-speech-3.3-8b" "nyrahealth/CrisperWhisper" "Qwen/Qwen2.5-Omni-7B" "Qwen/Qwen2-Audio-7B" "Qwen/Qwen2-Audio-7B-Instruct")

base_cmd="python -m tools.evaluate --serial 997 --warmup_steps 2 --max_eval_samples 4 --batch_size 2"

# Iterate through all combinations
for i in "${!models[@]}"; do
    # Execute the command
    cmd="$base_cmd --model_id ${models[$i]}"
    echo "$cmd"
    $cmd
done
