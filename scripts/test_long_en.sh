#!/bin/bash

dataset_paths=("hf-audio/asr-leaderboard-longform" "hf-audio/asr-leaderboard-longform" "hf-audio/asr-leaderboard-longform" "distil-whisper/meanwhile" "distil-whisper/rev16" "distil-whisper/tedlium-long-form" "espnet/floras" "espnet/floras")

configs=("earnings21" "earnings22" "tedlium" "default" "whisper_subset" "default" "monolingual" "monolingual")

splits=("test" "test" "test" "test" "test" "validation" "dev" "test")

base_cmd="python -m tools.evaluate --serial 999 --warmup_steps 2 --max_eval_samples 4 --batch_size 2 --long_form --long_form_tricks --eval_metrics wer_all"

# Iterate through all combinations
for i in "${!dataset_paths[@]}"; do
    # Execute the command
    cmd="$base_cmd --dataset_path ${dataset_paths[$i]} --dataset ${configs[$i]} --split ${splits[$i]}"
    echo "Running: $cmd"
    $cmd
done
