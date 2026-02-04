#!/bin/bash

dataset_paths=("espnet/floras" "espnet/floras" "espnet/floras" "espnet/floras" "espnet/floras" "espnet/floras")

configs=("multilingual_ar" "multilingual_ar" "multilingual_es" "multilingual_es" "multilingual_eng" "multilingual_en")

splits=("dev" "test" "dev" "test" "dev" "test")

base_cmd="python -m tools.evaluate --serial 999 --warmup_steps 2 --max_eval_samples 4 --batch_size 2 --long_form --long_form_tricks --model openai/whisper-tiny"

# Iterate through all combinations
for i in "${!dataset_paths[@]}"; do
    # Execute the command
    cmd="$base_cmd --dataset_path ${dataset_paths[$i]} --dataset ${configs[$i]} --split ${splits[$i]}"
    echo "Running: $cmd"
    $cmd
done
