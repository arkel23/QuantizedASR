#!/bin/bash

CONFIG_FILES=(
    "openasr_longform_earnings21.yaml"
    "openasr_longform_earnings22.yaml"
    "openasr_longform_tedlium.yaml"
    "distil_meanwhile.yaml"
    "distil_rev16_whisper.yaml"
    "distil_tedlium_longform.yaml"
    "floras_dev.yaml"
    "floras_test.yaml"
)
base_cmd="python -m tools.evaluate --serial 994 --warmup_steps 2 --max_eval_samples 4 --batch_size 2"

# Iterate through all combinations
for cfg in "${CONFIG_FILES[@]}"; do
    # Execute the command
    cmd="$base_cmd --config configs/datasets/long_en/$cfg"
    echo "$cmd"
    $cmd
done
