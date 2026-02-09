#!/bin/bash

CONFIG_FILES=(
    "bnb_w4.yaml"
    "bnb_w8.yaml"
    "quanto_wfloat8.yaml"
    "quanto_wint2.yaml"
    "quanto_wint4.yaml"
    "quanto_wint8.yaml"
    "hqq_wint1.yaml"
    "hqq_wint2.yaml"
    "hqq_wint3.yaml"
    "hqq_wint4.yaml"
    "hqq_wint8.yaml"
    "torchao_wfloat8_afloat8.yaml"
    "torchao_wint4_aint8.yaml"
    "torchao_wint8_aint8.yaml"
    "torchao_wint1.yaml"
    "torchao_wint2.yaml"
    "torchao_wint3.yaml"
    "torchao_wint4.yaml"
    "torchao_wint5.yaml"
    "torchao_wint6.yaml"
    "torchao_wint7.yaml"
    "torchao_wint8.yaml"
    "torchao_wfloat8.yaml"
)

# torchao_wint4 requires fbgemm
# quanto wint2/4/wfloat8 require cuda support
base_cmd="python -m tools.evaluate --serial 998 --warmup_steps 2 --max_eval_samples 4"

# Iterate through all combinations
for cfg in "${CONFIG_FILES[@]}"; do
    # Execute the command
    cmd="$base_cmd --config configs/quant/$cfg"
    echo "$cmd"
    $cmd
done
