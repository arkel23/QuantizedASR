#!/bin/bash

DATASET_CONFIGS=(
    "openasr_longform_earnings21.yaml"
    "openasr_longform_earnings22.yaml"
    "openasr_longform_tedlium.yaml"
    "distil_meanwhile.yaml"
    "distil_rev16_whisper.yaml"
    "distil_tedlium_longform.yaml"
    "floras_dev.yaml"
)

# floras_test consistently fails due to oom
#     "floras_test.yaml"

MODEL_CONFIGS=(
    "whisper_tiny.yaml"
    "whisper_small.yaml"
    "whisper_base.yaml"
    "whisper_medium.yaml"
    "whisper_large.yaml"
    "whisper_large_v2.yaml"
    "whisper_large_v3.yaml"
    "whisper_large_v3_turbo.yaml"

    "lite_whisper_large_v3_acc.yaml"
    "lite_whisper_large_v3.yaml"
    "lite_whisper_large_v3_fast.yaml"
    "lite_whisper_large_v3_turbo_acc.yaml"
    "lite_whisper_large_v3_turbo.yaml"
    "lite_whisper_large_v3_turbo_fast.yaml"
)


# voxtral only works with datasets whose max length
# is not that long (~1500 s ~ 25 mins tested)
# but some such as rev are ~7500 s ~ 125 mins ~ 2 hours
# earnings21/22 are around ~5300 s and ~5700 s 
# around ~90 and ~95 mins respectively
# paper mentions chunking into 10 minutes but
# huggingface does not do this so need to manually implement
# earnings21 can process (without oom) but results are bad
# probably due to max_token_limits (8192)?
# --max_new_tokens 8192 --batch_size 2/4
    # "voxtral_small_24b.yaml"
    # "voxtral_mini_3b.yaml"

# dont work with long-form using default processors and
# not support long_form arguments that whisper uses
    # "qwen_25_omni_7b.yaml"
    # "qwen_2_audio_7b.yaml"
    # "qwen_2_audio_7b_instruct.yaml"

batch_size='128'
others=''

VALID_ARGS=$(getopt  -o '' --long batch_size:,others: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --batch_size)
        batch_size=${2}
        shift 2
        ;;
    --others)
        others=${2}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done


base_cmd="python -m tools.evaluate --serial 400 --batch_size $batch_size $others"

# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/long_en/$dataset_cfg --wandb_save_figs"
        echo "$cmd"
        $cmd
    done
done
