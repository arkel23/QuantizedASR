#!/bin/bash

DATASET_CONFIGS=(
    "uea_aishell1.yaml"
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
    "uea_cv15_yue.yaml"
)

# no current model works well for hokkien without ft
    # "taiwan_tongues_hokkien.yaml"


    # does not work in hpc / no standardized tones romanization for hakka
    # "uea_kespeech.yaml"
    # "taiwan_tongues_hakka.yaml"

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

    "voxtral_mini_3b.yaml"
    "voxtral_small_24b.yaml"
    "qwen_25_omni_7b.yaml"
    "qwen_2_audio_7b.yaml"
)

# qwen2 instruct does does not perform well in any benchmark
# probably prompt issue
    # "qwen_2_audio_7b_instruct.yaml"


# QUANT_CONFIGS=(
    # "bnb_w4.yaml"
    # "bnb_w8.yaml"
# )

    # "bnb_w4.yaml"
    # "bnb_w8.yaml"
    # "quanto_wint8.yaml"
    # "hqq_wint1.yaml"
    # "hqq_wint2.yaml"
    # "hqq_wint3.yaml"
    # "hqq_wint4.yaml"
    # "hqq_wint8.yaml"
    # "torchao_wint1.yaml"
    # "torchao_wint2.yaml"
    # "torchao_wint3.yaml"
    # "torchao_wint4.yaml"
    # "torchao_wint5.yaml"
    # "torchao_wint6.yaml"
    # "torchao_wint7.yaml"
    # "torchao_wint8.yaml"
    # "torchao_wfloat8.yaml"
    # "torchao_wfloat8_afloat8.yaml"
    # "torchao_wint4_aint8.yaml"
    # "torchao_wint8_aint8.yaml"


# SERIALS=(
#     "101"
#     "102"
#     "103"
#     "104"
#     "105"
#     "106"
#     "107"
#     "108"
#     "109"
#     "110"
#     "111"
#     "112"
#     "113"
#     "114"
#     "115"
#     "116"
#     "117"
#     "118"
#     "119"
#     "120"
# )

quant_config='bnb_w4.yaml'
others=' --serial 101'

VALID_ARGS=$(getopt  -o '' --long quant_config:,others: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --quant_config)
        quant_config=${2}
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

base_cmd="python -m tools.evaluate --batch_size 128"

# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/short_zh/$dataset_cfg configs/quant/$quant_config --wandb_save_figs$others"
        echo "$cmd"
        $cmd
    done
done
