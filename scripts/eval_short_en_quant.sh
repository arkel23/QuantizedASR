#!/bin/bash

DATASET_CONFIGS=(
    "openasr_ami.yaml"
    "openasr_common_voice.yaml"
    "openasr_earnings22.yaml"
    "openasr_gigaspeech.yaml"
    "openasr_librispeech_test_clean.yaml"
    "openasr_librispeech_test_other.yaml"
    "openasr_spgispeech.yaml"
    "openasr_tedlium.yaml"
    "openasr_voxpopuli.yaml"
    "uea_peoples_speech.yaml"
    "uea_audio_MNIST.yaml"
    "uea_librispeech_dev_clean.yaml"
    "uea_librispeech_dev_other.yaml"
    "uea_librispeech_test_clean.yaml"
    "uea_librispeech_test_other.yaml"
    "uea_tedlium_test.yaml"
    "uea_tedlium_release1.yaml"
    "uea_tedlium_release2.yaml"
    "uea_tedlium_release3.yaml"
    "taiwan_tongues_en.yaml"
    "uea_llama_questions.yaml"
    "uea_speech_chatbot_alpaca_eval.yaml"
    "uea_speech_web_questions.yaml"
    "uea_speech_trivia_qa.yaml"
)

# error
#     "uea_air_chat.yaml"


MODEL_CONFIGS=(
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

# does not usually work with < 4 bits quant configs
#     "whisper_tiny.yaml"

# qwen2 instruct does does not perform well in any benchmark
# probably prompt issue
    # "qwen_2_audio_7b_instruct.yaml"


# QUANT_CONFIGS=(
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
# )


# SERIALS=(
#     "201"
#     "202"
#     "203"
#     "204"
#     "205"
#     "206"
#     "207"
#     "208"
#     "209"
#     "210"
#     "211"
#     "212"
#     "213"
#     "214"
#     "215"
#     "216"
#     "217"
#     "218"
#     "219"
#     "220"
# )

quant_config='bnb_w4.yaml'
others=' --serial 201'

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
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/short_en/$dataset_cfg configs/quant/$quant_config --wandb_save_figs$others"
        echo "$cmd"
        $cmd
    done
done
