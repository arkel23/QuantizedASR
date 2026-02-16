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
    "uea_air_chat.yaml"
)

MODEL_CONFIGS=(
    "voxtral_small_24b.yaml"
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


base_cmd="python -m tools.evaluate --serial 200 --batch_size 128"

# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/short_en/$dataset_cfg --wandb_save_figs"
        echo "$cmd"
        $cmd
    done
done
