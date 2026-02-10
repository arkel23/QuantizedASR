#!/bin/bash

CONFIG_FILES=(
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

base_cmd="python -m tools.evaluate --serial 991 --warmup_steps 2 --max_eval_samples 4"

# Iterate through all combinations
for cfg in "${CONFIG_FILES[@]}"; do
    # Execute the command
    cmd="$base_cmd --config configs/datasets/short_en/$cfg"
    echo "$cmd"
    $cmd
done
