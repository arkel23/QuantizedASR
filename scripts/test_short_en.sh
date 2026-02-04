#!/bin/bash

dataset_paths=("hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "hf-audio/esb-datasets-test-only-sorted" "TwinkStart/peoples_speech" "TwinkStart/audio-MNIST" "TwinkStart/librispeech" "TwinkStart/librispeech" "TwinkStart/librispeech" "TwinkStart/librispeech" "TwinkStart/tedlium" "TwinkStart/tedlium" "TwinkStart/tedlium" "TwinkStart/tedlium" "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en" "TwinkStart/llama-questions" "TwinkStart/speech-chatbot-alpaca-eval" "TwinkStart/speech-web-questions" "TwinkStart/speech-triavia-qa" "TwinkStart/air-chat")

configs=("ami" "common_voice" "earnings22" "gigaspeech" "librispeech" "librispeech" "spgispeech" "tedlium" "voxpopuli" "default" "default" "default" "default" "default" "default" "default" "release1" "release2" "release3" "default" "default" "default" "default" "default" "default")

splits=("test" "test" "test" "test" "test.clean" "test.other" "test" "test" "test" "test" "test" "dev_clean" "dev_other" "test_clean" "test_other" "test" "test" "test" "test" "test" "test" "test" "test" "test" "test")

base_cmd="python -m tools.evaluate --serial 999 --warmup_steps 2 --max_eval_samples 4"

# Iterate through all combinations
for i in "${!dataset_paths[@]}"; do
    # Execute the command
    cmd="$base_cmd --dataset_path ${dataset_paths[$i]} --dataset ${configs[$i]} --split ${splits[$i]}"
    echo "Running: $cmd"
    $cmd
done
