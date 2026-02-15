#!/bin/bash

DATASET_CONFIGS=(
    "openasr_ml_fleurs_bg.yaml"
    "openasr_ml_fleurs_cs.yaml"
    "openasr_ml_fleurs_da.yaml"
    "openasr_ml_fleurs_de.yaml"
    "openasr_ml_fleurs_el.yaml"
    "openasr_ml_fleurs_en.yaml"
    "openasr_ml_fleurs_es.yaml"
    "openasr_ml_fleurs_et.yaml"
    "openasr_ml_fleurs_fi.yaml"
    "openasr_ml_fleurs_fr.yaml"
    "openasr_ml_fleurs_hr.yaml"
    "openasr_ml_fleurs_hu.yaml"
    "openasr_ml_fleurs_it.yaml"
    "openasr_ml_fleurs_lt.yaml"
    "openasr_ml_fleurs_lv.yaml"
    "openasr_ml_fleurs_mt.yaml"
    "openasr_ml_fleurs_nl.yaml"
    "openasr_ml_fleurs_pl.yaml"
    "openasr_ml_fleurs_pt.yaml"
    "openasr_ml_fleurs_ro.yaml"
    "openasr_ml_fleurs_ru.yaml"
    "openasr_ml_fleurs_sk.yaml"
    "openasr_ml_fleurs_sl.yaml"
    "openasr_ml_fleurs_sv.yaml"
    "openasr_ml_fleurs_uk.yaml"
    "openasr_ml_mcv_de.yaml"
    "openasr_ml_mcv_en.yaml"
    "openasr_ml_mcv_es.yaml"
    "openasr_ml_mcv_et.yaml"
    "openasr_ml_mcv_fr.yaml"
    "openasr_ml_mcv_it.yaml"
    "openasr_ml_mcv_lv.yaml"
    "openasr_ml_mcv_nl.yaml"
    "openasr_ml_mcv_pt.yaml"
    "openasr_ml_mcv_ru.yaml"
    "openasr_ml_mcv_sl.yaml"
    "openasr_ml_mcv_sv.yaml"
    "openasr_ml_mcv_uk.yaml"
    "openasr_ml_mls_es.yaml"
    "openasr_ml_mls_fr.yaml"
    "openasr_ml_mls_it.yaml"
    "openasr_ml_mls_nl.yaml"
    "openasr_ml_mls_pl.yaml"
    "openasr_ml_mls_pt.yaml"
    "uea_mls_dutch.yaml"
    "uea_mls_french.yaml"
    "uea_mls_german.yaml"
    "uea_mls_italian.yaml"
    "uea_mls_polish.yaml"
    "uea_mls_portuguese.yaml"
    "uea_mls_spanish.yaml"
    "uea_cv15_en.yaml"
    "uea_cv15_fr.yaml"
)

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
    "qwen_2_audio_7b_instruct.yaml"
)


base_cmd="python -m tools.evaluate --serial 300 --batch_size 128"

# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/short_ml/$dataset_cfg --wandb_save_figs"
        echo "$cmd"
        $cmd
    done
done
