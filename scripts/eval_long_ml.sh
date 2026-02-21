#!/bin/bash

DATASET_CONFIGS=(
    "floras_ru_dev.yaml"
    "floras_ru_test.yaml"
    "floras_es_dev.yaml"
    "floras_es_test.yaml"
    "floras_de_dev.yaml"
    "floras_de_test.yaml"
    "floras_fr_dev.yaml"
    "floras_fr_test.yaml"
    "floras_it_dev.yaml"
    "floras_it_test.yaml"
    "floras_id_dev.yaml"
    "floras_id_test.yaml"
    "floras_ja_dev.yaml"
    "floras_ja_test.yaml"
    "floras_pt_dev.yaml"
    "floras_pt_test.yaml"
    "floras_tr_dev.yaml"
    "floras_tr_test.yaml"
    "floras_pl_dev.yaml"
    "floras_pl_test.yaml"
    "floras_zh_dev.yaml"
    "floras_zh_test.yaml"
    "floras_nl_dev.yaml"
    "floras_nl_test.yaml"
    "floras_hu_dev.yaml"
    "floras_hu_test.yaml"
    "floras_eu_dev.yaml"
    "floras_eu_test.yaml"
    "floras_hi_dev.yaml"
    "floras_hi_test.yaml"
    "floras_vi_dev.yaml"
    "floras_vi_test.yaml"
    "floras_fi_dev.yaml"
    "floras_fi_test.yaml"
    "floras_uk_dev.yaml"
    "floras_uk_test.yaml"
    "floras_el_dev.yaml"
    "floras_el_test.yaml"
    "floras_ro_dev.yaml"
    "floras_ro_test.yaml"
    "floras_ca_dev.yaml"
    "floras_ca_test.yaml"
    "floras_cs_dev.yaml"
    "floras_cs_test.yaml"
    "floras_th_dev.yaml"
    "floras_th_test.yaml"
    "floras_et_dev.yaml"
    "floras_et_test.yaml"
    "floras_ms_dev.yaml"
    "floras_ms_test.yaml"
    "floras_fa_dev.yaml"
    "floras_fa_test.yaml"
    "floras_ta_dev.yaml"
    "floras_ta_test.yaml"
    "floras_sk_dev.yaml"
    "floras_sk_test.yaml"
    "floras_sl_dev.yaml"
    "floras_sl_test.yaml"
    "floras_hr_dev.yaml"
    "floras_hr_test.yaml"
    "floras_da_dev.yaml"
    "floras_da_test.yaml"
    "floras_sr_dev.yaml"
    "floras_sr_test.yaml"
    "floras_la_dev.yaml"
    "floras_la_test.yaml"
    "floras_ar_dev.yaml"
    "floras_ar_test.yaml"
    "floras_uz_dev.yaml"
    "floras_uz_test.yaml"
    "floras_bg_dev.yaml"
    "floras_bg_test.yaml"
    "floras_sv_dev.yaml"
    "floras_sv_test.yaml"
    "floras_ur_dev.yaml"
    "floras_ur_test.yaml"
    "floras_gl_dev.yaml"
    "floras_gl_test.yaml"
    "floras_bn_dev.yaml"
    "floras_bn_test.yaml"
    "floras_cy_dev.yaml"
    "floras_cy_test.yaml"
    "floras_ka_dev.yaml"
    "floras_ka_test.yaml"
    "floras_az_dev.yaml"
    "floras_az_test.yaml"
    "floras_mi_dev.yaml"
    "floras_mi_test.yaml"
)

# whisper does not support esperanto (eo)
#     "floras_eo_dev.yaml"
#    "floras_eo_test.yaml"

# does not output anything (filtered out?)
    # "floras_bs_dev.yaml"
    # "floras_bs_test.yaml"
    # "floras_ku_dev.yaml"
    # "floras_ku_test.yaml"
    # "floras_ky_dev.yaml"
    # "floras_ky_test.yaml"


MODEL_CONFIGS=(
    "whisper_large_v3.yaml"

)

# takes a while so this is just for a test
    # "whisper_tiny.yaml"
    # "whisper_small.yaml"
    # "whisper_base.yaml"
    # "whisper_medium.yaml"
    # "whisper_large.yaml"
    # "whisper_large_v2.yaml"
    # "whisper_large_v3_turbo.yaml"


    # "lite_whisper_large_v3_acc.yaml"
    # "lite_whisper_large_v3.yaml"
    # "lite_whisper_large_v3_fast.yaml"
    # "lite_whisper_large_v3_turbo_acc.yaml"
    # "lite_whisper_large_v3_turbo.yaml"
    # "lite_whisper_large_v3_turbo_fast.yaml"


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


base_cmd="python -m tools.evaluate --serial 500 --batch_size $batch_size $others"


# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/long_ml/$dataset_cfg --wandb_save_figs"
        echo "$cmd"
        $cmd
    done
done
