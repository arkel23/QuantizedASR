import os
import yaml

# 1. Custom Representer Logic
class QuotedStr(str):
    """A helper class to tell PyYAML to use single quotes."""
    pass

def quoted_scalar(dumper, data):
    """Tell the dumper to always use single quotes for QuotedStr."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

# Register the representer
yaml.add_representer(QuotedStr, quoted_scalar)

# 1. Provide total control through explicit lists
dataset_paths = [
    "nithinraok/asr-leaderboard-datasets"] * 44 + [
    "TwinkStart/facebook_multilingual_librispeech"] * 7 + [
    "TwinkStart/CommonVoice_15"] * 2

configs = [
    "fleurs_bg", "fleurs_cs", "fleurs_da", "fleurs_de", "fleurs_el", "fleurs_en", "fleurs_es", "fleurs_et", "fleurs_fi", "fleurs_fr", "fleurs_hr", "fleurs_hu", "fleurs_it", "fleurs_lt", "fleurs_lv", "fleurs_mt", "fleurs_nl", "fleurs_pl", "fleurs_pt", "fleurs_ro", "fleurs_ru", "fleurs_sk", "fleurs_sl", "fleurs_sv", "fleurs_uk", 
    "mcv_de", "mcv_en", "mcv_es", "mcv_et", "mcv_fr", "mcv_it", "mcv_lv", "mcv_nl", "mcv_pt", "mcv_ru", "mcv_sl", "mcv_sv", "mcv_uk", 
    "mls_es", "mls_fr", "mls_it", "mls_nl", "mls_pl", "mls_pt", 
    "default", "default", "default", "default", "default", "default", "default", 
    "default", "default"
]

splits = [
    "test"] * 44 + [
    "mls_dutch", "mls_french", "mls_german", "mls_italian", "mls_polish", "mls_portuguese", "mls_spanish", 
    "en", "fr"
]

names = [
    "openasr_ml_fleurs_bg", "openasr_ml_fleurs_cs", "openasr_ml_fleurs_da", "openasr_ml_fleurs_de", "openasr_ml_fleurs_el", "openasr_ml_fleurs_en", "openasr_ml_fleurs_es", "openasr_ml_fleurs_et", "openasr_ml_fleurs_fi", "openasr_ml_fleurs_fr", "openasr_ml_fleurs_hr", "openasr_ml_fleurs_hu", "openasr_ml_fleurs_it", "openasr_ml_fleurs_lt", "openasr_ml_fleurs_lv", "openasr_ml_fleurs_mt", "openasr_ml_fleurs_nl", "openasr_ml_fleurs_pl", "openasr_ml_fleurs_pt", "openasr_ml_fleurs_ro", "openasr_ml_fleurs_ru", "openasr_ml_fleurs_sk", "openasr_ml_fleurs_sl", "openasr_ml_fleurs_sv", "openasr_ml_fleurs_uk",
    "openasr_ml_mcv_de", "openasr_ml_mcv_en", "openasr_ml_mcv_es", "openasr_ml_mcv_et", "openasr_ml_mcv_fr", "openasr_ml_mcv_it", "openasr_ml_mcv_lv", "openasr_ml_mcv_nl", "openasr_ml_mcv_pt", "openasr_ml_mcv_ru", "openasr_ml_mcv_sl", "openasr_ml_mcv_sv", "openasr_ml_mcv_uk",
    "openasr_ml_mls_es", "openasr_ml_mls_fr", "openasr_ml_mls_it", "openasr_ml_mls_nl", "openasr_ml_mls_pl", "openasr_ml_mls_pt",
    "uea_mls_dutch", "uea_mls_french", "uea_mls_german", "uea_mls_italian", "uea_mls_polish", "uea_mls_portuguese", "uea_mls_spanish",
    "uea_cv15_en", "uea_cv15_fr"
]

languages = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
    "de", "en", "es", "et", "fr", "it", "lv", "nl", "pt", "ru", "sl", "sv", "uk",
    "es", "fr", "it", "nl", "pl", "pt",
    "nl", "fr", "de", "it", "pl", "pt", "es",
    "en", "fr"
]

# 2. Setup output directory
output_dir = "configs/datasets/short_ml"
os.makedirs(output_dir, exist_ok=True)

# 3. Simple loop using the 4-list mapping
for i in range(len(dataset_paths)):
    filename = f"{names[i]}.yaml"
    filepath = os.path.join(output_dir, filename)

    yaml_data = {
        "dataset_path": QuotedStr(dataset_paths[i]),
        "dataset": QuotedStr(configs[i]),
        "split": QuotedStr(splits[i]),
        "force_asr_language": QuotedStr(languages[i]),
    }

    if languages[i] == 'en':
        yaml_data.update({"norm_english": True})

    with open(filepath, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {filename}")

