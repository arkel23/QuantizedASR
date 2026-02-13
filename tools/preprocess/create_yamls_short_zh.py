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
    "TwinkStart/AISHELL-1", "TwinkStart/kespeech", "TwinkStart/WenetSpeech", 
    "TwinkStart/WenetSpeech", "TwinkStart/speech-CMMLU", "TwinkStart/speech-HSK", 
    "TwinkStart/speech-HSK", "TwinkStart/speech-HSK", "TwinkStart/speech-HSK", 
    "TwinkStart/speech-HSK", "TwinkStart/speech-HSK", "TwinkStart/CommonVoice_15", 
    "JacobLinCool/common_voice_19_0_zh-TW", "JacobLinCool/common_voice_19_0_zh-TW", 
    "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-zhtw", "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hokkien", 
    "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-hakka", "TwinkStart/CommonVoice_15"
]

configs = [
    "default", "default", "default", "default", "default", "default", 
    "default", "default", "default", "default", "default", "default", 
    "default", "default", "default", "default", "default", "default"
]

splits = [
    "test", "test", "test_meeting", "test_net", "train", "hsk1", 
    "hsk2", "hsk3", "hsk4", "hsk5", "hsk6", "zh", 
    "validated_without_test", "test", "test", "test", "test", "yue"
]

names = [
    "uea_aishell1", "uea_kespeech", "uea_wenetspeech_meeting", 
    "uea_wenetspeech_net", "uea_cmmlu", "uea_hsk1", 
    "uea_hsk2", "uea_hsk3", "uea_hsk4", "uea_hsk5", "uea_hsk6", "uea_cv15_zh", 
    "mcv19_zhtw_validated", "mcv19_zhtw_test", "taiwan_tongues_zhtw", "taiwan_tongues_hokkien", 
    "taiwan_tongues_hakka", "uea_cv15_yue"
]

# 2. Setup output directory
output_dir = "configs/datasets/short_zh"
os.makedirs(output_dir, exist_ok=True)

# 3. Simple loop using the 4-list mapping
for i in range(len(dataset_paths)):
    filename = f"{names[i]}.yaml"
    filepath = os.path.join(output_dir, filename)

    yaml_data = {
        "dataset_path": QuotedStr(dataset_paths[i]),
        "dataset": QuotedStr(configs[i]),
        "split": QuotedStr(splits[i]),
        "norm_chinese": True,
        "eval_metrics": ['cer', 'ter', 'bert'],
        "force_asr_language": QuotedStr("zh"),
    }

    if 'hakka' in filename:
        yaml_data.update({"language": 'hak', "eval_metrics": ['cer', 'bert']})
    if 'hokkien' in filename:
        yaml_data.update({"language": 'nan'})
    elif 'yue' in filename:
        yaml_data.update({"language": 'yue'})
    else:
        yaml_data.update({"language": 'zh'})

    # tried this with whisper and it does not help with hsk datasets
    # some of the genargs make it give an error for qwen
    # if 'speech' in dataset_paths[i]:
    #     yaml_data.update({
    #         "long_form": True,
    #         "long_form_tricks": True,
    #     })

    with open(filepath, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {filename}")
