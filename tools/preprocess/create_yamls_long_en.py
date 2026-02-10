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
    "hf-audio/asr-leaderboard-longform", "hf-audio/asr-leaderboard-longform", 
    "hf-audio/asr-leaderboard-longform", "distil-whisper/meanwhile", 
    "distil-whisper/rev16", "distil-whisper/tedlium-long-form", 
    "espnet/floras", "espnet/floras"
]

configs = [
    "earnings21", "earnings22", "tedlium", "default", 
    "whisper_subset", "default", "monolingual", "monolingual"
]

splits = [
    "test", "test", "test", "test", 
    "test", "validation", "dev", "test"
]

names = [
    "openasr_longform_earnings21", "openasr_longform_earnings22", "openasr_longform_tedlium", 
    "distil_meanwhile", "distil_rev16_whisper", "distil_tedlium_longform", 
    "floras_dev", "floras_test"
]

# 2. Setup output directory
output_dir = "configs/datasets/long_en"
os.makedirs(output_dir, exist_ok=True)

# 3. Simple loop using the 4-list mapping
for i in range(len(dataset_paths)):
    filename = f"{names[i]}.yaml"
    filepath = os.path.join(output_dir, filename)

    yaml_data = {
        "dataset_path": QuotedStr(dataset_paths[i]),
        "dataset": QuotedStr(configs[i]),
        "split": QuotedStr(splits[i]),
        "long_form": True,
        "long_form_tricks": True,
        "eval_metrics": ['wer_all'],
    }

    with open(filepath, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {filename}")

