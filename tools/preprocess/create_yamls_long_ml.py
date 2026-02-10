import os
import yaml

# --- 1. Define the Languages from the Plot ---
# Extracted left-to-right from the provided image
floras_codes = [
    "en", "ru", "es", "de", "fr", "it", "id", "ja", "pt", "tr", 
    "pl", "zh", "nl", "hu", "eu", "hi", "vi", "fi", "uk", "el", 
    "ro", "ca", "cs", "th", "et", "ms", "eo", "fa", "ta", "sk", 
    "sl", "hr", "da", "sr", "la", "ar", "uz", "bg", "sv", "ur", 
    "gl", "bn", "bs", "cy", "ku", "ky", "ka", "az", "mi"
]

# --- 2. Generate the Lists Dynamically ---
dataset_paths = []
configs = []
splits = []
names = []
languages = []

for code in floras_codes:
    # We generate a 'dev' and 'test' entry for every language
    for split_name in ["dev", "test"]:
        dataset_paths.append("espnet/floras")
        
        # Config name convention: multilingual_{code}
        configs.append(f"multilingual_{code}")
        
        splits.append(split_name)
        
        # Unique file name: floras_{code}_{split}
        names.append(f"floras_{code}_{split_name}")
        
        # ISO language code
        languages.append(code)

# --- 3. Custom Representer for Quoted Strings ---
class QuotedStr(str): pass
def quoted_scalar(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
yaml.add_representer(QuotedStr, quoted_scalar)

# --- 4. Write the Files ---
output_dir = "configs/datasets/long_ml"
os.makedirs(output_dir, exist_ok=True)

for i in range(len(dataset_paths)):
    filename = f"{names[i]}.yaml"
    filepath = os.path.join(output_dir, filename)

    yaml_data = {
        "dataset_path": QuotedStr(dataset_paths[i]),
        "dataset": QuotedStr(configs[i]),
        "split": QuotedStr(splits[i]),
        "force_asr_language": QuotedStr(languages[i]),
        "long_form": True,
        "long_form_tricks": True,
        "eval_metrics": ['wer_all'],
    }

    with open(filepath, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    # Optional: Print progress
    print(f"Created: {filename}")

print(f"Successfully generated {len(names)} config files for {len(floras_codes)} languages.")

