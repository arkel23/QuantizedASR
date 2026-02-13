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
    "hf-audio/esb-datasets-test-only-sorted", "hf-audio/esb-datasets-test-only-sorted", 
    "hf-audio/esb-datasets-test-only-sorted", "hf-audio/esb-datasets-test-only-sorted", 
    "hf-audio/esb-datasets-test-only-sorted", "hf-audio/esb-datasets-test-only-sorted", 
    "hf-audio/esb-datasets-test-only-sorted", "hf-audio/esb-datasets-test-only-sorted", 
    "hf-audio/esb-datasets-test-only-sorted", "TwinkStart/peoples_speech", 
    "TwinkStart/audio-MNIST", "TwinkStart/librispeech", "TwinkStart/librispeech", 
    "TwinkStart/librispeech", "TwinkStart/librispeech", "TwinkStart/tedlium", 
    "TwinkStart/tedlium", "TwinkStart/tedlium", "TwinkStart/tedlium", 
    "adi-gov-tw/Taiwan-Tongues-ASR-CE-dataset-en", "TwinkStart/llama-questions", 
    "TwinkStart/speech-chatbot-alpaca-eval", "TwinkStart/speech-web-questions", 
    "TwinkStart/speech-triavia-qa", "TwinkStart/air-chat"
]

configs = [
    "ami", "common_voice", "earnings22", "gigaspeech", "librispeech", "librispeech", 
    "spgispeech", "tedlium", "voxpopuli", "default", "default", "default", "default", 
    "default", "default", "default", "release1", "release2", "release3", "default", 
    "default", "default", "default", "default", "default"
]

splits = [
    "test", "test", "test", "test", "test.clean", "test.other", "test", "test", "test", 
    "test", "test", "dev_clean", "dev_other", "test_clean", "test_other", "test", 
    "test", "test", "test", "test", "test", "test", "test", "test", "test"
]

# The "Control List" - generated based on your naming requirements
names = [
    "openasr_ami", "openasr_common_voice", "openasr_earnings22", "openasr_gigaspeech", 
    "openasr_librispeech_test_clean", "openasr_librispeech_test_other", "openasr_spgispeech", 
    "openasr_tedlium", "openasr_voxpopuli", "uea_peoples_speech", "uea_audio_MNIST", 
    "uea_librispeech_dev_clean", "uea_librispeech_dev_other", "uea_librispeech_test_clean", 
    "uea_librispeech_test_other", "uea_tedlium_test", "uea_tedlium_release1", 
    "uea_tedlium_release2", "uea_tedlium_release3", "taiwan_tongues_en", 
    "uea_llama_questions", "uea_speech_chatbot_alpaca_eval", "uea_speech_web_questions", 
    "uea_speech_trivia_qa", "uea_air_chat"
]

# 2. Setup output directory
output_dir = "configs/datasets/short_en"
os.makedirs(output_dir, exist_ok=True)

# 3. Simple loop using the 4-list mapping
for i in range(len(dataset_paths)):
    filename = f"{names[i]}.yaml"
    filepath = os.path.join(output_dir, filename)

    yaml_data = {
        "dataset_path": QuotedStr(dataset_paths[i]),
        "dataset": QuotedStr(configs[i]),
        "split": QuotedStr(splits[i]),
        "norm_english": True,
    }

    with open(filepath, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {filename}")

