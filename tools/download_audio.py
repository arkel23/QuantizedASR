import argparse
import json
from datasets import load_dataset
import soundfile as sf
from pathlib import Path


def download_audio_from_dataset(
    dataset_path: str,
    dataset_name: str,
    split: str,
    audio_filepath: str,
    output_path: str = None,
    streaming: bool = False
):
    """
    Download audio file from HuggingFace dataset.
    
    Args:
        dataset_path: HuggingFace dataset path (e.g., 'mozilla-foundation/common_voice_11_0')
        dataset_name: Dataset config name (e.g., 'en')
        split: Dataset split (e.g., 'test', 'validation')
        audio_filepath: The audio_filepath from your JSONL (e.g., 'sample_0', 'sample_123')
        output_path: Where to save the WAV file (default: '{audio_filepath}.wav')
        streaming: Use streaming mode for large datasets
    """
    # Load dataset
    print(f"Loading dataset {dataset_path}/{dataset_name} split={split}")
    dataset = load_dataset(
        dataset_path,
        dataset_name,
        split=split,
        streaming=streaming,
    )
    
    # Extract sample index from audio_filepath (assuming format like 'sample_0', 'sample_123')
    if audio_filepath.startswith('sample_'):
        sample_idx = int(audio_filepath.split('_')[1])
    else:
        # Try to parse as integer directly
        try:
            sample_idx = int(audio_filepath)
        except:
            print(f"Cannot parse audio_filepath: {audio_filepath}")
            return
    
    print(f"Extracting sample at index {sample_idx}")
    
    # Get the specific sample
    if streaming:
        # For streaming datasets, iterate to the index
        for i, sample in enumerate(dataset):
            if i == sample_idx:
                target_sample = sample
                break
    else:
        # For non-streaming, direct access
        target_sample = dataset[sample_idx]
    
    # Extract audio data
    audio_data = target_sample['audio']
    audio_array = audio_data['array']
    sampling_rate = audio_data['sampling_rate']
    
    # Set output path
    if output_path is None:
        output_path = f"{audio_filepath}.wav"
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as WAV file
    sf.write(output_path, audio_array, sampling_rate)
    print(f"Audio saved to: {output_path}")
    print(f"Duration: {len(audio_array) / sampling_rate:.2f} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Also print the reference text if available
    if 'text' in target_sample:
        print(f"Reference text: {target_sample['text']}")


def batch_download_from_jsonl(
    jsonl_path: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    output_folder: str = 'audio_samples',
    max_samples: int = None
):
    """
    Download multiple audio files listed in a JSONL comparison file.
    """
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_to_download = data[:max_samples] if max_samples else data
    
    print(f"Downloading {len(samples_to_download)} audio samples...")
    
    for i, entry in enumerate(samples_to_download):
        audio_filepath = entry['audio_filepath']
        output_path = output_dir / f"{audio_filepath}.wav"
        
        print(f"\n[{i+1}/{len(samples_to_download)}] Processing {audio_filepath}")
        
        try:
            download_audio_from_dataset(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                split=split,
                audio_filepath=audio_filepath,
                output_path=str(output_path)
            )
        except Exception as e:
            print(f"Error downloading {audio_filepath}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download audio files from HuggingFace dataset'
    )
    
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='HuggingFace dataset path')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset config name')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split (default: test)')
    parser.add_argument('--audio_filepath', type=str,
                       help='Single audio filepath to download (e.g., sample_0)')
    parser.add_argument('--jsonl_file', type=str,
                       help='JSONL file with comparison results to batch download')
    parser.add_argument('--output_folder', type=str, default='audio_samples',
                       help='Output folder for audio files (default: audio_samples)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to download from JSONL')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode for large datasets')
    
    args = parser.parse_args()
    
    if args.jsonl_file:
        # Batch download mode
        batch_download_from_jsonl(
            jsonl_path=args.jsonl_file,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset,
            split=args.split,
            output_folder=args.output_folder,
            max_samples=args.max_samples
        )
    elif args.audio_filepath:
        # Single download mode
        download_audio_from_dataset(
            dataset_path=args.dataset_path,
            dataset_name=args.dataset,
            split=args.split,
            audio_filepath=args.audio_filepath,
            streaming=args.streaming
        )
    else:
        parser.error("Must specify either --audio_filepath or --jsonl_file")


if __name__ == '__main__':
    main()
