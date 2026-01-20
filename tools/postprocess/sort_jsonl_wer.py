import argparse
import json
from pathlib import Path
import statistics
from jiwer import wer


def compute_and_sort_wer(input_file: str, output_folder: str, filter_wer: float = None) -> None:
    """
    Read JSONL file, compute WER for each sample, sort by WER, and save results.
    
    Args:
        input_file: Path to input JSONL file
        output_folder: Folder to save output file
    """
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    
    print(f"Loaded {len(data)} samples from {input_file}")
    
    # Compute WER for each sample
    all_references = []
    all_predictions = []
    
    for entry in data:
        reference = entry['text']
        prediction = entry['pred_text']
        entry['wer'] = wer(reference, prediction)
        
        all_references.append(reference)
        all_predictions.append(prediction)
    
    # Sort by WER (highest to lowest)
    data.sort(key=lambda x: x['wer'], reverse=True)
    
    # Create output path
    input_path = Path(input_file)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_path.stem}_sorted{input_path.suffix}"
    
    # Save sorted results
    # Filter data if threshold is specified
    data_to_save = data
    if filter_wer is not None:
        threshold = filter_wer / 100  # Convert percentage to decimal
        data_to_save = [entry for entry in data if entry['wer'] > threshold]
        print(f"Filtered to {len(data_to_save)} samples with WER > {filter_wer}%")

    # Save sorted results
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data_to_save:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data_to_save)} sorted samples to {output_file}")
    
    # Print statistics
    if data:
        # Corpus-level WER (matches evaluation script)
        corpus_wer = wer(all_references, all_predictions)
        
        # Average of individual WERs (for comparison)
        sample_wers = [e['wer'] for e in data]
        avg_sample_wer = statistics.mean(sample_wers)
        median_sample_wer = statistics.median(sample_wers)
        std_sample_wer = statistics.stdev(sample_wers) if len(sample_wers) > 1 else 0.0
        
        print(f"\nWER Statistics:")
        print(f"  Corpus-level WER: {corpus_wer * 100:.2f}%")
        print(f"  Average sample WER: {avg_sample_wer * 100:.2f}%")
        print(f"  Median sample WER: {median_sample_wer * 100:.2f}%")
        print(f"  Std Dev sample WER: {std_sample_wer * 100:.2f}%")
        print(f"  Highest sample WER: {data[0]['wer'] * 100:.2f}% (audio: {data[0]['audio_filepath']})")
        print(f"  Lowest sample WER: {data[-1]['wer'] * 100:.2f}% (audio: {data[-1]['audio_filepath']})")


def main():
    parser = argparse.ArgumentParser(
        description='Compute WER for each sample and sort from highest to lowest'
    )
    
    parser.add_argument('input_file', type=str,
                       help='Path to input JSONL file')
    parser.add_argument('--output_folder', type=str, default='results_sorted',
                       help='Output folder for sorted results (default: results)')
    parser.add_argument('--filter_wer', type=float, default=None,
                       help='Only save samples with WER above this threshold (e.g., 10 for 10%)')

    args = parser.parse_args()

    compute_and_sort_wer(args.input_file, args.output_folder, args.filter_wer)


if __name__ == '__main__':
    main()
