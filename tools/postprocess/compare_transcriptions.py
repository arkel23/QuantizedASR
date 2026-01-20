import argparse
import json
from pathlib import Path
from typing import List, Dict
from jiwer import wer, cer


def load_jsonl(filepath: str) -> Dict[str, Dict]:
    """Load JSONL file and create a dictionary keyed by audio_filepath."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            key = entry['audio_filepath']
            data[key] = entry
    return data


def compare_transcriptions(file_a: str, file_b: str, metric: str = 'wer') -> List[Dict]:
    """Compare transcriptions from two JSONL files and calculate differences."""
    data_a = load_jsonl(file_a)
    data_b = load_jsonl(file_b)

    differences = []
    common_keys = set(data_a.keys()) & set(data_b.keys())

    print(f"Found {len(common_keys)} common audio samples")
    print(f"File A has {len(data_a)} samples, File B has {len(data_b)} samples")

    for key in common_keys:
        pred_a = data_a[key]['pred_text']
        pred_b = data_b[key]['pred_text']

        if pred_a == pred_b:
            continue

        diff_entry = {
            'audio_filepath': key,
            'duration': data_a[key]['duration'],
            'time': data_a[key]['time'],
            'text': data_a[key].get('text', ''),
            'pred_text_a': pred_a,
            'pred_text_b': pred_b,
        }

        if metric in ['wer', 'both']:
            diff_entry['wer'] = wer(pred_a, pred_b)

        if metric in ['cer', 'both']:
            diff_entry['cer'] = cer(pred_a, pred_b)

        differences.append(diff_entry)

    print(f"Found {len(differences)} samples with different transcriptions")
    return differences


def rank_and_save_differences(differences: List[Dict], 
                              output_folder: str,
                              output_file: str,
                              label_a: str,
                              label_b: str,
                              metric: str = 'wer',
                              top_n: int = None) -> None:
    """Rank differences by specified metric and save to JSONL file."""
    if not differences:
        print("No differences to rank and save")
        return
    
    if metric not in differences[0]:
        metric = 'wer' if 'wer' in differences[0] else 'cer'
    
    sorted_diffs = sorted(differences, key=lambda x: x[metric], reverse=True)
    
    if top_n:
        sorted_diffs = sorted_diffs[:top_n]
    
    output_data = []
    for diff in sorted_diffs:
        entry = {
            'audio_filepath': diff['audio_filepath'],
            'duration': diff['duration'],
            'time': diff['time'],
            'text': diff['text'],
            f'pred_text_{label_a}': diff['pred_text_a'],
            f'pred_text_{label_b}': diff['pred_text_b'],
        }
        
        if 'wer' in diff:
            entry['wer'] = diff['wer']
        if 'cer' in diff:
            entry['cer'] = diff['cer']
        
        output_data.append(entry)
    
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{output_file}.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(output_data)} ranked differences to {output_path}")
    print(f"Ranking metric: {metric}")
    if output_data:
        print(f"Top difference {metric}: {output_data[0][metric]:.4f}")
        print(f"Lowest difference {metric}: {output_data[-1][metric]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare ASR transcriptions from two JSONL files and rank differences'
    )
    
    parser.add_argument('--file_a', type=str,
                        default='results/MODEL_mistralai-Voxtral-Mini-3B-2507_DATASET_hf-audio-esb-datasets-test-only-sorted_gigaspeech_test.jsonl',
                        help='Path to first JSONL file')
    parser.add_argument('--file_b', type=str,
                        default='results_2048tokens/MODEL_mistralai-Voxtral-Mini-3B-2507_DATASET_hf-audio-esb-datasets-test-only-sorted_gigaspeech_test.jsonl',
                        help='Path to second JSONL file')
    parser.add_argument('--metric', type=str, default='wer', 
                       choices=['wer', 'cer', 'both'],
                       help='Metric to use for comparison (default: wer)')
    parser.add_argument('--rank_metric', type=str, default='wer',
                       choices=['wer', 'cer'],
                       help='Metric to use for ranking (default: wer)')
    parser.add_argument('--output_folder', type=str, default='results_comparison',
                       help='Output folder for results (default: results_comparison)')
    parser.add_argument('--output_file', type=str, default='transcription_differences',
                       help='Output file name without extension (default: transcription_differences)')
    parser.add_argument('--label_a', type=str, default='512t',
                       help='Label for first file predictions (default: a)')
    parser.add_argument('--label_b', type=str, default='2048t',
                       help='Label for second file predictions (default: b)')
    parser.add_argument('--top_n', type=int, default=None,
                       help='Save only top N differences (default: all)')
    
    args = parser.parse_args()
    
    print(f"Comparing {args.file_a} and {args.file_b}")
    print(f"Using metric: {args.metric}")
    
    differences = compare_transcriptions(args.file_a, args.file_b, args.metric)
    
    if differences:
        rank_and_save_differences(
            differences=differences,
            output_folder=args.output_folder,
            output_file=args.output_file,
            label_a=args.label_a,
            label_b=args.label_b,
            metric=args.rank_metric,
            top_n=args.top_n
        )
    else:
        print("No differences found between the two files")


if __name__ == '__main__':
    main()
