# try:
#     from .alignment import align_sequences, align_sequences_pinyin_hanzi
# except:
#     from alignment import align_sequences, align_sequences_pinyin_hanzi
from typing import List, Dict

import numpy as np
import datasets
import evaluate

# import matplotlib.pyplot as plt
# import seaborn as sns

try:
    from .romantization import get_base, get_tone_number, pinyinize_results, Pinyinizer
except:
    from romantization import get_base, get_tone_number, pinyinize_results, Pinyinizer


_CITATION = """\
Placeholder
"""

_DESCRIPTION = """
Placeholder
"""

_KWARGS_DESCRIPTION = """
Placeholder
"""

def maybe_print(*args, debugging=False):
    if debugging:
        print(*args)


def compute_ter(token_pairs_list: list,
                language: str = 'zh',
                debugging: bool = False,
                ):
    """
    Tone Error Rate: % of syllables with correct base but wrong tone
    """
    if language == 'zh':
        ntones = 5
    else:
        raise NotImplementedError
    confusion_matrix = np.zeros((ntones, ntones), dtype=int)


    total_syllables = 0
    total_pinyin_errors = 0
    total_matched_base = 0
    total_tone_errors = 0

    for aligned in token_pairs_list:
        for ref_syl, pred_syl, _, _ in aligned:
            # Count reference syllables (standard denominator)
            if ref_syl is not None:
                total_syllables += 1

            if ref_syl is None or pred_syl is None:
                total_pinyin_errors += 1
                maybe_print('Insertion/deletion error: ', ref_syl, pred_syl, debugging=debugging)
                continue  # Skip insertions/deletions

            # Split base and tone
            ref_base = get_base(ref_syl)
            ref_tone = get_tone_number(ref_syl)
            pred_base = get_base(pred_syl)
            pred_tone = get_tone_number(pred_syl)

            if ref_syl != pred_syl:
                total_pinyin_errors += 1
                maybe_print('Substitution error: ', ref_syl, pred_syl, debugging=debugging)

            # Count if bases match
            if ref_base == pred_base:
                total_matched_base += 1

                # Record in confusion matrix
                confusion_matrix[ref_tone-1][pred_tone-1] += 1

                if ref_tone != pred_tone:
                    maybe_print('Tone error: ', ref_syl, pred_syl, debugging=debugging)
                    total_tone_errors += 1

    # Original TER (from paper): tone errors / matched bases
    ter = total_tone_errors / max(total_matched_base, 1)

    # Proportion: tone errors / all syllable errors
    # tep = total_tone_errors / max(total_pinyin_errors, 1) if total_pinyin_errors > 0 else 0.0
    tep = total_tone_errors / max(total_pinyin_errors, 1)

    # Absolute tone error rate: tone errors / all syllables (similar to wer but for tones)
    ater = total_tone_errors / max(total_syllables, 1)

    result = {
        'ter': ter,
        'tep': tep,
        'ater': ater,
        'total_tone_errors': total_tone_errors,
        'total_matched_base': total_matched_base,
        'total_pinyin_errors': total_pinyin_errors,
        'total_syllables': total_syllables,
        'confusion_matrix': confusion_matrix,
    }
    maybe_print(result, debugging=debugging)

    return result


'''

def compute_homophone_error_rate(ref_chars_list: List[List[str]],
                                  pred_chars_list: List[List[str]],
                                  ref_pinyin_list: List[List[str]],
                                  pred_pinyin_list: List[List[str]]) -> dict:
    """
    Homophone Error Rate: Correct pronunciation, wrong character
    
    FIXED: Align at character level, use pinyin to check if homophone
    Since chars and pinyin have matching lengths, alignment works correctly
    """
    total_chars = 0
    total_errors = 0
    homophone_errors = 0
    
    for ref_chars, pred_chars, ref_pinyin, pred_pinyin in zip(
        ref_chars_list, pred_chars_list, ref_pinyin_list, pred_pinyin_list
    ):
        # Both should have same length (guaranteed by preprocessing)
        assert len(ref_chars) == len(ref_pinyin), "Char-pinyin length mismatch in ref"
        assert len(pred_chars) == len(pred_pinyin), "Char-pinyin length mismatch in pred"
        
        # Align at character level
        aligned_chars = align_sequences(ref_chars, pred_chars)
        
        # Track indices to get corresponding pinyin
        ref_idx = 0
        pred_idx = 0
        
        for ref_char, pred_char in aligned_chars:
            if ref_char is None:
                # Insertion: only pred has char
                pred_idx += 1
                continue
            elif pred_char is None:
                # Deletion: only ref has char
                ref_idx += 1
                continue
            else:
                # Match or substitution
                total_chars += 1
                
                # Get corresponding pinyin
                ref_pin = ref_pinyin[ref_idx]
                pred_pin = pred_pinyin[pred_idx]
                
                if ref_char != pred_char:
                    total_errors += 1
                    # Check if pinyin matches (homophone)
                    if ref_pin == pred_pin:
                        homophone_errors += 1
                
                ref_idx += 1
                pred_idx += 1
    
    return {
        'homophone_error_rate': homophone_errors / max(total_chars, 1),
        'homophone_proportion': homophone_errors / max(total_errors, 1) if total_errors > 0 else 0.0,
        'total_errors': total_errors,
        'homophone_errors': homophone_errors,
        'total_chars': total_chars
    }


def plot_tone_confusion_matrix(confusion_matrix, save_path=None):
    """
    Plot tone confusion matrix
    
    Args:
        confusion_matrix: NxN numpy array from compute_ter()
        tone_labels: List of tone labels ['1', '2', '3', '4', '5']
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Use seaborn heatmap
    sns.heatmap(confusion_matrix, 
                annot=True,  # Show numbers
                fmt='d',  # Integer format
                cmap='YlOrRd',
                # xticklabels=tone_labels,
                # yticklabels=tone_labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Tone')
    plt.ylabel('Reference Tone')
    plt.title('Tone Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
'''


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TonalASRMetrics(evaluate.Metric):
    """
    Tonal ASR evaluation metrics
    
    Design:
    - Preprocessing done ONCE in _compute()
    - Individual metrics are STANDALONE FUNCTIONS (easy to test)
    - Class only orchestrates and aggregates results
    """

    def __init__(self, language='zh', *args, **kwargs):
        """
        Initialize metrics and optional pinyinizer
        
        Args:
            pinyinizer: Optional custom pinyinizer object
                       If None, uses default pypinyin
        """
        super().__init__(*args, **kwargs)
        
        # Initialize reusable metrics
        self.language = language
        self.pinyinizer = Pinyinizer(language=language)
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/jitsi/jiwer/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Word_error_rate",
            ],
        )
    
    def _compute(self, predictions, references, return_pinyin: bool = False):
        """
        Compute all metrics
        
        Args:
            predictions: List of predicted Chinese texts
            references: List of reference Chinese texts
            return_confusion_matrix: If True, include tone confusion matrix
        
        Returns:
            Dictionary with all metrics (percentages 0-100)
        """
        results_pinyin = pinyinize_results(references, predictions, self.pinyinizer, language=self.language)

        # COMPUTE METRICS: Call standalone functions
        metrics = {}
        
        # TER with optional confusion matrix
        # ter_result = compute_ter(ref_pinyin_tone, pred_pinyin_tone, 
        #                          return_confusion_matrix=return_confusion_matrix)
        # metrics['ter'] = ter_result['ter'] * 100
        # if return_confusion_matrix:
        #     metrics['ter_confusion_matrix'] = ter_result['confusion_matrix']
        #     metrics['ter_tone_labels'] = ter_result['tone_labels']
        ter_result = compute_ter(results_pinyin['pinyin_hanzi_ref_pred_pairs'])
        for k, v in ter_result.items():
            if k in ['ter', 'tep', 'ater']:
                metrics[k] = v * 100

        # ConER
        # metrics['coner'] = compute_coner_from_lists(ref_initials, pred_initials) * 100
        
        # # VER
        # metrics['ver'] = compute_ver(ref_finals, pred_finals) * 100
        
        # # WER-T
        # metrics['wer_t'] = compute_wer_t(ref_pinyin_notone, pred_pinyin_notone) * 100
        
        # # Homophone Error Rate (THE KEY METRIC)
        # homo_results = compute_homophone_error_rate(
        #     ref_chars, pred_chars, 
        #     ref_pinyin_tone, pred_pinyin_tone
        # )
        # metrics['homophone_error_rate'] = homo_results['homophone_error_rate'] * 100
        # metrics['homophone_proportion'] = homo_results['homophone_proportion'] * 100
        # metrics['homophone_errors'] = homo_results['homophone_errors']
        # metrics['total_errors'] = homo_results['total_errors']

        if return_pinyin:
            return metrics, results_pinyin
        return metrics


if __name__ == '__main__':
    metric = TonalASRMetrics(language='zh')

    results = {}
    results['references'] = ["我沒咧驚", "我发现了问题", "你好世界", "並加速建構具備韌性的非紅供應鏈", "你再发一遍吧"]
    results['predictions'] = ["我沒驚", "我发线了问题", "你号世界", "加速建構具備韌性的供應", "麻烦您再发一遍吧"]
    results['references'] = ["我沒咧驚", "我发现了问题", "你好世界"]
    results['predictions'] = ["我沒驚", "我发线了问题", "你号世界"]
    print(results)

    scores, pinyin = metric.compute(
        references=results['references'],
        predictions=results['predictions'],
        return_pinyin=True
    )
    print(scores)
    print(pinyin)
