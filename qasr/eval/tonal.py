import numpy as np
import datasets
import evaluate

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


def compute_ter_her(token_pairs_list: list,
                language: str = 'zh',
                debugging: bool = False,
                ):
    """
    Tone Error Rate: % of syllables with correct base but wrong tone
    """
    if language == 'zh':
        ntones = 5
    elif language == 'yue':
        ntones = 6
    elif language == 'nan':
        ntones = 8

    else:
        raise NotImplementedError
    confusion_matrix = np.zeros((ntones, ntones), dtype=int)


    total_syllables = 0
    total_pinyin_errors = 0
    total_matched_base = 0
    total_tone_errors = 0
    total_homophone_errors = 0
    total_char_errors = 0

    for aligned in token_pairs_list:
        for ref_syl, pred_syl, ref_char, pred_char in aligned:
            # Count reference syllables (standard denominator)
            if ref_syl is not None:
                total_syllables += 1

            if ref_syl is None or pred_syl is None:
                total_pinyin_errors += 1
                if ref_char is None or pred_char is None:
                    total_char_errors += 1
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

            if ref_char != pred_char:
                maybe_print('Character error: ', ref_char, pred_char, debugging=debugging)
                total_char_errors += 1

            # Count if bases match
            if ref_base == pred_base:
                total_matched_base += 1

                # Record in confusion matrix
                confusion_matrix[ref_tone-1][pred_tone-1] += 1

                if ref_tone != pred_tone:
                    maybe_print('Tone error: ', ref_syl, pred_syl, debugging=debugging)
                    total_tone_errors += 1

                elif ref_tone == pred_tone and ref_char != pred_char:
                    maybe_print('Homophone error: ', ref_syl, pred_syl, ref_char, pred_char, debugging=debugging)
                    total_homophone_errors += 1


    # Original TER (from paper): tone errors / matched bases
    ter = total_tone_errors / max(total_matched_base, 1)

    # Proportion: tone errors / all syllable errors
    # tep = total_tone_errors / max(total_pinyin_errors, 1) if total_pinyin_errors > 0 else 0.0
    tep = total_tone_errors / max(total_pinyin_errors, 1)

    # Absolute tone error rate: tone errors / all syllables (similar to wer but for tones)
    ater = total_tone_errors / max(total_syllables, 1)


    # homophone error proportion
    hep = total_homophone_errors / max(total_char_errors, 1)

    # homophone error rate (similar to wer but for homophones)
    her = total_homophone_errors / max(total_syllables, 1)


    result = {
        'ter': ter,
        'tep': tep,
        'ater': ater,
        'total_tone_errors': total_tone_errors,
        'total_matched_base': total_matched_base,
        'total_pinyin_errors': total_pinyin_errors,
        'total_syllables': total_syllables,
        'confusion_matrix': confusion_matrix,

        'her': her,
        'hep': hep,
        'total_homophone_errors': total_homophone_errors,
        'total_char_errors': total_char_errors,
    }
    maybe_print(result, debugging=debugging)

    return result


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
    
    def _compute(self, references, predictions, return_pinyin: bool = False):
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

        # CER-T same as CER
        metrics['cer_t'] = self.cer_metric.compute(
            references=references,
            predictions=predictions
        )

        # WER-T
        metrics['wer_t'] = self.wer_metric.compute(
            references=results_pinyin['references_base'],
            predictions=results_pinyin['predictions_base']
        )

        # TER and HER
        ter_result = compute_ter_her(results_pinyin['pinyin_hanzi_ref_pred_pairs'], language=self.language)
        for k, v in ter_result.items():
            if k in ['ter', 'tep', 'ater', 'her', 'hep']:
                metrics[k] = v
            elif k in ['confusion_matrix']:
                metrics[k] = v

        if self.language == 'zh':
            # ConER 聖母
            try:
                metrics['coner'] = self.wer_metric.compute(
                    references=results_pinyin['references_initials'],
                    predictions=results_pinyin['predictions_initials']
                )
            except:
                metrics['coner'] = np.nan

            # VER 韻母
            try:
                metrics['ver'] = self.wer_metric.compute(
                    references=results_pinyin['references_finals'],
                    predictions=results_pinyin['predictions_finals']
                )
            except:
                metrics['ver'] = np.nan


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

    for r, p in zip(results['references'], results['predictions']):
        print(r, p)
        scores, pinyin = metric.compute(
            references=[r],
            predictions=[p],
            return_pinyin=True
        )
        print(scores)
        print(pinyin)
