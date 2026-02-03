# https://github.com/huggingface/evaluate/blob/main/metrics/wer/wer.py
# Copyright 2021 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Word Error Ratio (WER) metric. """

from statistics import mean, stdev

import datasets
import jiwer
import evaluate


EMBEDDER = 'microsoft/deberta-large-mnli'


_CITATION = """\
@inproceedings{inproceedings,
    author = {Morris, Andrew and Maier, Viktoria and Green, Phil},
    year = {2004},
    month = {01},
    pages = {},
    title = {From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition.}
}
"""

_DESCRIPTION = """\
Word error rate (WER) is a common metric of the performance of an automatic speech recognition system.

The general difficulty of measuring performance lies in the fact that the recognized word sequence can have a different length from the reference word sequence (supposedly the correct one). The WER is derived from the Levenshtein distance, working at the word level instead of the phoneme level. The WER is a valuable tool for comparing different systems as well as for evaluating improvements within one system. This kind of measurement, however, provides no details on the nature of translation errors and further work is therefore required to identify the main source(s) of error and to focus any research effort.

This problem is solved by first aligning the recognized word sequence with the reference (spoken) word sequence using dynamic string alignment. Examination of this issue is seen through a theory called the power law that states the correlation between perplexity and word error rate.

Word error rate can then be computed as:

WER = (S + D + I) / N = (S + D + I) / (S + D + C)

where

S is the number of substitutions,
D is the number of deletions,
I is the number of insertions,
C is the number of correct words,
N is the number of words in the reference (N=S+D+C).

This value indicates the average number of errors per reference word. The lower the value, the better the
performance of the ASR system with a WER of 0 being a perfect score.
"""

_KWARGS_DESCRIPTION = """
Compute WER score of transcribed segments against references.

Args:
    references: List of references for each speech input.
    predictions: List of transcriptions to score.
    concatenate_texts (bool, default=False): Whether to concatenate all input texts or compute WER iteratively.

Returns:
    (float): the word error rate

Examples:

    >>> predictions = ["this is the prediction", "there is an other sample"]
    >>> references = ["this is the reference", "there is another one"]
    >>> wer = evaluate.load("wer")
    >>> wer_score = wer.compute(predictions=predictions, references=references)
    >>> print(wer_score)
    0.5
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class WERMetrics(evaluate.Metric):
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
    
    def _compute(self, predictions=None, references=None, concatenate_texts=True, return_all_metrics=True):
        if hasattr(jiwer, "compute_measures"):
            if concatenate_texts:
                measures = jiwer.compute_measures(references, predictions)
                if return_all_metrics:
                    return {
                        'wer': measures['wer'],
                        'mer': measures['mer'],
                        'wil': measures['wil'],
                        'wip': measures['wip'],
                        'substitutions': measures['substitutions'],
                        'deletions': measures['deletions'],
                        'insertions': measures['insertions'],
                        'hits': measures['hits'],
                    }
                return measures["wer"]
            else:
                # Aggregate metrics across samples
                totals = {
                    'substitutions': 0,
                    'deletions': 0,
                    'insertions': 0,
                    'hits': 0,
                }
                for prediction, reference in zip(predictions, references):
                    measures = jiwer.compute_measures(reference, prediction)
                    totals['substitutions'] += measures['substitutions']
                    totals['deletions'] += measures['deletions']
                    totals['insertions'] += measures['insertions']
                    totals['hits'] += measures['hits']
                
                incorrect = (totals['substitutions'] + totals['deletions'] + totals['insertions'])
                total_words = totals['substitutions'] + totals['deletions'] + totals['hits']
                wer = incorrect / total_words
                
                if return_all_metrics:
                    return {
                        'wer': wer,
                        'substitutions': totals['substitutions'],
                        'deletions': totals['deletions'],
                        'insertions': totals['insertions'],
                        'hits': totals['hits'],
                        # MER/WIL/WIP need concatenated text, compute separately if needed
                    }
                return wer
        else:
            # Fallback for older jiwer versions
            if concatenate_texts:
                measures = jiwer.process_words(references, predictions)
                if return_all_metrics:
                    return {
                        'wer': measures.wer,
                        'mer': measures.mer,
                        'wil': measures.wil,
                        'wip': measures.wip,
                        'substitutions': measures.substitutions,
                        'deletions': measures.deletions,
                        'insertions': measures.insertions,
                        'hits': measures.hits,
                    }
                return measures.wer
            else:
                totals = {
                    'substitutions': 0,
                    'deletions': 0,
                    'insertions': 0,
                    'hits': 0,
                }
                for prediction, reference in zip(predictions, references):
                    measures = jiwer.process_words(reference, prediction)
                    totals['substitutions'] += measures.substitutions
                    totals['deletions'] += measures.deletions
                    totals['insertions'] += measures.insertions
                    totals['hits'] += measures.hits
                
                incorrect = (totals['substitutions'] + totals['deletions'] + totals['insertions'])
                total_words = totals['substitutions'] + totals['deletions'] + totals['hits']
                wer = incorrect / total_words
                
                if return_all_metrics:
                    return {
                        'wer': wer,
                        'substitutions': totals['substitutions'],
                        'deletions': totals['deletions'],
                        'insertions': totals['insertions'],
                        'hits': totals['hits'],
                    }
                return wer


def compute_metrics(results, eval_metric):
    scores_dic = {}
    
    # Normalize to list
    if isinstance(eval_metric, str):
        eval_metrics = [eval_metric]
    else:
        eval_metrics = eval_metric
    
    for metric_name in eval_metrics:
        # Load metric
        if metric_name in ['wer', 'cer']:
            metric = evaluate.load(metric_name)
        elif metric_name == 'wer_all':
            metric = WERMetrics()
        elif metric_name == 'bert':
            metric = evaluate.load('bertscore')
        
        # Compute scores
        if metric_name in ['wer', 'cer', 'wer_all']:
            scores = metric.compute(
                references=results['references'], 
                predictions=results['predictions']
            )
        elif metric_name == 'bert':
            scores = metric.compute(
                references=results['references'], 
                predictions=results['predictions'],
                model_type=EMBEDDER,
            )
        
        # Process scores
        if metric_name == 'wer_all':
            for k, v in scores.items():
                if k in ['wer', 'mer', 'wil', 'wip']:
                    scores_dic[k] = round(v * 100, 2)
        
        elif metric_name in ['wer', 'cer']:
            scores_dic[metric_name] = round(scores * 100, 2)
        
        elif metric_name == 'bert':
            for k, v in scores.items():
                if k != 'hashcode':
                    scores_dic[f'bert_{k}_mean'] = round(mean(scores[k]), 2)
                    scores_dic[f'bert_{k}_std'] = round(stdev(scores[k]), 3)
    
    return scores_dic


if __name__ == '__main__':
    results = {}

    results['references'] = ["this is the reference", "there is another one", " 能吞虾玻璃而 不霜身体啦"]
    results['predictions'] = ["this is the prediction", "there is an other sample", "我能吞下玻璃而不伤身体"]

    compute_metrics(results, 'wer_all')
    compute_metrics(results, ['wer_all'])
    compute_metrics(results, ['wer_all', 'cer'])
    compute_metrics(results, ['wer_all', 'cer', 'wer'])
    compute_metrics(results, ['wer_all', 'cer', 'bert'])
