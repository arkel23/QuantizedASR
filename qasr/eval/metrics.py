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

import evaluate

try:
    from .wer import WERMetrics
    from .tonal import TonalASRMetrics
except:
    from wer import WERMetrics
    from tonal import TonalASRMetrics

EMBEDDER = 'microsoft/deberta-large-mnli'


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
        elif metric_name == 'ter':
            metric = TonalASRMetrics()

        # Compute scores
        if metric_name in ['wer', 'cer', 'wer_all', 'ter']:
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
                    try:
                        scores_dic[f'bert_{k}_std'] = round(stdev(scores[k]), 3)
                    except:
                        scores_dic[f'bert_{k}_std'] = 0

        elif metric_name == 'ter':
            for k, v in scores.items():
                if k in ['confusion_matrix']:
                    scores_dic[k] = v
                else:
                    scores_dic[k] = round(v * 100, 2)

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

    results['references'] = ["我沒咧驚", "我发现了问题", "你好世界"]
    results['predictions'] = ["我沒驚", "我发线了问题", "你号世界"]
    scores_dic = compute_metrics(results, ['cer', 'ter', 'bert'])
    print(scores_dic)
