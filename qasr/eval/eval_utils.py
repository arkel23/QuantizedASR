import os
import glob
import json
import time
from collections import defaultdict

import wandb
import evaluate
from tqdm import tqdm
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import StoppingCriteria, StoppingCriteriaList

from qasr.data.data_utils import preprocess_batch, postprocess_predictions


class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    # https://github.com/huggingface/open_asr_leaderboard/blob/main/phi/run_eval.py
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)


def count_params(model):
    return sum([p.numel() for p in model.parameters()])


def read_manifest(manifest_path: str):
    '''
    Reads a manifest file (jsonl format) and returns a list of dictionaries containing samples.
    '''
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) > 0:
                datum = json.loads(line)
                data.append(datum)
    return data


def write_manifest(
    references: list,
    transcriptions: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    audio_length: list = None,
    transcription_time: list = None,
):
    '''
    Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        references: Ground truth reference texts.
        transcriptions: Model predicted transcriptions.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.
        audio_length: Length of each audio sample in seconds.
        transcription_time: Transcription time of each sample in seconds.

    Returns:
        Path to the manifest file.
    '''
    model_id = model_id.replace('/', '-')
    dataset_path = dataset_path.replace('/', '-')
    dataset_name = dataset_name.replace('/', '-')

    if len(references) != len(transcriptions):
        raise ValueError(
            f'The number of samples in `references` ({len(references)}) '
            f'must match `transcriptions` ({len(transcriptions)}).'
        )

    if audio_length is not None and len(audio_length) != len(references):
        raise ValueError(
            f'The number of samples in `audio_length` ({len(audio_length)}) '
            f'must match `references` ({len(references)}).'
        )
    if transcription_time is not None and len(transcription_time) != len(references):
        raise ValueError(
            f'The number of samples in `transcription_time` ({len(transcription_time)}) '
            f'must match `references` ({len(references)}).'
        )

    audio_length = (
        audio_length if audio_length is not None else len(references) * [None]
    )
    transcription_time = (
        transcription_time
        if transcription_time is not None
        else len(references) * [None]
    )

    basedir = './results/'
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    manifest_path = os.path.join(
        basedir, f'MODEL_{model_id}_DATASET_{dataset_path}_{dataset_name}_{split}.jsonl'
    )

    with open(manifest_path, 'w', encoding='utf-8') as f:
        for idx, (text, transcript, audio_length, transcription_time) in enumerate(
            zip(references, transcriptions, audio_length, transcription_time)
        ):
            datum = {
                'audio_filepath': f'sample_{idx}',  # dummy value for Speech Data Processor
                'duration': audio_length,
                'time': transcription_time,
                'text': text,
                'pred_text': transcript,
            }
            f.write(f'{json.dumps(datum, ensure_ascii=False)}\n')
    return manifest_path


def score_results(directory: str, model_id: str = None):
    '''
    Scores all result files in a directory and returns a composite score over all evaluated datasets.

    Args:
        directory: Path to the result directory, containing one or more jsonl files.
        model_id: Optional, model name to filter out result files based on model name.

    Returns:
        Composite score over all evaluated datasets and a dictionary of all results.
    '''

    # Strip trailing slash
    if directory.endswith(os.pathsep):
        directory = directory[:-1]

    # Find all result files in the directory
    result_files = list(glob.glob(f'{directory}/**/*.jsonl', recursive=True))
    result_files = list(sorted(result_files))

    # Filter files belonging to a specific model id
    if model_id is not None and model_id != '':
        print('Filtering models by id:', model_id)
        model_id = model_id.replace('/', '-')
        result_files = [fp for fp in result_files if model_id in fp]

    # Check if any result files were found
    if len(result_files) == 0:
        raise ValueError(f'No result files found in {directory}')

    # Utility function to parse the file path and extract model id, dataset path, dataset name and split
    def parse_filepath(fp: str):
        model_index = fp.find('MODEL_')
        fp = fp[model_index:]
        ds_index = fp.find('DATASET_')
        model_id = fp[:ds_index].replace('MODEL_', '').rstrip('_')
        author_index = model_id.find('-')
        model_id = model_id[:author_index] + '/' + model_id[author_index + 1 :]

        ds_fp = fp[ds_index:]
        dataset_id = ds_fp.replace('DATASET_', '').rstrip('.jsonl')
        return model_id, dataset_id

    # Compute WER results per dataset, and RTFx over all datasets
    results = {}
    wer_metric = evaluate.load('wer')

    for result_file in result_files:
        manifest = read_manifest(result_file)
        model_id_of_file, dataset_id = parse_filepath(result_file)

        references = [datum['text'] for datum in manifest]
        predictions = [datum['pred_text'] for datum in manifest]

        time = [datum['time'] for datum in manifest]
        duration = [datum['duration'] for datum in manifest]
        compute_rtfx = all(time) and all(duration)

        wer = wer_metric.compute(references=references, predictions=predictions)
        wer = round(100 * wer, 2)

        if compute_rtfx:
            audio_length = sum(duration)
            inference_time = sum(time)
            rtfx = round(sum(duration) / sum(time), 4)
        else:
            audio_length = inference_time = rtfx = None

        result_key = f'{model_id_of_file} | {dataset_id}'
        results[result_key] = {'wer': wer, 'audio_length': audio_length, 'inference_time': inference_time, 'rtfx': rtfx}

    print('*' * 80)
    print('Results per dataset:')
    print('*' * 80)

    for k, v in results.items():
        metrics = f"{k}: WER = {v['wer']:0.2f} %"
        if v["rtfx"] is not None:
            metrics += f", RTFx = {v['rtfx']:0.2f}"
        print(metrics)

    # composite WER should be computed over all datasets and with the same key
    composite_wer = defaultdict(float)
    composite_audio_length = defaultdict(float)
    composite_inference_time = defaultdict(float)
    count_entries = defaultdict(int)
    for k, v in results.items():
        key = k.split('|')[0].strip()
        composite_wer[key] += v['wer']
        if v['rtfx'] is not None:
            composite_audio_length[key] += v['audio_length']
            composite_inference_time[key] += v['inference_time']
        else:
            composite_audio_length[key] = composite_inference_time[key] = None
        count_entries[key] += 1

    # normalize scores & print
    print()
    print('*' * 80)
    print('Composite Results:')
    print('*' * 80)
    for k, v in composite_wer.items():
        wer = v / count_entries[k]
        print(f'{k}: WER = {wer:0.2f} %')
    for k in composite_audio_length:
        if composite_audio_length[k] is not None:
            rtfx = composite_audio_length[k] / composite_inference_time[k]
            print(f'{k}: RTFx = {rtfx:0.2f}')
    print('*' * 80)
    return composite_wer, results


# ================================
# Inference
# ================================

def run_inference(model, inputs, gen_kwargs, args, min_new_tokens=None):
    dtype = getattr(torch, args.act_dtype, None) if getattr(args, 'act_dtype', None) else torch.float32

    if args.flash_attn:
        with torch.no_grad():
            with torch.autocast(device_type=model.device.type, dtype=dtype):
                with sdpa_kernel(SDPBackend.MATH if args.torch_compile else SDPBackend.FLASH_ATTENTION):
                    if model.can_generate():
                        # 2.1 Auto-regressive generation for encoder-decoder models
                        return model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)
                    else:
                        # 2.2. Single forward pass for CTC
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            return logits.argmax(-1)

    with torch.no_grad():
        with torch.autocast(device_type=model.device.type, dtype=dtype):
            if model.can_generate():
                if 'moonshine' in args.model_id:
                    # from moonshine/run_eval.py
                    # Create a mask for output tokens to limit length based on input audio clip length.
                    # Add 2 to token limits to account for <sot> and <eot>.
                    audios = inputs.pop('audios')
                    token_generation_limits = [len(clip) * 6.5 // 16000 + 2 for clip in audios]
                    max_new_tokens = torch.tensor(token_generation_limits).reshape((-1, 1)).to(model.device)
                    pred_ids = model.generate(**inputs, max_new_tokens=max_new_tokens.max())

                    output_mask = torch.arange(pred_ids.shape[-1]).repeat((pred_ids.shape[0], 1)).to(model.device)
                    output_mask = output_mask > max_new_tokens

                    eot_token = model.config.eos_token_id
                    pred_ids.masked_fill(output_mask, eot_token)

                    return pred_ids

                elif 'Qwen2.5-Omni' in args.model_id:
                    pred_ids = model.generate(
                        **inputs, **gen_kwargs, return_audio=False,
                        thinker_max_new_tokens=getattr(args, 'max_think_tokens', 256), thinker_do_sample=False
                    )
                    return pred_ids

                elif 'Phi4' in args.model_id:
                    stopping_criteria = StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(
                        gen_kwargs['stop_token_ids'],
                        batch_size=args.num_beams * inputs.input_ids.shape
                    )])
                    gen_kwargs['stopping_criteria'] = stopping_criteria

                return model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)

            else:
                # 2.2. Single forward pass for CTC
                with torch.no_grad():
                    logits = model(**inputs).logits
                    return logits.argmax(-1)


# ================================
# Benchmark function (HF map)
# ================================

def make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args):
    def benchmark(batch, min_new_tokens=None):
        start = time.time()

        inputs, padding_size, minibatch_size = preprocess_batch(
            batch, processor, model, model_input_name, args
        )

        pred_ids = run_inference(model, inputs, gen_kwargs, args, min_new_tokens)
        preds = postprocess_predictions(pred_ids, padding_size, inputs, processor, normalizer)

        runtime = time.time() - start

        batch['predictions'] = preds
        batch['references'] = batch['norm_text']
        batch['transcription_time_s'] = minibatch_size * [runtime / minibatch_size]

        return batch

    return benchmark


def run_warmup(dataset, benchmark, args):
    dataset = iter(
        dataset.map(
            benchmark,
            batch_size=args.batch_size,
            batched=True,
            # use max new tokens for early compilation / preparation for worst case
            # this causes problems with mistral tokenizer
            fn_kwargs={'min_new_tokens': args.max_new_tokens},
        )
    )

    for _ in tqdm(dataset, desc='Warming up...'):
        pass

    return 0


def quantization_calibration(dataset, benchmark, model, args):
    from optimum.quanto import Calibration, freeze

    print("Calibrating ...")
    with Calibration():
        dataset = iter(
            dataset.map(
                benchmark,
                batch_size=args.batch_size,
                batched=True,
            )
        )

        for _ in tqdm(dataset, desc='Warming up...'):
            pass

        # evaluate_model(model, processor, processed_dataset, wer, args.batch_size)

    freeze(model)

    return 0


def evaluate_dataset(dataset, benchmark, args):
    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=['audio'],
    )

    results = {
        'audio_length_s': [],
        'transcription_time_s': [],
        'predictions': [],
        'references': [],
    }

    for sample in tqdm(iter(dataset), desc='Samples...'):
        for k in results:
            results[k].append(sample[k])

    return results


def compute_and_log_metrics(results, model, args):
    manifest_path = write_manifest(
        results['references'],
        results['predictions'],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results['audio_length_s'],
        transcription_time=results['transcription_time_s'],
    )

    wer_metric = evaluate.load('wer')
    wer = round(100 * wer_metric.compute(
        references=results['references'], predictions=results['predictions']
    ), 2)

    rtfx = round(
        sum(results['audio_length_s']) / sum(results['transcription_time_s']), 2
    )

    no_params = round(count_params(model) / 1e6, 4)

    if torch.cuda.is_available():
        max_memory = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 4)
    else:
        max_memory = 0

    print('Results saved at path:', os.path.abspath(manifest_path))
    print('WER:', wer, '%  RTFx:', rtfx)

    wandb.log({
        'wer': wer, 'rtfx': rtfx,
        'max_memory': max_memory,
        'no_params': no_params
    })

    return 0
