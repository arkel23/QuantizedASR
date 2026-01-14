import argparse
import os
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoProcessor, VoxtralForConditionalGeneration
import evaluate
import time
from tqdm import tqdm
import wandb

from normalizer import data_utils


wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')

def main(args):
    if args.wandb_project and args.wandb_entity:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        wandb.run.name = f'{args.model_id}_{args.dataset}_{args.split}'

    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device if args.device != -1 else "cpu"
    )

    # if args.torch_compile:
    #     model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
    #     if model.can_generate():
    #         # enable static k/v cache for autoregressive models
    #         model.generation_config.cache_implementation = "static"

    # Set generation parameters
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,  # Greedy decoding for deterministic transcription
        "num_beams": 1,  # Greedy search
    }

    def benchmark(batch, **kwargs):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # 1. Pre-Processing
        # Pad audios to max batch size if needed for consistent batching
        padding_size = None
        if minibatch_size != args.batch_size and args.batch_size > minibatch_size:
            padding_size = args.batch_size - minibatch_size
            padding_audios = [audios[-1] for _ in range(padding_size)]
            audios.extend(padding_audios)

        # Use apply_transcription_request for Voxtral
        # This method handles the audio processing and creates proper input format
        inputs = processor.apply_transcription_request(
            audio=audios,
            language="en",  # Set language for better accuracy
            model_id=args.model_id,
            format=["WAV"] * len(audios),  # Voxtral needs to know what kind of inputs
            sampling_rate=16000,
        )

        # Move inputs to device
        device = f"cuda:{args.device}" if args.device != -1 else "cpu"
        inputs = inputs.to(device, dtype=torch.bfloat16)

        # 2. Model Inference - Generate transcription
        # with sdpa_kernel(SDPBackend.MATH if args.torch_compile else SDPBackend.FLASH_ATTENTION):
        #     with torch.no_grad():
        #         pred_ids = model.generate(**inputs, **gen_kwargs)
        with torch.no_grad():
            pred_ids = model.generate(**inputs, **gen_kwargs)

        # 3. Post-processing
        # Strip padded predictions if we added padding
        if padding_size is not None:
            pred_ids = pred_ids[:-padding_size, ...]

        # Decode predictions - skip the prompt tokens
        # Voxtral includes prompt tokens in output, so we slice from input_ids length
        pred_text = processor.batch_decode(
            pred_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        # END TIMING
        runtime = time.time() - start_time

        # Normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # Normalize transcriptions with English normalizer
        batch["predictions"] = [data_utils.normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    # Warmup phase
    if args.warmup_steps is not None and args.warmup_steps > 0:
        dataset = data_utils.load_data(args)
        dataset = data_utils.prepare_data(dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))
        
        warmup_dataset = iter(warmup_dataset.map(
            benchmark, 
            batch_size=args.batch_size, 
            batched=True
        ))

        for _ in tqdm(warmup_dataset, desc="Warming up..."):
            continue

    # Load evaluation dataset
    dataset = data_utils.load_data(args)
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))
    
    dataset = data_utils.prepare_data(dataset)

    # Run evaluation
    dataset = dataset.map(
        benchmark, 
        batch_size=args.batch_size, 
        batched=True, 
        remove_columns=["audio"],
    )

    # Collect all results
    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    # Calculate and print metrics
    wer = wer_metric.compute(
        references=all_results["references"], 
        predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 
        2
    )
    print("WER:", wer, "%", "RTFx:", rtfx)

    if args.wandb_project and args.wandb_entity:
        wandb.log({'wer': wer, 'rtfx': rtfx})
        wandb.finish()

    return 0

def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default='mistralai/Voxtral-Mini-3B-2507',
        help="Model identifier. Should be a Voxtral model from 🤗 Transformers (e.g., mistralai/Voxtral-Mini-3B-2507)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hf-audio/esb-datasets-test-only-sorted",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='voxpopuli',
        help="Dataset name. *E.g.* `'librispeech_asr'` for the LibriSpeech ASR dataset, or `'common_voice'` for Common Voice.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation'` for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for transcription.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to JIT compile the forward pass of the model.",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="max-autotune",
        help="Mode for torch compiling model forward pass. Can be either 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs'.",
    )
    parser.add_argument(
        '--wandb_project', 
        type=str, 
        default='OpenASR',
        help="Weights & Biases project name for logging"
    )
    parser.add_argument(
        '--wandb_entity', 
        type=str, 
        default='nycu_pcs',
        help="Weights & Biases entity name for logging"
    )

    parser.set_defaults(streaming=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = load_args()

    main(args)
