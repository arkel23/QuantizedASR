import argparse
import os
import time
import torch
import evaluate
import wandb
from tqdm import tqdm
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
)
from qasr.data.data_utils import load_and_prepare_dataset
from qasr.eval.eval_utils import write_manifest


torch.set_float32_matmul_precision("high")


# ================================
# Model & Processor
# ================================

def load_model_and_processor(args):
    config = AutoConfig.from_pretrained(args.model_id)
    cls = AutoModelForSpeechSeq2Seq if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else AutoModelForCTC

    model = cls.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model_input_name = processor.model_input_names[0]

    gen_kwargs = None
    if model.can_generate():
        gen_kwargs = {"max_new_tokens": args.max_new_tokens}
        # for multilingual Whisper-checkpoints we see a definitive WER boost by setting the language and task args
        if getattr(model.generation_config, "is_multilingual", False):
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"
    elif args.max_new_tokens:
        raise ValueError("max_new_tokens is only valid for seq2seq models")

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = "static"

    return model, processor, model_input_name, gen_kwargs


# ================================
# Preprocessing
# ================================

def preprocess_batch(batch, processor, model, model_input_name, args):
    audios = [a["array"] for a in batch["audio"]]
    minibatch_size = len(audios)

    # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
    padding_size = None
    if minibatch_size != args.batch_size and args.torch_compile:
        padding_size = args.batch_size - minibatch_size
        audios.extend([audios[-1]] * padding_size)

    if not model.can_generate():
        # 1.2 Either CTC pre-processing (normalize to mean 0, std 1), or long-form Whisper processing
        inputs = processor(
            audios,
            sampling_rate=16_000,
            truncation=False,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
    else:
        # 1.3 Standard Whisper processing: pad audios to 30-seconds and converted to log-mel
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", device=args.device)

    inputs = inputs.to(args.device)
    inputs[model_input_name] = inputs[model_input_name].to(torch.bfloat16)

    return inputs, padding_size, minibatch_size


# ================================
# Inference
# ================================

def run_inference(model, inputs, gen_kwargs, args, min_new_tokens=None):
    with sdpa_kernel(SDPBackend.MATH if args.torch_compile else SDPBackend.FLASH_ATTENTION):
        if model.can_generate():
            # 2.1 Auto-regressive generation for encoder-decoder models
            return model.generate(**inputs, **gen_kwargs, min_new_tokens=min_new_tokens)
        else:
            # 2.2. Single forward pass for CTC
            with torch.no_grad():
                logits = model(**inputs).logits
                return logits.argmax(-1)


# ================================
# Post-processing
# ================================

def postprocess_predictions(pred_ids, padding_size, processor, normalizer):
    # 3.1 Strip padded ids from predictions
    if padding_size is not None:
        pred_ids = pred_ids[:-padding_size, ...]

    # 3.2 Convert token ids to text transcription
    texts = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # normalize transcriptions with English normalizer
    preds = [normalizer(t) for t in texts]
    return preds


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
        preds = postprocess_predictions(pred_ids, padding_size, processor, normalizer)

        runtime = time.time() - start

        batch["predictions"] = preds
        batch["references"] = batch["norm_text"]
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        return batch

    return benchmark


# ================================
# Warmup
# ================================

def run_warmup(dataset, benchmark, args):
    num = args.warmup_steps * args.batch_size
    if args.streaming:
        warmup = dataset.take(num)
    else:
        warmup = dataset.select(range(min(num, len(dataset))))

    warmup = iter(
        warmup.map(
            benchmark,
            batch_size=args.batch_size,
            batched=True,
            fn_kwargs={"min_new_tokens": args.max_new_tokens},
        )
    )

    for _ in tqdm(warmup, desc="Warming up..."):
        pass


# ================================
# Evaluation Loop
# ================================

def evaluate_dataset(dataset, benchmark, args):
    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }

    for sample in tqdm(iter(dataset), desc="Samples..."):
        for k in results:
            results[k].append(sample[k])

    return results


# ================================
# Metrics & Logging
# ================================

def compute_and_log_metrics(results, args):
    manifest = write_manifest(
        results["references"],
        results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=results["audio_length_s"],
        transcription_time=results["transcription_time_s"],
    )

    wer_metric = evaluate.load("wer")
    wer = round(100 * wer_metric.compute(
        references=results["references"], predictions=results["predictions"]
    ), 2)

    rtfx = round(
        sum(results["audio_length_s"]) / sum(results["transcription_time_s"]), 2
    )

    print("Results:", os.path.abspath(manifest))
    print("WER:", wer, "%  RTFx:", rtfx)
    wandb.log({"wer": wer, "rtfx": rtfx})

    return 0


# ================================
# Main
# ================================

def main(args):
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    wandb.run.name = f"{args.model_id}_{args.dataset}_{args.split}_{args.serial}"

    model, processor, model_input_name, gen_kwargs = load_model_and_processor(args)

    dataset, normalizer = load_and_prepare_dataset(args)

    benchmark = make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args)

    if args.warmup_steps:
        run_warmup(dataset, benchmark, args)

    dataset, normalizer = load_and_prepare_dataset(args)
    if args.max_eval_samples:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    results = evaluate_dataset(dataset, benchmark, args)
    compute_and_log_metrics(results, args)

    wandb.finish()
    return 0

# ================================
# CLI
# ================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', type=int, default=0)

    parser.add_argument("--model_id", type=str, default='openai/whisper-tiny.en')
    parser.add_argument("--max_new_tokens", type=int, default=None)

    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, default='ami')
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="max-autotune")

    parser.add_argument("--wandb_project", type=str, default="OpenASR")
    parser.add_argument("--wandb_entity", type=str, default="nycu_pcs")

    parser.set_defaults(streaming=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
