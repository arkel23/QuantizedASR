import torch
from datasets import load_dataset, Audio

from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def make_normalizer(english=True):
    if english:
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicMultilingualTextNormalizer()
    return normalizer


def make_normalize_fn(normalizer):
    def normalize(batch):
        batch["original_text"] = get_text(batch)
        batch["norm_text"] = normalizer(batch["original_text"])
        return batch
    return normalize


def load_data(args):
    dataset = load_dataset(
        args.dataset_path,
        args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )
    return dataset


def prepare_data(dataset):
    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    normalizer = make_normalizer()
    normalize = make_normalize_fn(normalizer)
    dataset = dataset.map(normalize)

    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])
    return dataset, normalizer


def load_and_prepare_dataset(args):
    dataset = load_data(args)
    dataset, normalizer = prepare_data(dataset)
    return dataset, normalizer


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
