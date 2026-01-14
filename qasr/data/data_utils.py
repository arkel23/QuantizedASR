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
