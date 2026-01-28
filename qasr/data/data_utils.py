import torch
from datasets import load_dataset, Audio
from transformers import VoxtralProcessor, GraniteSpeechProcessor, \
    Qwen2AudioProcessor, Qwen2_5OmniProcessor
try:
    from transformers import AudioFlamingo3Processor
except:
    # placeholder way to make sure it is a lalm style processor
    AudioFlamingo3Processor = VoxtralProcessor

from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer


def is_target_text_in_range(ref):
    if ref.strip() == 'ignore time segment in scoring':
        return False
    else:
        return ref.strip() != ''


def get_text(sample):
    if 'text' in sample:
        return sample['text']
    elif 'sentence' in sample:
        return sample['sentence']
    elif 'normalized_text' in sample:
        return sample['normalized_text']
    elif 'transcript' in sample:
        return sample['transcript']
    elif 'transcription' in sample:
        return sample['transcription']
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
        batch['original_text'] = get_text(batch)
        batch['norm_text'] = normalizer(batch['original_text'])
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
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16_000))

    normalizer = make_normalizer()
    normalize = make_normalize_fn(normalizer)
    dataset = dataset.map(normalize)

    dataset = dataset.filter(is_target_text_in_range, input_columns=['norm_text'])
    return dataset, normalizer


def load_and_prepare_dataset(args, warmup=False):
    dataset = load_data(args)
    dataset, normalizer = prepare_data(dataset)

    if warmup:
        num = args.warmup_steps * args.batch_size
        if args.streaming:
            dataset = dataset.take(num)
        else:
            dataset = dataset.select(range(min(num, len(dataset))))

        return dataset, normalizer

    if args.max_eval_samples:
        print(f'Subsampling dataset to first {args.max_eval_samples} samples!')
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    return dataset, normalizer


# ================================
# Preprocessing
# ================================

def preprocess_batch(batch, processor, model, model_input_name, args):
    audios = [a['array'] for a in batch['audio']]
    minibatch_size = len(audios)

    # 1.1 Pad audios to max batch size if using torch compile to prevent re-compilations
    padding_size = None
    if minibatch_size != args.batch_size and args.torch_compile:
        padding_size = args.batch_size - minibatch_size
        audios.extend([audios[-1]] * padding_size)

    # This method handles the audio processing and creates proper input format
    if 'Voxtral' in args.model_id:
        # Use apply_transcription_request for Voxtral
        inputs = processor.apply_transcription_request(
            audio=audios,
            sampling_rate=16_000,
            # Set language for better accuracy
            language=args.force_asr_language if getattr(args, 'force_asr_language', None) else 'en',
            model_id=args.model_id,
            format=['WAV'] * len(audios),  # Voxtral needs to know what kind of inputs
            # device=args.device,
        )
    elif 'audio-flamingo' in args.model_id:
        inputs = processor.apply_transcription_request(
            audio=audios,
        )
    elif 'Qwen2.5-Omni' in args.model_id:
        inputs = processor(
            text=processor.prompt_asr,
            # text=[processor.prompt_asr] * len(audios),
            audio=audios,
            # sampling_rate=16_000,
            return_tensors='pt',
            padding=True,
        )
    elif 'Qwen2-Audio' in args.model_id:
        inputs = processor(
            text=[processor.prompt_asr] * len(audios),
            audio=audios,
            sampling_rate=16_000,
            return_tensors='pt',
            padding=True,
        )
    elif 'Phi4' in args.model_id:
        inputs = processor(
            text=[processor.prompt_asr] * len(audios),
            audios=audios,
            return_tensors='pt',
            # device=args.device,
        )
    elif 'granite' in args.model_id:
        inputs = processor(
            [processor.prompt_asr] * len(audios),
            audios,
            return_tensors='pt',
            device=args.device,
        )
    elif 'moonshine' in args.model_id:
        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors='pt',
            padding=True,
        )
        inputs['audios'] = audios
    elif not model.can_generate():
        # 1.2 Either CTC pre-processing (normalize to mean 0, std 1), or long-form Whisper processing
        inputs = processor(
            audios,
            sampling_rate=16_000,
            truncation=False,
            padding='longest',
            return_tensors='pt',
            return_attention_mask=True,
        )
    else:
        # 1.3 Standard Whisper processing: pad audios to 30-seconds and converted to log-mel
        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors='pt',
            device=args.device,
        )

    dtype = getattr(torch, args.data_dtype, None) if getattr(args, 'data_dtype', None) else torch.float32

    inputs = inputs.to(args.device)
    # if 'Qwen2-Audio' in args.model_id or 'Qwen2.5-Omni' in args.model_id:
    if 'Qwen2' in args.model_id or 'audio-flamingo' in args.model_id:
        inputs['input_features'] = inputs['input_features'].to(dtype)
    else:
        inputs[model_input_name] = inputs[model_input_name].to(dtype)

    return inputs, padding_size, minibatch_size


# ================================
# Post-processing
# ================================

def postprocess_predictions(pred_ids, padding_size, inputs, processor, normalizer):
    # 3.1 Strip padded ids from predictions
    if padding_size is not None:
        pred_ids = pred_ids[:-padding_size, ...]

    # 3.2 Convert token ids to text transcription

    '''
    # phi4
    if type(processor) == Phi4Processor:
        # Gather the sequence index of the stop token
        stop_tokens_idx = gen_kwargs["stopping_criteria"][0].stop_tokens_idx.reshape(inputs.input_ids.shape[0], -1)[:, 0]

        # If a stop token was produced, we need to remove its length from the found index,
        # however there might be a chance that the stop token was not produced and the index
        # returned is the length of the generated sequence
        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - processor.stop_tokens_ids.shape[-1],
            pred_ids.shape[-1],
        )

        # Convert token ids to text transcription
        pred_text = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(pred_ids, stop_tokens_idx)
        ]
    '''

    if type(processor) in [AudioFlamingo3Processor]:
        texts = processor.batch_decode(
            pred_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            strip_prefix=True,
        )

    elif type(processor) in [VoxtralProcessor, GraniteSpeechProcessor, Qwen2AudioProcessor, Qwen2_5OmniProcessor]:
        # Decode predictions - skip the prompt tokens
        # Voxtral includes prompt tokens in output, so we slice from input_ids length
        texts = processor.batch_decode(
            pred_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
            # add_special_tokens=False,
        )

    else:
        texts = processor.batch_decode(pred_ids, skip_special_tokens=True)


    # normalize transcriptions with English normalizer
    preds = [normalizer(t) for t in texts]
    return preds
