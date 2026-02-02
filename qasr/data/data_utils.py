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


def prepare_filter_language(target_language='en'):
    target_language = 'eng' if target_language == 'en' else target_language

    def is_language(language):
        return language == target_language

    return is_language


# filter data that is shorter than min_input_length or longer than
# max_input_length
def is_audio_in_length_range(length, min_input_length, max_input_length):
    return length > min_input_length and length < max_input_length


def is_target_text_in_range(ref):
    if ref.strip() == 'ignore time segment in scoring':
        return False
    else:
        return ref.strip() != ''

def get_audio(sample):
    if 'audio' in sample:
        return sample['audio']
    # adi-gov-tw
    elif 'mp3' in sample:
        return sample.pop('mp3')
    # asr
    elif 'flac' in sample:
        return sample.pop('flac')
    else:
        raise ValueError(
            f"Expected transcript column of either 'audio', or 'mp3'. Got sample of "
            ".join{sample.keys()}. Ensure a audio column name is present in the dataset."
        )


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
    elif 'txt' in sample:
        return sample['txt']
    elif 'label' in sample:
        return sample['label']
    elif 'Digit' in sample:
        return sample['Digit']
    elif 'Text' in sample:
        return sample['Text']
    elif 'sentence' in sample:
        return sample['sentence']
    elif 'question_text' in sample:
        return sample['question_text']
    elif 'Questions' in sample:
        return sample['Questions']
    elif 'instruction' in sample:
        return sample['instruction']
    elif 'question' in sample:
        return sample['question']
    elif 'meta_info' in sample:
        return sample['meta_info']
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def make_normalizer(english=True):
    if english:
        normalizer = EnglishTextNormalizer()
    # if chinese hokkien need to decide if chinese characters or pinyin
    # eg 我沒咧驚（gu? b? leh kian)
    # if hsk2/3/4/5 need to remove the "女：", "男：", "问：" from 'text' fields
    # if hsk6 need to eliminate the punctuation such as "样子。”" in the text

    # for air-chat need to concatenate "transcription" key from a list of dictionaries in meta_info
    # {'meta_info': '[{"speaker": "speaker 1", "speaking time": "(0.00,13.39)", "transcription": "so it\'s a lot more than the like two thousand dollar computers that like you or i can afford you know [laughter] so ah they so they put a lot of effort and this is the kind of stuff that gets sold to the government"}, {"speaker": "speaker 2", "speaking time": "(5.33,6.88)", "transcription": "right right"}, {"speaker": "speaker 2", "speaking time": "(13.01,13.91)", "transcription": "mhm"}, {"speaker": "speaker 1", "speaking time": "(13.30,20.20)", "transcription": "and that\'s why and our government has a massive sum of money and that\'s why i"}, {"speaker": "speaker 2", "speaking time": "(19.29,21.60)", "transcription": "isn\'t it amazing how it has"}, {"speaker": "speaker 1", "speaking time": "(21.30,26.59)", "transcription": "i mean they they ah how what\'s the national debt six trillion plus"}, {"speaker": "speaker 2", "speaking time": "(25.88,28.08)", "transcription": "[laughter]"}]'
    else:
        normalizer = BasicMultilingualTextNormalizer()
    return normalizer


def make_normalize_fn(normalizer):
    def normalize(batch):
        batch['original_text'] = get_text(batch)
        batch['norm_text'] = normalizer(batch['original_text'])

        batch['audio'] = get_audio(batch)

        return batch
    return normalize


def load_data(
        dataset_path='hf-audio/esb-datasets-test-only',
        dataset_config='tedlium',
        split='test',
        streaming=True
        ):

    # filter based on language for floras
    if 'floras' in dataset_path and 'multilingual' in dataset_config:
        dataset = load_dataset(
            dataset_path,
            'multilingual',
            split=split,
            streaming=streaming,
            token=True,
        )

        is_language = prepare_filter_language(target_language=dataset_config.split('_')[-1])
        dataset = dataset.filter(is_language, input_columns=['language'])

    else:
        dataset = load_dataset(
            dataset_path,
            dataset_config,
            split=split,
            streaming=streaming,
            token=True,
        )

    if 'hf-audio' in dataset_path or '_en' in dataset_config or dataset_config == 'monolingual' or split == 'en':
        english = True
    else:
        english = False

    print(dataset_path, dataset, 'english dataset: ', english)

    return dataset, english


def prepare_data(dataset, english):
    # may need to coordinate with the audio_lengths function in preprocess_batch
    # also convert to a uniform format and may need to process from multichannel to single

    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16_000))

    normalizer = make_normalizer(english)
    normalize = make_normalize_fn(normalizer)
    # the map function can take num_proc to control number of workers
    dataset = dataset.map(normalize)

    dataset = dataset.filter(is_target_text_in_range, input_columns=['norm_text'])

    # in the future may want to use this to filter out samples
    # dataset = dataset.filter(is_audio_in_length_range, input_columns=['input_length'])

    return dataset, normalizer


def load_and_prepare_dataset(args, warmup=False):
    dataset, english = load_data(args.dataset_path, args.dataset, args.split, args.streaming)
    dataset, normalizer = prepare_data(dataset, english)

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

    # this assumes sampling rate is 16 kHz, if different then need flag
    ds_sampling_rate = getattr(args, 'ds_sampling_rate', 16_000)
    audio_lengths = [a.shape[0] / ds_sampling_rate for a in audios]

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
    elif not model.can_generate() or args.long_form:
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

    return inputs, padding_size, minibatch_size, audio_lengths


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
        stop_tokens_idx = gen_kwargs['stopping_criteria'][0].stop_tokens_idx.reshape(inputs.input_ids.shape[0], -1)[:, 0]

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
            processor.decode(_pred_ids[inputs['input_ids'].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
