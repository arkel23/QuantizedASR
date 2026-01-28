import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    VoxtralForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    AutoModelForCausalLM,
    GenerationConfig,
)

from .quant_utils import get_dtype_quantization_config

try:
    from transformers import AudioFlamingo3ForConditionalGeneration
except:
    print('HF transformers>5 can use AudioFlamingo3')


def add_transcription_prompt_to_processor(processor, model_id, language='en'):
    if 'granite' in model_id:
        # create text prompt
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|>can you transcribe the speech into a written format?",
            }
        ]

        text = processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
        )

        processor.prompt_asr = text
    elif 'Qwen2-Audio' in model_id:
        # https://huggingface.co/Qwen/Qwen2-Audio-7B
        # https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/evaluate_asr.py
        # prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|{language}|>"
        processor.prompt_asr = prompt

    elif 'Qwen2.5-Omni' in model_id:
        # https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
        prompt = "Transcribe the English audio into text without any punctuation marks."
        system_prompt='You are a speech recognition model.'
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                    {"type": "audio", "audio": "audio_to_transcribe.wav"},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor.prompt_asr = text

    elif 'Phi4' in model_id:
        user = "<|user|>"
        assistant = "<|assistant|>"
        prompt_suffix = "<|end|>"
        user_prompt = "Transcribe the audio clip into text."

        prompt = f"{user}<|audio_1|>{user_prompt}{prompt_suffix}{assistant}"

        processor.prompt_asr = prompt

        stop_tokens = [prompt_suffix, processor.tokenizer.eos_token]
        stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]

        processor.stop_tokens_ids = stop_tokens_ids

    return processor


def prepare_processor(args):
    if 'lite-whisper' in args.model_id:
        # id = 'openai/whisper-large-v3-turbo' if 'turbo' in args.model_id else 'openai/whisper-large-v3'
        processor = AutoProcessor.from_pretrained('openai/whisper-large-v3-turbo', trust_remote_code=True)
    # elif 'Qwen2_Omni' in args.model_id:
    #     processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    else:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor = add_transcription_prompt_to_processor(processor, args.model_id, args.force_asr_language)
    return processor


def load_model_and_processor(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    model_dtype, quantization_config = get_dtype_quantization_config(args)


    if 'Voxtral' in args.model_id:
        cls = VoxtralForConditionalGeneration
    elif 'Qwen2.5-Omni' in args.model_id:
        cls = Qwen2_5OmniForConditionalGeneration
    elif 'Qwen2-Audio' in args.model_id:
        cls = Qwen2AudioForConditionalGeneration
    elif 'audio-flamingo' in args.model_id:
        cls = AudioFlamingo3ForConditionalGeneration
    elif 'Phi4' in args.model_id:
        cls = AutoModelForCausalLM
    elif 'lite-whisper' in args.model_id:
        cls = AutoModel
    elif type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        cls = AutoModelForSpeechSeq2Seq
    else:
        cls = AutoModelForCTC

    if 'granite' in args.model_id:
        model = cls.from_pretrained(
            args.model_id,
            torch_dtype=model_dtype, # also known as dtype in transformers > 5
            device_map='auto',
            quantization_config=quantization_config,
        )
    elif 'Phi4' in args.model_id:
        raise NotImplementedError
    else:
        # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        model = cls.from_pretrained(
            args.model_id,
            torch_dtype=model_dtype, # also known as dtype in transformers > 5
            # https://huggingface.co/docs/transformers/en/attention_interface
            # sdpa uses pytorch default, can autotune with context manager
            attn_implementation='sdpa',
            # for large models that need to be split
            # device map can be cpu, cuda:1
            device_map='auto',
            # tp_plan='auto',
            # a dic to be used with bitsandbytes or gptq
            quantization_config=quantization_config,
            trust_remote_code=True,
        )


    processor = prepare_processor(args)
    model_input_name = processor.model_input_names[0]


    gen_kwargs = None
    if model.can_generate():
        # Set generation parameters
        gen_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': False,  # Greedy decoding for deterministic transcription
            'num_beams': 1,  # Greedy search
        }

        if 'granite' in args.model_id:
            gen_kwargs.update({
                'bos_token_id': processor.tokenizer.bos_token_id,
                'pad_token_id': processor.tokenizer.pad_token_id,
                'eos_token_id': processor.tokenizer.eos_token_id,
                # 'repetition_penalty': 1.0, 1.0 means no penalty
            })

        if 'Phi4' in args.model_id:
            stop_token_ids = processor.stop_tokens_ids.to(model.device)
            gen_kwargs.update({
                'stop_tokens_ids': stop_token_ids,
                'ad_token_id': processor.tokenizer.pad_tokenizer_id,
                'eos_token_id': processor.tokenizer.eos_token_id,
            })

        if args.force_asr_language:
            gen_kwargs['language'] = args.force_asr_language
            gen_kwargs['task'] = 'transcribe'
            gen_kwargs['generation_config'] = GenerationConfig.from_pretrained(args.model_id)

        # for multilingual Whisper-checkpoints we see a definitive WER boost by setting the language and task args
        if getattr(model.generation_config, 'is_multilingual', False):
            gen_kwargs['language'] = 'en'
            gen_kwargs['task'] = 'transcribe'

        if 'lite-whisper' in args.model_id:
            gen_kwargs['language'] = 'en'
            gen_kwargs['task'] = 'transcribe'
            gen_kwargs['generation_config'] = GenerationConfig.from_pretrained("openai/whisper-large-v3-turbo")

        if args.quant_config == 'quanto' and args.quant_dtype_acts is not None:
            gen_kwargs['supress_tokens'] = []
            gen_kwargs['begin_suppress_tokens'] = []
            gen_kwargs['forced_decoder_ids'] = None


    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = 'static'


    return model, processor, model_input_name, gen_kwargs
