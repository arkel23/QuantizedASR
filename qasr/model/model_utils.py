import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    # Qwen2_5OmniProcessor,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    VoxtralForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)

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


def get_dtype_quantization_config(args):
    model_dtype = getattr(torch, args.model_dtype, 'auto') if getattr(args, 'model_dtype', None) else 'auto'

    act_dtype = getattr(torch, args.act_dtype, None) if getattr(args, 'act_dtype', None) else torch.float32

    quantization_config=None
    if args.quant_config == 'bnb' and args.quant_dtype_weights == '8bit':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=getattr(args, 'bnb_int8_threshold', 6.0),
        )
    elif args.quant_config == 'bnb' and args.quant_dtype_weights == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=act_dtype,
            bnb_4bit_use_double_quant=getattr(args, 'bnb_4bit_use_double_quant', False),
            bnb_4bit_quant_type=getattr(args, 'bnb_4bit_quant_type', 'fp4'), # can be fp4 or nf4
        )

    return model_dtype, quantization_config


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
        raise NotImplementedError
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

    # elif args.max_new_tokens:
    #     raise ValueError('max_new_tokens is only valid for seq2seq models')

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = 'static'

    return model, processor, model_input_name, gen_kwargs
