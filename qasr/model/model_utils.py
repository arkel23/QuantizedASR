import torch

from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    VoxtralForConditionalGeneration,
    BitsAndBytesConfig,
)


def get_dtype_quantization_config(args):
    # default is torch.float (fp32), others: torch.float16/bfloat16
    model_dtype = getattr(torch, args.model_dtype, 'auto')
    # if args.model_dtype == 'bfloat16':
    #     model_dtype = torch.bfloat16
    # elif args.model_dtype == 'float16':
    #     model_dtype = torch.float16
    # elif args.model_dtype == 'float32':
    #     model_dtype = torch.float
    # else:
    #     model_dtype = 'auto'

    act_dtype = getattr(torch, args.act_dtype, torch.float32)

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
    config = AutoConfig.from_pretrained(args.model_id)

    if 'Voxtral' in args.model_id:
        cls = VoxtralForConditionalGeneration
    elif type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        cls = AutoModelForSpeechSeq2Seq
    else:
        cls = AutoModelForCTC

    model_dtype, quantization_config = get_dtype_quantization_config(args)

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
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    model_input_name = processor.model_input_names[0]

    gen_kwargs = None
    if model.can_generate():
        # Set generation parameters
        gen_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': False,  # Greedy decoding for deterministic transcription
            'num_beams': 1,  # Greedy search
        }

        # for multilingual Whisper-checkpoints we see a definitive WER boost by setting the language and task args
        if getattr(model.generation_config, 'is_multilingual', False):
            gen_kwargs['language'] = 'en'
            gen_kwargs['task'] = 'transcribe'

    elif args.max_new_tokens:
        raise ValueError('max_new_tokens is only valid for seq2seq models')

    if args.torch_compile:
        model.forward = torch.compile(model.forward, mode=args.compile_mode, fullgraph=True)
        if model.can_generate():
            # enable static k/v cache for autoregressive models
            model.generation_config.cache_implementation = 'static'

    return model, processor, model_input_name, gen_kwargs
