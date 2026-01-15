import torch

from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    VoxtralForConditionalGeneration,
)

# ================================
# Model & Processor
# ================================

def load_model_and_processor(args):
    config = AutoConfig.from_pretrained(args.model_id)

    if 'Voxtral' in args.model_id:
        cls = VoxtralForConditionalGeneration
    elif type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING:
        cls = AutoModelForSpeechSeq2Seq
    else:
        cls = AutoModelForCTC
    # cls = AutoModelForSpeechSeq2Seq if type(config) in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING else AutoModelForCTC

    dtype = getattr(torch, args.model_dtype, 'auto')
    # if args.model_dtype == 'bfloat16':
    #     model_dtype = torch.bfloat16
    # elif args.model_dtype == 'float16':
    #     model_dtype = torch.float16
    # elif args.model_dtype == 'float32':
    #     model_dtype = torch.float
    # else:
    #     model_dtype = 'auto'

    # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    model = cls.from_pretrained(
        args.model_id,
        # default is torch.float (fp32), others: torch.float16/bfloat16
        torch_dtype=dtype,
        # https://huggingface.co/docs/transformers/en/attention_interface
        # sdpa uses pytorch default, can autotune with context manager
        attn_implementation='sdpa',
        # for large models that need to be split
        # device map can be cpu, cuda:1
        # device_map='auto',
        # tp_plan='auto',
        # a dic to be used with bitsandbytes or gptq
        # quantization_config=QUantizationConfigMixin, Dict
    ).to(args.device)
    # to use device_map='auto' need to install accelerate
    # .to(args.device)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model_input_name = processor.model_input_names[0]

    gen_kwargs = None
    if model.can_generate():
        # gen_kwargs = {'max_new_tokens': args.max_new_tokens}

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
