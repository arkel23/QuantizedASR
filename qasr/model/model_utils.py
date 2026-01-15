import torch

from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoProcessor,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
)

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

