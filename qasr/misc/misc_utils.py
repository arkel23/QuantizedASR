import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--model_id', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                         help='Max num of tokens to generate')
    parser.add_argument('--max_think_tokens', type=int, default=256,
                         help='Max num of tokens to generate')
    parser.add_argument('--model_dtype', type=str, default='float32',
                        choices=['auto', 'bfloat16', 'float16', 'float32'])
    parser.add_argument('--force_asr_language', type=str, default=None)

    parser.add_argument('--quant_config', type=str, default=None,
                        choices=['bnb'])
    parser.add_argument('--quant_dtype_weights', type=str, default=None)

    # bitsandbytes
    parser.add_argument('--bnb_int8_threshold', type=float, default=6.0)
    parser.add_argument('--bnb_4bit_use_double_quant', action='store_true')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='fp4',
                        choices=['fp4', 'nf4'])

    parser.add_argument('--dataset_path', type=str, default='hf-audio/esb-datasets-test-only-sorted')
    parser.add_argument('--dataset', type=str, default='voxpopuli')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--data_dtype', type=str, default='float32',
                         choices=['bfloat16', 'float16', 'float32'])

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_eval_samples', type=int, default=None)
    parser.add_argument('--no_streaming', dest='streaming', action='store_false')

    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--compile_mode', type=str, default='max-autotune')
    parser.add_argument('--flash_attn', action='store_true',
                        help='only available for 5090/h100 with certain models')
    parser.add_argument('--act_dtype', type=str, default=None,
                        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--float32_matmul_prec', type=str, default=None,
                        choices=['medium', 'high', 'highest'])

    parser.add_argument('--warmup_steps', type=int, default=10)

    parser.add_argument('--wandb_project', type=str, default='OpenASR')
    parser.add_argument('--wandb_entity', type=str, default='nycu_pcs')

    parser.set_defaults(streaming=True)
    args = parser.parse_args()

    args.device = f"cuda:{args.device}" if args.device != -1 else "cpu"

    if any([model_family in args.model_id for model_family in ['Voxtral', 'Qwen', 'granite', 'flamingo']]) and args.max_new_tokens is None:
        args.max_new_tokens = 200

    if 'Qwen2.5-Omni' in args.model_id:
        args.batch_size = 1

    return args
