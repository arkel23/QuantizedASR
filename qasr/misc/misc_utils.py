import argparse

import yaml
import wandb
import torch


def load_config_from_yaml(yaml_path):
    """Load config from YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def merge_yaml_with_args(parser, yaml_config):
    """Merge YAML config with argparse, CLI args take precedence"""
    # Parse command line args
    args = parser.parse_args()
    
    # If no YAML provided, return CLI args
    if not yaml_config:
        return args
    
    # Convert args to dict
    args_dict = vars(args)
    
    # Update with YAML values only if not set via CLI
    defaults = vars(parser.parse_args([]))  # Get default values
    for key, value in yaml_config.items():
        # Only override if CLI arg is still at default value
        if key in args_dict and args_dict[key] == defaults.get(key):
            args_dict[key] = value
    
    return argparse.Namespace(**args_dict)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, 
                       help='Path to YAML config file')

    parser.add_argument('--serial', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['wer_all', 'bert'],
                        choices=['wer_all', 'wer', 'cer', 'bert'])

    parser.add_argument('--force_asr_language', type=str, default=None)
    parser.add_argument('--long_form', action='store_true',
                        help='no truncation and return_attention_mask for whisper > 30 s')
    parser.add_argument('--long_form_tricks', action='store_true',
                        help='from hf https://github.com/huggingface/transformers/pull/27658')

    parser.add_argument('--model_id', type=str, default='openai/whisper-tiny.en')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                         help='Max num of tokens to generate')
    parser.add_argument('--max_think_tokens', type=int, default=256,
                         help='Max num of tokens to generate')
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                        choices=['auto', 'bfloat16', 'float16', 'float32'])

    parser.add_argument('--quant_config', type=str, default=None,
                        choices=['bnb', 'quanto', 'hqq', 'torchao'])
    parser.add_argument('--quant_dtype_weights', type=str, default=None)
    parser.add_argument('--quant_dtype_acts', type=str, default=None)

    # bitsandbytes
    parser.add_argument('--bnb_int8_threshold', type=float, default=6.0)
    parser.add_argument('--bnb_4bit_use_double_quant', action='store_true')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default='fp4',
                        choices=['fp4', 'nf4'])

    parser.add_argument('--dataset_path', type=str, default='hf-audio/esb-datasets-test-only-sorted')
    parser.add_argument('--dataset', type=str, default='voxpopuli')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--data_dtype', type=str, default='bfloat16',
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

    # Load YAML if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        args = merge_yaml_with_args(parser, yaml_config)
    
    if torch.cuda.is_available() and args.device != '-1':
        args.device = f'cuda:{args.device}'
    else:
        args.device = 'cpu'

    if any([model_family in args.model_id for model_family in [
        'Voxtral', 'Qwen', 'granite', 'flamingo']]) and args.max_new_tokens is None:
        args.max_new_tokens = 200

    if 'Qwen2.5-Omni' in args.model_id:
        args.batch_size = 1

    return args


def init_procedure(args):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if getattr(args, 'float32_matmul_prec', None):
        torch.set_float32_matmul_precision(args.float32_matmul_prec)

    q_weights = f'_w{args.quant_dtype_weights}' if args.quant_dtype_weights else ''
    q_acts = f'_a{args.quant_dtype_acts}' if args.quant_dtype_acts else ''
    q_config = f'_{args.quant_config}' if args.quant_config else ''
    q_all = f'{q_config}{q_weights}{q_acts}'

    ds = f'{args.dataset_path}_{args.dataset}_{args.split}'

    args.run_name = f'{args.model_id}{q_all}_{ds}_{args.serial}'
    args.run_name_legacy = f'MODEL_{args.model_id}{q_all}_DATASET_{ds}_{args.serial}'

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    wandb.run.name = args.run_name

    return 0