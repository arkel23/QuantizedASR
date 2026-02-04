import matplotlib.pyplot as plt
import re

from qasr.misc.misc_utils import parse_args, init_procedure
from qasr.model.model_utils import load_model_and_processor
from qasr.model.quant_utils import quantization_calibration
from qasr.data.data_utils import load_and_prepare_dataset
from qasr.eval.eval_utils import make_benchmark_fn


def plot_model_weights(
        weights, matched_layers, model_id=None, layer_pattern=None,
        output_path="weights_distribution.png"):
    """
    Plot weight distribution of a HuggingFace model
    
    Args:
        model_id: HuggingFace model ID
        output_path: Where to save the plot
        layer_pattern: None (all layers), exact name ("encoder.layers.0.self_attn"), 
                      or regex pattern (r".*self_attn.*")
    """

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=100, edgecolor='black', alpha=0.7, log=True)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency (log scale)')
    
    # Title shows what was plotted
    if layer_pattern is None:
        title = f'All Weights: {model_id}'
    elif len(matched_layers) == 1:
        title = f'Layer: {matched_layers[0]}'
    else:
        title = f'Pattern "{layer_pattern}": {len(matched_layers)} layers'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matched {len(matched_layers)} layers")
    print(f"Saved to {output_path}")
    return matched_layers


def match_layers(model, layer_pattern):
    # Collect weights based on pattern
    weights = []
    matched_layers = []
    
    for name, param in model.named_parameters():
        # No pattern: collect all
        if layer_pattern is None:
            weights.extend(param.data.cpu().flatten().numpy())
            matched_layers.append(name)
            fn = 'weights_distribution'
        # Exact match
        elif layer_pattern == name:
            weights.extend(param.data.cpu().flatten().numpy())
            matched_layers.append(name)
            fn = name
            break  # Found exact match, stop
        # Regex pattern
        elif re.search(layer_pattern, name):
            weights.extend(param.data.cpu().flatten().numpy())
            matched_layers.append(name)
            fn = layer_pattern

    if not weights:
        raise ValueError(f"No layers matched pattern: {layer_pattern}")

    return matched_layers, fn


def setup_env():
    args = parse_args()

    init_procedure(args, use_wandb=False)

    model, processor, model_input_name, gen_kwargs = load_model_and_processor(args)

    dataset, normalizer = load_and_prepare_dataset(args, warmup=True)

    benchmark = make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args)

    if args.quant_config == 'quanto' and args.quant_dtype_acts is not None:
        quantization_calibration(dataset, benchmark, model, args)

    weights, matched_layers, fn = match_layers(model, args.vis_layer_pattern)

    return args, weights, matched_layers, fn


if __name__ == '__main__':
    # Usage examples:

    args, weights, matched_layers, fn = setup_env()

    plot_model_weights(
        weights, matched_layers, model_id=args.model_id,
        layer_pattern=args.vis_layer_pattern,
        output_path=f"{fn}.png")

    '''
    # All weights
    plot_model_weights("openai/whisper-tiny.en")

    # Exact layer name
    plot_model_weights(
        "openai/whisper-tiny.en", 
        layer_pattern="encoder.layers.0.self_attn.q_proj.weight",
        output_path="layer_0_attn.png"
    )

    # Regex: all attention layers
    plot_model_weights(
        "openai/whisper-tiny.en",
        layer_pattern=r".*self_attn.*",
        output_path="all_attention.png"
    )

    # Regex: all encoder layers
    plot_model_weights(
        "openai/whisper-tiny.en",
        layer_pattern=r"^encoder\..*",
        output_path="encoder_only.png"
    )

    # Regex: specific layer numbers
    plot_model_weights(
        "openai/whisper-tiny.en",
        layer_pattern=r"layers\.[0-2]\..*",  # First 3 layers only
        output_path="first_3_layers.png"
    )
    '''
