from tqdm import tqdm
import torch
from transformers import (
    BitsAndBytesConfig,
    QuantoConfig,
    HqqConfig,
    TorchAoConfig,
)

try:
    from torchao.quantization import (
        IntxWeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        Float8WeightOnlyConfig,
        Int8DynamicActivationInt4WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
        Float8DynamicActivationFloat8WeightConfig
    )
except:
    print('torchao not installed')

try:
    from optimum.quanto import (
        Calibration,
        quantize,
        freeze,
        qint2,
        qint4,
        qint8,
        qfloat8,
    )
except:
    print('optimum-quanto not installed')


def get_dtype_quantization_config(args):
    model_dtype = getattr(torch, args.model_dtype, 'auto') if getattr(args, 'model_dtype', None) else 'auto'

    act_dtype = getattr(torch, args.act_dtype, None) if getattr(args, 'act_dtype', None) else torch.float32

    quantization_config=None
    if args.quant_config == 'bnb' and args.quant_dtype_weights == '8':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=getattr(args, 'bnb_int8_threshold', 6.0),
        )
    elif args.quant_config == 'bnb' and args.quant_dtype_weights == '4':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=act_dtype,
            bnb_4bit_use_double_quant=getattr(args, 'bnb_4bit_use_double_quant', False),
            bnb_4bit_quant_type=getattr(args, 'bnb_4bit_quant_type', 'fp4'), # can be fp4 or nf4
        )

    elif args.quant_config == 'quanto' and args.quant_dtype_acts is None:
        # weights can be None, int2, int4, int8, float8
        # acts can be None, int8, float8
        # acts need to use quanto library directly
        quantization_config = QuantoConfig(
            weights=args.quant_dtype_weights,
        )

    elif args.quant_config == 'hqq':
        assert args.quant_dtype_weights.isnumeric(), 'hqq requires numeric quant_dtype_weights'

        quantization_config = HqqConfig(
            # can be 1, 2, 3, 4, or 8 bits
            nbits=int(args.quant_dtype_weights),
            group_size=getattr(args, 'quant_group_size', 64),
            # view_as_float if True quantized param is viewed as float instead of int
            # can specify specific layers
            # dynamic_config={
            # 'self_attn.q_proj': q4_config,
            # 'mlp.up_proj': q3_config,
            # }
            # https://github.com/dropbox/hqq/blob/master/examples/models/whisper.py
        )

    elif args.quant_config == 'torchao':

        # in active development
        # https://huggingface.co/docs/transformers/main/quantization/torchao

        # it has support for regex matching for layers for specific configs
        # set default to int4 (for linears), and skip quantizing `model.layers.0.self_attn.q_proj`
        # quant_config = FqnToConfig({"_default": config, "model.layers.0.self_attn.q_proj": None})

        # from torchao.dtypes import MarlinSparseLayout
        # quant_config = Int4WeightOnlyConfig(layout=MarlinSparseLayout())

        if args.quant_dtype_weights == 'int4' and args.quant_dtype_acts == 'int8':
            quantization_config = TorchAoConfig(quant_type=Int8DynamicActivationInt4WeightConfig())
        elif args.quant_dtype_weights == 'int8' and args.quant_dtype_acts == 'int8':
            quantization_config = TorchAoConfig(quant_type=Int8DynamicActivationInt8WeightConfig())
        elif args.quant_dtype_weights == 'float8' and args.quant_dtype_acts == 'float8':
            quantization_config = TorchAoConfig(quant_type=Float8DynamicActivationFloat8WeightConfig())
        elif args.quant_dtype_weights == 'int4':
            # requires fpgemm-gpu-genai >= 1.2.0
            quantization_config = TorchAoConfig(quant_type=Int4WeightOnlyConfig())
        elif args.quant_dtype_weights == 'int8':
            quantization_config = TorchAoConfig(quant_type=Int8WeightOnlyConfig())
        elif args.quant_dtype_weights == 'float8':
            quantization_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
        elif args.quant_dtype_weights and args.quant_dtype_weights.isnumeric():
            dtype = getattr(torch, f'int{args.quant_dtype_weights}', 'int8')
            quantization_config = TorchAoConfig(quant_type=IntxWeightOnlyConfig(weight_dtype=dtype))

    return model_dtype, quantization_config


def keyword_to_dtype(k):
    kw_dtype_dic = {
        "none": None,
        "int2": qint2,
        "int4": qint4,
        "int8": qint8,
        "float8": qfloat8,
    }
    return kw_dtype_dic.get(k, None)


def quantization_calibration(dataset, benchmark, model, args):
    # keyword_to_itype = {"none": None, "int8": qint8, "int4": qint4}[k]

    weights = keyword_to_dtype(args.quant_dtype_weights)
    activations = keyword_to_dtype(args.quant_dtype_acts)

    model = quantize(model, weights=weights, activations=activations)

    print("Calibrating ...")
    with Calibration():
        dataset = iter(
            dataset.map(
                benchmark,
                batch_size=args.batch_size,
                batched=True,
            )
        )

        for _ in tqdm(dataset, desc='Warming up...'):
            pass

        # evaluate_model(model, processor, processed_dataset, wer, args.batch_size)

    freeze(model)

    return 0