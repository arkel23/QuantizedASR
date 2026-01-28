from qasr.misc.misc_utils import parse_args, init_procedure
from qasr.model.model_utils import load_model_and_processor
from qasr.model.quant_utils import quantization_calibration
from qasr.data.data_utils import load_and_prepare_dataset
from qasr.eval.eval_utils import make_benchmark_fn, run_warmup, \
    evaluate_dataset, compute_and_log_metrics


def main():
    args = parse_args()

    init_procedure(args)

    model, processor, model_input_name, gen_kwargs = load_model_and_processor(args)

    dataset, normalizer = load_and_prepare_dataset(args, warmup=True)

    benchmark = make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args)

    if args.warmup_steps:
        run_warmup(dataset, benchmark, args)

    if args.quant_config == 'quanto' and args.quant_dtype_acts is not None:
        quantization_calibration(dataset, benchmark, model, args)

    dataset, normalizer = load_and_prepare_dataset(args)

    results = evaluate_dataset(dataset, benchmark, args)
    compute_and_log_metrics(results, model, args)

    return 0


if __name__ == '__main__':
    main()
