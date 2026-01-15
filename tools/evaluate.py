import torch
import wandb

from qasr.misc.misc_utils import parse_args
from qasr.model.model_utils import load_model_and_processor
from qasr.data.data_utils import load_and_prepare_dataset
from qasr.eval.eval_utils import make_benchmark_fn, run_warmup, evaluate_dataset, compute_and_log_metrics


torch.set_float32_matmul_precision("high")


def main():
    args = parse_args()

    model, processor, model_input_name, gen_kwargs = load_model_and_processor(args)

    dataset, normalizer = load_and_prepare_dataset(args)

    benchmark = make_benchmark_fn(model, processor, normalizer, model_input_name, gen_kwargs, args)

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    wandb.run.name = f"{args.model_id}_{args.dataset}_{args.split}_{args.serial}"

    if args.warmup_steps:
        run_warmup(dataset, benchmark, args)

    dataset, normalizer = load_and_prepare_dataset(args)
    if args.max_eval_samples:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    results = evaluate_dataset(dataset, benchmark, args)
    compute_and_log_metrics(results, args)

    wandb.finish()
    return 0


if __name__ == "__main__":
    main()
