import argparse


# ================================
# CLI
# ================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', type=int, default=0)

    parser.add_argument("--model_id", type=str, default='openai/whisper-tiny.en')
    parser.add_argument("--max_new_tokens", type=int, default=None)

    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, default='ami')
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="max-autotune")

    parser.add_argument("--wandb_project", type=str, default="OpenASR")
    parser.add_argument("--wandb_entity", type=str, default="nycu_pcs")

    parser.set_defaults(streaming=False)
    args = parser.parse_args()
    return args
