"""Generate an answer to an evaluation question according to the recipe."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from benchmark import BENCHMARK_CHOICES, get_benchmark, _load_records
from checkpoint import load_georeader, load_sft
from inference import predict


def main():
    """Answer one evaluation question with a checkpoint."""
    parser = argparse.ArgumentParser(
        description="Run a checkpoint on a single evaluation question."
    )
    parser.add_argument("--checkpoint", required=True, help="Model directory.")
    parser.add_argument(
        "--data-root", required=True, help="Dataset root that contains eval/."
    )
    parser.add_argument(
        "--benchmark",
        choices=(*BENCHMARK_CHOICES, "mindcube"),
        required=True,
        help="Benchmark to read the question from.",
    )
    parser.add_argument(
        "--method",
        choices=("georeader", "sft"),
        default="georeader",
        help="Use 'sft' for Vanilla SFT checkpoints.",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="0-based index of the question."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum answer length in tokens (default: 128, as in the paper).",
    )
    parser.add_argument(
        "--recipe",
        choices=("sixbench_qwen3_8b", "mindcube_da3_qwen25_3b"),
        default="sixbench_qwen3_8b",
        help="Use mindcube_da3_qwen25_3b for the MindCube model.",
    )
    args = parser.parse_args()
    if args.recipe == "mindcube_da3_qwen25_3b":
        from mindcube_runtime import run_infer

        return run_infer(args)

    if args.index < 0 or args.max_new_tokens <= 0:
        parser.error(
            "--index must be non-negative and --max-new-tokens must be positive"
        )
    benchmark = get_benchmark(args.data_root, args.benchmark)
    records = _load_records(benchmark.jsonl_path, args.index + 1)
    if args.index >= len(records):
        raise IndexError(f"--index {args.index} is out of range")
    model, processor, _ = (load_georeader if args.method == "georeader" else load_sft)(
        args.checkpoint
    )
    record = records[args.index]
    answer, views, extra = predict(
        model,
        processor,
        record,
        benchmark,
        method=args.method,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        json.dumps(
            {
                "id": record.get("id"),
                "question": record["question"],
                "answer": answer,
                "ground_truth": record["gt_answer"],
                "n_images": views,
                **extra,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
