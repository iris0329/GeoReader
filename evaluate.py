"""Select evaluation data and run the official scorers for the chosen recipe."""

import argparse
import json
from pathlib import Path
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from benchmark import (
    BENCHMARK_CHOICES,
    get_benchmark,
    _load_records,
    _run_benchmark_scorer,
)
from checkpoint import load_georeader, load_sft
from inference import predict


def evaluate_one(model, processor, args, name, output):
    """Save questions, model responses and official scores; mark partial runs as non-final."""
    benchmark = get_benchmark(args.data_root, name)
    records = _load_records(benchmark.jsonl_path, args.max_samples)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    with (output / "eval_results.jsonl").open("w") as handle:
        for record in records:
            answer, views, extra = predict(
                model,
                processor,
                record,
                benchmark,
                method=args.method,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                key: value for key, value in record.items() if not key.startswith("_")
            }
            row.update(
                benchmark=name,
                id=record.get("id", f"line_{record['_line_idx']}"),
                line_idx=record["_line_idx"],
                n_images=views,
                input_mode="multi_image",
                answer=answer,
                generated=answer,
                **extra,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    summary = {
        "benchmark": name,
        "method": args.method,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "jsonl": str(benchmark.jsonl_path),
        "processed_rows": len(records),
        "total_records": len(records),
        "max_new_tokens": args.max_new_tokens,
        "input_mode": "multi_image",
        "vsi_num_frames": 32,
        "is_final": args.max_samples is None,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _run_benchmark_scorer(str(output), benchmark, is_final=args.max_samples is None)
    return json.loads((output / "eval_summary.json").read_text())


def main():
    """Load one model and evaluate one or all six benchmarks without overwriting existing results."""
    parser = argparse.ArgumentParser(
        description="Evaluate a GeoReader or Vanilla SFT checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model directory to evaluate."
    )
    parser.add_argument(
        "--data-root", required=True, help="Dataset root that contains eval/."
    )
    parser.add_argument(
        "--method",
        choices=("georeader", "sft"),
        default="georeader",
        help="Use 'sft' for Vanilla SFT checkpoints.",
    )
    parser.add_argument(
        "--benchmark",
        choices=(*BENCHMARK_CHOICES, "mindcube", "all"),
        default="all",
        help="Benchmark to run; 'all' runs the six benchmarks.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for predictions and scores.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N questions per benchmark, for a quick check.",
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
        from mindcube_runtime import run_eval

        return run_eval(args)

    if args.max_new_tokens <= 0 or (
        args.max_samples is not None and args.max_samples <= 0
    ):
        parser.error("--max-new-tokens and --max-samples must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Output directory already exists: {output}")
    model, processor, _ = (load_georeader if args.method == "georeader" else load_sft)(
        args.checkpoint
    )
    names = BENCHMARK_CHOICES if args.benchmark == "all" else (args.benchmark,)
    summaries = {
        name: evaluate_one(
            model,
            processor,
            args,
            name,
            output / name if args.benchmark == "all" else output,
        )
        for name in names
    }
    result = {
        "method": args.method,
        "is_final": args.max_samples is None,
        "benchmarks": {name: row["accuracy"] for name, row in summaries.items()},
    }
    if len(summaries) == 6:
        result["mean_accuracy"] = sum(row["accuracy"] for row in summaries.values()) / 6
        (output / "six_benchmark_summary.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
