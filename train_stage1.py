"""Select Reader training inputs according to the recipe, shared by the Stage1 trainer."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    """Prepare Stage 1 inputs from the dataset root and launch Reader training."""
    parser = argparse.ArgumentParser(
        description="Train a GeoReader Stage 1 Reader (camera or depth)."
    )
    parser.add_argument(
        "--branch",
        choices=("camera", "depth"),
        required=True,
        help="Which Reader to train.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Local base model directory for the selected recipe, e.g. Qwen3-VL-8B-Instruct.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root, e.g. the extracted GeoReader-MultiSource directory.",
    )
    parser.add_argument(
        "--data-file",
        help="Train on this Stage 1 JSONL instead of the one in --data-root.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for checkpoints and logs."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume training from a checkpoint inside --output-dir.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a few steps on one GPU to check the setup.",
    )
    parser.add_argument(
        "--recipe",
        choices=("sixbench_qwen3_8b", "mindcube_da3_qwen25_3b"),
        default="sixbench_qwen3_8b",
        help="Training recipe; settings are in configs/<recipe>/.",
    )
    args = parser.parse_args()
    if args.recipe == "mindcube_da3_qwen25_3b":
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from mindcube_runtime import run_stage1

        return run_stage1(args)

    os.environ["WANDB_MODE"] = "offline"
    os.environ["DEPTH_FROM_SCRATCH"] = "1"
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from preparation import (
        dataset_entries,
        prepare_depth,
        validate_output_dir,
        validate_base_model,
    )

    validate_base_model(args.base_model)
    validate_output_dir(args.output_dir, args.resume_from_checkpoint)
    if args.data_file:
        source = Path(args.data_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        selected = [{"absolute": str(source), "signal": args.branch}]
    else:
        entries = dataset_entries(args.data_root, "stage1")
        selected = [row for row in entries if row["signal"] == args.branch]
    if len(selected) != 1:
        raise RuntimeError(
            "Expected exactly one Stage 1 input for the selected --branch."
        )
    if not args.smoke and int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError(
            "Formal Stage1 requires torchrun with four cards, 4 GPU(s), batch size 1 per GPU, gradient accumulation 1"
        )
    values = json.loads(
        (root / f"configs/sixbench_qwen3_8b/stage1_{args.branch}.json").read_text()
    )["arguments"]
    values.update(
        model=str(Path(args.base_model).resolve()),
        data_path=selected[0]["absolute"],
        image_folder=str(Path(args.data_root).resolve()),
        save_model_path=str(Path(args.output_dir).resolve()),
        cache_dir=str(Path(args.output_dir).resolve() / "cache"),
    )
    if args.branch == "depth":
        # Explicit new JSONL supports small-sample validation; default published inputs still strictly verify the original 8B window/frame count.
        data, report = prepare_depth(
            selected[0]["absolute"],
            args.output_dir,
            expected_counts=None if args.data_file else (35422, 228691),
        )
        values.update(data_path=str(data), data_provenance_report=str(report))
    if args.smoke:
        # Only validate updates, first logs, and saves; do not modify the official configuration file.
        values.update(
            max_samples=2,
            max_steps=1,
            logging_steps=1,
            save_steps=1,
            save_total_limit=1,
            dataloader_num_workers=0,
            save_only_model="1",
            skip_final_save="1",
        )
    if args.resume_from_checkpoint:
        values["resume_from_checkpoint"] = args.resume_from_checkpoint
    sys.argv = [sys.argv[0]] + [
        piece for key, value in values.items() for piece in (f"--{key}", str(value))
    ]
    from main_camera_interpreter import main as train

    train()


if __name__ == "__main__":
    main()
