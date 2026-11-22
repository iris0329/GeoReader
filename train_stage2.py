"""Select QA training with GeoReader or Vanilla SFT based on the recipe."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    """Launch Stage 2 training for GeoReader or the Vanilla SFT baseline."""
    parser = argparse.ArgumentParser(
        description="Stage 2 QA training for GeoReader or the Vanilla SFT baseline."
    )
    parser.add_argument(
        "--method",
        choices=("georeader", "sft"),
        required=True,
        help="'georeader' trains with frozen Readers; 'sft' trains the plain VLM baseline.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Local base model directory, e.g. Qwen3-VL-8B-Instruct.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root, e.g. the extracted GeoReader-MultiSource directory.",
    )
    parser.add_argument(
        "--data-file",
        nargs="+",
        help="Train on these Stage 2 JSONL files instead of the ones in --data-root.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for checkpoints and logs."
    )
    parser.add_argument(
        "--camera-reader",
        help="Stage 1 camera Reader checkpoint, or a released GeoReader model directory.",
    )
    parser.add_argument(
        "--depth-reader",
        help="Stage 1 depth Reader checkpoint, or a released GeoReader model directory.",
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
        if args.data_file and len(args.data_file) != 1:
            parser.error("MindCube recipe only accepts one --data-file")
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from mindcube_runtime import run_stage2

        return run_stage2(args)

    if args.method == "georeader" and not (args.camera_reader and args.depth_reader):
        parser.error("GeoReader requires --camera-reader and --depth-reader")
    if args.method == "sft" and (args.camera_reader or args.depth_reader):
        parser.error(
            "--method sft does not use Readers; remove --camera-reader and --depth-reader"
        )
    if not args.smoke and int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError(
            "Formal Stage2 requires four GPUs with torchrun, 4 GPU(s), batch size 1 per GPU, gradient accumulation 2"
        )
    os.environ["WANDB_MODE"] = "offline"
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from preparation import prepare_stage2, validate_output_dir, validate_base_model

    cfg = json.loads((root / "configs/sixbench_qwen3_8b/recipe.json").read_text())[
        "stage2"
    ]
    validate_base_model(args.base_model)
    validate_output_dir(args.output_dir, args.resume_from_checkpoint)
    recipe, recipe_path, manifest = prepare_stage2(
        args.data_root, args.output_dir, args.smoke, data_files=args.data_file
    )
    if args.method == "georeader":
        command = [
            "--model",
            args.base_model,
            "--data-root",
            args.data_root,
            "--camera",
            args.camera_reader,
            "--depth",
            args.depth_reader,
            "--qa-manifest",
            str(manifest),
            "--output-dir",
            args.output_dir,
            "--keep-resume-state",
        ]
        if args.smoke:
            command += ["--smoke", "--smoke-rows", recipe["qa_data_paths"][0]]
        if args.resume_from_checkpoint:
            command += ["--resume-from-checkpoint", args.resume_from_checkpoint]
        from main_camera_stage2_qwen3 import main as train
    else:
        os.environ["RGB_TRAIN_RECIPE"] = str(recipe_path)
        os.environ["STRICT_RELOAD"] = "1" if args.smoke else "0"
        # Two methods unified ep2, per card b1, accumulated 2; pixel, learning rate, and warmup consistent with the already rated model.
        values = dict(
            model=args.base_model,
            image_folder=args.data_root,
            save_model_path=args.output_dir,
            epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_per_gpu"],
            gradient_accumulation_steps=cfg["gradient_accumulation"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg["warmup_steps"],
            weight_decay=cfg["weight_decay"],
            min_pixels=3136,
            max_pixels=150528,
            expected_image_size="mixed",
            train_sampler_mode="random",
            gradient_checkpointing="1",
            draw_spar_marker_specs="1",
            logging_steps=1 if args.smoke else cfg["logging_steps"],
            save_steps=4 if args.smoke else cfg["save_steps"],
            save_total_limit=1,
            max_steps=4 if args.smoke else cfg["max_steps"],
            expected_final_step=4 if args.smoke else cfg["max_steps"],
            save_only_model="1" if args.smoke else "0",
            dataloader_num_workers=0 if args.smoke else 8,
        )
        if args.resume_from_checkpoint:
            values["resume_from_checkpoint"] = args.resume_from_checkpoint
        command = [
            piece for key, value in values.items() for piece in (f"--{key}", str(value))
        ]
        command += ["--qa_data_path", *recipe["qa_data_paths"]]
        from main_stage2_rgb_sft import main as train
    sys.argv = [sys.argv[0], *command]
    train()


if __name__ == "__main__":
    main()
