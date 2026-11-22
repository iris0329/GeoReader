"""Locate training inputs according to the release checklist and automatically expand frames as needed internally for training."""

import json
import os
from pathlib import Path

from filelock import FileLock

from georeader_data import resolve_record_paths
from depth_targets import validate_source_record, make_frame_record


def validate_base_model(base_model):
    """Only supports Qwen3-VL-8B; refuses to load weights for other bases prior to loading."""
    config = json.loads(
        (Path(base_model).expanduser().resolve() / "config.json").read_text()
    )
    text = config["text_config"]
    if (config["model_type"], text["hidden_size"], text["num_hidden_layers"]) != (
        "qwen3_vl",
        4096,
        36,
    ):
        raise RuntimeError(
            "This recipe requires Qwen3-VL-8B-Instruct as the base model"
        )


def validate_output_dir(output_dir, resume):
    """New training cannot overwrite existing checkpoints; continuing training must explicitly point to the same output directory."""
    output = Path(output_dir).expanduser().resolve()
    if resume:
        checkpoint = Path(resume).expanduser().resolve()
        if (
            not checkpoint.is_relative_to(output)
            or not (checkpoint / "trainer_state.json").is_file()
        ):
            raise RuntimeError(
                "--resume-from-checkpoint must point to a checkpoint inside --output-dir that contains trainer_state.json"
            )
    elif output.exists() and any(output.glob("checkpoint-*")):
        raise RuntimeError(
            f"Output directory already has checkpoints; pass --resume-from-checkpoint or choose a new directory: {output}"
        )


def dataset_entries(data_root, phase):
    """Only accepts published fixed lists, with file order consistent with the original training recipe."""
    root = Path(data_root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "georeader-sixbench":
        raise RuntimeError(
            "--data-root is not a GeoReader-MultiSource dataset directory."
        )
    entries = [
        dict(row, absolute=str(root / row["path"]))
        for row in manifest["datasets"]
        if row["phase"] == phase
    ]
    for row in entries:
        if not Path(row["absolute"]).is_file():
            raise FileNotFoundError(row["absolute"])
    return entries


def prepare_depth(source, output_dir, *, expected_counts=(35422, 228691)):
    """Continue original conversion, split GT description frame by frame; do not alter numbers, frame order, or geometry files."""
    source = Path(source).resolve()
    target = Path(output_dir).resolve() / "prepared_data/depth.jsonl"
    report = target.with_suffix(".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Multiple ranks in torchrun share the same converted result to avoid simultaneous truncation of outputs.
    with FileLock(str(target) + ".lock"):
        if target.exists() != report.exists():
            raise RuntimeError(
                f"Per-frame depth file is incomplete; delete it and retry: {target}"
            )
        if report.exists():
            saved = json.loads(report.read_text())
            if saved["input_path"] != str(source) or saved["output_path"] != str(
                target
            ):
                raise RuntimeError(
                    "Existing per-frame input was built from a different --data-root"
                )
            return target, report
        windows = frames = 0
        identities = set()
        partial = target.with_suffix(".partial")
        with source.open() as reader, partial.open("w") as writer:
            for index, line in enumerate(reader):
                record = resolve_record_paths(json.loads(line), source)
                identity = int(record["idx"])
                if identity in identities:
                    raise RuntimeError(f"Duplicate window id: {identity}")
                identities.add(identity)
                count, targets = validate_source_record(
                    record, context=f"{source}:{index + 1}"
                )
                for frame, caption in enumerate(targets):
                    row = make_frame_record(
                        record,
                        source_window_idx=index,
                        frame_index=frame,
                        n_images=count,
                        target=caption,
                        expanded_idx=frames,
                    )
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    frames += 1
                windows += 1
        if expected_counts is not None and (windows, frames) != expected_counts:
            raise RuntimeError(
                f"Unexpected per-frame input count: windows={windows}, frames={frames}"
            )
        os.replace(partial, target)
        report.write_text(
            json.dumps(
                {
                    "format": "georeader-depth-expansion",
                    "status": "passed",
                    "input_path": str(source),
                    "output_path": str(target),
                    "contract": {
                        "source_window_rows": windows,
                        "expanded_rows": frames,
                        # Expand without random sampling; record in sequence using the original 8B generator, with training seed separately noted in checkpoint.
                        "data_order_mode": "source_window_then_frame_order",
                        "data_order_seed": 0,
                        "new_cot_version": "depth_layout_perframe_caption_microbatch",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    return target, report


def prepare_stage2(data_root, output_dir, smoke, *, data_files=None):
    """Read the publication list or explicitly generate files; formal training retains a budget of 135399 samples."""
    if data_files:
        paths = [Path(path).expanduser().resolve() for path in data_files]
        if len(set(paths)) != len(paths):
            raise RuntimeError(
                "--data-file cannot reference the same file multiple times"
            )
        entries = []
        for path in paths:
            rows = 0
            with path.open() as reader:
                for index, line in enumerate(reader):
                    if not line.strip() or not isinstance(json.loads(line), dict):
                        raise RuntimeError(
                            f"Stage2 JSONL line invalid: {path}:{index + 1}"
                        )
                    rows += 1
            if not rows:
                raise RuntimeError(f"Stage2 JSONL is empty: {path}")
            entries.append(dict(absolute=str(path), rows=rows))
        if not smoke and sum(row["rows"] for row in entries) != 135399:
            raise RuntimeError(
                "The full recipe uses 135,399 questions; smaller subsets are only allowed with --smoke"
            )
    else:
        entries = dataset_entries(data_root, "stage2")
        if len(entries) != 36 or sum(row["rows"] for row in entries) != 135399:
            raise RuntimeError(
                "Stage 2 input is not the 36-file, 135,399-question recipe"
            )
    prepared = Path(output_dir).resolve() / "prepared_data"
    prepared.mkdir(parents=True, exist_ok=True)
    with FileLock(str(prepared / "stage2.lock")):
        recipe = {
            "qa_data_paths": [row["absolute"] for row in entries],
            "qa_data_rows": [row["rows"] for row in entries],
            "qa_rows_total": 135399,
        }
        if smoke:
            buckets = {
                (pose, depth): [] for pose in (False, True) for depth in (False, True)
            }
            for entry in entries:
                with Path(entry["absolute"]).open() as reader:
                    for line in reader:
                        row = json.loads(line)
                        key = bool(row.get("has_pose")), bool(row.get("has_depth"))
                        # Small-sample validation uses the real branch; formal training still reads the full recipe and original frame count.
                        if len(buckets[key]) < 2 and int(row["n_images"]) <= 2:
                            buckets[key].append(
                                resolve_record_paths(row, entry["absolute"])
                            )
                        if all(len(rows) == 2 for rows in buckets.values()):
                            break
                if all(len(rows) == 2 for rows in buckets.values()):
                    break
            if any(len(rows) != 2 for rows in buckets.values()):
                raise RuntimeError(
                    "--smoke needs two questions for each of the four modality settings."
                )
            sample_path = prepared / "smoke.jsonl"
            sample_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for rows in buckets.values()
                    for row in rows
                )
            )
            recipe = {
                "qa_data_paths": [str(sample_path)],
                "qa_data_rows": [8],
                "qa_rows_total": 8,
            }
        recipe_path = prepared / "stage2_recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
        (prepared / "qa_manifest.txt").write_text(
            "\n".join(recipe["qa_data_paths"]) + "\n"
        )
    return recipe, recipe_path, prepared / "qa_manifest.txt"
