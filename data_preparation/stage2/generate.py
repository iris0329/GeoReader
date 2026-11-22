"""Supervise Stage2 answers from upstream QA and ordered geometry index generation, without reading previous ready training files."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import image_paths, read_jsonl, write_json, write_jsonl


def parse_args():
    """Explicitly specify the original QA, image root directory, and geometry index; do not infer data sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=("sixbench_qwen3_8b", "mindcube_da3_qwen25_3b"),
        required=True,
    )
    parser.add_argument(
        "--qa",
        required=True,
        help="Source training QA: question/answer/id rows for the six-benchmark recipe, or the official MindCube 10K QA.",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Prefix to strip from image paths in the QA file.",
    )
    parser.add_argument("--image-field", default="image_input")
    parser.add_argument(
        "--geometry-index",
        required=True,
        help="DA3 groups.jsonl, or a geometry index in the same format.",
    )
    parser.add_argument("--output-dir", required=True)
    # workers only control parallel file acceptance for independent graph groups; when dealing with many large depth files, reduce this value to lower memory and I/O usage.
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Process only the first N rows for a quick check; the report is marked as incomplete.",
    )
    return parser.parse_args()


def strip_view_hint(question, images):
    """Follow MindCube's original rules to remove the viewpoint hint at the beginning of the question, without altering the turning conditions within the actual problem."""
    question = " ".join(question.split())
    question = question.replace(
        "two opposing views (front and back) of the same scene captured during rotation",
        "two views showing the same scene",
    )
    category = Path(images[0]).parts[-3]
    if category == "rotation" and not question.startswith("Based on "):
        if " Based on " not in question:
            raise RuntimeError("MindCube rotation problem statement format abnormal")
        prefix = question.split(". ", 1)[0]
        prefix = re.sub(
            r" from (?:two |three |four )?different viewpoints$", "", prefix
        )
        return prefix + ". Based on " + question.split(" Based on ", 1)[1]
    return re.sub(r" from different viewpoints[^:]*:", ":", question, count=1)


def index_path(value, index_file):
    """The relative path of the geometry file is resolved only relative to the index position, not by filename search."""
    path = Path(value)
    if not path.is_absolute():
        path = index_file.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def modality_flag(value, name):
    """Accept only explicit boolean values or 0/1; avoid misinterpreting string "false" as true."""
    if type(value) not in (bool, int) or value not in (0, 1):
        raise ValueError(f"{name}Must be a boolean or 0/1")
    return bool(value)


def load_geometry(index_file, image_root):
    """Index by complete ordered image paths, prohibit sorting image order or ignoring directory names of same-named files."""
    result = {}
    for row in read_jsonl(index_file):
        images = tuple(image_paths(row, "image_input", image_root, ""))
        if images in result:
            raise RuntimeError(f"Geometric index repeated ordered group: {images}")
        geometry = {"image_input": list(images)}
        for flag, field in (("has_pose", "pose_path"), ("has_depth", "depth_path")):
            enabled = modality_flag(row[flag], flag)
            path = row[field]
            if enabled != (isinstance(path, str) and bool(path)) or (
                not enabled and path is not None
            ):
                raise ValueError(f"{flag}and{field}Inconsistent")
            geometry[flag] = enabled
            geometry[field] = index_path(path, index_file) if enabled else None
        if geometry["has_pose"] or geometry["has_depth"]:
            if not isinstance(row["geometry_mode"], str) or not row["geometry_mode"]:
                raise ValueError(
                    "Geometry groups that have geometry must record geometry_mode"
                )
            geometry["geometry_mode"] = row["geometry_mode"]
            if "geometry_source" in row:
                geometry["geometry_source"] = row["geometry_source"]
        result[images] = geometry
    if not result:
        raise RuntimeError("Geometry index is empty")
    return result


def validate_geometry(geometry):
    """Each actual used graph group is validated only once for RGB dimensions, geometric arrays, and NPZ image order."""
    images = geometry["image_input"]
    sizes = []
    for path in images:
        with Image.open(path) as image:
            sizes.append(list(image.size))
    count = len(images)
    for enabled, field in (
        (geometry["has_pose"], "pose_path"),
        (geometry["has_depth"], "depth_path"),
    ):
        if not enabled:
            continue
        with np.load(geometry[field], allow_pickle=False) as values:
            # Geometry must point to the current QA image and cannot integrate a different frame order from the same scene.
            stored = [
                str(Path(path).resolve()) for path in values["image_input"].astype(str)
            ]
            if stored != images or not np.array_equal(values["image_sizes"], sizes):
                raise RuntimeError(
                    f"NPZ image order or RGB dimensions do not match: {geometry[field]}"
                )
            if field == "pose_path":
                e, k = values["extrinsics"], values["intrinsics"]
                if e.shape != (count, 3, 4) or k.shape != (count, 3, 3):
                    raise RuntimeError(
                        f"Camera array shape mismatch: {geometry[field]}"
                    )
                if (
                    not np.isfinite(e).all()
                    or not np.isfinite(k).all()
                    or np.any(k[:, (0, 1), (0, 1)] <= 0)
                ):
                    raise RuntimeError(
                        f"Camera contains invalid values: {geometry[field]}"
                    )
            else:
                depth = values["depth_maps"]
                if (
                    depth.ndim != 3
                    or depth.shape[0] != count
                    or min(depth.shape[1:]) <= 0
                ):
                    raise RuntimeError(f"Depth array shape mismatch: {geometry[field]}")
                if not np.isfinite(depth).all() or np.any(depth <= 0):
                    raise RuntimeError(
                        f"Depth must be a finite positive number: {geometry[field]}"
                    )
                if str(values["unit"]) != "meter":
                    raise RuntimeError(f"Depth unit must be meter: {geometry[field]}")
                # Like the training reader, only allow a 3% error in aspect ratio, and do not forcibly attach depth to RGB of the other dimension.
                if any(
                    abs((depth.shape[2] / depth.shape[1]) / (width / height) - 1) > 0.03
                    for width, height in sizes
                ):
                    raise RuntimeError(
                        f"Depth and RGB have inconsistent aspect ratio: {geometry[field]}"
                    )
    return dict(geometry, image_sizes=sizes, n_images=count)


def build_row(spec, recipe):
    """Answers come only from upstream QA; geometry is only added from the group index and does not participate in answer generation."""
    index, raw, geometry = spec
    if recipe == "mindcube_da3_qwen25_3b":
        if (
            not geometry["has_pose"]
            or geometry["has_depth"]
            or geometry["n_images"] not in (2, 3, 4)
        ):
            raise RuntimeError(
                "MindCube recipe requires 2/3/4 images, with DA3 geometry indices for only Camera."
            )
        if (
            geometry.get("geometry_source") != "da3_nested"
            or geometry["geometry_mode"] != "ordered_multiview"
        ):
            raise RuntimeError(
                "MindCube recipe must use DA3 Nested ordered multi-image geometry"
            )
        question = strip_view_hint(raw["text_input"], geometry["image_input"])
        row = dict(
            geometry,
            idx=raw["idx"],
            dataset="mindcube",
            subtask=Path(geometry["image_input"][0]).parts[-3],
        )
    else:
        # The six-benchmark input follows the upstream annotation schema; existing ready JSONL is not a generation template.
        if any(
            name in raw
            for name in (
                "text_output",
                "pose_path",
                "depth_path",
                "geometry_provenance",
            )
        ):
            raise RuntimeError(
                "The six-benchmark recipe needs source QA with question/answer/id, not already-built training rows"
            )
        question = raw["question"]
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object")
        protected = set(geometry) | {
            "idx",
            "sample_id",
            "text_input",
            "text_output",
            "answer",
        }
        if set(metadata) & protected:
            raise RuntimeError(
                f"metadata covers essential training fields: {sorted(set(metadata) & protected)}"
            )
        # The bounding box/coordinate rewriting rules are explicitly provided by the original annotations; retain them in the real training reader, not just export QA text.
        row = dict(metadata, **geometry, idx=index, sample_id=str(raw["id"]))
    answer = raw["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The question must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "The answer must be a non-empty string provided by the upstream annotation."
        )
    row.update(
        text_input=question,
        answer=answer,
        text_output="<think></think>\n<answer>" + answer + "</answer>",
    )
    return row


def main():
    """Connect QA with geometry in upstream order; remove CoT, do not extract questions, do not alter answers or multi-image order."""
    args = parse_args()
    if args.workers <= 0 or (args.max_rows is not None and args.max_rows <= 0):
        raise ValueError("workers/max-rows must be a positive number")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    geometry = load_geometry(Path(args.geometry_index).resolve(), args.image_root)
    records, keys, used, identities = [], [], {}, set()
    total = 0
    for index, raw in enumerate(read_jsonl(args.qa)):
        total += 1
        if args.max_rows is not None and index >= args.max_rows:
            continue
        images = tuple(
            image_paths(raw, args.image_field, args.image_root, args.image_prefix)
        )
        if images not in geometry:
            raise KeyError(
                f"Source QA row {index + 1}: no geometry found with the same image order"
            )
        identity = raw["idx"] if args.recipe == "mindcube_da3_qwen25_3b" else raw["id"]
        if identity in identities:
            raise RuntimeError(f"QA identity duplication: {identity}")
        identities.add(identity)
        records.append(raw)
        keys.append(images)
        used[images] = geometry[images]
    if not records:
        raise RuntimeError("Source QA file is empty")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checked = dict(zip(used, pool.map(validate_geometry, used.values())))
    rows = [
        build_row((i, row, checked[key]), args.recipe)
        for i, (row, key) in enumerate(zip(records, keys))
    ]
    write_jsonl(output / "qa.jsonl", rows)
    report = dict(
        format="georeader-stage2-generation",
        recipe=args.recipe,
        qa_input=str(Path(args.qa).resolve()),
        geometry_index=str(Path(args.geometry_index).resolve()),
        output=str(output / "qa.jsonl"),
        input_rows=total,
        rows=len(rows),
        is_complete=total == len(rows),
        ordered_image_groups=len(used),
        modality_counts=dict(
            Counter(f"{int(row['has_pose'])}{int(row['has_depth'])}" for row in rows)
        ),
        supervision="answer_only",
        selection="all_input_rows_in_original_order",
        old_ready_jsonl_used=False,
    )
    write_json(output / "generation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
