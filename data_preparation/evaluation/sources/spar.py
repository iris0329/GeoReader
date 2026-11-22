#!/usr/bin/env python3
"""Convert SPAR-Bench parquet shard to current standard eval JSONL format and export images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image


# Public entry does not depend on the original cluster path; formal invocation via CLI explicitly passes the official raw location.
DEFAULT_DATA_ROOT = Path("raw/evaluation/spar")
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_ROOT / "processed" / "spar_bench_test_local.jsonl"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT / "processed" / "images"
DEFAULT_STATS_FILE = DEFAULT_DATA_ROOT / "meta" / "spar_bench_stats.json"

MCA_QUESTION_TYPES = {
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
}
NA_QUESTION_TYPES = {
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
}
SPECIAL_QUESTION_TYPES = {"view_change_infer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="SPAR-Bench root directory with raw/, processed/ and meta/.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL in the GeoReader evaluation format.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=str(DEFAULT_IMAGE_ROOT),
        help="Directory for decoded images.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=str(DEFAULT_STATS_FILE),
        help="Output JSON with summary statistics.",
    )
    parser.add_argument(
        "--force-rewrite-images",
        action="store_true",
        help="Overwrite decoded images that already exist.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Keep only the first N rows (0 = all).",
    )
    return parser.parse_args()


def load_parquet_rows(path: Path) -> List[Dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return table.to_pylist()
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    except ImportError as exc:
        raise RuntimeError(
            "Failed to read parquet: current environment lacks pyarrow / pandas.\nPlease switch to the environment with the parquet dependency installed before running this script."
        ) from exc


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def hash_file_bytes(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def as_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if isinstance(image_bytes, (bytes, bytearray)):
            return Image.open(BytesIO(bytes(image_bytes))).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    raise ValueError(f"unsupported image field type: {type(value).__name__}")


def save_images(
    images: Sequence[object],
    sample_id: int,
    image_root: Path,
    force_rewrite: bool,
) -> List[Dict[str, str]]:
    saved: List[Dict[str, str]] = []
    for index, image_value in enumerate(images):
        rel_path = Path(str(sample_id)) / f"{sample_id}_{index}.jpg"
        out_path = image_root / rel_path
        ensure_parent(out_path)
        if not out_path.exists() or force_rewrite:
            as_image(image_value).save(out_path, format="JPEG", quality=95)
        saved.append(
            {"rel_path": rel_path.as_posix(), "file_sha1": hash_file_bytes(out_path)}
        )
    return saved


def resolve_metric_type(task: str) -> str:
    if task in MCA_QUESTION_TYPES:
        return "accuracy"
    if task in NA_QUESTION_TYPES:
        return "mra"
    if task in SPECIAL_QUESTION_TYPES:
        return "special"
    raise ValueError(f"unknown SPAR task: {task}")


def convert_rows(
    rows: Iterable[Dict],
    image_root: Path,
    force_rewrite_images: bool,
) -> List[Dict]:
    converted: List[Dict] = []
    seen_ids = set()

    for row in rows:
        raw_id = int(row["id"])
        sample_id = f"spar_{raw_id}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicated sample id: {sample_id}")
        seen_ids.add(sample_id)

        images = row.get("image") or []
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError(f"invalid image field for id={raw_id}")

        task = normalize_text(row.get("task"))
        saved_images = save_images(
            images,
            sample_id=raw_id,
            image_root=image_root,
            force_rewrite=force_rewrite_images,
        )
        converted.append(
            {
                "id": sample_id,
                "question": normalize_text(row.get("question")),
                "images": [item["rel_path"] for item in saved_images],
                "gt_answer": normalize_text(row.get("answer")),
                "task": task,
                "metric_type": resolve_metric_type(task),
                "image_type": normalize_text(row.get("img_type")),
                "img_type": normalize_text(row.get("img_type")),
                "format_type": normalize_text(row.get("format_type")),
                "source": normalize_text(row.get("source")),
                "original_id": raw_id,
                "_image_file_sha1": [item["file_sha1"] for item in saved_images],
            }
        )
    return converted


def assign_scene_ids(rows: Sequence[Dict]) -> List[Dict]:
    if not rows:
        return []

    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hash_to_rows: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for image_hash in row.get("_image_file_sha1", []):
            if image_hash:
                hash_to_rows[image_hash].append(index)

    for row_indices in hash_to_rows.values():
        if len(row_indices) <= 1:
            continue
        base = row_indices[0]
        for other in row_indices[1:]:
            union(base, other)

    component_to_rows: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(rows)):
        component_to_rows[find(index)].append(index)

    source_to_components: Dict[str, List[tuple[int, List[int]]]] = defaultdict(list)
    for root, row_indices in component_to_rows.items():
        source = normalize_text(rows[row_indices[0]].get("source"))
        source_to_components[source].append((root, row_indices))

    component_scene_id: Dict[int, str] = {}
    for source, components in source_to_components.items():
        components.sort(
            key=lambda item: (
                min(int(rows[idx].get("original_id", idx)) for idx in item[1]),
                item[0],
            )
        )
        source_prefix = source or "unknown"
        for rank, (root, _) in enumerate(components, start=1):
            component_scene_id[root] = f"{source_prefix}_scenecomp_{rank:05d}"

    assigned: List[Dict] = []
    for index, row in enumerate(rows):
        root = find(index)
        component_rows = component_to_rows[root]
        copied = dict(row)
        copied["scene_id"] = component_scene_id[root]
        copied["scene_name"] = component_scene_id[root]
        copied["scene_id_source"] = "shared_image_component"
        copied["scene_component_size"] = len(component_rows)
        copied.pop("_image_file_sha1", None)
        assigned.append(copied)
    return assigned


def build_stats(rows: Sequence[Dict]) -> Dict:
    by_task = Counter()
    by_image_type = Counter()
    by_metric = Counter()
    by_source = Counter()
    by_num_images = Counter()
    by_scene_id_source = Counter()
    by_scene_component_size = Counter()
    unique_scene_ids = set()

    for row in rows:
        by_task[row["task"]] += 1
        by_image_type[row["image_type"]] += 1
        by_metric[row["metric_type"]] += 1
        by_source[row["source"]] += 1
        by_num_images[len(row["images"])] += 1
        if row.get("scene_id"):
            unique_scene_ids.add(row["scene_id"])
        if row.get("scene_id_source"):
            by_scene_id_source[row["scene_id_source"]] += 1
        if row.get("scene_component_size"):
            by_scene_component_size[int(row["scene_component_size"])] += 1

    return {
        "total_rows": len(rows),
        "by_task": dict(sorted(by_task.items())),
        "by_image_type": dict(sorted(by_image_type.items())),
        "by_metric": dict(sorted(by_metric.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_num_images": dict(sorted(by_num_images.items())),
        "unique_scene_ids": len(unique_scene_ids),
        "avg_rows_per_scene_component": round(len(rows) / len(unique_scene_ids), 4)
        if unique_scene_ids
        else 0.0,
        "max_scene_component_size": max(by_scene_component_size)
        if by_scene_component_size
        else 0,
        "by_scene_id_source": dict(sorted(by_scene_id_source.items())),
        "by_scene_component_size": dict(sorted(by_scene_component_size.items())),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    output_file = Path(args.output_file).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    stats_file = Path(args.stats_file).expanduser().resolve()

    parquet_files = sorted((raw_dir / "data").glob("test-*.parquet"))
    if not parquet_files:
        parquet_files = sorted(raw_dir.glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"no SPAR-Bench test parquet found under {raw_dir}. "
            "Please run download_spar_bench_assets.py first."
        )

    merged_rows: List[Dict] = []
    for parquet_file in parquet_files:
        merged_rows.extend(load_parquet_rows(parquet_file))

    converted = convert_rows(
        merged_rows,
        image_root=image_root,
        force_rewrite_images=args.force_rewrite_images,
    )
    converted = assign_scene_ids(converted)
    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    stats = build_stats(converted)
    write_jsonl(output_file, converted)
    ensure_parent(stats_file)
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("SPAR-Bench local eval data ready.")
    print(f"  parquet_files: {len(parquet_files)}")
    print(f"  output_file:   {output_file}")
    print(f"  image_root:    {image_root}")
    print(f"  stats_file:    {stats_file}")
    print(f"  total_rows:    {stats['total_rows']}")


if __name__ == "__main__":
    main()
