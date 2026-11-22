#!/usr/bin/env python3
"""Convert MMSI-Bench parquet to current standard eval JSONL and extract images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


# Public entry does not depend on the original cluster path; formal invocation via CLI explicitly passes the official raw location.
DEFAULT_DATA_ROOT = Path("raw/evaluation/mmsi")
DEFAULT_INPUT_FILE = DEFAULT_DATA_ROOT / "raw" / "MMSI_Bench.parquet"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_ROOT / "processed" / "mmsi_bench_test_local.jsonl"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT / "processed" / "images"
DEFAULT_STATS_FILE = DEFAULT_DATA_ROOT / "meta" / "mmsi_bench_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Input parquet file downloaded from Hugging Face.",
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
        help="Overwrite images that already exist.",
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


def as_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if isinstance(image_bytes, bytes):
            return image_bytes
        if isinstance(image_bytes, bytearray):
            return bytes(image_bytes)
        if isinstance(image_bytes, memoryview):
            return image_bytes.tobytes()
    raise ValueError(f"unsupported image bytes type: {type(value).__name__}")


def save_images(
    image_list: Sequence[object],
    sample_id: int,
    image_root: Path,
    force_rewrite: bool,
) -> List[str]:
    rel_paths: List[str] = []
    for index, image_data in enumerate(image_list):
        rel_path = Path(f"{sample_id}") / f"{sample_id}_{index}.jpg"
        out_path = image_root / rel_path
        ensure_parent(out_path)
        if not out_path.exists() or force_rewrite:
            out_path.write_bytes(as_bytes(image_data))
        rel_paths.append(rel_path.as_posix())
    return rel_paths


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def convert_rows(
    rows: Iterable[Dict],
    image_root: Path,
    force_rewrite_images: bool,
) -> List[Dict]:
    converted: List[Dict] = []
    for row in rows:
        sample_id = int(row["id"])
        images = row.get("images")
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError(f"invalid images field for id={sample_id}")

        rel_paths = save_images(
            images,
            sample_id=sample_id,
            image_root=image_root,
            force_rewrite=force_rewrite_images,
        )
        converted.append(
            {
                "id": f"mmsi_{sample_id}",
                "question": normalize_text(row.get("question")),
                "images": rel_paths,
                "gt_answer": normalize_text(row.get("answer")),
                "question_type": normalize_text(row.get("question_type")),
                "difficulty": str(row.get("difficulty")),
                "thought": normalize_text(row.get("thought")),
                "human_normed_duration_seconds": row.get(
                    "mean_normed_duration_seconds"
                ),
                "original_id": sample_id,
            }
        )
    return converted


def build_stats(rows: Sequence[Dict]) -> Dict:
    by_question_type = Counter()
    by_difficulty = Counter()
    num_images = Counter()

    for row in rows:
        by_question_type[row["question_type"]] += 1
        by_difficulty[row["difficulty"]] += 1
        num_images[len(row["images"])] += 1

    return {
        "total_rows": len(rows),
        "by_question_type": dict(sorted(by_question_type.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "by_num_images": dict(sorted(num_images.items())),
    }


def main() -> None:
    args = parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    stats_file = Path(args.stats_file).expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"input parquet not found: {input_file}")

    rows = load_parquet_rows(input_file)
    converted = convert_rows(
        rows,
        image_root=image_root,
        force_rewrite_images=args.force_rewrite_images,
    )
    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    stats = build_stats(converted)
    write_jsonl(output_file, converted)
    ensure_parent(stats_file)
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("MMSI-Bench local eval data ready.")
    print(f"  output_file: {output_file}")
    print(f"  image_root:  {image_root}")
    print(f"  stats_file:  {stats_file}")
    print(f"  total_rows:  {stats['total_rows']}")
    print(f"  by_type:     {stats['by_question_type']}")
    print(f"  by_diff:     {stats['by_difficulty']}")


if __name__ == "__main__":
    main()
