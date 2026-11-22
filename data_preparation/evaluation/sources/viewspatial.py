#!/usr/bin/env python3
"""Convert ViewSpatial-Bench annotations and images ZIP to current standard eval JSONL format."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


# Public entry does not depend on the original cluster path; formal invocation via CLI explicitly passes the official raw location.
DEFAULT_DATA_ROOT = Path("raw/evaluation/viewspatial")
DEFAULT_INPUT_FILE = DEFAULT_DATA_ROOT / "raw" / "ViewSpatial-Bench.json"
DEFAULT_OUTPUT_FILE = (
    DEFAULT_DATA_ROOT / "processed" / "viewspatial_bench_test_local.jsonl"
)
DEFAULT_STATS_FILE = DEFAULT_DATA_ROOT / "meta" / "viewspatial_bench_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="ViewSpatial-Bench root directory with raw/, processed/ and meta/.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Input ViewSpatial-Bench.json file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL in the GeoReader evaluation format.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=str(DEFAULT_STATS_FILE),
        help="Output JSON with summary statistics.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not extract the image zip files.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract the image zip files again even if they were already extracted.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Keep only the first N rows (0 = all).",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_choice_lines(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [normalize_text(x) for x in value.splitlines()]
        return [x for x in lines if x]
    if isinstance(value, (list, tuple)):
        return [normalize_text(x) for x in value if normalize_text(x)]
    return [normalize_text(value)]


def normalize_letter(value: str) -> str:
    match = re.match(r"^\s*([A-E])(?:[\.\)]|\s|$)", value, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"failed to normalize letter from {value!r}")
    return match.group(1).upper()


def extract_zip(zip_path: Path, output_dir: Path, force_extract: bool) -> Path:
    marker = output_dir / ".extract_complete"
    if output_dir.exists() and marker.exists() and not force_extract:
        return output_dir

    if output_dir.exists() and force_extract:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return output_dir


def load_json_rows(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_relative_image_path(image_path: str) -> Path:
    normalized = normalize_text(image_path)
    prefix = "ViewSpatial-Bench/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    path_obj = Path(normalized)
    if not path_obj.parts:
        raise ValueError(f"invalid image_path: {image_path!r}")
    top = path_obj.parts[0]
    return Path("raw") / "extracted" / top / path_obj


def build_question(question: str, choice_lines: Sequence[str]) -> str:
    if not choice_lines:
        return normalize_text(question)
    return f"{normalize_text(question)} {' '.join(choice_lines)}"


def convert_rows(rows: Iterable[Dict], data_root: Path) -> List[Dict]:
    converted: List[Dict] = []
    for index, row in enumerate(rows):
        image_paths = row.get("image_path") or []
        if not isinstance(image_paths, (list, tuple)) or not image_paths:
            raise ValueError(f"invalid image_path field at row {index}")
        rel_paths = [
            resolve_relative_image_path(str(path)).as_posix() for path in image_paths
        ]
        for rel_path in rel_paths:
            if not (data_root / rel_path).exists():
                raise FileNotFoundError(
                    f"missing resolved image file: {data_root / rel_path}"
                )

        choice_lines = normalize_choice_lines(row.get("choices"))
        raw_answer = normalize_text(row.get("answer"))
        converted.append(
            {
                "id": f"viewspatial_{index}",
                "question": build_question(
                    str(row.get("question") or ""), choice_lines
                ),
                "images": rel_paths,
                "gt_answer": normalize_letter(raw_answer),
                "raw_answer": raw_answer,
                "choices": choice_lines,
                "task": normalize_text(row.get("question_type")),
                "question_type": normalize_text(row.get("question_type")),
                "source_split": "scannetv2_val"
                if "scannetv2_val" in rel_paths[0]
                else "val2017",
                "original_index": index,
            }
        )
    return converted


def build_stats(rows: Sequence[Dict]) -> Dict:
    by_task = Counter()
    by_source = Counter()
    by_num_images = Counter()

    for row in rows:
        by_task[row["question_type"]] += 1
        by_source[row["source_split"]] += 1
        by_num_images[len(row["images"])] += 1

    return {
        "total_rows": len(rows),
        "by_task": dict(sorted(by_task.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_num_images": dict(sorted(by_num_images.items())),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    stats_file = Path(args.stats_file).expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"input json not found: {input_file}")

    if not args.skip_extract:
        scannet_zip = raw_dir / "scannetv2_val.zip"
        coco_zip = raw_dir / "val2017.zip"
        if not scannet_zip.exists() or not coco_zip.exists():
            raise FileNotFoundError(
                "missing ViewSpatial-Bench image zip files. Please run download_viewspatial_bench_assets.py first."
            )
        extract_zip(
            scannet_zip,
            raw_dir / "extracted" / "scannetv2_val",
            force_extract=args.force_extract,
        )
        extract_zip(
            coco_zip,
            raw_dir / "extracted" / "val2017",
            force_extract=args.force_extract,
        )

    rows = load_json_rows(input_file)
    converted = convert_rows(rows, data_root=data_root)
    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    stats = build_stats(converted)
    write_jsonl(output_file, converted)
    ensure_parent(stats_file)
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("ViewSpatial-Bench local eval data ready.")
    print(f"  output_file: {output_file}")
    print(f"  stats_file:  {stats_file}")
    print(f"  total_rows:  {stats['total_rows']}")
    print(f"  by_task:     {stats['by_task']}")


if __name__ == "__main__":
    main()
