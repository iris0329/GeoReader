#!/usr/bin/env python3
"""Convert CV-Bench parquet to current standard eval JSONL and export images to local directory."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image


# Public entry does not depend on the original cluster path; formal invocation via CLI explicitly passes the official raw location.
DEFAULT_DATA_ROOT = Path("raw/evaluation/cvbench")
DEFAULT_RAW_DIR = DEFAULT_DATA_ROOT / "raw"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_ROOT / "processed" / "cv_bench_test_local.jsonl"
DEFAULT_IMAGE_ROOT = DEFAULT_DATA_ROOT / "processed" / "images"
DEFAULT_STATS_FILE = DEFAULT_DATA_ROOT / "meta" / "cv_bench_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory with test_2d.parquet and test_3d.parquet.",
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
        help="Keep only the first N samples after merging 2D and 3D (0 = all).",
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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def normalize_choice_lines(choices: object) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, str):
        text = normalize_whitespace(choices)
        matches = re.findall(r"([A-Z]\.\s*.*?)(?=(?:\s+[A-Z]\.\s)|$)", text)
        if matches:
            return [normalize_whitespace(match) for match in matches]
        return [text] if text else []
    if isinstance(choices, (list, tuple)):
        normalized = [
            normalize_whitespace(item) for item in choices if normalize_whitespace(item)
        ]
        if normalized and all(
            not re.match(r"^\s*[A-Z][\.\)]", item, flags=re.IGNORECASE)
            for item in normalized
        ):
            return [
                f"{chr(ord('A') + idx)}. {item}" for idx, item in enumerate(normalized)
            ]
        return normalized
    return [normalize_whitespace(str(choices))]


def build_question(row: Dict, choice_lines: List[str]) -> str:
    prompt = normalize_whitespace(str(row.get("prompt") or ""))
    if prompt:
        return prompt
    question = normalize_whitespace(str(row.get("question") or ""))
    if not question:
        raise ValueError(f"row missing question/prompt: idx={row.get('idx')!r}")
    if choice_lines:
        return f"{question} {' '.join(choice_lines)}"
    return question


def extract_bytes_from_image_field(image_field: object) -> Optional[bytes]:
    if image_field is None:
        return None
    if isinstance(image_field, bytes):
        return image_field
    if isinstance(image_field, bytearray):
        return bytes(image_field)
    if isinstance(image_field, memoryview):
        return image_field.tobytes()
    if isinstance(image_field, dict):
        image_bytes = image_field.get("bytes")
        if isinstance(image_bytes, bytes):
            return image_bytes
        if isinstance(image_bytes, bytearray):
            return bytes(image_bytes)
        if isinstance(image_bytes, memoryview):
            return image_bytes.tobytes()
    return None


def save_image_field(
    image_field: object,
    output_path: Path,
    force_rewrite: bool,
) -> None:
    ensure_parent(output_path)
    if output_path.exists() and not force_rewrite:
        return

    image_bytes = extract_bytes_from_image_field(image_field)
    if image_bytes is not None:
        output_path.write_bytes(image_bytes)
        return

    if isinstance(image_field, Image.Image):
        image_field.save(output_path)
        return

    raise ValueError(
        f"unsupported image field type: {type(image_field).__name__} for {output_path}"
    )


def sanitize_filename(path_like: str, *, sample_id: str) -> str:
    text = normalize_whitespace(path_like)
    if not text:
        raise ValueError(f"missing filename for sample_id={sample_id}")
    filename = Path(text).name
    if not filename:
        raise ValueError(f"invalid filename={path_like!r} for sample_id={sample_id}")
    return filename


def normalize_gt_answer(answer: object, choice_lines: Sequence[str]) -> str:
    text = normalize_whitespace(str(answer or ""))
    if not text:
        raise ValueError("empty ground-truth answer")

    m = re.match(
        r"^[\(\[\{]?([A-Z])[\)\]\}]?(?:[\.\)]|\s|$)", text, flags=re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    normalized_gt = normalize_choice_content(text)
    for line in choice_lines:
        letter = extract_choice_letter(line)
        content = normalize_choice_content(line)
        if letter and normalized_gt == content:
            return letter

    raise ValueError(f"failed to normalize gt answer: {text!r}")


def extract_choice_letter(choice_line: str) -> Optional[str]:
    m = re.match(r"^\s*[\(\[\{]?([A-Z])[\)\]\}]?\.", choice_line, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper()


def normalize_choice_content(text: str) -> str:
    stripped = re.sub(
        r"^\s*[\(\[\{]?[A-Z][\)\]\}]?[\.\)]\s*", "", text, flags=re.IGNORECASE
    )
    return normalize_whitespace(stripped).lower()


def convert_rows(
    rows: Iterable[Dict],
    image_root: Path,
    force_rewrite_images: bool,
) -> List[Dict]:
    converted: List[Dict] = []
    seen_ids = set()

    for row in rows:
        idx = row.get("idx")
        cv_type = normalize_whitespace(str(row.get("type") or ""))
        task = normalize_whitespace(str(row.get("task") or ""))
        source = normalize_whitespace(str(row.get("source") or ""))
        if not cv_type:
            raise ValueError(f"missing type for idx={idx!r}")
        if not task:
            raise ValueError(f"missing task for idx={idx!r}")
        if not source:
            raise ValueError(f"missing source for idx={idx!r}")

        sample_id = f"cvbench_{cv_type.lower()}_{idx}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicated sample id: {sample_id}")
        seen_ids.add(sample_id)

        choice_lines = normalize_choice_lines(row.get("choices"))
        question = build_question(row, choice_lines)
        gt_answer = normalize_gt_answer(row.get("answer"), choice_lines)

        filename = sanitize_filename(
            str(row.get("filename") or ""),
            sample_id=sample_id,
        )
        # Include the task in each image path to separate different images or markings with the same filename.
        rel_image_path = Path(source) / cv_type / task.lower() / filename
        out_image_path = image_root / rel_image_path
        save_image_field(
            row.get("image"),
            out_image_path,
            force_rewrite=force_rewrite_images,
        )

        converted.append(
            {
                "id": sample_id,
                "question": question,
                "images": [rel_image_path.as_posix()],
                "gt_answer": gt_answer,
                "raw_answer": normalize_whitespace(str(row.get("answer") or "")),
                "cv_type": cv_type,
                "task": task,
                "source": source,
                "source_dataset": normalize_whitespace(
                    str(row.get("source_dataset") or "")
                ),
                "filename": filename,
                # Preserve source media identity for overlap checks.
                "source_filename": row["source_filename"],
                "choices": choice_lines,
                "original_idx": idx,
            }
        )

    return converted


def build_stats(rows: Sequence[Dict]) -> Dict:
    by_type = Counter()
    by_source = Counter()
    by_task = Counter()
    by_type_source: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        cv_type = row["cv_type"]
        source = row["source"]
        task = row["task"]
        by_type[cv_type] += 1
        by_source[source] += 1
        by_task[task] += 1
        by_type_source[cv_type][source] += 1

    return {
        "total_rows": len(rows),
        "by_type": dict(sorted(by_type.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_type_source": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_type_source.items())
        },
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    stats_file = Path(args.stats_file).expanduser().resolve()

    parquet_paths = [
        raw_dir / "test_2d.parquet",
        raw_dir / "test_3d.parquet",
    ]
    for path in parquet_paths:
        if not path.exists():
            raise FileNotFoundError(f"required parquet not found: {path}")

    merged_rows: List[Dict] = []
    for path in parquet_paths:
        merged_rows.extend(load_parquet_rows(path))

    converted = convert_rows(
        merged_rows,
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

    print("CV-Bench local eval data ready.")
    print(f"  output_file: {output_file}")
    print(f"  image_root:  {image_root}")
    print(f"  stats_file:  {stats_file}")
    print(f"  total_rows:  {stats['total_rows']}")
    print(f"  by_type:     {stats['by_type']}")
    print(f"  by_source:   {stats['by_source']}")


if __name__ == "__main__":
    main()
