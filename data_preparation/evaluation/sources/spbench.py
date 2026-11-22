#!/usr/bin/env python3
"""Convert SPBench parquet to current standard eval JSONL and automatically unzip images."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


# Public entry does not depend on the original cluster path; formal invocation via CLI explicitly passes the official raw location.
DEFAULT_DATA_ROOT = Path("raw/evaluation/spbench")
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_ROOT / "processed" / "spbench_test_local.jsonl"
DEFAULT_STATS_FILE = DEFAULT_DATA_ROOT / "meta" / "spbench_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="SPBench root directory with raw/, processed/ and meta/.",
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
        help="Keep only the first N merged samples (0 = all).",
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


def normalize_choice_lines(choices: object) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, str):
        parts = [normalize_text(x) for x in choices.splitlines()]
        return [x for x in parts if x]
    if isinstance(choices, (list, tuple)):
        return [normalize_text(x) for x in choices if normalize_text(x)]
    return [normalize_text(choices)]


def build_question(question: str, choice_lines: Sequence[str]) -> str:
    question = normalize_text(question)
    if choice_lines:
        return f"{question} {' '.join(choice_lines)}"
    return question


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


def resolve_image_path(scene_name: str, image_name: str, roots: Sequence[Path]) -> Path:
    image_name = normalize_text(image_name)
    scene_name = normalize_text(scene_name)
    for root in roots:
        candidates = [
            root / scene_name / image_name,
            root / "images" / scene_name / image_name,
            root / root.name / scene_name / image_name,
        ]
        for child in root.iterdir() if root.exists() else []:
            if child.is_dir():
                candidates.append(child / scene_name / image_name)
                candidates.append(child / "images" / scene_name / image_name)
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = [
            match for match in root.rglob(image_name) if scene_name in match.parts
        ]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"failed to resolve image '{image_name}' under scene '{scene_name}' from roots={roots}"
    )


def normalize_subset_tag(subset_name: str) -> str:
    return subset_name.lower().replace("-", "_")


def convert_rows(
    rows: Iterable[Dict],
    *,
    subset_name: str,
    data_root: Path,
    image_roots: Sequence[Path],
) -> List[Dict]:
    subset_tag = normalize_subset_tag(subset_name)
    converted: List[Dict] = []
    seen_ids = set()

    for row in rows:
        raw_id = int(row["id"])
        scene_name = normalize_text(row.get("scene_name"))
        question_type = normalize_text(row.get("question_type"))
        choice_lines = normalize_choice_lines(row.get("options"))
        image_names = row.get("images") or []
        if not isinstance(image_names, (list, tuple)) or not image_names:
            raise ValueError(
                f"invalid images field for subset={subset_name}, id={raw_id}"
            )

        rel_image_paths: List[str] = []
        for image_name in image_names:
            image_path = resolve_image_path(scene_name, str(image_name), image_roots)
            rel_image_paths.append(image_path.relative_to(data_root).as_posix())

        sample_id = f"{subset_tag}_{raw_id}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicated sample id: {sample_id}")
        seen_ids.add(sample_id)

        converted.append(
            {
                "id": sample_id,
                "question": build_question(
                    str(row.get("question") or ""), choice_lines
                ),
                "images": rel_image_paths,
                "gt_answer": normalize_text(row.get("ground_truth")),
                "subset": subset_name,
                "task": question_type,
                "question_type": question_type,
                "metric_type": "accuracy" if choice_lines else "mra",
                "choices": choice_lines,
                "scene_name": scene_name,
                "source_dataset": normalize_text(row.get("dataset")),
                "original_id": raw_id,
            }
        )
    return converted


def build_stats(rows: Sequence[Dict]) -> Dict:
    by_subset = Counter()
    by_task = Counter()
    by_metric = Counter()
    by_num_images = Counter()

    for row in rows:
        by_subset[row["subset"]] += 1
        by_task[f"{row['subset']}::{row['question_type']}"] += 1
        by_metric[row["metric_type"]] += 1
        by_num_images[len(row["images"])] += 1

    return {
        "total_rows": len(rows),
        "by_subset": dict(sorted(by_subset.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_metric": dict(sorted(by_metric.items())),
        "by_num_images": dict(sorted(by_num_images.items())),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    output_file = Path(args.output_file).expanduser().resolve()
    stats_file = Path(args.stats_file).expanduser().resolve()

    si_parquet = raw_dir / "SPBench-SI.parquet"
    mv_parquet = raw_dir / "SPBench-MV.parquet"
    if not si_parquet.exists() or not mv_parquet.exists():
        raise FileNotFoundError(
            "missing SPBench parquet files.\n"
            f"expected: {si_parquet}\n"
            f"expected: {mv_parquet}"
        )

    extracted_roots: Dict[str, List[Path]] = {"SPBench-SI": [], "SPBench-MV": []}
    if not args.skip_extract:
        si_zip = raw_dir / "SPBench-SI-images.zip"
        mv_zip = raw_dir / "SPBench-MV-images.zip"
        if not si_zip.exists() or not mv_zip.exists():
            raise FileNotFoundError(
                "missing SPBench image zip files. Please run download_spbench_assets.py first."
            )
        extracted_roots["SPBench-SI"].append(
            extract_zip(
                si_zip,
                raw_dir / "extracted" / "spbench_si",
                force_extract=args.force_extract,
            )
        )
        extracted_roots["SPBench-MV"].append(
            extract_zip(
                mv_zip,
                raw_dir / "extracted" / "spbench_mv",
                force_extract=args.force_extract,
            )
        )

    for subset_name, extra_dir in (
        ("SPBench-SI", raw_dir / "extracted" / "spbench_si"),
        ("SPBench-MV", raw_dir / "extracted" / "spbench_mv"),
    ):
        if extra_dir.exists():
            extracted_roots[subset_name].append(extra_dir)

    si_rows = convert_rows(
        load_parquet_rows(si_parquet),
        subset_name="SPBench-SI",
        data_root=data_root,
        image_roots=extracted_roots["SPBench-SI"],
    )
    mv_rows = convert_rows(
        load_parquet_rows(mv_parquet),
        subset_name="SPBench-MV",
        data_root=data_root,
        image_roots=extracted_roots["SPBench-MV"],
    )
    merged = si_rows + mv_rows
    if args.max_samples > 0:
        merged = merged[: args.max_samples]

    stats = build_stats(merged)
    write_jsonl(output_file, merged)
    ensure_parent(stats_file)
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("SPBench local eval data ready.")
    print(f"  output_file: {output_file}")
    print(f"  stats_file:  {stats_file}")
    print(f"  total_rows:  {stats['total_rows']}")
    print(f"  by_subset:   {stats['by_subset']}")
    print(f"  by_metric:   {stats['by_metric']}")


if __name__ == "__main__":
    main()
