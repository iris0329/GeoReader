#!/usr/bin/env python3
"""Score responses.jsonl according to the MCQ format of MMSI-Bench."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# Use shared answer extraction so the scorer can run independently.
BENCHMARK_DATASETS_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DATASETS_DIR))
from common_extractors import extract_answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", required=True, help="eval_results.jsonl written by evaluate.py."
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--summary-file", default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_letter(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.match(r"^\s*([A-E])(?:[\.\)]|\s|$)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def safe_acc(correct: int, total: int) -> Optional[float]:
    if total == 0:
        return None
    return correct / total


def evaluate(rows: Sequence[Dict]) -> Dict:
    parsed = 0
    correct = 0

    for row in rows:
        gt = normalize_letter(str(row.get("gt_answer") or ""))
        if gt is None:
            raise ValueError(
                f"failed to normalize gt answer for row id={row.get('id')}"
            )

        pred = extract_answer(str(row.get("answer") or ""))

        if pred is not None:
            parsed += 1
        if pred is not None and pred.upper() == gt:
            correct += 1

    return {
        "total_rows": len(rows),
        "parsed_rows": parsed,
        # MMSI reports overall MCQ accuracy; bucket statistics are excluded from official_eval.
        "overall_accuracy": safe_acc(correct, len(rows)),
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_summary(result: Dict, input_file: Path) -> str:
    lines = [
        "MMSI-Bench Evaluation Summary",
        f"input_file: {input_file}",
        f"total_rows: {result['total_rows']}",
        f"parsed_rows: {result['parsed_rows']}",
        f"overall_accuracy: {fmt_pct(result['overall_accuracy'])}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else input_path.with_suffix(input_path.suffix + ".eval.json")
    )
    summary_file = (
        Path(args.summary_file).expanduser().resolve()
        if args.summary_file
        else input_path.with_suffix(input_path.suffix + ".summary.txt")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    result = evaluate(rows)
    summary = build_summary(result, input_path)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with summary_file.open("w", encoding="utf-8") as f:
        f.write(summary)

    print(summary, end="")
    print(f"saved json -> {output_json}")
    print(f"saved txt  -> {summary_file}")


if __name__ == "__main__":
    main()
