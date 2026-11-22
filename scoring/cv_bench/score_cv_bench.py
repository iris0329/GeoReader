#!/usr/bin/env python3
"""Score responses.jsonl according to CV-Bench official criteria."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence


EXPECTED_SOURCES = ("ADE20K", "COCO", "Omni3D")
EXPECTED_TYPES = ("2D", "3D")


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


def extract_option_letter(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    patterns = [
        r"^\s*[\(\[\{]?([A-Z])[\)\]\}]?(?:[\.\)]|\s|$)",
        r"<answer>\s*[\(\[\{]?([A-Z])[\)\]\}]?\s*</answer>",
        r"\banswer:?\s*[\(\[\{]?([A-Z])[\)\]\}]?\b",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, raw, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()
    return None


def normalize_choice_letter(text: str, choices: Sequence[str]) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.match(
        r"^\s*[\(\[\{]?([A-Z])[\)\]\}]?(?:[\.\)]|\s|$)", raw, flags=re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    normalized = normalize_choice_content(raw)
    for choice_line in choices:
        letter = extract_choice_letter(choice_line)
        content = normalize_choice_content(choice_line)
        if letter and normalized == content:
            return letter
    return None


def extract_choice_letter(choice_line: str) -> Optional[str]:
    match = re.match(
        r"^\s*[\(\[\{]?([A-Z])[\)\]\}]?\.", str(choice_line or ""), flags=re.IGNORECASE
    )
    if not match:
        return None
    return match.group(1).upper()


def normalize_choice_content(text: str) -> str:
    stripped = re.sub(
        r"^\s*[\(\[\{]?[A-Z][\)\]\}]?[\.\)]\s*",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", stripped.strip()).lower()


def safe_acc(correct: int, total: int) -> Optional[float]:
    if total == 0:
        return None
    return correct / total


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def evaluate(rows: Sequence[Dict]) -> Dict:
    source_totals: Counter = Counter()
    source_correct: Counter = Counter()

    parsed = 0

    unexpected_sources = sorted(
        {
            str(row.get("source") or "").strip()
            for row in rows
            if str(row.get("source") or "").strip() not in EXPECTED_SOURCES
        }
    )
    if unexpected_sources:
        raise ValueError(f"unsupported CV-Bench source values: {unexpected_sources}")

    unexpected_types = sorted(
        {
            str(row.get("cv_type") or "").strip()
            for row in rows
            if str(row.get("cv_type") or "").strip() not in EXPECTED_TYPES
        }
    )
    if unexpected_types:
        raise ValueError(f"unsupported CV-Bench type values: {unexpected_types}")

    for row in rows:
        source = str(row["source"]).strip()
        choices = row.get("choices") or []
        if not isinstance(choices, list):
            choices = [str(choices)]

        pred_letter = extract_option_letter(str(row.get("answer") or ""))
        gt_letter = normalize_choice_letter(str(row.get("gt_answer") or ""), choices)
        if gt_letter is None:
            gt_letter = normalize_choice_letter(
                str(row.get("raw_answer") or ""), choices
            )
        if gt_letter is None:
            raise ValueError(
                f"failed to normalize gt answer for row id={row.get('id')}"
            )

        source_totals[source] += 1

        if pred_letter is not None:
            parsed += 1
        if pred_letter is not None and pred_letter.upper() == gt_letter.upper():
            source_correct[source] += 1

    source_accuracy = {
        source: safe_acc(source_correct[source], source_totals[source])
        for source in EXPECTED_SOURCES
    }

    # Official report only view source bucket -> 2D/3D -> combined; no longer include pooled exact-match, to avoid misinterpretation as primary result.
    two_d_accuracy = mean([source_accuracy["ADE20K"], source_accuracy["COCO"]])
    three_d_accuracy = source_accuracy["Omni3D"]
    combined_accuracy = mean([two_d_accuracy, three_d_accuracy])

    return {
        "total_rows": len(rows),
        "parsed_rows": parsed,
        # CV-Bench official reporting metrics; source/2D/3D are only for internal computation and not written into result files.
        "combined_accuracy": combined_accuracy,
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_summary(result: Dict, input_file: Path) -> str:
    lines = [
        "CV-Bench Evaluation Summary",
        f"input_file: {input_file}",
        f"total_rows: {result['total_rows']}",
        f"parsed_rows: {result['parsed_rows']}",
        f"combined_accuracy: {fmt_pct(result['combined_accuracy'])}",
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
