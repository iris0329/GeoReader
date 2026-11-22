#!/usr/bin/env python3
"""Score responses.jsonl according to ViewSpatial-Bench official criteria."""

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
from common_extractors import extract_answer, strip_chat_special_tokens  # noqa: E402


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


def normalize_text(value: object) -> str:
    # Preserve line boundaries when extracting answer options.
    return strip_chat_special_tokens(value).strip()


def normalize_choice_content(text: str) -> str:
    stripped = re.sub(r"^\s*[A-E][\.\)]\s*", "", text, flags=re.IGNORECASE)
    # Only merge whitespace when comparing option text, without affecting the recognition of the last line option above.
    return re.sub(r"\s+", " ", normalize_text(stripped)).lower()


def extract_choice_letter(choice_line: str) -> Optional[str]:
    match = re.match(
        r"^\s*([A-E])(?:[\.\)]|\s|$)", str(choice_line or ""), flags=re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    return None


def extract_tag_answer(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    return normalize_text(raw)


def score_row(row: Dict) -> Dict:
    raw_pred = extract_tag_answer(str(row.get("answer") or ""))
    pred_letter = extract_answer(raw_pred)
    gt_letter = extract_choice_letter(str(row.get("gt_answer") or ""))
    if gt_letter is None:
        gt_letter = extract_choice_letter(str(row.get("raw_answer") or ""))

    result = {"score": 0.0, "parsed": False, "pred_norm": None, "gt_norm": gt_letter}
    if pred_letter is not None and gt_letter is not None:
        result["score"] = 1.0 if pred_letter.upper() == gt_letter.upper() else 0.0
        result["parsed"] = True
        result["pred_norm"] = pred_letter
        return result

    normalized_pred = normalize_choice_content(raw_pred)
    normalized_gt = normalize_choice_content(str(row.get("raw_answer") or ""))
    if normalized_pred:
        result["parsed"] = True
        result["pred_norm"] = normalized_pred
        result["gt_norm"] = normalized_gt
        result["score"] = 1.0 if normalized_pred == normalized_gt else 0.0
    return result


def safe_acc(correct: float, total: int) -> Optional[float]:
    if total == 0:
        return None
    return correct / total


def evaluate(rows: Sequence[Dict]) -> Dict:
    parsed = 0
    correct = 0.0

    for row in rows:
        scored = score_row(row)
        if scored["parsed"]:
            parsed += 1
        if scored["score"] > 0:
            correct += 1

    return {
        "total_rows": len(rows),
        "parsed_rows": parsed,
        # ViewSpatial reports only overall accuracy; task/source/person/camera bucketing does not write to official_eval.
        "overall_accuracy": safe_acc(correct, len(rows)),
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_summary(result: Dict, input_file: Path) -> str:
    lines = [
        "ViewSpatial-Bench Evaluation Summary",
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
