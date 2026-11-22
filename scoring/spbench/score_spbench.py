#!/usr/bin/env python3
"""Score responses.jsonl according to SPBench official criteria."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
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


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", strip_chat_special_tokens(text).strip())


def extract_choice_letter(choice_line: str) -> Optional[str]:
    match = re.match(r"^\s*([A-E])[\.\)]", str(choice_line or ""), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def normalize_choice_content(text: str) -> str:
    stripped = re.sub(r"^\s*[A-E][\.\)]\s*", "", text, flags=re.IGNORECASE)
    return normalize_text(stripped).lower()


def normalize_gt_letter(raw_gt: str, choices: Sequence[str]) -> Optional[str]:
    text = normalize_text(raw_gt)
    match = re.match(r"^([A-E])(?:[\.\)]|\s|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    normalized_gt = normalize_choice_content(text)
    for choice in choices:
        letter = extract_choice_letter(choice)
        if letter and normalize_choice_content(choice) == normalized_gt:
            return letter
    return None


def extract_tag_answer(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    return normalize_text(raw)


def fuzzy_matching_num(pred: str) -> Optional[float]:
    text = normalize_text(pred).lower()
    number_words = {
        "zero": "0",
        "a": "1",
        "an": "1",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
    }
    for word, digit in number_words.items():
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            return float(digit)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0:
        return abs(pred - target)
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred: float, target: float) -> float:
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    norm = abs_dist_norm(pred, target)
    return sum(norm <= 1 - t for t in thresholds) / len(thresholds)


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def safe_div(num: float, den: int) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def score_row(row: Dict) -> Dict:
    metric_type = str(row.get("metric_type") or "").strip()
    raw_pred = extract_tag_answer(str(row.get("answer") or ""))
    gt_answer = normalize_text(row.get("gt_answer"))
    result = {
        "metric_type": metric_type,
        "raw_prediction": raw_pred,
        "score": 0.0,
        "parsed": False,
        "pred_norm": None,
        "gt_norm": None,
    }

    if metric_type == "accuracy":
        choices = row.get("choices") or []
        if not isinstance(choices, list):
            choices = [str(choices)]
        pred_letter = extract_answer(raw_pred)
        gt_letter = normalize_gt_letter(gt_answer, choices)
        if pred_letter is not None and gt_letter is not None:
            result["parsed"] = True
            result["pred_norm"] = pred_letter
            result["gt_norm"] = gt_letter
            result["score"] = 1.0 if pred_letter.upper() == gt_letter.upper() else 0.0
            return result

        normalized_pred = normalize_choice_content(raw_pred)
        normalized_gt = normalize_choice_content(gt_answer)
        result["pred_norm"] = normalized_pred
        result["gt_norm"] = normalized_gt
        if normalized_pred:
            result["parsed"] = True
            result["score"] = 1.0 if normalized_pred == normalized_gt else 0.0
        return result

    if metric_type == "mra":
        pred_num = fuzzy_matching_num(raw_pred)
        gt_num = fuzzy_matching_num(gt_answer)
        result["pred_norm"] = pred_num
        result["gt_norm"] = gt_num
        if pred_num is not None and gt_num is not None:
            result["parsed"] = True
            result["score"] = mean_relative_accuracy(pred_num, gt_num)
        return result

    raise ValueError(
        f"unsupported metric_type={metric_type!r} for row id={row.get('id')}"
    )


def aggregate_subset(rows: Sequence[Dict]) -> Dict:
    by_task_scores: Dict[str, List[float]] = defaultdict(list)
    parsed = 0

    for row in rows:
        scored = score_row(row)
        row["_score"] = scored["score"]
        row["_parsed"] = scored["parsed"]
        if scored["parsed"]:
            parsed += 1
        by_task_scores[str(row["question_type"])].append(scored["score"])

    per_task = {
        task: {
            "count": len(scores),
            "score": safe_div(sum(scores), len(scores)),
        }
        for task, scores in sorted(by_task_scores.items())
    }

    official_like_overall = mean([info["score"] for info in per_task.values()])
    return {
        "total_rows": len(rows),
        "parsed_rows": parsed,
        # Only retain task-macro scores for the final official mean within the subset.
        "official_like_overall": official_like_overall,
    }


def evaluate(rows: Sequence[Dict]) -> Dict:
    subset_groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        subset_groups[str(row.get("subset") or "")].append(dict(row))

    by_subset = {
        subset: aggregate_subset(subset_rows)
        for subset, subset_rows in sorted(subset_groups.items())
    }
    subset_official_scores = [
        info["official_like_overall"] for info in by_subset.values()
    ]

    overall_total = sum(info["total_rows"] for info in by_subset.values())
    overall_parsed = sum(info["parsed_rows"] for info in by_subset.values())

    return {
        "total_rows": overall_total,
        "parsed_rows": overall_parsed,
        # SPBench reports main metrics, retaining only the unweighted mean of two subsets.
        "combined_unweighted_subset_mean": mean(subset_official_scores),
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_summary(result: Dict, input_file: Path) -> str:
    lines = [
        "SPBench Evaluation Summary",
        f"input_file: {input_file}",
        f"total_rows: {result['total_rows']}",
        f"parsed_rows: {result['parsed_rows']}",
        f"combined_unweighted_subset_mean: {fmt_pct(result['combined_unweighted_subset_mean'])}",
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
