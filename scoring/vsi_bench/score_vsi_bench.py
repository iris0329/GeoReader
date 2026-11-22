#!/usr/bin/env python3
"""Score responses.jsonl according to VSI-Bench official criteria."""

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
# Expose the scorer directory when loading it through importlib.
VSI_SCORER_DIR = Path(__file__).resolve().parent
if str(VSI_SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(VSI_SCORER_DIR))
from common_extractors import extract_answer, strip_chat_special_tokens  # noqa: E402

MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


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
    # Preserve newlines for last-line option extraction.
    return strip_chat_special_tokens(value).strip()


def extract_tag_answer(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    return normalize_text(raw)


def fuzzy_matching_mca(pred: str) -> str:
    raw = normalize_text(pred)
    letter = extract_answer(raw)
    if letter is not None:
        return letter
    return raw.split(" ")[0].rstrip(".").strip().lower()


def fuzzy_matching_num(pred: str) -> Optional[float]:
    # Digital extraction is independent of MCQ, avoid articles, frame numbers, and object attribute digits from competing for the final answer.
    from vsi_answer import extract_number

    return extract_number(pred)


def exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0:
        return abs(pred - target)
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred: float, target: float) -> float:
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    norm = abs_dist_norm(pred, target)
    return sum(norm <= 1 - t for t in thresholds) / len(thresholds)


def score_row(row: Dict) -> Dict:
    task = str(row.get("question_type") or row.get("task") or "").strip()
    raw_pred = extract_tag_answer(str(row.get("answer") or ""))
    gt = normalize_text(row.get("gt_answer"))
    result = {"score": 0.0, "parsed": False}

    if task in MCA_QUESTION_TYPES:
        pred_norm = fuzzy_matching_mca(raw_pred)
        gt_norm = fuzzy_matching_mca(gt)
        result["parsed"] = pred_norm in {"A", "B", "C", "D"}
        result["score"] = exact_match(pred_norm, gt_norm)
        return result

    if task in NA_QUESTION_TYPES:
        pred_num = fuzzy_matching_num(raw_pred)
        gt_num = fuzzy_matching_num(gt)
        if pred_num is not None and gt_num is not None:
            result["parsed"] = True
            result["score"] = mean_relative_accuracy(pred_num, gt_num)
        return result

    raise ValueError(f"unknown VSI question type: {task}")


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def aggregate(rows: Sequence[Dict]) -> Dict:
    per_task_values: Dict[str, List[float]] = defaultdict(list)
    parsed = 0

    for row in rows:
        task = str(row.get("question_type") or row.get("task") or "").strip()
        scored = score_row(row)
        per_task_values[task].append(scored["score"])
        if scored["parsed"]:
            parsed += 1

    output = {
        f"{task}": sum(values) / len(values)
        for task, values in sorted(per_task_values.items())
    }

    if (
        "object_rel_direction" not in output
        and "object_rel_direction_easy" in output
        and "object_rel_direction_medium" in output
        and "object_rel_direction_hard" in output
    ):
        output["object_rel_direction"] = mean(
            [
                output["object_rel_direction_easy"],
                output["object_rel_direction_medium"],
                output["object_rel_direction_hard"],
            ]
        )

    metric_values = []
    for task_name, value in output.items():
        if task_name in {
            "object_rel_direction_easy",
            "object_rel_direction_medium",
            "object_rel_direction_hard",
        }:
            continue
        metric_values.append(value)

    return {
        "total_rows": len(rows),
        "parsed_rows": parsed,
        # The final table for VSI-Bench only references overall_accuracy; MCA/NA/task bucketed results are not written to official_eval.
        "overall_accuracy": mean(metric_values),
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_summary(result: Dict, input_file: Path) -> str:
    lines = [
        "VSI-Bench Evaluation Summary",
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
    result = aggregate(rows)
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
