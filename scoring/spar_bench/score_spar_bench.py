#!/usr/bin/env python3
"""Score responses.jsonl according to SPAR-Bench official criteria."""

from __future__ import annotations

import argparse
import json
import math
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
VCI_ACTION_PAIRS = [
    ("move_right", "move_left"),
    ("move_up", "move_down"),
    ("move_forward", "move_backward"),
    ("rotate_right", "rotate_left"),
    ("rotate_up", "rotate_down"),
]
VCI_VALID_ACTIONS = {
    action for action_pair in VCI_ACTION_PAIRS for action in action_pair
}
VCI_METRIC_MODE_CHOICES = (
    "official",
    "fixed_order",
    "strict_official_order",
    "strict_fixed",
)

LOW_TASKS = {
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
}
MIDDLE_TASKS = {"view_change_infer", "position_matching", "camera_motion_infer"}
HIGH_TASKS = {
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", required=True, help="eval_results.jsonl written by evaluate.py."
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--summary-file", default=None)
    # The official scorer inherits the lmms-eval argument-order bug in mean_relative_accuracy(gt, pred),
    # Denominator follows the predicted value; empty answers or failed parses with all-zero predictions can still earn ~31 points (GT zero-axis matching).
    # fixed_order adjusts parameter ordering to minimize mean_relative_accuracy(pred, gt); empty answers no longer receive inflated scores.
    # To reproduce the official scores from the paper, pass --vci-metric-mode official.
    parser.add_argument(
        "--vci-metric-mode",
        default="fixed_order",
        choices=VCI_METRIC_MODE_CHOICES,
        help="Scoring rule for view_change_infer: 'fixed_order' uses the correct MRA argument order; 'official' keeps the reversed order used by lmms-eval.",
    )
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
    return re.sub(r"\s+", " ", strip_chat_special_tokens(value).strip())


def extract_tag_answer(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    return normalize_text(raw)


def exact_match(pred: str, target: str) -> float:
    pred_norm = normalize_text(pred).lower()
    target_norm = normalize_text(target).lower()
    if not pred_norm or not target_norm:
        return 0.0
    if pred_norm == target_norm:
        return 1.0
    if pred_norm in target_norm:
        return 1.0
    if pred_norm[0] == target_norm:
        return 1.0
    pred_letter = extract_answer(pred_norm)
    target_letter = extract_answer(target_norm)
    if (
        pred_letter is not None
        and target_letter is not None
        and pred_letter == target_letter
    ):
        return 1.0
    return 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0.0:
        return abs(pred - target)
    return abs((pred - target) / target)


def mean_relative_accuracy(pred: float, target: float) -> float:
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    norm = abs_dist_norm(pred, target)
    return sum(norm <= 1 - t for t in thresholds) / len(thresholds)


def process_na(pred: str, task: str) -> float:
    numbers = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)
    extracted = [float(num) if "." in num else int(num) for num in numbers]
    if task in {
        "depth_prediction_oc_mv",
        "depth_prediction_oo_mv",
        "distance_prediction_oc_mv",
        "distance_prediction_oo_mv",
    }:
        if not extracted:
            extracted = [-1]
        extracted = [extracted[-1]]
    if not extracted:
        raise ValueError(f"no numeric value parsed from {pred!r}")
    return float(extracted[0])


def clean_vci_instruction_text(instruction: str) -> str:
    """Clean up residual tokens from the chat template in view_change_infer answer."""

    # Strip chat-template delimiters before converting numerical answers.
    return strip_chat_special_tokens(instruction).strip()


def parse_instruction(instruction: str) -> Dict[str, float]:
    """Parse the action string of view_change_infer into a mapping from actions to numerical values."""

    parsed: Dict[str, float] = {}
    for item in clean_vci_instruction_text(instruction).split(","):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = float(value.strip())
    return parsed


def parse_strict_instruction(instruction: str) -> Dict[str, float]:
    """Strictly parse view_change_infer: 5 action groups must each provide one direction."""

    parsed = parse_instruction(instruction)
    unknown_actions = sorted(set(parsed) - VCI_VALID_ACTIONS)
    if unknown_actions:
        raise ValueError(f"unknown view_change_infer actions: {unknown_actions}")

    for positive_action, negative_action in VCI_ACTION_PAIRS:
        present_count = int(positive_action in parsed) + int(negative_action in parsed)
        if present_count != 1:
            raise ValueError(
                f"invalid view_change_infer action pair: "
                f"{positive_action}/{negative_action}, present_count={present_count}"
            )
    return parsed


def signed_vci_values(action_dict: Dict[str, float]) -> List[float]:
    """Synthesize the forward and backward actions of view_change_infer into 5 signed components."""

    signed_values = []
    for positive_action, negative_action in VCI_ACTION_PAIRS:
        signed_values.append(
            action_dict.get(positive_action, 0) - action_dict.get(negative_action, 0)
        )
    return signed_values


def compute_vci_metric(
    pred: str, answer: str, metric_mode: str = "fixed_order"
) -> float:
    """Calculate view_change_infer score based on specified aperture."""

    if metric_mode not in VCI_METRIC_MODE_CHOICES:
        raise ValueError(f"unknown view_change_infer metric mode: {metric_mode}")

    parse_func = (
        parse_strict_instruction
        if metric_mode in {"strict_official_order", "strict_fixed"}
        else parse_instruction
    )
    answer_dict = parse_func(pred)
    gt_dict = parse_func(answer)
    answer_list = signed_vci_values(answer_dict)
    gt_list = signed_vci_values(gt_dict)

    if metric_mode in {"fixed_order", "strict_fixed"}:
        # Audit correction: the first parameter of mean_relative_accuracy is the predicted value, and the second parameter is the ground truth.
        # When the official call is reversed, the denominator follows the predicted value, and raw numbers or empty parsing are more likely to receive zero rewards.
        mra_list = [
            mean_relative_accuracy(pred_v, gt_v)
            for pred_v, gt_v in zip(answer_list, gt_list)
        ]
    else:
        # Official lmms-eval metric: intentionally retains mean_relative_accuracy(gt, pred),
        # For reproducing the official table and existing eval_summary, not to be cited as corrected metric.
        mra_list = [
            mean_relative_accuracy(gt_v, pred_v)
            for pred_v, gt_v in zip(answer_list, gt_list)
        ]
    return sum(mra_list) / len(mra_list)


def score_row(row: Dict, vci_metric_mode: str = "fixed_order") -> Dict:
    """Score a single SPAR result, defaulting to fixed_order."""

    task = str(row.get("task") or "").strip()
    pred = extract_tag_answer(str(row.get("answer") or ""))
    gt = normalize_text(row.get("gt_answer"))
    result = {"score": 0.0, "parsed": False}

    if task in MCA_QUESTION_TYPES:
        result["parsed"] = bool(pred)
        result["score"] = exact_match(pred, gt)
        return result
    if task in NA_QUESTION_TYPES:
        try:
            pred_num = process_na(pred, task)
            gt_num = float(gt)
            result["parsed"] = True
            result["score"] = mean_relative_accuracy(pred_num, gt_num)
        except Exception:
            result["score"] = 0.0
        return result
    if task in SPECIAL_QUESTION_TYPES:
        try:
            result["score"] = compute_vci_metric(pred, gt, metric_mode=vci_metric_mode)
            result["parsed"] = True
        except Exception:
            # Count a row as parsed only after metric calculation succeeds.
            result["parsed"] = False
            result["score"] = 0.0
        return result
    raise ValueError(f"unknown SPAR task: {task}")


def safe_div(num: float, den: int) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def macro_subset_score(
    metric_scores: Dict[str, float],
    tasks_in_category: Sequence[str],
) -> Optional[float]:
    values = []
    for task_name in tasks_in_category:
        key = f"{task_name}__score"
        if key not in metric_scores:
            continue
        values.append(metric_scores[key])
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(rows: Sequence[Dict], vci_metric_mode: str = "fixed_order") -> Dict:
    """Aggregate SPAR scores; view_change_infer can switch between official and audit metrics."""

    task_values: Dict[str, List[float]] = defaultdict(list)
    parsed = 0

    for row in rows:
        task = str(row.get("task") or "").strip()
        scored = score_row(row, vci_metric_mode=vci_metric_mode)
        task_values[task].append(scored["score"])
        if scored["parsed"]:
            parsed += 1

    overall_metrics: Dict[str, float] = {}
    for task, values in sorted(task_values.items()):
        overall_metrics[f"{task}__score"] = sum(values) / len(values)

    # Compute the official SPAR score as a macro average over task-level scores.
    overall_accuracy = macro_subset_score(overall_metrics, sorted(task_values))

    return {
        "total_rows": len(rows),
        "parsed_rows": parsed,
        # eval_multibench reads nested path overall.overall_accuracy; here only this official reporting main metric is retained.
        "overall": {"overall_accuracy": overall_accuracy},
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_summary(result: Dict, input_file: Path) -> str:
    overall = result["overall"]
    lines = [
        "SPAR-Bench Evaluation Summary",
        f"input_file: {input_file}",
        f"total_rows: {result['total_rows']}",
        f"parsed_rows: {result['parsed_rows']}",
        f"vci_metric_mode: {result['vci_metric_mode']}",
        f"overall_accuracy: {fmt_pct(overall.get('overall_accuracy'))}",
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
    result = aggregate(rows, vci_metric_mode=args.vci_metric_mode)
    result["vci_metric_mode"] = args.vci_metric_mode
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
