"""Six-benchmark data loading, original prompts and official scoring, without model loading."""

from __future__ import annotations

import json

import math

import os

import re

import subprocess

import sys

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, List, Optional, Sequence

import torch

from PIL import Image

from stage2_config import (
    SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_AFTER_QUESTION,
    SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION,
    SPATIAL_FUSION_TYPE_LATENT_TOKENS,
)

from coordinate_rewriter import (
    coordinate_rewrite_plan_for_record,
    get_processed_size,
    rewrite_coordinates_in_text,
)

from main_camera_interpreter import (
    _assemble_camera_prompt,
    _build_camera_token_block,
    _build_depth_token_block,
    _resolve_record_pose_path,
    _resolve_record_depth_path,
)

from georeader_data import resolve_record_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_benchmark(data_root, name):
    """Only list files from the downloaded and extracted data root directory, without referencing the benchmark directory on the training machine."""
    root = Path(data_root).expanduser().resolve()
    manifest = root / "manifest.json"
    if (
        not manifest.is_file()
        or json.loads(manifest.read_text()).get("format") != "georeader-sixbench"
    ):
        raise RuntimeError(f"Not a GeoReader six-benchmark dataset root: {root}")
    if name not in BENCHMARK_CHOICES:
        raise ValueError(f"Unknown benchmark: {name}")
    path = root / "eval" / name / "qa.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return BenchmarkConfig(
        name=name, display_name=name, jsonl_path=path, image_root=root
    )


BENCHMARK_CHOICES = ("vsi", "mmsi", "spbench", "viewspatial", "cvbench", "spar")

MCQ_LETTERS = set("ABCDEF")

MCQ_ANSWER_INSTRUCTION = "Answer with the option letter only."

CVBENCH_MCQ_ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)

NUMERIC_ANSWER_INSTRUCTION = "Answer with a single number only."

SPAR_OPTION_COORD_RE = re.compile(
    r"\b[A-F]\.\s*\[\s*-?\d{1,4}\s*,\s*-?\d{1,4}\s*,\s*-?\d{1,4}\s*,\s*-?\d{1,4}\s*\]"
)

SPAR_POINT_COORD_RE = re.compile(r"\(\s*\d{1,5}\s*,\s*\d{1,5}\s*\)")

COORDINATE_REWRITE_AUDIT_NOTE = "Note: The image path in eval_results.jsonl remains the original image path; the Qwen processor will resize it dynamically during evaluation. The question field retains the original benchmark question text, while model_prompt_question is the actual question fed to the model. Coordinates have been converted to Qwen-processed image pixels and cannot be directly plotted on the original image to assess correctness."

SPAR_COORDINATE_TASK_EXPECTATIONS = {
    # ── SPAR-Bench Coordinate Space Verification ──
    # SPAR-Bench's position_matching and camera_motion_infer two subtasks problem statements
    # Include normalized bbox/point coordinates to [0, 1000], to be processed by coordinate_rewriter.py during evaluation.
    # Convert to the pixel space actually output by the Qwen processor.
    #
    # Here is not redefining task routing; the actual "change which image/coordinate system" comes from coordinate_rewriter.py.
    # eval only validates the SPAR-Bench problem format (regex matching + key phrases) to ensure the benchmark file structure
    # After changing, the prompt won't silently correct errors anymore.
    "position_matching": {
        "coord_re": SPAR_OPTION_COORD_RE,
        "missing_desc": "A./B. format bbox options",
        "required_phrase": "second image",
        "original_coordinate_space": "normalized_1000_bbox_on_second_image",
        "target_image_role": "second_image",
    },
    "camera_motion_infer": {
        "coord_re": SPAR_POINT_COORD_RE,
        "missing_desc": "(x,y) point option",
        "required_phrase": "first image",
        "original_coordinate_space": "normalized_1000_point_on_first_image",
        "target_image_role": "first_image",
    },
}

VSI_MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}

VSI_NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}

VSI_VIDEO_CONTEXT = "These are frames of a video."

VSI_MCQ_ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)

VSI_NUMERIC_ANSWER_INSTRUCTION = (
    "Please answer the question using a single word or phrase."
)

BENCHMARK_SCORER_CONFIG = {
    "cvbench": (
        "cv_bench/score_cv_bench.py",
        ("combined_accuracy",),
        "combined_accuracy",
    ),
    "viewspatial": (
        "viewspatial_bench/score_viewspatial_bench.py",
        ("overall_accuracy",),
        "overall_accuracy",
    ),
    "mmsi": (
        "mmsi_bench/score_mmsi_bench.py",
        ("overall_accuracy",),
        "overall_accuracy",
    ),
    "spar": (
        "spar_bench/score_spar_bench.py",
        ("overall", "overall_accuracy"),
        "overall.overall_accuracy",
    ),
    "spbench": (
        "spbench/score_spbench.py",
        ("combined_unweighted_subset_mean",),
        "combined_unweighted_subset_mean",
    ),
    "vsi": ("vsi_bench/score_vsi_bench.py", ("overall_accuracy",), "overall_accuracy"),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    display_name: str
    jsonl_path: Path
    image_root: Path
    # Geometry is supplied per record; the dataset-level pose directory must be None.
    pose_dir: Optional[Path] = None


def _load_records(jsonl_path: Path, max_samples: Optional[int]) -> List[dict]:
    """Load samples by line number in JSONL; pose/depth are fully specified by complete paths within each line."""
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"benchmark JSONL not found: {jsonl_path}")
    records: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            record = resolve_record_paths(json.loads(line), jsonl_path)
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Sample must be dict: path={jsonl_path}, line={line_idx}"
                )
            record["_line_idx"] = line_idx
            _validate_record(record, jsonl_path=jsonl_path, line_idx=line_idx)
            records.append(record)
            if max_samples is not None and len(records) >= int(max_samples):
                break
    if not records:
        raise RuntimeError(f"benchmark JSONL is empty: {jsonl_path}")
    return records


def _validate_record(record: dict, jsonl_path: Path, line_idx: int):
    """Check benchmark base fields and unified modality path contract; bad paths must error during loading."""
    sample_id = record.get("id", f"line_{line_idx}")
    question = str(record.get("question", "")).strip()
    if not question:
        raise RuntimeError(
            f"Sample question is empty: path={jsonl_path}, sample={sample_id}"
        )
    images = record.get("images") or record.get("image_input")
    if not isinstance(images, list) or not images:
        raise RuntimeError(
            f"Sample images must be a non-empty list: path={jsonl_path}, sample={sample_id}"
        )
    if "n_images" in record and int(record["n_images"]) != len(images):
        raise RuntimeError(
            f"n_images and images count do not match: path={jsonl_path}, sample={sample_id}, n_images={record['n_images']}, images={len(images)}"
        )
    # Validate has_pose and has_depth through the shared geometry path resolver.
    for legacy_field in ("pose_available", "depth_available"):
        if legacy_field in record:
            raise RuntimeError(
                f"eval new contract no longer receives {legacy_field}: path={jsonl_path}, sample={sample_id}; changed to has_pose/has_depth + full path"
            )
    context = f"eval path={jsonl_path}, line={line_idx}, sample={sample_id}"
    pose_path = _resolve_record_pose_path(record, context=context)
    depth_path = _resolve_record_depth_path(record, context=context)
    if pose_path is not None and len(images) <= 1:
        raise RuntimeError(
            f"Single-image eval samples cannot write pose_path: {context}"
        )
    record["_pose_path"] = pose_path
    record["_depth_path"] = depth_path
    record["_has_pose"] = int(pose_path is not None)
    record["_has_depth"] = int(depth_path is not None)
    gt_answer = str(record.get("gt_answer", "")).strip()
    if not gt_answer:
        raise RuntimeError(
            f"Sample gt_answer is empty: path={jsonl_path}, sample={sample_id}"
        )


def _resolve_image_path(image_root: Path, raw_path: str) -> str:
    raw_path = os.path.expanduser(str(raw_path))
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(str(image_root), raw_path)


def _answer_type_from_gt(gt_answer: str) -> str:
    normalized = str(gt_answer).strip().upper()
    if normalized in MCQ_LETTERS:
        return "mcq"
    _parse_required_number(normalized, context="gt_answer")
    return "numeric"


def _extract_number(text: str) -> Optional[float]:
    """Extract the first number from the model output; return None if no number is present, indicating prediction failure."""
    clean = re.sub(r"<\|[^>]+?\|>", " ", str(text))
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", clean)
    if match is None:
        return None
    value = float(match.group(0))
    if not math.isfinite(value):
        raise RuntimeError(
            f"Model output is not a finite number: value={value}, text={text!r}"
        )
    return value


def _parse_required_number(text: str, context: str) -> float:
    value = _extract_number(str(text))
    if value is None:
        raise RuntimeError(f"{context} Not a parseable number: {text!r}")
    return value


def _load_rgb_images(
    image_paths: List[str], *, benchmark_name: str, sample_id: str
) -> List[Image.Image]:
    """Shared bounded parallel decoding, preserving frame order and missing image error semantics."""
    from eval_images import load_rgb_images

    return load_rgb_images(
        image_paths, benchmark_name=benchmark_name, sample_id=sample_id
    )


def _build_prompt_text(
    record: dict,
    benchmark: BenchmarkConfig,
    camera_latent_size: int,
    use_camera_tokens: bool,
    cv_prompt_style: str,
    camera_token_position: str = "before_question",
    depth_latent_size: int = 0,
    use_depth_tokens: bool = False,
    camera_token_layout: str = "global",
    depth_token_layout: str = "global",
    num_frames: int = 1,
    direct_patch_depth: bool = False,
    direct_patch_token_count: Optional[int] = None,
    spatial_fusion_type: str = SPATIAL_FUSION_TYPE_LATENT_TOKENS,
):
    """(camera_prefix_item, prompt_text)"""
    answer_type = _answer_type_from_gt(record["gt_answer"])
    question = str(record["question"]).strip()
    if benchmark.name == "vsi":
        question_type = str(record.get("question_type", "")).strip()
        if question_type in VSI_MCA_QUESTION_TYPES:
            if answer_type != "mcq":
                raise RuntimeError(
                    f"VSI MCA sample gt_answer is not an option letter: id={record.get('id')}, gt={record.get('gt_answer')!r}"
                )
            instruction = VSI_MCQ_ANSWER_INSTRUCTION
        elif question_type in VSI_NA_QUESTION_TYPES:
            if answer_type != "numeric":
                raise RuntimeError(
                    f"VSI NA sample gt_answer is not numeric: id={record.get('id')}, gt={record.get('gt_answer')!r}"
                )
            instruction = VSI_NUMERIC_ANSWER_INSTRUCTION
        else:
            raise RuntimeError(
                f"Unknown VSI question_type: id={record.get('id')}, question_type={question_type!r}"
            )
        question_with_instruction = f"{VSI_VIDEO_CONTEXT}\n{question}\n{instruction}"
    elif benchmark.name == "cvbench" and answer_type == "mcq":
        if cv_prompt_style == "official":
            instruction = CVBENCH_MCQ_ANSWER_INSTRUCTION
        elif cv_prompt_style == "legacy":
            instruction = MCQ_ANSWER_INSTRUCTION
        else:
            raise RuntimeError(f"Unknown --cv_prompt_style: {cv_prompt_style}")
        question_with_instruction = f"{question}\n{instruction}"
    else:
        instruction = (
            MCQ_ANSWER_INSTRUCTION
            if answer_type == "mcq"
            else NUMERIC_ANSWER_INSTRUCTION
        )
        question_with_instruction = f"{question}\n{instruction}"
    # Use the training slot-count function so per-frame padding equals frame count times K.
    camera_tokens = _build_camera_token_block(
        camera_latent_size,
        use_camera_tokens,
        token_layout=camera_token_layout,
        num_frames=num_frames,
    )
    if direct_patch_depth:
        if not use_depth_tokens or direct_patch_token_count is None:
            raise RuntimeError(
                "Direct Patch Depth prompt missing patch depth padding number."
            )
        # Direct Patch depth inputs align one-to-one with the RGB patch grid.
        depth_tokens = _build_depth_token_block(
            depth_latent_size=1,
            use_depth_tokens=True,
            patch_token_count=int(direct_patch_token_count),
        )
        if camera_token_position != "after_question":
            raise RuntimeError(
                f"Direct Patch Camera block must come after Question, current={camera_token_position!r}"
            )
        if (
            spatial_fusion_type
            == SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION
        ):
            # A: [RGB][P_depth][Question][Camera][Answer]。
            return None, f"{depth_tokens}\n{question_with_instruction}\n{camera_tokens}"
        if (
            spatial_fusion_type
            == SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_AFTER_QUESTION
        ):
            # B: [RGB][Question][Camera][P_depth][Answer]。
            return None, f"{question_with_instruction}\n{camera_tokens}\n{depth_tokens}"
        raise RuntimeError(
            f"Direct Patch Depth spatial_fusion_type illegal: {spatial_fusion_type!r}"
        )
    camera_prefix = camera_tokens + _build_depth_token_block(
        depth_latent_size,
        use_depth_tokens,
        token_layout=depth_token_layout,
        num_frames=num_frames,
    )
    return _assemble_camera_prompt(
        camera_prefix, question_with_instruction, camera_token_position
    )


def _build_user_content(
    record: dict,
    *,
    benchmark: BenchmarkConfig,
    images: List[Image.Image],
    image_paths: List[str],
    raw_video_path: Optional[str],
    camera_latent_size: int,
    use_camera_tokens: bool,
    input_mode: str,
    vsi_video_source: str,
    vsi_num_frames: int,
    cv_prompt_style: str,
    camera_token_position: str = "before_question",
    depth_latent_size: int = 0,
    use_depth_tokens: bool = False,
    camera_token_layout: str = "global",
    depth_token_layout: str = "global",
    num_frames: int = 1,
    direct_patch_depth: bool = False,
    direct_patch_token_count: Optional[int] = None,
    spatial_fusion_type: str = SPATIAL_FUSION_TYPE_LATENT_TOKENS,
) -> List[dict]:
    """Construct a general-purpose benchmark with Qwen user content, maintaining strict visual order as specified in JSONL."""
    camera_prefix, prompt_text = _build_prompt_text(
        record,
        benchmark,
        camera_latent_size,
        use_camera_tokens,
        cv_prompt_style,
        camera_token_position=camera_token_position,
        depth_latent_size=depth_latent_size,
        use_depth_tokens=use_depth_tokens,
        camera_token_layout=camera_token_layout,
        depth_token_layout=depth_token_layout,
        num_frames=num_frames,
        direct_patch_depth=direct_patch_depth,
        direct_patch_token_count=direct_patch_token_count,
        spatial_fusion_type=spatial_fusion_type,
    )
    user_content = []
    if camera_prefix:
        user_content.append({"type": "text", "text": camera_prefix})
    if input_mode == "video":
        if benchmark.name != "vsi":
            raise RuntimeError(
                f"Only VSI-Bench allows video input: benchmark={benchmark.name}"
            )
        if vsi_video_source == "frames":
            if len(images) != len(image_paths):
                raise RuntimeError(
                    f"VSI video frame count and path count do not match: images={len(images)}, paths={len(image_paths)}"
                )
            user_content.append({"type": "video", "video": image_paths})
        elif vsi_video_source == "raw_video":
            if raw_video_path is None:
                raise RuntimeError(
                    f"VSI raw_video mode is missing raw_video_path: id={record.get('id')}"
                )
            user_content.append(
                {
                    "type": "video",
                    "video": raw_video_path,
                    "nframes": int(vsi_num_frames),
                }
            )
        else:
            raise RuntimeError(f"Unknown --vsi_video_source: {vsi_video_source}")
    elif input_mode == "multi_image":
        user_content.extend({"type": "image", "image": image} for image in images)
    else:
        raise RuntimeError(f"Unknown input_mode: {input_mode}")
    user_content.append({"type": "text", "text": prompt_text})
    return user_content


def _spar_coordinate_rewrite_plan(
    record: dict,
    benchmark: BenchmarkConfig,
) -> Optional[tuple[int, str, str]]:
    """Use the shared coordinate rewriter to determine whether a SPAR-Bench prompt needs coordinate conversion."""
    if benchmark.name != "spar":
        return None
    route_record = dict(record)
    route_record["benchmark"] = "spar"
    if "task_family" not in route_record:
        # SPAR-Bench eval files use the task field; SPAR training JSONL files use task_family.
        # Only normalize field names here, without writing another task judgment rule in eval.
        route_record["task_family"] = str(record.get("task", "")).strip()
    image_idx, coord_source, rewrite_source = coordinate_rewrite_plan_for_record(
        route_record
    )
    if not str(rewrite_source).startswith("spar_"):
        return None
    return image_idx, coord_source, rewrite_source


def _rewrite_spar_prompt_record(
    record: dict,
    images: List[Image.Image],
    image_grid_thw: torch.Tensor,
    *,
    rewrite_plan: tuple[int, str, str],
) -> dict:
    """Rewrite SPAR-Bench prompt coordinates according to shared routing; keep GT option letters unchanged."""
    task = str(record.get("task_family", record.get("task", ""))).strip()
    expectation = SPAR_COORDINATE_TASK_EXPECTATIONS.get(task)
    if expectation is None:
        raise RuntimeError(
            f"Shared coordinate routing returned an unregistered SPAR-Bench task: id={record.get('id')}, task={task!r}"
        )

    target_image_idx, coord_source, rewrite_source = rewrite_plan
    if target_image_idx < 0 or target_image_idx >= len(images):
        raise RuntimeError(
            f"SPAR {task} Coordinate rewrite target image out of bounds: id={record.get('id')}, image_idx={target_image_idx}, images={len(images)}, rewrite_source={rewrite_source}"
        )

    question = str(record.get("question", ""))
    coord_re = expectation["coord_re"]
    if not coord_re.search(question):
        raise RuntimeError(
            f"SPAR {task} Not found in the question {expectation['missing_desc']}: id={record.get('id')}"
        )
    required_phrase = str(expectation["required_phrase"])
    if required_phrase not in question.lower():
        raise RuntimeError(
            f"SPAR {task} Currently only support {required_phrase} id={record.get('id')}, question={question!r}"
        )

    # Share coordinate-system and target-image routing with training.
    target_image = images[target_image_idx]
    processed_w, processed_h = get_processed_size(
        image_grid_thw, image_idx=target_image_idx
    )
    rewritten_question = rewrite_coordinates_in_text(
        question,
        original_w=target_image.width,
        original_h=target_image.height,
        processed_w=processed_w,
        processed_h=processed_h,
        coord_source=coord_source,
    )
    if (
        coord_source == "normalized_1000"
        and (processed_w != 1000 or processed_h != 1000)
        and rewritten_question == question
    ):
        raise RuntimeError(
            f"SPAR {task} Coordinate rewrite not taking effect: id={record.get('id')}, processed=({processed_w},{processed_h}), rewrite_source={rewrite_source}"
        )
    rewritten = dict(record)
    rewritten["question"] = rewritten_question
    # Audit Evaluation: Image paths still point to the original images, but the prompt coordinates are now processed pixel coordinates after the processor.
    # When analyzing eval_results.jsonl subsequently, you must refer to this metadata; do not directly draw the model_prompt_question coordinates back onto the original image.
    rewritten["_coordinate_rewrite_meta"] = {
        "note": COORDINATE_REWRITE_AUDIT_NOTE,
        "coordinate_space": "qwen_processed_image_pixels",
        "original_question_coordinate_space": expectation["original_coordinate_space"],
        "target_image_index": target_image_idx,
        "target_image_role": expectation["target_image_role"],
        "coord_source": coord_source,
        "rewrite_source": rewrite_source,
        "task_family": task,
        "original_image_size": [int(target_image.width), int(target_image.height)],
        "processed_image_size": [int(processed_w), int(processed_h)],
        "image_grid_thw": [int(x) for x in image_grid_thw[target_image_idx].tolist()],
        "saved_image_path_is_original": True,
        "model_input_image_is_runtime_resized": True,
    }
    return rewritten


def _nested_get(payload: Dict[str, object], keys: Sequence[str]) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(
                f"benchmark scorer output missing field: path={'.'.join(keys)}, current={current}"
            )
        current = current[key]
    return current


def _run_benchmark_scorer(
    output_dir: str, benchmark: BenchmarkConfig, *, is_final: bool
) -> Dict[str, object]:
    """Run the benchmark scorer under preprocessing, and use only this score for the final table."""
    if benchmark.name not in BENCHMARK_SCORER_CONFIG:
        raise RuntimeError(
            f"Missing benchmark scorer configuration: benchmark={benchmark.name}"
        )

    scorer_rel, metric_path, metric_name = BENCHMARK_SCORER_CONFIG[benchmark.name]
    scorer_path = PROJECT_ROOT / "scoring" / scorer_rel
    if not scorer_path.is_file():
        raise FileNotFoundError(f"benchmark scorer not found: {scorer_path}")

    results_path = Path(output_dir) / "eval_results.jsonl"
    # Use the same official scorer for live progress and final results.
    output_stem = "official_eval" if is_final else "live_eval"
    output_json = Path(output_dir) / f"{output_stem}.json"
    summary_txt = Path(output_dir) / f"{output_stem}.txt"
    cmd = [
        sys.executable,
        str(scorer_path),
        "--input-file",
        str(results_path),
        "--output-json",
        str(output_json),
        "--summary-file",
        str(summary_txt),
    ]
    # Write the official benchmark metric into eval_summary.json after generation.
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=None if is_final else subprocess.PIPE,
        stderr=None if is_final else subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark scorer failed: benchmark={benchmark.name}, is_final={is_final}, stderr={completed.stderr}"
        )

    with output_json.open("r", encoding="utf-8") as f:
        official_payload = json.load(f)
    metric_value = _nested_get(official_payload, metric_path)
    if not isinstance(metric_value, (int, float)):
        raise RuntimeError(
            f"benchmark scorer main metric is not numeric: benchmark={benchmark.name}, value={metric_value!r}"
        )

    summary_path = Path(output_dir) / "eval_summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        # Expose the official primary metric as the single top-level accuracy.
        summary["accuracy"] = round(float(metric_value), 6)
        summary["metric_name"] = metric_name
        summary["is_final"] = bool(is_final)
        summary["scorer_json"] = str(output_json.resolve())
        summary["scorer_summary"] = str(summary_txt.resolve())
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "accuracy": round(float(metric_value), 6),
        "metric_name": metric_name,
        "is_final": bool(is_final),
        "scorer_json": str(output_json.resolve()),
        "scorer_summary": str(summary_txt.resolve()),
    }
