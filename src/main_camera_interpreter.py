import argparse
import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Use a separate Triton cache per rank to avoid concurrent compiler writes.
if os.environ.get("TRITON_CACHE_DIR") and "LOCAL_RANK" in os.environ:
    os.environ["TRITON_CACHE_DIR"] = str(
        Path(os.environ["TRITON_CACHE_DIR"]) / f"rank_{os.environ['LOCAL_RANK']}"
    )
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from georeader_data import resolve_npz_paths, resolve_records
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers import TrainingArguments

from camera_interpreter import (
    CAMERA_FRAME_INDEX_CHOICES,
    CAMERA_FRAME_INDEX_RELATIVE,
    CAMERA_TRANSLATION_ENCODING_CHOICES,
    CAMERA_TRANSLATION_ENCODING_NORMALIZED,
    CAMERA_TRANSLATION_ENCODING_NORMALIZED_LOG_SCALE,
    CAMERA_TRANSLATION_ENCODING_RAW_METERS,
    OUTPUT_CONTRACT_CHOICES,
    OUTPUT_CONTRACT_DIRECT_EMBEDDING,
    OUTPUT_GATE_TYPE_CHOICES,
    CameraInterpreter,
    CameraInterpreterConfig,
    build_pose_only_image_grid_thw,
    geometry_dim_for_camera_translation_encoding,
    get_llm_image_token_counts,
    parse_camera_interpreter_config_dict,
)
from depth_interpreter import (
    DEPTH_ENCODER_TYPE_CHOICES,
    DEPTH_ENCODER_TYPE_STATS_10D,
    DepthInterpreter,
    DepthInterpreterConfig,
    build_depth_patch_stats,
    depth_geometry_dim_for_encoder_type,
    parse_depth_interpreter_config_dict,
)
from depth_patch_tokenizer import (
    DEPTH_PATCH_OUTPUT_CONTRACT,
    DepthPatchTokenizer,
    DepthPatchTokenizerConfig,
    parse_depth_patch_tokenizer_config_dict,
)
from pixel_budget_defaults import DEFAULT_IMAGE_MAX_PIXELS, DEFAULT_IMAGE_MIN_PIXELS
from trainer_camera_interpreter import CustomTrainerCameraInterpreter
from visual_encoder_utils import encode_visual_in_chunks


CAMERA_START = "<|camera_start|>"
CAMERA_PAD = "<|camera_pad|>"
CAMERA_END = "<|camera_end|>"
DEPTH_START = "<|depth_start|>"
DEPTH_PAD = "<|depth_pad|>"
DEPTH_END = "<|depth_end|>"
TOKEN_LAYOUT_CHOICES = ("global", "per_frame")
DEPTH_STAGE1_OUTPUT_CONTRACT_SUMMARY = "summary_interpreter"
DEPTH_STAGE1_OUTPUT_CONTRACT_DIRECT_PATCH = DEPTH_PATCH_OUTPUT_CONTRACT
DEPTH_STAGE1_OUTPUT_CONTRACT_CHOICES = (
    DEPTH_STAGE1_OUTPUT_CONTRACT_SUMMARY,
    DEPTH_STAGE1_OUTPUT_CONTRACT_DIRECT_PATCH,
)
DEFAULT_USER_PROMPT = "Describe only the camera motion across these views using Image 1 as the reference frame."
DEFAULT_DEPTH_USER_PROMPT = (
    "Describe the depth layout and object positions in each view."
)
CAMERA_PLANAR_METRIC_COT_VERSION = (
    "camera_observer_planar_metric_motion_image1_content_supervision"
)
CAMERA_LOSS_MODE_FULL_CAPTION = "full_caption"
CAMERA_LOSS_MODE_MOTION_CONTENT = "motion_content"
CAMERA_LOSS_MODE_CHOICES = (
    CAMERA_LOSS_MODE_FULL_CAPTION,
    CAMERA_LOSS_MODE_MOTION_CONTENT,
)
DEPTH_STAGE1_COT_VERSION = "depth_layout_instance_surface_projection_compact_ninegrid"
DEPTH_NUMERIC_READOUT_COT_VERSION = "depth_layout_numeric_readout_top5_position_first"
DEPTH_LOSS_MODE_FULL_CAPTION = "full_caption"
DEPTH_LOSS_MODE_NUMBER_ONLY = "number_only"
DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA = "caption_pair_delta"
DEPTH_LOSS_MODE_CHOICES = (
    DEPTH_LOSS_MODE_FULL_CAPTION,
    DEPTH_LOSS_MODE_NUMBER_ONLY,
    DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA,
)
PROMPT_SOURCE_CHOICES = ("fixed", "record", "none")
DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW = "window_all_frames"
DEPTH_FRAME_SUPERVISION_SCOPE_OWN = "own_frame_only"
DEPTH_FRAME_SUPERVISION_SCOPE_CHOICES = (
    DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
    DEPTH_FRAME_SUPERVISION_SCOPE_OWN,
)
DEPTH_FRAME_MICROBATCH_MODE_DISABLED = "disabled"
DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS = "window_to_rows"
DEPTH_FRAME_MICROBATCH_MODE_CHOICES = (
    DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
    DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS,
)
DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN = "token_mean"
DEPTH_FRAME_LOSS_REDUCTION_WINDOW_MEAN = "mean_per_source_window"
DEPTH_FRAME_LOSS_REDUCTION_CHOICES = (
    DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
    DEPTH_FRAME_LOSS_REDUCTION_WINDOW_MEAN,
)
DEPTH_PERFRAME_CAPTION_MICROBATCH_COT_VERSION = (
    "depth_layout_perframe_caption_microbatch"
)
DEPTH_PERFRAME_CAPTION_MICROBATCH_TASK = "depth_perframe_caption_microbatch"
DEPTH_OBJECT_QUERY_COT_VERSION = "depth_object_query_numeric_single"
DEPTH_CAPTION_PAIR_DELTA_COT_VERSION = "depth_layout_caption_pair_delta_numeric"
DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION = (
    "depth_layout_caption_pair_delta_crossframe_numeric"
)
DEPTH_NUMBER_PATTERN = re.compile(r"at ([0-9]+\.[0-9]) meters")
DEPTH_RELATION_ANSWER_PATTERN = re.compile(
    r"The absolute depth difference is ([0-9]+\.[0-9]) meters\."
)
DEPTH_DISTANCE_FIELD_PATTERN = re.compile(r"at ([^\s]+) meters")
DEPTH_METER_NUMBER_PATTERN = re.compile(r"[0-9]\.[0-9]")
# Fixed at 0.1m with alignment to one decimal place; this freezes the Design constant and does not expose a second tunable hyperparameter.
DEPTH_METER_LOSS_BETA_M = 0.1
CAMERA_MOTION_LINE_PATTERN = re.compile(
    r"^compared with image 1, the observer in image (?P<frame>[0-9]+) "
    r"(?:(?P<stationary>stays nearly at the same position)|"
    r"moves (?P<distance1>[0-9]+\.[0-9]) meters (?P<translation1>left|right|forward|backward)"
    r"(?: and (?P<distance2>[0-9]+\.[0-9]) meters "
    r"(?P<translation2>left|right|forward|backward))?), and "
    r"(?:(?P<unchanged>keeps nearly the same heading)|"
    r"turns (?P<rotation>left|right) by about (?P<angle>[0-9]+) degrees)\.$"
)
# Continue using the original MindCube DA3 description syntax, do not interpret it as an approximate motion template for SixDoF.
MINDCUBE_DA3_REFERENCE = (
    "image 1 is the reference view. distances are measured in predicted meters."
)
MINDCUBE_DA3_LINE_PATTERN = re.compile(
    r"^compared with image 1, the observer in image (?P<frame>[0-9]+) "
    r"(?:(?P<stationary>is at the same position)|"
    r"is (?P<distance1>[0-9]+(?:\.[0-9]+)?) meters? (?P<translation1>left|right|forward|backward)"
    r"(?: and (?P<distance2>[0-9]+(?:\.[0-9]+)?) meters? (?P<translation2>left|right|forward|backward))? of the reference position), and "
    r"(?:(?P<unchanged>faces the same direction)|"
    r"(?P<opposite>faces the opposite direction) \((?P<opposite_angle>180) degrees\)|"
    r"faces (?P<angle>[0-9]+) degrees to the (?P<rotation>left|right) of the reference heading)\.$"
)
MIXED_IMAGE_SIZE = "mixed"
DEPTH_RGB_ASPECT_RATIO_REL_TOL = 0.03  # The aspect ratio of depth/RGB allows a 3% error; prevents new data sources from mixing with depth data having different FOV/crop.
DEPTH_MIN_REASONABLE_MEDIAN_M = 0.01  # Median depth below 1cm is generally not normal indoor metric depth; it usually indicates a range or unit error.
DEPTH_MAX_REASONABLE_MEDIAN_M = 500.0  # Outdoor airport/mountain scenes can exceed 100m; overall exceeding 500m triggers suspected millimeter-level inspection.
DEPTH_MAX_REASONABLE_LOWER_QUARTILE_M = 500.0  # P25 exceeds 500m only to indicate an overall scale anomaly in the entire image, which more resembles mm without unit conversion.
CAMERA_ROT_DET_TOL = 0.08  # native pose must be a right-handed rotation; must maintain the same tolerance as the pose-rescue frame-level gate.
CAMERA_FRAME0_ATOL = 1e-4  # Canonicalized image 1 should be [I|0]; larger errors indicate a broken coordinate agreement.


def _coerce_modality_flag(raw_value, *, field_name: str, context: str) -> int:
    """Strictly validate has_pose/has_depth as 0/1, avoiding implicit truthy/falsy string conversions."""
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int) and raw_value in (0, 1):
        return int(raw_value)
    if isinstance(raw_value, str) and raw_value.strip() in {"0", "1"}:
        return int(raw_value.strip())
    raise RuntimeError(f"{field_name} Can only be 0/1: {context}, value={raw_value!r}")


def _resolve_record_modality_path(
    record: Dict,
    *,
    path_field: str,
    availability_field: str,
    context: str,
) -> Optional[str]:
    """Parse modalities according to the unified JSONL contract; never read residual paths when explicitly unavailable."""
    if not isinstance(record, dict):
        raise RuntimeError(
            f"Sample must be a dict, cannot parse {path_field}: {context}, type={type(record)}"
        )
    explicit = None
    if availability_field in record:
        explicit = _coerce_modality_flag(
            record[availability_field],
            field_name=availability_field,
            context=context,
        )
        if explicit == 0:
            # A disabled modality must not access its geometry file.
            return None
    if path_field not in record:
        raw_path = None
        resolved_path = None
    else:
        raw_path = record.get(path_field)
        if raw_path is None or not str(raw_path).strip():
            raise RuntimeError(
                f"{path_field} Written to JSONL but value is empty: {context}, value={raw_path!r}"
            )
        raw_text = str(raw_path).strip()
        if not os.path.isabs(raw_text):
            raise RuntimeError(
                f"{path_field} Must be a complete absolute path: {context}, value={raw_path!r}"
            )
        resolved_path = os.path.abspath(os.path.expanduser(raw_text))
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"{path_field} The specified npz file does not exist: {resolved_path} ({context})"
            )

    derived = int(resolved_path is not None)
    if explicit is not None:
        if explicit != derived:
            raise RuntimeError(
                f"{availability_field} and {path_field} There is a contradiction: {context}, {availability_field}={explicit}, {path_field}={raw_path!r}"
            )
    return resolved_path


def _resolve_record_pose_path(record: Dict, *, context: str) -> Optional[str]:
    """Parse pose_path; this is the only positioning method for Camera Stage1/Stage2/eval."""
    return _resolve_record_modality_path(
        record,
        path_field="pose_path",
        availability_field="has_pose",
        context=context,
    )


def _resolve_record_depth_path(record: Dict, *, context: str) -> Optional[str]:
    """Parse depth_path; explicitly indicate not to inject depth token when the field is missing."""
    return _resolve_record_modality_path(
        record,
        path_field="depth_path",
        availability_field="has_depth",
        context=context,
    )


def _normalize_image_path_sequence(paths: Sequence[str]) -> tuple[str, ...]:
    """Convert ordered image path strings into absolute path tuples for sequential item-by-item comparison in npz/JSONL files."""
    return tuple(os.path.abspath(os.path.expanduser(str(path))) for path in paths)


def _image_file_identity_sequence(paths: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Return RGB file system identities in order; hard link aliases should be considered the same image."""
    identities = []
    for path in _normalize_image_path_sequence(paths):
        stat_result = os.stat(path)
        identities.append((int(stat_result.st_dev), int(stat_result.st_ino)))
    return tuple(identities)


def _validate_npz_image_order(
    npz_payload: Dict[str, np.ndarray],
    *,
    expected_image_paths: Optional[Sequence[str]],
    expected_frame_ids: Optional[Sequence[int]],
    path: str,
    expected_original_image_paths: Optional[Sequence[str]] = None,
    expected_geometry_image_paths: Optional[Sequence[str]] = None,
    unmarked_rgb_migration: bool = False,
) -> None:
    """Check RGB order against NPZ ordering metadata; Camera Window NPZ files must include both ordering fields."""
    # Publish NPZ directly saves relative image order within the package; only parse in memory from the download root directory, without needing to restore copies or mapping tables.
    npz_payload = resolve_npz_paths(npz_payload, path)
    if expected_image_paths is not None and "image_input" in npz_payload:
        stored = np.asarray(npz_payload["image_input"])
        if stored.ndim != 1:
            raise RuntimeError(
                f"camera/depth npz's image_input must be one-dimensional: shape={stored.shape}, path={path}"
            )
        stored_paths = _normalize_image_path_sequence(stored.tolist())
        expected_paths = _normalize_image_path_sequence(expected_image_paths)
        if expected_original_image_paths is not None:
            # Check the declared ordered image sequences without changing geometry values.
            original = _normalize_image_path_sequence(expected_original_image_paths)
            if stored_paths != original or len(original) != len(expected_paths):
                raise RuntimeError(
                    f"NPZ image order does not match the declared audit sequence: path={path}"
                )
            if expected_geometry_image_paths is None:
                raise RuntimeError(f"Migration RGB missing frame order: path={path}")
            geometry = _normalize_image_path_sequence(expected_geometry_image_paths)
            if "geometry_image_input" in npz_payload:
                stored_geometry = _normalize_image_path_sequence(
                    np.asarray(npz_payload["geometry_image_input"]).tolist()
                )
                if stored_geometry != geometry:
                    raise RuntimeError(
                        f"NPZ unbounded image sequence inconsistent with JSONL: path={path}"
                    )
            elif not unmarked_rgb_migration or geometry != expected_paths:
                # Require a separate geometry image sequence unless the contract explicitly permits shared RGB.
                raise RuntimeError(
                    f"Migration RGB missing independent frame order: path={path}"
                )
            _image_file_identity_sequence(expected_paths)
            _image_file_identity_sequence(geometry)
        else:
            # Compare file identities so hard-linked image aliases retain the correct frame order.
            if _image_file_identity_sequence(
                stored_paths
            ) != _image_file_identity_sequence(expected_paths):
                raise RuntimeError(
                    f"RGB and npz frame order inconsistency: path={path}, stored={stored_paths}, expected={expected_paths}"
                )
    if expected_frame_ids is not None and "frame_ids" in npz_payload:
        stored_ids = np.asarray(npz_payload["frame_ids"])
        if stored_ids.ndim != 1:
            raise RuntimeError(
                f"camera/depth NPZ frame_ids must be one-dimensional: shape={stored_ids.shape}, path={path}"
            )
        expected_ids = np.asarray(
            [int(item) for item in expected_frame_ids], dtype=np.int64
        )
        if not np.array_equal(stored_ids.astype(np.int64), expected_ids):
            raise RuntimeError(
                f"JSONL window_frame_ids and npz frame_ids are out of order: path={path}, stored={stored_ids.tolist()}, expected={expected_ids.tolist()}"
            )


def _coerce_has_pose_value(raw_value, *, context: str) -> int:
    """Strictly validate the has_pose field to be 0 or 1; report an error for any invalid values."""
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int) and raw_value in (0, 1):
        return int(raw_value)
    if isinstance(raw_value, str) and raw_value.strip() in {"0", "1"}:
        return int(raw_value.strip())
    raise RuntimeError(f"has_pose can only be 0/1: {context}, value={raw_value!r}")


def _resolve_has_pose(record: Dict, n_images: int, *, context: str) -> int:
    """Parse has_pose by sample field and number of images; single image is default unavailable, multi-image is default available."""
    if not isinstance(record, dict):
        raise RuntimeError(
            f"Sample must be a dict; cannot parse has_pose: {context}, type={type(record)}"
        )
    n_images = int(n_images)
    if n_images <= 0:
        raise RuntimeError(
            f"Number of images must be greater than 0, cannot parse has_pose: {context}, n_images={n_images}"
        )
    has_field = "has_pose" in record
    # Key: If a single image does not have an explicit field, it is treated as no_camera by default to avoid leaving empty camera slots in the prompt.
    raw_value = record["has_pose"] if has_field else (1 if n_images > 1 else 0)
    has_pose = _coerce_has_pose_value(raw_value, context=context)
    if n_images <= 1 and has_pose != 0:
        raise RuntimeError(
            f"Single-image samples cannot declare has_pose=1, otherwise it will reintroduce empty slot tax: {context}, n_images={n_images}"
        )
    return has_pose


def _should_insert_camera_tokens(has_pose: int, n_images: int, *, context: str) -> bool:
    """Determine whether the current sample should insert a camera token segment in the prompt."""
    has_pose = _coerce_has_pose_value(has_pose, context=context)
    n_images = int(n_images)
    if n_images <= 0:
        raise RuntimeError(
            f"Number of images must be greater than 0, cannot determine camera token insertion: {context}, n_images={n_images}"
        )
    if n_images <= 1 and has_pose != 0:
        raise RuntimeError(
            f"Single-image samples must not contain camera tokens: {context}, n_images={n_images}, has_pose={has_pose}"
        )
    return bool(has_pose)


def _geometry_prompt_token_count(
    latent_size: int,
    token_layout: str,
    num_frames: int,
) -> int:
    """Calculate the number of Geometry pads to insert for the current sample based on the layout."""
    latent_size = int(latent_size)
    num_frames = int(num_frames)
    if latent_size <= 0 or num_frames <= 0:
        raise RuntimeError(
            f"latent_size/num_frames must be > 0, current={latent_size}/{num_frames}"
        )
    if token_layout == "global":
        return latent_size
    if token_layout == "per_frame":
        return num_frames * latent_size
    raise RuntimeError(
        f"token_layout must be {TOKEN_LAYOUT_CHOICES}current ={token_layout!r}"
    )


def _build_camera_token_block(
    camera_latent_size: int,
    use_camera_tokens: bool,
    *,
    token_layout: str = "global",
    num_frames: int = 1,
) -> str:
    """Construct camera special token segment as needed; return empty string when no camera is used."""
    if not bool(use_camera_tokens):
        return ""
    pad_count = _geometry_prompt_token_count(
        camera_latent_size,
        token_layout,
        num_frames,
    )
    return CAMERA_START + (CAMERA_PAD * pad_count) + CAMERA_END


def _build_depth_token_block(
    depth_latent_size: int,
    use_depth_tokens: bool,
    *,
    token_layout: str = "global",
    num_frames: int = 1,
    patch_token_count: Optional[int] = None,
) -> str:
    """Construct Depth token block according to layout, ensuring pad count matches Interpreter output."""
    if not bool(use_depth_tokens):
        return ""
    if patch_token_count is None:
        pad_count = _geometry_prompt_token_count(
            depth_latent_size,
            token_layout,
            num_frames,
        )
    else:
        pad_count = int(patch_token_count)
        if pad_count <= 0:
            raise RuntimeError(
                f"Depth patch token count must be a positive number: {pad_count}"
            )
    return DEPTH_START + (DEPTH_PAD * pad_count) + DEPTH_END


CAMERA_TOKEN_POSITION_CHOICES = ("before_image", "before_question", "after_question")


def _assemble_camera_prompt(
    camera_tokens: str, question: str, camera_token_position: str
):
    """return (camera_prefix_item, prompt_text)

    camera_prefix_item: "before_image" is a string for camera_tokens
                         To be inserted as an independent text content item before images;
                         None when other position.
    prompt_text: Text containing the question.
    """
    if not camera_tokens:
        return None, question
    if camera_token_position == "before_image":
        return camera_tokens, question
    if camera_token_position == "before_question":
        return None, f"{camera_tokens}\n{question}"
    if camera_token_position == "after_question":
        return None, f"{question}\n{camera_tokens}"
    raise RuntimeError(f"Unknown camera_token_position: {camera_token_position!r}")


def _assert_special_token_layout(
    input_ids: torch.Tensor,
    *,
    start_id: int,
    pad_id: int,
    end_id: int,
    expected_pad_count: int,
    context: str,
    token_name: str,
):
    """Check the number and consecutive order of special-token blocks."""
    if input_ids is None or input_ids.dim() != 2:
        raise RuntimeError(
            f"input_ids must be [batch, seq], cannot be checked {token_name} token: {context}"
        )
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError(
            f"Only supports batch_size=1 {token_name} Token check: {context}, shape={tuple(input_ids.shape)}"
        )
    expected_pad_count = int(expected_pad_count)
    if expected_pad_count < 0:
        raise RuntimeError(
            f"expected_pad_count cannot be negative: {context}, expected={expected_pad_count}"
        )
    start_count = int((input_ids == int(start_id)).sum().item())
    pad_count = int((input_ids == int(pad_id)).sum().item())
    end_count = int((input_ids == int(end_id)).sum().item())
    if expected_pad_count == 0:
        if start_count != 0 or pad_count != 0 or end_count != 0:
            raise RuntimeError(
                f"The current sample should not appear at all {token_name} token: {context}, start={start_count}, pad={pad_count}, end={end_count}"
            )
        return
    if start_count != 1 or pad_count != expected_pad_count or end_count != 1:
        raise RuntimeError(
            f"{token_name} Invalid token count: {context}, start={start_count}, pad={pad_count}, end={end_count}, expected_pad={expected_pad_count}"
        )
    start_index = int(
        (input_ids[0] == int(start_id)).nonzero(as_tuple=False)[0, 0].item()
    )
    block = input_ids[0, start_index : start_index + expected_pad_count + 2]
    expected_block = torch.tensor(
        [int(start_id)] + [int(pad_id)] * expected_pad_count + [int(end_id)],
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    if block.shape != expected_block.shape or not torch.equal(block, expected_block):
        raise RuntimeError(
            f"{token_name} token must be a unique contiguous start/pad/end block: {context}, start_index={start_index}, expected_pad={expected_pad_count}"
        )


def _assert_camera_token_layout(
    input_ids: torch.Tensor,
    *,
    camera_start_id: int,
    camera_pad_id: int,
    camera_end_id: int,
    expected_pad_count: int,
    context: str,
):
    """Check the number of Camera blocks and their sequential order."""
    _assert_special_token_layout(
        input_ids,
        start_id=camera_start_id,
        pad_id=camera_pad_id,
        end_id=camera_end_id,
        expected_pad_count=expected_pad_count,
        context=context,
        token_name="Camera",
    )


def _assert_depth_token_layout(
    input_ids: torch.Tensor,
    *,
    depth_start_id: int,
    depth_pad_id: int,
    depth_end_id: int,
    expected_pad_count: int,
    context: str,
):
    """Check the number of Depth blocks and their sequential order."""
    _assert_special_token_layout(
        input_ids,
        start_id=depth_start_id,
        pad_id=depth_pad_id,
        end_id=depth_end_id,
        expected_pad_count=expected_pad_count,
        context=context,
        token_name="Depth",
    )


def project_root_from_this_file() -> str:
    return str(Path(__file__).resolve().parents[1])


def assert_cpu_affinity_for_training():
    if os.environ.get("CAMERA_INTERPRETER_SKIP_CPU_AFFINITY_CHECK", "0") == "1":
        return
    affinity = sorted(os.sched_getaffinity(0))
    preview = (
        affinity if len(affinity) <= 16 else affinity[:8] + ["..."] + affinity[-8:]
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[env] cpu_affinity_count={len(affinity)} cpu_affinity={preview}")
    if len(affinity) <= 1:
        raise RuntimeError(
            "[FATAL] CPU affinity is too narrow for Stage1 Camera Interpreter training. Please start using srun/sbatch with the --cpus-per-task option inside the login node tmux."
        )


def get_args():
    root = project_root_from_this_file()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=os.path.join(root, "models/qwen25_vl_3b")
    )
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="+",
        required=True,
        help="One or more CoT data JSON/JSONL paths; multiple paths are concatenated in order, and max_samples applies to the total samples in the concatenated result.",
    )
    # Derived training data can include a generated audit report; checkpoints will strictly verify and preserve the original data, script, and random seed identity.
    parser.add_argument(
        "--data_provenance_report",
        type=str,
        default=None,
        help="Data report for the Stage 1 training JSONL; its SHA256 is checked and recorded in the checkpoint metadata.",
    )
    parser.add_argument("--image_folder", type=str, default=root)
    parser.add_argument(
        "--feature_vggt_dir",
        type=str,
        default=None,
        help="Deprecated: Camera Stage1 only reads the complete pose_path for each sample, no longer concatenates feature_dir/feature_idx.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=os.path.join(root, "checkpoints/stage1/camera"),
    )
    parser.add_argument("--cache_dir", type=str, default=os.path.join(root, "cache"))
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="stage1-camera-interpreter")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # Load Interpreter weights only for the completed Stage1; optimizer/scheduler must be freshly created for the current experiment.
    parser.add_argument(
        "--init_interpreter_checkpoint",
        type=str,
        default=None,
        help="Load only the Interpreter weight initialization checkpoint; cannot be used with resume_from_checkpoint.",
    )

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # When max_steps > 0, it serves as an upper limit for precise optimizer steps; the formal metric experiment is fixed at 73,630.
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument(
        "--save_only_model",
        type=str,
        default=os.environ.get("SAVE_ONLY_MODEL", "1"),
        choices=["0", "1"],
    )
    parser.add_argument(
        "--disable_tqdm",
        type=str,
        default=os.environ.get("DISABLE_TQDM", "1"),
        choices=["0", "1"],
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    # Group samples by frame count to reduce waiting between training ranks.
    parser.add_argument(
        "--train_sampler_mode",
        type=str,
        default="length_grouped",
        choices=["random", "length_grouped"],
    )
    # visual_chunk_size: Number of images encoded at once by the frozen visual encoder; default is 8 for speedup, reduced to 4/2/1 when out of memory (OOM).
    parser.add_argument(
        "--visual_chunk_size",
        type=int,
        default=int(os.environ.get("VISUAL_CHUNK_SIZE", "8")),
        help="In Stage1 imgpose mode, the image patch size for the visual encoder; pose-only mode does not run the visual encoder.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_IMAGE_MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument(
        "--expected_image_size",
        type=str,
        default="480x640",
        help="Expected input image size as WxH; use 'mixed' for images of different sizes.",
    )

    parser.add_argument("--camera_latent_size", type=int, default=8)
    # global=entire window shares K; per_frame=K output per frame; Depth allows Kd=1/2, Camera remains fixed Kc=1.
    parser.add_argument(
        "--camera_token_layout",
        type=str,
        default="global",
        choices=TOKEN_LAYOUT_CHOICES,
    )
    parser.add_argument("--camera_num_layers", type=int, default=2)
    parser.add_argument("--camera_num_heads", type=int, default=8)
    parser.add_argument("--camera_geom_rank", type=int, default=32)
    parser.add_argument("--camera_dropout", type=float, default=0.0)
    # output_contract controls the semantic of Interpreter projection; direct_embedding indicates complete replacement of pad.
    parser.add_argument(
        "--output_contract",
        type=str,
        default="gated_delta",
        choices=OUTPUT_CONTRACT_CHOICES,
    )
    # direct_embedding can only use none; the configuration class will block invalid combinations directly.
    parser.add_argument(
        "--output_gate_type",
        type=str,
        default="scalar",
        choices=OUTPUT_GATE_TYPE_CHOICES,
    )
    parser.add_argument("--camera_output_gate_init", type=float, default=0.1)
    parser.add_argument("--camera_geom_gate_init", type=float, default=0.1)
    # Choose a translation encoding consistent with the available metric scale.
    parser.add_argument(
        "--camera_translation_encoding",
        type=str,
        default=os.environ.get(
            "CAMERA_TRANSLATION_ENCODING", CAMERA_TRANSLATION_ENCODING_NORMALIZED
        ),
        choices=CAMERA_TRANSLATION_ENCODING_CHOICES,
        help=(
            "Camera translation encoding methods: normalized = normalize the entire trajectory by the maximum relative displacement, retaining only direction and shape (default for Pred Camera); raw_meters = retain original displacement; normalized_log_scale = normalize and additionally append log(center_scale), used only for scale reliability ablation."
        ),
    )
    # Fixed frame indexing encoding makes the image i from paired training share the same index features with image i from multi-frame Stage2.
    parser.add_argument(
        "--camera_frame_index_encoding",
        type=str,
        default=CAMERA_FRAME_INDEX_RELATIVE,
        choices=CAMERA_FRAME_INDEX_CHOICES,
    )
    parser.add_argument(
        "--camera_pair_rows",
        type=str,
        default="0",
        choices=["0", "1"],
        help="1 = expand each window into one training row per target image, paired with image 1.",
    )
    # motion_content retains the full caption for teacher forcing, but only supervises the content slots dependent on pose.
    parser.add_argument(
        "--camera_loss_mode",
        type=str,
        default=CAMERA_LOSS_MODE_FULL_CAPTION,
        choices=CAMERA_LOSS_MODE_CHOICES,
        help="Camera Stage1 loss: full_caption = complete answer CE; motion_content = only supervises the motion content slot.",
    )
    # cot_version only describes the target text version and cannot be inferred from loss_mode; new experiments must explicitly pass it via wrapper.
    parser.add_argument(
        "--camera_cot_version",
        type=str,
        default=None,
        help="The cot_version of Camera assistant target; and camera_loss_mode are saved and verified independently.",
    )
    # 0 turns off additional raw-geometry attention score; the geometry MLP -> memory K/V main path remains enabled.
    parser.add_argument(
        "--camera_use_geometry_attention_bias",
        type=str,
        default="1",
        choices=["0", "1"],
        help="1 = enable geometry attention bias; 0 = use standard QK cross-attention only.",
    )
    parser.add_argument(
        "--llm_use_image_tokens",
        type=str,
        default="1",
        choices=["0", "1"],
        help="1=QwenVL LLM views RGB image tokens; 0=LLM does not view images (pure geometric ablation).",
    )
    parser.add_argument(
        "--interpreter_use_image_tokens",
        type=str,
        default="0",
        choices=["0", "1"],
        help="1=Interpreter cross-attends RGB image tokens; 0=Interpreter only looks at geometric input.",
    )
    parser.add_argument(
        "--camera_token_position",
        type=str,
        default="before_question",
        choices=list(CAMERA_TOKEN_POSITION_CHOICES),
        help="Position of camera tokens in the prompt: before_image=before the image; before_question=before the question (default); after_question=after the question.",
    )
    parser.add_argument(
        "--use_camera_interpreter",
        type=str,
        default="1",
        choices=["0", "1"],
        help="1=Enable Camera Interpreter (default); 0=Do not create, for depth-only training.",
    )
    # B control group freezes the trained Camera token as context; all three conditions must be met simultaneously to prevent accidental activation of the standard joint Stage1.
    parser.add_argument(
        "--frozen_camera_checkpoint",
        type=str,
        default=None,
        help="B control group freezes Camera checkpoint; only loads camera_interpreter.pt.",
    )
    parser.add_argument(
        "--freeze_camera_interpreter",
        type=str,
        default="0",
        choices=["0", "1"],
        help="1=Strictly freeze Camera Interpreter, only allow B control group to use.",
    )
    parser.add_argument(
        "--depth_camera_caption_control",
        type=str,
        default="0",
        choices=["0", "1"],
        help="1 = Caption-only control group with frozen Camera and trainable Depth.",
    )
    # ── Depth Interpreter Parameters ──
    parser.add_argument(
        "--use_depth_interpreter",
        type=str,
        default="0",
        choices=["0", "1"],
        help="1=Enable independent Depth Interpreter; 0=Disable (default).",
    )
    # summary_interpreter produces summary tokens; direct_patch_embedding produces patch embeddings.
    parser.add_argument(
        "--depth_stage1_output_contract",
        type=str,
        default=DEPTH_STAGE1_OUTPUT_CONTRACT_SUMMARY,
        choices=DEPTH_STAGE1_OUTPUT_CONTRACT_CHOICES,
        help="Depth Stage1 output interface; direct_patch_embedding only allows this round's RGB-MRoPE alignment to train patch by patch.",
    )
    parser.add_argument(
        "--depth_encoder_type",
        type=str,
        default=DEPTH_ENCODER_TYPE_STATS_10D,
        choices=DEPTH_ENCODER_TYPE_CHOICES,
        help="Depth geometry encoding methods: stats_6d = 6-dimensional raw depth statistics; stats_6d_log = 6-dimensional log depth statistics; stats_10d = graph coordinates/frame ID + raw depth statistics (default); stats_10d_log = graph coordinates/frame ID + log depth statistics; stats_fourier = 52-dimensional Fourier statistics; center_frame_fourier = 12-dimensional center depth + absolute frame Fourier; scalar_5d = 5-dimensional center depth + fixed frame_idx/31.0.",
    )
    parser.add_argument(
        "--depth_latent_size",
        type=int,
        default=8,
        help="Number of tokens output by Depth Interpreter.",
    )
    # global=All windows share the same K; per_frame=Output K per frame, and K=1 is fixed for this round.
    parser.add_argument(
        "--depth_token_layout",
        type=str,
        default="global",
        choices=TOKEN_LAYOUT_CHOICES,
    )
    parser.add_argument(
        "--depth_token_position",
        type=str,
        default="before_question",
        choices=list(CAMERA_TOKEN_POSITION_CHOICES),
        help="Position of Depth tokens in the prompt; this round of direct replacement fixes before_question.",
    )
    # depth_loss_mode: full_caption retains the original Stage1 behavior; number_only only supervises the numeric value in "at X.X meters".
    parser.add_argument(
        "--depth_loss_mode",
        type=str,
        default=DEPTH_LOSS_MODE_FULL_CAPTION,
        choices=DEPTH_LOSS_MODE_CHOICES,
        help="Depth Stage1 loss: full_caption = complete answer CE; number_only = only depth numbers CE.",
    )
    # Allow full-caption supervision independently of the description template.
    parser.add_argument(
        "--depth_cot_version",
        type=str,
        default=None,
        help="The cot_version of the Depth Assistant Target; and depth_loss_mode are saved and verified independently.",
    )
    # This weight scales metric SmoothL1 relative to number-only cross-entropy; zero disables it.
    parser.add_argument(
        "--depth_meter_loss_weight",
        type=float,
        default=0.0,
        help="Depth numeric expects SmoothL1 weight; must be >=0, and positive values are only allowed for Depth-only number_only.",
    )
    # Group weights for relation numerical CE; Caption group weights are fixed at 1-weight, and this round is frozen at 0.5.
    parser.add_argument(
        "--depth_relation_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the relation group of caption_pair_delta; only 0.5 is allowed this round.",
    )
    # Only controls the additional attention score for Depth raw-geometry; 0 still retains the geometry MLP main path.
    parser.add_argument(
        "--depth_use_geometry_attention_bias",
        type=str,
        default="1",
        choices=["0", "1"],
        help="1 = enable geometry attention bias; 0 = use standard QK cross-attention only.",
    )
    parser.add_argument("--user_prompt", type=str, default=DEFAULT_USER_PROMPT)
    parser.add_argument(
        "--omit_user_question",
        type=str,
        default="0",
        choices=["0", "1"],
        help="1=Delete fixed Stage1 Question, retain only RGB, Camera token, and assistant Caption.",
    )
    # prompt_source controls whether the Question comes from a fixed string, JSONL with one Query per line, or if no natural language Question is provided during this round of frame-by-frame training.
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="fixed",
        choices=PROMPT_SOURCE_CHOICES,
        help="fixed=use --user_prompt; record=read each line Query; none=no natural language Question.",
    )
    # These three parameters jointly define the frame-wise caption responsibility scope, physical batch representation, and loss reduction.
    # Only allow complete combinations approved by Design, to avoid silently enabling partial semantics.
    parser.add_argument(
        "--depth_frame_supervision_scope",
        type=str,
        default=DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
        choices=DEPTH_FRAME_SUPERVISION_SCOPE_CHOICES,
        help="own_frame_only means that each Caption row can only read its own X_i/D_i.",
    )
    parser.add_argument(
        "--depth_frame_microbatch_mode",
        type=str,
        default=DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
        choices=DEPTH_FRAME_MICROBATCH_MODE_CHOICES,
        help="window_to_rows expands a source Window into V isolated Qwen batch rows.",
    )
    parser.add_argument(
        "--depth_frame_loss_reduction",
        type=str,
        default=DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        choices=DEPTH_FRAME_LOSS_REDUCTION_CHOICES,
        help="mean_per_source_window first computes the mean of Caption CE per frame, preventing long windows from adding unnecessary weight.",
    )
    parser.add_argument("--skip_final_save", type=str, default="0", choices=["0", "1"])
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256_file(path: str) -> str:
    """Compute SHA256 for small source code files, preserving exact code identity for checkpoint."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_code_snapshot_identity() -> dict:
    """Record Stage1 forward and save the local source code hash directly dependent on the chain link."""
    src_dir = Path(__file__).resolve().parent
    filenames = (
        "main_camera_interpreter.py",
        "georeader_data.py",
        "camera_interpreter.py",
        "depth_interpreter.py",
        "depth_patch_tokenizer.py",
        "trainer_camera_interpreter.py",
        "n_images_length_sampler.py",
        "pixel_budget_defaults.py",
        "visual_encoder_utils.py",
    )
    files = {}
    for filename in filenames:
        path = src_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Stage1 code snapshot missing source code: {path}")
        files[str(path)] = _sha256_file(str(path))
    aggregate = hashlib.sha256(
        "\n".join(f"{path}\t{files[path]}" for path in sorted(files)).encode("utf-8")
    ).hexdigest()
    return {"sha256": aggregate, "files": files}


def _load_data_provenance(
    report_path: Optional[str], data_paths: List[str]
) -> Optional[dict]:
    """Strictly verify the derived data report and return the data identity chain that can be saved with checkpoint."""
    if report_path is None:
        return None
    resolved_report_path = Path(report_path).resolve()
    if not resolved_report_path.is_file():
        raise FileNotFoundError(f"Data report not found: {resolved_report_path}")
    with resolved_report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise RuntimeError(
            f"Data report did not pass checks: path={resolved_report_path}, status={(report.get('status') if isinstance(report, dict) else None)!r}"
        )
    if report.get("format") == "georeader-depth-expansion":
        # Internal release entry expands while preserving original frame order/target checks; no need to repeatedly scan large files to calculate summaries.
        paths = [str(Path(path).resolve()) for path in data_paths]
        if paths != [report["output_path"]] or not Path(report["input_path"]).is_file():
            raise RuntimeError(
                "Depth frame-by-frame input inconsistent with auto-expanded report"
            )
        contract = report["contract"]
        if (contract["source_window_rows"], contract["expanded_rows"]) != (
            35422,
            228691,
        ):
            raise RuntimeError("Per-frame depth input is not supported by this recipe.")
        return {
            "report_path": str(resolved_report_path),
            "input_path": report["input_path"],
            "output_path": report["output_path"],
            "contract": contract,
            "data_order_mode": contract["data_order_mode"],
            "data_order_seed": contract["data_order_seed"],
            "cot_version": contract["new_cot_version"],
        }
    required_sections = {"contract", "input", "output", "generator", "audit"}
    missing_sections = sorted(required_sections - set(report))
    if missing_sections:
        raise RuntimeError(
            f"Data report is missing sections {missing_sections}: {resolved_report_path}"
        )
    contract = report["contract"]
    if (
        not isinstance(contract, dict)
        or not contract.get("data_order_mode")
        or "data_order_seed" not in contract
        or not contract.get("new_cot_version")
    ):
        raise RuntimeError(
            f"Data report is missing mode/seed/cot_version: {resolved_report_path}"
        )
    resolved_data_paths = [str(Path(path).resolve()) for path in data_paths]
    if len(resolved_data_paths) != 1:
        raise RuntimeError(
            f"Stage 1 accepts exactly one training JSONL with a data report: data_paths={resolved_data_paths}"
        )
    expected_output_path = str(Path(report["output"]["path"]).resolve())
    if resolved_data_paths != [expected_output_path]:
        raise RuntimeError(
            f"Training data path does not match the data report: data_paths={resolved_data_paths}, report_output={expected_output_path}"
        )
    for section_name in ("input", "output", "generator"):
        section = report[section_name]
        if (
            not isinstance(section, dict)
            or not section.get("path")
            or not section.get("sha256")
        ):
            raise RuntimeError(
                f"Data report section {section_name} is missing path/sha256: {resolved_report_path}"
            )
        artifact_path = Path(section["path"]).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Data report section {section_name}: file not found: {artifact_path}"
            )
        actual_sha256 = _sha256_file(str(artifact_path))
        if actual_sha256 != str(section["sha256"]):
            raise RuntimeError(
                f"Data report section {section_name}: SHA256 mismatch: path={artifact_path}, saved={section['sha256']}, actual={actual_sha256}"
            )
    if report["audit"].get("repeat_generation_sha256_match") is not True:
        raise RuntimeError(
            f"Data report did not pass the regeneration check: {resolved_report_path}"
        )
    return {
        "report_path": str(resolved_report_path),
        "report_sha256": _sha256_file(str(resolved_report_path)),
        "data_order_mode": str(contract["data_order_mode"]),
        "data_order_seed": int(contract["data_order_seed"]),
        "cot_version": str(contract["new_cot_version"]),
        "input_path": str(Path(report["input"]["path"]).resolve()),
        "input_sha256": str(report["input"]["sha256"]),
        "output_path": expected_output_path,
        "output_sha256": str(report["output"]["sha256"]),
        "generator_path": str(Path(report["generator"]["path"]).resolve()),
        "generator_sha256": str(report["generator"]["sha256"]),
        "contract": json.loads(json.dumps(contract, ensure_ascii=False)),
    }


def _build_stage1_checkpoint_meta(
    *,
    args,
    model,
    train_dataset,
    branch: str,
    resolved_min_pixels: int,
    resolved_max_pixels: int,
) -> dict:
    """The complete effective config that allows independent reconstruction of Stage1 semantics from a checkpoint."""
    if branch not in {"Camera", "Depth"}:
        raise RuntimeError(f"Unknown Stage1 branch: {branch!r}")
    view_counts = Counter(int(record["n_images"]) for record in train_dataset.records)
    if branch == "Camera":
        if model.camera_interpreter is None:
            raise RuntimeError(
                "When building Camera meta, the model does not have a Camera Interpreter."
            )
        interpreter_config = model.camera_interpreter.config.to_dict()
        input_recipe = {
            "geometry_source": "gt_camera_pose",
            "path_field": "pose_path",
            "reference_frame": "image_1_canonicalized_frame_0",
            "translation_encoding": interpreter_config["camera_translation_encoding"],
            "frame_index_encoding": interpreter_config["camera_frame_index_encoding"],
            "npz_schema": {
                "extrinsics": "[V,3,4] float32, frame 0 = [I|0]",
                "intrinsics": "[V,3,3] float32",
                "image_sizes": "[V,2] width,height",
                "order_checks": "image_paths/frame_ids when present",
            },
        }
        loss_mode = model.camera_loss_mode
        cot_version = model.camera_cot_version
        token_position = interpreter_config["camera_token_position"]
        if cot_version == "mindcube_da3_camera_image1_meter":
            # Original shared meta must hardcode GT; MindCube DA3 must record the true source to prevent downstream misjudgment of GT recipes.
            input_recipe.update(
                geometry_source="da3_nested",
                runtime_unit="meter",
                metric_ground_truth=False,
                geometry_status="da3_training_geometry",
            )
    else:
        if model.depth_interpreter is None and model.depth_patch_tokenizer is None:
            raise RuntimeError(
                "When building Depth meta, the model does not have a Depth Interpreter or DepthPatchTokenizer."
            )
        if model.depth_patch_tokenizer is not None:
            interpreter_config = model.depth_patch_tokenizer.config.to_dict()
            geometry_mode = interpreter_config["depth_encoder_type"]
            k_source = "Qwen image_grid_thw exact per-patch RGB alignment"
            token_position = interpreter_config["token_position"]
        else:
            interpreter_config = model.depth_interpreter.config.to_dict()
            geometry_mode = interpreter_config["depth_encoder_type"]
            k_source = "Qwen image_grid_thw aligned depth patches"
            token_position = interpreter_config["depth_token_position"]
        input_recipe = {
            "geometry_source": "gt_depth",
            "path_field": "depth_paths",
            "npz_schema": {
                "keys": ["depth_map", "depth_size"],
                "stored_dtype": "uint16",
                "stored_unit": "millimeter",
                "runtime_dtype": "float32",
                "runtime_unit": "meter",
                "invalid_value": 0,
            },
            "k_source": k_source,
            "geometry_mode": geometry_mode,
        }
        loss_mode = model.depth_loss_mode
        cot_version = model.depth_cot_version
    if not cot_version:
        raise RuntimeError(
            f"{branch} checkpoint meta is missing cot_version; new direct checkpoint does not allow target identity to be empty."
        )
    data_metadata = {
        "paths": list(train_dataset.data_paths),
        "source_rows": dict(train_dataset.source_row_counts),
        "training_rows": len(train_dataset),
    }
    if getattr(train_dataset, "depth_perframe_caption_microbatch", False):
        data_metadata.update(
            {
                "source_window_rows": len(train_dataset),
                "expanded_rows": sum(train_dataset.source_row_counts.values()),
            }
        )
    data_provenance = _load_data_provenance(
        args.data_provenance_report,
        list(train_dataset.data_paths),
    )
    if data_provenance is not None:
        if data_provenance["cot_version"] != cot_version:
            raise RuntimeError(
                f"cot_version in the data report does not match the dataset: report={data_provenance['cot_version']!r}, dataset={cot_version!r}"
            )
        data_metadata["provenance"] = data_provenance
        if getattr(train_dataset, "depth_perframe_caption_microbatch", False):
            contract = data_provenance["contract"]
            source_window_rows = int(contract.get("source_window_rows", -1))
            expanded_rows = int(contract.get("expanded_rows", -1))
            physical_rows = sum(train_dataset.source_row_counts.values())
            if (
                source_window_rows <= 0
                or expanded_rows != physical_rows
                or len(train_dataset) > source_window_rows
            ):
                raise RuntimeError(
                    f"Frame count in the data report does not match the dataset: source_windows={source_window_rows}, expanded={expanded_rows}, physical={physical_rows}, effective={len(train_dataset)}"
                )
            # Save the complete dataset identity separately from the number of samples used in a run.
            data_metadata["source_window_rows"] = source_window_rows
            data_metadata["expanded_rows"] = expanded_rows
            data_metadata["effective_training_rows"] = len(train_dataset)
    initialization = {"mode": "random"}
    if args.init_interpreter_checkpoint:
        checkpoint_dir = os.path.abspath(
            os.path.expanduser(str(args.init_interpreter_checkpoint))
        )
        if branch == "Depth" and model.depth_interpreter is not None:
            artifact_names = (
                "depth_interpreter.pt",
                "depth_interpreter_config.json",
            )
        elif branch == "Camera" and model.camera_interpreter is not None:
            artifact_names = (
                "camera_interpreter.pt",
                "camera_interpreter_config.json",
            )
        else:
            raise RuntimeError(
                f"init_interpreter_checkpoint does not match current Stage1 branch: branch={branch}, checkpoint={checkpoint_dir}"
            )
        artifacts_sha256 = {}
        for filename in artifact_names:
            artifact_path = os.path.join(checkpoint_dir, filename)
            if not os.path.isfile(artifact_path):
                raise FileNotFoundError(
                    f"Initialization checkpoint missing artifact: {artifact_path}"
                )
            artifacts_sha256[filename] = _sha256_file(artifact_path)
        initialization = {
            "mode": "weights_only_fresh_optimizer_scheduler",
            "source_checkpoint": checkpoint_dir,
            "artifacts_sha256": artifacts_sha256,
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "trainer_state_loaded": False,
        }
    elif args.resume_from_checkpoint:
        resume_dir = os.path.abspath(
            os.path.expanduser(str(args.resume_from_checkpoint))
        )
        resume_meta_path = os.path.join(resume_dir, "camera_checkpoint.json")
        if not os.path.isfile(resume_meta_path):
            raise FileNotFoundError(
                f"Stage1 resume lacks initialization of identity source: {resume_meta_path}"
            )
        with open(resume_meta_path, "r", encoding="utf-8") as handle:
            resume_meta = json.load(handle)
        saved_initialization = resume_meta.get("initialization")
        if not isinstance(saved_initialization, dict) or not saved_initialization.get(
            "mode"
        ):
            raise RuntimeError(
                f"Stage1 resume metadata lacks initialization contract: {resume_meta_path}"
            )
        # Preserve initialization provenance when resuming training.
        initialization = json.loads(
            json.dumps(saved_initialization, ensure_ascii=False)
        )
        initialization["resumed_from_checkpoint"] = resume_dir
    return {
        "stage1_meta_schema_version": 1,
        "stage1_branch": branch,
        "base_model": model.base_model_name_or_path,
        "initialization": initialization,
        "data": data_metadata,
        "code_snapshot_identity": _build_code_snapshot_identity(),
        "input_recipe": input_recipe,
        "frame_order": {
            "image_field": "image_input",
            "frame_id_field": "window_frame_ids",
            "view_count_distribution": {
                str(num_views): int(count)
                for num_views, count in sorted(view_counts.items())
            },
        },
        "processor": {
            "min_pixels": int(resolved_min_pixels),
            "max_pixels": int(resolved_max_pixels),
            "expected_image_size": str(args.expected_image_size),
        },
        "interpreter": {
            "config": interpreter_config,
            "token_position": token_position,
            "llm_use_image_tokens": args.llm_use_image_tokens == "1",
            "interpreter_use_image_tokens": args.interpreter_use_image_tokens == "1",
        },
        "training_target": {
            "loss_mode": loss_mode,
            "cot_version": cot_version,
            "prompt_source": str(args.prompt_source),
            "prompt": str(args.user_prompt),
            "omit_user_question": args.omit_user_question == "1",
            "llm_trainable": False,
            "vision_encoder_trainable": False,
            "interpreter_trainable": True,
            **(
                {
                    "stage1_task": "camera_reference_target_pairs",
                    "pair_scope": "image_1_plus_original_target_frame",
                    # Count pairs from the windows used for training and retain the complete source row count.
                    "source_window_rows": len(train_dataset.records),
                    "expanded_pair_rows": int(
                        sum(int(row["n_images"]) - 1 for row in train_dataset.records)
                    ),
                    "optimizer_sample_unit": "source_window",
                    "pair_loss_reduction": "target_frame_mean_per_source_window",
                    "target_frame_weighting": "equal_all_targets",
                    "frame_index_encoding": str(args.camera_frame_index_encoding),
                }
                if args.camera_pair_rows == "1"
                else {}
            ),
            **(
                {
                    "stage1_task": "depth_frozen_camera_caption_control",
                    "camera_trainable": False,
                    "depth_trainable": True,
                    "frozen_camera_checkpoint": str(args.frozen_camera_checkpoint),
                    "frozen_camera_artifacts_sha256": {
                        filename: _sha256_file(
                            os.path.join(args.frozen_camera_checkpoint, filename)
                        )
                        for filename in (
                            "camera_interpreter.pt",
                            "camera_interpreter_config.json",
                        )
                    },
                    "camera_token_layout": model.camera_interpreter.config.to_dict(),
                }
                if args.depth_camera_caption_control == "1"
                else {}
            ),
            **(
                {
                    # crossframe still uses the same loss; metadata must distinguish whether the second number exists to support Stage2 and strict reload tracing.
                    "stage1_task": (
                        "depth_caption_pair_delta_crossframe"
                        if cot_version
                        == DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION
                        else "depth_caption_pair_delta"
                    ),
                    "assistant_target_field": "depth_pair_answer",
                    "relation_mask_source_field": "depth_pair_relation_answer",
                    "caption_mask_source_field": "depth_scene_cot",
                    "relation_loss_weight": float(args.depth_relation_loss_weight),
                    "caption_loss_weight": float(1.0 - args.depth_relation_loss_weight),
                    "relation_pair_scope": (
                        "same_and_cross_frame"
                        if cot_version
                        == DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION
                        else "same_frame"
                    ),
                }
                if loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA
                else {}
            ),
            **(
                {
                    "stage1_task": DEPTH_PERFRAME_CAPTION_MICROBATCH_TASK,
                    "assistant_target_field": "depth_single_frame_caption",
                    "frame_supervision_scope": args.depth_frame_supervision_scope,
                    "frame_microbatch_mode": args.depth_frame_microbatch_mode,
                    "frame_loss_reduction": args.depth_frame_loss_reduction,
                    "source_window_rows": int(data_metadata["source_window_rows"]),
                    "expanded_rows": int(data_metadata["expanded_rows"]),
                    "optimizer_sample_unit": "source_window",
                    # Frames without numbers under number_only have no CE target; valid target frames still occupy their own slots.
                    "target_frame_weighting": "equal_nonempty_numeric_targets",
                    "qwen_batch_semantics": "one_row_per_source_frame_no_cross_row_attention",
                    "initialization_policy": (
                        "from_scratch"
                        if os.environ.get("DEPTH_FROM_SCRATCH", "0") == "1"
                        else "weights_only"
                    ),
                }
                if args.depth_frame_microbatch_mode
                == DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS
                else {}
            ),
        },
        "optimization": {
            "seed": 42,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_batch_size": int(os.environ.get("WORLD_SIZE", "1"))
            * int(args.per_device_train_batch_size)
            * int(args.gradient_accumulation_steps),
            "epochs": float(args.epochs),
            "max_steps": int(args.max_steps),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "learning_rate": float(args.learning_rate),
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "warmup_steps": int(args.warmup_steps),
            "weight_decay": float(args.weight_decay),
            "train_sampler_mode": str(args.train_sampler_mode),
            "save_steps": int(args.save_steps),
            "save_total_limit": int(args.save_total_limit),
        },
    }


def _parse_expected_image_size(value: Optional[str]) -> Optional[tuple]:
    if value is None:
        raise RuntimeError(
            "expected_image_size cannot be empty; training image size check is not allowed to be disabled."
        )
    text = str(value).strip().lower()
    if text == MIXED_IMAGE_SIZE:
        return MIXED_IMAGE_SIZE
    if text in {"", "none", "null", "0", "off", "disable", "disabled"}:
        raise RuntimeError(
            f"expected_image_size is not allowed to be disabled, current: {value!r}"
        )
    text = text.replace(",", "x")
    parts = [p.strip() for p in text.split("x") if p.strip()]
    if len(parts) != 2:
        raise RuntimeError(
            f"expected_image_size must be in WxH format, for example 480x640, current: {value!r}"
        )
    width, height = [int(p) for p in parts]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"expected_image_size must be positive, current: {value!r}")
    return width, height


def _assert_pil_image_size(
    image: Image.Image, path: str, expected_size: Optional[tuple], context: str
):
    if expected_size == MIXED_IMAGE_SIZE:
        return
    if expected_size is None:
        raise RuntimeError(
            f"{context} expected_image_size cannot be empty; image size validation is not allowed to be disabled."
        )
    actual = tuple(image.size)
    if actual != tuple(expected_size):
        raise RuntimeError(
            f"{context} Image size does not match training specification: path={path}, actual PIL(W,H)={actual}, expected PIL(W,H)={tuple(expected_size)}."
        )


def _set_image_processor_pixel_budget(
    image_processor, *, min_pixels: Optional[int], max_pixels: Optional[int]
):
    if image_processor is None:
        raise RuntimeError(
            "processor is missing image_processor, cannot set pixel budget."
        )
    size = getattr(image_processor, "size", None)
    # Support processor size containers through their mapping interface.
    if size is None or not all(
        hasattr(size, name) for name in ("__getitem__", "__setitem__")
    ):
        raise RuntimeError(
            f"image_processor.size must support key reading and writing, current={type(size)}"
        )
    # The local Qwen image processor actually participates in smart_resize through the size dictionary.
    # Only changing the image_processor.max_pixels property won't take effect; you need to modify both places together to prevent the original image training from silently using the default large grid.
    if min_pixels is not None:
        min_pixels = int(min_pixels)
        if min_pixels <= 0:
            raise RuntimeError(f"min_pixels must be > 0, current={min_pixels}")
        image_processor.min_pixels = min_pixels
        image_processor.size["shortest_edge"] = min_pixels
    if max_pixels is not None:
        max_pixels = int(max_pixels)
        if max_pixels <= 0:
            raise RuntimeError(f"max_pixels must be > 0, current={max_pixels}")
        image_processor.max_pixels = max_pixels
        image_processor.size["longest_edge"] = max_pixels
    min_value = int(image_processor.size["shortest_edge"])
    max_value = int(image_processor.size["longest_edge"])
    if min_value > max_value:
        raise RuntimeError(
            f"image processor pixel budget illegal: min_pixels={min_value}, max_pixels={max_value}"
        )
    return min_value, max_value


def _read_processor_pixel_budget_config(
    checkpoint_dir: Optional[str],
) -> Optional[tuple]:
    """Read min/max_pixels from checkpoint/preprocessor_config.json; return None if not configured."""
    if checkpoint_dir is None:
        return None
    config_path = os.path.join(
        os.path.abspath(os.path.expanduser(str(checkpoint_dir))),
        "preprocessor_config.json",
    )
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    size = cfg.get("size") if isinstance(cfg.get("size"), dict) else {}
    min_pixels = cfg.get("min_pixels", size.get("shortest_edge"))
    max_pixels = cfg.get("max_pixels", size.get("longest_edge"))
    if min_pixels is None or max_pixels is None:
        raise RuntimeError(
            f"preprocessor_config is missing min/max_pixels: {config_path}"
        )
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise RuntimeError(
            f"preprocessor_config pixel budget is invalid: {config_path}, min_pixels={min_pixels}, max_pixels={max_pixels}"
        )
    return min_pixels, max_pixels, config_path


def _resolve_pixel_budget_from_checkpoint_configs(
    *,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    expected_min_pixels: Optional[int] = None,
    expected_max_pixels: Optional[int] = None,
    checkpoint_dirs: Optional[List[Optional[str]]] = None,
    context: str,
) -> tuple:
    """Parse pixel budget from checkpoint processor configuration to prevent Stage2/eval from overriding Stage1's max_pixels."""
    budgets = []
    for checkpoint_dir in checkpoint_dirs or []:
        budget = _read_processor_pixel_budget_config(checkpoint_dir)
        if budget is not None:
            budgets.append(budget)
    ckpt_min = None
    ckpt_max = None
    source_paths = []
    for cur_min, cur_max, source_path in budgets:
        if ckpt_min is None:
            ckpt_min = int(cur_min)
            ckpt_max = int(cur_max)
        elif int(cur_min) != ckpt_min or int(cur_max) != ckpt_max:
            raise RuntimeError(
                f"{context} Multiple checkpoint preprocessor_config.json files are inconsistent and cannot be mixed: expected=({ckpt_min},{ckpt_max}), got=({cur_min},{cur_max}), path={source_path}"
            )
        source_paths.append(source_path)

    manual_values = {
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "expected_min_pixels": expected_min_pixels,
        "expected_max_pixels": expected_max_pixels,
    }
    if ckpt_min is not None:
        provided = {
            key: value for key, value in manual_values.items() if value is not None
        }
        if provided:
            raise RuntimeError(
                f"{context} Checkpoint/preprocessor_config.json is already bound, manual pixel budget transmission is prohibited: provided={provided}, checkpoint=({ckpt_min},{ckpt_max}), sources={source_paths}"
            )

    base_min = int(ckpt_min) if ckpt_min is not None else int(DEFAULT_IMAGE_MIN_PIXELS)
    base_max = int(ckpt_max) if ckpt_max is not None else int(DEFAULT_IMAGE_MAX_PIXELS)
    resolved_min = base_min if min_pixels is None else int(min_pixels)
    resolved_max = base_max if max_pixels is None else int(max_pixels)
    if resolved_min <= 0 or resolved_max <= 0 or resolved_min > resolved_max:
        raise RuntimeError(
            f"{context} pixel budget illegal: min_pixels={resolved_min}, max_pixels={resolved_max}"
        )
    if ckpt_min is not None and (resolved_min != ckpt_min or resolved_max != ckpt_max):
        raise RuntimeError(
            f"{context} pixel budget and checkpoint processor are inconsistent: args=({resolved_min},{resolved_max}), checkpoint=({ckpt_min},{ckpt_max}), sources={source_paths}"
        )

    resolved_expected_min = (
        resolved_min if expected_min_pixels is None else int(expected_min_pixels)
    )
    resolved_expected_max = (
        resolved_max if expected_max_pixels is None else int(expected_max_pixels)
    )
    if resolved_expected_min != resolved_min or resolved_expected_max != resolved_max:
        raise RuntimeError(
            f"{context} expected pixel budget and actual pixel budget are inconsistent: actual=({resolved_min},{resolved_max}), expected=({resolved_expected_min},{resolved_expected_max})"
        )
    source = ",".join(source_paths) if source_paths else "project default"
    return (
        resolved_min,
        resolved_max,
        resolved_expected_min,
        resolved_expected_max,
        source,
    )


def _get_record_image_sizes(
    record: dict, *, row_idx: Optional[int] = None, required: bool = False
) -> Optional[List[tuple]]:
    loc = f"row={row_idx}, " if row_idx is not None else ""
    if "image_sizes" not in record:
        if required:
            raise RuntimeError(
                f"Sample missing image_sizes, cannot construct pose-only grid aligned with Qwen: {loc}idx={record.get('idx')}, keys={sorted(record.keys())}"
            )
        return None
    raw_sizes = record["image_sizes"]
    if not isinstance(raw_sizes, list) or not raw_sizes:
        raise RuntimeError(
            f"image_sizes must be a non-empty list: {loc}idx={record.get('idx')}, value={raw_sizes!r}"
        )
    sizes = []
    for size_idx, item in enumerate(raw_sizes):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError(
                f"image_sizes[{size_idx}Must be [W,H]: {loc}idx={record.get('idx')}, value={item!r}"
            )
        width, height = [int(x) for x in item]
        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"image_sizes[{size_idx}must be positive: {loc}idx={record.get('idx')}, value={item!r}"
            )
        sizes.append((width, height))
    return sizes


def _require_record_feature_dir(
    record: dict, *, row_idx: Optional[int] = None, context: str = "sample"
) -> str:
    loc = f"row={row_idx}, " if row_idx is not None else ""
    sample_id = record.get("idx", record.get("group_id", "N/A"))
    if "feature_dir" not in record:
        raise RuntimeError(
            f"{context} Missing feature_dir, cannot use --feature_vggt_dir or fallback to default directory: {loc}idx={sample_id}, keys={sorted(record.keys())}"
        )
    raw = record.get("feature_dir")
    if raw is None or not str(raw).strip():
        raise RuntimeError(
            f"{context} feature_dir cannot be empty: {loc}idx={sample_id}"
        )
    return os.path.abspath(os.path.expanduser(str(raw).strip()))


def _infer_eval_bucket(record: dict, *, row_idx: Optional[int] = None) -> str:
    """Bucket eval during training; throw errors for unrecognized data structures."""
    loc = f"row={row_idx}, " if row_idx is not None else ""
    dataset = str(record.get("dataset", "")).strip().lower()
    source = str(record.get("source", "")).strip().lower()
    image_input = record.get("image_input")
    first_image = ""
    if isinstance(image_input, list) and image_input:
        first_image = str(image_input[0]).replace("\\", "/")

    if dataset == "ego3d":
        if not source:
            raise RuntimeError(
                f"Ego3D sample missing source, cannot perform eval bucketing: {loc}idx={record.get('idx')}"
            )
        return f"ego3d:{source}"
    if source in {"scannet", "scannetpp"}:
        return source

    if "/among/" in first_image:
        return "mindcube:among"
    if "/around/" in first_image:
        return "mindcube:around"
    if "/rotation/" in first_image:
        return "mindcube:rotation"

    if dataset == "mindcube":
        raise RuntimeError(
            f"MindCube samples cannot be recognized as among/around/rotation from image_input, and cannot be bucketed for evaluation: {loc}idx={record.get('idx')}, image={first_image!r}"
        )
    if dataset:
        return dataset
    raise RuntimeError(
        f"Sample lacks dataset, and image_input does not contain among/around/rotation, cannot perform eval bucketing: {loc}idx={record.get('idx')}, image={first_image!r}"
    )


def _load_json_or_jsonl(path: str) -> List[dict]:
    """Read training records and resolve dataset-package paths."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]
    if not isinstance(records, list):
        raise RuntimeError(f"data_path must be a JSON array or JSONL: {path}")
    return resolve_records(records, path)


def _group_depth_perframe_microbatch_rows(
    records: List[dict], *, data_path: str
) -> List[dict]:
    """Reorganize expanded rows strictly by source triplets into Dataset's Window item."""
    grouped = []
    current = None
    for row_idx, record in enumerate(records):
        context = f"file={data_path}, row={row_idx}"
        required = {
            "source_window_idx",
            "source_frame_index",
            "source_n_images",
            "source_record_idx",
            "stage1_task",
            "cot_version",
            "depth_single_frame_caption",
            "source",
            "scene_id",
            "n_images",
            "image_input",
            "image_sizes",
            "window_frame_ids",
            "depth_paths",
            "has_depth",
        }
        missing = sorted(required - set(record))
        if missing:
            raise RuntimeError(
                f"Per-frame expanded row is missing fields {missing}: {context}"
            )
        if record["stage1_task"] != DEPTH_PERFRAME_CAPTION_MICROBATCH_TASK:
            raise RuntimeError(
                f"stage1_task mismatch: {context}, value={record['stage1_task']!r}"
            )
        if record["cot_version"] != DEPTH_PERFRAME_CAPTION_MICROBATCH_COT_VERSION:
            raise RuntimeError(
                f"cot_version mismatch: {context}, value={record['cot_version']!r}"
            )
        if int(record["n_images"]) != 1:
            raise RuntimeError(
                f"Unfolding frame by frame, n_images must be 1: {context}"
            )
        for field in ("image_input", "image_sizes", "window_frame_ids", "depth_paths"):
            value = record[field]
            if not isinstance(value, list) or len(value) != 1:
                raise RuntimeError(
                    f"Frame-by-frame unroll row {field} Exactly one of: {context}"
                )
        window_idx = int(record["source_window_idx"])
        frame_idx = int(record["source_frame_index"])
        n_images = int(record["source_n_images"])
        if window_idx < 0 or frame_idx < 0 or n_images <= 0 or frame_idx >= n_images:
            raise RuntimeError(
                f"source Window/frame contract illegal: {context}, window={window_idx}, frame={frame_idx}, V={n_images}"
            )
        if current is None or window_idx != current["source_window_idx"]:
            if current is not None:
                if len(current["depth_single_frame_captions"]) != int(
                    current["n_images"]
                ):
                    raise RuntimeError(
                        f"The previous source Window was not fully reorganized: window={current['source_window_idx']}, rows={len(current['depth_single_frame_captions'])}, V={current['n_images']}"
                    )
                grouped.append(current)
            if window_idx != len(grouped) or frame_idx != 0:
                raise RuntimeError(
                    f"source Window must be arranged consecutively in the range 0..N, frame 0..V-1: {context}, window={window_idx}, frame={frame_idx}, expected_window={len(grouped)}"
                )
            current = {
                "idx": int(record["source_record_idx"]),
                "source_window_idx": window_idx,
                "source": str(record["source"]),
                "scene_id": str(record["scene_id"]),
                "n_images": n_images,
                "image_input": [],
                "image_sizes": [],
                "window_frame_ids": [],
                "depth_paths": [],
                "depth_single_frame_captions": [],
                "has_depth": record["has_depth"],
                "cot_version": str(record["cot_version"]),
                "stage1_task": str(record["stage1_task"]),
            }
            if "eval_bucket" in record:
                current["eval_bucket"] = record["eval_bucket"]
        expected_frame = len(current["depth_single_frame_captions"])
        if frame_idx != expected_frame:
            raise RuntimeError(
                f"Frame order is discontinuous for the same source Window: {context}, actual={frame_idx}, expected={expected_frame}"
            )
        invariants = {
            "source_n_images": n_images,
            "source_record_idx": int(record["source_record_idx"]),
            "source": str(record["source"]),
            "scene_id": str(record["scene_id"]),
            "has_depth": record["has_depth"],
        }
        expected_invariants = {
            "source_n_images": int(current["n_images"]),
            "source_record_idx": int(current["idx"]),
            "source": current["source"],
            "scene_id": current["scene_id"],
            "has_depth": current["has_depth"],
        }
        if invariants != expected_invariants:
            raise RuntimeError(
                f"Identity field changed for the same source Window: {context}, actual={invariants}, expected={expected_invariants}"
            )
        target = str(record["depth_single_frame_caption"])
        if not target.startswith("depth layout:\n") or "\nimage " in target:
            raise RuntimeError(
                f"Frame-by-frame Caption target format is invalid: {context}, target={target!r}"
            )
        current["image_input"].append(record["image_input"][0])
        current["image_sizes"].append(record["image_sizes"][0])
        current["window_frame_ids"].append(record["window_frame_ids"][0])
        current["depth_paths"].append(record["depth_paths"][0])
        current["depth_single_frame_captions"].append(target)
    if current is not None:
        if len(current["depth_single_frame_captions"]) != int(current["n_images"]):
            raise RuntimeError(
                f"The last source Window was not fully reorganized: window={current['source_window_idx']}, rows={len(current['depth_single_frame_captions'])}, V={current['n_images']}"
            )
        grouped.append(current)
    if not grouped:
        raise RuntimeError(
            f"Frame-by-frame data has no recombinaable source Window: {data_path}"
        )
    return grouped


class GlobalSceneCoTDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_folder: str,
        feature_vggt_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        expected_image_size: Optional[tuple] = (480, 640),
        load_images: bool = True,
        use_camera_cot: bool = True,
        camera_loss_mode: str = CAMERA_LOSS_MODE_FULL_CAPTION,
        camera_cot_version: Optional[str] = None,
        depth_loss_mode: str = DEPTH_LOSS_MODE_FULL_CAPTION,
        depth_cot_version: Optional[str] = None,
        prompt_source: str = "fixed",
        depth_frame_supervision_scope: str = DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
        depth_frame_microbatch_mode: str = DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
        depth_frame_loss_reduction: str = DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        camera_context_only: bool = False,
        camera_pair_rows: bool = False,
    ):
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = list(data_path)
        if not data_paths:
            raise RuntimeError("--data_path must specify at least one JSON/JSONL file.")
        self.data_paths = [os.path.abspath(os.path.expanduser(p)) for p in data_paths]
        self.data_path = ",".join(self.data_paths)
        self.image_folder = os.path.abspath(os.path.expanduser(image_folder))
        self.feature_vggt_dir = (
            os.path.abspath(os.path.expanduser(feature_vggt_dir))
            if feature_vggt_dir is not None
            else None
        )
        self.expected_image_size = expected_image_size
        self.load_images = bool(load_images)
        self.use_camera_cot = bool(use_camera_cot)
        self.camera_pair_rows = bool(camera_pair_rows)
        if self.camera_pair_rows and (
            not self.use_camera_cot
            or camera_loss_mode != CAMERA_LOSS_MODE_MOTION_CONTENT
        ):
            raise RuntimeError(
                "Camera pairing lines only allow Camera-only + motion_content."
            )
        # B's supervised text belongs to Depth; the Camera provides frozen context only when has_pose=1 in this line.
        self.camera_context_only = bool(camera_context_only)
        self.camera_loss_mode = str(camera_loss_mode)
        if self.camera_loss_mode not in CAMERA_LOSS_MODE_CHOICES:
            raise RuntimeError(
                f"camera_loss_mode must be {CAMERA_LOSS_MODE_CHOICES}: actual={self.camera_loss_mode!r}"
            )
        if (
            not self.use_camera_cot
            and self.camera_loss_mode != CAMERA_LOSS_MODE_FULL_CAPTION
        ):
            raise RuntimeError(
                "Depth-only Stage1 cannot use Camera motion_content loss."
            )
        self.camera_cot_version = (
            str(camera_cot_version).strip() if camera_cot_version is not None else None
        )
        if self.camera_cot_version == "":
            raise RuntimeError("camera_cot_version cannot be an empty string.")
        self.depth_loss_mode = str(depth_loss_mode)
        if self.depth_loss_mode not in DEPTH_LOSS_MODE_CHOICES:
            raise RuntimeError(
                f"depth_loss_mode must be {DEPTH_LOSS_MODE_CHOICES}: actual={self.depth_loss_mode!r}"
            )
        if self.use_camera_cot and self.depth_loss_mode != DEPTH_LOSS_MODE_FULL_CAPTION:
            raise RuntimeError(
                "Camera Stage1 can only use full_caption; local loss for Depth cannot be used with Camera."
            )
        self.depth_cot_version = (
            str(depth_cot_version).strip() if depth_cot_version is not None else None
        )
        if self.depth_cot_version == "":
            raise RuntimeError("depth_cot_version cannot be an empty string.")
        self.prompt_source = str(prompt_source).strip()
        if self.prompt_source not in PROMPT_SOURCE_CHOICES:
            raise RuntimeError(
                f"prompt_source must be {PROMPT_SOURCE_CHOICES}: actual={self.prompt_source!r}"
            )
        self.depth_frame_supervision_scope = str(depth_frame_supervision_scope)
        self.depth_frame_microbatch_mode = str(depth_frame_microbatch_mode)
        self.depth_frame_loss_reduction = str(depth_frame_loss_reduction)
        self.depth_perframe_caption_microbatch = (
            self.depth_frame_microbatch_mode
            == DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS
        )
        expected_frame_contract = (
            DEPTH_FRAME_SUPERVISION_SCOPE_OWN,
            DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS,
            DEPTH_FRAME_LOSS_REDUCTION_WINDOW_MEAN,
        )
        actual_frame_contract = (
            self.depth_frame_supervision_scope,
            self.depth_frame_microbatch_mode,
            self.depth_frame_loss_reduction,
        )
        if self.depth_perframe_caption_microbatch:
            if actual_frame_contract != expected_frame_contract:
                raise RuntimeError(
                    f"Frame-by-frame Caption micro-batch must fully utilize own_frame_only/window_to_rows/mean_per_source_window: actual={actual_frame_contract}"
                )
            if (
                self.use_camera_cot
                or self.camera_context_only
                or self.depth_loss_mode
                not in {DEPTH_LOSS_MODE_FULL_CAPTION, DEPTH_LOSS_MODE_NUMBER_ONLY}
                or self.prompt_source not in {"none", "fixed"}
                or self.depth_cot_version
                != DEPTH_PERFRAME_CAPTION_MICROBATCH_COT_VERSION
            ):
                raise RuntimeError(
                    f"Frame-by-frame Caption micro-batch only allows Depth-only + full_caption/number_only + prompt_source=none/fixed + cot_version={DEPTH_PERFRAME_CAPTION_MICROBATCH_COT_VERSION}: camera={self.use_camera_cot}, control={self.camera_context_only}, loss={self.depth_loss_mode}, prompt={self.prompt_source}, cot={self.depth_cot_version}"
                )
        elif actual_frame_contract != (
            DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
            DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
            DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        ):
            raise RuntimeError(
                f"Frame supervision/microbatch/reduction cannot be modified independently without frame-wise training: actual={actual_frame_contract}"
            )
        if self.prompt_source == "none" and not self.depth_perframe_caption_microbatch:
            raise RuntimeError(
                "prompt_source=none allows only frame-by-frame Caption micro-batch."
            )
        if self.prompt_source == "record" and (
            (self.use_camera_cot and not self.camera_context_only)
            or self.depth_loss_mode
            not in {DEPTH_LOSS_MODE_NUMBER_ONLY, DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA}
        ):
            raise RuntimeError(
                "prompt_source=record allows only Depth-only + number_only/caption_pair_delta Stage1."
            )
        expected_record_cot_version = {
            DEPTH_LOSS_MODE_NUMBER_ONLY: (DEPTH_OBJECT_QUERY_COT_VERSION,),
            DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA: (
                DEPTH_CAPTION_PAIR_DELTA_COT_VERSION,
                DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION,
            ),
        }.get(self.depth_loss_mode)
        if (
            self.prompt_source == "record"
            and self.depth_cot_version not in expected_record_cot_version
        ):
            raise RuntimeError(
                f"prompt_source=record's cot_version and depth_loss_mode do not match: loss_mode={self.depth_loss_mode!r}, actual={self.depth_cot_version!r}, expected={expected_record_cot_version!r}"
            )

        kept = []
        total_loaded = 0
        source_row_counts = {}
        required_fields = ["n_images", "image_input"]
        if self.use_camera_cot:
            # Read camera supervision only from camera_text.
            required_fields.extend(["camera_text", "pose_path"])
        else:
            # Read each depth frame through the depth_paths declared in the JSONL.
            required_fields.extend(
                [
                    "source",
                    "scene_id",
                    "image_sizes",
                    "window_frame_ids",
                    "depth_paths",
                    "has_depth",
                    "cot_version",
                ]
            )
            if self.depth_perframe_caption_microbatch:
                required_fields.extend(
                    [
                        "stage1_task",
                        "source_window_idx",
                        "depth_single_frame_captions",
                    ]
                )
            elif (
                self.prompt_source == "record"
                and self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY
            ):
                required_fields.append("depth_object_query")
            elif self.prompt_source == "record":
                required_fields.extend(
                    [
                        "stage1_task",
                        "relation_eligible",
                        "depth_pair_query",
                        "depth_pair_relation_answer",
                        "depth_scene_cot",
                        "depth_pair_answer",
                    ]
                )
            else:
                required_fields.append("depth_scene_cot")
            if self.prompt_source == "record":
                required_fields.append(
                    "depth_object_answer"
                    if self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY
                    else "depth_pair_answer"
                )
        for dp in self.data_paths:
            if not os.path.isfile(dp):
                raise FileNotFoundError(f"data_path does not exist: {dp}")
            records = _load_json_or_jsonl(dp)
            total_loaded += len(records)
            source_row_counts[dp] = len(records)
            if self.depth_perframe_caption_microbatch:
                # The physical file contains 228,691 single-line entries; the Dataset must be strictly reorganized into 35,422 Window items first.
                # Weights for sampler, epoch, and optimizer are still calculated based on source Window.
                records = _group_depth_perframe_microbatch_rows(records, data_path=dp)
            for row_idx, record in enumerate(records):
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"Sample must be dict: file={dp}, row={row_idx}, type={type(record)}"
                    )
                for field in required_fields:
                    if field not in record:
                        raise RuntimeError(
                            f"Sample missing field {field!r}: file={dp}, row={row_idx}, keys={sorted(record.keys())}"
                        )
                if "idx" not in record and "group_id" not in record:
                    raise RuntimeError(
                        f"Sample missing idx/group_id: file={dp}, row={row_idx}, keys={sorted(record.keys())}"
                    )
                if self.use_camera_cot and not str(record["camera_text"]).strip():
                    raise RuntimeError(
                        f"camera_text cannot be empty: file={dp}, row={row_idx}, idx={record.get('idx')}"
                    )
                if self.use_camera_cot and self.camera_cot_version is not None:
                    if str(record.get("cot_version", "")) != self.camera_cot_version:
                        raise RuntimeError(
                            f"Camera Stage1 cot_version mismatch: file={dp}, row={row_idx}, idx={record.get('idx')}, actual={record.get('cot_version')!r}, expected={self.camera_cot_version!r}"
                        )
                if not self.use_camera_cot:
                    if self.depth_perframe_caption_microbatch:
                        target_field = "depth_single_frame_captions"
                    elif self.prompt_source == "record":
                        target_field = (
                            "depth_object_answer"
                            if self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY
                            else "depth_pair_answer"
                        )
                    else:
                        target_field = "depth_scene_cot"
                    target_value = record[target_field]
                    if self.depth_perframe_caption_microbatch:
                        if (
                            not isinstance(target_value, list)
                            or len(target_value) != int(record["n_images"])
                            or any(not str(item).strip() for item in target_value)
                        ):
                            raise RuntimeError(
                                f"depth_single_frame_captions must be a non-empty list of text with length n_images: file={dp}, row={row_idx}, idx={record.get('idx')}"
                            )
                    elif not str(target_value).strip():
                        raise RuntimeError(
                            f"{target_field} Cannot be empty: file={dp}, row={row_idx}, idx={record.get('idx')}"
                        )
                    if self.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
                        expected_stage1_task = (
                            "depth_caption_pair_delta_crossframe"
                            if self.depth_cot_version
                            == DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION
                            else "depth_caption_pair_delta"
                        )
                        if record["stage1_task"] != expected_stage1_task:
                            raise RuntimeError(
                                f"stage1_task mismatch: file={dp}, row={row_idx}, actual={record['stage1_task']!r}, expected={expected_stage1_task!r}"
                            )
                        relation_eligible = record["relation_eligible"]
                        if not isinstance(relation_eligible, bool):
                            raise RuntimeError(
                                f"relation_eligible must be bool: file={dp}, row={row_idx}, value={relation_eligible!r}"
                            )
                        relation_answer = str(record["depth_pair_relation_answer"])
                        caption = str(record["depth_scene_cot"])
                        expected_answer = (
                            relation_answer + "\n\n" + caption
                            if relation_eligible
                            else caption
                        )
                        if str(record["depth_pair_answer"]) != expected_answer:
                            raise RuntimeError(
                                f"depth_pair_answer does not meet the fixed concatenation contract: file={dp}, row={row_idx}, idx={record.get('idx')}"
                            )
                        if relation_eligible:
                            relation_matches = list(
                                DEPTH_RELATION_ANSWER_PATTERN.finditer(relation_answer)
                            )
                            if (
                                not relation_matches
                                or " ".join(
                                    match.group(0) for match in relation_matches
                                )
                                != relation_answer
                            ):
                                raise RuntimeError(
                                    f"eligible relation answer format is invalid: file={dp}, row={row_idx}, answer={relation_answer!r}"
                                )
                            if (
                                self.depth_cot_version
                                == DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION
                            ):
                                crossframe_eligible = record.get(
                                    "crossframe_relation_eligible"
                                )
                                if not isinstance(crossframe_eligible, bool):
                                    raise RuntimeError(
                                        f"crossframe_relation_eligible must be bool: file={dp}, row={row_idx}, value={crossframe_eligible!r}"
                                    )
                                expected_relation_count = (
                                    2 if crossframe_eligible else 1
                                )
                                if len(relation_matches) != expected_relation_count:
                                    raise RuntimeError(
                                        f"Number of cross-frame relations and eligible count do not match: file={dp}, row={row_idx}, matches={len(relation_matches)}, crossframe_eligible={crossframe_eligible}"
                                    )
                        elif relation_answer != "":
                            raise RuntimeError(
                                f"caption-only relation answer must be empty: file={dp}, row={row_idx}"
                            )
                if (
                    not isinstance(record["image_input"], list)
                    or not record["image_input"]
                ):
                    raise RuntimeError(
                        f"image_input must be a non-empty list: file={dp}, row={row_idx}, idx={record.get('idx')}"
                    )
                n_images = int(record["n_images"])
                if n_images != len(record["image_input"]):
                    raise RuntimeError(
                        f"n_images and image_input count do not match: file={dp}, row={row_idx}, n_images={n_images}, image_input={len(record['image_input'])}"
                    )
                record_image_sizes = _get_record_image_sizes(
                    record, row_idx=row_idx, required=False
                )
                if record_image_sizes is not None and len(record_image_sizes) != len(
                    record["image_input"]
                ):
                    raise RuntimeError(
                        f"image_sizes and image_input quantities do not match: file={dp}, row={row_idx}, idx={record.get('idx')}, image_sizes={len(record_image_sizes)}, image_input={len(record['image_input'])}"
                    )
                if self.use_camera_cot:
                    pose_path = _resolve_record_pose_path(
                        record,
                        context=f"Stage1 sample file={dp}, row={row_idx}, idx={record.get('idx')}",
                    )
                    if pose_path is None:
                        raise RuntimeError(
                            f"Camera Stage1 samples must have pose_path: file={dp}, row={row_idx}"
                        )
                    record["_pose_path"] = pose_path
                else:
                    context = f"Depth Stage1 sample file={dp}, row={row_idx}, idx={record.get('idx')}"
                    has_depth = _coerce_modality_flag(
                        record["has_depth"], field_name="has_depth", context=context
                    )
                    if has_depth != 1:
                        raise RuntimeError(
                            f"Depth Stage 1 samples must have has_depth=1: {context}"
                        )
                    # Validate description identity separately from the selected loss mode.
                    if (
                        self.depth_cot_version is not None
                        and str(record["cot_version"]) != self.depth_cot_version
                    ):
                        raise RuntimeError(
                            f"Depth Stage1 cot_version mismatch: {context}, loss_mode={self.depth_loss_mode!r}, actual={record['cot_version']!r}, expected={self.depth_cot_version!r}"
                        )
                    source = str(record["source"])
                    scene_id = str(record["scene_id"])
                    if (
                        source not in {"scannet", "scannetpp"}
                        or not scene_id
                        or "/" in scene_id
                    ):
                        raise RuntimeError(
                            f"Depth Stage1 source/scene_id illegal: {context}, {source!r}/{scene_id!r}"
                        )
                    depth_paths = record["depth_paths"]
                    frame_ids = record["window_frame_ids"]
                    if (
                        not isinstance(depth_paths, list)
                        or len(depth_paths) != n_images
                    ):
                        raise RuntimeError(
                            f"depth_paths must be a list of length n_images: {context}, depth_paths={type(depth_paths).__name__}, len={(len(depth_paths) if isinstance(depth_paths, list) else 'N/A')}"
                        )
                    if not isinstance(frame_ids, list) or len(frame_ids) != n_images:
                        raise RuntimeError(
                            f"window_frame_ids must be a list of length n_images: {context}, frame_ids={type(frame_ids).__name__}, len={(len(frame_ids) if isinstance(frame_ids, list) else 'N/A')}"
                        )
                    normalized_depth_paths = []
                    normalized_frame_ids = []
                    for view_index, (
                        raw_depth_path,
                        raw_frame_id,
                        raw_image_path,
                    ) in enumerate(zip(depth_paths, frame_ids, record["image_input"])):
                        image_path = str(raw_image_path).strip()
                        if not os.path.isabs(image_path):
                            raise RuntimeError(
                                f"Depth Stage1 image_input[{view_index}must be an absolute path: {context}, value={raw_image_path!r}"
                            )
                        image_path = os.path.abspath(os.path.expanduser(image_path))
                        depth_path = str(raw_depth_path).strip()
                        if not os.path.isabs(depth_path):
                            raise RuntimeError(
                                f"depth_paths[{view_index}must be an absolute path: {context}, value={raw_depth_path!r}"
                            )
                        depth_path = os.path.abspath(os.path.expanduser(depth_path))
                        if not os.path.isfile(depth_path):
                            raise FileNotFoundError(
                                f"depth_paths[{view_index}] Not found: {context}, path={depth_path}"
                            )
                        try:
                            frame_id = int(raw_frame_id)
                            rgb_frame_id = int(Path(image_path).stem)
                            depth_frame_id = int(Path(depth_path).stem)
                        except ValueError as exc:
                            raise RuntimeError(
                                f"RGB/Depth filenames and window_frame_ids must be integers: {context}, view={view_index}"
                            ) from exc
                        if frame_id != rgb_frame_id or frame_id != depth_frame_id:
                            raise RuntimeError(
                                f"JSONL RGB/Depth/frame id order is inconsistent: {context}, view={view_index}, frame_id={frame_id}, rgb={rgb_frame_id}, depth={depth_frame_id}"
                            )
                        image_parts = Path(image_path).parts
                        depth_parts = Path(depth_path).parts
                        expected_image_suffix = (
                            "images",
                            source,
                            scene_id,
                            f"{frame_id}.jpg",
                        )
                        expected_depth_suffix = (
                            "gt_depth",
                            source,
                            scene_id,
                            f"{frame_id}.npz",
                        )
                        if tuple(image_parts[-4:]) != expected_image_suffix:
                            raise RuntimeError(
                                f"RGB path and source/scene/frame identity are inconsistent: {context}, view={view_index}, path={image_path}"
                            )
                        if tuple(depth_parts[-4:]) != expected_depth_suffix:
                            raise RuntimeError(
                                f"Depth path and source/scene/frame identity are inconsistent: {context}, view={view_index}, path={depth_path}"
                            )
                        normalized_depth_paths.append(depth_path)
                        normalized_frame_ids.append(frame_id)
                    record["_depth_paths"] = normalized_depth_paths
                    record["_window_frame_ids"] = normalized_frame_ids
                    if self.camera_context_only:
                        # Merge JSONL, only append the original pose_path to rows where JOIN succeeds, without adding extra status fields.
                        # A single image naturally has no inter-frame pose; un-matched multi-image sets are also dropped according to Stage2 contract without inserting C_i.
                        pose_path = _resolve_record_pose_path(record, context=context)
                        has_pose = int(n_images > 1 and pose_path is not None)
                        record["_has_pose"] = has_pose
                        if has_pose:
                            record["_pose_path"] = pose_path
                record["_data_path"] = dp
                _infer_eval_bucket(record, row_idx=row_idx)
                kept.append(record)
                if max_samples is not None and len(kept) >= int(max_samples):
                    break
            if max_samples is not None and len(kept) >= int(max_samples):
                break
        if not kept:
            raise RuntimeError(f"No trainable samples: {self.data_path}")
        self._has_inline_depth_cot = any("depth_scene_cot" in r for r in kept[:10])
        if self._has_inline_depth_cot:
            depth_miss = sum(1 for r in kept if "depth_scene_cot" not in r)
            if depth_miss > 0:
                raise RuntimeError(
                    f"JSONL embedded depth_scene_cot is missing {depth_miss}/{len(kept)} Line."
                )
        if self.camera_pair_rows:
            # Keep each window as one training sample and expand its pairs inside the collator.
            for source in kept:
                views = int(source["n_images"])
                lines = str(source["camera_text"]).splitlines()
                if (
                    views < 2
                    or len(lines) != views
                    or lines[0] != "image 1 is the reference view."
                ):
                    raise RuntimeError(
                        f"Invalid camera pair source Window/COT: idx={source.get('idx')}, V={views}"
                    )
        self.records = kept
        self.source_row_counts = source_row_counts
        if self.use_camera_cot and self.camera_cot_version is None:
            versions = {
                str(record.get("cot_version", "")).strip() for record in self.records
            }
            if "" in versions or len(versions) != 1:
                raise RuntimeError(
                    f"When Camera Stage1 does not explicitly specify cot_version, the training data must provide a unique non-empty version: actual={sorted(versions)}"
                )
            self.camera_cot_version = next(iter(versions))
            print(
                "[dataset] camera_cot_version resolved from data: "
                f"{self.camera_cot_version}"
            )
        if not self.use_camera_cot and self.depth_cot_version is None:
            versions = {
                str(record.get("cot_version", "")).strip() for record in self.records
            }
            if "" in versions or len(versions) != 1:
                raise RuntimeError(
                    f"When Depth Stage1 does not explicitly specify cot_version, the training data must provide a unique non-empty version: actual={sorted(versions)}"
                )
            self.depth_cot_version = next(iter(versions))
            print(
                "[dataset] depth_cot_version resolved from data: "
                f"{self.depth_cot_version}"
            )
        print(f"[dataset] data_files={self.data_paths}")
        print(f"[dataset] loaded={total_loaded} kept={len(self.records)}")
        if self._has_inline_depth_cot:
            print(
                f"[dataset] depth_scene_cot: inline in JSONL, all {len(kept)} samples have it"
            )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _sample_id(record: dict) -> int:
        return int(record["idx"] if "idx" in record else record["group_id"])

    def _resolve_image_path(self, raw_path: str) -> str:
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(self.image_folder, raw_path)

    def __getitem__(self, index):
        record = dict(self.records[int(index)])
        image_paths = [self._resolve_image_path(p) for p in record["image_input"]]
        record_image_sizes = _get_record_image_sizes(record, required=False)
        images = []
        if self.load_images:
            for image_idx, path in enumerate(image_paths):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Cannot find input image: {path}")
                image = Image.open(path).convert("RGB")
                if record_image_sizes is not None:
                    expected_record_size = tuple(record_image_sizes[image_idx])
                    if tuple(image.size) != expected_record_size:
                        raise RuntimeError(
                            f"image_sizes metadata and actual image dimensions do not match: path={path}, actual PIL(W,H)={tuple(image.size)}, metadata={expected_record_size}"
                        )
                _assert_pil_image_size(
                    image, path, self.expected_image_size, "Stage1 Camera Interpreter"
                )
                images.append(image)
        n_images = int(record.get("n_images", len(image_paths)))
        if n_images != len(image_paths):
            raise RuntimeError(
                f"n_images and image_input count do not match: idx={record.get('idx')}, n_images={n_images}, image_input={len(image_paths)}"
            )
        camera_cot = str(record["camera_text"]).strip() if self.use_camera_cot else ""
        if self.depth_perframe_caption_microbatch:
            depth_cot = ""
            prompt = None
        elif (
            self.prompt_source == "record"
            and self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY
        ):
            depth_cot = str(record["depth_object_answer"]).strip()
            prompt = str(record["depth_object_query"]).strip()
        elif self.prompt_source == "record":
            # The only Assistant target of Pair Delta is the full stitching field; the two sub-fields generate mutually exclusive masks.
            depth_cot = str(record["depth_pair_answer"])
            prompt = str(record["depth_pair_query"]).strip()
        else:
            depth_cot = (
                str(record["depth_scene_cot"]).strip()
                if "depth_scene_cot" in record
                else ""
            )
            prompt = None
        if camera_cot and depth_cot:
            combined_cot = camera_cot + "\n" + depth_cot
        else:
            combined_cot = camera_cot or depth_cot
        result = {
            "idx": self._sample_id(record),
            "n_images": n_images,
            "global_scene_cot": combined_cot,
            "image_input": image_paths,
        }
        if self.depth_perframe_caption_microbatch:
            # The Collator expands this group of targets into Qwen's batch dimension; here, we continue to retain one Dataset item = one Window.
            result["depth_single_frame_captions"] = list(
                record["depth_single_frame_captions"]
            )
            result["source_window_idx"] = int(record["source_window_idx"])
        if prompt is not None:
            result["user_prompt"] = prompt
            if self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY:
                result["depth_object_query"] = prompt
            else:
                result["depth_pair_query"] = prompt
                result["relation_eligible"] = bool(record["relation_eligible"])
                result["depth_pair_relation_answer"] = str(
                    record["depth_pair_relation_answer"]
                )
                result["depth_scene_cot"] = str(record["depth_scene_cot"])
        if self.load_images:
            result["images"] = images
        if record_image_sizes is not None:
            result["image_sizes"] = record_image_sizes
        if self.use_camera_cot:
            result["pose_path"] = str(record["_pose_path"])
            result["has_pose"] = 1
            if "window_frame_ids" in record:
                result["window_frame_ids"] = [
                    int(item) for item in record["window_frame_ids"]
                ]
        elif self.camera_context_only:
            result["has_pose"] = int(record["_has_pose"])
            if int(record["_has_pose"]):
                result["pose_path"] = str(record["_pose_path"])
        if not self.use_camera_cot:
            result["source"] = str(record["source"])
            result["scene_id"] = str(record["scene_id"])
            result["depth_paths"] = list(record["_depth_paths"])
            result["window_frame_ids"] = list(record["_window_frame_ids"])
            result["has_depth"] = 1
        result["eval_bucket"] = _infer_eval_bucket(record)
        return result


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer(token, add_special_tokens=False, return_tensors=None)[
        "input_ids"
    ]
    if len(token_ids) != 1:
        raise RuntimeError(f"{token} Should be a single token, current ids={token_ids}")
    return int(token_ids[0])


def _resolve_depth_meter_token_contract(tokenizer) -> tuple[List[int], int]:
    """At startup, ensure that the tokenizer splits for 0–9, decimal points, and X.X satisfy the metric loss contract."""
    digit_token_ids = [_single_token_id(tokenizer, str(value)) for value in range(10)]
    if len(set(digit_token_ids)) != 10:
        raise RuntimeError(
            f"The token IDs from 0 to 9 must be distinct: digit_token_ids={digit_token_ids}"
        )
    decimal_token_id = _single_token_id(tokenizer, ".")
    probe_ids = tokenizer(
        "1.3",
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]
    expected_probe_ids = [
        digit_token_ids[1],
        decimal_token_id,
        digit_token_ids[3],
    ]
    if [int(token_id) for token_id in probe_ids] != expected_probe_ids:
        raise RuntimeError(f"actual={probe_ids}, expected={expected_probe_ids}")
    return digit_token_ids, decimal_token_id


def _find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:
    pat_len = int(pattern.numel())
    for start in range(0, int(row.numel()) - pat_len + 1):
        if torch.equal(row[start : start + pat_len], pattern):
            return start
    return -1


def _build_labels_after_assistant(
    input_ids: torch.Tensor,
    answer_start_pattern: torch.Tensor,
    pad_token_id: int,
    image_token_id: int,
    camera_token_id: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    for b in range(labels.shape[0]):
        start = _find_subsequence(labels[b], answer_start_pattern)
        if start < 0:
            raise RuntimeError(
                f"at batch index {b} Cannot find assistant start pattern in the sample, unable to construct labels. input_ids shape={input_ids.shape}"
            )
        labels[b, : start + answer_start_pattern.numel()] = -100
        labels[b, labels[b] == int(pad_token_id)] = -100
        labels[b, labels[b] == int(image_token_id)] = -100
        labels[b, labels[b] == int(camera_token_id)] = -100
    return labels


def _find_unique_assistant_caption_span(
    full_text: str,
    caption: str,
    *,
    context: str,
) -> tuple[int, int]:
    """In the full chat text, locate the unique character range of the assistant's Caption."""
    if not caption:
        raise RuntimeError(f"Caption cannot be empty: {context}")
    starts = []
    search_from = 0
    while True:
        start = full_text.find(caption, search_from)
        if start < 0:
            break
        starts.append(start)
        search_from = start + 1
    if len(starts) != 1:
        raise RuntimeError(
            f"Caption must appear exactly once in the full chat text: {context}, occurrences={len(starts)}"
        )
    start = starts[0]
    assistant_marker = "<|im_start|>assistant"
    marker_start = full_text.rfind(assistant_marker, 0, start + 1)
    if marker_start < 0:
        raise RuntimeError(
            f"Cannot find assistant start marker before caption: {context}, caption_start={start}"
        )
    return start, start + len(caption)


def _build_number_only_labels(
    *,
    full_texts: Sequence[str],
    captions: Sequence[str],
    input_ids: torch.Tensor,
    offset_mapping: torch.Tensor,
    tokenizer,
    contexts: Sequence[str],
    allow_empty: bool = False,
) -> tuple[torch.Tensor, List[int]]:
    """Only supervise the numeric token in assistant caption `at X.X meters`."""
    if input_ids.dim() != 2 or offset_mapping.dim() != 3:
        raise RuntimeError(
            f"number_only requires input_ids=[B,S], offset_mapping=[B,S,2]: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    if (
        tuple(offset_mapping.shape[:2]) != tuple(input_ids.shape)
        or int(offset_mapping.shape[2]) != 2
    ):
        raise RuntimeError(
            f"offset_mapping and input_ids shape are inconsistent: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    batch_size = int(input_ids.shape[0])
    if not (len(full_texts) == len(captions) == len(contexts) == batch_size):
        raise RuntimeError(
            f"number_only text, caption, context, and batch size inconsistent: texts={len(full_texts)}, captions={len(captions)}, contexts={len(contexts)}, batch={batch_size}"
        )

    labels = torch.full_like(input_ids, -100)
    span_counts: List[int] = []
    for batch_index in range(batch_size):
        full_text = str(full_texts[batch_index])
        caption = str(captions[batch_index])
        context = str(contexts[batch_index])
        caption_start, caption_end = _find_unique_assistant_caption_span(
            full_text,
            caption,
            context=context,
        )
        matches = list(DEPTH_NUMBER_PATTERN.finditer(caption))
        if not matches:
            if not allow_empty:
                raise RuntimeError(
                    f"number_only samples have no depth numbers: {context}"
                )
            # When a single frame explicitly states "no significant objects," no numerical supervision is available; this row remains all -100.
            span_counts.append(0)
            continue
        numeric_spans = [
            (
                caption_start + match.start(1),
                caption_start + match.end(1),
                match.group(1),
            )
            for match in matches
        ]
        if any(
            span_start < caption_start or span_end > caption_end
            for span_start, span_end, _ in numeric_spans
        ):
            raise RuntimeError(f"numeric span crossed assistant Caption: {context}")

        tokens_by_span: List[List[int]] = [[] for _ in numeric_spans]
        covered_chars_by_span = [set() for _ in numeric_spans]
        for token_index, raw_offsets in enumerate(offset_mapping[batch_index]):
            token_start, token_end = [int(value) for value in raw_offsets.tolist()]
            # Fast tokenizer uses (0,0) to represent the special token with no original character interval; they are never included in the loss.
            if token_start == 0 and token_end == 0:
                continue
            if (
                token_start < 0
                or token_end <= token_start
                or token_end > len(full_text)
            ):
                raise RuntimeError(
                    f"Invalid token offset: {context}, token={token_index}, offset=({token_start},{token_end}), text_len={len(full_text)}"
                )
            overlaps = [
                span_index
                for span_index, (span_start, span_end, _) in enumerate(numeric_spans)
                if token_start < span_end and token_end > span_start
            ]
            if len(overlaps) > 1:
                raise RuntimeError(
                    f"A token crosses two depth values: {context}, token={token_index}, offset=({token_start},{token_end}), spans={overlaps}"
                )
            if not overlaps:
                continue
            span_index = overlaps[0]
            span_start, span_end, _ = numeric_spans[span_index]
            # The tokenizer may combine a leading space before a number with the first digit to form a single token; it allows pure whitespace to overflow.
            # But if the token simultaneously covers non-whitespace characters such as at/meters/ punctuation, it will directly report an error.
            outside_text = (
                full_text[token_start : min(token_end, span_start)]
                + full_text[max(token_start, span_end) : token_end]
            )
            if outside_text.strip():
                raise RuntimeError(
                    f"Digital tokens also cover non-digit text: {context}, token={token_index}, offset=({token_start},{token_end}), outside={outside_text!r}"
                )
            labels[batch_index, token_index] = input_ids[batch_index, token_index]
            tokens_by_span[span_index].append(token_index)
            covered_start = max(token_start, span_start)
            covered_end = min(token_end, span_end)
            covered_chars_by_span[span_index].update(range(covered_start, covered_end))

        for span_index, (span_start, span_end, number_text) in enumerate(numeric_spans):
            token_indices = tokens_by_span[span_index]
            if not token_indices:
                raise RuntimeError(
                    f"Depth values have not been mapped to any token: {context}, number={number_text!r}, span=({span_start},{span_end})"
                )
            expected_chars = set(range(span_start, span_end))
            if covered_chars_by_span[span_index] != expected_chars:
                raise RuntimeError(
                    f"Digital characters were not fully covered by tokens: {context}, number={number_text!r}, covered={sorted(covered_chars_by_span[span_index])}, expected={sorted(expected_chars)}"
                )
            decoded = tokenizer.decode(
                input_ids[batch_index, token_indices].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if number_text not in decoded and "".join(decoded.split()) != number_text:
                raise RuntimeError(
                    f"Supervised token decode cannot backtrack to original digits: {context}, number={number_text!r}, decoded={decoded!r}, tokens={token_indices}"
                )
        span_counts.append(len(numeric_spans))

    return labels, span_counts


def _build_caption_pair_delta_labels(
    *,
    full_texts: Sequence[str],
    pair_answers: Sequence[str],
    relation_answers: Sequence[str],
    scene_captions: Sequence[str],
    relation_eligible: Sequence[bool],
    input_ids: torch.Tensor,
    offset_mapping: torch.Tensor,
    tokenizer,
    contexts: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """Construct mutually exclusive relation/caption digital label masks from the same Pair Answer Forward."""
    batch_size = int(input_ids.shape[0])
    values = (
        full_texts,
        pair_answers,
        relation_answers,
        scene_captions,
        relation_eligible,
        contexts,
    )
    if any(len(value) != batch_size for value in values):
        raise RuntimeError(
            f"caption_pair_delta text field and batch size are inconsistent: batch={batch_size}, lengths={[len(value) for value in values]}"
        )
    caption_labels, caption_span_counts = _build_number_only_labels(
        full_texts=full_texts,
        captions=scene_captions,
        input_ids=input_ids,
        offset_mapping=offset_mapping,
        tokenizer=tokenizer,
        contexts=contexts,
    )
    relation_labels = torch.full_like(input_ids, -100)
    relation_span_counts: List[int] = []
    for batch_index in range(batch_size):
        context = str(contexts[batch_index])
        relation_answer = str(relation_answers[batch_index])
        scene_caption = str(scene_captions[batch_index])
        pair_answer = str(pair_answers[batch_index])
        eligible = bool(relation_eligible[batch_index])
        expected_answer = (
            relation_answer + "\n\n" + scene_caption if eligible else scene_caption
        )
        if pair_answer != expected_answer:
            raise RuntimeError(
                f"depth_pair_answer does not meet the fixed concatenation contract: {context}, eligible={eligible}"
            )
        if not eligible:
            if relation_answer:
                raise RuntimeError(
                    f"caption-only relation answer must be empty: {context}"
                )
            relation_span_counts.append(0)
            continue

        relation_matches = list(DEPTH_RELATION_ANSWER_PATTERN.finditer(relation_answer))
        if (
            not relation_matches
            or " ".join(match.group(0) for match in relation_matches) != relation_answer
        ):
            raise RuntimeError(
                f"Invalid format for eligible relation answer: {context}, answer={relation_answer!r}"
            )
        relation_start, _ = _find_unique_assistant_caption_span(
            str(full_texts[batch_index]),
            relation_answer,
            context=context,
        )
        for relation_match in relation_matches:
            span_start = relation_start + relation_match.start(1)
            span_end = relation_start + relation_match.end(1)
            number_text = relation_match.group(1)
            token_indices = []
            covered_chars = set()
            for token_index, raw_offsets in enumerate(offset_mapping[batch_index]):
                token_start, token_end = [int(value) for value in raw_offsets.tolist()]
                if token_start == 0 and token_end == 0:
                    continue
                if (
                    token_start < 0
                    or token_end <= token_start
                    or token_end > len(full_texts[batch_index])
                ):
                    raise RuntimeError(
                        f"Invalid relation token offset: {context}, token={token_index}, offset=({token_start},{token_end})"
                    )
                if token_start >= span_end or token_end <= span_start:
                    continue
                outside_text = (
                    str(full_texts[batch_index])[
                        token_start : min(token_end, span_start)
                    ]
                    + str(full_texts[batch_index])[
                        max(token_start, span_end) : token_end
                    ]
                )
                if outside_text.strip():
                    raise RuntimeError(
                        f"relation numeric token simultaneously covers non-numeric text: {context}, token={token_index}, outside={outside_text!r}"
                    )
                relation_labels[batch_index, token_index] = input_ids[
                    batch_index, token_index
                ]
                token_indices.append(token_index)
                covered_chars.update(
                    range(max(token_start, span_start), min(token_end, span_end))
                )
            if covered_chars != set(range(span_start, span_end)) or not token_indices:
                raise RuntimeError(
                    f"The relation number is not fully covered by the token: {context}, number={number_text!r}, tokens={token_indices}"
                )
            decoded = tokenizer.decode(
                input_ids[batch_index, token_indices].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if number_text not in decoded and "".join(decoded.split()) != number_text:
                raise RuntimeError(
                    f"relation token decode cannot backtrack to original number: {context}, number={number_text!r}, decoded={decoded!r}"
                )
        relation_span_counts.append(len(relation_matches))

    if bool(((relation_labels != -100) & (caption_labels != -100)).any().item()):
        raise RuntimeError(
            "caption_pair_delta relation/caption label mask overlap occurs."
        )
    return relation_labels, caption_labels, relation_span_counts, caption_span_counts


def _build_depth_meter_targets(
    *,
    full_text: str,
    caption: str,
    input_ids: torch.Tensor,
    offset_mapping: torch.Tensor,
    digit_token_ids: Sequence[int],
    decimal_token_id: int,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map strict X.X depth to integer bit, fractional bit label position, and metric GT."""
    if tuple(input_ids.shape[:1]) != (1,) or tuple(offset_mapping.shape[:1]) != (1,):
        raise RuntimeError(
            f"meter loss is currently bound to Stage1 batch=1: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    if input_ids.dim() != 2 or offset_mapping.dim() != 3:
        raise RuntimeError(
            f"meter loss requires input_ids=[1,S], offset_mapping=[1,S,2]: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    if tuple(offset_mapping.shape[1:]) != (int(input_ids.shape[1]), 2):
        raise RuntimeError(
            f"meter loss offset_mapping and input_ids shape inconsistent: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    if (
        len(digit_token_ids) != 10
        or len(set(int(value) for value in digit_token_ids)) != 10
    ):
        raise RuntimeError(
            f"meter loss digit token contract illegal: ids={list(digit_token_ids)}"
        )

    caption_start, caption_end = _find_unique_assistant_caption_span(
        full_text,
        caption,
        context=context,
    )
    fields = list(DEPTH_DISTANCE_FIELD_PATTERN.finditer(caption))
    if not fields:
        raise RuntimeError(
            f"meter loss sample lacks the `at X.X meters` depth field: {context}"
        )

    integer_positions: List[int] = []
    fractional_positions: List[int] = []
    targets_m: List[float] = []
    row_offsets = offset_mapping[0]
    row_input_ids = input_ids[0]
    for field in fields:
        number_text = field.group(1)
        if DEPTH_METER_NUMBER_PATTERN.fullmatch(number_text) is None:
            # Unsupported 10.5, integers, multi-decimal numbers, or negative numbers must be directly exposed and cannot be silently overlooked by regex.
            raise RuntimeError(
                f"meter loss supports only strict [0-9].[0-9]: {context}, actual={number_text!r}"
            )
        span_start = caption_start + field.start(1)
        span_end = caption_start + field.end(1)
        if span_start < caption_start or span_end > caption_end:
            raise RuntimeError(
                f"meter loss number span exceeds {context}, number={number_text!r}"
            )

        token_indices: List[int] = []
        covered_chars = set()
        for token_index, raw_offsets in enumerate(row_offsets):
            token_start, token_end = [int(value) for value in raw_offsets.tolist()]
            if token_start == 0 and token_end == 0:
                continue
            if token_start < span_end and token_end > span_start:
                outside_text = (
                    full_text[token_start : min(token_end, span_start)]
                    + full_text[max(token_start, span_end) : token_end]
                )
                if outside_text.strip():
                    raise RuntimeError(
                        f"meter loss token simultaneously covers non-whitespace text outside numbers: {context}, number={number_text!r}, token={token_index}, outside={outside_text!r}"
                    )
                token_indices.append(token_index)
                covered_chars.update(
                    range(max(token_start, span_start), min(token_end, span_end))
                )
        if covered_chars != set(range(span_start, span_end)):
            raise RuntimeError(
                f"Meter loss numeric characters were not fully tokenized: {context}, number={number_text!r}, tokens={token_indices}"
            )
        if len(token_indices) != 3:
            raise RuntimeError(
                f"meter loss each X.X must strictly correspond to 3 tokens: {context}, number={number_text!r}, tokens={token_indices}"
            )

        expected_ids = [
            int(digit_token_ids[int(number_text[0])]),
            int(decimal_token_id),
            int(digit_token_ids[int(number_text[2])]),
        ]
        actual_ids = [int(row_input_ids[position].item()) for position in token_indices]
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"meter loss X.X token split does not match digit + dot + digit: {context}, number={number_text!r}, actual={actual_ids}, expected={expected_ids}"
            )
        integer_positions.append(token_indices[0])
        fractional_positions.append(token_indices[2])
        targets_m.append(float(int(number_text[0])) + 0.1 * float(int(number_text[2])))

    return (
        torch.tensor(integer_positions, dtype=torch.long),
        torch.tensor(fractional_positions, dtype=torch.long),
        torch.tensor(targets_m, dtype=torch.float32),
    )


def _build_motion_content_labels(
    *,
    full_texts: Sequence[str],
    captions: Sequence[str],
    input_ids: torch.Tensor,
    offset_mapping: torch.Tensor,
    tokenizer,
    contexts: Sequence[str],
    expected_frame_numbers: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, List[int]]:
    """Only supervise distance, direction, angle, and static state tokens in Camera Caption."""
    if input_ids.dim() != 2 or offset_mapping.dim() != 3:
        raise RuntimeError(
            f"motion_content requires input_ids=[B,S], offset_mapping=[B,S,2]: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    if (
        tuple(offset_mapping.shape[:2]) != tuple(input_ids.shape)
        or int(offset_mapping.shape[2]) != 2
    ):
        raise RuntimeError(
            f"motion_content offset_mapping and input_ids shape mismatch: input_ids={tuple(input_ids.shape)}, offsets={tuple(offset_mapping.shape)}"
        )
    batch_size = int(input_ids.shape[0])
    if not (len(full_texts) == len(captions) == len(contexts) == batch_size):
        raise RuntimeError(
            f"motion_content text, caption, context, and batch size are inconsistent: texts={len(full_texts)}, captions={len(captions)}, contexts={len(contexts)}, batch={batch_size}"
        )
    if expected_frame_numbers is not None and len(expected_frame_numbers) != batch_size:
        raise RuntimeError(
            "Camera pair answer frame number count and batch size do not match."
        )

    labels = torch.full_like(input_ids, -100)
    span_counts: List[int] = []
    supervised_groups = (
        "stationary",
        "distance1",
        "translation1",
        "distance2",
        "translation2",
        "unchanged",
        "rotation",
        "angle",
    )
    for batch_index in range(batch_size):
        full_text = str(full_texts[batch_index])
        caption = str(captions[batch_index])
        context = str(contexts[batch_index])
        caption_start, caption_end = _find_unique_assistant_caption_span(
            full_text,
            caption,
            context=context,
        )
        lines = caption.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\n") not in {
            "image 1 is the reference view.",
            MINDCUBE_DA3_REFERENCE,
        }:
            raise RuntimeError(
                f"motion_content Caption reference line does not conform to fixed format: {context}"
            )

        content_spans: List[tuple[int, int, str]] = []
        relative_line_start = len(lines[0])
        if expected_frame_numbers is not None and len(lines) != 2:
            raise RuntimeError(
                f"Camera pairing answers must exactly one target motion: {context}"
            )
        for local_frame, raw_line in enumerate(lines[1:], start=2):
            expected_frame = (
                int(expected_frame_numbers[batch_index])
                if expected_frame_numbers is not None
                else local_frame
            )
            line = raw_line.rstrip("\n")
            # Accept the two supported camera-description templates.
            pattern = (
                MINDCUBE_DA3_LINE_PATTERN
                if lines[0].rstrip("\n") == MINDCUBE_DA3_REFERENCE
                else CAMERA_MOTION_LINE_PATTERN
            )
            match = pattern.fullmatch(line)
            if match is None:
                raise RuntimeError(
                    f"motion_content Caption line does not conform to fixed syntax: {context}, line={line!r}"
                )
            if int(match.group("frame")) != expected_frame:
                raise RuntimeError(
                    f"motion_content frame index is discontinuous: {context}, actual={match.group('frame')}, expected={expected_frame}"
                )
            for group_name in supervised_groups:
                group_text = match.groupdict().get(group_name)
                if group_text is None:
                    continue
                group_start, group_end = match.span(group_name)
                content_spans.append(
                    (
                        caption_start + relative_line_start + group_start,
                        caption_start + relative_line_start + group_end,
                        group_text,
                    )
                )
            relative_line_start += len(raw_line)
        if not content_spans:
            raise RuntimeError(
                f"motion_content Caption no motion content slot: {context}"
            )
        if relative_line_start != len(caption):
            raise RuntimeError(
                f"motion_content Caption line offset does not cover the entire text: {context}, covered={relative_line_start}, caption_len={len(caption)}"
            )
        if any(
            start < caption_start or end > caption_end
            for start, end, _ in content_spans
        ):
            raise RuntimeError(
                f"motion_content span crosses assistant Caption: {context}"
            )

        tokens_by_span: List[List[int]] = [[] for _ in content_spans]
        covered_chars_by_span = [set() for _ in content_spans]
        for token_index, raw_offsets in enumerate(offset_mapping[batch_index]):
            token_start, token_end = [int(value) for value in raw_offsets.tolist()]
            if token_start == 0 and token_end == 0:
                continue
            if (
                token_start < 0
                or token_end <= token_start
                or token_end > len(full_text)
            ):
                raise RuntimeError(
                    f"motion_content token offset illegal: {context}, token={token_index}, offset=({token_start},{token_end}), text_len={len(full_text)}"
                )
            overlaps = [
                span_index
                for span_index, (span_start, span_end, _) in enumerate(content_spans)
                if token_start < span_end and token_end > span_start
            ]
            if len(overlaps) > 1:
                raise RuntimeError(
                    f"A token spans multiple motion content spans: {context}, token={token_index}, spans={overlaps}"
                )
            if not overlaps:
                continue
            span_index = overlaps[0]
            span_start, span_end, _ = content_spans[span_index]
            # Templates and punctuation must all be labeled as label=-100; if content and non-blank templates are combined, an error will be reported directly.
            outside_text = (
                full_text[token_start : min(token_end, span_start)]
                + full_text[max(token_start, span_end) : token_end]
            )
            if outside_text.strip():
                raise RuntimeError(
                    f"motion content token simultaneously covers template text: {context}, token={token_index}, offset=({token_start},{token_end}), outside={outside_text!r}"
                )
            labels[batch_index, token_index] = input_ids[batch_index, token_index]
            tokens_by_span[span_index].append(token_index)
            covered_chars_by_span[span_index].update(
                range(max(token_start, span_start), min(token_end, span_end))
            )

        for span_index, (span_start, span_end, span_text) in enumerate(content_spans):
            token_indices = tokens_by_span[span_index]
            if not token_indices:
                raise RuntimeError(
                    f"motion content span not mapped to token: {context}, span={span_text!r}"
                )
            if covered_chars_by_span[span_index] != set(range(span_start, span_end)):
                raise RuntimeError(
                    f"Motion content characters were not fully tokenized: {context}, span={span_text!r}"
                )
            decoded = tokenizer.decode(
                input_ids[batch_index, token_indices].tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if span_text not in decoded and "".join(decoded.split()) != "".join(
                span_text.split()
            ):
                raise RuntimeError(
                    f"motion content token decode cannot backtrack to original content: {context}, span={span_text!r}, decoded={decoded!r}, tokens={token_indices}"
                )
        span_counts.append(len(content_spans))

    return labels, span_counts


def _load_camera_pose_from_path(
    path: str,
    expected_views: int,
    *,
    expected_image_paths: Optional[Sequence[str]] = None,
    expected_image_sizes: Optional[Sequence[Sequence[int]]] = None,
    expected_frame_ids: Optional[Sequence[int]] = None,
    expected_original_image_paths: Optional[Sequence[str]] = None,
    expected_geometry_image_paths: Optional[Sequence[str]] = None,
    unmarked_rgb_migration: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load standard camera.npz from complete pose_path and strictly verify coordinates, shape, frame count, and available metadata order."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find camera pose: {path}")
    with np.load(path) as data:
        required = {"extrinsics", "intrinsics", "image_sizes"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"camera pose missing fields {missing}: {path}")
        payload = {key: data[key] for key in data.files}
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        image_sizes = np.asarray(data["image_sizes"], dtype=np.float32)
    # Require camera arrays shaped (V, ...); normalize batch dimensions during data preparation.
    if tuple(extrinsics.shape) != (expected_views, 3, 4):
        raise RuntimeError(
            f"extrinsics shape does not match the number of images: shape={extrinsics.shape}, expected={(expected_views, 3, 4)}"
        )
    if tuple(intrinsics.shape) != (expected_views, 3, 3):
        raise RuntimeError(
            f"intrinsics shape does not match the number of images: shape={intrinsics.shape}, expected={(expected_views, 3, 3)}"
        )
    if (
        image_sizes.ndim != 2
        or int(image_sizes.shape[0]) != expected_views
        or int(image_sizes.shape[1]) != 2
    ):
        raise RuntimeError(
            f"image_sizes shape is invalid: shape={image_sizes.shape}, expected={(expected_views, 2)}"
        )
    if (
        not np.isfinite(extrinsics).all()
        or not np.isfinite(intrinsics).all()
        or not np.isfinite(image_sizes).all()
    ):
        raise RuntimeError(f"camera.npz contains NaN/Inf: {path}")
    if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
        raise RuntimeError(f"camera intrinsics must have fx/fy > 0: path={path}")
    if np.any(image_sizes <= 0.0):
        raise RuntimeError(
            f"camera image_sizes must all be positive numbers: path={path}, values={image_sizes.tolist()}"
        )
    dets = np.linalg.det(extrinsics[:, :3, :3].astype(np.float64))
    bad_det = np.where(
        (~np.isfinite(dets)) | (np.abs(dets - 1.0) > CAMERA_ROT_DET_TOL)
    )[0]
    if bad_det.size:
        first = int(bad_det[0])
        raise RuntimeError(
            f"camera rotation det(R) is not close to +1: path={path}, view={first}, det={float(dets[first]):.6f}"
        )
    identity = np.eye(4, dtype=np.float32)[:3, :4]
    frame0_error = float(np.max(np.abs(extrinsics[0] - identity)))
    if frame0_error > CAMERA_FRAME0_ATOL:
        raise RuntimeError(
            f"camera frame 0 not canonicalized to [I|0]: path={path}, max_error={frame0_error:.6g}"
        )
    if expected_image_sizes is not None:
        expected_sizes = np.asarray(expected_image_sizes, dtype=np.float32)
        if tuple(expected_sizes.shape) != (expected_views, 2) or not np.array_equal(
            image_sizes, expected_sizes
        ):
            raise RuntimeError(
                f"JSONL image_sizes and camera.npz are inconsistent: path={path}, stored={image_sizes.tolist()}, expected={expected_sizes.tolist()}"
            )
    _validate_npz_image_order(
        payload,
        expected_image_paths=expected_image_paths,
        expected_frame_ids=expected_frame_ids,
        path=path,
        expected_original_image_paths=expected_original_image_paths,
        expected_geometry_image_paths=expected_geometry_image_paths,
        unmarked_rgb_migration=unmarked_rgb_migration,
    )
    return {
        "extrinsics": torch.from_numpy(np.ascontiguousarray(extrinsics)).unsqueeze(0),
        "intrinsics": torch.from_numpy(np.ascontiguousarray(intrinsics)).unsqueeze(0),
        "image_sizes": torch.from_numpy(np.ascontiguousarray(image_sizes)),
    }


def _load_camera_pose(
    feature_root: str, idx: int, expected_views: int
) -> Dict[str, torch.Tensor]:
    """Load camera geometry from a feature directory and sample index."""
    path = os.path.join(
        os.path.abspath(os.path.expanduser(feature_root)), str(int(idx)), "camera.npz"
    )
    return _load_camera_pose_from_path(path, expected_views)


def _assert_depth_value_range(
    depth: np.ndarray, *, idx: int, path: str, check_distribution: bool = True
) -> None:
    """Check for finite values; for unapproved data, also verify complete percentiles and metric ranges."""
    if not np.isfinite(depth).all():
        raise RuntimeError(f"depth contains NaN/Inf: idx={idx}, path={path}")
    if not np.any(depth > 0.0):
        raise RuntimeError(f"depth has no valid positive value: idx={idx}, path={path}")
    if not check_distribution:
        # Geometry published at disk write is fully verified numerically; training reads only retain limited values/positive checks per iteration.
        # No longer sort the same Depth repeatedly to calculate median and P25. Other sources still follow the original complete gate.
        return
    valid = depth[depth > 0.0]
    median_d = float(np.median(valid))
    lower_quartile_d = float(np.percentile(valid, 25.0))
    # Use joint depth quantiles to detect unit errors while allowing distant scenes.
    if (
        median_d > DEPTH_MAX_REASONABLE_MEDIAN_M
        and lower_quartile_d > DEPTH_MAX_REASONABLE_LOWER_QUARTILE_M
    ):
        raise RuntimeError(
            f"median depth {median_d:.1f}m and P25={lower_quartile_d:.1f}m are all abnormally large, possibly the unit is mm without dividing by 1000: idx={idx}, path={path}"
        )
    if median_d < DEPTH_MIN_REASONABLE_MEDIAN_M:
        raise RuntimeError(
            f"median depth {median_d:.4f}m is abnormally small, possibly due to incorrect value range: idx={idx}, path={path}"
        )


def _assert_depth_rgb_aspect_ratio(
    depth: np.ndarray, image_sizes, *, idx: int, path: str
) -> None:
    """Check that the depth and RGB aspect ratios are consistent to prevent resizing depth with different FOV/crop to fit the RGB token grid."""
    if image_sizes is None:
        return
    if isinstance(image_sizes, torch.Tensor):
        sizes = image_sizes.detach().cpu().numpy()
    else:
        sizes = np.asarray(image_sizes)
    if (
        sizes.ndim != 2
        or int(sizes.shape[0]) != int(depth.shape[0])
        or int(sizes.shape[1]) != 2
    ):
        raise RuntimeError(
            f"image_sizes shape and depth view are inconsistent, cannot verify depth/RGB alignment: idx={idx}, image_sizes_shape={sizes.shape}, depth_shape={depth.shape}, path={path}"
        )
    sizes = sizes.astype(np.float32)
    rgb_w = sizes[:, 0]
    rgb_h = sizes[:, 1]
    if np.any(rgb_w <= 0) or np.any(rgb_h <= 0):
        raise RuntimeError(
            f"image_sizes contains invalid width and height: idx={idx}, image_sizes={sizes.tolist()}, path={path}"
        )
    depth_h = float(depth.shape[1])
    depth_w = float(depth.shape[2])
    depth_ratio = depth_w / depth_h
    rgb_ratio = rgb_w / rgb_h
    # Only checks aspect ratio; no requirement for identical resolution; ScanNet++ allows 256x192 depth to be aligned to 640x480 RGB.
    rel_err = np.abs(rgb_ratio - depth_ratio) / np.maximum(rgb_ratio, depth_ratio)
    bad = np.where(rel_err > DEPTH_RGB_ASPECT_RATIO_REL_TOL)[0]
    if bad.size > 0:
        first = int(bad[0])
        raise RuntimeError(
            f"Inconsistent aspect ratio between depth and RGB, resizing causes spatial misalignment: idx={idx}, view={first}, depth_hw=({int(depth_h)},{int(depth_w)}), rgb_wh=({int(rgb_w[first])},{int(rgb_h[first])}), depth_ratio={depth_ratio:.4f}, rgb_ratio={float(rgb_ratio[first]):.4f}, rel_err={float(rel_err[first]):.4f}, path={path}"
        )


def _load_depth_maps_from_path(
    path: str,
    expected_views: int,
    image_sizes=None,
    *,
    expected_image_paths: Optional[Sequence[str]] = None,
    expected_frame_ids: Optional[Sequence[int]] = None,
    expected_original_image_paths: Optional[Sequence[str]] = None,
    expected_geometry_image_paths: Optional[Sequence[str]] = None,
    check_distribution: bool = True,  # Depth that has been released and fully accepted can disable duplicate quantiles; unknown data remains True.
) -> torch.Tensor:
    """Load complete depth_path (V,H,W) metric depth and verify frame count, value range, and available metadata order.

    Compatible with two schemas:
      - eval: key="depth_maps", dtype=float32 (already in meters)
      - training: key="depth_maps", dtype=uint16 (mm, divide by 1000)
    image_sizes optional; checks aspect ratio of depth/RGB after input, blocks new data sources with different FOV/crop in advance.
    check_distribution=False only validates QA geometry that has been fully accepted at release; finite, positive, shape, and image order are still checked sequentially.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find depth map: {path}")
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
        if "depth_maps" in data:
            raw = data["depth_maps"]
        elif "depth" in data:
            raw = data["depth"]
        else:
            raise RuntimeError(
                f"depth.npz is missing the depth_maps/depth field: {path}, keys={list(data.keys())}"
            )
        if raw.dtype == np.uint16:
            depth = raw.astype(np.float32) / 1000.0
        else:
            depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim == 4 and int(depth.shape[0]) == 1:
        depth = depth[0]
    if depth.ndim != 3 or int(depth.shape[0]) != expected_views:
        raise RuntimeError(
            f"depth shape invalid: shape={depth.shape}, expected=({expected_views}, H, W), path={path}"
        )
    _assert_depth_value_range(
        depth, idx=-1, path=path, check_distribution=check_distribution
    )
    _assert_depth_rgb_aspect_ratio(depth, image_sizes, idx=-1, path=path)
    _validate_npz_image_order(
        payload,
        expected_image_paths=expected_image_paths,
        expected_frame_ids=expected_frame_ids,
        path=path,
        expected_original_image_paths=expected_original_image_paths,
        expected_geometry_image_paths=expected_geometry_image_paths,
    )
    return torch.from_numpy(np.ascontiguousarray(depth))


def _load_depth_maps(
    feature_root: str, idx: int, expected_views: int, image_sizes=None
) -> torch.Tensor:
    """Load a depth sample from a feature directory and sample index."""
    path = os.path.join(
        os.path.abspath(os.path.expanduser(feature_root)), str(int(idx)), "depth.npz"
    )
    return _load_depth_maps_from_path(path, expected_views, image_sizes=image_sizes)


def _load_depth_maps_from_paths(
    depth_paths: Sequence[str],
    expected_views: int,
    *,
    image_sizes: Sequence[Sequence[int]],
    expected_image_paths: Sequence[str],
    expected_frame_ids: Sequence[int],
    idx: int,
) -> torch.Tensor:
    """Strictly read ordered frame-by-frame uint16 millimeter depth, and directly stack into [V,H,W] metric float32."""

    if not isinstance(depth_paths, (list, tuple)) or len(depth_paths) != expected_views:
        raise RuntimeError(
            f"The number of depth_paths must equal n_images: idx={idx}, depth_paths={(len(depth_paths) if isinstance(depth_paths, (list, tuple)) else type(depth_paths))}, expected_views={expected_views}"
        )
    if (
        len(expected_image_paths) != expected_views
        or len(expected_frame_ids) != expected_views
    ):
        raise RuntimeError(
            f"Number of RGB/frame IDs must equal n_images: idx={idx}, images={len(expected_image_paths)}, frame_ids={len(expected_frame_ids)}, expected_views={expected_views}"
        )
    if len(image_sizes) != expected_views:
        raise RuntimeError(
            f"image_sizes count must equal n_images: idx={idx}, image_sizes={len(image_sizes)}, expected_views={expected_views}"
        )

    frames = []
    first_shape = None
    for view_index, (raw_depth_path, raw_image_path, expected_frame_id) in enumerate(
        zip(depth_paths, expected_image_paths, expected_frame_ids)
    ):
        depth_path = os.path.abspath(os.path.expanduser(str(raw_depth_path)))
        image_path = os.path.abspath(os.path.expanduser(str(raw_image_path)))
        if not os.path.isabs(str(raw_depth_path)) or not os.path.isfile(depth_path):
            raise FileNotFoundError(
                f"The depth_paths must be existing absolute files: idx={idx}, view={view_index}, path={raw_depth_path}"
            )
        try:
            depth_frame_id = int(Path(depth_path).stem)
            image_frame_id = int(Path(image_path).stem)
            expected_frame_id = int(expected_frame_id)
        except ValueError as exc:
            raise RuntimeError(
                f"RGB/Depth filenames must be integer frame ids: idx={idx}, view={view_index}"
            ) from exc
        if depth_frame_id != expected_frame_id or image_frame_id != expected_frame_id:
            raise RuntimeError(
                f"RGB/Depth/frame id order inconsistency: idx={idx}, view={view_index}, expected={expected_frame_id}, rgb={image_frame_id}, depth={depth_frame_id}"
            )
        with np.load(depth_path, allow_pickle=False) as data:
            if set(data.files) != {"depth_map", "depth_size"}:
                raise RuntimeError(
                    f"Frame-by-frame Depth NPZ must contain only depth_map/depth_size: idx={idx}, view={view_index}, path={depth_path}, keys={sorted(data.files)}"
                )
            raw = np.asarray(data["depth_map"])
            depth_size = np.asarray(data["depth_size"])
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise RuntimeError(
                f"Frame-by-frame Depth must be 2D uint16 in millimeters: idx={idx}, view={view_index}, path={depth_path}, dtype={raw.dtype}, shape={raw.shape}"
            )
        if depth_size.shape != (2,) or not np.array_equal(
            depth_size.astype(np.int64), raw.shape
        ):
            raise RuntimeError(
                f"depth_size and depth_map shape are inconsistent: idx={idx}, view={view_index}, path={depth_path}, depth_size={depth_size.tolist()}, shape={raw.shape}"
            )
        if not bool(np.any(raw > 0)):
            raise RuntimeError(
                f"Frame-by-frame depth all invalid 0: idx={idx}, view={view_index}, path={depth_path}"
            )
        if first_shape is None:
            first_shape = raw.shape
        elif raw.shape != first_shape:
            # No resize: Different windows in the same grid cannot be losslessly stacked into [V,H,W], which directly exposes data errors.
            raise RuntimeError(
                f"Inconsistent Depth shape within the same Window: idx={idx}, view={view_index}, first_shape={first_shape}, current_shape={raw.shape}, path={depth_path}"
            )
        frames.append(raw.astype(np.float32) / 1000.0)

    depth = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.float32)
    _assert_depth_value_range(
        depth, idx=idx, path=",".join(str(path) for path in depth_paths)
    )
    _assert_depth_rgb_aspect_ratio(
        depth, image_sizes, idx=idx, path=",".join(str(path) for path in depth_paths)
    )
    return torch.from_numpy(depth)


class CameraInterpreterCollator:
    def __init__(
        self,
        processor,
        feature_vggt_dir: Optional[str],
        camera_latent_size: int,
        user_prompt: str,
        llm_use_image_tokens: bool = True,
        interpreter_use_image_tokens: bool = False,
        expected_image_size: Optional[tuple] = (480, 640),
        camera_token_position: str = "before_question",
        depth_token_position: str = "before_question",
        use_depth_interpreter: bool = False,
        use_depth_patch_tokenizer: bool = False,
        depth_latent_size: int = 8,
        camera_token_layout: str = "global",
        depth_token_layout: str = "global",
        use_camera_interpreter: bool = True,
        camera_loss_mode: str = CAMERA_LOSS_MODE_FULL_CAPTION,
        camera_cot_version: Optional[str] = None,
        depth_loss_mode: str = DEPTH_LOSS_MODE_FULL_CAPTION,
        depth_cot_version: Optional[str] = None,
        prompt_source: str = "fixed",
        depth_frame_supervision_scope: str = DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
        depth_frame_microbatch_mode: str = DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
        depth_frame_loss_reduction: str = DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        depth_meter_loss_weight: float = 0.0,
        depth_relation_loss_weight: float = 0.5,
        depth_digit_token_ids: Optional[Sequence[int]] = None,
        depth_decimal_token_id: Optional[int] = None,
        depth_camera_caption_control: bool = False,
        camera_pair_rows: bool = False,
    ):
        self.processor = processor
        self.feature_vggt_dir = feature_vggt_dir
        self.camera_latent_size = int(camera_latent_size)
        self.user_prompt = str(user_prompt)
        self.prompt_source = str(prompt_source).strip()
        self.use_depth_interpreter = bool(use_depth_interpreter)
        self.use_depth_patch_tokenizer = bool(use_depth_patch_tokenizer)
        self.use_camera_interpreter = bool(use_camera_interpreter)
        self.depth_camera_caption_control = bool(depth_camera_caption_control)
        self.camera_pair_rows = bool(camera_pair_rows)
        if self.prompt_source not in PROMPT_SOURCE_CHOICES:
            raise RuntimeError(
                f"prompt_source must be {PROMPT_SOURCE_CHOICES}: actual={self.prompt_source!r}"
            )
        self.llm_use_image_tokens = bool(llm_use_image_tokens)
        self.interpreter_use_image_tokens = bool(interpreter_use_image_tokens)
        if self.use_depth_patch_tokenizer:
            # Direct Patch reuses RGB MRoPE pixel by pixel, requiring a one-to-one grid generated from a real RGB image block.
            if not self.use_depth_interpreter or self.use_camera_interpreter:
                raise RuntimeError("DepthPatchTokenizer only allows Depth-only Stage1.")
            if not self.llm_use_image_tokens or self.interpreter_use_image_tokens:
                raise RuntimeError(
                    "DepthPatchTokenizer must have llm_use_image_tokens=1 and interpreter_use_image_tokens=0; it only reuses the RGB grid and does not read RGB content."
                )
        self.camera_token_position = str(camera_token_position)
        self.depth_token_position = str(depth_token_position)
        self.expected_image_size = expected_image_size
        self.depth_latent_size = int(depth_latent_size)
        self.camera_token_layout = str(camera_token_layout)
        self.depth_token_layout = str(depth_token_layout)
        if self.use_camera_interpreter and self.use_depth_interpreter:
            if self.camera_token_position != self.depth_token_position:
                raise RuntimeError(
                    f"Camera+Depth Stage1 shares a spatial block; the two token positions must be consistent: camera={self.camera_token_position!r}, depth={self.depth_token_position!r}"
                )
        self.spatial_token_position = (
            self.camera_token_position
            if self.use_camera_interpreter
            else self.depth_token_position
        )
        if self.camera_token_layout not in TOKEN_LAYOUT_CHOICES:
            raise RuntimeError(
                f"camera_token_layout must be {TOKEN_LAYOUT_CHOICES}: {self.camera_token_layout!r}"
            )
        if self.depth_token_layout not in TOKEN_LAYOUT_CHOICES:
            raise RuntimeError(
                f"depth_token_layout must be {TOKEN_LAYOUT_CHOICES}: {self.depth_token_layout!r}"
            )
        if (
            self.use_camera_interpreter
            and self.camera_token_layout == "per_frame"
            and self.camera_latent_size != 1
        ):
            raise RuntimeError(
                "Camera per_frame requires camera_latent_size=1 in this configuration."
            )
        if (
            self.use_depth_interpreter
            and self.depth_token_layout == "per_frame"
            and self.depth_latent_size not in {1, 2}
        ):
            # Accept only the supported per-frame depth summary sizes.
            raise RuntimeError("Depth per_frame only allows depth_latent_size=1 or 2.")
        if self.use_depth_patch_tokenizer and self.depth_token_layout != "per_frame":
            raise RuntimeError(
                "The pad value for DepthPatchTokenizer comes from the number of real patches per frame; the CLI layout must fix per_frame."
            )
        self.camera_loss_mode = str(camera_loss_mode)
        if self.camera_loss_mode not in CAMERA_LOSS_MODE_CHOICES:
            raise RuntimeError(
                f"camera_loss_mode must be {CAMERA_LOSS_MODE_CHOICES}: actual={self.camera_loss_mode!r}"
            )
        if self.camera_loss_mode == CAMERA_LOSS_MODE_MOTION_CONTENT:
            if not self.use_camera_interpreter or self.use_depth_interpreter:
                raise RuntimeError(
                    "motion_content allows only Camera-only Stage1: use_camera_interpreter=1 and use_depth_interpreter=0."
                )
        self.camera_cot_version = (
            str(camera_cot_version).strip() if camera_cot_version is not None else None
        )
        self.depth_loss_mode = str(depth_loss_mode)
        if self.depth_loss_mode not in DEPTH_LOSS_MODE_CHOICES:
            raise RuntimeError(
                f"depth_loss_mode must be {DEPTH_LOSS_MODE_CHOICES}: actual={self.depth_loss_mode!r}"
            )
        self.depth_frame_supervision_scope = str(depth_frame_supervision_scope)
        self.depth_frame_microbatch_mode = str(depth_frame_microbatch_mode)
        self.depth_frame_loss_reduction = str(depth_frame_loss_reduction)
        self.depth_perframe_caption_microbatch = (
            self.depth_frame_microbatch_mode
            == DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS
        )
        actual_frame_contract = (
            self.depth_frame_supervision_scope,
            self.depth_frame_microbatch_mode,
            self.depth_frame_loss_reduction,
        )
        if self.depth_perframe_caption_microbatch:
            expected_frame_contract = (
                DEPTH_FRAME_SUPERVISION_SCOPE_OWN,
                DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS,
                DEPTH_FRAME_LOSS_REDUCTION_WINDOW_MEAN,
            )
            if actual_frame_contract != expected_frame_contract:
                raise RuntimeError(
                    f"Frame-by-frame Caption micro-batch parameter combination incomplete: actual={actual_frame_contract}"
                )
            if (
                not self.use_depth_interpreter
                or self.use_depth_patch_tokenizer
                or self.use_camera_interpreter
                or self.depth_loss_mode
                not in {DEPTH_LOSS_MODE_FULL_CAPTION, DEPTH_LOSS_MODE_NUMBER_ONLY}
                or self.prompt_source not in {"none", "fixed"}
                or self.depth_token_layout != "per_frame"
                or self.depth_latent_size != 1
                or self.depth_token_position != "before_question"
            ):
                raise RuntimeError(
                    "Frame-by-frame Caption micro-batch only allows summary Depth-only/per_frame Kd=1/before_question/full_caption or number_only/prompt_source=none/fixed."
                )
        elif actual_frame_contract != (
            DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
            DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
            DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        ):
            raise RuntimeError(
                f"Non-frame-by-frame training cannot independently modify the frame contract: actual={actual_frame_contract}"
            )
        if self.prompt_source == "none" and not self.depth_perframe_caption_microbatch:
            raise RuntimeError(
                "prompt_source=none allows only frame-by-frame Caption micro-batch."
            )
        # Validate prompt fields after all loss-mode attributes are initialized.
        if self.prompt_source == "record" and (
            (self.use_camera_interpreter and not self.depth_camera_caption_control)
            or self.depth_loss_mode
            not in {DEPTH_LOSS_MODE_NUMBER_ONLY, DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA}
        ):
            raise RuntimeError(
                "prompt_source=record allows only Depth-only + number_only/caption_pair_delta Stage1."
            )
        if (
            self.depth_loss_mode
            in {
                DEPTH_LOSS_MODE_NUMBER_ONLY,
                DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA,
            }
            and not self.depth_camera_caption_control
        ):
            if not self.use_depth_interpreter or self.use_camera_interpreter:
                raise RuntimeError(
                    "Depth local loss is only allowed for Depth-only Stage1: use_depth_interpreter=1 and use_camera_interpreter=0."
                )
        self.depth_cot_version = (
            str(depth_cot_version).strip() if depth_cot_version is not None else None
        )
        self.depth_meter_loss_weight = float(depth_meter_loss_weight)
        if (
            not np.isfinite(self.depth_meter_loss_weight)
            or self.depth_meter_loss_weight < 0.0
        ):
            raise RuntimeError(
                f"depth_meter_loss_weight must be finite and >=0: {self.depth_meter_loss_weight}"
            )
        self.depth_digit_token_ids = (
            [int(value) for value in depth_digit_token_ids]
            if depth_digit_token_ids is not None
            else None
        )
        self.depth_decimal_token_id = (
            int(depth_decimal_token_id) if depth_decimal_token_id is not None else None
        )
        if self.depth_meter_loss_weight > 0.0:
            if self.depth_loss_mode != DEPTH_LOSS_MODE_NUMBER_ONLY:
                raise RuntimeError(
                    "meter loss only allows depth_loss_mode=number_only."
                )
            if (
                self.depth_digit_token_ids is None
                or self.depth_decimal_token_id is None
            ):
                raise RuntimeError(
                    "When enabling meter loss, you must provide tokenizer digit and decimal point token contracts."
                )
        self.depth_relation_loss_weight = float(depth_relation_loss_weight)
        if not np.isfinite(self.depth_relation_loss_weight):
            raise RuntimeError(
                f"depth_relation_loss_weight must be finite: {self.depth_relation_loss_weight}"
            )
        if self.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
            if self.depth_relation_loss_weight != 0.5:
                raise RuntimeError(
                    f"caption_pair_delta requires depth_relation_loss_weight=0.5: actual={self.depth_relation_loss_weight}"
                )
        elif self.depth_relation_loss_weight != 0.5:
            raise RuntimeError(
                "caption_pair_delta does not consume depth_relation_loss_weight, and non-default values cannot be passed."
            )
        self.camera_token_id = (
            _single_token_id(processor.tokenizer, CAMERA_PAD)
            if self.use_camera_interpreter
            else -1
        )
        self.depth_token_id = (
            _single_token_id(processor.tokenizer, DEPTH_PAD)
            if self.use_depth_interpreter
            else -1
        )
        self.camera_start_id = (
            _single_token_id(processor.tokenizer, CAMERA_START)
            if self.use_camera_interpreter
            else -1
        )
        self.camera_end_id = (
            _single_token_id(processor.tokenizer, CAMERA_END)
            if self.use_camera_interpreter
            else -1
        )
        self.depth_start_id = (
            _single_token_id(processor.tokenizer, DEPTH_START)
            if self.use_depth_interpreter
            else -1
        )
        self.depth_end_id = (
            _single_token_id(processor.tokenizer, DEPTH_END)
            if self.use_depth_interpreter
            else -1
        )
        self.image_token_id = _single_token_id(processor.tokenizer, "<|image_pad|>")
        if processor.tokenizer.pad_token_id is None:
            raise RuntimeError(
                "processor.tokenizer.pad_token_id is empty; Camera Interpreter collator does not automatically replace pad tokens."
            )
        self.pad_token_id = int(processor.tokenizer.pad_token_id)
        self.answer_start_pattern = torch.tensor(
            processor.tokenizer("<|im_start|>assistant", add_special_tokens=False)[
                "input_ids"
            ],
            dtype=torch.long,
        )
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError(
                "processor lacks image_processor, cannot construct pose-only geometry grid."
            )
        self.image_processor_patch_size = int(
            getattr(image_processor, "patch_size", 14)
        )
        self.image_processor_merge_size = int(
            getattr(
                image_processor,
                "merge_size",
                getattr(image_processor, "spatial_merge_size", 2),
            )
        )
        self.image_processor_min_pixels = int(
            getattr(image_processor, "min_pixels", DEFAULT_IMAGE_MIN_PIXELS)
        )
        self.image_processor_max_pixels = int(
            getattr(image_processor, "max_pixels", DEFAULT_IMAGE_MAX_PIXELS)
        )
        self._logged_label_contracts = set()
        # Construct the pose-only token grid from image_sizes using the processor dimensions.

    def _expand_image_tokens(self, text: str, image_grid_thw: torch.Tensor) -> str:
        # Expand <|image_pad|> tokens explicitly because the tokenizer does not know the patch count for each separately processed image.
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        merge_length = int(self.image_processor_merge_size) ** 2
        expanded = str(text)
        for idx, thw in enumerate(image_grid_thw):
            if image_token not in expanded:
                raise RuntimeError(
                    f"image token count in chat template is less than number of images: missing index={idx}, token={image_token!r}"
                )
            repeat = int(thw.prod().item()) // merge_length
            if repeat <= 0:
                raise RuntimeError(
                    f"Invalid number of image tokens derived from image_grid_thw: thw={thw.tolist()}, repeat={repeat}"
                )
            expanded = expanded.replace(image_token, "<|placeholder|>" * repeat, 1)
        if image_token in expanded:
            raise RuntimeError(
                f"image token count in chat template exceeds number of images: token={image_token!r}"
            )
        return expanded.replace("<|placeholder|>", image_token)

    def _process_images_individually(
        self,
        text: str,
        images: List[Image.Image],
    ) -> tuple[Dict[str, torch.Tensor], str]:
        """Process RGB image by image, and return the complete chat text after fully expanding the image placeholder."""
        if not images:
            raise RuntimeError(
                "When llm_use_image_tokens=True, images cannot be empty."
            )
        pixel_values = []
        grids = []
        for image_idx, image in enumerate(images):
            # Do not submit a set of images all at once to the Qwen processor:
            # The local processor will resize all images in this group to the dimensions of the first image. For datasets like Ego3D/Waymo, samples with mixed width and height will be stretched.
            # Processing images individually can preserve each camera's own aspect ratio, thus eliminating the need for 480x640 black-bordered images.
            image_inputs = self.processor.image_processor(
                images=[image], return_tensors="pt"
            )
            if (
                "pixel_values" not in image_inputs
                or "image_grid_thw" not in image_inputs
            ):
                raise RuntimeError(
                    f"processor.image_processor outputs missing field: image_idx={image_idx}, keys={list(image_inputs.keys())}"
                )
            grid = image_inputs["image_grid_thw"]
            if tuple(grid.shape) != (1, 3):
                raise RuntimeError(
                    f"Single image image_grid_thw shape should be (1,3): image_idx={image_idx}, shape={tuple(grid.shape)}"
                )
            pixel_values.append(image_inputs["pixel_values"])
            grids.append(grid)
        image_grid_thw = torch.cat(grids, dim=0)
        expanded_text = self._expand_image_tokens(text, image_grid_thw)
        batch = self.processor.tokenizer(
            text=[expanded_text],
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=(
                self.depth_loss_mode
                in {DEPTH_LOSS_MODE_NUMBER_ONLY, DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA}
                or self.camera_loss_mode == CAMERA_LOSS_MODE_MOTION_CONTENT
                or (
                    self.camera_loss_mode == CAMERA_LOSS_MODE_FULL_CAPTION
                    and self.camera_cot_version == CAMERA_PLANAR_METRIC_COT_VERSION
                )
                or (
                    self.depth_loss_mode == DEPTH_LOSS_MODE_FULL_CAPTION
                    and self.depth_cot_version == DEPTH_NUMERIC_READOUT_COT_VERSION
                )
            ),
        )
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        batch["image_grid_thw"] = image_grid_thw
        return batch, expanded_text

    def _preview_image_grid_thw(self, images: List[Image.Image]) -> torch.Tensor:
        """Before assembling the prompt, obtain the true Qwen grid for each RGB image to write an equal number of Depth pads."""
        grids = []
        for image_idx, image in enumerate(images):
            image_inputs = self.processor.image_processor(
                images=[image], return_tensors="pt"
            )
            grid = image_inputs.get("image_grid_thw")
            if grid is None or tuple(grid.shape) != (1, 3):
                raise RuntimeError(
                    f"DepthPatchTokenizer cannot obtain single-image Qwen grid: image_idx={image_idx}, shape={(None if grid is None else tuple(grid.shape))}"
                )
            grids.append(grid)
        if not grids:
            raise RuntimeError("DepthPatchTokenizer must have at least one RGB image.")
        return torch.cat(grids, dim=0)

    def build_pose_only_grid(self, ex: dict, extrinsics: torch.Tensor) -> torch.Tensor:
        # pose-only fair ablation: replicate Qwen smart_resize using the actual dimensions of each original image
        # Ensure the geometry grid token count matches exactly with the img+pose mode (only remove image content, do not compress the grid).
        image_sizes = ex.get("image_sizes")
        if image_sizes is None:
            raise RuntimeError(
                "pose-only grid geometry requires samples to include image_sizes."
            )
        if len(image_sizes) != int(ex["n_images"]):
            raise RuntimeError(
                f"image_sizes and n_images count do not match: idx={ex['idx']}, image_sizes={len(image_sizes)}, n_images={ex['n_images']}"
            )
        grid_rows = []
        one_view_extrinsics = extrinsics[:, :1].contiguous()
        for image_width, image_height in image_sizes:
            # pose-only does not read image pixels, but here it must replicate Qwen's grid according to the original image dimensions for each image.
            # The only difference between this and img+pose is whether image token content is included, avoiding the confound of "spatial grid compressed into 1 token/view".
            one_grid = build_pose_only_image_grid_thw(
                one_view_extrinsics,
                spatial_merge_size=self.image_processor_merge_size,
                patch_size=self.image_processor_patch_size,
                image_width=int(image_width),
                image_height=int(image_height),
                min_pixels=self.image_processor_min_pixels,
                max_pixels=self.image_processor_max_pixels,
            )
            grid_rows.append(one_grid[0])
        return torch.stack(grid_rows, dim=0)

    def _build_grid_from_image_sizes(self, ex: dict) -> torch.Tensor:
        """depth-only + pose-only mode: Build a grid from JSONL image_sizes and align it with Qwen smart_resize."""
        image_sizes = ex.get("image_sizes")
        if image_sizes is None:
            raise RuntimeError(
                f"depth-only grid requires samples to include image_sizes: idx={ex.get('idx')}"
            )
        dummy_ext = torch.zeros(1, 1, 3, 4)
        grid_rows = []
        for image_width, image_height in image_sizes:
            one_grid = build_pose_only_image_grid_thw(
                dummy_ext,
                spatial_merge_size=self.image_processor_merge_size,
                patch_size=self.image_processor_patch_size,
                image_width=int(image_width),
                image_height=int(image_height),
                min_pixels=self.image_processor_min_pixels,
                max_pixels=self.image_processor_max_pixels,
            )
            grid_rows.append(one_grid[0])
        return torch.stack(grid_rows, dim=0)

    def _build_image_sizes_tensor(self, ex: dict) -> torch.Tensor:
        """Build a (V, 2) tensor [width, height] from JSONL image_sizes."""
        image_sizes = ex.get("image_sizes")
        if image_sizes is None:
            raise RuntimeError(
                f"depth-only requires samples to include image_sizes: idx={ex.get('idx')}"
            )
        return torch.tensor(image_sizes, dtype=torch.long)

    def _build_camera_pair_microbatch(self, ex: dict) -> Dict[str, torch.Tensor]:
        """A single original Window generates V-1 isolated graphs of Figure 1 + target figure Qwen row, with the outer layer still optimizing only once."""
        views = int(ex["n_images"])
        images = ex.get("images")
        captions = str(ex["global_scene_cot"]).splitlines()
        if (
            views < 2
            or not isinstance(images, list)
            or len(images) != views
            or len(captions) != views
        ):
            raise RuntimeError(
                f"Inconsistent RGB/Caption counts for Camera pair Window: idx={ex.get('idx')}, V={views}"
            )
        if captions[0] != "image 1 is the reference view.":
            raise RuntimeError(
                f"Camera pair missing reference frame description: idx={ex.get('idx')}"
            )
        if (
            not self.user_prompt.strip()
            or self.camera_token_position != "before_question"
        ):
            raise RuntimeError(
                "Camera pair must retain the fixed prompt and the Camera token must precede the prompt."
            )

        # RGB is preprocessed once per frame; in Qwen row, Figure 1 is repeated, but geometry is computed only once per original Window frame number.
        processed = []
        for image in images:
            item = self.processor.image_processor(images=[image], return_tensors="pt")
            if tuple(item["image_grid_thw"].shape) != (1, 3):
                raise RuntimeError(f"Camera pair RGB grid invalid: idx={ex.get('idx')}")
            processed.append(item)
        source_grid = torch.cat([item["image_grid_thw"] for item in processed], dim=0)
        camera_tokens = _build_camera_token_block(
            1, True, token_layout="per_frame", num_frames=2
        )
        texts, pair_captions, pair_grids = [], [], []
        for target in range(1, views):
            caption = captions[0] + "\n" + captions[target]
            pair_grid = torch.cat(
                [processed[0]["image_grid_thw"], processed[target]["image_grid_thw"]],
                dim=0,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[target]},
                        {
                            "type": "text",
                            "text": f"{camera_tokens}\n{self.user_prompt}",
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": caption}]},
            ]
            rendered = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(self._expand_image_tokens(rendered, pair_grid))
            pair_captions.append(caption)
            pair_grids.append(pair_grid)
        batch = self.processor.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
        )
        offsets = batch.pop("offset_mapping")
        labels, spans = _build_motion_content_labels(
            full_texts=texts,
            captions=pair_captions,
            input_ids=batch["input_ids"],
            offset_mapping=offsets,
            tokenizer=self.processor.tokenizer,
            contexts=[
                f"idx={ex['idx']} target={target + 1}" for target in range(1, views)
            ],
            expected_frame_numbers=list(range(2, views + 1)),
        )
        for row in range(views - 1):
            _assert_camera_token_layout(
                batch["input_ids"][row : row + 1],
                camera_start_id=self.camera_start_id,
                camera_pad_id=self.camera_token_id,
                camera_end_id=self.camera_end_id,
                expected_pad_count=2,
                context=f"Camera pair idx={ex['idx']} row={row}",
            )
        pose = _load_camera_pose_from_path(
            ex["pose_path"],
            expected_views=views,
            expected_image_paths=ex.get("image_input"),
            expected_image_sizes=ex.get("image_sizes"),
            expected_frame_ids=ex.get("window_frame_ids"),
        )
        batch.update(pose)
        batch.update(
            {
                "labels": labels,
                # Encode each image once per window and reuse its features across reference-target pairs.
                "pixel_values": torch.cat(
                    [item["pixel_values"] for item in processed], dim=0
                ),
                "image_grid_thw": torch.cat(pair_grids, dim=0),
                "camera_source_grid_thw": source_grid,
                "camera_frame_indices": torch.arange(views, dtype=torch.long),
                "source_window_size": torch.tensor([views], dtype=torch.long),
                "has_pose": torch.tensor([1], dtype=torch.long),
                "idx": torch.tensor([int(ex["idx"])], dtype=torch.long),
                "supervised_token_count": (labels != -100).sum(dim=1),
            }
        )
        if not hasattr(self, "_logged_camera_pair_contracts"):
            self._logged_camera_pair_contracts = set()
        if views not in self._logged_camera_pair_contracts:
            self._logged_camera_pair_contracts.add(views)
            print(
                "[stage1-camera-pair-contract] "
                f"frames={views} qwen_batch={views - 1} camera_pads_per_row=2 "
                f"loss=motion_content reduction=target_frame_mean_per_source_window spans={spans}",
                flush=True,
            )
        return batch

    def _build_depth_perframe_caption_microbatch(
        self, ex: dict
    ) -> Dict[str, torch.Tensor]:
        """Construct a source Window into V isolated single-frame Qwen rows."""
        num_frames = int(ex["n_images"])
        captions = ex.get("depth_single_frame_captions")
        if (
            not isinstance(captions, list)
            or len(captions) != num_frames
            or any(not str(caption).strip() for caption in captions)
        ):
            raise RuntimeError(
                f"Frame-by-frame Caption counts must equal source Window frame count: idx={ex.get('idx')}, captions={type(captions).__name__}, V={num_frames}"
            )
        images = ex.get("images")
        if (
            not self.llm_use_image_tokens
            or not isinstance(images, list)
            or len(images) != num_frames
        ):
            raise RuntimeError(
                f"Frame-by-frame Caption micro-batch must provide one RGB image per frame: idx={ex.get('idx')}, images={(0 if images is None else len(images))}, V={num_frames}"
            )
        depth_tokens = _build_depth_token_block(
            self.depth_latent_size,
            True,
            token_layout="per_frame",
            # Each Qwen row contains only one D_i; the Depth Interpreter still receives the entire V frame at once to preserve the original frame index.
            num_frames=1,
        )
        texts = []
        pixel_values = []
        grids = []
        for frame_index, (image, caption) in enumerate(zip(images, captions)):
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": depth_tokens},
            ]
            # Fix the prompt to be added only to the current frame's row, maintaining no Qwen attention edges between different frames.
            if self.prompt_source == "fixed":
                if not self.user_prompt.strip():
                    raise RuntimeError(
                        "The frame-by-frame fixed prompt cannot be empty."
                    )
                user_content.append({"type": "text", "text": self.user_prompt})
            messages = [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(caption)}],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False)
            image_inputs = self.processor.image_processor(
                images=[image], return_tensors="pt"
            )
            if (
                "pixel_values" not in image_inputs
                or "image_grid_thw" not in image_inputs
            ):
                raise RuntimeError(
                    f"Frame-by-frame processor output missing fields: idx={ex.get('idx')}, frame={frame_index}, keys={sorted(image_inputs.keys())}"
                )
            grid = image_inputs["image_grid_thw"]
            if tuple(grid.shape) != (1, 3):
                raise RuntimeError(
                    f"Frame-by-frame image_grid_thw must be [1,3]: idx={ex.get('idx')}, frame={frame_index}, shape={tuple(grid.shape)}"
                )
            texts.append(self._expand_image_tokens(text, grid))
            pixel_values.append(image_inputs["pixel_values"])
            grids.append(grid)
        batch = self.processor.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY,
        )
        image_grid_thw = torch.cat(grids, dim=0)
        for frame_index in range(num_frames):
            row_ids = batch["input_ids"][frame_index : frame_index + 1]
            _assert_depth_token_layout(
                row_ids,
                depth_start_id=self.depth_start_id,
                depth_pad_id=self.depth_token_id,
                depth_end_id=self.depth_end_id,
                expected_pad_count=1,
                context=f"Depth per-frame idx={ex.get('idx')} frame={frame_index}",
            )
            expected_image_tokens = get_llm_image_token_counts(
                image_grid_thw[frame_index : frame_index + 1],
                spatial_merge_size=self.image_processor_merge_size,
            )[0]
            actual_image_tokens = int((row_ids == self.image_token_id).sum().item())
            if actual_image_tokens != expected_image_tokens:
                raise RuntimeError(
                    f"Frame-by-frame RGB placeholder and grid are inconsistent: idx={ex.get('idx')}, frame={frame_index}, tokens={actual_image_tokens}, expected={expected_image_tokens}"
                )
        if self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY:
            # Retain caption context while supervising only the depth number for the selected frame.
            offset_mapping = batch.pop("offset_mapping")
            labels, _ = _build_number_only_labels(
                full_texts=texts,
                captions=captions,
                input_ids=batch["input_ids"],
                offset_mapping=offset_mapping,
                tokenizer=self.processor.tokenizer,
                contexts=[f"idx={ex['idx']} frame={i}" for i in range(num_frames)],
                allow_empty=True,
            )
        else:
            labels = _build_labels_after_assistant(
                batch["input_ids"],
                self.answer_start_pattern,
                pad_token_id=self.pad_token_id,
                image_token_id=self.image_token_id,
                camera_token_id=self.camera_token_id,
            )
        labels[batch["input_ids"] == self.depth_token_id] = -100
        supervised_counts = (labels != -100).sum(dim=1)
        if bool((supervised_counts <= 0).all().item()) or (
            self.depth_loss_mode != DEPTH_LOSS_MODE_NUMBER_ONLY
            and bool((supervised_counts <= 0).any().item())
        ):
            raise RuntimeError(
                f"Frame-by-frame Caption has no available supervised digits, or full-caption has empty row: idx={ex.get('idx')}, counts={supervised_counts.tolist()}"
            )
        depth_maps = _load_depth_maps_from_paths(
            ex.get("depth_paths"),
            expected_views=num_frames,
            image_sizes=ex.get("image_sizes"),
            expected_image_paths=ex.get("image_input"),
            expected_frame_ids=ex.get("window_frame_ids"),
            idx=int(ex["idx"]),
        )
        contract_key = ("depth_perframe_caption_microbatch", num_frames)
        if contract_key not in self._logged_label_contracts:
            self._logged_label_contracts.add(contract_key)
            print(
                "[stage1-label-contract] "
                f"branch=depth frames={num_frames} qwen_batch={num_frames} "
                f"loss_mode={self.depth_loss_mode} frame_scope=own_frame_only "
                f"reduction=mean_per_source_window prompt_source={self.prompt_source} "
                f"supervised_tokens={supervised_counts.tolist()}",
                flush=True,
            )
        batch.update(
            {
                "labels": labels,
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": image_grid_thw,
                "depth_maps": depth_maps,
                "image_sizes": self._build_image_sizes_tensor(ex),
                "idx": torch.full((num_frames,), int(ex["idx"]), dtype=torch.long),
                "n_images": torch.ones(num_frames, dtype=torch.long),
                "source_window_size": torch.tensor([num_frames], dtype=torch.long),
                "supervised_token_count": supervised_counts,
            }
        )
        return batch

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise RuntimeError(
                f"Camera Interpreter v1 only supports per_device_train_batch_size=1, current batch_size={len(examples)}"
            )
        ex = examples[0]
        if self.camera_pair_rows:
            return self._build_camera_pair_microbatch(ex)
        if self.depth_perframe_caption_microbatch:
            return self._build_depth_perframe_caption_microbatch(ex)
        num_frames = int(ex["n_images"])
        has_pose = _resolve_has_pose(
            ex,
            num_frames,
            context=f"Stage1 collator idx={ex['idx']}",
        )
        insert_camera_tokens = (
            self.use_camera_interpreter
            and _should_insert_camera_tokens(
                has_pose,
                num_frames,
                context=f"Stage1 collator idx={ex['idx']}",
            )
        )
        camera_tokens = _build_camera_token_block(
            self.camera_latent_size,
            insert_camera_tokens,
            token_layout=self.camera_token_layout,
            num_frames=num_frames,
        )
        depth_patch_token_count = None
        if self.use_depth_patch_tokenizer:
            if "images" not in ex:
                raise RuntimeError(
                    "DepthPatchTokenizer requires loaded RGB images to determine patch alignment."
                )
            preview_grid = self._preview_image_grid_thw(ex["images"])
            if int(preview_grid.shape[0]) != num_frames:
                raise RuntimeError(
                    f"DepthPatchTokenizer preview grid and sample frame count do not match: idx={ex['idx']}, grid={tuple(preview_grid.shape)}, n_images={num_frames}"
                )
            depth_patch_token_count = sum(
                get_llm_image_token_counts(
                    preview_grid,
                    spatial_merge_size=self.image_processor_merge_size,
                )
            )
        depth_tokens = _build_depth_token_block(
            self.depth_latent_size,
            self.use_depth_interpreter,
            token_layout=self.depth_token_layout,
            num_frames=num_frames,
            patch_token_count=depth_patch_token_count,
        )
        combined_tokens = camera_tokens + depth_tokens
        prompt = ex.get("user_prompt", self.user_prompt)
        if self.prompt_source == "record":
            expected_prompt_field = (
                "depth_object_query"
                if self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY
                else "depth_pair_query"
            )
            if not str(prompt).strip() or expected_prompt_field not in ex:
                raise RuntimeError(
                    f"prompt_source=record sample missing line-by-line Question: idx={ex['idx']}, field={expected_prompt_field}"
                )
            prompt = str(prompt).strip()
        camera_prefix, prompt_text = _assemble_camera_prompt(
            combined_tokens,
            prompt,
            self.spatial_token_position,
        )
        user_content = []
        if camera_prefix:
            user_content.append({"type": "text", "text": camera_prefix})
        if self.llm_use_image_tokens:
            if "images" not in ex:
                raise RuntimeError(
                    "When llm_use_image_tokens=True, the sample must contain images."
                )
            user_content.extend(
                {"type": "image", "image": image} for image in ex["images"]
            )
        user_content.append({"type": "text", "text": prompt_text})
        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ex["global_scene_cot"]}],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        tokenized_text = text
        if self.llm_use_image_tokens:
            batch, tokenized_text = self._process_images_individually(
                text, ex["images"]
            )
        else:
            if (
                self.depth_loss_mode
                in {DEPTH_LOSS_MODE_NUMBER_ONLY, DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA}
                or self.camera_loss_mode == CAMERA_LOSS_MODE_MOTION_CONTENT
                or (
                    self.camera_loss_mode == CAMERA_LOSS_MODE_FULL_CAPTION
                    and self.camera_cot_version == CAMERA_PLANAR_METRIC_COT_VERSION
                )
                or (
                    self.depth_loss_mode == DEPTH_LOSS_MODE_FULL_CAPTION
                    and self.depth_cot_version == DEPTH_NUMERIC_READOUT_COT_VERSION
                )
            ):
                batch = self.processor.tokenizer(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    return_offsets_mapping=True,
                )
            else:
                batch = self.processor(text=[text], return_tensors="pt", padding=True)
        if self.use_camera_interpreter:
            _assert_camera_token_layout(
                batch["input_ids"],
                camera_start_id=self.camera_start_id,
                camera_pad_id=self.camera_token_id,
                camera_end_id=self.camera_end_id,
                expected_pad_count=_geometry_prompt_token_count(
                    self.camera_latent_size,
                    self.camera_token_layout,
                    num_frames,
                )
                if insert_camera_tokens
                else 0,
                context=f"Stage1 Camera idx={ex['idx']}",
            )
        if self.use_depth_interpreter:
            _assert_depth_token_layout(
                batch["input_ids"],
                depth_start_id=self.depth_start_id,
                depth_pad_id=self.depth_token_id,
                depth_end_id=self.depth_end_id,
                expected_pad_count=_geometry_prompt_token_count(
                    self.depth_latent_size,
                    self.depth_token_layout,
                    num_frames,
                )
                if depth_patch_token_count is None
                else depth_patch_token_count,
                context=f"Stage1 Depth idx={ex['idx']}",
            )
        if insert_camera_tokens:
            # Load pose_path directly and validate image order, dimensions, and the reference frame.
            pair_indices = ex.get("camera_frame_indices")
            camera = _load_camera_pose_from_path(
                ex["pose_path"],
                expected_views=(
                    len(ex["camera_source_image_input"])
                    if pair_indices is not None
                    else int(ex["n_images"])
                ),
                expected_image_paths=ex.get(
                    "camera_source_image_input", ex.get("image_input")
                ),
                expected_image_sizes=ex.get(
                    "camera_source_image_sizes", ex.get("image_sizes")
                ),
                expected_frame_ids=ex.get(
                    "camera_source_frame_ids", ex.get("window_frame_ids")
                ),
            )
            if pair_indices is not None:
                # Validate the complete pose sequence before selecting the reference and target views.
                if len(pair_indices) != 2 or int(pair_indices[0]) != 0:
                    raise RuntimeError(f"Invalid camera pair_indices: {pair_indices}")
                camera["extrinsics"] = camera["extrinsics"][
                    :, pair_indices
                ].contiguous()
                camera["intrinsics"] = camera["intrinsics"][
                    :, pair_indices
                ].contiguous()
                camera["image_sizes"] = camera["image_sizes"][pair_indices].contiguous()
                camera["camera_frame_indices"] = torch.tensor(
                    pair_indices, dtype=torch.long
                )
            batch.update(camera)
            if not self.llm_use_image_tokens:
                batch["image_grid_thw"] = self.build_pose_only_grid(
                    ex, batch["extrinsics"]
                )
        if self.use_camera_interpreter:
            # Same as Stage2: when has_pose=0, the prompt has no C_i, and forward must never call Camera.
            batch["has_pose"] = torch.tensor([has_pose], dtype=torch.long)
        if self.use_depth_interpreter:
            # New Depth mainline strictly verifies frame id, order, dtype, shape, aspect ratio, and value range per file; no resizing, no normalization.
            depth_maps = _load_depth_maps_from_paths(
                ex.get("depth_paths"),
                expected_views=int(ex["n_images"]),
                image_sizes=ex.get("image_sizes"),
                expected_image_paths=ex.get("image_input"),
                expected_frame_ids=ex.get("window_frame_ids"),
                idx=int(ex["idx"]),
            )
            batch["depth_maps"] = depth_maps
            if "image_grid_thw" not in batch:
                batch["image_grid_thw"] = self._build_grid_from_image_sizes(ex)
            if "image_sizes" not in batch:
                batch["image_sizes"] = self._build_image_sizes_tensor(ex)
        if self.camera_loss_mode == CAMERA_LOSS_MODE_MOTION_CONTENT:
            if "offset_mapping" not in batch:
                raise RuntimeError(
                    "Local tokenizer does not return offset_mapping, cannot safely construct motion_content labels."
                )
            offset_mapping = batch.pop("offset_mapping")
            batch["labels"], span_counts = _build_motion_content_labels(
                full_texts=[tokenized_text],
                captions=[ex["global_scene_cot"]],
                input_ids=batch["input_ids"],
                offset_mapping=offset_mapping,
                tokenizer=self.processor.tokenizer,
                contexts=[f"idx={ex['idx']}"],
                expected_frame_numbers=(
                    [int(ex["camera_pair_target_frame"])]
                    if "camera_pair_target_frame" in ex
                    else None
                ),
            )
            batch["motion_span_count"] = torch.tensor(span_counts, dtype=torch.long)
            batch["supervised_token_count"] = torch.tensor(
                [int((batch["labels"] != -100).sum().item())],
                dtype=torch.long,
            )
        elif self.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
            if "offset_mapping" not in batch:
                raise RuntimeError(
                    "Local tokenizer does not return offset_mapping, cannot safely construct caption_pair_delta labels."
                )
            offset_mapping = batch.pop("offset_mapping")
            relation_labels, caption_labels, relation_counts, caption_counts = (
                _build_caption_pair_delta_labels(
                    full_texts=[tokenized_text],
                    pair_answers=[ex["global_scene_cot"]],
                    relation_answers=[ex["depth_pair_relation_answer"]],
                    scene_captions=[ex["depth_scene_cot"]],
                    relation_eligible=[bool(ex["relation_eligible"])],
                    input_ids=batch["input_ids"],
                    offset_mapping=offset_mapping,
                    tokenizer=self.processor.tokenizer,
                    contexts=[f"idx={ex['idx']}"],
                )
            )
            # labels retain Caption mask for the Trainer to recognize supervision fields; the model calculates loss using only two explicit masks in this mode.
            batch["labels"] = caption_labels
            batch["depth_pair_relation_labels"] = relation_labels
            batch["depth_pair_caption_labels"] = caption_labels
            batch["depth_pair_relation_eligible"] = torch.tensor(
                [bool(ex["relation_eligible"])], dtype=torch.bool
            )
            batch["relation_numeric_span_count"] = torch.tensor(
                relation_counts, dtype=torch.long
            )
            batch["caption_numeric_span_count"] = torch.tensor(
                caption_counts, dtype=torch.long
            )
            batch["supervised_token_count"] = torch.tensor(
                [
                    int((relation_labels != -100).sum().item())
                    + int((caption_labels != -100).sum().item())
                ],
                dtype=torch.long,
            )
            contract_key = (
                "depth",
                int(ex["n_images"]),
                DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA,
                bool(ex["relation_eligible"]),
            )
            if contract_key not in self._logged_label_contracts:
                self._logged_label_contracts.add(contract_key)
                print(
                    "[stage1-label-contract] "
                    f"branch=depth frames={contract_key[1]} "
                    f"loss_mode={DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA} "
                    f"cot_version={self.depth_cot_version} "
                    f"relation_eligible={contract_key[3]} "
                    f"relation_spans={relation_counts[0]} "
                    f"caption_spans={caption_counts[0]} "
                    f"relation_tokens={int((relation_labels != -100).sum().item())} "
                    f"caption_tokens={int((caption_labels != -100).sum().item())}",
                    flush=True,
                )
        elif self.depth_loss_mode == DEPTH_LOSS_MODE_NUMBER_ONLY:
            if "offset_mapping" not in batch:
                raise RuntimeError(
                    "Local tokenizer does not return offset_mapping, cannot safely construct number_only labels."
                )
            offset_mapping = batch.pop("offset_mapping")
            batch["labels"], span_counts = _build_number_only_labels(
                full_texts=[tokenized_text],
                captions=[ex["global_scene_cot"]],
                input_ids=batch["input_ids"],
                offset_mapping=offset_mapping,
                tokenizer=self.processor.tokenizer,
                contexts=[f"idx={ex['idx']}"],
            )
            batch["numeric_span_count"] = torch.tensor(span_counts, dtype=torch.long)
            batch["supervised_token_count"] = torch.tensor(
                [int((batch["labels"] != -100).sum().item())],
                dtype=torch.long,
            )
            contract_key = ("depth", int(ex["n_images"]), DEPTH_LOSS_MODE_NUMBER_ONLY)
            if contract_key not in self._logged_label_contracts:
                self._logged_label_contracts.add(contract_key)
                print(
                    "[stage1-label-contract] "
                    f"branch=depth frames={contract_key[1]} "
                    f"loss_mode={DEPTH_LOSS_MODE_NUMBER_ONLY} "
                    f"cot_version={self.depth_cot_version} "
                    f"numeric_spans={span_counts[0]} "
                    f"supervised_tokens={int((batch['labels'] != -100).sum().item())}",
                    flush=True,
                )
            if self.depth_meter_loss_weight > 0.0:
                (
                    batch["depth_meter_integer_positions"],
                    batch["depth_meter_fractional_positions"],
                    batch["depth_meter_targets_m"],
                ) = _build_depth_meter_targets(
                    full_text=tokenized_text,
                    caption=ex["global_scene_cot"],
                    input_ids=batch["input_ids"],
                    offset_mapping=offset_mapping,
                    digit_token_ids=self.depth_digit_token_ids,
                    decimal_token_id=self.depth_decimal_token_id,
                    context=f"idx={ex['idx']}",
                )
                if int(batch["depth_meter_targets_m"].numel()) != int(span_counts[0]):
                    raise RuntimeError(
                        f"meter target count and number_only span count are inconsistent: idx={ex['idx']}, meter={batch['depth_meter_targets_m'].numel()}, number_only={span_counts[0]}"
                    )
        else:
            # full_caption supervises the complete assistant target.
            batch["labels"] = _build_labels_after_assistant(
                batch["input_ids"],
                self.answer_start_pattern,
                pad_token_id=self.pad_token_id,
                image_token_id=self.image_token_id,
                camera_token_id=self.camera_token_id,
            )
            if self.use_depth_interpreter:
                batch["labels"][batch["input_ids"] == self.depth_token_id] = -100
            full_count = int((batch["labels"] != -100).sum().item())
            baseline_count = None
            baseline_name = None
            if (
                self.use_camera_interpreter
                and self.camera_cot_version == CAMERA_PLANAR_METRIC_COT_VERSION
            ):
                if "offset_mapping" not in batch:
                    raise RuntimeError(
                        "Camera full_caption requires offset_mapping and cannot skip local supervision counting."
                    )
                baseline_labels, _ = _build_motion_content_labels(
                    full_texts=[tokenized_text],
                    captions=[ex["global_scene_cot"]],
                    input_ids=batch["input_ids"],
                    offset_mapping=batch["offset_mapping"],
                    tokenizer=self.processor.tokenizer,
                    contexts=[f"idx={ex['idx']}"],
                )
                baseline_count = int((baseline_labels != -100).sum().item())
                baseline_name = CAMERA_LOSS_MODE_MOTION_CONTENT
            elif (
                self.use_depth_interpreter
                and self.depth_cot_version == DEPTH_NUMERIC_READOUT_COT_VERSION
            ):
                if "offset_mapping" not in batch:
                    raise RuntimeError(
                        "Depth full_caption requires offset_mapping and cannot skip local supervision counting."
                    )
                baseline_labels, _ = _build_number_only_labels(
                    full_texts=[tokenized_text],
                    captions=[ex["global_scene_cot"]],
                    input_ids=batch["input_ids"],
                    offset_mapping=batch["offset_mapping"],
                    tokenizer=self.processor.tokenizer,
                    contexts=[f"idx={ex['idx']}"],
                )
                baseline_count = int((baseline_labels != -100).sum().item())
                baseline_name = DEPTH_LOSS_MODE_NUMBER_ONLY
            if baseline_count is not None:
                # The only experimental variable in full-caption is expanding the supervision scope; no increase in quantity indicates the mask has not genuinely switched.
                if full_count <= baseline_count:
                    raise RuntimeError(
                        f"full_caption must supervise more tokens than content-only supervision: idx={ex['idx']}, full={full_count}, {baseline_name}={baseline_count}"
                    )
                batch["local_baseline_supervised_token_count"] = torch.tensor(
                    [baseline_count],
                    dtype=torch.long,
                )
            batch["supervised_token_count"] = torch.tensor(
                [full_count],
                dtype=torch.long,
            )
            contract_key = (
                "camera" if self.use_camera_interpreter else "depth",
                int(ex["n_images"]),
            )
            if contract_key not in self._logged_label_contracts:
                self._logged_label_contracts.add(contract_key)
                print(
                    "[stage1-label-contract] "
                    f"branch={contract_key[0]} frames={contract_key[1]} "
                    f"loss_mode=full_caption cot_version="
                    f"{self.camera_cot_version if self.use_camera_interpreter else self.depth_cot_version} "
                    f"supervised_tokens={full_count} "
                    f"local_baseline={baseline_name}:{baseline_count}",
                    flush=True,
                )
        batch.pop("offset_mapping", None)
        if not bool((batch["labels"] != -100).any().item()):
            raise RuntimeError(f"No supervised token for sample: idx={ex['idx']}")
        if self.llm_use_image_tokens and int(batch["image_grid_thw"].shape[0]) != int(
            ex["n_images"]
        ):
            raise RuntimeError(
                f"processor image_grid_thw count and n_images are inconsistent: idx={ex['idx']}, grid={tuple(batch['image_grid_thw'].shape)}, n_images={ex['n_images']}"
            )
        batch["idx"] = torch.tensor([int(ex["idx"])], dtype=torch.long)
        batch["n_images"] = torch.tensor([int(ex["n_images"])], dtype=torch.long)
        return batch


class CameraInterpreterForCausalLM(nn.Module):
    """Freeze QwenVL, train only the Camera/Depth Interpreter Stage1 wrapper model."""

    def __init__(
        self,
        qwen_model: Qwen2_5_VLForConditionalGeneration,
        interpreter_config: Optional[CameraInterpreterConfig],
        *,
        camera_token_id: int,
        base_model_name_or_path: str,
        visual_chunk_size: int = 1,
        depth_interpreter_config: Optional[DepthInterpreterConfig] = None,
        depth_patch_tokenizer_config: Optional[DepthPatchTokenizerConfig] = None,
        depth_token_id: int = -1,
        llm_use_image_tokens: bool = True,
        interpreter_use_image_tokens: bool = False,
        camera_loss_mode: str = CAMERA_LOSS_MODE_FULL_CAPTION,
        camera_cot_version: Optional[str] = None,
        depth_loss_mode: str = DEPTH_LOSS_MODE_FULL_CAPTION,
        depth_cot_version: Optional[str] = None,
        depth_frame_supervision_scope: str = DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
        depth_frame_microbatch_mode: str = DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
        depth_frame_loss_reduction: str = DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        depth_meter_loss_weight: float = 0.0,
        depth_relation_loss_weight: float = 0.5,
        depth_digit_token_ids: Optional[Sequence[int]] = None,
        freeze_camera_interpreter: bool = False,
        camera_pair_rows: bool = False,
    ):
        super().__init__()
        self.qwen_model = qwen_model
        # Use native vision and multimodal position interfaces for each backbone.
        self.qwen3_vl = qwen_model.config.model_type in {
            "qwen3_vl",
            "qwen3_5",
            "qwen2_5_vl",
        }
        self.camera_interpreter = (
            CameraInterpreter(interpreter_config)
            if interpreter_config is not None
            else None
        )
        self.freeze_camera_interpreter = bool(freeze_camera_interpreter)
        self.camera_pair_rows = bool(camera_pair_rows)
        if self.freeze_camera_interpreter and self.camera_interpreter is None:
            raise RuntimeError("freeze_camera_interpreter=1 but no Camera Interpreter.")
        self.camera_token_id = int(camera_token_id)
        self.base_model_name_or_path = str(base_model_name_or_path)
        self.llm_use_image_tokens = bool(llm_use_image_tokens)
        self.interpreter_use_image_tokens = bool(interpreter_use_image_tokens)
        self.camera_loss_mode = str(camera_loss_mode)
        if self.camera_loss_mode not in CAMERA_LOSS_MODE_CHOICES:
            raise RuntimeError(
                f"camera_loss_mode must be {CAMERA_LOSS_MODE_CHOICES}: actual={self.camera_loss_mode!r}"
            )
        self.camera_cot_version = (
            str(camera_cot_version).strip() if camera_cot_version is not None else None
        )
        self.depth_loss_mode = str(depth_loss_mode)
        if self.depth_loss_mode not in DEPTH_LOSS_MODE_CHOICES:
            raise RuntimeError(
                f"depth_loss_mode must be {DEPTH_LOSS_MODE_CHOICES}: actual={self.depth_loss_mode!r}"
            )
        self.depth_cot_version = (
            str(depth_cot_version).strip() if depth_cot_version is not None else None
        )
        self.depth_frame_supervision_scope = str(depth_frame_supervision_scope)
        self.depth_frame_microbatch_mode = str(depth_frame_microbatch_mode)
        self.depth_frame_loss_reduction = str(depth_frame_loss_reduction)
        self.depth_perframe_caption_microbatch = (
            self.depth_frame_microbatch_mode
            == DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS
        )
        actual_frame_contract = (
            self.depth_frame_supervision_scope,
            self.depth_frame_microbatch_mode,
            self.depth_frame_loss_reduction,
        )
        expected_frame_contract = (
            DEPTH_FRAME_SUPERVISION_SCOPE_OWN,
            DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS,
            DEPTH_FRAME_LOSS_REDUCTION_WINDOW_MEAN,
        )
        if self.depth_perframe_caption_microbatch:
            if actual_frame_contract != expected_frame_contract:
                raise RuntimeError(
                    f"Frame-by-frame Caption model frame contract incomplete: actual={actual_frame_contract}"
                )
            if (
                depth_interpreter_config is None
                or depth_patch_tokenizer_config is not None
                or interpreter_config is not None
                or self.depth_loss_mode
                not in {DEPTH_LOSS_MODE_FULL_CAPTION, DEPTH_LOSS_MODE_NUMBER_ONLY}
                or self.depth_cot_version
                != DEPTH_PERFRAME_CAPTION_MICROBATCH_COT_VERSION
            ):
                raise RuntimeError(
                    "Frame-by-frame Caption model only allows summary Depth-only/full_caption or number_only/specified cot_version."
                )
        elif actual_frame_contract != (
            DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
            DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
            DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        ):
            raise RuntimeError(
                f"Non-frame-by-frame model cannot independently modify frame contract: actual={actual_frame_contract}"
            )
        self.depth_meter_loss_weight = float(depth_meter_loss_weight)
        self.depth_meter_loss_beta_m = DEPTH_METER_LOSS_BETA_M
        if (
            not np.isfinite(self.depth_meter_loss_weight)
            or self.depth_meter_loss_weight < 0.0
        ):
            raise RuntimeError(
                f"depth_meter_loss_weight must be finite and >=0: {self.depth_meter_loss_weight}"
            )
        self.depth_relation_loss_weight = float(depth_relation_loss_weight)
        if not np.isfinite(self.depth_relation_loss_weight):
            raise RuntimeError(
                f"depth_relation_loss_weight must be finite: {self.depth_relation_loss_weight}"
            )
        if self.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
            if self.depth_relation_loss_weight != 0.5:
                raise RuntimeError(
                    f"caption_pair_delta requires depth_relation_loss_weight=0.5: actual={self.depth_relation_loss_weight}"
                )
        elif self.depth_relation_loss_weight != 0.5:
            raise RuntimeError(
                "caption_pair_delta does not consume depth_relation_loss_weight, and non-default values cannot be passed."
            )
        self.config = qwen_model.config
        self.visual_chunk_size = int(visual_chunk_size)
        if self.visual_chunk_size <= 0:
            raise RuntimeError(
                f"visual_chunk_size must be a positive integer, current={self.visual_chunk_size}"
            )

        if (
            depth_interpreter_config is not None
            and depth_patch_tokenizer_config is not None
        ):
            raise RuntimeError(
                "DepthInterpreter and DepthPatchTokenizer cannot be created simultaneously."
            )
        self.depth_interpreter = None
        self.depth_patch_tokenizer = None
        self.depth_token_id = int(depth_token_id)
        if depth_interpreter_config is not None:
            self.depth_interpreter = DepthInterpreter(depth_interpreter_config)
        if depth_patch_tokenizer_config is not None:
            self.depth_patch_tokenizer = DepthPatchTokenizer(
                depth_patch_tokenizer_config
            )
        if self.depth_meter_loss_weight > 0.0:
            if (
                self.depth_interpreter is None
                or self.camera_interpreter is not None
                or self.depth_loss_mode != DEPTH_LOSS_MODE_NUMBER_ONLY
            ):
                raise RuntimeError(
                    "meter loss only allows Depth-only + depth_loss_mode=number_only."
                )
            if depth_digit_token_ids is None:
                raise RuntimeError(
                    "When meter loss is enabled, token IDs from the tokenizer must be provided in the range 0–9."
                )
            normalized_digit_ids = [int(value) for value in depth_digit_token_ids]
            if len(normalized_digit_ids) != 10 or len(set(normalized_digit_ids)) != 10:
                raise RuntimeError(
                    f"meter loss requires 10 distinct digit token IDs: {normalized_digit_ids}"
                )
        else:
            normalized_digit_ids = []
        # digit ids are used only to select columns from frozen lm_head logits, and do not belong to Interpreter parameters or checkpoint state.
        self.register_buffer(
            "_depth_digit_token_ids",
            torch.tensor(normalized_digit_ids, dtype=torch.long),
            persistent=False,
        )
        self._logged_camera_token_contracts = set()
        self._logged_depth_token_contracts = set()

        for param in self.qwen_model.parameters():
            param.requires_grad = False
        self.qwen_model.eval()
        if self.freeze_camera_interpreter:
            # B's only learnable path is Caption number CE -> Depth; Camera provides only fixed context.
            for param in self.camera_interpreter.parameters():
                param.requires_grad = False
            self.camera_interpreter.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.qwen_model.eval()
        if self.camera_interpreter is not None:
            if self.freeze_camera_interpreter:
                self.camera_interpreter.eval()
            else:
                self.camera_interpreter.train(mode)
        if self.depth_interpreter is not None:
            self.depth_interpreter.train(mode)
        if self.depth_patch_tokenizer is not None:
            self.depth_patch_tokenizer.train(mode)
        return self

    def _encode_visual_in_chunks(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run the frozen visual encoder in image blocks to avoid OOM from visual attention when processing multiple images at once."""
        # Share visual-patch splitting and validate its alignment with image_grid_thw.
        return encode_visual_in_chunks(
            self.qwen_model.visual,
            pixel_values,
            image_grid_thw,
            self.visual_chunk_size,
            context="Stage1 Interpreter visual encoder",
        )

    def _replace_image_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        camera_source_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values is None:
            raise RuntimeError(
                "Camera Interpreter training must input pixel_values; no text-only automatic replacement is performed here."
            )
        if self.camera_pair_rows:
            if camera_source_grid_thw is None:
                raise RuntimeError(
                    "Camera pair lacks the original window RGB grid, preventing reuse of visual features from the reference image."
                )
            source_views = int(camera_source_grid_thw.shape[0])
            if int(input_ids.shape[0]) != source_views - 1 or int(
                image_grid_thw.shape[0]
            ) != 2 * (source_views - 1):
                raise RuntimeError("Camera pair row count does not match RGB grid.")
            expected_pair_grid = torch.stack(
                [
                    torch.stack(
                        (camera_source_grid_thw[0], camera_source_grid_thw[target])
                    )
                    for target in range(1, source_views)
                ]
            ).reshape(-1, 3)
            if not torch.equal(image_grid_thw, expected_pair_grid):
                raise RuntimeError(
                    "Camera pair RGB grid must correspond sequentially to the source image 1 and target image in the original window."
                )
            with torch.no_grad():
                source_embeds = self._encode_visual_in_chunks(
                    pixel_values, camera_source_grid_thw
                )
            source_counts = get_llm_image_token_counts(
                camera_source_grid_thw,
                spatial_merge_size=int(
                    self.camera_interpreter.config.camera_merge_size
                ),
            )
            if sum(source_counts) != int(source_embeds.shape[0]):
                raise RuntimeError(
                    "The number of visual features in the original camera window does not match the RGB grid."
                )
            # Only copy the already computed frozen features; each row is still placed in its own sequence as [RGB1][RGB_i].
            source_slices = list(torch.split(source_embeds, source_counts, dim=0))
            image_embeds = torch.cat(
                [
                    part
                    for target in range(1, source_views)
                    for part in (source_slices[0], source_slices[target])
                ],
                dim=0,
            )
        else:
            with torch.no_grad():
                image_embeds = self._encode_visual_in_chunks(
                    pixel_values, image_grid_thw
                )

        image_token_id = int(self.qwen_model.config.image_token_id)
        n_image_tokens = int((input_ids == image_token_id).sum().item())
        if n_image_tokens != int(image_embeds.shape[0]):
            raise RuntimeError(
                f"tokens={n_image_tokens}, features={int(image_embeds.shape[0])}"
            )
        image_mask = (
            (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        image_embeds = image_embeds.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        return inputs_embeds.masked_scatter(
            image_mask.to(inputs_embeds.device), image_embeds
        )

    def _inject_camera_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_sizes: torch.Tensor,
        camera_frame_indices: Optional[torch.Tensor] = None,
        camera_source_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.camera_pair_rows and input_ids.shape[0] != 1:
            raise RuntimeError(
                f"Camera Interpreter v1 only supports batch_size=1, current={input_ids.shape[0]}"
            )
        image_token_id = int(self.qwen_model.config.image_token_id)
        image_mask = input_ids == image_token_id
        camera_mask = input_ids == self.camera_token_id

        image_tokens = None
        if self.interpreter_use_image_tokens:
            image_tokens = inputs_embeds[0][image_mask[0]].contiguous()
            if image_tokens.numel() == 0:
                raise RuntimeError(
                    "interpreter_use_image_tokens=True but the sample has no image token."
                )
            expected_counts = get_llm_image_token_counts(
                image_grid_thw,
                spatial_merge_size=int(
                    self.camera_interpreter.config.camera_merge_size
                ),
            )
            if sum(expected_counts) != int(image_tokens.shape[0]):
                raise RuntimeError(
                    f"token count derived from image_grid_thw does not match input_ids: expected={sum(expected_counts)}, actual={int(image_tokens.shape[0])}, per_image_counts={expected_counts}"
                )

        camera_count = int(camera_mask.sum().item())
        geometry_grid = (
            camera_source_grid_thw if self.camera_pair_rows else image_grid_thw
        )
        if geometry_grid is None:
            raise RuntimeError(
                "Camera pair is missing the original Window's geometry grid."
            )
        num_frames = int(geometry_grid.shape[0])
        expected_camera = self.camera_interpreter.config.output_token_count(num_frames)
        expected_pads = (
            2 * (num_frames - 1) if self.camera_pair_rows else expected_camera
        )
        if camera_count != expected_pads or (
            self.camera_pair_rows and int(input_ids.shape[0]) != num_frames - 1
        ):
            raise RuntimeError(
                f"The <|camera_pad|> count must equal the Interpreter output count: layout={self.camera_interpreter.config.token_layout}, frames={num_frames}, actual={camera_count}, expected={expected_pads}"
            )

        camera_output_batch = self.camera_interpreter(
            image_tokens=image_tokens,
            image_grid_thw=geometry_grid,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_sizes=image_sizes,
            frame_indices=camera_frame_indices,
        )
        expected_shape = (1, expected_camera, int(inputs_embeds.shape[-1]))
        if tuple(camera_output_batch.shape) != expected_shape:
            raise RuntimeError(
                f"Camera token shape inconsistent: expected={expected_shape}, actual={tuple(camera_output_batch.shape)}"
            )
        contract_key = (self.camera_interpreter.config.token_layout, num_frames)
        if contract_key not in self._logged_camera_token_contracts:
            self._logged_camera_token_contracts.add(contract_key)
            output_rms = float(
                camera_output_batch.detach().float().square().mean().sqrt().item()
            )
            print(
                "[stage1-camera-token-contract] "
                f"layout={self.camera_interpreter.config.token_layout} "
                f"frames={num_frames} prompt_pads={camera_count} "
                f"output_shape={tuple(camera_output_batch.shape)} "
                f"output_contract={self.camera_interpreter.config.output_contract} "
                f"embedding_rms={output_rms:.6f}",
                flush=True,
            )
        camera_output = camera_output_batch[0].to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if self.camera_pair_rows:
            # Each Qwen row independently obtains C1, Ci; repeated C1 shares the same Interpreter gradient.
            targets = torch.arange(1, num_frames, device=camera_output.device)
            camera_output = torch.stack(
                [camera_output[0].expand(num_frames - 1, -1), camera_output[targets]],
                dim=1,
            ).reshape(expected_pads, -1)
        if (
            self.camera_interpreter.config.output_contract
            == OUTPUT_CONTRACT_DIRECT_EMBEDDING
        ):
            # direct replacement: pad ID is only responsible for positioning, and the E_camera_pad obtained via lookup does not enter the Transformer.
            camera_embeds = camera_output
        else:
            base_camera = inputs_embeds[camera_mask].view(expected_pads, -1)
            camera_embeds = base_camera + camera_output

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[camera_mask] = camera_embeds
        return inputs_embeds

    def _inject_depth_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        depth_maps: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if self.depth_interpreter is None:
            raise RuntimeError(
                "_inject_depth_tokens is called but depth_interpreter is not initialized."
            )
        depth_mask = input_ids == self.depth_token_id
        depth_count = int(depth_mask.sum().item())
        num_frames = int(image_grid_thw.shape[0])
        expected_depth = self.depth_interpreter.config.output_token_count(num_frames)
        if depth_count != expected_depth:
            raise RuntimeError(
                f"The <|depth_pad|> count must equal the Interpreter output count: layout={self.depth_interpreter.config.token_layout}, frames={num_frames}, actual={depth_count}, expected={expected_depth}"
            )
        image_token_id = int(self.qwen_model.config.image_token_id)
        image_mask = input_ids == image_token_id
        image_tokens = None
        if self.interpreter_use_image_tokens:
            image_tokens = inputs_embeds[0][image_mask[0]].contiguous()

        depth_output_batch = self.depth_interpreter(
            image_tokens=image_tokens,
            image_grid_thw=image_grid_thw,
            depth_maps=depth_maps,
            image_sizes=image_sizes,
        )
        expected_shape = (1, expected_depth, int(inputs_embeds.shape[-1]))
        if tuple(depth_output_batch.shape) != expected_shape:
            raise RuntimeError(
                f"Depth token shape inconsistency: expected={expected_shape}, actual={tuple(depth_output_batch.shape)}"
            )
        contract_key = (self.depth_interpreter.config.token_layout, num_frames)
        if contract_key not in self._logged_depth_token_contracts:
            self._logged_depth_token_contracts.add(contract_key)
            output_rms = float(
                depth_output_batch.detach().float().square().mean().sqrt().item()
            )
            print(
                "[stage1-depth-token-contract] "
                f"layout={self.depth_interpreter.config.token_layout} "
                f"frames={num_frames} prompt_pads={depth_count} "
                f"output_shape={tuple(depth_output_batch.shape)} "
                f"output_contract={self.depth_interpreter.config.output_contract} "
                f"embedding_rms={output_rms:.6f}",
                flush=True,
            )
        depth_output = depth_output_batch[0].to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if self.depth_perframe_caption_microbatch:
            batch_size = int(input_ids.shape[0])
            if (
                self.depth_interpreter.config.token_layout != "per_frame"
                or int(self.depth_interpreter.config.num_depth_tokens) != 1
                or batch_size != num_frames
            ):
                raise RuntimeError(
                    f"Frame-by-frame caption injection requires Qwen batch=V, Depth per_frame Kd=1: batch={batch_size}, V={num_frames}, layout={self.depth_interpreter.config.token_layout}, Kd={self.depth_interpreter.config.num_depth_tokens}"
                )
            pads_per_row = depth_mask.sum(dim=1)
            if not torch.equal(pads_per_row, torch.ones_like(pads_per_row)):
                raise RuntimeError(
                    f"Frame-by-frame caption: Each Qwen row must have exactly one D_i slot: pads_per_row={pads_per_row.tolist()}"
                )
            # Inside Depth Interpreter, it still constructs a complete V-frame using frame_idx/31, and isolates the patch within the frame-batch.
            # Here, only write back the frame-major [V,d] one-by-one into Qwen's V independent rows.
            depth_output = depth_output.reshape(num_frames, -1)
        if (
            self.depth_interpreter.config.output_contract
            == OUTPUT_CONTRACT_DIRECT_EMBEDDING
        ):
            # direct replacement: pad ID is only responsible for positioning, and the E_depth_pad obtained via lookup does not enter the Transformer at all.
            depth_embeds = depth_output
        else:
            base_depth = inputs_embeds[depth_mask].view(expected_depth, -1)
            depth_embeds = base_depth + depth_output
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[depth_mask] = depth_embeds
        return inputs_embeds

    def _inject_depth_patch_tokens(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        depth_maps: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Use scalar_5d to generate full tokens patch by patch, and strictly replace all Depth pad."""
        if self.depth_patch_tokenizer is None:
            raise RuntimeError(
                "_inject_depth_patch_tokens is called but DepthPatchTokenizer is not initialized."
            )
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"DepthPatchTokenizer currently only supports batch_size=1 to avoid ambiguity in patch alignment across samples: batch={input_ids.shape[0]}"
            )
        config = self.depth_patch_tokenizer.config
        expected_counts = get_llm_image_token_counts(
            image_grid_thw,
            spatial_merge_size=int(config.spatial_merge_size),
        )
        if len(expected_counts) != int(depth_maps.shape[0]):
            raise RuntimeError(
                f"DepthPatchTokenizer depth map frame count and image_grid_thw are inconsistent: depth_frames={depth_maps.shape[0]}, grid_frames={len(expected_counts)}"
            )
        depth_geometry = build_depth_patch_stats(
            depth_maps=depth_maps,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=int(config.spatial_merge_size),
            patch_size=int(config.patch_size),
            image_sizes=image_sizes,
            encoder_type=str(config.depth_encoder_type),
        )
        expected_total = int(sum(expected_counts))
        if int(depth_geometry.shape[0]) != expected_total:
            raise RuntimeError(
                f"DepthPatchTokenizer geometry patch count and RGB LLM patch count are inconsistent: geometry={depth_geometry.shape[0]}, rgb={expected_total}, per_frame={expected_counts}"
            )
        depth_mask = input_ids == self.depth_token_id
        depth_count = int(depth_mask.sum().item())
        if depth_count != expected_total:
            raise RuntimeError(
                f"<|depth_pad|> tokens must align one-to-one with RGB patches: pads={depth_count}, rgb={expected_total}, per_frame={expected_counts}"
            )
        depth_output = self.depth_patch_tokenizer(depth_geometry).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        expected_shape = (expected_total, int(inputs_embeds.shape[-1]))
        if tuple(depth_output.shape) != expected_shape:
            raise RuntimeError(
                f"DepthPatchTokenizer output must have one complete Qwen token per patch: actual={tuple(depth_output.shape)}, expected={expected_shape}"
            )
        contract_key = (tuple(expected_counts), expected_total)
        if contract_key not in self._logged_depth_token_contracts:
            self._logged_depth_token_contracts.add(contract_key)
            print(
                "[stage1-depth-patch-token-contract] "
                f"per_frame={expected_counts} prompt_pads={depth_count} "
                f"output_shape={tuple(depth_output.shape)} "
                f"position_ids={config.position_ids_contract}",
                flush=True,
            )
        # The full semantics of direct_patch_embedding are replacement, not addition with <|depth_pad|> lookup.
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[depth_mask] = depth_output
        return inputs_embeds

    def _build_rgb_aligned_depth_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Copy Qwen's computed RGB 3D MRoPE to the corresponding Depth patch slot."""
        if self.depth_patch_tokenizer is None:
            raise RuntimeError(
                "Only DepthPatchTokenizer can construct RGB-aligned MRoPE."
            )
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"DepthPatchTokenizer MRoPE alignment currently supports only batch_size=1: batch={input_ids.shape[0]}"
            )
        default_position_ids, _ = self.qwen_model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        image_slots = torch.nonzero(
            input_ids[0] == int(self.qwen_model.config.image_token_id),
            as_tuple=False,
        ).flatten()
        depth_slots = torch.nonzero(
            input_ids[0] == self.depth_token_id, as_tuple=False
        ).flatten()
        if (
            int(image_slots.numel()) != int(depth_slots.numel())
            or int(image_slots.numel()) == 0
        ):
            raise RuntimeError(
                f"RGB/Depth patch slot cannot be one-to-one copied MRoPE: rgb={image_slots.numel()}, depth={depth_slots.numel()}"
            )
        position_ids = default_position_ids.clone()
        position_ids[:, 0, depth_slots] = default_position_ids[:, 0, image_slots]
        if not torch.equal(
            position_ids[:, 0, depth_slots], default_position_ids[:, 0, image_slots]
        ):
            raise RuntimeError(
                "After Depth patch MRoPE gather/scatter, it does not equal RGB MRoPE element-wise."
            )
        # No positions other than depth_pad are allowed to be rewritten, especially delimiter, Question, and Answer must remain at their default text position.
        untouched = torch.ones(
            input_ids.shape[1], dtype=torch.bool, device=input_ids.device
        )
        untouched[depth_slots] = False
        if not torch.equal(
            position_ids[:, 0, untouched], default_position_ids[:, 0, untouched]
        ):
            raise RuntimeError(
                "RGB-MRoPE misaligned errors rewritten non-Depth patch position_ids."
            )
        return position_ids

    @staticmethod
    def _masked_causal_mean_ce(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        component_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the standard causal next-token mean CE on an explicit label mask per sample."""
        if logits.dim() != 3 or labels.dim() != 2:
            raise RuntimeError(
                f"{component_name} CE shape illegal: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )
        if tuple(logits.shape[:2]) != tuple(labels.shape):
            raise RuntimeError(
                f"{component_name} CE logits/labels sequence shape inconsistency: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )
        shift_labels = labels[:, 1:].contiguous()
        token_mask = shift_labels.ne(-100)
        token_counts = token_mask.sum(dim=1)
        per_sample_losses = []
        for batch_index in range(int(logits.shape[0])):
            mask = token_mask[batch_index]
            if not bool(mask.any().item()):
                # The relation mask in caption-only rows may be empty; construct a scalar that is connected but zero using logits.
                per_sample_losses.append(logits[batch_index, 0, 0].float() * 0.0)
                continue
            # Only take the true supervised digit positions, avoid duplicating the full [B,S,Vocab] logits as float32 twice.
            selected_logits = logits[batch_index, :-1, :][mask].float()
            selected_labels = shift_labels[batch_index][mask]
            per_sample_losses.append(
                F.cross_entropy(selected_logits, selected_labels, reduction="mean")
            )
        per_sample = torch.stack(per_sample_losses)
        if not bool(torch.isfinite(per_sample).all().item()):
            raise RuntimeError(
                f"{component_name} causal CE produces NaN/Inf: per_sample={per_sample}"
            )
        return per_sample, token_counts

    def _compute_depth_meter_components(
        self,
        logits: torch.Tensor,
        integer_positions: torch.Tensor,
        fractional_positions: torch.Tensor,
        targets_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute metric SmoothL1 and MAE from predicted logits of the current digit's previous digit."""
        if logits.dim() != 3 or int(logits.shape[0]) != 1:
            raise RuntimeError(
                f"meter loss currently requires logits=[1,S,V]: actual={tuple(logits.shape)}"
            )
        integer_positions = integer_positions.reshape(-1).to(
            device=logits.device,
            dtype=torch.long,
        )
        fractional_positions = fractional_positions.reshape(-1).to(
            device=logits.device,
            dtype=torch.long,
        )
        targets_m = targets_m.reshape(-1).to(device=logits.device, dtype=torch.float32)
        count = int(targets_m.numel())
        if count <= 0:
            raise RuntimeError("Meter loss requires at least one depth value.")
        if (
            int(integer_positions.numel()) != count
            or int(fractional_positions.numel()) != count
        ):
            raise RuntimeError(
                f"mismatch between meter loss position and target count: integer={integer_positions.numel()}, fractional={fractional_positions.numel()}, targets={count}"
            )
        sequence_length = int(logits.shape[1])
        for name, positions in (
            ("integer", integer_positions),
            ("fractional", fractional_positions),
        ):
            if bool((positions <= 0).any().item()) or bool(
                (positions >= sequence_length).any().item()
            ):
                raise RuntimeError(
                    f"meter loss {name} label position out of bounds: positions={positions.tolist()}, sequence_length={sequence_length}"
                )

        digit_token_ids = self._depth_digit_token_ids.to(logits.device)
        if int(digit_token_ids.numel()) != 10:
            raise RuntimeError(
                f"meter loss digit token ids count should be 10: {digit_token_ids.tolist()}"
            )
        if bool((digit_token_ids < 0).any().item()) or bool(
            (digit_token_ids >= int(logits.shape[-1])).any().item()
        ):
            raise RuntimeError(
                f"meter loss digit token id out of vocabulary: ids={digit_token_ids.tolist()}, vocab={int(logits.shape[-1])}"
            )

        # The logits[t-1] of a Causal LM predict label[t]; directly taking the label position reveals the ground truth, causing teacher-forcing leakage.
        integer_logits = (
            logits[0]
            .index_select(0, integer_positions - 1)
            .index_select(
                1,
                digit_token_ids,
            )
        )
        fractional_logits = (
            logits[0]
            .index_select(
                0,
                fractional_positions - 1,
            )
            .index_select(1, digit_token_ids)
        )
        digit_values = torch.arange(10, device=logits.device, dtype=torch.float32)
        # Normalize only within the ten digits; probabilities of non-digit tokens in the full vocabulary remain penalized by original CE.
        integer_expectation = (
            torch.softmax(integer_logits.float(), dim=-1) * digit_values
        ).sum(dim=-1)
        fractional_expectation = (
            torch.softmax(fractional_logits.float(), dim=-1) * digit_values
        ).sum(dim=-1)
        predicted_m = integer_expectation + 0.1 * fractional_expectation
        meter_loss = F.smooth_l1_loss(
            predicted_m,
            targets_m,
            reduction="mean",
            beta=self.depth_meter_loss_beta_m,
        )
        meter_mae = torch.mean(torch.abs(predicted_m - targets_m))
        if not bool(torch.isfinite(meter_loss).item()) or not bool(
            torch.isfinite(meter_mae).item()
        ):
            raise RuntimeError(
                f"meter loss/MAE appears NaN or Inf: loss={meter_loss}, mae={meter_mae}"
            )
        return meter_loss, meter_mae

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        camera_frame_indices: Optional[torch.Tensor] = None,
        camera_source_grid_thw: Optional[torch.Tensor] = None,
        has_pose: Optional[torch.Tensor] = None,
        depth_maps: Optional[torch.Tensor] = None,
        depth_meter_integer_positions: Optional[torch.Tensor] = None,
        depth_meter_fractional_positions: Optional[torch.Tensor] = None,
        depth_meter_targets_m: Optional[torch.Tensor] = None,
        depth_pair_relation_labels: Optional[torch.Tensor] = None,
        depth_pair_caption_labels: Optional[torch.Tensor] = None,
        depth_pair_relation_eligible: Optional[torch.Tensor] = None,
        source_window_size: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.camera_interpreter is not None:
            if has_pose is None or int(has_pose.numel()) != 1:
                raise RuntimeError(
                    "The Stage1 forward that includes Camera Interpreter must pass has_pose with batch_size=1."
                )
            has_pose_value = int(has_pose.reshape(-1)[0].item())
            if has_pose_value not in (0, 1):
                raise RuntimeError(
                    f"has_pose can only be 0/1, current={has_pose_value}"
                )
            if has_pose_value and (
                image_grid_thw is None or extrinsics is None or intrinsics is None
            ):
                raise RuntimeError(
                    "Camera Interpreter forward requires image_grid_thw / extrinsics / intrinsics; pose-only also requires image_grid_thw to preserve per-patch geometry grid."
                )

        embedding_layer = (
            self.qwen_model.model.language_model.embed_tokens
            if self.qwen3_vl
            else self.qwen_model.model.embed_tokens
        )
        inputs_embeds = embedding_layer(input_ids)
        if self.llm_use_image_tokens and not self.qwen3_vl:
            inputs_embeds = self._replace_image_tokens(
                input_ids,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
                camera_source_grid_thw=camera_source_grid_thw,
            )
        if self.camera_interpreter is not None and has_pose_value:
            inputs_embeds = self._inject_camera_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                image_sizes=image_sizes,
                camera_frame_indices=camera_frame_indices,
                camera_source_grid_thw=camera_source_grid_thw,
            )
        elif self.camera_interpreter is not None:
            # Depth rows without pose are not placed in C_i; checking prevents empty Camera slots from sneaking into training.
            if bool((input_ids == self.camera_token_id).any().item()):
                raise RuntimeError(
                    "When has_pose=0, no <|camera_*> tokens are allowed in the input sequence."
                )
        if self.depth_interpreter is not None and depth_maps is not None:
            inputs_embeds = self._inject_depth_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                depth_maps=depth_maps,
                image_sizes=image_sizes,
            )
        if self.depth_patch_tokenizer is not None:
            if image_grid_thw is None or depth_maps is None or image_sizes is None:
                raise RuntimeError(
                    "DepthPatchTokenizer forward requires image_grid_thw / depth_maps / image_sizes, and cannot return padded embeddings."
                )
            inputs_embeds = self._inject_depth_patch_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                depth_maps=depth_maps,
                image_sizes=image_sizes,
            )
            position_ids = self._build_rgb_aligned_depth_position_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
            )
        else:
            position_ids = None
        # It is safe to pass input_ids + inputs_embeds simultaneously to the outer Qwen2_5_VLForConditionalGeneration.forward():
        #   1. Outer layer sees inputs_embeds is not None → Skip embed_tokens lookup (no redundant embedding)
        #   2. Outer layer uses input_ids to call get_rope_index() to compute M-RoPE position_ids (needs to know image token boundaries)
        #   3. Inner self.model() receives input_ids=None → XOR checksum passed
        # pixel_values=None skips the outer visual encoder (which has already been handled in _replace_image_tokens)
        qwen_labels = (
            None
            if (
                self.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA
                or self.depth_perframe_caption_microbatch
                or self.camera_pair_rows
            )
            else labels
        )
        if self.qwen3_vl:
            if (
                not self.llm_use_image_tokens
                or image_grid_thw is None
                or pixel_values is None
            ):
                raise RuntimeError(
                    "Qwen3-VL Stage1 requires RGB, image_grid_thw, and pixel_values."
                )
            image_token_id = int(self.qwen_model.config.image_token_id)
            mm_token_type_ids = (input_ids == image_token_id).long()
            position_ids, _ = self.qwen_model.model.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            if self.camera_pair_rows:
                if camera_source_grid_thw is None:
                    raise RuntimeError(
                        "Qwen3 Camera pair missing original window RGB grid."
                    )
                raw_counts = camera_source_grid_thw.prod(-1).tolist()
                if sum(raw_counts) != int(pixel_values.shape[0]):
                    raise RuntimeError(
                        "Qwen3 Camera pair original window pixel patch count does not match grid."
                    )
                raw_views = torch.split(pixel_values, raw_counts, dim=0)
                # Use the native Qwen3 visual pathway and reorder DeepStack features with the target views.
                pixel_values = torch.cat(
                    [
                        part
                        for target in range(1, len(raw_views))
                        for part in (raw_views[0], raw_views[target])
                    ],
                    dim=0,
                )
            # Qwen3 does not accept both input_ids and inputs_embeds simultaneously; use original IDs to compute M-RoPE first.
            # Inject RGB and DeepStack into the native forward, preserving gradients for Camera/Depth tokens.
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=qwen_labels,
                use_cache=False,
                return_dict=True,
            )
        else:
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                labels=qwen_labels,
                pixel_values=None,
                image_grid_thw=image_grid_thw if self.llm_use_image_tokens else None,
                use_cache=False,
                return_dict=True,
            )
        if self.camera_pair_rows:
            if labels is None or source_window_size is None:
                raise RuntimeError(
                    "Camera pair forward missing labels/source_window_size."
                )
            views = int(source_window_size.reshape(-1)[0].item())
            if (
                int(source_window_size.numel()) != 1
                or int(outputs.logits.shape[0]) != views - 1
            ):
                raise RuntimeError(
                    "Camera pair Qwen batch must equal the original Window frame count minus one."
                )
            pair_losses, token_counts = self._masked_causal_mean_ce(
                outputs.logits, labels, component_name="camera_pair_motion"
            )
            if bool((token_counts <= 0).any().item()):
                raise RuntimeError(
                    f"Camera pair with empty motion supervision row: {token_counts.tolist()}"
                )
            # Give each target frame equal loss weight.
            window_loss = pair_losses.mean()
            outputs["loss"] = window_loss
            outputs["camera_pair_loss"] = window_loss.detach()
            outputs["pair_count"] = pair_losses.new_tensor(float(views - 1))
            return outputs
        if self.depth_perframe_caption_microbatch:
            if labels is None or source_window_size is None:
                raise RuntimeError(
                    "Frame-by-frame Caption forward is missing labels/source_window_size."
                )
            window_sizes = source_window_size.reshape(-1)
            if int(window_sizes.numel()) != 1:
                raise RuntimeError(
                    f"Each dataloader item must contain exactly one source Window: source_window_size={window_sizes.tolist()}"
                )
            num_frames = int(window_sizes[0].item())
            if num_frames <= 0 or num_frames != int(outputs.logits.shape[0]):
                raise RuntimeError(
                    f"source Window frame count and Qwen batch are inconsistent: V={num_frames}, logits_batch={outputs.logits.shape[0]}"
                )
            frame_losses, frame_token_counts = self._masked_causal_mean_ce(
                outputs.logits,
                labels,
                component_name="frame_caption",
            )
            valid_rows = frame_token_counts > 0
            if not bool(valid_rows.any().item()) or (
                self.depth_loss_mode != DEPTH_LOSS_MODE_NUMBER_ONLY
                and not bool(valid_rows.all().item())
            ):
                raise RuntimeError(
                    f"Frame-by-frame Caption: Current Window has no available supervised token: counts={frame_token_counts.tolist()}"
                )
            # number_only skips lines with "no significant objects" (zero numeric rows); for other frames, perform independent CE first, then average with equal weights.
            window_loss = frame_losses[valid_rows].mean()
            if not bool(torch.isfinite(window_loss).item()):
                raise RuntimeError(
                    f"NaN/Inf appears in frame-by-frame Caption Window loss: {window_loss}"
                )
            outputs["loss"] = window_loss
            outputs["frame_caption_loss"] = window_loss.detach()
            outputs["frame_count"] = frame_losses.new_tensor(float(num_frames))
            return outputs
        if self.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
            if (
                labels is None
                or depth_pair_relation_labels is None
                or depth_pair_caption_labels is None
                or depth_pair_relation_eligible is None
            ):
                raise RuntimeError(
                    "caption_pair_delta forward missing labels/relation labels/caption labels/eligible."
                )
            if not torch.equal(labels, depth_pair_caption_labels):
                raise RuntimeError(
                    "caption_pair_delta generic labels must match caption labels item by item."
                )
            relation_per_sample, relation_token_counts = self._masked_causal_mean_ce(
                outputs.logits,
                depth_pair_relation_labels,
                component_name="relation",
            )
            caption_per_sample, caption_token_counts = self._masked_causal_mean_ce(
                outputs.logits,
                depth_pair_caption_labels,
                component_name="caption",
            )
            eligible = depth_pair_relation_eligible.reshape(-1).bool()
            if int(eligible.numel()) != int(outputs.logits.shape[0]):
                raise RuntimeError(
                    f"caption_pair_delta eligible count and batch size do not match: eligible={tuple(eligible.shape)}, logits={tuple(outputs.logits.shape)}"
                )
            if bool((caption_token_counts <= 0).any().item()):
                raise RuntimeError(
                    f"caption_pair_delta each sample must have Caption token: counts={caption_token_counts.tolist()}"
                )
            if bool(((relation_token_counts > 0) != eligible).any().item()):
                raise RuntimeError(
                    f"caption_pair_delta relation token and eligible are inconsistent: counts={relation_token_counts.tolist()}, eligible={eligible.tolist()}"
                )
            weight = self.depth_relation_loss_weight
            total_per_sample = torch.where(
                eligible,
                weight * relation_per_sample + (1.0 - weight) * caption_per_sample,
                caption_per_sample,
            )
            total_loss = total_per_sample.mean()
            if not bool(torch.isfinite(total_loss).item()):
                raise RuntimeError(
                    f"caption_pair_delta total loss appears NaN/Inf: {total_loss}"
                )
            zero = caption_per_sample.sum() * 0.0
            relation_loss = (
                relation_per_sample[eligible].mean()
                if bool(eligible.any().item())
                else zero
            )
            outputs["loss"] = total_loss
            outputs["relation_loss"] = relation_loss
            outputs["caption_loss"] = caption_per_sample.mean()
            outputs["relation_eligible_fraction"] = eligible.float().mean()
            return outputs
        if self.depth_meter_loss_weight == 0.0:
            return outputs
        if outputs.loss is None:
            raise RuntimeError(
                "When meter loss is enabled, Qwen forward must return number_only CE loss."
            )
        if (
            depth_meter_integer_positions is None
            or depth_meter_fractional_positions is None
            or depth_meter_targets_m is None
        ):
            raise RuntimeError(
                "When meter loss is enabled, the batch lacks numeric positions or metric ground truth."
            )
        meter_loss, meter_mae = self._compute_depth_meter_components(
            outputs.logits,
            depth_meter_integer_positions,
            depth_meter_fractional_positions,
            depth_meter_targets_m,
        )
        ce_loss = outputs.loss
        total_loss = ce_loss + self.depth_meter_loss_weight * meter_loss
        # Components remain in ModelOutput for the Trainer to perform in-window aggregation; only total_loss participates in optimization.
        outputs["loss"] = total_loss
        outputs["ce_loss"] = ce_loss
        outputs["meter_loss"] = meter_loss
        outputs["meter_mae"] = meter_mae.detach()
        return outputs

    def save_camera_interpreter(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        if self.camera_interpreter is not None and not self.freeze_camera_interpreter:
            payload = {
                "camera_interpreter": self.camera_interpreter.state_dict(),
                "camera_interpreter_config": self.camera_interpreter.config.to_dict(),
                "camera_token_id": self.camera_token_id,
                "base_model_name_or_path": self.base_model_name_or_path,
            }
            if self.depth_interpreter is not None:
                payload["depth_interpreter"] = self.depth_interpreter.state_dict()
                payload["depth_interpreter_config"] = (
                    self.depth_interpreter.config.to_dict()
                )
                payload["depth_token_id"] = self.depth_token_id
            torch.save(payload, os.path.join(output_dir, "camera_interpreter.pt"))
            with open(
                os.path.join(output_dir, "camera_interpreter_config.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    payload["camera_interpreter_config"],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        if self.depth_interpreter is not None:
            depth_payload = {
                "depth_interpreter": self.depth_interpreter.state_dict(),
                "depth_interpreter_config": self.depth_interpreter.config.to_dict(),
                "depth_token_id": self.depth_token_id,
                "base_model_name_or_path": self.base_model_name_or_path,
            }
            torch.save(depth_payload, os.path.join(output_dir, "depth_interpreter.pt"))
            with open(
                os.path.join(output_dir, "depth_interpreter_config.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    depth_payload["depth_interpreter_config"],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        if self.depth_patch_tokenizer is not None:
            patch_payload = {
                "depth_patch_tokenizer": self.depth_patch_tokenizer.state_dict(),
                "depth_patch_tokenizer_config": self.depth_patch_tokenizer.config.to_dict(),
                "depth_token_id": self.depth_token_id,
                "base_model_name_or_path": self.base_model_name_or_path,
            }
            torch.save(
                patch_payload, os.path.join(output_dir, "depth_patch_tokenizer.pt")
            )
            with open(
                os.path.join(output_dir, "depth_patch_tokenizer_config.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    patch_payload["depth_patch_tokenizer_config"],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    def load_camera_interpreter(self, checkpoint_dir: str):
        if self.freeze_camera_interpreter:
            # Reload the trainable Depth Reader while keeping the Camera Reader fixed.
            if self.depth_interpreter is None:
                raise RuntimeError(
                    "B strict reload lacks a trainable Depth Interpreter."
                )
            depth_path = os.path.join(checkpoint_dir, "depth_interpreter.pt")
            if not os.path.isfile(depth_path):
                raise FileNotFoundError(
                    f"B checkpoint missing Depth Interpreter: {depth_path}"
                )
            depth_payload = torch.load(depth_path, map_location="cpu")
            self._load_depth_interpreter_payload(
                payload=depth_payload,
                weight_path=depth_path,
                config_path=os.path.join(
                    checkpoint_dir, "depth_interpreter_config.json"
                ),
            )
            return
        if self.camera_interpreter is not None:
            path = os.path.join(checkpoint_dir, "camera_interpreter.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Cannot find Camera Interpreter checkpoint: {path}"
                )
            payload = torch.load(path, map_location="cpu")
            self._load_camera_interpreter_payload(
                payload=payload,
                weight_path=path,
                config_path=os.path.join(
                    checkpoint_dir, "camera_interpreter_config.json"
                ),
            )
            if self.depth_interpreter is not None and "depth_interpreter" in payload:
                self._load_depth_interpreter_payload(
                    payload=payload,
                    weight_path=path,
                    config_path=os.path.join(
                        checkpoint_dir, "depth_interpreter_config.json"
                    ),
                )
                return
        if self.depth_interpreter is not None:
            depth_path = os.path.join(checkpoint_dir, "depth_interpreter.pt")
            if not os.path.isfile(depth_path):
                raise FileNotFoundError(
                    f"Cannot find Depth Interpreter checkpoint: {depth_path}"
                )
            depth_payload = torch.load(depth_path, map_location="cpu")
            self._load_depth_interpreter_payload(
                payload=depth_payload,
                weight_path=depth_path,
                config_path=os.path.join(
                    checkpoint_dir, "depth_interpreter_config.json"
                ),
            )
        if self.depth_patch_tokenizer is not None:
            patch_path = os.path.join(checkpoint_dir, "depth_patch_tokenizer.pt")
            if not os.path.isfile(patch_path):
                raise FileNotFoundError(
                    f"Cannot find DepthPatchTokenizer checkpoint: {patch_path}"
                )
            patch_payload = torch.load(patch_path, map_location="cpu")
            self._load_depth_patch_tokenizer_payload(
                payload=patch_payload,
                weight_path=patch_path,
                config_path=os.path.join(
                    checkpoint_dir, "depth_patch_tokenizer_config.json"
                ),
            )

    def load_frozen_camera_interpreter(self, checkpoint_dir: str):
        """Strictly load B's Camera weights and reconfirm that all Camera parameters are frozen."""
        if self.camera_interpreter is None or not self.freeze_camera_interpreter:
            raise RuntimeError(
                "Only B's frozen Camera mode can call load_frozen_camera_interpreter."
            )
        path = os.path.join(checkpoint_dir, "camera_interpreter.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot find frozen Camera checkpoint: {path}")
        payload = torch.load(path, map_location="cpu")
        self._load_camera_interpreter_payload(
            payload=payload,
            weight_path=path,
            config_path=os.path.join(checkpoint_dir, "camera_interpreter_config.json"),
        )
        for name, parameter in self.camera_interpreter.named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(
                    f"Frozen Camera parameters unexpectedly trainable: {name}"
                )
        self.camera_interpreter.eval()

    def _load_camera_interpreter_payload(
        self,
        *,
        payload: dict,
        weight_path: str,
        config_path: str,
    ) -> None:
        """Strictly verify Camera JSON, weight payload, and current model config before loading weights."""
        if self.camera_interpreter is None:
            raise RuntimeError(
                "The current model has no Camera Interpreter and cannot load the Camera checkpoint."
            )
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Camera checkpoint missing config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        saved_config = parse_camera_interpreter_config_dict(config_dict, config_path)
        expected_config = self.camera_interpreter.config.to_dict()
        if saved_config.to_dict() != expected_config:
            raise RuntimeError(
                f"Camera checkpoint config inconsistent with current model: saved={saved_config.to_dict()}, expected={expected_config}, path={config_path}"
            )
        payload_config = payload.get("camera_interpreter_config")
        if payload_config is None:
            raise RuntimeError(
                f"Camera checkpoint payload is missing camera_interpreter_config: {weight_path}"
            )
        normalized_payload_config = parse_camera_interpreter_config_dict(
            payload_config,
            f"{weight_path}::camera_interpreter_config",
        )
        if normalized_payload_config.to_dict() != saved_config.to_dict():
            raise RuntimeError(
                f"Camera checkpoint payload config and JSON are inconsistent: payload={normalized_payload_config.to_dict()}, json={saved_config.to_dict()}, path={weight_path}"
            )
        camera_state = payload.get("camera_interpreter")
        if camera_state is None:
            raise RuntimeError(
                f"checkpoint missing camera_interpreter state_dict: {weight_path}"
            )
        self.camera_interpreter.load_state_dict(camera_state, strict=True)

    def _load_depth_interpreter_payload(
        self,
        *,
        payload: dict,
        weight_path: str,
        config_path: str,
    ) -> None:
        """Strictly verify Depth JSON, weight payload, and current model config before loading weights."""
        if self.depth_interpreter is None:
            raise RuntimeError(
                "The current model does not have a Depth Interpreter and cannot load the Depth checkpoint."
            )
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Depth checkpoint missing config: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        saved_config = parse_depth_interpreter_config_dict(config_dict, config_path)
        expected_config = self.depth_interpreter.config.to_dict()
        if saved_config.to_dict() != expected_config:
            raise RuntimeError(
                f"Depth checkpoint config inconsistent with current model: saved={saved_config.to_dict()}, expected={expected_config}, path={config_path}"
            )
        payload_config = payload.get("depth_interpreter_config")
        if payload_config is None:
            raise RuntimeError(
                f"Depth checkpoint payload is missing depth_interpreter_config: {weight_path}"
            )
        normalized_payload_config = parse_depth_interpreter_config_dict(
            payload_config,
            f"{weight_path}::depth_interpreter_config",
        )
        if normalized_payload_config.to_dict() != saved_config.to_dict():
            raise RuntimeError(
                f"Depth checkpoint payload config and JSON are inconsistent: payload={normalized_payload_config.to_dict()}, json={saved_config.to_dict()}, path={weight_path}"
            )
        depth_state = payload.get("depth_interpreter")
        if depth_state is None:
            raise RuntimeError(
                f"checkpoint missing depth_interpreter state_dict: {weight_path}"
            )
        # strict=True will block missing keys, extra keys, and 12D/52D shape drift directly.
        self.depth_interpreter.load_state_dict(depth_state, strict=True)

    def _load_depth_patch_tokenizer_payload(
        self,
        *,
        payload: dict,
        weight_path: str,
        config_path: str,
    ) -> None:
        """Strictly verify Direct Patch Tokenizer JSON, payload, and current contract construction."""
        if self.depth_patch_tokenizer is None:
            raise RuntimeError(
                "The current model does not have DepthPatchTokenizer and cannot load the direct patch checkpoint."
            )
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"DepthPatchTokenizer checkpoint missing config: {config_path}"
            )
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        saved_config = parse_depth_patch_tokenizer_config_dict(config_dict, config_path)
        expected_config = self.depth_patch_tokenizer.config.to_dict()
        if saved_config.to_dict() != expected_config:
            raise RuntimeError(
                f"DepthPatchTokenizer checkpoint config does not match current model: saved={saved_config.to_dict()}, expected={expected_config}, path={config_path}"
            )
        payload_config = payload.get("depth_patch_tokenizer_config")
        if payload_config is None:
            raise RuntimeError(
                f"DepthPatchTokenizer checkpoint payload missing config: {weight_path}"
            )
        normalized_payload_config = parse_depth_patch_tokenizer_config_dict(
            payload_config,
            f"{weight_path}::depth_patch_tokenizer_config",
        )
        if normalized_payload_config.to_dict() != saved_config.to_dict():
            raise RuntimeError(
                f"DepthPatchTokenizer payload config and JSON are inconsistent: payload={normalized_payload_config.to_dict()}, json={saved_config.to_dict()}"
            )
        patch_state = payload.get("depth_patch_tokenizer")
        if patch_state is None:
            raise RuntimeError(
                f"checkpoint missing depth_patch_tokenizer state_dict: {weight_path}"
            )
        self.depth_patch_tokenizer.load_state_dict(patch_state, strict=True)

    def stage1_artifact_names(self) -> tuple[str, str]:
        """Return the filename of the weights and configuration file that the current Stage1 branch should save."""
        if self.camera_interpreter is not None and not self.freeze_camera_interpreter:
            return "camera_interpreter.pt", "camera_interpreter_config.json"
        if self.depth_interpreter is not None:
            return "depth_interpreter.pt", "depth_interpreter_config.json"
        if self.depth_patch_tokenizer is not None:
            return "depth_patch_tokenizer.pt", "depth_patch_tokenizer_config.json"
        raise RuntimeError("Stage1 model has no savable Camera/Depth modules.")


def _init_new_special_token_embeddings(model, tokenizer, tokens: List[str]):
    """Initialize the newly added special token with the mean of the original vocabulary to avoid inheriting strong semantics like EOS."""
    input_emb = model.get_input_embeddings().weight
    output_emb = (
        model.get_output_embeddings().weight
        if model.get_output_embeddings() is not None
        else None
    )
    token_ids = [_single_token_id(tokenizer, token) for token in tokens]
    original_vocab_size = min(int(token_id) for token_id in token_ids)
    if original_vocab_size <= 0:
        raise RuntimeError(
            f"Cannot infer original vocabulary size from added token: token_ids={token_ids}, original_vocab_size={original_vocab_size}"
        )
    if max(int(token_id) for token_id in token_ids) >= int(input_emb.shape[0]):
        raise RuntimeError(
            f"Added token id exceeds input embedding size: token_ids={token_ids}, input_vocab={input_emb.shape[0]}"
        )
    if output_emb is not None and max(int(token_id) for token_id in token_ids) >= int(
        output_emb.shape[0]
    ):
        raise RuntimeError(
            f"Added token id exceeds output embedding size: token_ids={token_ids}, output_vocab={output_emb.shape[0]}"
        )
    with torch.no_grad():
        # Use the mean of the original vocabulary as a neutral placeholder, and do not include newly added tokens from the resized output in the mean.
        mean_input = input_emb[:original_vocab_size].mean(dim=0)
        mean_output = (
            output_emb[:original_vocab_size].mean(dim=0)
            if output_emb is not None
            else None
        )
        for token_id in token_ids:
            input_emb[token_id].copy_(mean_input)
            if output_emb is not None:
                output_emb[token_id].copy_(mean_output)


def main():
    assert_cpu_affinity_for_training()
    args = get_args()
    if args.omit_user_question == "1":
        # Ablation only removes fixed user text; data, caption, motion_content mask, and token positions remain unchanged.
        if args.prompt_source != "fixed":
            raise RuntimeError(
                "--omit_user_question=1 allows only prompt_source=fixed."
            )
        args.user_prompt = ""
    seed_everything(42)
    use_depth = args.use_depth_interpreter == "1"
    use_camera = args.use_camera_interpreter == "1"
    if args.camera_pair_rows == "1" and (
        not use_camera
        or use_depth
        or args.camera_token_layout != "per_frame"
        or int(args.camera_latent_size) != 1
        or args.camera_loss_mode != CAMERA_LOSS_MODE_MOTION_CONTENT
        or args.camera_frame_index_encoding != "fixed_31"
        or args.camera_token_position != "before_question"
        or args.llm_use_image_tokens != "1"
        or args.interpreter_use_image_tokens != "0"
    ):
        raise RuntimeError(
            "Camera pairing lines must use Camera-only/per_frame K1/motion_content/fixed_31/RGB Qwen/Geometry Interpreter."
        )
    depth_camera_caption_control = args.depth_camera_caption_control == "1"
    use_depth_patch_tokenizer = (
        args.depth_stage1_output_contract == DEPTH_STAGE1_OUTPUT_CONTRACT_DIRECT_PATCH
    )
    depth_perframe_caption_microbatch = (
        args.depth_frame_microbatch_mode == DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS
    )
    if not use_camera and not use_depth:
        raise RuntimeError(
            "--use_camera_interpreter=0 and --use_depth_interpreter=0: At least one interpreter must be enabled."
        )
    if args.camera_loss_mode == CAMERA_LOSS_MODE_MOTION_CONTENT and (
        not use_camera or use_depth
    ):
        raise RuntimeError(
            "motion_content only allows Camera-only Stage1: --use_camera_interpreter=1 and --use_depth_interpreter=0."
        )
    if depth_perframe_caption_microbatch:
        # This is an indivisible training semantic package; any drift will alter token visibility range, frame identity, or window weights.
        actual_contract = {
            "use_depth": use_depth,
            "use_camera": use_camera,
            "depth_stage1_output_contract": args.depth_stage1_output_contract,
            "depth_encoder_type": args.depth_encoder_type,
            "depth_token_layout": args.depth_token_layout,
            "depth_latent_size": int(args.depth_latent_size),
            "depth_token_position": args.depth_token_position,
            "depth_loss_mode": args.depth_loss_mode,
            "depth_cot_version": args.depth_cot_version,
            "prompt_source": args.prompt_source,
            "depth_frame_supervision_scope": args.depth_frame_supervision_scope,
            "depth_frame_microbatch_mode": args.depth_frame_microbatch_mode,
            "depth_frame_loss_reduction": args.depth_frame_loss_reduction,
            "llm_use_image_tokens": args.llm_use_image_tokens,
            "interpreter_use_image_tokens": args.interpreter_use_image_tokens,
            "depth_use_geometry_attention_bias": args.depth_use_geometry_attention_bias,
            "output_contract": args.output_contract,
            "output_gate_type": args.output_gate_type,
            "depth_meter_loss_weight": float(args.depth_meter_loss_weight),
            "depth_relation_loss_weight": float(args.depth_relation_loss_weight),
            "omit_user_question": args.omit_user_question,
        }
        expected_contract = {
            "use_depth": True,
            "use_camera": False,
            "depth_stage1_output_contract": DEPTH_STAGE1_OUTPUT_CONTRACT_SUMMARY,
            "depth_encoder_type": "scalar_5d",
            "depth_token_layout": "per_frame",
            "depth_latent_size": 1,
            "depth_token_position": "before_question",
            "depth_loss_mode": args.depth_loss_mode,
            "depth_cot_version": DEPTH_PERFRAME_CAPTION_MICROBATCH_COT_VERSION,
            "prompt_source": args.prompt_source,
            "depth_frame_supervision_scope": DEPTH_FRAME_SUPERVISION_SCOPE_OWN,
            "depth_frame_microbatch_mode": DEPTH_FRAME_MICROBATCH_MODE_WINDOW_TO_ROWS,
            "depth_frame_loss_reduction": DEPTH_FRAME_LOSS_REDUCTION_WINDOW_MEAN,
            "llm_use_image_tokens": "1",
            "interpreter_use_image_tokens": "0",
            "depth_use_geometry_attention_bias": "0",
            "output_contract": "gated_delta",
            "output_gate_type": "scalar",
            "depth_meter_loss_weight": 0.0,
            "depth_relation_loss_weight": 0.5,
            "omit_user_question": "0",
        }
        mismatches = {
            key: (actual_contract[key], expected)
            for key, expected in expected_contract.items()
            if actual_contract[key] != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Frame-by-frame Caption micro-batch violates frozen Design parameter contract: mismatches={mismatches}"
            )
        depth_from_scratch = os.environ.get("DEPTH_FROM_SCRATCH", "0") == "1"
        if depth_from_scratch and args.init_interpreter_checkpoint:
            raise RuntimeError(
                "Depth training from scratch cannot also initialize from Reader weights."
            )
        if (
            not depth_from_scratch
            and not args.init_interpreter_checkpoint
            and not args.resume_from_checkpoint
        ):
            raise RuntimeError(
                "Frame-wise caption training requires Reader weights or DEPTH_FROM_SCRATCH=1."
            )
        if args.prompt_source == "none":
            args.user_prompt = ""
        elif not args.user_prompt.strip():
            raise RuntimeError("The frame-by-frame fixed prompt cannot be empty.")
    else:
        default_frame_contract = (
            DEPTH_FRAME_SUPERVISION_SCOPE_WINDOW,
            DEPTH_FRAME_MICROBATCH_MODE_DISABLED,
            DEPTH_FRAME_LOSS_REDUCTION_TOKEN_MEAN,
        )
        actual_frame_contract = (
            args.depth_frame_supervision_scope,
            args.depth_frame_microbatch_mode,
            args.depth_frame_loss_reduction,
        )
        if actual_frame_contract != default_frame_contract:
            raise RuntimeError(
                f"Non-frame-by-frame caption training cannot independently modify frame contracts: actual={actual_frame_contract}, expected={default_frame_contract}"
            )
        if args.prompt_source == "none":
            raise RuntimeError(
                "prompt_source=none allows only frame-by-frame Caption micro-batch."
            )
    if args.depth_loss_mode in {
        DEPTH_LOSS_MODE_NUMBER_ONLY,
        DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA,
    } and (not use_depth or (use_camera and not depth_camera_caption_control)):
        raise RuntimeError(
            "Depth local loss is only allowed for Depth-only Stage1: --use_depth_interpreter=1 and --use_camera_interpreter=0."
        )
    if depth_camera_caption_control:
        # B's only studied variable is the appearance of frozen C_i in the same Forward; all other joint Stage1 combinations are rejected.
        control_contract = {
            "use_camera": use_camera,
            "use_depth": use_depth,
            "frozen_camera_checkpoint": bool(args.frozen_camera_checkpoint),
            "freeze_camera_interpreter": args.freeze_camera_interpreter,
            "camera_latent_size": int(args.camera_latent_size),
            "camera_token_layout": args.camera_token_layout,
            "camera_token_position": args.camera_token_position,
            "camera_translation_encoding": args.camera_translation_encoding,
            "camera_use_geometry_attention_bias": args.camera_use_geometry_attention_bias,
            "depth_latent_size": int(args.depth_latent_size),
            "depth_token_layout": args.depth_token_layout,
            "depth_token_position": args.depth_token_position,
            "depth_encoder_type": args.depth_encoder_type,
            "depth_loss_mode": args.depth_loss_mode,
            "camera_loss_mode": args.camera_loss_mode,
            "prompt_source": args.prompt_source,
            "depth_meter_loss_weight": float(args.depth_meter_loss_weight),
            "depth_relation_loss_weight": float(args.depth_relation_loss_weight),
            "llm_use_image_tokens": args.llm_use_image_tokens,
            "interpreter_use_image_tokens": args.interpreter_use_image_tokens,
            "output_contract": args.output_contract,
            "output_gate_type": args.output_gate_type,
            "depth_use_geometry_attention_bias": args.depth_use_geometry_attention_bias,
        }
        expected_control_contract = {
            "use_camera": True,
            "use_depth": True,
            "frozen_camera_checkpoint": True,
            "freeze_camera_interpreter": "1",
            "camera_latent_size": 1,
            "camera_token_layout": "per_frame",
            "camera_token_position": "before_question",
            "camera_translation_encoding": "raw_meters_planar",
            "depth_latent_size": 1,
            "camera_use_geometry_attention_bias": "0",
            "depth_token_layout": "per_frame",
            "depth_token_position": "before_question",
            "depth_encoder_type": "scalar_5d",
            "depth_loss_mode": DEPTH_LOSS_MODE_NUMBER_ONLY,
            "camera_loss_mode": CAMERA_LOSS_MODE_FULL_CAPTION,
            "prompt_source": "fixed",
            "depth_meter_loss_weight": 0.0,
            "depth_relation_loss_weight": 0.5,
            "llm_use_image_tokens": "1",
            "interpreter_use_image_tokens": "0",
            "output_contract": "gated_delta",
            "output_gate_type": "scalar",
            "depth_use_geometry_attention_bias": "0",
        }
        mismatches = {
            key: (control_contract[key], value)
            for key, value in expected_control_contract.items()
            if control_contract[key] != value
        }
        if mismatches:
            raise RuntimeError(
                f"B frozen Camera Caption control group parameters contract not satisfied: {mismatches}"
            )
        args.frozen_camera_checkpoint = os.path.abspath(
            os.path.expanduser(args.frozen_camera_checkpoint)
        )
        if not os.path.isdir(args.frozen_camera_checkpoint):
            raise FileNotFoundError(
                f"B frozen Camera checkpoint does not exist: {args.frozen_camera_checkpoint}"
            )
    if args.init_interpreter_checkpoint and args.resume_from_checkpoint:
        raise RuntimeError(
            "--init_interpreter_checkpoint and --resume_from_checkpoint have different semantics and cannot be set simultaneously."
        )
    if use_depth_patch_tokenizer:
        # Freeze Design: Direct Patch only tests scalar_5d -> full token + RGB-MRoPE
        # Any summary query, RGB reader, Gate, base add, or other geometry encoding is directly rejected.
        required_values = {
            "use_depth": use_depth,
            "use_camera": use_camera,
            "depth_encoder_type": args.depth_encoder_type,
            "depth_token_layout": args.depth_token_layout,
            "depth_latent_size": int(args.depth_latent_size),
            "depth_token_position": args.depth_token_position,
            "output_contract": args.output_contract,
            "output_gate_type": args.output_gate_type,
            "depth_use_geometry_attention_bias": args.depth_use_geometry_attention_bias,
            "interpreter_use_image_tokens": args.interpreter_use_image_tokens,
            "llm_use_image_tokens": args.llm_use_image_tokens,
            "depth_loss_mode": args.depth_loss_mode,
        }
        expected_values = {
            "use_depth": True,
            "use_camera": False,
            "depth_encoder_type": "scalar_5d",
            "depth_token_layout": "per_frame",
            "depth_latent_size": 1,
            "depth_token_position": "before_question",
            "output_contract": OUTPUT_CONTRACT_DIRECT_EMBEDDING,
            "output_gate_type": "none",
            "depth_use_geometry_attention_bias": "0",
            "interpreter_use_image_tokens": "0",
            "llm_use_image_tokens": "1",
            "depth_loss_mode": DEPTH_LOSS_MODE_NUMBER_ONLY,
        }
        mismatches = {
            key: (required_values[key], value)
            for key, value in expected_values.items()
            if required_values[key] != value
        }
        if mismatches:
            raise RuntimeError(
                f"direct_patch_embedding violates frozen Design parameter contract: mismatches={mismatches}"
            )
    if args.prompt_source == "record":
        if (
            use_camera
            or not use_depth
            or args.depth_loss_mode
            not in {
                DEPTH_LOSS_MODE_NUMBER_ONLY,
                DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA,
            }
        ):
            raise RuntimeError(
                "prompt_source=record allows only Depth-only + number_only/caption_pair_delta Stage1."
            )
        expected_record_version = {
            DEPTH_LOSS_MODE_NUMBER_ONLY: (DEPTH_OBJECT_QUERY_COT_VERSION,),
            DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA: (
                DEPTH_CAPTION_PAIR_DELTA_COT_VERSION,
                DEPTH_CAPTION_PAIR_DELTA_CROSSFRAME_COT_VERSION,
            ),
        }[args.depth_loss_mode]
        if args.depth_cot_version not in expected_record_version:
            raise RuntimeError(
                f"prompt_source=record's cot_version and loss mode do not match: actual={args.depth_cot_version!r}, expected={expected_record_version!r}"
            )
    selected_cot_version = (
        args.depth_cot_version
        if depth_camera_caption_control or not use_camera
        else args.camera_cot_version
    )
    if (
        args.output_contract == OUTPUT_CONTRACT_DIRECT_EMBEDDING
        and not str(selected_cot_version or "").strip()
    ):
        raise RuntimeError(
            "direct_embedding Stage1 must explicitly pass in the current branch's cot_version; it can no longer be inferred from loss_mode or directory name to determine the target text version."
        )
    if (
        not np.isfinite(args.depth_meter_loss_weight)
        or args.depth_meter_loss_weight < 0.0
    ):
        raise RuntimeError(
            f"--depth_meter_loss_weight must be finite and >=0: {args.depth_meter_loss_weight}"
        )
    if args.depth_meter_loss_weight > 0.0 and (
        not use_depth
        or use_camera
        or args.depth_loss_mode != DEPTH_LOSS_MODE_NUMBER_ONLY
    ):
        raise RuntimeError(
            "Positive --depth_meter_loss_weight only allows Depth-only + --depth_loss_mode=number_only."
        )
    if not np.isfinite(args.depth_relation_loss_weight):
        raise RuntimeError(
            f"--depth_relation_loss_weight must be finite: {args.depth_relation_loss_weight}"
        )
    if args.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
        if args.depth_relation_loss_weight != 0.5:
            raise RuntimeError(
                f"caption_pair_delta requires --depth_relation_loss_weight=0.5: actual={args.depth_relation_loss_weight}"
            )
        if args.depth_meter_loss_weight != 0.0:
            raise RuntimeError("caption_pair_delta does not allow stacking meter loss.")
        pair_contract = {
            "depth_token_layout": args.depth_token_layout,
            "depth_latent_size": int(args.depth_latent_size),
            "depth_token_position": args.depth_token_position,
            "depth_encoder_type": args.depth_encoder_type,
            "camera_num_layers": int(args.camera_num_layers),
            "camera_num_heads": int(args.camera_num_heads),
            "camera_geom_rank": int(args.camera_geom_rank),
            "depth_use_geometry_attention_bias": args.depth_use_geometry_attention_bias,
            "interpreter_use_image_tokens": args.interpreter_use_image_tokens,
            "llm_use_image_tokens": args.llm_use_image_tokens,
            "output_contract": args.output_contract,
            "output_gate_type": args.output_gate_type,
        }
        expected_pair_contract = {
            "depth_token_layout": "per_frame",
            "depth_latent_size": 1,
            "depth_token_position": "before_question",
            "depth_encoder_type": "scalar_5d",
            "camera_num_layers": 2,
            "camera_num_heads": 8,
            "camera_geom_rank": 32,
            "depth_use_geometry_attention_bias": "0",
            "interpreter_use_image_tokens": "0",
            "llm_use_image_tokens": "1",
            "output_contract": "gated_delta",
            "output_gate_type": "scalar",
        }
        mismatches = {
            key: (pair_contract[key], expected_value)
            for key, expected_value in expected_pair_contract.items()
            if pair_contract[key] != expected_value
        }
        if mismatches:
            raise RuntimeError(
                f"caption_pair_delta must replicate checkpoint-88560's Interpreter/input contract: mismatches={mismatches}"
            )
        if not args.init_interpreter_checkpoint:
            raise RuntimeError(
                "caption_pair_delta must explicitly pass --init_interpreter_checkpoint=checkpoint-88560; random initialization is not allowed."
            )
    expected_image_size = _parse_expected_image_size(args.expected_image_size)
    llm_use_image_tokens = args.llm_use_image_tokens == "1"
    interpreter_use_image_tokens = args.interpreter_use_image_tokens == "1"
    if interpreter_use_image_tokens and not llm_use_image_tokens:
        raise RuntimeError(
            "Current training does not support interpreter_use_image_tokens=1 and llm_use_image_tokens=0: The Interpreter's image tokens come from the image tokens in the LLM prompt; when the LLM does not process images, there are no reusable image tokens."
        )

    rank = int(os.environ.get("RANK", "0"))
    os.environ["HF_HOME"] = args.cache_dir
    if args.log_file is None:
        args.log_file = os.path.join(args.save_model_path, "log.txt")
    if rank == 0:
        os.makedirs(args.save_model_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(args.log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logging.info(args)

    processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    resolved_min_pixels, resolved_max_pixels = _set_image_processor_pixel_budget(
        processor.image_processor,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if rank == 0:
        print(f"[image_processor] min_pixels={resolved_min_pixels}")
        print(f"[image_processor] max_pixels={resolved_max_pixels}")
    all_new_tokens = []
    if use_camera:
        all_new_tokens.extend([CAMERA_START, CAMERA_PAD, CAMERA_END])
    if use_depth:
        all_new_tokens.extend([DEPTH_START, DEPTH_PAD, DEPTH_END])
    for token in all_new_tokens:
        processor.tokenizer.add_tokens(token, special_tokens=True)
    depth_digit_token_ids = None
    depth_decimal_token_id = None
    if args.depth_meter_loss_weight > 0.0:
        depth_digit_token_ids, depth_decimal_token_id = (
            _resolve_depth_meter_token_contract(processor.tokenizer)
        )

    model_type = AutoConfig.from_pretrained(
        args.model, cache_dir=args.cache_dir
    ).model_type
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
        model_class = Qwen3VLForConditionalGeneration
    elif model_type == "qwen3_5":
        # The model class, visual tower, and MRoPE of Qwen3.5 are provided via native interfaces; only the geometric Interpreter is reused.
        from transformers import Qwen3_5ForConditionalGeneration

        config = AutoConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
        model_class = Qwen3_5ForConditionalGeneration
    elif model_type == "qwen2_5_vl":
        config = Qwen2_5_VLConfig.from_pretrained(args.model, cache_dir=args.cache_dir)
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        raise RuntimeError(
            f"Stage1 does not support the backbone model_type: {model_type}"
        )
    config.stage = "stage1"
    config.latent_recur = False
    config.stage1_enable_feature_proj = False
    config.use_prototype_injection = False
    config.prototype_image_dropout = False

    qwen_model = model_class.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    )
    qwen_model.resize_token_embeddings(len(processor.tokenizer))
    if use_camera:
        qwen_model.config.camera_token_id = _single_token_id(
            processor.tokenizer, CAMERA_PAD
        )
        qwen_model.config.camera_start_id = _single_token_id(
            processor.tokenizer, CAMERA_START
        )
        qwen_model.config.camera_end_id = _single_token_id(
            processor.tokenizer, CAMERA_END
        )
    else:
        qwen_model.config.camera_token_id = -1
        qwen_model.config.camera_start_id = -1
        qwen_model.config.camera_end_id = -1
    _init_new_special_token_embeddings(
        qwen_model,
        processor.tokenizer,
        all_new_tokens,
    )

    vision_cfg = qwen_model.config.vision_config
    text_hidden_size = int(qwen_model.config.text_config.hidden_size)
    interpreter_cfg = None
    if use_camera:
        geometry_dim = geometry_dim_for_camera_translation_encoding(
            args.camera_translation_encoding
        )
        interpreter_cfg = CameraInterpreterConfig(
            hidden_size=text_hidden_size,
            num_camera_tokens=int(args.camera_latent_size),
            token_layout=str(args.camera_token_layout),
            num_layers=int(args.camera_num_layers),
            num_heads=int(args.camera_num_heads),
            dropout=float(args.camera_dropout),
            geometry_dim=int(geometry_dim),
            camera_translation_encoding=str(args.camera_translation_encoding),
            camera_frame_index_encoding=str(args.camera_frame_index_encoding),
            geometry_rank=int(args.camera_geom_rank),
            output_contract=str(args.output_contract),
            output_gate_type=str(args.output_gate_type),
            output_gate_init=float(args.camera_output_gate_init),
            geom_gate_init=float(args.camera_geom_gate_init),
            use_geometry_attention_bias=(
                args.camera_use_geometry_attention_bias == "1"
            ),
            patch_size=int(getattr(vision_cfg, "patch_size", 14)),
            # Use camera_merge_size when constructing the Camera Reader configuration.
            camera_merge_size=int(getattr(vision_cfg, "spatial_merge_size", 2)),
            use_image_tokens=interpreter_use_image_tokens,
            camera_token_position=str(args.camera_token_position),
        )
    depth_interpreter_cfg = None
    depth_patch_tokenizer_cfg = None
    depth_token_id = -1
    if use_depth:
        depth_geometry_dim = depth_geometry_dim_for_encoder_type(
            args.depth_encoder_type
        )
        if use_depth_patch_tokenizer:
            depth_patch_tokenizer_cfg = DepthPatchTokenizerConfig(
                hidden_size=text_hidden_size,
                depth_geometry_dim=int(depth_geometry_dim),
                depth_encoder_type=str(args.depth_encoder_type),
                patch_size=int(getattr(vision_cfg, "patch_size", 14)),
                spatial_merge_size=int(getattr(vision_cfg, "spatial_merge_size", 2)),
            )
        else:
            depth_interpreter_cfg = DepthInterpreterConfig(
                hidden_size=text_hidden_size,
                num_depth_tokens=int(args.depth_latent_size),
                token_layout=str(args.depth_token_layout),
                num_layers=int(args.camera_num_layers),
                num_heads=int(args.camera_num_heads),
                dropout=float(args.camera_dropout),
                depth_geometry_dim=int(depth_geometry_dim),
                depth_encoder_type=str(args.depth_encoder_type),
                geometry_rank=int(args.camera_geom_rank),
                output_contract=str(args.output_contract),
                output_gate_type=str(args.output_gate_type),
                output_gate_init=float(args.camera_output_gate_init),
                geom_gate_init=float(args.camera_geom_gate_init),
                # Apply the depth attention-bias setting only to the Depth Reader.
                use_geometry_attention_bias=(
                    args.depth_use_geometry_attention_bias == "1"
                ),
                patch_size=int(getattr(vision_cfg, "patch_size", 14)),
                # Use spatial_merge_size when constructing the Depth Reader configuration.
                spatial_merge_size=int(getattr(vision_cfg, "spatial_merge_size", 2)),
                use_image_tokens=interpreter_use_image_tokens,
                depth_token_position=str(args.depth_token_position),
            )
        depth_token_id = _single_token_id(processor.tokenizer, DEPTH_PAD)

    model = CameraInterpreterForCausalLM(
        qwen_model=qwen_model,
        interpreter_config=interpreter_cfg,
        camera_token_id=qwen_model.config.camera_token_id,
        base_model_name_or_path=args.model,
        visual_chunk_size=args.visual_chunk_size,
        depth_interpreter_config=depth_interpreter_cfg,
        depth_patch_tokenizer_config=depth_patch_tokenizer_cfg,
        depth_token_id=depth_token_id,
        llm_use_image_tokens=llm_use_image_tokens,
        interpreter_use_image_tokens=interpreter_use_image_tokens,
        camera_loss_mode=args.camera_loss_mode,
        camera_cot_version=args.camera_cot_version,
        depth_loss_mode=args.depth_loss_mode,
        depth_cot_version=args.depth_cot_version,
        depth_frame_supervision_scope=args.depth_frame_supervision_scope,
        depth_frame_microbatch_mode=args.depth_frame_microbatch_mode,
        depth_frame_loss_reduction=args.depth_frame_loss_reduction,
        depth_meter_loss_weight=args.depth_meter_loss_weight,
        depth_relation_loss_weight=args.depth_relation_loss_weight,
        depth_digit_token_ids=depth_digit_token_ids,
        freeze_camera_interpreter=depth_camera_caption_control,
        camera_pair_rows=args.camera_pair_rows == "1",
    )
    if depth_camera_caption_control:
        # Load the fixed Camera Reader independently of fresh Depth initialization.
        model.load_frozen_camera_interpreter(args.frozen_camera_checkpoint)
        if rank == 0:
            print(
                "[stage1-control] mode=depth_frozen_camera_caption_control "
                f"camera_checkpoint={args.frozen_camera_checkpoint} depth_init=random",
                flush=True,
            )
    if args.init_interpreter_checkpoint:
        init_checkpoint = os.path.abspath(
            os.path.expanduser(str(args.init_interpreter_checkpoint))
        )
        if not os.path.isdir(init_checkpoint):
            raise FileNotFoundError(
                f"Interpreter initialized checkpoint is not a directory: {init_checkpoint}"
            )
        # The third startup method only loads strictly via config + state_dict; the Trainer still creates optimizer/scheduler from step 0.
        model.load_camera_interpreter(init_checkpoint)
        args.init_interpreter_checkpoint = init_checkpoint
        if rank == 0:
            print(
                "[stage1-init] mode=weights_only "
                f"checkpoint={init_checkpoint} optimizer=fresh scheduler=fresh global_step=0",
                flush=True,
            )

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if use_depth and not use_camera:
        # Depth-only Smoke/Official training must only update the Depth Interpreter; new loss/configs cannot misinterpret freezing the LLM or Vision.
        unexpected_trainable = [
            name
            for name in trainable_names
            if not name.startswith(
                "depth_patch_tokenizer."
                if use_depth_patch_tokenizer
                else "depth_interpreter."
            )
        ]
        if not trainable_names or unexpected_trainable:
            raise RuntimeError(
                f"Depth-only requires_grad contract failed: trainable_count={len(trainable_names)}, unexpected={unexpected_trainable[:20]}"
            )
    if depth_camera_caption_control:
        unexpected_trainable = [
            name
            for name in trainable_names
            if not name.startswith("depth_interpreter.")
        ]
        if not trainable_names or unexpected_trainable:
            raise RuntimeError(
                f"B requires_grad contract failed: Only Depth Interpreter can be trained: trainable_count={len(trainable_names)}, unexpected={unexpected_trainable[:20]}"
            )
    if use_camera and not use_depth:
        # Camera-only formal tasks must update only the Camera Interpreter; frozen geom parameters with bias disabled should not affect this contract.
        unexpected_trainable = [
            name
            for name in trainable_names
            if not name.startswith("camera_interpreter.")
        ]
        if not trainable_names or unexpected_trainable:
            raise RuntimeError(
                f"Camera-only requires_grad contract failed: trainable_count={len(trainable_names)}, unexpected={unexpected_trainable[:20]}"
            )

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[model] trainable_params={trainable:,} total_params={total:,}")
        if use_camera:
            mode = "imgpose" if llm_use_image_tokens else "pose_only"
        elif use_depth:
            mode = "img+depth" if llm_use_image_tokens else "depth_only"
        else:
            mode = "unknown"
        if interpreter_cfg is not None:
            print(f"[camera] config={interpreter_cfg.to_dict()}")
            print(f"[camera] loss_mode={model.camera_loss_mode}")
            print(f"[camera] cot_version={model.camera_cot_version}")
        else:
            print("[camera] disabled (depth-only training)")
        print(f"[model] input_mode={mode}")
        print(f"[model] llm_use_image_tokens={llm_use_image_tokens}")
        print(f"[model] interpreter_use_image_tokens={interpreter_use_image_tokens}")
        if depth_interpreter_cfg is not None:
            print(f"[depth] config={depth_interpreter_cfg.to_dict()}")
            print(f"[depth] loss_mode={model.depth_loss_mode}")
            print(f"[depth] cot_version={model.depth_cot_version}")
            print(
                "[depth] meter_loss "
                f"weight={model.depth_meter_loss_weight} "
                f"beta_m={model.depth_meter_loss_beta_m}"
            )
            if model.depth_loss_mode == DEPTH_LOSS_MODE_CAPTION_PAIR_DELTA:
                print(
                    "[depth] caption_pair_delta "
                    f"relation_weight={model.depth_relation_loss_weight} "
                    f"caption_weight={1.0 - model.depth_relation_loss_weight}"
                )
            if not use_camera:
                print(
                    "[depth] trainable_modules=depth_interpreter_only "
                    f"trainable_tensor_count={len(trainable_names)}"
                )
        elif depth_patch_tokenizer_cfg is not None:
            print(
                f"[depth] patch_tokenizer_config={depth_patch_tokenizer_cfg.to_dict()}"
            )
            print(f"[depth] loss_mode={model.depth_loss_mode}")
            print(f"[depth] cot_version={model.depth_cot_version}")
            print(
                "[depth] trainable_modules=depth_patch_tokenizer_only "
                f"trainable_tensor_count={len(trainable_names)}"
            )
        print(f"[model] visual_chunk_size={model.visual_chunk_size}")
        print(f"[image_size_assert] expected PIL(W,H)={expected_image_size}")

    load_images = llm_use_image_tokens or interpreter_use_image_tokens
    train_dataset = GlobalSceneCoTDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        feature_vggt_dir=args.feature_vggt_dir,
        max_samples=args.max_samples,
        expected_image_size=expected_image_size,
        load_images=load_images,
        use_camera_cot=use_camera and not depth_camera_caption_control,
        camera_loss_mode=args.camera_loss_mode,
        camera_cot_version=args.camera_cot_version,
        depth_loss_mode=args.depth_loss_mode,
        depth_cot_version=args.depth_cot_version,
        prompt_source=args.prompt_source,
        depth_frame_supervision_scope=args.depth_frame_supervision_scope,
        depth_frame_microbatch_mode=args.depth_frame_microbatch_mode,
        depth_frame_loss_reduction=args.depth_frame_loss_reduction,
        camera_context_only=depth_camera_caption_control,
        camera_pair_rows=args.camera_pair_rows == "1",
    )
    # Share the dataset-resolved supervision identity across model, collator, and metadata.
    if use_camera and not depth_camera_caption_control:
        model.camera_cot_version = train_dataset.camera_cot_version
    else:
        model.depth_cot_version = train_dataset.depth_cot_version
    collator = CameraInterpreterCollator(
        processor=processor,
        feature_vggt_dir=args.feature_vggt_dir,
        camera_latent_size=args.camera_latent_size,
        user_prompt=args.user_prompt,
        llm_use_image_tokens=llm_use_image_tokens,
        interpreter_use_image_tokens=interpreter_use_image_tokens,
        expected_image_size=expected_image_size,
        camera_token_position=interpreter_cfg.camera_token_position
        if interpreter_cfg
        else "before_question",
        depth_token_position=(
            depth_interpreter_cfg.depth_token_position
            if depth_interpreter_cfg is not None
            else "before_question"
        ),
        use_depth_interpreter=use_depth,
        use_depth_patch_tokenizer=use_depth_patch_tokenizer,
        depth_latent_size=args.depth_latent_size,
        camera_token_layout=(
            interpreter_cfg.token_layout if interpreter_cfg is not None else "global"
        ),
        depth_token_layout=(
            depth_interpreter_cfg.token_layout
            if depth_interpreter_cfg is not None
            else ("per_frame" if depth_patch_tokenizer_cfg is not None else "global")
        ),
        use_camera_interpreter=use_camera,
        camera_loss_mode=args.camera_loss_mode,
        camera_cot_version=train_dataset.camera_cot_version,
        depth_loss_mode=args.depth_loss_mode,
        depth_cot_version=train_dataset.depth_cot_version,
        prompt_source=args.prompt_source,
        depth_frame_supervision_scope=args.depth_frame_supervision_scope,
        depth_frame_microbatch_mode=args.depth_frame_microbatch_mode,
        depth_frame_loss_reduction=args.depth_frame_loss_reduction,
        depth_meter_loss_weight=args.depth_meter_loss_weight,
        depth_relation_loss_weight=args.depth_relation_loss_weight,
        depth_digit_token_ids=depth_digit_token_ids,
        depth_decimal_token_id=depth_decimal_token_id,
        depth_camera_caption_control=depth_camera_caption_control,
        camera_pair_rows=args.camera_pair_rows == "1",
    )

    # The release environment does not require wandb; metrics and training settings are saved in the Trainer logs and checkpoint JSON files.
    report_to = []
    training_args = TrainingArguments(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch_fused",
        bf16=True,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=args.wandb_name,
        # Write training logs under output_dir.
        logging_strategy="steps",
        disable_tqdm=(args.disable_tqdm == "1"),
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,
        save_only_model=(args.save_only_model == "1"),
    )
    # The custom Trainer uses group_by_length to select the frame-count sampler.
    training_args.group_by_length = args.train_sampler_mode == "length_grouped"
    # HF native fields only save bool; additional readable sampler names are retained for strict sampler switching audit during checkpoint resume.
    training_args.train_sampler_mode = args.train_sampler_mode
    model.stage1_checkpoint_meta = _build_stage1_checkpoint_meta(
        args=args,
        model=model,
        train_dataset=train_dataset,
        branch="Depth" if depth_camera_caption_control or not use_camera else "Camera",
        resolved_min_pixels=resolved_min_pixels,
        resolved_max_pixels=resolved_max_pixels,
    )
    if train_dataset.camera_cot_version == "mindcube_da3_camera_image1_meter":
        # Continue original MindCube repair: forward does not consume num_items_in_batch; must make Trainer normalize by a4.
        # Otherwise, the average CE from four parts is directly summed, causing the update magnitude to be amplified fourfold.
        model.accepts_loss_kwargs = False
    trainer = CustomTrainerCameraInterpreter(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        # Save full processor: Stage2 needs to read min/max_pixels from Stage1 checkpoint to ensure resize and camera grid alignment.
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if rank == 0 and args.skip_final_save != "1":
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
