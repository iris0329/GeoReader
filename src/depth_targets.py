"""Reuse the depth frame-by-frame target from the original training, without regenerating geometry or descriptors."""

import re
from typing import Dict

OLD_COT_VERSION = "depth_layout_numeric_readout_top5_position_first"

NEW_COT_VERSION = "depth_layout_perframe_caption_microbatch"

STAGE1_TASK = "depth_perframe_caption_microbatch"

HEADER = "depth layout:"

IMAGE_LINE_RE = re.compile(r"^image ([1-9][0-9]*): (.+)$")


def parse_frame_captions(caption: str, *, n_images: int, context: str) -> list[str]:
    """Remove the 'image i' prefix, but retain a consistent depth layout header for each frame target."""
    lines = str(caption).splitlines()
    if not lines or lines[0] != HEADER:
        raise RuntimeError(f"Invalid caption header: {context}, first={lines[:1]}")
    if len(lines) != n_images + 1:
        raise RuntimeError(
            f"Caption lines and n_images are inconsistent: {context}, caption_frames={len(lines) - 1}, n_images={n_images}"
        )
    targets = []
    for frame_index, line in enumerate(lines[1:]):
        match = IMAGE_LINE_RE.fullmatch(line)
        expected_number = frame_index + 1
        if match is None or int(match.group(1)) != expected_number:
            raise RuntimeError(
                f"Invalid caption image number: {context}, line={line!r}, expected={expected_number}"
            )
        body = match.group(2).strip()
        if not body:
            raise RuntimeError(
                f"Caption body for single frame cannot be empty: {context}, frame={frame_index}"
            )
        targets.append(f"{HEADER}\n{body}")
    return targets


def validate_source_record(record: Dict, *, context: str) -> tuple[int, list[str]]:
    """Verify the ordered RGB/Depth/size/frame number contract for the source window."""
    if record.get("cot_version") != OLD_COT_VERSION:
        raise RuntimeError(
            f"Source cot_version does not match: {context}, actual={record.get('cot_version')!r}"
        )
    if "idx" not in record:
        raise RuntimeError(f"Source Window missing idx: {context}")
    n_images = int(record.get("n_images", -1))
    if n_images <= 0:
        raise RuntimeError(
            f"n_images must be a positive integer: {context}, value={n_images}"
        )
    for field in ("image_input", "image_sizes", "window_frame_ids", "depth_paths"):
        values = record.get(field)
        if not isinstance(values, list) or len(values) != n_images:
            raise RuntimeError(
                f"{field} Must be a list of length n_images: {context}, type={type(values).__name__}, len={(len(values) if isinstance(values, list) else 'N/A')}"
            )
    targets = parse_frame_captions(
        str(record.get("depth_scene_cot", "")),
        n_images=n_images,
        context=context,
    )
    return n_images, targets


def make_frame_record(
    record: Dict,
    *,
    source_window_idx: int,
    frame_index: int,
    n_images: int,
    target: str,
    expanded_idx: int,
) -> Dict:
    """Construct a single-line; the source triplet is the sole identity for reassembling the window during training."""
    output = {
        "idx": expanded_idx,
        "source": record["source"],
        "scene_id": record["scene_id"],
        "n_images": 1,
        "image_input": [record["image_input"][frame_index]],
        "image_sizes": [record["image_sizes"][frame_index]],
        "window_frame_ids": [record["window_frame_ids"][frame_index]],
        "depth_paths": [record["depth_paths"][frame_index]],
        "has_depth": record["has_depth"],
        "cot_version": NEW_COT_VERSION,
        "stage1_task": STAGE1_TASK,
        "depth_single_frame_caption": target,
        # 0-based frame index directly corresponds to the source array position; during training, the frames must be reorganized in this order and cannot be sorted by filename.
        "source_window_idx": source_window_idx,
        "source_frame_index": frame_index,
        "source_n_images": n_images,
        "source_record_idx": int(record["idx"]),
    }
    # eval_bucket is used only for sampler/audit tags; its value is carried over verbatim to the single-line output when present.
    if "eval_bucket" in record:
        output["eval_bucket"] = record["eval_bucket"]
    return output
