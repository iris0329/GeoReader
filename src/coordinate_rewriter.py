#!/usr/bin/env python3
"""Rewrite question coordinates into the image processor coordinate space.

Use image_grid_thw to obtain processed image dimensions and scale each axis
separately. SPAR position_matching uses normalized boxes in the second image;
camera_motion_infer uses normalized points in the first image. Both raw-pixel
and normalized 0-1000 inputs are supported."""

from __future__ import annotations

import re
import torch

PATCH_SIZE = 14

COORDINATE_REWRITE_META_KEYS = (
    "source_dataset",
    "data_source",
    "benchmark",
    "task",
    "task_family",
    "task_type",
    "spar_marker_raw_type",
    "spar_marker_coordinate_space",
    "spar_marker_specs",
)


def copy_coordinate_rewrite_metadata(source_record: dict, target_record: dict) -> dict:
    """Pass the required sample metadata to the collator to rewrite coordinates, avoiding loss of task-specific information after Dataset cropping."""
    for meta_key in COORDINATE_REWRITE_META_KEYS:
        if meta_key in source_record:
            target_record[meta_key] = source_record[meta_key]
    return target_record


# ── From image_grid_thw compute processed dimensions ─────────────────────────────


def get_processed_size(
    image_grid_thw: torch.Tensor,
    image_idx: int = 0,
) -> tuple:
    """Calculate (processed_w, processed_h) for the image at index image_idx from image_grid_thw.

    Qwen2.5-VL image_processor outputs image_grid_thw with shape=(N_images, 3) i.e. [T, grid_h, grid_w].
    Actual pixel size = grid * PATCH_SIZE(14).
    """
    if image_grid_thw.dim() != 2 or image_grid_thw.shape[1] != 3:
        raise RuntimeError(
            f"image_grid_thw shape should be (N,3), current={tuple(image_grid_thw.shape)}"
        )
    if image_idx < 0 or image_idx >= image_grid_thw.shape[0]:
        raise RuntimeError(
            f"image_idx={image_idx} out of range [0, {image_grid_thw.shape[0]})"
        )
    grid_h = int(image_grid_thw[image_idx, 1].item())
    grid_w = int(image_grid_thw[image_idx, 2].item())
    return grid_w * PATCH_SIZE, grid_h * PATCH_SIZE


# ── Regular Expression ──────────────────────────────────────────────────────

# -- raw_pixel mode: match absolute pixel coordinates of the original image (integer >=10, to avoid misassigning small numbers)

# "bbox_2d": [x1, y1, x2, y2] (SpatialLadder grounding answer format)
_RE_BBOX_2D_JSON = re.compile(
    r'("bbox_2d"\s*:\s*\[)'
    r"(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)"
    r"(\])",
)

# "point_2d": [x, y]
_RE_POINT_2D_JSON = re.compile(
    r'("point_2d"\s*:\s*\[)'
    r"(\s*\d+\s*,\s*\d+\s*)"
    r"(\])",
)

# Accept edge-aligned bbox coordinates; filter small-number false matches in the callback.
_RE_BRACKET_4COORD = re.compile(
    r"(?<!\w)"
    r"(\[\s*)"
    r"(\d{1,5})\s*,\s*(\d{1,5})\s*,\s*(\d{1,5})\s*,\s*(\d{1,5})"
    r"(\s*\])"
    r"(?![a-zA-Z])",
)

# Match both compact and natural-language pixel points, then filter non-pixel coordinates.
_RE_PAREN_2COORD = re.compile(
    r"(\(\s*)"
    r"(\d{1,5})\s*,\s*(\d{1,5})"
    r"(\s*\))",
)

# -- normalized_1000 mode: normalized coordinates from 0 to 1000, allowing unit numbers and edge values -1

# Allow slightly negative bbox coordinates only in normalized_1000 mode.
_RE_BRACKET_4COORD_NORM = re.compile(
    r"(\[\s*)"
    r"(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})\s*,\s*(-?\d{1,4})"
    r"(\s*\])",
)

# Interpret camera_motion_infer points in the normalized 0-1000 coordinate system.
_RE_PAREN_2COORD_NORM = re.compile(
    r"(\(\s*)"
    r"(-?\d{1,4})\s*,\s*(-?\d{1,4})"
    r"(\s*\))",
)


# ── Coordinate Scaling Callback ────────────────────────────────────────────────────


def _scale_xy(x: int, y: int, sx: float, sy: float) -> tuple:
    """Scale by scale_x and scale_y respectively."""
    return round(x * sx), round(y * sy)


def _clamp(value: int, lower: int, upper: int) -> int:
    """Clip coordinates to the valid image range to prevent negative or slightly out-of-bounds coordinates from being passed to the model."""
    return max(int(lower), min(int(upper), int(value)))


def _looks_like_pixel_coord(values: list) -> bool:
    """Determine if an independent numeric list resembles pixel coordinates, and avoid mistakenly modifying small numeric lists like [1, 2, 3, 4]."""
    return any(int(value) >= 10 for value in values)


def _replace_bbox_2d_json(match: re.Match, sx: float, sy: float) -> str:
    """Scale "bbox_2d": [x1,y1,x2,y2]."""
    prefix, coords_str, suffix = match.group(1), match.group(2), match.group(3)
    vals = [int(v.strip()) for v in coords_str.split(",")]
    x1, y1 = _scale_xy(vals[0], vals[1], sx, sy)
    x2, y2 = _scale_xy(vals[2], vals[3], sx, sy)
    return f"{prefix}{x1}, {y1}, {x2}, {y2}{suffix}"


def _replace_point_2d_json(match: re.Match, sx: float, sy: float) -> str:
    """Scale "point_2d": [x, y]."""
    prefix, coords_str, suffix = match.group(1), match.group(2), match.group(3)
    vals = [int(v.strip()) for v in coords_str.split(",")]
    x, y = _scale_xy(vals[0], vals[1], sx, sy)
    return f"{prefix}{x}, {y}{suffix}"


def _replace_bracket_4coord(match: re.Match, sx: float, sy: float) -> str:
    """Scale [x1, y1, x2, y2]."""
    prefix = match.group(1)
    v1, v2, v3, v4 = (
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
    )
    suffix = match.group(6)
    if not _looks_like_pixel_coord([v1, v2, v3, v4]):
        return match.group(0)
    x1, y1 = _scale_xy(v1, v2, sx, sy)
    x2, y2 = _scale_xy(v3, v4, sx, sy)
    return f"{prefix}{x1}, {y1}, {x2}, {y2}{suffix}"


def _replace_bracket_4coord_normalized(
    match: re.Match,
    sx: float,
    sy: float,
    processed_w: int,
    processed_h: int,
) -> str:
    """Scale 0-1000 normalized bbox; in explicit mode, 0/unit count/edge -1 must all be converted."""
    prefix = match.group(1)
    v1, v2, v3, v4 = (
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
    )
    suffix = match.group(6)
    x1, y1 = _scale_xy(v1, v2, sx, sy)
    x2, y2 = _scale_xy(v3, v4, sx, sy)
    # Clamp edge-aligned normalized boxes to the processed image boundaries.
    x1 = _clamp(x1, 0, processed_w)
    y1 = _clamp(y1, 0, processed_h)
    x2 = _clamp(x2, 0, processed_w)
    y2 = _clamp(y2, 0, processed_h)
    return f"{prefix}{x1}, {y1}, {x2}, {y2}{suffix}"


def _replace_paren_2coord_normalized(
    match: re.Match,
    sx: float,
    sy: float,
    processed_w: int,
    processed_h: int,
) -> str:
    """Scale 0-1000 normalized point; in explicit normalized mode, 0/unit count must also be converted."""
    prefix = match.group(1)
    x, y = int(match.group(2)), int(match.group(3))
    suffix = match.group(4)
    nx, ny = _scale_xy(x, y, sx, sy)
    # Normalized data may occasionally border beyond by 1 unit; Qwen input has no negative pixels, uniformly clamped to valid range.
    nx = _clamp(nx, 0, processed_w)
    ny = _clamp(ny, 0, processed_h)
    return f"{prefix}{nx}, {ny}{suffix}"


def _replace_paren_2coord(match: re.Match, sx: float, sy: float) -> str:
    """Scale (x, y) image plane coordinates."""
    prefix = match.group(1)
    x, y = int(match.group(2)), int(match.group(3))
    suffix = match.group(4)
    if not _looks_like_pixel_coord([x, y]):
        return match.group(0)
    nx, ny = _scale_xy(x, y, sx, sy)
    return f"{prefix}{nx}, {ny}{suffix}"


# ── Internal: raw_pixel mode rewrite ────────────────────────────────────────


def _rewrite_raw_pixel(text: str, sx: float, sy: float) -> str:
    """Rewrite original image absolute pixel coordinates. Process according to priority, use placeholder to prevent double scaling."""
    placeholders: list = []

    def _bbox_with_ph(match: re.Match) -> str:
        replaced = _replace_bbox_2d_json(match, sx, sy)
        token = f"\x00BBOX{len(placeholders)}\x00"
        placeholders.append(replaced)
        return token

    def _point_with_ph(match: re.Match) -> str:
        replaced = _replace_point_2d_json(match, sx, sy)
        token = f"\x00PT{len(placeholders)}\x00"
        placeholders.append(replaced)
        return token

    # Steps 1-2: Process JSON key format, replace with placeholder for protection
    text = _RE_BBOX_2D_JSON.sub(_bbox_with_ph, text)
    text = _RE_POINT_2D_JSON.sub(_point_with_ph, text)

    # Step 3: Process independent bracket coordinates (JSON coordinates have been replaced with placeholders to avoid misalignment)
    text = _RE_BRACKET_4COORD.sub(lambda m: _replace_bracket_4coord(m, sx, sy), text)

    # Step 4: Process (x,y) coordinates
    text = _RE_PAREN_2COORD.sub(lambda m: _replace_paren_2coord(m, sx, sy), text)

    # Step 5: Restore placeholder
    for i, original in enumerate(placeholders):
        text = text.replace(f"\x00BBOX{i}\x00", original)
        text = text.replace(f"\x00PT{i}\x00", original)

    return text


# ── Internal: normalized_1000 mode rewrite ──────────────────────────────────


def _rewrite_normalized_1000(
    text: str, sx: float, sy: float, processed_w: int, processed_h: int
) -> str:
    """Convert 0-1000 normalized coordinates to processed absolute pixels."""
    text = _RE_BRACKET_4COORD_NORM.sub(
        lambda m: _replace_bracket_4coord_normalized(
            m, sx, sy, processed_w, processed_h
        ),
        text,
    )
    text = _RE_PAREN_2COORD_NORM.sub(
        lambda m: _replace_paren_2coord_normalized(m, sx, sy, processed_w, processed_h),
        text,
    )
    return text


# ── Main Entry ──────────────────────────────────────────────────────────


def rewrite_coordinates_in_text(
    text: str,
    *,
    original_w: int,
    original_h: int,
    processed_w: int,
    processed_h: int,
    coord_source: str = "raw_pixel",
) -> str:
    """Convert bbox/point coordinates from original sensor scale to processed pixel scale.

    coord_source="raw_pixel":
        Original pixel coordinates → processed pixel coordinates.
        scale_x = processed_w / original_w, scale_y = processed_h / original_h。

    coord_source="normalized_1000":
        0-1000 normalized → processed pixel.
        new_x = round(old_x / 1000 * processed_w)。
        In this mode, original_w / original_h is not involved in the calculation, but must still be provided to maintain interface consistency.

    Args:
        Original text
        original_w: Original image width (PIL Image.width)
        original_h: Original image height (PIL Image.height)
        processed_w: actual output width of the processor (grid_w * PATCH_SIZE)
        processed_h: actual output height of processor (grid_h * PATCH_SIZE)
        coord_source: Coordinate source, "raw_pixel" or "normalized_1000"

    Returns:
        Coordinate has been rewritten text
    """
    if coord_source == "raw_pixel":
        if processed_w == original_w and processed_h == original_h:
            return text
        sx = processed_w / original_w
        sy = processed_h / original_h
        return _rewrite_raw_pixel(text, sx, sy)
    elif coord_source == "normalized_1000":
        sx = processed_w / 1000.0
        sy = processed_h / 1000.0
        return _rewrite_normalized_1000(text, sx, sy, processed_w, processed_h)
    else:
        raise ValueError(
            f"Unknown coord_source: {coord_source!r}, supports 'raw_pixel' or 'normalized_1000'"
        )


def _is_spar_record(record: dict) -> bool:
    """Determine if the sample originates from SPAR; SPAR training data contains two clues: task_family and marker fields."""
    source_values = {
        str(record.get("source_dataset", "")).strip().lower(),
        str(record.get("data_source", "")).strip().lower(),
        str(record.get("benchmark", "")).strip().lower(),
    }
    if source_values.intersection({"spar", "spar7m"}):
        return True
    if "spar_marker_coordinate_space" in record or "spar_marker_specs" in record:
        return True
    task = str(record.get("task_family", record.get("task", ""))).strip()
    return task in {"position_matching", "camera_motion_infer"}


def coordinate_rewrite_plan_for_record(record: dict) -> tuple[int, str, str]:
    """Return sample coordinate rewrite plan: (target_image_idx, coord_source, debug_label).

    Training and evaluation share this task routing table to ensure consistent coordinate rewriting logic.

    Return value description:
      target_image_idx: The coordinate corresponds to the image number (0-indexed) used to obtain the processed dimensions of that image.
      coord_source: coordinate source specification, determines scaling formula
                          "raw_pixel"       → scale = processed / original
                          "normalized_1000" → scale = processed / 1000
      debug_label: Debug identifier, helps locate which route failed when an error occurs
    """
    task = str(record.get("task_family", record.get("task", ""))).strip()
    if _is_spar_record(record) and task == "position_matching":
        # position_matching options refer to normalized boxes in the second image.
        return 1, "normalized_1000", "spar_position_normalized_1000_to_processed_pixels"
    if _is_spar_record(record) and task == "camera_motion_infer":
        # camera_motion_infer options refer to normalized points in the first image.
        return (
            0,
            "normalized_1000",
            "spar_camera_motion_normalized_1000_to_processed_pixels",
        )
    # Default: Most data containing coordinates (such as SpatialLadder, CAVQA, etc.) use absolute pixel coordinates from the original image.
    return 0, "raw_pixel", "raw_pixel_first_image_to_processed_pixels"


def _has_rewritable_numeric_coordinates(text: str) -> bool:
    """Check whether the text contains numeric coordinates handled by the coordinate rewriter."""
    return bool(
        _RE_BBOX_2D_JSON.search(text)
        or _RE_POINT_2D_JSON.search(text)
        or _RE_BRACKET_4COORD.search(text)
        or _RE_PAREN_2COORD.search(text)
        or _RE_BRACKET_4COORD_NORM.search(text)
    )


def _assert_spar_numeric_coordinate_task_is_registered(
    record: dict, texts: dict, *, context: str
) -> None:
    """Register the coordinate convention for SPAR tasks containing numeric coordinates before applying raw_pixel conversion."""
    task = str(record.get("task_family", record.get("task", ""))).strip()
    if not _is_spar_record(record) or task in {
        "position_matching",
        "camera_motion_infer",
    }:
        return
    for key, value in texts.items():
        if _has_rewritable_numeric_coordinates(str(value)):
            raise RuntimeError(
                f"Encountered an unregistered SPAR numeric coordinate task; raw_pixel cannot be used by default: {context}, task={task!r}, field={key!r}"
            )


def rewrite_sample_coordinates_in_texts(
    record: dict,
    texts: dict,
    images: list,
    image_grid_thw: torch.Tensor,
    *,
    context: str,
) -> dict:
    """Rewrite coordinate fields in multiple text fields according to the sample task scope, ensuring training and evaluation use the same coordinate system."""
    if not images:
        raise RuntimeError(
            f"Coordinate rewriting requires at least one image: {context}"
        )
    _assert_spar_numeric_coordinate_task_is_registered(record, texts, context=context)
    image_idx, coord_source, rewrite_source = coordinate_rewrite_plan_for_record(record)
    if image_idx < 0 or image_idx >= len(images):
        raise RuntimeError(
            f"Coordinate rewrite target image out of bounds: {context}, image_idx={image_idx}, n_images={len(images)}, rewrite_source={rewrite_source}"
        )
    target_image = images[image_idx]
    processed_w, processed_h = get_processed_size(image_grid_thw, image_idx=image_idx)
    rewritten = {}
    for key, value in texts.items():
        rewritten[key] = rewrite_coordinates_in_text(
            str(value),
            original_w=int(target_image.width),
            original_h=int(target_image.height),
            processed_w=processed_w,
            processed_h=processed_h,
            coord_source=coord_source,
        )
    return rewritten


# ── Self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate image_grid_thw: Original image 1296x968 → processed 616x476
    # grid_h=34, grid_w=44 → 34*14=476, 44*14=616
    grid = torch.tensor([[1, 34, 44]])
    pw, ph = get_processed_size(grid, image_idx=0)
    print(f"processed size: {pw}x{ph}")
    assert (pw, ph) == (616, 476), f"Expected (616,476), actual ({pw},{ph})"

    # 1. raw_pixel: SpatialLadder bbox_2d
    sl_text = '[{"bbox_2d": [976, 61, 1296, 248], "label": "pillow"}]'
    result = rewrite_coordinates_in_text(
        sl_text,
        original_w=1296,
        original_h=968,
        processed_w=616,
        processed_h=476,
    )
    print(f"SL Original: {sl_text}")
    print(f"SL rewritten: {result}")

    # 2. raw_pixel: CAVQA bracket
    cavqa = "What is at [618, 486, 732, 751]?"
    result = rewrite_coordinates_in_text(
        cavqa,
        original_w=1296,
        original_h=968,
        processed_w=616,
        processed_h=476,
    )
    print(f"CAVQA Original: {cavqa}")
    print(f"CAVQA rewritten: {result}")

    # 3. raw_pixel: SPAR compact point format (x,y),depth
    spar = "(447, 386),2.6"
    result = rewrite_coordinates_in_text(
        spar,
        original_w=1296,
        original_h=968,
        processed_w=616,
        processed_h=476,
    )
    print(f"SPAR Original: {spar}")
    print(f"SPAR rewritten: {result}")
    assert result == "(212, 190),2.6", (
        f"SPAR compact point should be rewritten, actual: {result}"
    )

    # Match raw-pixel points in natural-language SPAR questions.
    spar_prose = (
        "<answer>The second image's observer would be located at (447,386) "
        "in the first image, at a depth of 2.6 meters.</answer>"
    )
    result = rewrite_coordinates_in_text(
        spar_prose,
        original_w=1296,
        original_h=968,
        processed_w=616,
        processed_h=476,
    )
    print(f"SPAR prose point original: {spar_prose}")
    print(f"SPAR prose point rewritten: {result}")
    assert "(212, 190)" in result, (
        f"SPAR natural language point should be rewritten, actual: {result}"
    )

    # 4. normalized_1000: SPAR-Bench position_matching
    spar_bench = "A. [0, 0, 556, 720] B. [529, 1, 1000, 632]"
    result = rewrite_coordinates_in_text(
        spar_bench,
        original_w=1296,
        original_h=968,
        processed_w=616,
        processed_h=476,
        coord_source="normalized_1000",
    )
    print(f"SPAR-Bench Original: {spar_bench}")
    print(f"SPAR-Bench rewritten: {result}")

    # 5. normalized_1000: SPAR-Bench edge-aligned negative coordinates, -1 semantically represents the upper boundary and must be clamped to 0
    spar_bench_negative = "D. [917, -1, 1000, 468]"
    result = rewrite_coordinates_in_text(
        spar_bench_negative,
        original_w=1296,
        original_h=968,
        processed_w=616,
        processed_h=476,
        coord_source="normalized_1000",
    )
    print(f"SPAR-Bench negative coordinate original: {spar_bench_negative}")
    print(f"SPAR-Bench negative coordinate rewrite: {result}")
    assert result == "D. [565, 0, 616, 223]", (
        f"Negative coordinates should be clamped to 0, actual: {result}"
    )

    # 6. No resize (processed == original)
    no_change = '[{"bbox_2d": [100, 200, 300, 400], "label": "obj"}]'
    result = rewrite_coordinates_in_text(
        no_change,
        original_w=616,
        original_h=476,
        processed_w=616,
        processed_h=476,
    )
    assert result == no_change, "processed == original should return unchanged"
    print("No resize: return unchanged ✓")

    # 7. placeholder prevent double scaling
    double_test = (
        '[{"bbox_2d": [100, 200, 300, 400], "label": "obj"}]  [100, 200, 300, 400]'
    )
    result = rewrite_coordinates_in_text(
        double_test,
        original_w=1000,
        original_h=1000,
        processed_w=500,
        processed_h=500,
    )
    print(f"Anti-duplicate scaling: {result}")
    # bbox_2d coordinates [100,200,300,400] become [50,100,150,200] after one scaling operation.
    # Independent bracket [100,200,300,400] → [50,100,150,200] (scaled once)
    # No bbox_2d scaling will be independently bracketed and scaled again.
    assert '"bbox_2d": [50, 100, 150, 200]' in result, (
        "bbox_2d should be scaled only once"
    )
    print("placeholder prevents double scaling ✓")

    # 8. Sample-level rewriting: SPAR position_matching must be converted to processed pixel coordinates for the bounding boxes 0-1000 in the second image.
    class _DummyImage:
        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height

    sample_grid = torch.tensor([[1, 34, 44], [1, 34, 44]])
    sample_images = [_DummyImage(1296, 968), _DummyImage(1296, 968)]
    sample_texts = {
        "text_input": "B. [914, 2, 1000, 690]",
        "text_output": "<answer>B</answer>",
    }
    rewritten = rewrite_sample_coordinates_in_texts(
        {"data_source": "spar7m", "task_family": "position_matching"},
        sample_texts,
        sample_images,
        sample_grid,
        context="self-test position_matching",
    )
    assert rewritten["text_input"] == "B. [563, 1, 616, 328]", rewritten["text_input"]
    assert rewritten["text_output"] == "<answer>B</answer>", rewritten["text_output"]
    print("SPAR position_matching sample-level rewrite ✓")

    # 9. Sample-level rewriting: SPAR camera_motion_infer processes the first image's 0-1000 points to processed pixel coordinates
    rewritten = rewrite_sample_coordinates_in_texts(
        {"data_source": "spar7m", "task_family": "camera_motion_infer"},
        {"text_input": "A. Image Coor:(569, 18), Depth:1.1 meters"},
        sample_images,
        sample_grid,
        context="self-test camera_motion_infer",
    )
    assert "Image Coor:(351, 9)" in rewritten["text_input"], rewritten["text_input"]
    print("SPAR camera_motion_infer sample-level rewrite ✓")

    # 10. For other SPAR question types, if numeric coordinates appear in the future, the coordinate system must be explicitly registered; default raw_pixel is not allowed.
    try:
        rewrite_sample_coordinates_in_texts(
            {"data_source": "spar7m", "task_family": "future_spar_coord_task"},
            {"text_input": "A. [0, 0, 1000, 1000]"},
            sample_images,
            sample_grid,
            context="self-test unregistered spar coord task",
        )
        raise AssertionError(
            "Unregistered SPAR digital pose types should directly throw an error."
        )
    except RuntimeError as exc:
        assert "unregistered SPAR numeric coordinate task" in str(exc), str(exc)
    print("SPAR unregistered digital pose type fail-loud ✓")

    print("\nAll self-tests passed.")
