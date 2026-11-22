"""Generate pose and distance supervision from GT frame-by-frame visible facts, using the original training data generator."""

from __future__ import annotations
from typing import Any
import numpy as np
from PIL import Image
from depth_helpers import MAX_OBJECTS_PER_FRAME, should_skip_label

# Consistent with the six-field contract of the original GT instance assets; directly report errors for missing or extra unknown fields.
FRAME_NPZ_FIELDS = {
    "instance_ids",
    "surface_depth_m",
    "center_uv_px",
    "visible_ratio",
    "visible_pixel_count",
    "image_size",
}


def load_frame_fact(
    frame: dict[str, Any],
    instance_path: str,
    labels: dict[int, str],
) -> dict[str, Any]:
    """Fully re-read one frame of RGB/Pose/Depth/instance, return visible facts required for Caption."""

    context = f"source={frame['source']}, scene={frame['scene_id']}, frame={frame['frame_id']}"
    with Image.open(frame["image_path"]) as image:
        rgb_size = np.asarray(image.size, dtype=np.int32)
        image.verify()
    with np.load(frame["pose_path"], allow_pickle=False) as data:
        required = {"c2w", "intrinsic", "image_size"}
        if not required.issubset(data.files):
            raise RuntimeError(
                f"Pose missing fields {sorted(required - set(data.files))}: {context}"
            )
        c2w = np.asarray(data["c2w"])
        intrinsic = np.asarray(data["intrinsic"])
        pose_size = np.asarray(data["image_size"])
    if c2w.shape != (4, 4) or intrinsic.shape != (3, 3) or pose_size.shape != (2,):
        raise RuntimeError(f"Invalid pose shape: {context}")
    if (
        not np.isfinite(c2w).all()
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(pose_size).all()
    ):
        raise RuntimeError(f"Pose contains NaN/Inf: {context}")
    if not np.array_equal(rgb_size, pose_size.astype(np.int32)):
        raise RuntimeError(
            f"RGB/Pose image_size inconsistency: {context}, rgb={rgb_size.tolist()}, pose={pose_size.tolist()}"
        )

    with np.load(frame["depth_path"], allow_pickle=False) as data:
        if set(data.files) != {"depth_map", "depth_size"}:
            raise RuntimeError(
                f"Unexpected keys in depth file: {context}, keys={data.files}"
            )
        depth = np.asarray(data["depth_map"])
        depth_size = np.asarray(data["depth_size"])
    if depth.dtype != np.uint16 or depth.ndim != 2 or depth_size.shape != (2,):
        raise RuntimeError(
            f"Depth must be 2D uint16 millimeters: {context}, dtype={depth.dtype}, shape={depth.shape}"
        )
    if not np.array_equal(depth_size.astype(np.int64), depth.shape):
        raise RuntimeError(
            f"depth_size and shape are inconsistent: {context}, size={depth_size.tolist()}, shape={depth.shape}"
        )
    depth_ratio = float(depth.shape[1]) / float(depth.shape[0])
    rgb_ratio = float(rgb_size[0]) / float(rgb_size[1])
    if abs(depth_ratio - rgb_ratio) / max(depth_ratio, rgb_ratio) > 0.03:
        raise RuntimeError(
            f"RGB/Depth aspect ratio mismatch: {context}, rgb={rgb_size.tolist()}, depth={depth.shape}"
        )

    with np.load(instance_path, allow_pickle=False) as data:
        if set(data.files) != FRAME_NPZ_FIELDS:
            raise RuntimeError(
                f"Unexpected keys in instance file: path={instance_path}, keys={sorted(data.files)}"
            )
        instance_ids = np.asarray(data["instance_ids"])
        surface_depth = np.asarray(data["surface_depth_m"])
        centers = np.asarray(data["center_uv_px"])
        visible_ratio = np.asarray(data["visible_ratio"])
        visible_count = np.asarray(data["visible_pixel_count"])
        instance_size = np.asarray(data["image_size"])
    count = len(instance_ids)
    if instance_ids.dtype != np.int32 or instance_ids.shape != (count,):
        raise RuntimeError(f"instance_ids shape/dtype invalid: {instance_path}")
    if surface_depth.dtype != np.float32 or surface_depth.shape != (count,):
        raise RuntimeError(f"surface_depth_m shape/dtype illegal: {instance_path}")
    if centers.dtype != np.float32 or centers.shape != (count, 2):
        raise RuntimeError(f"center_uv_px shape/dtype invalid: {instance_path}")
    if visible_ratio.dtype != np.float32 or visible_ratio.shape != (count,):
        raise RuntimeError(f"visible_ratio shape/dtype invalid: {instance_path}")
    if visible_count.dtype != np.int32 or visible_count.shape != (count,):
        raise RuntimeError(f"visible_pixel_count shape/dtype invalid: {instance_path}")
    if instance_size.dtype != np.int32 or not np.array_equal(instance_size, rgb_size):
        raise RuntimeError(f"instance/RGB image_size inconsistent: {instance_path}")
    if count and (
        not np.isfinite(surface_depth).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(visible_ratio).all()
        or np.any(surface_depth <= 0.0)
        or np.any(visible_count <= 0)
    ):
        raise RuntimeError(
            f"Instance contains illegal values in visible facts: {instance_path}"
        )

    visible = []
    for index, instance_id_raw in enumerate(instance_ids):
        instance_id = int(instance_id_raw)
        if instance_id not in labels:
            raise RuntimeError(
                f"Frame-by-frame instance IDs not in scene metadata: {context}, id={instance_id}"
            )
        visible.append(
            {
                "instance_id": instance_id,
                "label": labels[instance_id],
                "depth_m": float(surface_depth[index]),
                "u": float(centers[index, 0]),
                "v": float(centers[index, 1]),
                "visible_ratio": float(visible_ratio[index]),
                "visible_pixel_count": int(visible_count[index]),
            }
        )
    return {
        "image_size": [int(rgb_size[0]), int(rgb_size[1])],
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "visible": visible,
    }


def nine_grid_position(u: float, v: float, image_size: list[int]) -> str:
    """Map pixel center to the name of the frozen 3x3 grid."""

    width, height = image_size
    if (
        not np.isfinite(u)
        or not np.isfinite(v)
        or not (0.0 <= u < width)
        or not (0.0 <= v < height)
    ):
        raise RuntimeError(
            f"center_uv_px out of bounds: uv=({u},{v}), image_size={image_size}"
        )
    horizontal = (
        "left" if u < width / 3.0 else ("right" if u > width * 2.0 / 3.0 else "center")
    )
    vertical = (
        "upper" if v < height / 3.0 else ("lower" if v > height * 2.0 / 3.0 else "")
    )
    if vertical and horizontal != "center":
        return f"{vertical}-{horizontal}"
    if vertical:
        return vertical
    return horizontal


def caption_visible_objects(frame_fact: dict[str, Any]) -> list[dict[str, Any]]:
    """Label filtering, Top-8 area selection, sorting from near to far, and 3x3 grid are performed only during the captioning stage."""

    filtered = [
        item for item in frame_fact["visible"] if not should_skip_label(item["label"])
    ]
    filtered.sort(key=lambda item: (-item["visible_pixel_count"], item["instance_id"]))
    filtered = filtered[:MAX_OBJECTS_PER_FRAME]
    filtered.sort(key=lambda item: (item["depth_m"], item["instance_id"]))
    result = []
    for item in filtered:
        result.append(
            {
                "label": item["label"],
                "depth_m": item["depth_m"],
                "position": nine_grid_position(
                    item["u"], item["v"], frame_fact["image_size"]
                ),
            }
        )
    return result
