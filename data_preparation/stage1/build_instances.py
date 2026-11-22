#!/usr/bin/env python3
"""Build reusable compact frame-by-frame GT instance assets from a frozen Window manifest."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


from depth_helpers import (
    DEPTH_CONSISTENCY_TAU_BASE,
    DEPTH_CONSISTENCY_TAU_REL,
    MAX_DEPTH_M,
    MAX_INSTANCE_POINTS,
    MIN_DEPTH_M,
    MIN_PROJECTED_POINTS,
    MIN_VISIBLE_DEPTH_PIXELS,
    MIN_VISIBLE_RATIO,
    _read_ply_vertices,
)

SCANNET_ROOT = None
SCANNETPP_ROOT = None

EXPECTED_ROWS = 35_650
EXPECTED_UNIQUE_FRAMES = 219_130
EXPECTED_BUCKETS = {1: 6_200, 2: 6_200, 4: 9_300, 8: 9_300, 16: 3_100, 32: 1_550}
EXPECTED_SOURCES = {"scannet": 27_163, "scannetpp": 8_487}
EXPECTED_SCENES = {"scannet": 1_181, "scannetpp": 369}
EXPECTED_MANIFEST_FIELDS = {
    "source",
    "scene_id",
    "image_paths",
    "pose_paths",
    "depth_paths",
}
FRAME_NPZ_FIELDS = {
    "instance_ids",
    "surface_depth_m",
    "center_uv_px",
    "visible_ratio",
    "visible_pixel_count",
    "image_size",
}
SCENE_POINTS_FIELDS = {"instance_ids", "point_offsets", "points_world"}
ASSET_VERSION = "depth_layout_instance_surface_projection_compact"


def parse_args() -> argparse.Namespace:
    """Parse build parameters; by default only accepts frozen full 35,650 Window."""

    parser = argparse.ArgumentParser(
        description="Build compact Stage 1 GT instance assets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--index-jsonl", type=Path, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        # Each worker loads only one scene at a time; the higher the concurrency, the higher the peak memory usage for meshes and shared disk I/O.
        help="Number of worker processes per scene.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        metavar="SOURCE/SCENE_ID",
        help="Build only this scene for a quick check (repeatable); the scene list is not changed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild scenes that are already complete.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing outputs; do not build.",
    )
    parser.add_argument("--scannet-root", type=Path, required=True)
    parser.add_argument("--scannetpp-root", type=Path, required=True)
    return parser.parse_args()


def ensure_slurm_job() -> None:
    """Public entry allows standalone workstations; cluster operation instructions require using compute nodes."""
    return None


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Atomic write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write index atomically in the order physical frames first appear in the frozen manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_npz(path: Path, **payload: np.ndarray) -> None:
    """Atomic write compression NPZ, avoid leaving an incomplete formal file after interruption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def require_absolute_file(raw_path: Any, *, field: str, context: str) -> Path:
    """Strictly parse absolute file paths from the manifest."""

    path = Path(str(raw_path))
    if not path.is_absolute():
        raise RuntimeError(
            f"{field} Must be an absolute path: {context}, value={raw_path!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"{field} Not found: {context}, path={path}")
    return path


def parse_frame_id(path: Path, *, context: str) -> int:
    """Parse integer frame id from per-frame RGB/Pose/Depth filenames."""

    try:
        return int(path.stem)
    except ValueError as exc:
        raise RuntimeError(
            f"Frame filename must be an integer: {context}, path={path}"
        ) from exc


def load_manifest(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Verify frozen manifest and organize unique physical frames and scene groups."""

    if not path.is_file():
        raise FileNotFoundError(f"Window manifest does not exist: {path}")
    windows = 0
    buckets: collections.Counter[int] = collections.Counter()
    sources: collections.Counter[str] = collections.Counter()
    scenes: dict[str, set[str]] = collections.defaultdict(set)
    frames: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for source_line_index, line in enumerate(handle):
            if not line.strip():
                raise RuntimeError(
                    f"Window manifest physical line empty: line={source_line_index}"
                )
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != EXPECTED_MANIFEST_FIELDS:
                raise RuntimeError(
                    f"Unexpected fields in the window manifest: line={source_line_index}, keys={(sorted(record) if isinstance(record, dict) else type(record))}"
                )
            source = str(record["source"])
            scene_id = str(record["scene_id"])
            if source not in EXPECTED_SOURCES or not scene_id or "/" in scene_id:
                raise RuntimeError(
                    f"source/scene_id illegal: line={source_line_index}, {source!r}/{scene_id!r}"
                )
            image_paths = record["image_paths"]
            pose_paths = record["pose_paths"]
            depth_paths = record["depth_paths"]
            if not isinstance(image_paths, list) or not image_paths:
                raise RuntimeError(
                    f"image_paths must be a non-empty list: line={source_line_index}"
                )
            if not isinstance(pose_paths, list) or len(pose_paths) != len(image_paths):
                raise RuntimeError(
                    f"pose_paths and image_paths have different lengths: line={source_line_index}"
                )
            if not isinstance(depth_paths, list) or len(depth_paths) != len(
                image_paths
            ):
                raise RuntimeError(
                    f"depth_paths and image_paths have unequal lengths: line={source_line_index}"
                )
            windows += 1
            buckets[len(image_paths)] += 1
            sources[source] += 1
            scenes[source].add(scene_id)
            for view_index, raw_paths in enumerate(
                zip(image_paths, pose_paths, depth_paths)
            ):
                context = f"line={source_line_index}, view={view_index}, source={source}, scene={scene_id}"
                image_path = require_absolute_file(
                    raw_paths[0], field="RGB", context=context
                )
                pose_path = require_absolute_file(
                    raw_paths[1], field="Pose", context=context
                )
                depth_path = require_absolute_file(
                    raw_paths[2], field="Depth", context=context
                )
                ids = [
                    parse_frame_id(item, context=context)
                    for item in (image_path, pose_path, depth_path)
                ]
                if len(set(ids)) != 1:
                    raise RuntimeError(
                        f"RGB/Pose/Depth frame id inconsistency: {context}, ids={ids}"
                    )
                key = (source, scene_id, ids[0])
                frame = {
                    "source": source,
                    "scene_id": scene_id,
                    "frame_id": ids[0],
                    "image_path": str(image_path),
                    "pose_path": str(pose_path),
                    "depth_path": str(depth_path),
                }
                previous = frames.setdefault(key, frame)
                if previous != frame:
                    raise RuntimeError(
                        f"Same physical frame pointing to different assets: key={key}, existing={previous}, new={frame}"
                    )

    actual_scenes = {source: len(values) for source, values in scenes.items()}
    if windows != EXPECTED_ROWS or dict(buckets) != EXPECTED_BUCKETS:
        raise RuntimeError(
            f"Window statistics change: rows={windows}, buckets={dict(buckets)}"
        )
    if dict(sources) != EXPECTED_SOURCES or actual_scenes != EXPECTED_SCENES:
        raise RuntimeError(
            f"source/scene statistics change: sources={dict(sources)}, scenes={actual_scenes}"
        )
    if len(frames) != EXPECTED_UNIQUE_FRAMES:
        raise RuntimeError(
            f"The only physical frame number change: actual={len(frames)}, expected={EXPECTED_UNIQUE_FRAMES}"
        )

    ordered_frames = list(frames.values())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for frame in ordered_frames:
        grouped[(frame["source"], frame["scene_id"])].append(frame)
    return ordered_frames, dict(grouped)


def raw_scene_paths(source: str, scene_id: str) -> tuple[Path, Path, Path]:
    """Return the paths to the frozen mesh, segment, and instance annotation data sources for two datasets."""

    if source == "scannet":
        scene_root = SCANNET_ROOT / scene_id
        return (
            scene_root / f"{scene_id}_vh_clean_2.ply",
            scene_root / f"{scene_id}_vh_clean_2.0.010000.segs.json",
            scene_root / f"{scene_id}.aggregation.json",
        )
    if source == "scannetpp":
        scene_root = SCANNETPP_ROOT / scene_id / "scans"
        return (
            scene_root / "mesh_aligned_0.05.ply",
            scene_root / "segments.json",
            scene_root / "segments_anno.json",
        )
    raise RuntimeError(f"Unsupported data source: {source!r}")


def load_scene_assets(
    source: str, scene_id: str
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Extract complete instance-level world coordinate points from original annotations, without filtering labels at the asset layer."""

    ply_path, segs_path, anno_path = raw_scene_paths(source, scene_id)
    for path in (ply_path, segs_path, anno_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing instance annotation: source={source}, scene={scene_id}, path={path}"
            )
    xyz = np.asarray(_read_ply_vertices(str(ply_path)), dtype=np.float32)
    with segs_path.open("r", encoding="utf-8") as handle:
        seg_payload = json.load(handle)
    if "segIndices" not in seg_payload:
        raise RuntimeError(f"segments JSON is missing segIndices: {segs_path}")
    seg_indices = np.asarray(seg_payload["segIndices"], dtype=np.int64)
    if seg_indices.ndim != 1 or len(seg_indices) != len(xyz):
        raise RuntimeError(
            f"segIndices and PLY vertex count do not match: segs={seg_indices.shape}, xyz={xyz.shape}"
        )
    if np.any(seg_indices < 0):
        raise RuntimeError(f"segIndices contains negative numbers: {segs_path}")
    with anno_path.open("r", encoding="utf-8") as handle:
        anno = json.load(handle)
    groups = anno.get("segGroups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError(f"annotation missing non-empty segGroups: {anno_path}")

    # Assign shared segments to every annotated instance and retain mesh vertex order.
    segment_order = np.argsort(seg_indices, kind="stable")
    sorted_segments = seg_indices[segment_order]
    unique_segments, segment_starts, segment_counts = np.unique(
        sorted_segments,
        return_index=True,
        return_counts=True,
    )
    segment_slices = {
        int(segment_id): (int(start), int(start + count))
        for segment_id, start, count in zip(
            unique_segments, segment_starts, segment_counts
        )
    }
    raw_meta: list[dict[str, Any]] = []
    seen_instance_ids: set[int] = set()
    for group in groups:
        instance_id = int(group["id"])
        if instance_id in seen_instance_ids:
            raise RuntimeError(
                f"instance id duplicate: source={source}, scene={scene_id}, id={instance_id}"
            )
        seen_instance_ids.add(instance_id)
        label = str(group.get("label", "")).strip()
        segments = np.asarray(
            sorted(set(int(item) for item in group.get("segments", []))), dtype=np.int64
        )
        if segments.size == 0:
            raise RuntimeError(
                f"instance has no segments: source={source}, scene={scene_id}, id={instance_id}"
            )
        if np.any(segments < 0):
            raise RuntimeError(
                f"instance segment id is negative: source={source}, scene={scene_id}, id={instance_id}"
            )
        raw_meta.append(
            {"instance_id": instance_id, "label": label, "segments": segments}
        )

    instances: list[dict[str, Any]] = []
    point_chunks: list[np.ndarray] = []
    point_offsets = [0]
    instance_ids: list[int] = []
    for meta in raw_meta:
        index_chunks = []
        for segment_id_raw in meta["segments"]:
            segment_id = int(segment_id_raw)
            if segment_id not in segment_slices:
                # Ignore annotated segments that contain no mesh vertices.
                continue
            start, end = segment_slices[segment_id]
            index_chunks.append(segment_order[start:end])
        if not index_chunks:
            continue
        vertex_indices = np.concatenate(index_chunks)
        vertex_indices.sort()
        points = np.ascontiguousarray(xyz[vertex_indices], dtype=np.float32)
        if points.shape[0] == 0:
            continue
        aabb_min = points.min(axis=0)
        aabb_max = points.max(axis=0)
        instance_ids.append(int(meta["instance_id"]))
        point_chunks.append(points)
        point_offsets.append(point_offsets[-1] + int(points.shape[0]))
        instances.append(
            {
                "instance_id": int(meta["instance_id"]),
                "label": str(meta["label"]),
                "aabb_min": [float(item) for item in aabb_min],
                "aabb_max": [float(item) for item in aabb_max],
                "point_count": int(points.shape[0]),
            }
        )
    if not instances:
        raise RuntimeError(
            f"scene has no available instance points: source={source}, scene={scene_id}"
        )
    points_world = np.ascontiguousarray(
        np.concatenate(point_chunks, axis=0), dtype=np.float32
    )
    return (
        instances,
        np.asarray(instance_ids, dtype=np.int32),
        np.asarray(point_offsets, dtype=np.int64),
        points_world,
        {
            "mesh": str(ply_path.resolve()),
            "segments": str(segs_path.resolve()),
            "annotation": str(anno_path.resolve()),
        },
    )


def load_pose_and_depth(
    frame: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one frame c2w/K/RGB with dimensions and uint16 millimeter Depth."""

    context = f"source={frame['source']}, scene={frame['scene_id']}, frame={frame['frame_id']}"
    with np.load(frame["pose_path"], allow_pickle=False) as data:
        required = {"c2w", "intrinsic", "image_size"}
        if not required.issubset(data.files):
            raise RuntimeError(
                f"Pose missing fields {sorted(required - set(data.files))}: {context}"
            )
        c2w = np.asarray(data["c2w"], dtype=np.float64)
        intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        image_size = np.asarray(data["image_size"])
    if c2w.shape != (4, 4) or intrinsic.shape != (3, 3) or image_size.shape != (2,):
        raise RuntimeError(
            f"Invalid pose shape: {context}, c2w={c2w.shape}, K={intrinsic.shape}, size={image_size.shape}"
        )
    if (
        not np.isfinite(c2w).all()
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(image_size).all()
    ):
        raise RuntimeError(f"Pose contains NaN/Inf: {context}")
    image_size_i32 = image_size.astype(np.int32)
    if not np.array_equal(image_size, image_size_i32) or np.any(image_size_i32 <= 0):
        raise RuntimeError(
            f"Invalid pose image_size: {context}, value={image_size.tolist()}"
        )
    with Image.open(frame["image_path"]) as image:
        actual_size = np.asarray(image.size, dtype=np.int32)
    if not np.array_equal(actual_size, image_size_i32):
        raise RuntimeError(
            f"RGB/Pose size mismatch: {context}, rgb={actual_size.tolist()}, pose={image_size_i32.tolist()}"
        )
    with np.load(frame["depth_path"], allow_pickle=False) as data:
        if set(data.files) != {"depth_map", "depth_size"}:
            raise RuntimeError(f"Depth NPZ field changes: {context}, keys={data.files}")
        raw_depth = np.asarray(data["depth_map"])
        depth_size = np.asarray(data["depth_size"])
    if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise RuntimeError(
            f"Depth must be 2D uint16 millimeters: {context}, dtype={raw_depth.dtype}, shape={raw_depth.shape}"
        )
    if depth_size.shape != (2,) or not np.array_equal(
        depth_size.astype(np.int64), raw_depth.shape
    ):
        raise RuntimeError(
            f"depth_size and depth_map shape are inconsistent: {context}, size={depth_size.tolist()}, shape={raw_depth.shape}"
        )
    depth_ratio = float(raw_depth.shape[1]) / float(raw_depth.shape[0])
    rgb_ratio = float(image_size_i32[0]) / float(image_size_i32[1])
    if abs(depth_ratio - rgb_ratio) / max(depth_ratio, rgb_ratio) > 0.03:
        raise RuntimeError(
            f"RGB/Depth aspect ratio mismatch: {context}, rgb={image_size_i32.tolist()}, depth={raw_depth.shape}"
        )
    depth_m = raw_depth.astype(np.float32) / 1000.0
    return c2w, intrinsic, image_size_i32, depth_m


def project_frame(
    instances: list[dict[str, Any]],
    instance_ids: np.ndarray,
    point_offsets: np.ndarray,
    points_world: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_size: np.ndarray,
    depth_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project every annotated instance without label filtering or object-count truncation."""

    img_w, img_h = [int(item) for item in image_size]
    depth_h, depth_w = depth_m.shape
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    visible: list[tuple[int, float, float, float, float, int]] = []
    if (
        len(instances) != len(instance_ids)
        or len(point_offsets) != len(instance_ids) + 1
    ):
        raise RuntimeError(
            "Scene instance metadata and point_offsets count do not match"
        )
    for instance_index, instance in enumerate(instances):
        instance_id = int(instance_ids[instance_index])
        if instance_id != int(instance["instance_id"]):
            raise RuntimeError(
                f"scene instance id order inconsistent: meta={instance['instance_id']}, points={instance_id}"
            )
        start, end = (
            int(point_offsets[instance_index]),
            int(point_offsets[instance_index + 1]),
        )
        points = points_world[start:end]
        if len(points) > MAX_INSTANCE_POINTS:
            # Preserve deterministic history projection with an 8k point limit; scene_points still retain all points, subsequent rules can reconstruct.
            keep = np.linspace(0, len(points) - 1, MAX_INSTANCE_POINTS).astype(np.int64)
            points = points[keep]
        points_camera = (
            points.astype(np.float64, copy=False) @ rotation.T + translation[None, :]
        )
        z = points_camera[:, 2]
        in_front = (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
        if int(in_front.sum()) < MIN_PROJECTED_POINTS:
            continue
        points_camera = points_camera[in_front]
        z = z[in_front]
        projected = points_camera @ intrinsic.T
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
        in_image = (u >= 0.0) & (u < img_w) & (v >= 0.0) & (v < img_h)
        if int(in_image.sum()) < MIN_PROJECTED_POINTS:
            continue
        u, v, z = u[in_image], v[in_image], z[in_image]
        depth_u = np.clip((u / float(img_w) * depth_w).astype(np.int64), 0, depth_w - 1)
        depth_v = np.clip((v / float(img_h) * depth_h).astype(np.int64), 0, depth_h - 1)

        # Object's own z-buffer: Only the frontmost instance vertex is retained for each Depth pixel.
        pixel_id = depth_v * depth_w + depth_u
        order = np.argsort(z, kind="stable")
        _, first_positions = np.unique(pixel_id[order], return_index=True)
        front = order[first_positions]
        u_front, v_front, z_front = u[front], v[front], z[front]
        depth_values = depth_m[depth_v[front], depth_u[front]]
        depth_valid = (
            np.isfinite(depth_values)
            & (depth_values >= MIN_DEPTH_M)
            & (depth_values <= MAX_DEPTH_M)
        )
        tolerance = DEPTH_CONSISTENCY_TAU_BASE + DEPTH_CONSISTENCY_TAU_REL * z_front
        visible_mask = depth_valid & (np.abs(z_front - depth_values) <= tolerance)
        visible_count = int(visible_mask.sum())
        if visible_count == 0:
            continue
        visible_ratio = visible_count / float(front.size)
        if (
            visible_count < MIN_VISIBLE_DEPTH_PIXELS
            or visible_ratio < MIN_VISIBLE_RATIO
        ):
            continue
        visible.append(
            (
                instance_id,
                float(np.median(depth_values[visible_mask])),
                float(np.median(u_front[visible_mask])),
                float(np.median(v_front[visible_mask])),
                visible_ratio,
                visible_count,
            )
        )
    visible.sort(key=lambda item: item[0])
    count = len(visible)
    return {
        "instance_ids": np.asarray([item[0] for item in visible], dtype=np.int32),
        "surface_depth_m": np.asarray([item[1] for item in visible], dtype=np.float32),
        "center_uv_px": np.asarray(
            [[item[2], item[3]] for item in visible], dtype=np.float32
        ).reshape(count, 2),
        "visible_ratio": np.asarray([item[4] for item in visible], dtype=np.float32),
        "visible_pixel_count": np.asarray(
            [item[5] for item in visible], dtype=np.int32
        ),
        "image_size": np.asarray(image_size, dtype=np.int32),
    }


def validate_frame_npz(
    path: Path, *, expected_image_size: np.ndarray | None = None
) -> int:
    """Re-read compact frame-by-frame NPZ and disallow any per-pixel mask fields."""

    if not path.is_file():
        raise FileNotFoundError(f"Frame-by-frame instance NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != FRAME_NPZ_FIELDS:
            raise RuntimeError(
                f"Unexpected keys in per-frame NPZ: path={path}, keys={sorted(data.files)}"
            )
        values = {key: np.asarray(data[key]) for key in data.files}
    count = int(values["instance_ids"].shape[0])
    if values["instance_ids"].dtype != np.int32 or values["instance_ids"].ndim != 1:
        raise RuntimeError(f"instance_ids must be a one-dimensional int32: path={path}")
    if values["surface_depth_m"].dtype != np.float32 or values[
        "surface_depth_m"
    ].shape != (count,):
        raise RuntimeError(f"surface_depth_m shape/dtype illegal: path={path}")
    if values["center_uv_px"].dtype != np.float32 or values["center_uv_px"].shape != (
        count,
        2,
    ):
        raise RuntimeError(f"center_uv_px shape/dtype invalid: path={path}")
    if values["visible_ratio"].dtype != np.float32 or values["visible_ratio"].shape != (
        count,
    ):
        raise RuntimeError(f"visible_ratio shape/dtype invalid: path={path}")
    if values["visible_pixel_count"].dtype != np.int32 or values[
        "visible_pixel_count"
    ].shape != (count,):
        raise RuntimeError(f"visible_pixel_count shape/dtype invalid: path={path}")
    image_size = values["image_size"]
    if (
        image_size.dtype != np.int32
        or image_size.shape != (2,)
        or np.any(image_size <= 0)
    ):
        raise RuntimeError(
            f"image_size shape/dtype/value illegal: path={path}, value={image_size.tolist()}"
        )
    if expected_image_size is not None and not np.array_equal(
        image_size, expected_image_size
    ):
        raise RuntimeError(
            f"instance/RGB image_size inconsistent: path={path}, stored={image_size.tolist()}, expected={expected_image_size.tolist()}"
        )
    if count:
        if len(np.unique(values["instance_ids"])) != count:
            raise RuntimeError(f"instance_ids duplicate: path={path}")
        if not np.isfinite(values["surface_depth_m"]).all() or np.any(
            values["surface_depth_m"] <= 0.0
        ):
            raise RuntimeError(f"surface_depth_m contains invalid values: path={path}")
        if (
            not np.isfinite(values["center_uv_px"]).all()
            or not np.isfinite(values["visible_ratio"]).all()
        ):
            raise RuntimeError(f"center/ratio contains NaN/Inf: path={path}")
        if np.any(values["visible_ratio"] < MIN_VISIBLE_RATIO) or np.any(
            values["visible_ratio"] > 1.0
        ):
            raise RuntimeError(f"visible_ratio out of bounds: path={path}")
        if np.any(values["visible_pixel_count"] < MIN_VISIBLE_DEPTH_PIXELS):
            raise RuntimeError(
                f"visible_pixel_count below asset threshold: path={path}"
            )
    return count


def validate_scene_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-read scene_points, ensuring all are unpickled numeric arrays."""

    if not path.is_file():
        raise FileNotFoundError(f"scene_points not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != SCENE_POINTS_FIELDS:
            raise RuntimeError(
                f"Unexpected keys in scene_points: path={path}, keys={sorted(data.files)}"
            )
        instance_ids = np.asarray(data["instance_ids"])
        point_offsets = np.asarray(data["point_offsets"])
        points_world = np.asarray(data["points_world"])
    if instance_ids.dtype != np.int32 or instance_ids.ndim != 1:
        raise RuntimeError(f"scene_points.instance_ids illegal: {path}")
    if point_offsets.dtype != np.int64 or point_offsets.shape != (
        len(instance_ids) + 1,
    ):
        raise RuntimeError(f"scene_points.point_offsets illegal: {path}")
    if (
        points_world.dtype != np.float32
        or points_world.ndim != 2
        or points_world.shape[1] != 3
    ):
        raise RuntimeError(f"scene_points.points_world illegal: {path}")
    if (
        point_offsets[0] != 0
        or point_offsets[-1] != len(points_world)
        or np.any(np.diff(point_offsets) <= 0)
    ):
        raise RuntimeError(f"Invalid value for scene_points.point_offsets: {path}")
    if not np.isfinite(points_world).all():
        raise RuntimeError(f"scene_points contain NaN/Inf: {path}")
    return instance_ids, point_offsets, points_world


def scene_is_complete(scene_dir: Path, frames: list[dict[str, Any]]) -> bool:
    """Use the most recently written scene_instances.json to determine if the scene is complete and reusable."""

    metadata_path = scene_dir / "scene_instances.json"
    points_path = scene_dir / "scene_points.npz"
    if not metadata_path.is_file() or not points_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("asset_version") != ASSET_VERSION:
            return False
        if metadata.get("frame_ids") != [int(item["frame_id"]) for item in frames]:
            return False
        ids, offsets, _ = validate_scene_points(points_path)
        if (
            len(ids) != len(metadata.get("instances", []))
            or len(offsets) != len(ids) + 1
        ):
            return False
        for frame in frames:
            validate_frame_npz(scene_dir / f"{frame['frame_id']}.npz")
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
        return False
    return True


def build_scene(
    task: tuple[str, str, list[dict[str, Any]], str, bool, bool],
) -> dict[str, Any]:
    """Worker entry: Loads and processes one scene at a time."""

    source, scene_id, frames, instance_root_raw, overwrite, verify_only = task
    scene_dir = Path(instance_root_raw) / source / scene_id
    if not overwrite and scene_is_complete(scene_dir, frames):
        return {
            "source": source,
            "scene_id": scene_id,
            "frames": len(frames),
            "status": "reused",
        }
    if verify_only:
        raise RuntimeError(
            f"verify-only found incomplete scene: source={source}, scene={scene_id}"
        )

    instances, instance_ids, point_offsets, points_world, raw_files = load_scene_assets(
        source, scene_id
    )
    atomic_write_npz(
        scene_dir / "scene_points.npz",
        instance_ids=instance_ids,
        point_offsets=point_offsets,
        points_world=points_world,
    )
    stored_ids, stored_offsets, stored_points = validate_scene_points(
        scene_dir / "scene_points.npz"
    )
    visible_total = 0
    for frame in frames:
        c2w, intrinsic, image_size, depth_m = load_pose_and_depth(frame)
        payload = project_frame(
            instances,
            stored_ids,
            stored_offsets,
            stored_points,
            c2w,
            intrinsic,
            image_size,
            depth_m,
        )
        frame_path = scene_dir / f"{frame['frame_id']}.npz"
        atomic_write_npz(frame_path, **payload)
        visible_total += validate_frame_npz(frame_path, expected_image_size=image_size)

    # metadata must be written; the scene can only be resumed and reused directly when it exists and frame_ids are complete.
    atomic_write_json(
        {
            "asset_version": ASSET_VERSION,
            "source": source,
            "scene_id": scene_id,
            "raw_files": raw_files,
            "instances": instances,
            "frame_ids": [int(item["frame_id"]) for item in frames],
        },
        scene_dir / "scene_instances.json",
    )
    return {
        "source": source,
        "scene_id": scene_id,
        "frames": len(frames),
        "instances": len(instances),
        "points": int(len(points_world)),
        "visible_records": visible_total,
        "status": "built",
    }


def parse_selected_scenes(raw_values: list[str]) -> set[tuple[str, str]]:
    """Parse explicit SOURCE/SCENE_ID small-sample selection."""

    selected: set[tuple[str, str]] = set()
    for raw in raw_values:
        parts = raw.strip().split("/", maxsplit=1)
        if len(parts) != 2 or parts[0] not in EXPECTED_SOURCES or not parts[1]:
            raise RuntimeError(
                f"--scene format must be scannet/SCENE or scannetpp/SCENE: {raw!r}"
            )
        selected.add((parts[0], parts[1]))
    return selected


def main() -> None:
    """Build the entire scene and output the unique physical frame index."""

    args = parse_args()
    global SCANNET_ROOT, SCANNETPP_ROOT
    SCANNET_ROOT = args.scannet_root.resolve()
    SCANNETPP_ROOT = args.scannetpp_root.resolve()
    ensure_slurm_job()
    if args.num_workers <= 0:
        raise RuntimeError(f"--num-workers must be > 0: {args.num_workers}")
    ordered_frames, grouped = load_manifest(args.manifest.resolve())
    selected = parse_selected_scenes(args.scene)
    if selected:
        missing = sorted(selected - set(grouped))
        if missing:
            raise RuntimeError(f"--scene is not in the scene list: {missing}")
        grouped = {key: grouped[key] for key in sorted(selected)}
        ordered_frames = [
            frame
            for frame in ordered_frames
            if (frame["source"], frame["scene_id"]) in selected
        ]

    instance_root = args.instance_root.resolve()
    tasks = [
        (
            source,
            scene_id,
            frames,
            str(instance_root),
            bool(args.overwrite),
            bool(args.verify_only),
        )
        for (source, scene_id), frames in grouped.items()
    ]
    counters: collections.Counter[str] = collections.Counter()
    completed_frames = 0
    with ProcessPoolExecutor(max_workers=min(args.num_workers, len(tasks))) as pool:
        futures = {pool.submit(build_scene, task): (task[0], task[1]) for task in tasks}
        for completed_scenes, future in enumerate(as_completed(futures), start=1):
            source, scene_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"scene build failed: source={source}, scene={scene_id}"
                ) from exc
            counters[result["status"]] += 1
            completed_frames += int(result["frames"])
            if (
                completed_scenes == 1
                or completed_scenes % 20 == 0
                or completed_scenes == len(tasks)
            ):
                print(
                    f"[instances] scenes={completed_scenes}/{len(tasks)} frames={completed_frames}/{len(ordered_frames)} "
                    f"built={counters['built']} reused={counters['reused']}",
                    flush=True,
                )

    index_rows = []
    for physical_index, frame in enumerate(ordered_frames):
        instance_path = (
            instance_root
            / frame["source"]
            / frame["scene_id"]
            / f"{frame['frame_id']}.npz"
        ).resolve()
        validate_frame_npz(instance_path)
        index_rows.append(
            {
                "physical_index": physical_index,
                **frame,
                "instance_path": str(instance_path),
            }
        )
    if not selected and len(index_rows) != EXPECTED_UNIQUE_FRAMES:
        raise RuntimeError(
            f"Unexpected number of frames in the instance index: actual={len(index_rows)}, expected={EXPECTED_UNIQUE_FRAMES}"
        )
    atomic_write_jsonl(index_rows, args.index_jsonl.resolve())
    print(
        f"[instances] done scenes={len(tasks)} frames={len(index_rows)} index={args.index_jsonl.resolve()} root={instance_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
