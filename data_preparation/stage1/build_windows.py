#!/usr/bin/env python3
"""Build a unique Stage 1 Window geometry asset from the original ScanNet / ScanNet++ sensor data.

This script is a self-contained entry point, not importing or calling any previous Stage 1 / pose_rescue scripts.
Supports four subcommands: scan, smoke, build, validate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import traceback
import uuid
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import lz4.block
import numpy as np
from PIL import Image


SCANNET_ROOT = None
SCANNETPP_ROOT = None
OUTPUT_ROOT = None
FFMPEG = None
FFPROBE = None
EXCLUDED_SCENES = None

# Fix seed to ensure scene-level sampling is unaffected by the number of workers and completion order.
RANDOM_SEED = 20260717
# Six-tier maximum quota per scene; long windows are prioritized for sampling to prevent short windows from consuming available frames for long trajectories.
WINDOW_QUOTAS = {1: 4, 2: 4, 4: 6, 8: 6, 16: 2, 32: 1}
WINDOW_LENGTHS_DESC = (32, 16, 8, 4, 2, 1)
# The proportion of common frames between any two Windows in the same scene cannot exceed this value relative to the shorter Window.
MAX_OVERLAP = 0.5
# Number of rejection attempts for candidates; the higher this value, the more likely the quota will be filled, but sampler CPU time will also increase.
SAMPLE_ATTEMPTS_PER_SLOT = 2000
# Maximum step length on the legal frame sequence within the window to prevent losing continuous scene semantics after crossing excessively long time intervals.
MAX_FRAME_STRIDE = 120
# All RGB images are resized to this [width, height].
TARGET_IMAGE_SIZE = (640, 480)
# ScanNet++ iPhone Depth's original fixed dimensions, still in sensor millimeter units.
SCANNETPP_DEPTH_SIZE = (192, 256)
SCANNETPP_DEPTH_BYTES = SCANNETPP_DEPTH_SIZE[0] * SCANNETPP_DEPTH_SIZE[1] * 2
# ScanNet++'s zlib variant saves float32 metric depth; during asset conversion, it is uniformly restored to uint16 millimeters.
SCANNETPP_DEPTH_FLOAT_BYTES = SCANNETPP_DEPTH_SIZE[0] * SCANNETPP_DEPTH_SIZE[1] * 4

MANIFEST_FIELDS = ("source", "scene_id", "image_paths", "pose_paths", "depth_paths")
SCANNET_SCENE_RE = re.compile(r"(?<![A-Za-z0-9])scene\d{4}_\d{2}(?![A-Za-z0-9])")
SCANNET_DIR_RE = re.compile(r"^scene\d{4}_\d{2}$")
# The current raw pool contains 415 ten-bit IDs and 1 nine-bit ID, both of which must be detectable by the Benchmark scan.
SCANNETPP_TOKEN_RE = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{9,10}(?![0-9A-Fa-f])")

# Numerical tolerance is hard-coded in the code to avoid different acceptance gates guessing their own standards.
BOTTOM_ROW_ATOL = 1e-4
ROTATION_ORTHO_ATOL = 5e-3
ROTATION_DET_ATOL = 5e-3
CANONICAL_ATOL = 1e-5


def parse_args() -> argparse.Namespace:
    """Parse subcommands and parallel parameters for the unique build entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "smoke", "build", "validate"))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=14,
        # Scene-level parallelism; too high will simultaneously compete for shared disk I/O and video decoding CPU.
        help="Number of worker threads (default: 14).",
    )
    parser.add_argument("--scannet-root", type=Path, required=True)
    parser.add_argument("--scannetpp-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--excluded-scenes",
        type=Path,
        default=Path(__file__).with_name("excluded_scenes.json"),
    )
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    return parser.parse_args()


def ensure_compute_node() -> str:
    """Record scheduling information; public entry also allows independent workstation operation."""
    return os.environ.get("SLURM_JOB_ID", "standalone")


def require_file(path: Path) -> None:
    """Check for existence of required files; report an error if any file is missing."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")


def read_exact(handle: BinaryIO, size: int, *, context: str) -> bytes:
    """Read binary bytes precisely; throw an error directly when the file is truncated."""

    payload = handle.read(size)
    if len(payload) != size:
        raise RuntimeError(
            f"{context} File truncation: required {size} bytes, actual {len(payload)} bytes"
        )
    return payload


def utc_now() -> str:
    """Return the UTC ISO time used by the report."""

    return datetime.now(timezone.utc).isoformat()


def temp_path_for(final_path: Path) -> Path:
    """Create a unique temporary path in the same directory as the final file, ensuring that subsequent os.replace does not cross file systems."""

    return final_path.with_name(
        f".{final_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write temporary JSON in the same directory, read and verify consistency, then atomically replace the official file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_path_for(path)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with tmp_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if loaded != payload:
            raise RuntimeError(f"JSON temporary file re-read inconsistency: {tmp_path}")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write temporary JSONL in the same directory, re-read line by line to ensure consistency, then atomically replace the official file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_path_for(path)
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded = load_jsonl(tmp_path)
        if loaded != rows:
            raise RuntimeError(
                f"JSONL temporary file re-read inconsistency: {tmp_path}"
            )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Strictly read JSONL; report errors directly for empty lines or non-object lines."""

    require_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(
                    f"JSONL does not allow empty lines: {path}:{line_no}"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Each line in JSONL must be an object: {path}:{line_no}"
                )
            rows.append(row)
    if not rows:
        raise RuntimeError(f"JSONL cannot be empty: {path}")
    return rows


def discover_input_scenes() -> dict[str, list[str]]:
    """Discover scenes directly from the two source data roots."""

    if not SCANNET_ROOT.is_dir():
        raise FileNotFoundError(f"ScanNet directory not found: {SCANNET_ROOT}")
    if not SCANNETPP_ROOT.is_dir():
        raise FileNotFoundError(f"ScanNet++ directory not found: {SCANNETPP_ROOT}")
    scannet = sorted(
        child.name
        for child in SCANNET_ROOT.iterdir()
        if child.is_dir() and SCANNET_DIR_RE.fullmatch(child.name)
    )
    scannetpp = sorted(
        child.name for child in SCANNETPP_ROOT.iterdir() if child.is_dir()
    )
    if not scannet or not scannetpp:
        raise RuntimeError(
            f"No scenes found: scannet={len(scannet)}, scannetpp={len(scannetpp)}"
        )
    return {"scannet": scannet, "scannetpp": scannetpp}


def build_benchmark_exclusion(input_scenes: dict[str, list[str]]) -> dict[str, Any]:
    """Read the scene exclusion list supplied with the code."""
    excluded = json.loads(EXCLUDED_SCENES.read_text())
    if set(excluded) != {"scannet", "scannetpp"}:
        raise RuntimeError(
            "The exclusion list must explicitly include scannet and scannetpp."
        )
    for source, scenes in excluded.items():
        if len(scenes) != len(set(scenes)):
            raise RuntimeError("Exclude scene duplicates")
    report = dict(
        excluded_scenes=excluded,
        excluded_scene_counts={key: len(value) for key, value in excluded.items()},
        input_scene_counts={key: len(value) for key, value in input_scenes.items()},
        exclusion_file=str(EXCLUDED_SCENES),
        matching_rule="Original Stage1 overall scene exclusion list; input scene still reads from the original data directory",
    )
    atomic_write_json(OUTPUT_ROOT / "reports/benchmark_scene_exclusion.json", report)
    return report


def scale_intrinsic(raw: Any, source_size: tuple[int, int]) -> np.ndarray:
    """Check the original K and rescale to the final 640×480 RGB coordinate system."""

    intrinsic = np.asarray(raw, dtype=np.float64)
    if intrinsic.size == 16:
        intrinsic = intrinsic.reshape(4, 4)[:3, :3]
    elif intrinsic.size == 9:
        intrinsic = intrinsic.reshape(3, 3)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RuntimeError(f"invalid intrinsic: shape={intrinsic.shape}")
    if float(intrinsic[0, 0]) <= 0.0 or float(intrinsic[1, 1]) <= 0.0:
        raise RuntimeError(
            f"intrinsic fx/fy must be positive: fx={intrinsic[0, 0]}, fy={intrinsic[1, 1]}"
        )
    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"Invalid RGB size: {source_size}")
    scaled = intrinsic.copy()
    scaled[0, :] *= TARGET_IMAGE_SIZE[0] / source_width
    scaled[1, :] *= TARGET_IMAGE_SIZE[1] / source_height
    if not (
        0.0 <= scaled[0, 2] <= TARGET_IMAGE_SIZE[0]
        and 0.0 <= scaled[1, 2] <= TARGET_IMAGE_SIZE[1]
    ):
        raise RuntimeError(
            f"After scaling, K principal point exceeds 640x480: cx={scaled[0, 2]}, cy={scaled[1, 2]}"
        )
    return scaled.astype(np.float32)


def normalize_c2w(raw: Any, *, context: str) -> np.ndarray:
    """Strictly check c2w's shape, finite values, homogeneous bottom row, rotation orthogonality, determinant, and invertibility."""

    c2w = np.asarray(raw, dtype=np.float64)
    if c2w.size == 16:
        c2w = c2w.reshape(4, 4)
    if c2w.shape != (4, 4):
        raise RuntimeError(f"{context} c2w shape is not [4,4]: {c2w.shape}")
    if not np.isfinite(c2w).all():
        raise RuntimeError(f"{context} c2w contains NaN/Inf")
    bottom = c2w[3]
    if (
        float(np.linalg.norm(bottom[:3])) > BOTTOM_ROW_ATOL
        or abs(float(bottom[3])) < 1e-8
    ):
        raise RuntimeError(
            f"{context} c2w homogeneous bottom row invalid: {bottom.tolist()}"
        )
    c2w = c2w / float(bottom[3])
    rotation = c2w[:3, :3]
    ortho_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if ortho_error > ROTATION_ORTHO_ATOL:
        raise RuntimeError(
            f"{context} rotation not orthogonal: max_error={ortho_error:.6g}"
        )
    if not math.isfinite(determinant) or abs(determinant - 1.0) > ROTATION_DET_ATOL:
        raise RuntimeError(f"{context} rotation det illegal: det={determinant:.6g}")
    try:
        np.linalg.inv(c2w)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"{context} c2w is not invertible") from exc
    return c2w.astype(np.float32)


def read_sens_header(handle: BinaryIO, *, path: Path) -> dict[str, Any]:
    """Read the ScanNet `.sens` file header, file pointer positioned at the first frame."""

    context = str(path)
    version = struct.unpack("<I", read_exact(handle, 4, context=context))[0]
    if version != 4:
        raise RuntimeError(
            f".sens version does not support: version={version}, path={path}"
        )
    name_length = struct.unpack("<Q", read_exact(handle, 8, context=context))[0]
    read_exact(handle, int(name_length), context=context)
    intrinsic_color = np.frombuffer(
        read_exact(handle, 64, context=context), dtype="<f4"
    ).reshape(4, 4)
    read_exact(handle, 64, context=context)  # extrinsic_color
    read_exact(handle, 64, context=context)  # intrinsic_depth
    read_exact(handle, 64, context=context)  # extrinsic_depth
    color_compression = struct.unpack("<i", read_exact(handle, 4, context=context))[0]
    depth_compression = struct.unpack("<i", read_exact(handle, 4, context=context))[0]
    if color_compression not in (1, 2):
        raise RuntimeError(
            f".sens RGB compression format not supported: type={color_compression}, path={path}"
        )
    if depth_compression != 1:
        raise RuntimeError(
            f".sens Depth is not zlib_ushort: type={depth_compression}, path={path}"
        )
    color_width = struct.unpack("<I", read_exact(handle, 4, context=context))[0]
    color_height = struct.unpack("<I", read_exact(handle, 4, context=context))[0]
    depth_width = struct.unpack("<I", read_exact(handle, 4, context=context))[0]
    depth_height = struct.unpack("<I", read_exact(handle, 4, context=context))[0]
    depth_shift = struct.unpack("<f", read_exact(handle, 4, context=context))[0]
    frame_count = struct.unpack("<Q", read_exact(handle, 8, context=context))[0]
    if depth_shift != 1000.0:
        raise RuntimeError(
            f".sens depth_shift is not 1000, cannot preserve millimeter scale: {depth_shift}, path={path}"
        )
    if min(color_width, color_height, depth_width, depth_height, frame_count) <= 0:
        raise RuntimeError(f".sens size or frame count invalid: path={path}")
    return {
        "intrinsic_color": intrinsic_color,
        "color_size": (int(color_width), int(color_height)),
        "depth_size": (int(depth_height), int(depth_width)),
        "frame_count": int(frame_count),
    }


def scan_scannet_scene(scene: str) -> dict[str, Any]:
    """Scan a ScanNet `.sens` Pose/K, passing only valid c2w frames to the sampler."""

    path = SCANNET_ROOT / scene / f"{scene}.sens"
    require_file(path)
    poses: dict[int, np.ndarray] = {}
    invalid_reasons: Counter[str] = Counter()
    with path.open("rb") as handle:
        header = read_sens_header(handle, path=path)
        intrinsic = scale_intrinsic(header["intrinsic_color"], header["color_size"])
        for frame_id in range(header["frame_count"]):
            c2w_raw = np.frombuffer(
                read_exact(handle, 64, context=str(path)), dtype="<f4"
            ).reshape(4, 4)
            read_exact(handle, 16, context=str(path))
            color_size = struct.unpack("<Q", read_exact(handle, 8, context=str(path)))[
                0
            ]
            depth_size = struct.unpack("<Q", read_exact(handle, 8, context=str(path)))[
                0
            ]
            handle.seek(int(color_size + depth_size), os.SEEK_CUR)
            try:
                poses[frame_id] = normalize_c2w(
                    c2w_raw, context=f"scannet/{scene}/{frame_id}"
                )
            except RuntimeError as exc:
                invalid_reasons[str(exc).split(":", 1)[-1].strip()] += 1
    if not poses:
        raise RuntimeError(f"ScanNet scene has no valid c2w: {path}")
    return {
        "path": str(path),
        "poses": poses,
        "intrinsics": {frame_id: intrinsic for frame_id in poses},
        "total_frames": header["frame_count"],
        "invalid_reasons": dict(invalid_reasons),
        "depth_size": header["depth_size"],
    }


def probe_scannetpp_video(scene: str) -> tuple[Path, tuple[int, int]]:
    """Use standalone ffprobe to read the original width and height of ScanNet++ RGB."""

    require_file(FFPROBE)
    video_path = SCANNETPP_ROOT / scene / "iphone/rgb.mkv"
    require_file(video_path)
    output = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    payload = json.loads(output)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise RuntimeError(
            f"ScanNet++ video stream count is not 1: scene={scene}, streams={len(streams or [])}"
        )
    source_size = (int(streams[0]["width"]), int(streams[0]["height"]))
    if min(source_size) <= 0:
        raise RuntimeError(
            f"ScanNet++ video dimension invalid: scene={scene}, size={source_size}"
        )
    return video_path, source_size


def scan_scannetpp_scene(scene: str) -> dict[str, Any]:
    """ScanNet++ JSON, accepts only frame-by-frame `aligned_pose` and intrinsic."""

    iphone_dir = SCANNETPP_ROOT / scene / "iphone"
    pose_path = iphone_dir / "pose_intrinsic_imu.json"
    depth_path = iphone_dir / "depth.bin"
    require_file(pose_path)
    require_file(depth_path)
    video_path, source_size = probe_scannetpp_video(scene)
    with pose_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ScanNet++ pose JSON must be an object: {pose_path}")

    poses: dict[int, np.ndarray] = {}
    intrinsics: dict[int, np.ndarray] = {}
    invalid_reasons: Counter[str] = Counter()
    frame_entries = [
        (key, value) for key, value in payload.items() if str(key).startswith("frame_")
    ]
    for key, entry in sorted(
        frame_entries, key=lambda item: int(str(item[0]).split("_")[1])
    ):
        frame_id = int(str(key).split("_")[1])
        try:
            if not isinstance(entry, dict):
                raise RuntimeError("frame entry is not an object")
            if "aligned_pose" not in entry:
                raise RuntimeError("Missing aligned_pose")
            if "intrinsic" not in entry:
                raise RuntimeError("Missing intrinsic")
            poses[frame_id] = normalize_c2w(
                entry["aligned_pose"], context=f"scannetpp/{scene}/{frame_id}"
            )
            intrinsics[frame_id] = scale_intrinsic(entry["intrinsic"], source_size)
        except RuntimeError as exc:
            poses.pop(frame_id, None)
            intrinsics.pop(frame_id, None)
            invalid_reasons[str(exc).split(":", 1)[-1].strip()] += 1
    if not poses:
        raise RuntimeError(
            f"ScanNet++ scene has no valid aligned_pose + K: {pose_path}"
        )
    return {
        "video_path": str(video_path),
        "depth_path": str(depth_path),
        "poses": poses,
        "intrinsics": intrinsics,
        "total_frames": len(frame_entries),
        "invalid_reasons": dict(invalid_reasons),
        "depth_size": SCANNETPP_DEPTH_SIZE,
    }


def scene_seed(source: str, scene: str) -> int:
    """Use a stable SHA256-derived scene seed to avoid Python hash variations across processes."""

    digest = hashlib.sha256(f"{RANDOM_SEED}:{source}:{scene}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def overlap_is_valid(candidate: set[int], accepted: list[set[int]]) -> bool:
    """Check the pairwise overlap rates between candidate Windows and accepted Windows in the same scene."""

    for previous in accepted:
        overlap = len(candidate & previous) / min(len(candidate), len(previous))
        if overlap > MAX_OVERLAP:
            return False
    return True


def sample_windows(
    source: str, scene: str, valid_frame_ids: list[int]
) -> list[list[int]]:
    """Sample six windows from a valid timeline according to fixed seed, quota, and overlap upper limit."""

    import random

    frame_ids = sorted(set(int(frame_id) for frame_id in valid_frame_ids))
    if not frame_ids:
        raise RuntimeError(f"Scene has no valid frames: {source}/{scene}")
    rng = random.Random(scene_seed(source, scene))
    accepted_sets: list[set[int]] = []
    accepted_windows: list[list[int]] = []
    frame_count = len(frame_ids)

    for length in WINDOW_LENGTHS_DESC:
        if frame_count < length:
            continue
        for _slot in range(WINDOW_QUOTAS[length]):
            selected: list[int] | None = None
            for _attempt in range(SAMPLE_ATTEMPTS_PER_SLOT):
                if length == 1:
                    positions = [rng.randrange(frame_count)]
                else:
                    max_stride = min(
                        MAX_FRAME_STRIDE, (frame_count - 1) // (length - 1)
                    )
                    if max_stride < 1:
                        break
                    stride = rng.randint(1, max_stride)
                    span = (length - 1) * stride + 1
                    start = rng.randrange(frame_count - span + 1)
                    positions = [start + offset * stride for offset in range(length)]
                candidate = [frame_ids[position] for position in positions]
                candidate_set = set(candidate)
                if len(candidate_set) != length:
                    continue
                if overlap_is_valid(candidate_set, accepted_sets):
                    selected = candidate
                    break
            if selected is None:
                # When the scene cannot accommodate more valid Windows, naturally reduce the number in accordance with the contract, without adding duplicate frames.
                break
            accepted_windows.append(selected)
            accepted_sets.append(set(selected))

    accepted_windows.sort(key=lambda window: (len(window), window))
    validate_sampled_windows(accepted_windows, context=f"{source}/{scene}")
    return accepted_windows


def validate_sampled_windows(windows: list[list[int]], *, context: str) -> None:
    """Independent acceptance sampler's tier, quota, order, unique frame, and pairwise overlap."""

    counts: Counter[int] = Counter()
    seen: set[tuple[int, ...]] = set()
    for window in windows:
        length = len(window)
        if length not in WINDOW_QUOTAS:
            raise RuntimeError(f"{context} Invalid frame count for Window: {length}")
        if window != sorted(window) or len(set(window)) != length:
            raise RuntimeError(
                f"{context} Window frame order or duplicate illegal: {window}"
            )
        identity = tuple(window)
        if identity in seen:
            raise RuntimeError(f"{context} Duplicate Window: {identity}")
        seen.add(identity)
        counts[length] += 1
    for length, count in counts.items():
        if count > WINDOW_QUOTAS[length]:
            raise RuntimeError(f"{context} {length} Frame Window Overquota: {count}")
    sets = [set(window) for window in windows]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            overlap = len(sets[left] & sets[right]) / min(
                len(sets[left]), len(sets[right])
            )
            if overlap > MAX_OVERLAP:
                raise RuntimeError(
                    f"{context} Window overlap rate exceeds limit: {overlap}"
                )


def load_scannet_payloads(
    scene: str, target_ids: list[int]
) -> dict[int, tuple[bytes, np.ndarray]]:
    """Second pass streaming scan `.sens`, only decode target frame RGB and original uint16 Depth."""

    path = SCANNET_ROOT / scene / f"{scene}.sens"
    remaining = set(target_ids)
    output: dict[int, tuple[bytes, np.ndarray]] = {}
    with path.open("rb") as handle:
        header = read_sens_header(handle, path=path)
        depth_height, depth_width = header["depth_size"]
        for frame_id in range(header["frame_count"]):
            read_exact(handle, 64 + 16, context=str(path))
            color_size = struct.unpack("<Q", read_exact(handle, 8, context=str(path)))[
                0
            ]
            depth_size = struct.unpack("<Q", read_exact(handle, 8, context=str(path)))[
                0
            ]
            if frame_id not in remaining:
                handle.seek(int(color_size + depth_size), os.SEEK_CUR)
                continue
            color_bytes = read_exact(handle, int(color_size), context=str(path))
            compressed_depth = read_exact(handle, int(depth_size), context=str(path))
            raw_depth = zlib.decompress(compressed_depth)
            expected_bytes = depth_height * depth_width * 2
            if len(raw_depth) != expected_bytes:
                raise RuntimeError(
                    f"ScanNet Depth decompression size error: scene={scene}, frame={frame_id}, expected={expected_bytes}, actual={len(raw_depth)}"
                )
            depth = (
                np.frombuffer(raw_depth, dtype="<u2")
                .reshape(depth_height, depth_width)
                .copy()
            )
            if not np.any(depth > 0):
                raise RuntimeError(
                    f"ScanNet Depth all zeros: scene={scene}, frame={frame_id}"
                )
            with Image.open(io.BytesIO(color_bytes)) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise RuntimeError(
                        f"ScanNet RGB size invalid: scene={scene}, frame={frame_id}"
                    )
            output[frame_id] = (color_bytes, depth)
            remaining.remove(frame_id)
            if not remaining:
                break
    if remaining:
        raise RuntimeError(
            f"ScanNet `.sens` missing target frames: scene={scene}, missing={sorted(remaining)[:20]}"
        )
    return output


def decode_scannetpp_depth_block(
    compressed: bytes, *, scene: str, frame_id: int
) -> np.ndarray:
    """Decode frame-by-frame LZ4 uint16 or zlib float32 Depth blocks in official format."""

    lz4_error: Exception | None = None
    try:
        raw = lz4.block.decompress(compressed, uncompressed_size=SCANNETPP_DEPTH_BYTES)
        if len(raw) != SCANNETPP_DEPTH_BYTES:
            raise RuntimeError(
                f"Decompression size for LZ4 {len(raw)}, expected {SCANNETPP_DEPTH_BYTES}"
            )
        depth = np.frombuffer(raw, dtype="<u2").reshape(SCANNETPP_DEPTH_SIZE).copy()
    except Exception as exc:
        lz4_error = exc
        try:
            raw = zlib.decompress(compressed, wbits=-zlib.MAX_WBITS)
        except Exception as zlib_exc:
            raise RuntimeError(
                f"ScanNet++ Depth block is neither a valid LZ4 nor a valid zlib: scene={scene}, frame={frame_id}, lz4_error={lz4_error}, zlib_error={zlib_exc}"
            ) from zlib_exc
        if len(raw) != SCANNETPP_DEPTH_FLOAT_BYTES:
            raise RuntimeError(
                f"ScanNet++ zlib Depth decompression size error: scene={scene}, frame={frame_id}, bytes={len(raw)}"
            )
        depth_m = np.frombuffer(raw, dtype="<f4").reshape(SCANNETPP_DEPTH_SIZE)
        if (
            not np.isfinite(depth_m).all()
            or np.any(depth_m < 0)
            or np.any(depth_m > 65.535)
        ):
            raise RuntimeError(
                f"ScanNet++ zlib Depth invalid metric value: scene={scene}, frame={frame_id}"
            )
        # Decode zlib float32 depth blocks into millimeters according to the sensor format.
        depth = (depth_m * 1000.0).astype("<u2")
    if not np.any(depth > 0):
        raise RuntimeError(
            f"ScanNet++ Depth all zeros: scene={scene}, frame={frame_id}"
        )
    return depth


def load_scannetpp_depth(scene: str, target_ids: list[int]) -> dict[int, np.ndarray]:
    """ScanNet++ `depth.bin` sequential scan, decompress only the LZ4/zlib blocks of target frames."""

    path = SCANNETPP_ROOT / scene / "iphone/depth.bin"
    require_file(path)
    remaining = set(target_ids)
    output: dict[int, np.ndarray] = {}
    with path.open("rb") as handle:
        frame_id = 0
        while remaining:
            size_bytes = handle.read(4)
            if len(size_bytes) != 4:
                raise RuntimeError(
                    f"ScanNet++ depth.bin frame count insufficient: scene={scene}, missing={sorted(remaining)[:20]}"
                )
            chunk_size = struct.unpack("<I", size_bytes)[0]
            if chunk_size <= 0:
                raise RuntimeError(
                    f"ScanNet++ depth.bin chunk_size illegal: scene={scene}, frame={frame_id}"
                )
            if frame_id in remaining:
                compressed = read_exact(handle, chunk_size, context=str(path))
                output[frame_id] = decode_scannetpp_depth_block(
                    compressed, scene=scene, frame_id=frame_id
                )
                remaining.remove(frame_id)
            else:
                handle.seek(chunk_size, os.SEEK_CUR)
            frame_id += 1
    return output


def prepare_jpeg_from_bytes(color_bytes: bytes, final_path: Path) -> Path:
    """Convert all sensor RGB to 640×480 JPEG, and after re-reading temporary files, return."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_path_for(final_path)
    try:
        with Image.open(io.BytesIO(color_bytes)) as image:
            resized = image.convert("RGB").resize(
                TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS
            )
            resized.save(tmp_path, format="JPEG", quality=92)
        validate_jpeg(tmp_path)
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def prepare_npz(final_path: Path, payload: dict[str, np.ndarray], *, kind: str) -> Path:
    """Write NPZ temporary files in the same directory and re-read them according to the Pose/Depth contract for verification."""

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_path_for(final_path).with_suffix(".npz")
    try:
        np.savez_compressed(tmp_path, **payload)
        if kind == "pose":
            validate_pose_npz(tmp_path)
        elif kind == "depth":
            validate_depth_npz(tmp_path)
        else:
            raise RuntimeError(f"Unknown NPZ type: {kind}")
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def prepare_scannetpp_jpegs(
    video_path: Path, scene_dir: Path, target_ids: list[int]
) -> dict[int, Path]:
    """Single-pass sequential decoding of ScanNet++ video, writing original frames as temporary JPEGs in the same directory."""

    require_file(FFMPEG)
    scene_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}-{uuid.uuid4().hex}"
    pattern = scene_dir / f".__ffmpeg_tmp_{tag}_%06d.jpg"
    select_expr = "+".join(f"eq(n\\,{frame_id})" for frame_id in target_ids)
    subprocess.run(
        [
            str(FFMPEG),
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select={select_expr},scale={TARGET_IMAGE_SIZE[0]}:{TARGET_IMAGE_SIZE[1]}",
            "-vsync",
            "0",
            "-q:v",
            "3",
            "-start_number",
            "0",
            "-y",
            str(pattern),
        ],
        check=True,
    )
    output: dict[int, Path] = {}
    try:
        for output_index, frame_id in enumerate(target_ids):
            tmp_path = scene_dir / f".__ffmpeg_tmp_{tag}_{output_index:06d}.jpg"
            if not tmp_path.is_file():
                raise RuntimeError(
                    f"ScanNet++ RGB frame missing: video={video_path}, frame={frame_id}"
                )
            validate_jpeg(tmp_path)
            output[frame_id] = tmp_path
        extras = sorted(scene_dir.glob(f".__ffmpeg_tmp_{tag}_*.jpg"))
        if len(extras) != len(target_ids):
            raise RuntimeError(
                f"ScanNet++ RGB output count inconsistency: video={video_path}, expected={len(target_ids)}, actual={len(extras)}"
            )
        return output
    except Exception:
        for path in scene_dir.glob(f".__ffmpeg_tmp_{tag}_*.jpg"):
            path.unlink(missing_ok=True)
        raise


def validate_jpeg(path: Path) -> None:
    """Re-read JPEG, confirm RGB dimensions are strictly 640×480."""

    require_file(path)
    with Image.open(path) as image:
        image.load()
        if image.size != TARGET_IMAGE_SIZE:
            raise RuntimeError(
                f"RGB size is not 640x480: path={path}, size={image.size}"
            )
        if image.mode != "RGB":
            raise RuntimeError(
                f"RGB file mode is not RGB: path={path}, mode={image.mode}"
            )


def validate_pose_npz(path: Path) -> dict[str, Any]:
    """Re-read frame-wise Pose NPZ files, verify c2w, K, RGB dimensions, and Pose source labels."""

    require_file(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"c2w", "intrinsic", "image_size", "pose_type"}
        if set(data.files) != required:
            raise RuntimeError(
                f"Pose NPZ field mismatch: path={path}, fields={data.files}"
            )
        c2w = normalize_c2w(data["c2w"], context=str(path))
        intrinsic = np.asarray(data["intrinsic"])
        image_size = np.asarray(data["image_size"])
        pose_type = str(np.asarray(data["pose_type"]).item())
    if c2w.dtype != np.float32 or intrinsic.dtype != np.float32:
        raise RuntimeError(f"Pose c2w/intrinsic must be float32: {path}")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RuntimeError(f"Invalid pose intrinsic: {path}")
    if float(intrinsic[0, 0]) <= 0.0 or float(intrinsic[1, 1]) <= 0.0:
        raise RuntimeError(f"Intrinsic parameters fx/fy are not positive: {path}")
    # When re-reading independently, verify the principal point and homogeneous bottom row again, directly proving that K belongs to the final 640×480 RGB coordinate system.
    if not (
        0.0 <= float(intrinsic[0, 2]) <= 640.0
        and 0.0 <= float(intrinsic[1, 2]) <= 480.0
    ):
        raise RuntimeError(
            f"Pose intrinsic principal point exceeds 640x480: path={path}, K={intrinsic.tolist()}"
        )
    if not np.allclose(
        intrinsic[2], np.asarray([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5
    ):
        raise RuntimeError(
            f"Pose intrinsic homogeneous last row invalid: path={path}, row={intrinsic[2].tolist()}"
        )
    if (
        image_size.dtype != np.int32
        or image_size.shape != (2,)
        or image_size.tolist() != [640, 480]
    ):
        raise RuntimeError(
            f"Pose image_size must be int32 [640,480]: path={path}, value={image_size}"
        )
    return {"c2w": c2w, "pose_type": pose_type}


def validate_depth_npz(path: Path) -> dict[str, Any]:
    """Re-read frame-by-frame depth NPZ, verify uint16, [H,W], and valid non-zero pixels."""

    require_file(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"depth_map", "depth_size"}:
            raise RuntimeError(
                f"Depth NPZ field mismatch: path={path}, fields={data.files}"
            )
        depth = np.asarray(data["depth_map"])
        depth_size = np.asarray(data["depth_size"])
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise RuntimeError(
            f"Depth must be uint16 [H,W]: path={path}, shape={depth.shape}, dtype={depth.dtype}"
        )
    if depth_size.dtype != np.int32 or depth_size.shape != (2,):
        raise RuntimeError(f"depth_size must be int32 [H,W]: {path}")
    if depth_size.tolist() != [depth.shape[0], depth.shape[1]]:
        raise RuntimeError(f"depth_size does not match depth_map: {path}")
    if not np.any(depth > 0):
        raise RuntimeError(f"No valid non-zero pixels in Depth: {path}")
    return {"depth_size": depth_size.tolist(), "nonzero": int(np.count_nonzero(depth))}


def frame_paths(
    root: Path, source: str, scene: str, frame_id: int
) -> tuple[Path, Path, Path]:
    """Generate unique standard paths for RGB, Pose, and Depth with fixed physical frame identities."""

    stem = str(frame_id)
    return (
        root / "images" / source / scene / f"{stem}.jpg",
        root / "gt_pose" / source / scene / f"{stem}.npz",
        root / "gt_depth" / source / scene / f"{stem}.npz",
    )


def build_scene(source: str, scene: str, root_text: str) -> dict[str, Any]:
    """Scan and sample a scene, decode its sensors, and atomically save each physical frame."""

    root = Path(root_text)
    if source == "scannet":
        scan = scan_scannet_scene(scene)
    elif source == "scannetpp":
        scan = scan_scannetpp_scene(scene)
    else:
        raise RuntimeError(f"Unknown source: {source}")
    windows = sample_windows(source, scene, sorted(scan["poses"]))
    if not windows:
        raise RuntimeError(f"scene cannot capture any Window: {source}/{scene}")
    target_ids = sorted({frame_id for window in windows for frame_id in window})

    if source == "scannet":
        payloads = load_scannet_payloads(scene, target_ids)
        depths = {frame_id: payloads[frame_id][1] for frame_id in target_ids}
    else:
        depths = load_scannetpp_depth(scene, target_ids)

    prepared: list[tuple[Path, Path]] = []
    prepared_tmp_paths: list[Path] = []
    try:
        if source == "scannet":
            rgb_temps = {
                frame_id: prepare_jpeg_from_bytes(
                    payloads[frame_id][0], frame_paths(root, source, scene, frame_id)[0]
                )
                for frame_id in target_ids
            }
        else:
            video_path = Path(scan["video_path"])
            rgb_scene_dir = root / "images" / source / scene
            rgb_temps = prepare_scannetpp_jpegs(video_path, rgb_scene_dir, target_ids)
        prepared_tmp_paths.extend(rgb_temps.values())

        for frame_id in target_ids:
            image_path, pose_path, depth_path = frame_paths(
                root, source, scene, frame_id
            )
            prepared.append((rgb_temps[frame_id], image_path))
            pose_type = "sens_c2w" if source == "scannet" else "aligned_pose"
            pose_tmp = prepare_npz(
                pose_path,
                {
                    "c2w": np.asarray(scan["poses"][frame_id], dtype=np.float32),
                    "intrinsic": np.asarray(
                        scan["intrinsics"][frame_id], dtype=np.float32
                    ),
                    "image_size": np.asarray(TARGET_IMAGE_SIZE, dtype=np.int32),
                    "pose_type": np.asarray(pose_type),
                },
                kind="pose",
            )
            depth = np.asarray(depths[frame_id], dtype=np.uint16)
            depth_tmp = prepare_npz(
                depth_path,
                {
                    "depth_map": depth,
                    "depth_size": np.asarray(depth.shape, dtype=np.int32),
                },
                kind="depth",
            )
            prepared_tmp_paths.extend((pose_tmp, depth_tmp))
            prepared.extend(((pose_tmp, pose_path), (depth_tmp, depth_path)))

        # All temporary RGB/NPZ files have been re-read and replaced atomically with the official paths, file by file.
        for tmp_path, final_path in prepared:
            os.replace(tmp_path, final_path)

        rows: list[dict[str, Any]] = []
        for window in windows:
            triplets = [
                frame_paths(root, source, scene, frame_id) for frame_id in window
            ]
            rows.append(
                {
                    "source": source,
                    "scene_id": scene,
                    "image_paths": [str(item[0]) for item in triplets],
                    "pose_paths": [str(item[1]) for item in triplets],
                    "depth_paths": [str(item[2]) for item in triplets],
                }
            )
        return {
            "source": source,
            "scene_id": scene,
            "rows": rows,
            "unique_frames": len(target_ids),
            "total_frames": int(scan["total_frames"]),
            "invalid_frame_reasons": scan["invalid_reasons"],
        }
    finally:
        for path in prepared_tmp_paths:
            path.unlink(missing_ok=True)


def parse_standard_frame_path(
    root: Path, path_text: str, kind: str
) -> tuple[str, str, int]:
    """Parse `(source, scene, frame_id)` from absolute standard asset paths; reject non-standard paths."""

    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError(f"manifest path is not an absolute path: {path}")
    expected_dir = {"image": "images", "pose": "gt_pose", "depth": "gt_depth"}[kind]
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Manifest path is not in the output root directory: {path}"
        ) from exc
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != expected_dir
        or parts[1] not in {"scannet", "scannetpp"}
    ):
        raise RuntimeError(f"manifest {kind} Invalid path layout: {path}")
    expected_suffix = ".jpg" if kind == "image" else ".npz"
    if Path(parts[3]).suffix != expected_suffix:
        raise RuntimeError(f"manifest {kind} Invalid suffix: {path}")
    try:
        frame_id = int(Path(parts[3]).stem)
    except ValueError as exc:
        raise RuntimeError(f"manifest frame_id is not an integer: {path}") from exc
    return parts[1], parts[2], frame_id


def validate_frame_triplet(item: tuple[str, str, str, str, str, int]) -> dict[str, Any]:
    """Parallelly re-read a unique physical frame's RGB, Pose, Depth, and verify source labels."""

    image_text, pose_text, depth_text, source, scene, frame_id = item
    image_path, pose_path, depth_path = (
        Path(image_text),
        Path(pose_text),
        Path(depth_text),
    )
    validate_jpeg(image_path)
    pose = validate_pose_npz(pose_path)
    depth = validate_depth_npz(depth_path)
    expected_pose_type = "sens_c2w" if source == "scannet" else "aligned_pose"
    if pose["pose_type"] != expected_pose_type:
        raise RuntimeError(
            f"Pose source label error: {source}/{scene}/{frame_id}, expected={expected_pose_type}, actual={pose['pose_type']}"
        )
    return {
        "source": source,
        "scene_id": scene,
        "frame_id": frame_id,
        "c2w": pose["c2w"],
        "depth_size": depth["depth_size"],
    }


def actual_asset_files(root: Path, subdir: str, suffix: str) -> set[str]:
    """/asset/subtree/file.txt"""

    asset_root = root / subdir
    if not asset_root.is_dir():
        raise FileNotFoundError(f"Asset directory does not exist: {asset_root}")
    return {str(path) for path in asset_root.rglob(f"*{suffix}") if path.is_file()}


def validate_manifest_rows(
    rows: list[dict[str, Any]],
    root: Path,
    excluded: dict[str, set[str]],
    num_workers: int,
) -> dict[str, Any]:
    """Re-read the acceptance manifest and all official assets field by field, window by window, and physical frame by physical frame."""

    if num_workers <= 0:
        raise RuntimeError(f"num_workers must be positive: {num_workers}")
    window_counts: dict[str, Counter[int]] = {
        "scannet": Counter(),
        "scannetpp": Counter(),
    }
    windows_by_scene: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    frame_triplets: dict[tuple[str, str, int], tuple[str, str, str]] = {}
    window_identities: set[tuple[str, str, tuple[str, ...]]] = set()

    for row_index, row in enumerate(rows):
        if tuple(row.keys()) != MANIFEST_FIELDS:
            raise RuntimeError(
                f"windows.jsonl row {row_index + 1} The row must strictly contain five predefined fields: keys={tuple(row.keys())}"
            )
        source = row["source"]
        scene = row["scene_id"]
        if (
            source not in {"scannet", "scannetpp"}
            or not isinstance(scene, str)
            or not scene
        ):
            raise RuntimeError(f"manifest source/scene illegal: row={row_index + 1}")
        if scene in excluded[source]:
            raise RuntimeError(
                f"Benchmark overlapping scene entered manifest: {source}/{scene}"
            )
        lists = [row["image_paths"], row["pose_paths"], row["depth_paths"]]
        if any(not isinstance(items, list) for items in lists):
            raise RuntimeError(
                f"manifest three path fields must be list: row={row_index + 1}"
            )
        if not (len(lists[0]) == len(lists[1]) == len(lists[2])):
            raise RuntimeError(
                f"manifest has three path lists with inconsistent lengths: row={row_index + 1}"
            )
        length = len(lists[0])
        if length not in WINDOW_QUOTAS:
            raise RuntimeError(
                f"manifest Window frame count invalid: row={row_index + 1}, length={length}"
            )
        identity = (source, scene, tuple(lists[0]))
        if identity in window_identities:
            raise RuntimeError(
                f"manifest Window identity duplicate: row={row_index + 1}, identity={identity}"
            )
        window_identities.add(identity)

        frame_ids: list[int] = []
        for image_text, pose_text, depth_text in zip(*lists):
            image_id = parse_standard_frame_path(root, image_text, "image")
            pose_id = parse_standard_frame_path(root, pose_text, "pose")
            depth_id = parse_standard_frame_path(root, depth_text, "depth")
            if image_id != pose_id or image_id != depth_id:
                raise RuntimeError(
                    f"manifest RGB/Pose/Depth frame identity misalignment: row={row_index + 1}"
                )
            if image_id[:2] != (source, scene):
                raise RuntimeError(
                    f"manifest row source/scene path inconsistency: row={row_index + 1}"
                )
            frame_ids.append(image_id[2])
            old = frame_triplets.setdefault(
                image_id, (image_text, pose_text, depth_text)
            )
            if old != (image_text, pose_text, depth_text):
                raise RuntimeError(
                    f"The same physical frame maps to multiple files: identity={image_id}"
                )
        if frame_ids != sorted(frame_ids) or len(set(frame_ids)) != length:
            raise RuntimeError(
                f"manifest Window frame order or duplicate illegal: row={row_index + 1}, ids={frame_ids}"
            )
        windows_by_scene[(source, scene)].append(frame_ids)
        window_counts[source][length] += 1

    for (source, scene), scene_windows in windows_by_scene.items():
        validate_sampled_windows(scene_windows, context=f"manifest {source}/{scene}")

    expected_images = {triplet[0] for triplet in frame_triplets.values()}
    expected_poses = {triplet[1] for triplet in frame_triplets.values()}
    expected_depths = {triplet[2] for triplet in frame_triplets.values()}
    actual_images = actual_asset_files(root, "images", ".jpg")
    actual_poses = actual_asset_files(root, "gt_pose", ".npz")
    actual_depths = actual_asset_files(root, "gt_depth", ".npz")
    for label, expected, actual in (
        ("RGB", expected_images, actual_images),
        ("Pose", expected_poses, actual_poses),
        ("Depth", expected_depths, actual_depths),
    ):
        if expected != actual:
            raise RuntimeError(
                f"{label} Inconsistent between official file and manifest: missing={sorted(expected - actual)[:10]}, orphan={sorted(actual - expected)[:10]}"
            )

    tasks = [
        (paths[0], paths[1], paths[2], source, scene, frame_id)
        for (source, scene, frame_id), paths in sorted(frame_triplets.items())
    ]
    c2ws: dict[tuple[str, str, int], np.ndarray] = {}
    depth_sizes: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(validate_frame_triplet, task): task for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            identity = (result["source"], result["scene_id"], result["frame_id"])
            c2ws[identity] = result["c2w"]
            depth_sizes[f"{result['depth_size'][0]}x{result['depth_size'][1]}"] += 1
            if completed % 5000 == 0 or completed == len(tasks):
                print(f"[validate frames] {completed}/{len(tasks)}", flush=True)

    max_canonical_error = 0.0
    for (source, scene), scene_windows in windows_by_scene.items():
        for frame_ids in scene_windows:
            w2cs = [
                np.linalg.inv(c2ws[(source, scene, frame_id)].astype(np.float64))
                for frame_id in frame_ids
            ]
            reference_c2w = np.linalg.inv(w2cs[0])
            relatives = [w2c @ reference_c2w for w2c in w2cs]
            error = float(np.max(np.abs(relatives[0] - np.eye(4))))
            max_canonical_error = max(max_canonical_error, error)
            if error > CANONICAL_ATOL:
                raise RuntimeError(
                    f"After canonicalizing the first frame of Window, it is not [I|0]: {source}/{scene}, error={error}"
                )

    return {
        "manifest_row_count": len(rows),
        "window_counts": {
            source: {
                str(length): int(window_counts[source][length])
                for length in sorted(WINDOW_QUOTAS)
            }
            for source in ("scannet", "scannetpp")
        },
        "unique_frame_counts": {
            "rgb": len(expected_images),
            "pose": len(expected_poses),
            "depth": len(expected_depths),
        },
        "successful_scene_counts_from_manifest": {
            source: len(
                {
                    scene
                    for current_source, scene in windows_by_scene
                    if current_source == source
                }
            )
            for source in ("scannet", "scannetpp")
        },
        "depth_sizes": dict(sorted(depth_sizes.items())),
        "max_first_frame_canonical_error": max_canonical_error,
        "benchmark_overlap_scene_count": 0,
        "all_paths_exist_and_reread": True,
        "all_scannetpp_pose_type": "aligned_pose",
    }


def referenced_asset_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Summarize the reference sets of three formal asset categories from the manifest line."""

    return {
        "images": {path for row in rows for path in row["image_paths"]},
        "gt_pose": {path for row in rows for path in row["pose_paths"]},
        "gt_depth": {path for row in rows for path in row["depth_paths"]},
    }


def cleanup_orphan_assets(root: Path, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Only delete failed rerun leftovers from the three formal asset subtrees not referenced by the current manifest."""

    referenced = referenced_asset_sets(rows)
    removed: dict[str, int] = {"images": 0, "gt_pose": 0, "gt_depth": 0, "tmp": 0}
    suffixes = {"images": ".jpg", "gt_pose": ".npz", "gt_depth": ".npz"}
    for subdir, suffix in suffixes.items():
        asset_root = root / subdir
        if not asset_root.is_dir():
            continue
        for path in sorted(asset_root.rglob("*"), reverse=True):
            if path.is_file():
                if ".tmp-" in path.name or path.name.startswith(".__ffmpeg_tmp_"):
                    path.unlink()
                    removed["tmp"] += 1
                elif path.suffix == suffix and str(path) not in referenced[subdir]:
                    path.unlink()
                    removed[subdir] += 1
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    return removed


def directory_bytes(root: Path) -> dict[str, int]:
    """Statistically count the actual byte size of the formal output root directory and its primary subtrees."""

    def size(path: Path) -> int:
        return (
            sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            if path.exists()
            else 0
        )

    result = {
        name: size(root / name)
        for name in (
            "manifests",
            "images",
            "gt_pose",
            "gt_depth",
            "reports",
            "scripts",
            "docs",
        )
    }
    result["total"] = size(root)
    return result


def write_validation_report(report: dict[str, Any]) -> None:
    """Atomically write the formal report and iterate to ensure the disk byte count and directory state after writing are identical."""

    report_path = OUTPUT_ROOT / "reports/validation_report.json"
    for _attempt in range(10):
        report["disk_bytes"] = directory_bytes(OUTPUT_ROOT)
        atomic_write_json(report_path, report)
        actual = directory_bytes(OUTPUT_ROOT)
        if report["disk_bytes"] == actual:
            return
    raise RuntimeError(
        f"Output size changed after the report was written: recorded={report['disk_bytes']}, actual={directory_bytes(OUTPUT_ROOT)}"
    )


def run_scan() -> dict[str, Any]:
    """Discover the original scene pool and use the original exclusion list provided with the code."""

    input_scenes = discover_input_scenes()
    exclusion = build_benchmark_exclusion(input_scenes)
    print(
        json.dumps(
            {
                "input_scene_counts": exclusion["input_scene_counts"],
                "excluded_scene_counts": exclusion["excluded_scene_counts"],
                "exclusion_file": exclusion["exclusion_file"],
                "report": str(OUTPUT_ROOT / "reports/benchmark_scene_exclusion.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return exclusion


def run_smoke(num_workers: int) -> None:
    """Auto-select two unpolluted scenes to run immediately deletable dual-source smoke."""

    del num_workers  # smoke fixes the two scene orders to run, avoiding unnecessary load on existing allocation.
    input_scenes = discover_input_scenes()
    exclusion_report = build_benchmark_exclusion(input_scenes)
    excluded = {
        source: set(exclusion_report["excluded_scenes"][source])
        for source in input_scenes
    }
    selected = {
        source: next(
            (scene for scene in input_scenes[source] if scene not in excluded[source]),
            None,
        )
        for source in ("scannet", "scannetpp")
    }
    if any(scene is None for scene in selected.values()):
        raise RuntimeError(
            f"No unused quick-check scene found for both ScanNet and ScanNet++: {selected}"
        )

    smoke_root = OUTPUT_ROOT / ".smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    try:
        results = [
            build_scene(source, selected[source], str(smoke_root))
            for source in ("scannet", "scannetpp")
        ]
        rows = [row for result in results for row in result["rows"]]
        rows.sort(
            key=lambda row: (
                row["source"],
                row["scene_id"],
                len(row["image_paths"]),
                row["image_paths"],
            )
        )
        smoke_excluded = {"scannet": set(), "scannetpp": set()}
        validation = validate_manifest_rows(
            rows, smoke_root, smoke_excluded, num_workers=2
        )
        print(
            json.dumps(
                {
                    "status": "smoke_passed_and_deleted",
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", "standalone"),
                    "selected_scenes": selected,
                    "validation": validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if smoke_root.exists():
            shutil.rmtree(smoke_root)
        if smoke_root.exists():
            raise RuntimeError(
                f"Failed to clean up the quick-check output: {smoke_root}"
            )


def run_build(num_workers: int) -> None:
    """Build the entire scene without contamination; only replace the official manifest atomically after all builds succeed."""

    if num_workers <= 0:
        raise RuntimeError(f"num_workers must be positive: {num_workers}")
    input_scenes = discover_input_scenes()
    exclusion_report = build_benchmark_exclusion(input_scenes)
    excluded = {
        source: set(exclusion_report["excluded_scenes"][source])
        for source in input_scenes
    }
    items = [
        (source, scene)
        for source in ("scannet", "scannetpp")
        for scene in input_scenes[source]
        if scene not in excluded[source]
    ]
    items.sort()
    manifest_path = OUTPUT_ROOT / "manifests/windows.jsonl"
    # Keep the manifest available for rollback while rebuilding data from sensor inputs.
    previous_rows = load_jsonl(manifest_path) if manifest_path.is_file() else []
    results: dict[tuple[str, str], dict[str, Any]] = {}
    failures: dict[str, dict[str, str]] = {"scannet": {}, "scannetpp": {}}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(build_scene, source, scene, str(OUTPUT_ROOT)): (
                source,
                scene,
            )
            for source, scene in items
        }
        for completed, future in enumerate(as_completed(futures), 1):
            source, scene = futures[future]
            try:
                results[(source, scene)] = future.result()
            except Exception as exc:
                failures[source][scene] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(
                    f"[scene failed] {source}/{scene}: {failures[source][scene]}",
                    flush=True,
                )
            if completed % 20 == 0 or completed == len(items):
                print(
                    f"[build scenes] {completed}/{len(items)} success={len(results)} "
                    f"failed={sum(len(value) for value in failures.values())}",
                    flush=True,
                )

    if any(failures[source] for source in failures):
        # Publish a manifest only after all required scenes succeed.
        cleanup = cleanup_orphan_assets(OUTPUT_ROOT, previous_rows)
        raise RuntimeError(
            f"Some scenes failed; the scene list was not updated. orphan_cleanup={cleanup}, failures={failures}"
        )

    rows = [row for key in sorted(results) for row in results[key]["rows"]]
    rows.sort(
        key=lambda row: (
            row["source"],
            row["scene_id"],
            len(row["image_paths"]),
            row["image_paths"],
        )
    )
    if not rows:
        raise RuntimeError("Full build did not produce any Window")
    try:
        validation = validate_manifest_rows(rows, OUTPUT_ROOT, excluded, num_workers)
    except Exception:
        # Full asset acceptance also serves as a release prerequisite; when it fails, the new manifest must not point to a dataset that has not passed inspection.
        cleanup_orphan_assets(OUTPUT_ROOT, previous_rows)
        raise
    atomic_write_jsonl(manifest_path, rows)
    cleanup = cleanup_orphan_assets(OUTPUT_ROOT, rows)

    success_counts = {
        source: sum(1 for result in results.values() if result["source"] == source)
        for source in ("scannet", "scannetpp")
    }
    excluded_counts = {source: len(excluded[source]) for source in excluded}
    failure_counts = {source: len(failures[source]) for source in failures}
    for source in ("scannet", "scannetpp"):
        conserved = (
            success_counts[source] + excluded_counts[source] + failure_counts[source]
        )
        if conserved != len(input_scenes[source]):
            raise RuntimeError(
                f"scene number conservation failed: source={source}, input={len(input_scenes[source])}, success+excluded+failed={conserved}"
            )
        if (
            validation["successful_scene_counts_from_manifest"][source]
            != success_counts[source]
        ):
            raise RuntimeError(
                f"manifest successful scene count does not match worker result: source={source}"
            )

    report = {
        "generated_at": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "standalone"),
        "config": {
            "random_seed": RANDOM_SEED,
            "window_quotas": {str(key): value for key, value in WINDOW_QUOTAS.items()},
            "max_overlap": MAX_OVERLAP,
            "sample_attempts_per_slot": SAMPLE_ATTEMPTS_PER_SLOT,
            "max_frame_stride": MAX_FRAME_STRIDE,
            "target_image_size": list(TARGET_IMAGE_SIZE),
            "num_workers": num_workers,
        },
        "inputs": {
            "roots": {
                "scannet": str(SCANNET_ROOT),
                "scannetpp": str(SCANNETPP_ROOT),
                "benchmark": str(EXCLUDED_SCENES),
            },
            "scene_counts": {
                source: len(input_scenes[source]) for source in input_scenes
            },
        },
        "scene_status": {
            "success_counts": success_counts,
            "success_scenes": {
                source: sorted(
                    scene
                    for current_source, scene in results
                    if current_source == source
                )
                for source in ("scannet", "scannetpp")
            },
            "benchmark_excluded_counts": excluded_counts,
            "benchmark_excluded_scenes": {
                source: sorted(excluded[source]) for source in excluded
            },
            "failed_counts": failure_counts,
            "failed_scenes": failures,
            "conservation_passed": True,
        },
        "build_stats": {
            "invalid_sensor_frames": {
                f"{source}/{scene}": result["invalid_frame_reasons"]
                for (source, scene), result in sorted(results.items())
                if result["invalid_frame_reasons"]
            },
            "orphan_cleanup": cleanup,
        },
        "validation": validation,
        "disk_bytes": {},
        "manifest_path": str(manifest_path),
        "benchmark_exclusion_report": str(
            OUTPUT_ROOT / "reports/benchmark_scene_exclusion.json"
        ),
    }
    report_path = OUTPUT_ROOT / "reports/validation_report.json"
    write_validation_report(report)
    print(
        json.dumps(
            {
                "status": "build_and_validation_completed",
                "scene_counts": report["scene_status"],
                "window_counts": validation["window_counts"],
                "unique_frame_counts": validation["unique_frame_counts"],
                "disk_bytes": report["disk_bytes"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_validate(num_workers: int) -> None:
    """Run independent re-validation of assets, scene conservation, and Benchmark scanning, and refresh the official report."""

    input_scenes = discover_input_scenes()
    exclusion_report = build_benchmark_exclusion(input_scenes)
    excluded = {
        source: set(exclusion_report["excluded_scenes"][source])
        for source in input_scenes
    }
    manifest_path = OUTPUT_ROOT / "manifests/windows.jsonl"
    rows = load_jsonl(manifest_path)
    validation = validate_manifest_rows(rows, OUTPUT_ROOT, excluded, num_workers)
    report_path = OUTPUT_ROOT / "reports/validation_report.json"
    require_file(report_path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    scene_status = report.get("scene_status")
    if not isinstance(scene_status, dict):
        raise RuntimeError(f"Report is missing scene_status: {report_path}")

    # Check that recorded failures account for every missing scene.
    for source in ("scannet", "scannetpp"):
        manifest_success = {
            str(row["scene_id"]) for row in rows if row["source"] == source
        }
        expected_failed = (
            set(input_scenes[source]) - excluded[source] - manifest_success
        )
        recorded_failed = set(scene_status["failed_scenes"][source])
        if expected_failed != recorded_failed:
            raise RuntimeError(
                f"Independent validate failure in scene inconsistent with build report: source={source}, missing_reasons={sorted(expected_failed - recorded_failed)}, stale_reasons={sorted(recorded_failed - expected_failed)}"
            )
        if len(input_scenes[source]) != len(manifest_success) + len(
            excluded[source]
        ) + len(recorded_failed):
            raise RuntimeError(
                f"Independent scene validation: number conservation failed: source={source}"
            )
        if set(scene_status["benchmark_excluded_scenes"][source]) != excluded[source]:
            raise RuntimeError(
                f"Independent Benchmark exclusion set and build report inconsistent: source={source}"
            )

    report["generated_at"] = utc_now()
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "standalone")
    report["validation"] = validation
    report["validation_refresh"] = {
        "mode": "independent_validate",
        "benchmark_scene_token_rule": "ScanNet regex + ScanNet++ 9-10 digit hexadecimal token to precisely query original scene pool",
        "scene_conservation_rechecked": True,
        # Record both supported compression formats in the validation report.
        "scannetpp_depth_codec_rule": "Frame-by-frame LZ4 uint16 mm or raw zlib float32 m, output uniformly as uint16 mm",
    }
    write_validation_report(report)
    print(
        json.dumps(
            {
                "status": "validation_passed",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "standalone"),
                "manifest": str(manifest_path),
                "validation": validation,
                "disk_bytes": report["disk_bytes"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """Run scan, dual-source smoke, atomic full build, or standalone validate by subcommand."""

    args = parse_args()
    global SCANNET_ROOT, SCANNETPP_ROOT, OUTPUT_ROOT, FFMPEG, FFPROBE, EXCLUDED_SCENES
    SCANNET_ROOT = args.scannet_root.resolve()
    SCANNETPP_ROOT = args.scannetpp_root.resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    FFMPEG = args.ffmpeg.resolve()
    FFPROBE = args.ffprobe.resolve()
    EXCLUDED_SCENES = args.excluded_scenes.resolve()
    ensure_compute_node()
    if args.mode == "scan":
        run_scan()
    elif args.mode == "smoke":
        run_smoke(args.num_workers)
    elif args.mode == "build":
        run_build(args.num_workers)
    elif args.mode == "validate":
        run_validate(args.num_workers)
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
