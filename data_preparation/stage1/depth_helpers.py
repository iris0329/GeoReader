"""Depth projection, label filtering, and geometric description helpers."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
SKIP_LABELS = frozenset(
    {
        "wall",
        "floor",
        "ceiling",
        "object",
        "objects",
        "otherstructure",
        "otherfurniture",
        "otherprop",
        "remove",
        "unknown",
        # Common cutting residuals or meaningless short labels in ScanNet++ are not real prominent objects.
        "split",
        "it",
        "c",
        # Small electrical/structural components frequently appear in CoT but are not primary objects for depth layout training.
        "light switch",
        "power switch",
        "power socket",
        "power sockets",
        "ceiling lamp",
        "coax outlet",
        "cable duct",
        "electrical duct",
        "closet wall",
        "shower wall",
        "wall tile",
        # 2026-07-07 Manual review: weak semantic structure tags added after instmask toy version review:
        # Such objects are geometrically visible, but provide no informative content for depth layout supervision and still occupy 8 slots per frame.
        "doorframe",
        "door frame",
        "radiator",
    }
)

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
SKIP_LABEL_KEYWORDS = (
    "wall",
    "pipe",
    "duct",
    "socket",
    "switch",
    "outlet",
    "cable",
    "wire",
    "cord",
)

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MIN_DEPTH_M = 0.3

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MAX_DEPTH_M = 15.0

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MAX_OBJECTS_PER_FRAME = 8

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MAX_INSTANCE_POINTS = 8000

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MIN_PROJECTED_POINTS = 20

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MIN_VISIBLE_DEPTH_PIXELS = 12

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
MIN_VISIBLE_RATIO = 0.45

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
DEPTH_CONSISTENCY_TAU_BASE = 0.12

# Keep the original Stage1 generator's filtering/projection thresholds; changing them will alter the training labels.
DEPTH_CONSISTENCY_TAU_REL = 0.05


def should_skip_label(label: str) -> bool:
    """Determine whether a 3D annotation label is unsuitable for depth CoT."""
    label_norm = label.strip().lower()
    if not label_norm:
        return True
    if label_norm in SKIP_LABELS:
        return True
    return any(keyword in label_norm for keyword in SKIP_LABEL_KEYWORDS)


def _read_ply_vertices(ply_path: str) -> np.ndarray:
    """Read vertex XYZ from binary_little_endian PLY and parse its header dynamically.

    Dynamic parsing is designed to simultaneously support ScanNet (RGBA, 16B/vertex) and ScanNet++ (RGB, 15B/vertex).
    """
    prop_dtypes = {
        "float": "f4",
        "double": "f8",
        "uchar": "u1",
        "int": "i4",
        "uint": "u4",
        "short": "i2",
        "ushort": "u2",
        "char": "i1",
    }
    with open(ply_path, "rb") as f:
        num_vertices = 0
        vertex_props = []
        in_vertex = False
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element ") and in_vertex:
                in_vertex = False
            elif (
                line.startswith("property")
                and in_vertex
                and not line.startswith("property list")
            ):
                parts = line.split()
                vertex_props.append((parts[2], prop_dtypes[parts[1]]))
            if line == "end_header":
                break
        if num_vertices <= 0 or not vertex_props:
            raise RuntimeError(f"Failed to parse PLY header: {ply_path}")
        vdtype = np.dtype(vertex_props)
        raw = f.read(num_vertices * vdtype.itemsize)
        if len(raw) != num_vertices * vdtype.itemsize:
            raise RuntimeError(f"PLY vertex data truncation: {ply_path}")
    vertices = np.frombuffer(raw, dtype=vdtype)
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(
        np.float64
    )


def format_depth_cot_single_image(
    visible: List[Dict],
    img_num: int,
) -> str:
    """Generate a compact depth description for a single frame, reducing unnecessary template words."""
    if not visible:
        return f"image {img_num}: no prominent objects in view."

    parts = []
    for v in visible:
        # Keep label, depth, and position targets compact for long frame sequences.
        parts.append(f"{v['label']} at {v['depth_m']:.1f} meters ({v['position']})")
    return f"image {img_num}: {', '.join(parts)}."


def format_depth_cot_multiview(
    all_visible: List[List[Dict]],
) -> str:
    """Generate compact depth CoT text for multiple frames (with header, one frame per line).

    Args:
        all_visible: [frame_0_visible, frame_1_visible, ...], where each frame is the output of project_objects_to_frame

    Format (compact):
        depth layout:
        image 1: table at 1.7 meters (center), couch at 2.3 meters (right).
        image 2: table at 1.2 meters (left), chair at 3.1 meters (center).
    """
    if not all_visible:
        return ""

    sentences = ["depth layout:"]
    for i, visible in enumerate(all_visible):
        img_num = i + 1
        sentences.append(format_depth_cot_single_image(visible, img_num))

    # Each image on a separate line: preserve clear frame boundaries to avoid self-regression repetition when 32 frames are concatenated into a long segment.
    return "\n".join(sentences)
