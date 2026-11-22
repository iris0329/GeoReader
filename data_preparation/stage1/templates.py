"""Keep the description templates already used by Sixbang and MindCube, without invoking external language models."""

import collections
import re
import numpy as np

TRANSLATION_THRESHOLD_M = 0.05

HEADING_THRESHOLD_DEG = 3.0

FORBIDDEN_TEXT = re.compile(
    r"\b(?:up|down|vertical|pitch|roll|matrix|matrices|camera)\b",
    flags=re.IGNORECASE,
)


def build_caption(extrinsics: np.ndarray) -> tuple[str, dict]:
    """Convert canonicalized w2c to relative right/forward metric motion and heading with respect to image 1."""

    if (
        extrinsics.ndim != 3
        or extrinsics.shape[1:] != (3, 4)
        or extrinsics.shape[0] < 2
    ):
        raise RuntimeError(f"extrinsics shape illegal: {extrinsics.shape}")
    rotations = extrinsics[:, :3, :3].astype(np.float64)
    translations = extrinsics[:, :3, 3].astype(np.float64)
    centers = -np.einsum("vji,vj->vi", rotations, translations)
    relative = centers - centers[:1]

    lines = ["image 1 is the reference view."]
    metrics = {
        "right_m": [],
        "forward_m": [],
        "yaw_deg": [],
        "translation_stationary": 0,
        "heading_unchanged": 0,
        "translation_direction_counts": collections.Counter(),
        "rotation_direction_counts": collections.Counter(),
        "content_span_count": 0,
    }
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    for view_index in range(1, int(extrinsics.shape[0])):
        # NPZ has been canonicalized with image 1, so x/z correspond to image 1's right/forward axes, and y is explicitly discarded.
        right_m = float(relative[view_index, 0])
        forward_m = float(relative[view_index, 2])
        camera_forward = rotations[view_index].T @ z_axis
        camera_forward /= max(float(np.linalg.norm(camera_forward)), 1e-12)
        yaw_deg = float(np.degrees(np.arctan2(camera_forward[0], camera_forward[2])))
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

        translation_parts: list[str] = []
        if abs(right_m) >= TRANSLATION_THRESHOLD_M:
            direction = "right" if right_m > 0.0 else "left"
            translation_parts.append(f"{abs(right_m):.1f} meters {direction}")
            metrics["translation_direction_counts"][direction] += 1
            metrics["content_span_count"] += 2
        if abs(forward_m) >= TRANSLATION_THRESHOLD_M:
            direction = "forward" if forward_m > 0.0 else "backward"
            translation_parts.append(f"{abs(forward_m):.1f} meters {direction}")
            metrics["translation_direction_counts"][direction] += 1
            metrics["content_span_count"] += 2
        if translation_parts:
            translation_text = "moves " + " and ".join(translation_parts)
        else:
            translation_text = "stays nearly at the same position"
            metrics["translation_stationary"] += 1
            metrics["content_span_count"] += 1

        if abs(yaw_deg) < HEADING_THRESHOLD_DEG:
            heading_text = "keeps nearly the same heading"
            metrics["heading_unchanged"] += 1
            metrics["content_span_count"] += 1
        else:
            direction = "right" if yaw_deg > 0.0 else "left"
            heading_text = f"turns {direction} by about {abs(yaw_deg):.0f} degrees"
            metrics["rotation_direction_counts"][direction] += 1
            metrics["content_span_count"] += 2

        lines.append(
            f"compared with image 1, the observer in image {view_index + 1} "
            f"{translation_text}, and {heading_text}."
        )
        metrics["right_m"].append(right_m)
        metrics["forward_m"].append(forward_m)
        metrics["yaw_deg"].append(yaw_deg)

    caption = "\n".join(lines)
    forbidden = FORBIDDEN_TEXT.search(caption)
    if forbidden:
        raise RuntimeError(
            f"Caption appears with disabled semantics {forbidden.group(0)!r}: {caption!r}"
        )
    return caption, metrics


def number(x):
    """Keep annotation values, without introducing metric approximations or motion thresholds."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def caption(motions):
    """All subsequent views are described within the same reference frame; text refers to relative pose relationships, without speculating intermediate trajectories."""
    lines = [
        "image 1 is the reference view. distances are measured in symbolic grid units."
    ]
    for i, (right, forward, yaw) in enumerate(motions, 2):
        parts = []
        for value, pos, neg in [
            (right, "right", "left"),
            (forward, "forward", "backward"),
        ]:
            if value:
                unit = "grid unit" if abs(value) == 1 else "grid units"
                parts.append(f"{number(abs(value))} {unit} {pos if value > 0 else neg}")
        position = (
            "is " + " and ".join(parts) + " of the reference position"
            if parts
            else "is at the same position"
        )
        if yaw == 0:
            heading = "faces the same direction"
        elif abs(yaw) == 180:
            heading = "faces the opposite direction (180 degrees)"
        else:
            heading = f"faces {number(abs(yaw))} degrees to the {'right' if yaw > 0 else 'left'} of the reference heading"
        lines.append(
            f"compared with image 1, the observer in image {i} {position}, and {heading}."
        )
    return "\n".join(lines)
