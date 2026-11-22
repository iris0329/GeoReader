"""SPAR marker online plotting tool."""

from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image, ImageDraw


SPAR_MARKER_COLORS = {"red", "blue", "green", "yellow"}


def get_optional_spar_marker_specs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read optional marker specifications for dataset statistics."""
    specs = record.get("spar_marker_specs", [])
    if specs is None or not isinstance(specs, list):
        return []
    return specs


def validate_spar_marker_specs(
    record: Dict[str, Any], *, row_label: str, n_images: int
) -> List[Dict[str, Any]]:
    """Verify SPAR marker specifications; if the switch is on, invalid annotations will fail immediately."""
    specs = record.get("spar_marker_specs", [])
    if specs is None:
        specs = []
    if not isinstance(specs, list):
        raise RuntimeError(
            f"spar_marker_specs must be a list: {row_label}, value={specs!r}"
        )
    for spec_idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise RuntimeError(
                f"spar_marker_specs item must be a dict: {row_label}, spec_idx={spec_idx}, value={spec!r}"
            )
        kind = spec.get("kind")
        if kind not in {"bbox", "point"}:
            raise RuntimeError(
                f"spar marker kind must be bbox/point: {row_label}, spec_idx={spec_idx}, value={spec!r}"
            )
        color = spec.get("color")
        if color not in SPAR_MARKER_COLORS:
            raise RuntimeError(
                f"spar marker color unknown: {row_label}, spec_idx={spec_idx}, value={spec!r}"
            )
        image_idx = int(spec.get("image_idx", -1))
        if image_idx < 0 or image_idx >= int(n_images):
            raise RuntimeError(
                f"spar marker image_idx out of range: {row_label}, spec_idx={spec_idx}, "
                f"image_idx={image_idx}, n_images={n_images}"
            )
        expected_len = 4 if kind == "bbox" else 2
        coords = spec.get("coords")
        if not isinstance(coords, list) or len(coords) != expected_len:
            raise RuntimeError(
                f"spar marker coords length mismatch: {row_label}, spec_idx={spec_idx}, "
                f"kind={kind}, coords={coords!r}"
            )
        parsed_coords = [float(value) for value in coords]
        # SPAR's official coordinates are normalized 0-1000; the raw data contains minor edge cases of -1/-2, so slight out-of-bounds is permitted.
        if any(value < -10 or value > 1010 for value in parsed_coords):
            raise RuntimeError(
                f"spar marker coords far outside normalized 0-1000 range: "
                f"{row_label}, spec_idx={spec_idx}, coords={parsed_coords}"
            )
        if kind == "bbox" and (
            parsed_coords[2] < parsed_coords[0] or parsed_coords[3] < parsed_coords[1]
        ):
            raise RuntimeError(
                f"spar marker bbox corner order invalid: {row_label}, spec_idx={spec_idx}, coords={parsed_coords}"
            )
    return specs


def _scale_spar_bbox(
    coords: List[Any], width: int, height: int
) -> tuple[int, int, int, int]:
    """Normalize SPAR 0-1000 bbox to PIL pixel coordinates."""
    x1, y1, x2, y2 = [float(value) for value in coords]
    return (
        int(round(x1 / 1000.0 * width)),
        int(round(y1 / 1000.0 * height)),
        int(round(x2 / 1000.0 * width)),
        int(round(y2 / 1000.0 * height)),
    )


def _scale_spar_point(coords: List[Any], width: int, height: int) -> tuple[int, int]:
    """Normalize SPAR 0-1000 point to PIL pixel coordinates."""
    x, y = [float(value) for value in coords]
    return int(round(x / 1000.0 * width)), int(round(y / 1000.0 * height))


def draw_spar_marker_specs(
    images: List[Image.Image], specs: List[Dict[str, Any]], *, sample_id: str
) -> None:
    """Draw bbox/point on PIL images in SPAR official style."""
    specs_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for spec in specs:
        image_idx = int(spec["image_idx"])
        specs_by_image.setdefault(image_idx, []).append(spec)
    for image_idx, image_specs in sorted(specs_by_image.items()):
        if image_idx < 0 or image_idx >= len(images):
            raise RuntimeError(
                f"spar marker image_idx out of range after load: sample_id={sample_id}, image_idx={image_idx}"
            )
        image = images[image_idx]
        draw = ImageDraw.Draw(image)
        width, height = image.size
        stroke = 20
        for spec in image_specs:
            # Key point: During training, draw markers only on PIL images in memory; do not read parquet files, nor generate large amounts of temporary JPGs.
            if spec["kind"] == "bbox":
                x1, y1, x2, y2 = _scale_spar_bbox(spec["coords"], width, height)
                extend = int(round(stroke * 7 / 8))
                draw.rectangle(
                    (x1 - extend, y1 - extend, x2 + extend, y2 + extend),
                    outline=spec["color"],
                    width=stroke,
                )
            elif spec["kind"] == "point":
                x, y = _scale_spar_point(spec["coords"], width, height)
                draw.ellipse((x - 20, y - 20, x + 20, y + 20), fill=spec["color"])
            else:
                raise RuntimeError(
                    f"unknown SPAR marker kind: sample_id={sample_id}, spec={spec!r}"
                )
