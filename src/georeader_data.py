"""Resolve relative paths in GeoReader dataset packages."""

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np


_ROOTS = {}
_IMAGE_FIELDS = ("image_input", "images", "media_paths")
_GEOMETRY_RGB_FIELDS = (
    "geometry_image_input",
    "geometry_rgb_paths",
    "qa_rgb_paths",
    "qa_rgb_path",
    "geometry_rgb_path",
)


@lru_cache(maxsize=128)
def bundle_root(data_file):
    """Find the containing GeoReader dataset package, if present."""
    path = Path(os.path.abspath(os.path.expanduser(str(data_file))))
    for root in _ROOTS:
        if path.is_relative_to(root):
            return root
    for parent in path.parents:
        manifest = parent / "manifest.json"
        if manifest.is_file():
            values = json.loads(manifest.read_text())
            if values.get("format") in {"georeader-sixbench", "georeader-mindcube"}:
                _ROOTS[parent] = values
                return parent
    return None


def _absolute(root, value):
    """Resolve a path within the dataset root."""
    path = Path(value)
    if path.is_absolute():
        if not path.is_relative_to(root):
            raise ValueError(
                f"The package runtime path points outside the package: {path}"
            )
        return str(path)
    if ".." in path.parts:
        raise ValueError(f"Package relative path out of bounds: {path}")
    return str(root / path)


def _mapped(value, root):
    """Resolve a path or an ordered list of paths."""
    if isinstance(value, str):
        return _absolute(root, value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [_absolute(root, item) for item in value]
    raise TypeError(f"Path is not a string or a list of strings: {type(value)}")


def resolve_record_paths(record, data_file):
    """Resolve record paths in memory."""
    root = bundle_root(str(data_file))
    if root is None:
        return record
    relative = Path(os.path.abspath(str(data_file))).relative_to(root)
    phase = relative.parts[:2]
    if (
        phase not in {("train", "stage1"), ("train", "stage2")}
        and relative.parts[0] != "eval"
    ):
        raise ValueError(f"Unrecognized dataset manifest location: {data_file}")
    result = dict(record)
    for name in _IMAGE_FIELDS:
        if name in record:
            result[name] = _mapped(record[name], root)
    for flag, name in (("has_pose", "pose_path"), ("has_depth", "depth_path")):
        if record.get(flag) and record.get(name):
            result[name] = _absolute(root, record[name])
    # Stage 2's depth_paths may contain outdated GT audits; only Stage 1 actually consumes frame-by-frame depth.
    if phase == ("train", "stage1") and record.get("depth_paths"):
        result["depth_paths"] = _mapped(record["depth_paths"], root)
    if record.get("geometry_provenance"):
        provenance = dict(record["geometry_provenance"])
        for name in _GEOMETRY_RGB_FIELDS:
            if provenance.get(name):
                provenance[name] = _mapped(provenance[name], root)
        result["geometry_provenance"] = provenance
    return result


def resolve_records(records, data_file):
    """Resolve paths in a list of dataset records."""
    if bundle_root(str(data_file)) is None:
        return records
    return [resolve_record_paths(row, data_file) for row in records]


def resolve_npz_paths(payload, path):
    """Resolve NPZ image paths without copying numeric arrays."""
    root = bundle_root(str(path))
    if root is None:
        return payload
    result = dict(payload)
    for name in ("image_input", "geometry_image_input"):
        if name not in payload:
            continue
        stored = np.asarray(payload[name])
        if stored.ndim != 1 or stored.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"NPZ image sequence is not a one-dimensional string: {path}:{name}"
            )
        result[name] = np.asarray(
            [_absolute(root, value) for value in stored.astype(str).tolist()]
        )
    return result


def iter_jsonl(path):
    """Yield dataset records with local absolute paths."""
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"JSONL with empty lines: {path}")
            yield resolve_record_paths(json.loads(line), path)


def load_npz(path):
    """Load geometry arrays and resolve their image paths."""
    bundle_root(str(path))
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    return resolve_npz_paths(payload, path)
