"""Shared ordered list, paths, and atomic writes for data production; independent of training machine directory."""

import json
import os
from pathlib import Path

import numpy as np


def read_jsonl(path):
    """Read strictly in original line order, report errors for empty lines or non-object lines."""
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(f"JSONL with empty lines: {path}:{index + 1}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"JSONL line is not an object: {path}:{index + 1}")
            yield row


def image_paths(row, field, root, prefix):
    """Replace an explicit image-path prefix with a configured root, preserving image order."""
    raw = row[field]
    if not isinstance(raw, list) or not raw:
        raise ValueError("The image field must be a non-empty list of paths")
    result = []
    for value in raw:
        path = Path(value)
        if prefix:
            path = path.relative_to(prefix)
        if not path.is_absolute():
            path = Path(root) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(str(path))
    return result


def write_json(path, value):
    """After writing temporary files in the same directory, atomically replace them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def write_jsonl(path, rows):
    """Maintain caller order and do not discard bad samples except for empty supervision lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_npz(path, **arrays):
    """Numerical values and image sequences are stored atomically in a single file; object arrays are prohibited."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
