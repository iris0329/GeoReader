"""Published files save only consumption fields and package image order, geometric values remain unchanged in bytes."""

import copy
from io import BytesIO
import os
from pathlib import Path
import zipfile

import numpy as np


STAGE1_FIELDS = frozenset(
    "idx group_id image_input image_sizes n_images pose_path depth_paths has_pose has_depth camera_text depth_scene_cot cot_version source scene_id window_frame_ids dataset source_n_images camera_geometry_unit".split()
)
# Title metadata is passed through the collator's key list; do not miss consuming these fields merely by searching row.get.
STAGE2_FIELDS = frozenset(
    "idx id sample_id image_input images media_paths image_sizes n_images pose_path depth_path has_pose has_depth text_input text_output answer window_frame_ids source_dataset data_source benchmark task task_family task_type spar_marker_specs spar_marker_raw_type spar_marker_coordinate_space".split()
)
EVAL_FIELDS = frozenset(
    "id images image_input image_sizes n_images question gt_answer pose_path depth_path has_pose has_depth choices cv_type raw_answer source source_dataset task task_family question_type metric_type subset format_type image_type img_type scene_component_size scene_id scene_name dataset frame_timestamps video_duration pruned subtask difficulty human_normed_duration_seconds".split()
)


def public_record(row, phase):
    """Export only the fields required for training and official scoring; keep the rewritten bounding box and coordinate fields."""
    fields = {"stage1": STAGE1_FIELDS, "stage2": STAGE2_FIELDS, "eval": EVAL_FIELDS}[
        phase
    ]
    return {key: value for key, value in row.items() if key in fields}


def rewrite_npz(source, target, replacements):
    """Only rewrite the specified string member; verify that the compressed bytes of all other ZIP members are identical."""
    source, target = Path(source), Path(target)
    temporary = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as old:
        members = old.infolist()
        names = [info.filename for info in members]
        if len(set(names)) != len(names) or set(replacements) - set(names):
            raise ValueError(
                f"NPZ member repeats or replaces non-existent field: {source}"
            )
        offsets = sorted(info.header_offset for info in members) + [old.start_dir]
        ends = dict(zip(offsets[:-1], offsets[1:]))
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as new:
            for info in members:
                if info.filename in replacements:
                    stored = np.load(
                        BytesIO(old.read(info.filename)), allow_pickle=False
                    )
                    values = replacements[info.filename]
                    if (
                        stored.ndim != 1
                        or stored.dtype.kind not in {"U", "S"}
                        or len(stored) != len(values)
                    ):
                        raise ValueError(
                            f"NPZ image sequence field type/length mismatch: {source}:{info.filename}"
                        )
                    if any(
                        Path(value).is_absolute() or ".." in Path(value).parts
                        for value in values
                    ):
                        raise ValueError(
                            f"NPZ file sequence must be relative paths within the package: {source}"
                        )
                    data = BytesIO()
                    np.save(data, np.asarray(values), allow_pickle=False)
                    replacement = zipfile.ZipInfo(info.filename)
                    replacement.compress_type = info.compress_type
                    new.writestr(replacement, data.getvalue())
                else:
                    # Copy local header, compressed data, and descriptor directly, avoiding re-encoding geometric floating-point values.
                    copied = copy.copy(info)
                    copied.header_offset = new.fp.tell()
                    old.fp.seek(info.header_offset)
                    remaining = ends[info.header_offset] - info.header_offset
                    while remaining:
                        block = old.fp.read(min(4 * 1024**2, remaining))
                        if not block:
                            raise IOError(
                                f"NPZ member truncated: {source}:{info.filename}"
                            )
                        new.fp.write(block)
                        remaining -= len(block)
                    new.filelist.append(copied)
                    new.NameToInfo[copied.filename] = copied
                    new.start_dir = new.fp.tell()
                    new._didModify = True
        with zipfile.ZipFile(temporary) as check:
            if check.namelist() != names:
                raise RuntimeError(f"NPZ member order changed: {source}")
            for info in members:
                actual = check.getinfo(info.filename)
                if info.filename in replacements:
                    values = np.load(
                        BytesIO(check.read(info.filename)), allow_pickle=False
                    )
                    if values.astype(str).tolist() != replacements[info.filename]:
                        raise RuntimeError(
                            f"NPZ image sequence replacement not effective: {source}"
                        )
                    continue
                length = ends[info.header_offset] - info.header_offset
                old.fp.seek(info.header_offset)
                check.fp.seek(actual.header_offset)
                while length:
                    size = min(4 * 1024**2, length)
                    if old.fp.read(size) != check.fp.read(size):
                        raise RuntimeError(
                            f"NPZ numeric member byte changed: {source}:{info.filename}"
                        )
                    length -= size
    # The original file may be a hard link to research data; it must replace the directory entry and cannot overwrite in place due to shared inode.
    os.chmod(temporary, 0o644)
    os.replace(temporary, target)


def clean_asset(entry, root):
    """Asset inventory only records relative paths, types, and sizes, without saving the path of the generating machine."""
    result = {key: entry[key] for key in ("path", "kind", "phase") if key in entry}
    result["bytes"] = (Path(root) / entry["path"]).stat().st_size
    return result
