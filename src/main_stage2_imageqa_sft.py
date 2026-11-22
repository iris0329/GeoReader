"""Qwen3-VL-8B Vanilla SFT data loading and checkpoint saving assistance."""

from __future__ import annotations

import json

import os

import re

from collections import Counter

from typing import List, Optional, Union

from PIL import Image

from torch.utils.data import Dataset

from main_camera_interpreter import (
    MIXED_IMAGE_SIZE,
    _assert_pil_image_size,
    _get_record_image_sizes,
    _parse_expected_image_size,
    assert_cpu_affinity_for_training,
    seed_everything,
)

from coordinate_rewriter import (
    copy_coordinate_rewrite_metadata,
    rewrite_sample_coordinates_in_texts,
)

from spar_marker_utils import (
    draw_spar_marker_specs,
    get_optional_spar_marker_specs,
    validate_spar_marker_specs,
)

from stage2_checkpoint_retention import (
    DropAllResumeStateCallback,
    save_final_checkpoint_with_latest_resume_state,
)


def _load_json_or_jsonl(path: str) -> List[dict]:
    """Read JSON array or JSONL; report error directly if format is incorrect."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]
    if not isinstance(records, list):
        raise RuntimeError(f"data file must be a JSON array or JSONL: {path}")
    return records


def _normalize_qa_data_paths(qa_data_path: Union[str, List[str]]) -> List[str]:
    """Convert single-path or multi-path parameters into an absolute path list."""
    raw_paths = [qa_data_path] if isinstance(qa_data_path, str) else list(qa_data_path)
    if not raw_paths:
        raise RuntimeError("--qa_data_path must specify at least one JSON/JSONL file.")
    paths = []
    for raw_path in raw_paths:
        path = os.path.abspath(os.path.expanduser(str(raw_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"QA data file not found: {path}")
        paths.append(path)
    return paths


def _require_text(
    record: dict, field: str, *, row_idx: int, row_label: Optional[str] = None
) -> str:
    """Extract non-empty text fields from the sample; fail directly if a field is missing or empty."""
    label = row_label if row_label is not None else f"row={row_idx}"
    if field not in record:
        raise RuntimeError(
            f"QA row missing {field!r}: {label}, keys={sorted(record.keys())}"
        )
    text = str(record[field]).strip()
    if not text:
        raise RuntimeError(
            f"QA row has empty {field!r}: {label}, idx={record.get('idx')}"
        )
    return text


def _has_nonempty_think(text: str) -> bool:
    """Determine if the assistant target contains a non-empty Think, and do not count empty labels as COT."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    return match is not None and bool(match.group(1).strip())


class ImageQAJsonlDataset(Dataset):
    """Only read image_input / text_input / text_output, do not read Camera/Depth."""

    def __init__(
        self,
        qa_data_path: Union[str, List[str]],
        image_folder: str,
        max_samples: Optional[int] = None,
        expected_image_size: Optional[tuple] = MIXED_IMAGE_SIZE,
        load_images: bool = True,
        draw_spar_marker_specs: bool = False,
    ):
        self.qa_data_paths = _normalize_qa_data_paths(qa_data_path)
        self.qa_data_path = ",".join(self.qa_data_paths)
        self.image_folder = os.path.abspath(os.path.expanduser(image_folder))
        self.expected_image_size = expected_image_size
        self.load_images = bool(load_images)
        self.draw_spar_marker_specs = bool(draw_spar_marker_specs)

        kept = []
        source_stats = Counter()
        image_count_stats = Counter()
        spar_marker_kind_stats = Counter()
        spar_marker_task_stats = Counter()
        spar_marker_rows = 0
        spar_marker_specs = 0
        coldstart_cot_rows = 0
        total_rows = 0
        rows_per_file = {}
        # Multiple JSONL files are concatenated directly in the input order without first synthesizing a temporary file; this reduces one intermediate step and retains the original files for debugging.
        for data_path in self.qa_data_paths:
            records = _load_json_or_jsonl(data_path)
            rows_per_file[data_path] = len(records)
            total_rows += len(records)
            for source_row_idx, record in enumerate(records):
                row_label = f"path={data_path}, row={source_row_idx}"
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"QA row must be a dict: {row_label}, type={type(record)}"
                    )
                for field in (
                    "idx",
                    "text_input",
                    "image_input",
                    "text_output",
                    "n_images",
                ):
                    if field not in record:
                        raise RuntimeError(
                            f"QA row missing {field!r}: {row_label}, keys={sorted(record.keys())}"
                        )
                _require_text(
                    record, "text_input", row_idx=source_row_idx, row_label=row_label
                )
                _require_text(
                    record, "text_output", row_idx=source_row_idx, row_label=row_label
                )
                image_input = record["image_input"]
                if not isinstance(image_input, list) or not image_input:
                    raise RuntimeError(
                        f"image_input must be a non-empty list: {row_label}, idx={record.get('idx')}"
                    )
                n_images = int(record["n_images"])
                if n_images != len(image_input):
                    raise RuntimeError(
                        f"n_images and image_input length mismatch: {row_label}, idx={record.get('idx')}, "
                        f"n_images={n_images}, image_input={len(image_input)}"
                    )
                if self.draw_spar_marker_specs:
                    # Default enabled: Samples with SPAR markers must be strictly validated and drawn online; non-SPAR samples have empty fields and are naturally no-op.
                    marker_specs = validate_spar_marker_specs(
                        record, row_label=row_label, n_images=n_images
                    )
                else:
                    marker_specs = get_optional_spar_marker_specs(record)
                if marker_specs:
                    spar_marker_rows += 1
                    spar_marker_specs += len(marker_specs)
                    if self.draw_spar_marker_specs:
                        spar_marker_kind_stats.update(
                            str(spec["kind"]) for spec in marker_specs
                        )
                    spar_marker_task_stats[
                        str(record.get("task_family", "unknown"))
                    ] += 1
                record_image_sizes = _get_record_image_sizes(
                    record, row_idx=source_row_idx, required=False
                )
                if record_image_sizes is not None and len(record_image_sizes) != len(
                    image_input
                ):
                    raise RuntimeError(
                        f"image_sizes and image_input length mismatch: {row_label}, idx={record.get('idx')}, "
                        f"image_sizes={len(record_image_sizes)}, image_input={len(image_input)}"
                    )
                source_stats[str(record.get("source_dataset", "unknown"))] += 1
                image_count_stats[n_images] += 1
                kept_record = dict(record)
                kept_record["_qa_data_path"] = data_path
                kept_record["_qa_row_idx"] = source_row_idx
                kept.append(kept_record)
                if _has_nonempty_think(str(record["text_output"])):
                    coldstart_cot_rows += 1
                if max_samples is not None and len(kept) >= int(max_samples):
                    break
            if max_samples is not None and len(kept) >= int(max_samples):
                break

        if not kept:
            raise RuntimeError(f"No image-QA samples loaded: {self.qa_data_path}")
        self.records = kept
        self.rows_per_file = rows_per_file
        self.total_rows = total_rows
        self.coldstart_cot_rows = coldstart_cot_rows
        print("[dataset] imageqa_sft summary")
        print(f"[dataset] qa_files={self.qa_data_paths}")
        print(f"[dataset] qa_rows_per_file={rows_per_file}")
        print(f"[dataset] qa_rows_total={total_rows}")
        print(f"[dataset] qa_rows_used={len(self.records)}")
        print(f"[dataset] coldstart_cot_rows={self.coldstart_cot_rows}")
        print(
            f"[dataset] max_samples={int(max_samples) if max_samples is not None else None}"
        )
        print(f"[dataset] source_dataset={dict(sorted(source_stats.items()))}")
        print(f"[dataset] n_images={dict(sorted(image_count_stats.items()))}")
        # The SPAR marker is only provided with the with-marker JSONL; when this field is absent in other training data, it is naturally 0.
        print(f"[dataset] draw_spar_marker_specs={int(self.draw_spar_marker_specs)}")
        print(f"[dataset] spar_marker_rows={spar_marker_rows}")
        print(f"[dataset] spar_marker_specs={spar_marker_specs}")
        print(
            f"[dataset] spar_marker_kind={dict(sorted(spar_marker_kind_stats.items()))}"
        )
        print(
            f"[dataset] spar_marker_task_rows={dict(sorted(spar_marker_task_stats.items()))}"
        )

    def __len__(self):
        return len(self.records)

    def _resolve_image_path(self, raw_path: str) -> str:
        """Convert relative image paths in JSONL to absolute paths."""
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(self.image_folder, raw_path)

    def __getitem__(self, index):
        record = dict(self.records[int(index)])
        image_paths = [
            self._resolve_image_path(str(path)) for path in record["image_input"]
        ]
        record_image_sizes = _get_record_image_sizes(record, required=False)
        images = []
        if self.load_images:
            for image_idx, path in enumerate(image_paths):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Input image not found: {path}")
                image = Image.open(path).convert("RGB")
                if record_image_sizes is not None:
                    expected_record_size = tuple(record_image_sizes[image_idx])
                    if tuple(image.size) != expected_record_size:
                        raise RuntimeError(
                            f"image_sizes metadata mismatch: path={path}, "
                            f"actual PIL(W,H)={tuple(image.size)}, metadata={expected_record_size}"
                        )
                # Here, hard validation is retained: if a JSONL record does not match the actual image dimensions, fail early.
                _assert_pil_image_size(
                    image, path, self.expected_image_size, "Stage2 image-only QA SFT"
                )
                images.append(image)
            marker_specs = record.get("spar_marker_specs", [])
            if self.draw_spar_marker_specs and marker_specs:
                # Key point: When the switch is turned on, the image_input in the with-marker JSONL file remains the original image.
                # Draw bounding boxes online based on annotation to ensure the training actually sees the marker.
                draw_spar_marker_specs(
                    images,
                    marker_specs,
                    sample_id=str(record.get("sample_id", record.get("idx"))),
                )

        result = {
            "idx": int(record["idx"]),
            "sample_id": str(record.get("sample_id", record.get("idx"))),
            "source_dataset": str(record.get("source_dataset", "unknown")),
            "n_images": int(record["n_images"]),
            "text_input": str(record["text_input"]).strip(),
            "text_output": str(record["text_output"]).strip(),
        }
        # Pass task metadata through the collator so coordinate rewriting selects the correct image.
        copy_coordinate_rewrite_metadata(record, result)
        if self.load_images:
            result["images"] = images
        if record_image_sizes is not None:
            result["image_sizes"] = record_image_sizes
        return result
