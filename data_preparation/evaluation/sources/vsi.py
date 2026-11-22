#!/usr/bin/env python3
"""Convert VSI-Bench's test.jsonl and video assets into the current standard eval JSONL format."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


# Public entry does not depend on the original cluster path; formal invocation via CLI explicitly passes the official raw location.
DEFAULT_DATA_ROOT = Path("raw/evaluation/vsi")
DEFAULT_INPUT_FILE = DEFAULT_DATA_ROOT / "raw" / "test.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_ROOT / "processed" / "vsi_bench_test_local.jsonl"
DEFAULT_FRAME_ROOT = DEFAULT_DATA_ROOT / "processed" / "frames"
DEFAULT_STATS_FILE = DEFAULT_DATA_ROOT / "meta" / "vsi_bench_stats.json"

MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="VSI-Bench root directory with raw/, processed/ and meta/.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Input test.jsonl downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output JSONL in the GeoReader evaluation format.",
    )
    parser.add_argument(
        "--frame-root",
        type=str,
        default=str(DEFAULT_FRAME_ROOT),
        help="Directory for sampled frames.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=str(DEFAULT_STATS_FILE),
        help="Output JSON with summary statistics.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames sampled from each video.",
    )
    parser.add_argument(
        "--target-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Approximate number of pixels per sampled frame.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not extract the three zip files.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract the zip files again even if they were already extracted.",
    )
    parser.add_argument(
        "--force-rewrite-frames",
        action="store_true",
        help="Overwrite sampled frames that already exist.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Keep only the first N samples (0 = all).",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_choice_lines(choices: object) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, str):
        return [normalize_text(x) for x in choices.splitlines() if normalize_text(x)]
    if isinstance(choices, (list, tuple)):
        return [normalize_text(x) for x in choices if normalize_text(x)]
    return [normalize_text(choices)]


def build_question(question: str, choice_lines: Sequence[str]) -> str:
    if choice_lines:
        return f"{normalize_text(question)} {' '.join(choice_lines)}"
    return normalize_text(question)


def extract_zip(zip_path: Path, output_dir: Path, force_extract: bool) -> Path:
    marker = output_dir / ".extract_complete"
    if output_dir.exists() and marker.exists() and not force_extract:
        return output_dir

    if output_dir.exists() and force_extract:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return output_dir


def calculate_target_size(
    original_width: int, original_height: int, target_pixels: int
) -> Tuple[int, int]:
    aspect_ratio = original_width / max(original_height, 1)
    target_height = max(1, int((target_pixels / max(aspect_ratio, 1e-6)) ** 0.5))
    target_width = max(1, int((target_pixels * max(aspect_ratio, 1e-6)) ** 0.5))
    return target_width, target_height


def resize_image_to_target_pixels(
    image: Image.Image, target_pixels: int
) -> Image.Image:
    target_width, target_height = calculate_target_size(
        image.width, image.height, target_pixels
    )
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def pick_evenly(items: Sequence, num_items: int) -> List:
    if not items:
        return []
    if len(items) <= num_items:
        if len(items) == num_items:
            return list(items)
        indices = np.linspace(0, len(items) - 1, num_items, dtype=int)
        return [items[idx] for idx in indices]
    indices = np.linspace(0, len(items) - 1, num_items, dtype=int)
    return [items[idx] for idx in indices]


def load_frames_from_video(
    video_path: Path, num_frames: int, target_pixels: int
) -> Tuple[List[Image.Image], List[int], int]:
    try:
        from torchvision.io import read_video  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading VSI video requires torchvision.io.read_video, but it is not available in the current environment."
        ) from exc

    frames_tensor, _, info = read_video(str(video_path), pts_unit="sec")
    if frames_tensor.numel() == 0:
        raise ValueError(f"empty video decoded from {video_path}")
    total_frames = frames_tensor.shape[0]
    fps = float(info.get("video_fps", 0) or 0)
    if fps <= 0:
        fps = 30.0

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for index in frame_indices:
        image = Image.fromarray(frames_tensor[index].numpy())
        frames.append(
            resize_image_to_target_pixels(image.convert("RGB"), target_pixels)
        )
    timestamps = [int(index / fps) for index in frame_indices]
    duration = int(total_frames / fps)
    return frames, timestamps, duration


def load_frames_from_directory(
    frame_dir: Path, num_frames: int, target_pixels: int
) -> Tuple[List[Image.Image], List[int], int]:
    frame_files = sorted(
        [
            path
            for path in frame_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if not frame_files:
        raise ValueError(f"no frame images found under {frame_dir}")
    sampled_files = pick_evenly(frame_files, num_frames)
    frames = [
        resize_image_to_target_pixels(Image.open(path).convert("RGB"), target_pixels)
        for path in sampled_files
    ]
    timestamps = list(range(len(sampled_files)))
    duration = len(frame_files)
    return frames, timestamps, duration


def normalize_pickle_frames(payload: object) -> List[Image.Image]:
    if isinstance(payload, dict):
        for key in ("frames", "images", "imgs"):
            if key in payload:
                return normalize_pickle_frames(payload[key])
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"unsupported pickle payload type: {type(payload).__name__}")

    frames: List[Image.Image] = []
    for item in payload:
        if isinstance(item, Image.Image):
            frames.append(item.convert("RGB"))
        elif isinstance(item, np.ndarray):
            frames.append(Image.fromarray(item).convert("RGB"))
        elif isinstance(item, dict):
            image_bytes = item.get("bytes")
            if isinstance(image_bytes, (bytes, bytearray)):
                frames.append(Image.open(BytesIO(bytes(image_bytes))).convert("RGB"))
        elif isinstance(item, (bytes, bytearray)):
            frames.append(Image.open(BytesIO(bytes(item))).convert("RGB"))
    if not frames:
        raise ValueError("pickle payload did not contain decodable frames")
    return frames


def load_frames_from_pickle(
    pickle_path: Path, num_frames: int, target_pixels: int
) -> Tuple[List[Image.Image], List[int], int]:
    with pickle_path.open("rb") as f:
        payload = pickle.load(f)
    frames = normalize_pickle_frames(payload)
    sampled = pick_evenly(frames, num_frames)
    resized = [resize_image_to_target_pixels(frame, target_pixels) for frame in sampled]
    timestamps = list(range(len(sampled)))
    duration = len(frames)
    return resized, timestamps, duration


def find_asset(scene_name: str, search_root: Path) -> Path:
    for suffix in (".mp4", ".pkl", ".pickle"):
        direct = search_root / f"{scene_name}{suffix}"
        if direct.exists():
            return direct

    direct_dir = search_root / scene_name
    if direct_dir.exists() and direct_dir.is_dir():
        return direct_dir

    for child in search_root.iterdir() if search_root.exists() else []:
        if child.is_dir():
            for suffix in (".mp4", ".pkl", ".pickle"):
                candidate = child / f"{scene_name}{suffix}"
                if candidate.exists():
                    return candidate
            candidate_dir = child / scene_name
            if candidate_dir.exists() and candidate_dir.is_dir():
                return candidate_dir

    for pattern in (f"{scene_name}.mp4", f"{scene_name}.pkl", f"{scene_name}.pickle"):
        matches = list(search_root.rglob(pattern))
        if matches:
            return matches[0]

    dir_matches = [path for path in search_root.rglob(scene_name) if path.is_dir()]
    if dir_matches:
        return dir_matches[0]

    raise FileNotFoundError(
        f"failed to find visual asset for scene_name={scene_name} under {search_root}"
    )


def load_visual_asset(
    asset_path: Path, num_frames: int, target_pixels: int
) -> Tuple[List[Image.Image], List[int], int]:
    if asset_path.is_dir():
        return load_frames_from_directory(asset_path, num_frames, target_pixels)
    if asset_path.suffix.lower() == ".mp4":
        return load_frames_from_video(asset_path, num_frames, target_pixels)
    if asset_path.suffix.lower() in {".pkl", ".pickle"}:
        return load_frames_from_pickle(asset_path, num_frames, target_pixels)
    raise ValueError(f"unsupported visual asset type: {asset_path}")


def save_sampled_frames(
    frames: Sequence[Image.Image],
    *,
    dataset: str,
    scene_name: str,
    frame_root: Path,
    force_rewrite: bool,
) -> List[str]:
    rel_paths: List[str] = []
    for index, frame in enumerate(frames):
        rel_path = (
            Path("processed") / "frames" / dataset / scene_name / f"{index:02d}.jpg"
        )
        out_path = frame_root.parent / rel_path.relative_to("processed")
        ensure_parent(out_path)
        if not out_path.exists() or force_rewrite:
            frame.save(out_path, format="JPEG", quality=95)
        rel_paths.append(rel_path.as_posix())
    return rel_paths


def build_stats(rows: Sequence[Dict]) -> Dict:
    by_dataset = {}
    by_question_type = {}
    by_metric = {}
    num_frames = {}
    for row in rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        by_question_type[row["question_type"]] = (
            by_question_type.get(row["question_type"], 0) + 1
        )
        by_metric[row["metric_type"]] = by_metric.get(row["metric_type"], 0) + 1
        num_frames[len(row["images"])] = num_frames.get(len(row["images"]), 0) + 1

    return {
        "total_rows": len(rows),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_question_type": dict(sorted(by_question_type.items())),
        "by_metric": dict(sorted(by_metric.items())),
        "by_num_frames": dict(sorted(num_frames.items())),
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_rows(
    rows: Iterable[Dict],
    *,
    extracted_roots: Dict[str, Path],
    frame_root: Path,
    num_frames: int,
    target_pixels: int,
    force_rewrite_frames: bool,
) -> List[Dict]:
    converted: List[Dict] = []
    scene_cache: Dict[Tuple[str, str], Tuple[List[str], List[int], int]] = {}
    for row in rows:
        dataset = normalize_text(row.get("dataset"))
        scene_name = str(row.get("scene_name"))
        question_type = normalize_text(row.get("question_type"))
        if dataset not in extracted_roots:
            raise ValueError(
                f"unsupported dataset field {dataset!r} for row id={row.get('id')}"
            )

        scene_key = (dataset, scene_name)
        if scene_key in scene_cache:
            rel_frame_paths, timestamps, duration = scene_cache[scene_key]
        else:
            asset_path = find_asset(scene_name, extracted_roots[dataset])
            frames, timestamps, duration = load_visual_asset(
                asset_path, num_frames, target_pixels
            )
            rel_frame_paths = save_sampled_frames(
                frames,
                dataset=dataset,
                scene_name=scene_name,
                frame_root=frame_root,
                force_rewrite=force_rewrite_frames,
            )
            scene_cache[scene_key] = (rel_frame_paths, timestamps, duration)
        choice_lines = normalize_choice_lines(row.get("options"))

        converted.append(
            {
                "id": f"vsi_{row['id']}",
                "question": build_question(
                    str(row.get("question") or ""), choice_lines
                ),
                "images": rel_frame_paths,
                "gt_answer": normalize_text(row.get("ground_truth")),
                "dataset": dataset,
                "scene_name": scene_name,
                "question_type": question_type,
                "task": question_type,
                "metric_type": "accuracy"
                if question_type in MCA_QUESTION_TYPES
                else "mra",
                "choices": choice_lines,
                "frame_timestamps": timestamps,
                "video_duration": duration,
                "pruned": bool(row.get("pruned", False)),
                "original_id": row.get("id"),
            }
        )
    return converted


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    raw_dir = data_root / "raw"
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    frame_root = Path(args.frame_root).expanduser().resolve()
    stats_file = Path(args.stats_file).expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"input jsonl not found: {input_file}")

    extracted_roots = {
        "arkitscenes": raw_dir / "extracted" / "arkitscenes",
        "scannet": raw_dir / "extracted" / "scannet",
        "scannetpp": raw_dir / "extracted" / "scannetpp",
    }

    if not args.skip_extract:
        for dataset, zip_name in (
            ("arkitscenes", "arkitscenes.zip"),
            ("scannet", "scannet.zip"),
            ("scannetpp", "scannetpp.zip"),
        ):
            zip_path = raw_dir / zip_name
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"missing {zip_path}. Please run download_vsi_bench_assets.py first."
                )
            extract_zip(
                zip_path, extracted_roots[dataset], force_extract=args.force_extract
            )

    for dataset, root in extracted_roots.items():
        if not root.exists():
            raise FileNotFoundError(
                f"missing extracted root for {dataset}: {root}. "
                "Either re-run converter without --skip-extract or extract zip files manually."
            )

    rows = read_jsonl(input_file)
    converted = convert_rows(
        rows,
        extracted_roots=extracted_roots,
        frame_root=frame_root,
        num_frames=args.num_frames,
        target_pixels=args.target_pixels,
        force_rewrite_frames=args.force_rewrite_frames,
    )
    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    stats = build_stats(converted)
    write_jsonl(output_file, converted)
    ensure_parent(stats_file)
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("VSI-Bench local eval data ready.")
    print(f"  output_file: {output_file}")
    print(f"  frame_root:  {frame_root}")
    print(f"  stats_file:  {stats_file}")
    print(f"  total_rows:  {stats['total_rows']}")
    print(f"  by_dataset:  {stats['by_dataset']}")


if __name__ == "__main__":
    main()
