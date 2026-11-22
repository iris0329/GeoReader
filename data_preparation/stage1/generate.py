"""Generate Stage 1 supervision from GT window facts or MindCube DA3 geometry."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import image_paths, read_jsonl, write_json, write_jsonl, write_npz
from da3.geometry import canonicalize_w2c
from depth_facts import load_frame_fact, caption_visible_objects
from templates import build_caption, caption


def parse_args():
    """The six-benchmark recipe reads GT windows and instance facts; MindCube reads raw QA image fields and the DA3 index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=("sixbench_qwen3_8b", "mindcube_da3_qwen25_3b"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--windows",
        help="Ordered GT window manifest from build_windows.py.",
    )
    parser.add_argument(
        "--instance-root",
        help="Root directory of GT instances from build_instances.py.",
    )
    parser.add_argument("--branch", choices=("camera", "depth", "both"), default="both")
    parser.add_argument(
        "--qa",
        help="Official MindCube 10K training QA; only the image fields are used.",
    )
    parser.add_argument("--image-root")
    parser.add_argument("--image-prefix", default="")
    parser.add_argument(
        "--geometry-index",
        help="groups.jsonl written by data_preparation/da3/extract.py.",
    )
    # workers only control parallelism for independent windows; default is 8 to avoid overloading shared disk I/O.
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Process only the first N windows for a quick check; the output is marked as incomplete.",
    )
    return parser.parse_args()


def build_gt_window(spec, args, output):
    """Generate two streams of labels directly from the GT pose/depth/instance in this window, preserving original physical line numbers."""
    index, row = spec
    required = {"source", "scene_id", "image_paths", "pose_paths", "depth_paths"}
    if set(row) != required:
        raise RuntimeError(
            "Input must be the GT window manifest, not an existing Stage 1 training file."
        )
    count = len(row["image_paths"])
    if count not in (1, 2, 4, 8, 16, 32) or any(
        len(row[key]) != count for key in ("pose_paths", "depth_paths")
    ):
        raise RuntimeError("GT window frame count or image order mismatch")
    images = [str(Path(path).resolve()) for path in row["image_paths"]]
    poses = []
    intrinsics = []
    sizes = []
    frame_ids = []
    for image_path, pose_path, depth_path in zip(
        images, row["pose_paths"], row["depth_paths"]
    ):
        ids = [int(Path(path).stem) for path in (image_path, pose_path, depth_path)]
        if len(set(ids)) != 1:
            raise RuntimeError("Inconsistent physical frame numbers for RGB/Pose/Depth")
        with np.load(pose_path, allow_pickle=False) as values:
            c2w = np.asarray(values["c2w"], dtype=np.float64)
            k = np.asarray(values["intrinsic"], dtype=np.float32)
            size = np.asarray(values["image_size"], dtype=np.int32)
        if (
            c2w.shape != (4, 4)
            or k.shape != (3, 3)
            or not np.isfinite(c2w).all()
            or not np.isfinite(k).all()
        ):
            raise RuntimeError("GT pose or K is invalid")
        with Image.open(image_path) as image:
            if not np.array_equal(image.size, size):
                raise RuntimeError("GT geometry and RGB size are inconsistent")
        poses.append(c2w)
        intrinsics.append(k)
        sizes.append(size)
        frame_ids.append(ids[0])
    base = dict(
        idx=index,
        source=row["source"],
        scene_id=row["scene_id"],
        n_images=count,
        image_input=images,
        image_sizes=np.asarray(sizes).tolist(),
        window_frame_ids=frame_ids,
    )
    camera = None
    depth = None
    if args.branch in ("camera", "both") and count > 1:
        # Consistent with the original pipeline: first invert each frame's c2w to float32 w2c, then normalize using the first frame's camera.
        raw = np.stack([np.linalg.inv(pose)[:3, :4] for pose in poses]).astype(
            np.float32
        )
        e = canonicalize_w2c(raw)
        pose_path = (
            output / "camera_windows" / row["source"] / row["scene_id"] / f"{index}.npz"
        )
        write_npz(
            pose_path,
            extrinsics=e,
            intrinsics=np.stack(intrinsics),
            image_sizes=np.asarray(sizes, dtype=np.int32),
            image_input=np.asarray(images),
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
        )
        text, _ = build_caption(e)
        camera = dict(
            base,
            pose_path=str(pose_path),
            has_pose=1,
            camera_text=text,
            cot_version="camera_observer_planar_metric_motion_image1_content_supervision",
        )
    if args.branch in ("depth", "both"):
        folder = Path(args.instance_root) / row["source"] / row["scene_id"]
        metadata = json.loads((folder / "scene_instances.json").read_text())
        if (metadata["source"], metadata["scene_id"]) != (
            row["source"],
            row["scene_id"],
        ):
            raise RuntimeError("Instance annotation scene mismatch")
        labels = {item["instance_id"]: item["label"] for item in metadata["instances"]}
        frames = []
        for i, frame_id in enumerate(frame_ids):
            frame = dict(
                source=row["source"],
                scene_id=row["scene_id"],
                frame_id=frame_id,
                image_path=images[i],
                pose_path=row["pose_paths"][i],
                depth_path=row["depth_paths"][i],
            )
            facts = load_frame_fact(frame, str(folder / f"{frame_id}.npz"), labels)
            frames.append(caption_visible_objects(facts))
        # Original rule: Select top 8 by area, then sort from near to far, retain the first 5; only remove if completely empty.
        if any(frames):
            lines = ["depth layout:"]
            for i, objects in enumerate(frames, 1):
                body = ", ".join(
                    f"{item['label']} ({item['position']}) at {item['depth_m']:.1f} meters"
                    for item in objects[:5]
                )
                lines.append(
                    f"image {i}: "
                    + (body + "." if objects else "no prominent objects in view.")
                )
            depth = dict(
                base,
                depth_paths=[str(Path(path).resolve()) for path in row["depth_paths"]],
                has_depth=1,
                depth_scene_cot="\n".join(lines),
                cot_version="depth_layout_numeric_readout_top5_position_first",
            )
    return camera, depth


def generate_gt(args, output):
    """Build GT descriptions from window manifests and geometric facts."""
    if not args.windows or (
        args.branch in ("depth", "both") and not args.instance_root
    ):
        raise RuntimeError(
            "The six-benchmark recipe requires --windows; Depth also requires --instance-root"
        )
    windows = list(enumerate(read_jsonl(args.windows)))
    selected = windows if args.max_windows is None else windows[: args.max_windows]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda spec: build_gt_window(spec, args, output), selected)
        )
    camera = [row for row, _ in results if row is not None]
    depth = [row for _, row in results if row is not None]
    if args.branch in ("camera", "both"):
        write_jsonl(output / "camera.jsonl", camera)
    if args.branch in ("depth", "both"):
        write_jsonl(output / "depth.jsonl", depth)
    return dict(
        windows=len(selected),
        total_windows=len(windows),
        camera_rows=len(camera),
        depth_rows=len(depth),
        is_complete=len(selected) == len(windows),
        inputs=[str(Path(args.windows).resolve()), args.instance_root],
        derived_caption_inputs=0,
    )


def generate_mindcube(args, output):
    """Same as raw QA ordered image pair, DA3 value generation template description, does not read QA answers."""
    if not args.qa or not args.image_root or not args.geometry_index:
        raise RuntimeError("MindCube requires --qa, --image-root, and --geometry-index")
    geometries = {
        tuple(row["image_input"]): row for row in read_jsonl(args.geometry_index)
    }
    groups = {}
    for index, row in enumerate(read_jsonl(args.qa)):
        images = tuple(
            image_paths(row, "image_input", args.image_root, args.image_prefix)
        )
        groups.setdefault(images, []).append(row["idx"])
    pairs = []
    selected = (
        list(groups.items())
        if args.max_windows is None
        else list(groups.items())[: args.max_windows]
    )
    for group_id, (images, qa_ids) in enumerate(selected):
        geometry = geometries[images]
        with np.load(geometry["pose_path"], allow_pickle=False) as values:
            payload = {key: values[key] for key in values.files}
        if (
            payload["image_input"].tolist() != list(images)
            or str(payload["unit"]) != "meter"
        ):
            raise RuntimeError("DA3 image order or unit mismatch")
        e = payload["extrinsics"]
        sizes = payload["image_sizes"]
        rotations = e[:, :, :3].astype(float)
        if not np.allclose(e[0], np.eye(4)[:3], atol=1e-5):
            raise RuntimeError("The DA3 front camera is not the reference origin.")
        centers = -np.einsum("vji,vj->vi", rotations, e[:, :, 3])
        motions = []
        for view in range(1, len(images)):
            forward = rotations[view].T @ np.array([0.0, 0.0, 1.0])
            if math.hypot(forward[0], forward[2]) <= 1e-6:
                raise RuntimeError(
                    "DA3 orientation cannot be projected onto the horizontal plane"
                )
            yaw = (math.degrees(math.atan2(forward[0], forward[2])) + 180) % 360 - 180
            angle = round(yaw)
            motions.append(
                (
                    round(float(centers[view, 0]), 1),
                    round(float(centers[view, 2]), 1),
                    180 if abs(angle) == 180 else angle,
                )
            )
        text = (
            caption(motions)
            .replace(
                "distances are measured in symbolic grid units.",
                "distances are measured in predicted meters.",
            )
            .replace("grid units", "meters")
            .replace("grid unit", "meter")
        )
        lines = text.splitlines()
        for view in range(1, len(images)):
            chosen = [0, view]
            pose_path = (
                output
                / "da3_camera_pair"
                / f"{group_id:06d}"
                / f"0_{view}"
                / "camera.npz"
            )
            # Only slice 0/i, keep the common origin of group prediction, without re-inference or re-normalization.
            sliced = {
                key: (
                    value[chosen]
                    if key
                    in (
                        "extrinsics",
                        "intrinsics",
                        "image_sizes",
                        "image_input",
                        "frame_ids",
                    )
                    else value
                )
                for key, value in payload.items()
            }
            write_npz(pose_path, **sliced)
            pairs.append(
                dict(
                    idx=len(pairs),
                    source="mindcube_train_da3_camera",
                    dataset="mindcube",
                    subtask=Path(images[0])
                    .relative_to(Path(args.image_root).resolve())
                    .parts[0],
                    source_group_idx=group_id,
                    source_n_images=len(images),
                    source_view_indices=chosen,
                    n_images=2,
                    image_input=[images[i] for i in chosen],
                    image_sizes=sizes[chosen].tolist(),
                    pose_path=str(pose_path),
                    has_pose=True,
                    has_depth=False,
                    camera_text=lines[0]
                    + "\n"
                    + lines[view].replace(
                        f"the observer in image {view + 1} ", "the observer in image 2 "
                    ),
                    cot_version="mindcube_da3_camera_image1_meter",
                    geometry_source="da3_nested",
                    camera_geometry_unit="meter",
                    qa_count=len(qa_ids),
                    qa_indices=qa_ids,
                    supervision_recipe="reference_pair_isolated",
                )
            )
    write_jsonl(output / "camera.jsonl", pairs)
    return dict(
        groups=len(selected),
        total_groups=len(groups),
        camera_rows=len(pairs),
        is_complete=len(selected) == len(groups),
        inputs=[str(Path(args.qa).resolve()), str(Path(args.geometry_index).resolve())],
        derived_caption_inputs=0,
    )


def main():
    """The current output directory must be empty; training data is always read-only."""
    args = parse_args()
    if args.workers <= 0 or (args.max_windows is not None and args.max_windows <= 0):
        raise ValueError("workers/max-windows must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = (generate_gt if args.recipe == "sixbench_qwen3_8b" else generate_mindcube)(
        args, output
    )
    result.update(status="passed", recipe=args.recipe, output_dir=str(output))
    write_json(output / "generation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
