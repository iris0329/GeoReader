"""Extract DA3 Nested geometry from ordered RGB image sequences without modifying the problem, answer, or training data."""

import argparse
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import image_paths, read_jsonl, write_json, write_jsonl, write_npz
from geometry import (
    canonicalize_w2c,
    scale_intrinsics_to_original,
    resize_depth_stack_to_rgb_aspect,
    assert_depth_rgb_aspect,
)


def file_identity(path):
    """Use original asset paths, sizes, and mtime to constrain caching, preventing accidental reuse after original images or weights are replaced."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def parse_args():
    """The input image list and weight directory are specified by the caller, and the original extraction parameters remain unchanged by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL. Only the image fields are read; questions and answers are ignored.",
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--image-field", default="image_input")
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Prefix to strip from image paths in the input JSONL.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local directory with the DA3 Nested weights and config.json.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--signals", choices=("camera", "both"), default="both")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip groups whose existing outputs match the current settings, weights and image order.",
    )
    # process_res controls the input resolution for DA3; the larger the value, the higher the computational cost; the original MindCube value is 504.
    parser.add_argument("--process-res", type=int, default=504)
    # The original extraction used upper_bound_resize and saddle_balanced; changing them will produce another prediction.
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=("upper_bound_resize",),
    )
    parser.add_argument(
        "--ref-view-strategy", default="saddle_balanced", choices=("saddle_balanced",)
    )
    # workers only control parallel image reading, defaulting to 4 to avoid overfilling shared storage.
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-groups",
        type=int,
        help="Process only the first N groups for a quick check; the output is marked as incomplete.",
    )
    return parser.parse_args()


def load_model(path):
    """Continue using the original Native Nested model, allowing only shared parameter aliases that are omitted in the weight file."""
    import torch
    from safetensors.torch import load_file
    from depth_anything_3.api import DepthAnything3

    path = Path(path).resolve()
    config = json.loads((path / "config.json").read_text())
    if config["model_name"] != "da3nested-giant-large":
        raise RuntimeError("Only DA3 Nested Giant-Large is supported.")
    model = DepthAnything3(**config)
    state = load_file(str(path / "model.safetensors"))
    loaded = model.load_state_dict(state, strict=False)
    parameters = model.state_dict(keep_vars=True)
    supplied = {parameters[key].data_ptr() for key in state if key in parameters}
    if loaded.unexpected_keys or any(
        parameters[key].data_ptr() not in supplied for key in loaded.missing_keys
    ):
        raise RuntimeError(f"DA3 weights cannot be strictly restored: {loaded}")
    del state, parameters
    return model.eval().to("cuda")


def check(folder, images, signals):
    """Accept image sequence, dimensions, units, and intrinsic matrix; cannot solely rely on directory existence to determine reusability."""
    if len(images) == 1:
        if signals != "both" or (folder / "camera.npz").exists():
            raise RuntimeError(
                "A single image has no camera motion; only depth can be extracted."
            )
        with np.load(folder / "depth.npz", allow_pickle=False) as values:
            sizes = values["image_sizes"]
            depth = values["depth_maps"]
            if (
                values["image_input"].tolist() != images
                or str(values["unit"]) != "meter"
            ):
                raise RuntimeError("Depth identity or unit mismatch in single image")
            if not np.isfinite(depth).all() or np.any(depth <= 0):
                raise RuntimeError("Single image depth contains invalid values")
            assert_depth_rgb_aspect(depth, sizes, context=str(folder))
        with Image.open(images[0]) as image:
            if list(image.size) != sizes[0].tolist():
                raise RuntimeError(
                    "Depth and RGB dimensions do not match for a single image"
                )
        return sizes.tolist()
    with np.load(folder / "camera.npz", allow_pickle=False) as values:
        if values["image_input"].tolist() != images or str(values["unit"]) != "meter":
            raise RuntimeError("camera image order or unit mismatch")
        e = values["extrinsics"]
        k = values["intrinsics"]
        sizes = values["image_sizes"]
        if (
            e.shape != (len(images), 3, 4)
            or k.shape != (len(images), 3, 3)
            or sizes.shape != (len(images), 2)
        ):
            raise RuntimeError("camera shape mismatch")
        if not np.isfinite(e).all() or not np.isfinite(k).all():
            raise RuntimeError("camera numerical value is not finite")
        if not np.allclose(e[0], np.eye(4)[:3], atol=1e-5):
            raise RuntimeError("First image not zeroed")
        rotation = e[:, :, :3]
        if not np.allclose(
            rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=2e-3
        ) or not np.allclose(np.linalg.det(rotation), 1, atol=2e-3):
            raise RuntimeError("Invalid rotation matrix prediction")
        if np.any(k[:, [0, 1], [0, 1]] <= 0):
            raise RuntimeError("Invalid focal length")
    for path, size in zip(images, sizes):
        with Image.open(path) as image:
            if list(image.size) != size.tolist():
                raise RuntimeError("RGB differs from geometric dimensions.")
    if signals == "both":
        with np.load(folder / "depth.npz", allow_pickle=False) as values:
            depth = values["depth_maps"]
            if (
                values["image_input"].tolist() != images
                or str(values["unit"]) != "meter"
            ):
                raise RuntimeError("Depth map order or unit mismatch")
            if (
                not np.array_equal(values["image_sizes"], sizes)
                or not np.isfinite(depth).all()
                or np.any(depth <= 0)
            ):
                raise RuntimeError("Invalid depth value or dimension")
            assert_depth_rgb_aspect(depth, sizes, context=str(folder))
    return sizes.tolist()


def main():
    """Duplicate the original image list by removing duplicates in the order of first appearance; write a completion record after each group is fully processed, and finally publish the index."""
    args = parse_args()
    if (
        args.workers <= 0
        or args.process_res <= 0
        or (args.max_groups is not None and args.max_groups <= 0)
    ):
        raise ValueError("workers/process-res/max-groups must be a positive number")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("DA3 extraction must use GPU")
    os.environ["WANDB_MODE"] = "offline"
    groups = {}
    for row in read_jsonl(args.input):
        images = tuple(
            image_paths(row, args.image_field, args.image_root, args.image_prefix)
        )
        groups.setdefault(images, len(groups))
    if not groups:
        raise RuntimeError("Empty input provided")
    if args.signals == "camera" and any(len(images) < 2 for images in groups):
        raise RuntimeError(
            "The camera-only entry requires multi-images; for single-image depth, use --signals both"
        )
    output = Path(args.output_dir).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(
        model="DA3NESTED-GIANT-LARGE-1.1",
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        ref_view_strategy=args.ref_view_strategy,
        signals=args.signals,
        unit="meter",
        pose_convention="w2c_frame0_identity",
    )
    contract["model_files"] = [
        file_identity(Path(args.model_dir) / name)
        for name in ("config.json", "model.safetensors")
    ]
    # Only perform cache acceptance when all groups are fully loaded, without reloading large models.
    model = None
    records = []
    selected = list(groups.items())[: args.max_groups]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for images, index in selected:
            images = list(images)
            folder = output / "geometry" / f"{index:06d}"
            if folder.exists():
                saved = json.loads((folder / "complete.json").read_text())
                if (
                    not args.resume
                    or saved["contract"] != contract
                    or saved["images"] != images
                    or saved["image_files"] != [file_identity(path) for path in images]
                ):
                    raise RuntimeError(
                        "Existing output was produced with different settings, weights, images or image order"
                    )
                sizes = check(folder, images, args.signals)
            else:
                if model is None:
                    model = load_model(args.model_dir)
                with torch.inference_mode():
                    prediction = model.inference(
                        images,
                        process_res=args.process_res,
                        process_res_method=args.process_res_method,
                        ref_view_strategy=args.ref_view_strategy,
                    )
                if not bool(prediction.is_metric):
                    raise RuntimeError(
                        "DA3 does not return metric results, cannot be saved as meter"
                    )
                depth = np.asarray(prediction.depth, dtype=np.float32)
                sizes = np.asarray(list(pool.map(read_size, images)), dtype=np.int32)
                processed = np.tile([depth.shape[2], depth.shape[1]], (len(images), 1))
                mode = "single_image" if len(images) == 1 else "ordered_multiview"
                common = dict(
                    image_input=np.asarray(images),
                    image_sizes=sizes,
                    unit=np.asarray("meter"),
                    geometry_mode=np.asarray(mode),
                )
                if len(images) > 1:
                    k = scale_intrinsics_to_original(
                        np.asarray(prediction.intrinsics, dtype=np.float32),
                        processed,
                        sizes,
                    )
                    e = canonicalize_w2c(
                        np.asarray(prediction.extrinsics, dtype=np.float32)[:, :3, :4]
                    )
                    write_npz(
                        folder / "camera.npz",
                        extrinsics=e,
                        intrinsics=k,
                        pose_convention=np.asarray("w2c_frame0_identity"),
                        **common,
                    )
                if args.signals == "both":
                    adjusted = resize_depth_stack_to_rgb_aspect(
                        depth, sizes, process_res=args.process_res
                    ).astype(np.float16)
                    write_npz(
                        folder / "depth.npz",
                        depth_maps=adjusted,
                        k_source=np.asarray("da3_nested_internal"),
                        **common,
                    )
                sizes = check(folder, images, args.signals)
                write_json(
                    folder / "complete.json",
                    dict(
                        contract=contract,
                        images=images,
                        image_files=[file_identity(path) for path in images],
                    ),
                )
            records.append(
                dict(
                    group=index,
                    image_input=images,
                    image_sizes=sizes,
                    n_images=len(images),
                    pose_path=str(folder / "camera.npz") if len(images) > 1 else None,
                    has_pose=len(images) > 1,
                    has_depth=args.signals == "both",
                    depth_path=str(folder / "depth.npz")
                    if args.signals == "both"
                    else None,
                    geometry_source="da3_nested",
                    geometry_mode="single_image"
                    if len(images) == 1
                    else "ordered_multiview",
                )
            )
            write_json(
                output / "progress.json",
                dict(
                    completed=len(records),
                    selected=len(selected),
                    total_groups=len(groups),
                ),
            )
    write_jsonl(output / "groups.jsonl", records)
    write_json(
        output / "manifest.json",
        dict(
            format="georeader-da3-groups",
            input=str(Path(args.input).resolve()),
            groups=len(records),
            total_groups=len(groups),
            is_complete=len(records) == len(groups),
            contract=contract,
            note="Ordered multiview extraction; supplied six-benchmark geometry retains its source-specific construction.",
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed", groups=len(records), index=str(output / "groups.jsonl")
            ),
            indent=2,
        )
    )


def read_size(path):
    """Read actual image dimensions with a bounded thread pool."""
    with Image.open(path) as image:
        return image.size


if __name__ == "__main__":
    main()
