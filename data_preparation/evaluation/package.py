"""Concatenate the official conversion result with the ordered DA3 geometry into a movable evaluation package, without modifying the problem statement or standard answers."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import read_jsonl, write_json, write_jsonl, write_npz
from release_format import public_record

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2"))
from generate import load_geometry, validate_geometry


EXPECTED_ROWS = dict(
    cvbench=2638,
    mmsi=1000,
    spar=7211,
    spbench=1328,
    viewspatial=5712,
    vsi=5130,
    mindcube=1050,
)


def main():
    """Each benchmark outputs to its own independent directory; a shared manifest records the locations of each benchmark without overwriting existing evaluation sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=(
            "cvbench",
            "mmsi",
            "spar",
            "spbench",
            "viewspatial",
            "vsi",
            "mindcube",
        ),
        required=True,
    )
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--geometry-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Use hardlink to save original image bytes within the same filesystem; across filesystems, explicitly choose copy, without silent switching.
    parser.add_argument("--copy-mode", choices=("link", "copy"), default="link")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow packaging a subset for a quick check; the output is marked as incomplete.",
    )
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    name = args.benchmark
    relative = Path(
        "eval/mindcube_tiny/da3.jsonl"
        if name == "mindcube"
        else f"eval/{name}/qa.jsonl"
    )
    destination = root / relative
    if destination.exists():
        raise FileExistsError(destination)
    manifest_file = root / "manifest.json"
    expected_format = (
        "georeader-mindcube" if name == "mindcube" else "georeader-sixbench"
    )
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["format"] != expected_format:
            raise RuntimeError("MindCube and Liubang must use separate data packages.")
    else:
        manifest = dict(format=expected_format, assets="assets.jsonl", datasets=[])
    geometries = load_geometry(
        args.geometry_index.resolve(), args.geometry_index.resolve().parent
    )
    original, identities = [], set()
    for row in read_jsonl(args.qa):
        if row["id"] in identities:
            raise RuntimeError(f"Official QA identity duplication: {row['id']}")
        identities.add(row["id"])
        paths = tuple(str(Path(path).resolve()) for path in row["images"])
        if paths not in geometries:
            raise RuntimeError(
                f"Cannot find matching image order geometry in official QA: {row['id']}"
            )
        if not isinstance(row["question"], str) or not str(row["gt_answer"]).strip():
            raise RuntimeError("Official question or answer fields are incomplete")
        original.append((row, paths))
    if not original:
        raise RuntimeError("Official QA is empty")
    complete = len(original) == EXPECTED_ROWS[name]
    if not complete and not args.allow_subset:
        raise RuntimeError(
            f"{name}Needed{EXPECTED_ROWS[name]}The small sample must be explicitly passed --allow-subset"
        )
    assets, pictures, predictions = [], {}, {}
    asset_root = (
        Path("eval") / ("mindcube_tiny" if name == "mindcube" else name) / "assets"
    )

    def register_image(path):
        """The original image is saved only once, with copies or hard links to the original bytes, without re-scaling or re-encoding."""
        if path not in pictures:
            source = Path(path)
            relative_image = (
                asset_root / "images" / f"{len(pictures):08d}{source.suffix.lower()}"
            )
            target = root / relative_image
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.copy_mode == "link":
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
            pictures[path] = str(relative_image)
            assets.append(
                dict(kind="rgb", path=str(relative_image), bytes=target.stat().st_size)
            )
        return pictures[path]

    rows = []
    for row, paths in original:
        mapped = [register_image(path) for path in paths]
        if paths not in predictions:
            source = validate_geometry(geometries[paths])
            group = len(predictions)
            sizes = []
            for path in paths:
                with Image.open(path) as image:
                    sizes.append(list(image.size))
            result = dict(
                has_pose=bool(source["has_pose"]),
                has_depth=bool(source["has_depth"]),
                pose_path=None,
                depth_path=None,
                image_sizes=sizes,
                n_images=len(paths),
            )
            if name == "mindcube" and (not result["has_pose"] or result["has_depth"]):
                raise RuntimeError(
                    "MindCube DA3 recipe only uses predicted camera geometry"
                )
            for flag, field, filename in (
                ("has_pose", "pose_path", "camera.npz"),
                ("has_depth", "depth_path", "depth.npz"),
            ):
                if not result[flag]:
                    if source[field] is not None:
                        raise RuntimeError(f"{flag}Closed but still provides a path")
                    continue
                path = Path(source[field])
                if not path.is_absolute():
                    path = args.geometry_index.resolve().parent / path
                with np.load(path, allow_pickle=False) as archive:
                    payload = {key: archive[key] for key in archive.files}
                if payload["image_input"].astype(str).tolist() != list(paths):
                    raise RuntimeError(
                        f"Geometry NPZ does not match the official image order: {path}"
                    )
                if not np.array_equal(payload["image_sizes"], sizes):
                    raise RuntimeError(
                        f"Geometry NPZ does not match official image dimensions: {path}"
                    )
                # Path members store relative positions, numerical arrays retain their original values; the reader can resume operation from any download root directory.
                payload["image_input"] = np.asarray(mapped)
                target = asset_root / "geometry" / f"{group:06d}" / filename
                write_npz(root / target, **payload)
                assets.append(
                    dict(
                        kind="camera" if field == "pose_path" else "depth",
                        path=str(target),
                        bytes=(root / target).stat().st_size,
                    )
                )
                result[field] = str(target)
            predictions[paths] = result
        record = dict(row, **predictions[paths])
        record["images"] = mapped
        if name == "mindcube":
            record.update(
                image_input=mapped,
                geometry_source="da3_nested",
                geometry_mode="ordered_multiview",
            )
        rows.append(public_record(record, "eval"))
    write_jsonl(destination, rows)
    index_file = root / manifest["assets"]
    existing = list(read_jsonl(index_file)) if index_file.exists() else []
    if {row["path"] for row in existing}.intersection(row["path"] for row in assets):
        raise RuntimeError("Conflict with existing asset when adding benchmark asset")
    write_jsonl(index_file, [*existing, *assets])
    manifest["datasets"].append(
        dict(phase="eval", benchmark=name, path=str(relative), rows=len(rows))
    )
    manifest["datasets"][-1]["is_complete"] = complete
    write_json(manifest_file, manifest)
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "src/georeader_data.py",
        root / "reader.py",
    )
    write_json(
        root
        / "eval"
        / ("mindcube_tiny" if name == "mindcube" else name)
        / "generation.json",
        dict(
            benchmark=name,
            rows=len(rows),
            ordered_groups=len(predictions),
            official_qa_unchanged=True,
            is_complete=complete,
            manual_restore_required=False,
        ),
    )
    print(json.dumps(dict(benchmark=name, rows=len(rows), output=str(root))))


if __name__ == "__main__":
    main()
