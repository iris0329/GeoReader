"""Convert official benchmark raw QA into evaluation JSONL; do not generate new questions or standard answers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BENCHMARKS = ("cvbench", "mmsi", "spar", "spbench", "viewspatial", "vsi", "mindcube")


def normalize_images(rows, benchmark, source_root, output):
    """Each converter uses different image roots; the unified entry point must be explicitly normalized before being passed to DA3."""
    for row in rows:
        paths = []
        for original in row["images"]:
            path = Path(original)
            if not path.is_absolute():
                if benchmark in ("cvbench", "mmsi", "spar"):
                    path = output / "images" / path
                elif benchmark == "vsi":
                    # Resolve frame paths against the configured frame root.
                    path = output / "frames" / path.relative_to("processed/frames")
                else:
                    path = source_root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(str(path))
        row["images"] = paths
        row["n_images"] = len(paths)
    return rows


def convert_mindcube(args, output):
    """MindCube official problem statement and solution remain unchanged, only explicitly specifying the root directory of the original image."""
    from PIL import Image

    if args.input_file is None:
        raise ValueError("MindCube must provide official --input-file")
    rows = []
    with args.input_file.open() as handle:
        for line in handle:
            row = json.loads(line)
            if args.max_samples is not None and len(rows) == args.max_samples:
                break
            if any(
                key in row for key in ("pose_path", "depth_path", "geometry_provenance")
            ):
                raise RuntimeError(
                    "MindCube conversion requires the official source QA file."
                )
            # Continue using the original MindCube image identity: the last three segments are category/scene/file, and the image order remains unchanged.
            pictures = row["images"]
            paths = [
                (args.source_root / Path(*Path(path).parts[-3:])).resolve()
                for path in pictures
            ]
            sizes = []
            for path in paths:
                with Image.open(path) as image:
                    sizes.append(list(image.size))
            rows.append(
                dict(
                    id=row["id"],
                    question=row["question"],
                    gt_answer=row["gt_answer"],
                    images=list(map(str, paths)),
                    image_input=list(map(str, paths)),
                    image_sizes=sizes,
                    n_images=len(paths),
                    dataset="mindcube",
                    subtask=Path(pictures[0]).parts[-3],
                )
            )
    with (output / "qa.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "conversion.json").write_text(
        json.dumps(
            dict(
                benchmark="mindcube",
                rows=len(rows),
                is_complete=args.max_samples is None,
                official_questions_answers_preserved=True,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    """Unified output position, official field conversion, image extraction, and video frame sampling continue to use the original implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Benchmark directory that contains raw/.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Official QA file (required for MMSI, ViewSpatial and VSI).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse media already extracted under <source-root>/raw/extracted.",
    )
    # Number of engineering inspection samples; omitting it processes the full official evaluation set, with the report explicitly recording the scope.
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive number")
    source, output = (
        args.source_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    if args.benchmark in ("mmsi", "viewspatial", "vsi") and not args.input_file:
        parser.error("The benchmark must specify the official --input-file")
    if args.input_file and not args.input_file.is_file():
        raise FileNotFoundError(args.input_file)
    output.mkdir(parents=True)
    if args.benchmark == "mindcube":
        return convert_mindcube(args, output)
    command = [
        sys.executable,
        str(HERE / "sources" / (args.benchmark + ".py")),
        "--output-file",
        str(output / "qa.jsonl"),
        "--stats-file",
        str(output / "conversion.json"),
    ]
    if args.benchmark == "cvbench":
        command += [
            "--raw-dir",
            str(source / "raw"),
            "--image-root",
            str(output / "images"),
        ]
    elif args.benchmark == "mmsi":
        command += [
            "--input-file",
            str(args.input_file.resolve()),
            "--image-root",
            str(output / "images"),
        ]
    else:
        command += ["--data-root", str(source)]
        if args.benchmark == "spar":
            command += ["--image-root", str(output / "images")]
        if args.benchmark in ("viewspatial", "vsi"):
            command += ["--input-file", str(args.input_file.resolve())]
        if args.benchmark == "vsi":
            # Consistent with the public 8B evaluation: fix 32 frames and original transformer pixel budget, do not secretly alter images to fit machine VRAM.
            command += ["--num-frames", "32", "--frame-root", str(output / "frames")]
        if args.skip_extract and args.benchmark in ("spbench", "viewspatial", "vsi"):
            command += ["--skip-extract"]
    if args.max_samples is not None:
        command += ["--max-samples", str(args.max_samples)]
    subprocess.run(command, check=True)
    with (output / "qa.jsonl").open() as handle:
        rows = normalize_images(
            [json.loads(line) for line in handle], args.benchmark, source, output
        )
    with (output / "qa.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = json.loads((output / "conversion.json").read_text())
    report.update(
        benchmark=args.benchmark,
        official_questions_preserved=True,
        official_answers_preserved=True,
        is_complete=args.max_samples is None,
        next_step="Extract geometry from qa.jsonl with --image-field images, then package the evaluation inputs.",
    )
    (output / "conversion.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
