"""Convert official evaluation data within the SixBangChaoShu workspace for shared use in split exclusion and independent evaluation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


SOURCES = dict(
    cvbench=("cv_bench", "cv_bench_test_local.jsonl", None),
    mmsi=("mmsi_bench", "mmsi_bench_test_local.jsonl", "MMSI_Bench.parquet"),
    spar=("spar_bench", "spar_bench_test_local.jsonl", None),
    spbench=("spbench", "spbench_test_local.jsonl", None),
    viewspatial=(
        "viewspatial_bench",
        "viewspatial_bench_test_local.jsonl",
        "ViewSpatial-Bench.json",
    ),
    vsi=("vsi_bench", "vsi_bench_test_local.jsonl", "test.jsonl"),
)


def main():
    """Native source split checks read from fixed positions; here only the format is converted, questions and answers retain official content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(SOURCES), required=True)
    args = parser.parse_args()
    folder, name, raw_name = SOURCES[args.benchmark]
    root = args.workspace.resolve() / "data/1_benchmark" / folder
    output = root / "processed"
    command = [
        sys.executable,
        str(Path(__file__).with_name("convert.py")),
        "--benchmark",
        args.benchmark,
        "--source-root",
        str(root),
        "--output-dir",
        str(output),
    ]
    if raw_name:
        command += ["--input-file", str(root / "raw" / raw_name)]
    subprocess.run(command, check=True)
    (output / "qa.jsonl").rename(output / name)
    report = json.loads((output / "conversion.json").read_text())
    report["output_file"] = str(output / name)
    (output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
