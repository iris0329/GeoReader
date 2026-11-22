# Convert official evaluation data

These converters preserve official evaluation questions and answers. They decode
Parquet images, extract benchmark media or sample the official videos, then
produce the QA JSONL consumed by GeoReader.

```text
official Parquet / JSON / video
             |
          convert.py
             |
  qa.jsonl + images/ or frames/
             + matching ordered geometry
             |
          package.py
             |
     eval/<benchmark>/qa.jsonl
```

Install `data_preparation/requirements.txt` and FFmpeg. Example:

```bash
python data_preparation/evaluation/convert.py \
  --benchmark mmsi --source-root raw/mmsi_bench \
  --input-file raw/mmsi_bench/raw/MMSI_Bench.parquet \
  --output-dir datasets/converted/mmsi
```

Required official inputs:

```text
benchmark    source-root/raw/
mindcube     official TinyBench JSONL and other_all_image/ (separate roots)
cvbench      test_2d.parquet, test_3d.parquet
mmsi         MMSI_Bench.parquet
spar         data/test-*.parquet
spbench      SPBench-SI.parquet, SPBench-MV.parquet, original media archives
viewspatial  ViewSpatial-Bench.json, original media archives
vsi          test.jsonl, original video archives
```

MindCube uses `--source-root raw/mindcube/other_all_image` and
`--input-file raw/mindcube/MindCube_tinybench.jsonl`. For `mmsi`, `viewspatial`
and `vsi`, pass `--input-file` explicitly. VSI retains 32 uniformly sampled frames
and the original conversion pixel budget. `--max-samples N` converts only the first
N rows for a quick check.

Geometry must refer to the same ordered RGB group and sizes. A `groups.jsonl`
index contains `image_input`, `has_pose`, `pose_path`, `has_depth`, `depth_path`
and `geometry_mode`, following the [geometry format](../da3/README.md).
`package.py` checks that each geometry file matches its images. To evaluate the
released checkpoint, use the packaged evaluation data and its geometry.

```bash
python data_preparation/evaluation/package.py \
  --benchmark mmsi --qa datasets/converted/mmsi/qa.jsonl \
  --geometry-index datasets/geometry/groups.jsonl \
  --output-dir datasets/rebuilt-evaluation
```

MindCube uses its own package with camera-only DA3 geometry. For the six
benchmarks, run `package.py` once per benchmark to add all six to the same package.
Packaged paths are relative, so the output directory can be moved. Use
`--allow-subset` to package a partial benchmark for a quick check.
