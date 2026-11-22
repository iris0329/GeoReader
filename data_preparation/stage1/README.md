# Generate Stage 1 supervision

The MindCube DA3 recipe uses DA3 predictions. The six-benchmark recipe uses GT geometry. Descriptions are generated from templates; no language model is used.

## MindCube DA3 data

First run the [DA3 extractor](../da3/README.md) on the original 10K training-QA image groups. Use the same image-root and prefix mapping in both commands:

```bash
python data_preparation/stage1/generate.py --recipe mindcube_da3_qwen25_3b \
  --qa raw/mindcube_train_10k.jsonl --image-root raw/mindcube/images \
  --image-prefix data/other_all_image_resize \
  --geometry-index datasets/mindcube_geometry/groups.jsonl \
  --output-dir datasets/mindcube_stage1

python train_stage1.py --recipe mindcube_da3_qwen25_3b --branch camera \
  --base-model models/Qwen2.5-VL-3B-Instruct --data-root raw/mindcube/images \
  --data-file datasets/mindcube_stage1/camera.jsonl --output-dir outputs/mindcube-stage1
```

The raw QA must provide `idx` and ordered `image_input`; questions and answers do not generate Stage 1 labels. Each image group uses its first view as the reference. Distances are rounded to 0.1 m and headings to 1 degree. Pair files select the original first and target views without re-estimating geometry. The official input produces 2,072 groups and 3,375 pairs.

Outputs are `camera.jsonl`, `da3_camera_pair/<group>/0_<target>/camera.npz`, and `generation.json`. The generation report lists the inputs and output counts.

## Six-benchmark GT data

Install the main repository dependencies and the sensor decoder:

```bash
pip install -r data_preparation/stage1/requirements.txt
```

Follow the [ScanNet download instructions](scannet.md)
and [ScanNet++ download instructions](scannetpp.md),
then prepare their raw `scans/` and `data/` directories. ScanNet scenes need the `.sens`, mesh, segmentation and aggregation files; ScanNet++ scenes need the iPhone RGB/depth/pose files and the aligned mesh and instance annotations under `scans/`. FFmpeg and FFprobe are required for video decoding.

`excluded_scenes.json` lists the excluded scenes. The released window selection is reproduced only with the same raw scene pool.

```bash
python data_preparation/stage1/build_windows.py build \
  --scannet-root raw/scannet/scans --scannetpp-root raw/scannetpp/data \
  --ffmpeg /usr/bin/ffmpeg --ffprobe /usr/bin/ffprobe \
  --output-root datasets/stage1_source --num-workers 8

python data_preparation/stage1/build_instances.py \
  --scannet-root raw/scannet/scans --scannetpp-root raw/scannetpp/data \
  --manifest datasets/stage1_source/manifests/windows.jsonl \
  --instance-root datasets/stage1_source/gt_instance \
  --index-jsonl datasets/stage1_source/manifests/instances.jsonl --num-workers 8

python data_preparation/stage1/generate.py --recipe sixbench_qwen3_8b \
  --windows datasets/stage1_source/manifests/windows.jsonl \
  --instance-root datasets/stage1_source/gt_instance \
  --output-dir datasets/stage1_generated --workers 8
```

`build_windows.py` also has `scan`, `smoke` (a quick check on a few scenes) and `validate` subcommands. `build_instances.py --scene scannet/SCENE_ID` builds a single scene for a quick check; `--verify-only` checks existing outputs. `generate.py --max-windows N` processes only the first N windows.

The data flow is:

```text
raw sensors
  → images/, gt_pose/, gt_depth/, manifests/windows.jsonl

raw mesh + segmentation + instance labels + GT pose/depth
  → gt_instance/<source>/<scene>/<frame>.npz

GT windows + instance facts
  → stage1_generated/camera.jsonl
  → stage1_generated/depth.jsonl
  → stage1_generated/camera_windows/<source>/<scene>/<window>.npz
```

Each window-manifest row contains `source`, `scene_id` and ordered `image_paths`, `pose_paths`, `depth_paths`. GT pose files contain `c2w`, `intrinsic`, `image_size`; GT depth files contain `depth_map` as uint16 millimeters and `depth_size`. The instance builder applies the recipe's visibility thresholds and label filtering.

Camera descriptions express right/forward displacement and heading relative to the first view. Depth descriptions first choose eight objects by visible area, sort them by surface distance and retain five; entirely empty windows are excluded. The full recipe yields 29,450 camera windows and 35,422 depth windows.

Pass the generated files to the common training entrypoint:

```bash
torchrun --standalone --nproc_per_node=4 train_stage1.py --branch camera \
  --base-model models/Qwen3-VL-8B-Instruct --data-root datasets/stage1_source \
  --data-file datasets/stage1_generated/camera.jsonl --output-dir outputs/stage1-camera

torchrun --standalone --nproc_per_node=4 train_stage1.py --branch depth \
  --base-model models/Qwen3-VL-8B-Instruct --data-root datasets/stage1_source \
  --data-file datasets/stage1_generated/depth.jsonl --output-dir outputs/stage1-depth
```

The depth entrypoint expands each window into per-frame records internally.
