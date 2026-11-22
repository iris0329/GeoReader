# Extract DA3 geometry

`extract.py` predicts geometry from ordered RGB image groups. It uses DA3 Nested Giant-Large 1.1 with the original settings: resolution 504, `upper_bound_resize`, and `saddle_balanced`.

Install the official [Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3) in a separate environment following its dependencies. We use commit `41736238f5bced4debf3f2a12375d2466874866d`. Download `depth-anything/DA3NESTED-GIANT-LARGE-1.1` with its `config.json` and `model.safetensors` into a local model directory.

```bash
pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@41736238f5bced4debf3f2a12375d2466874866d"
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --local-dir models/DA3NESTED-GIANT-LARGE-1.1
```

Input JSONL rows need an ordered `image_input` list. Extra QA fields are left untouched and are not used for inference. Identical ordered image groups are processed once. `--image-root` resolves relative paths; `--image-prefix` strips a prefix from input paths. Absolute paths are used as is. Use images without question markers (boxes).

For the MindCube camera-only recipe:

```bash
python data_preparation/da3/extract.py \
  --input raw/mindcube_train_10k.jsonl --image-root raw/mindcube/images \
  --image-prefix data/other_all_image_resize \
  --model-dir models/DA3NESTED-GIANT-LARGE-1.1 \
  --signals camera --output-dir datasets/mindcube_geometry
```

For pose, intrinsics and depth from a new image list:

```bash
python data_preparation/da3/extract.py \
  --input raw/image_groups.jsonl --image-root raw/images \
  --model-dir models/DA3NESTED-GIANT-LARGE-1.1 \
  --signals both --output-dir datasets/geometry
```

```text
geometry/
├── groups.jsonl
├── manifest.json
├── progress.json
└── geometry/<group>/
    ├── camera.npz   # multi-image groups only
    ├── depth.npz    # --signals both
    └── complete.json
```

Camera files contain extrinsics `[V,3,4]` in w2c convention, intrinsics `[V,3,3]` in original RGB pixel coordinates, ordered image paths, and image sizes `[V,2]` in width–height order. The first view is canonicalized to `[I|0]`. Depth files contain float16 `depth_maps` `[V,H,W]` in meters. Depth resizing preserves the RGB aspect ratio; incompatible view aspect ratios raise an error. Single-image groups produce depth only.

`--max-groups N` processes only the first N groups for a quick check. `--resume` skips groups that are already complete with the same settings and image order.

The released six-benchmark data uses source-specific geometry. To reproduce its results, use the packaged geometry rather than re-running this extractor.
