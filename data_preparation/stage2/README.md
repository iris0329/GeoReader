# Prepare Stage 2 QA

Stage 2 learns from questions and answers. Stage 1 descriptions are not QA targets. The training target is answer-only:

```text
<think></think>
<answer>ANSWER_FROM_THE_TRAINING_ANNOTATION</answer>
```

```text
upstream training QA / questions constructed from raw training annotations
                       +
ordered RGB and the corresponding ready-to-use geometry
                       |
                       v
                 generate.py
                       |
                       v
           qa.jsonl + generation.json
                       |
                       v
       train_stage2.py --data-file ...
```

The generator keeps input order, question/answer pairs and image order. It does not sample questions or use a model to invent answers. Geometry is joined by the complete ordered image list; missing groups, mismatched sizes and invalid files fail immediately. Generated files store absolute paths to local assets. The released packages use relative paths and work directly with `--data-root`.

## MindCube: original 10K QA + DA3

See [MindCube downloads](mindcube/README.md) for the fixed released package and
the original-source download links.

Use the original MindCube 10K training JSONL, which contains `idx`, `text_input`, `answer` and `image_input`. Its released CoT is not copied into the target. The original question preprocessing removes the introductory viewpoint hint; the answer and substantive question are retained. This is the Qwen2.5-VL-3B camera-only recipe.

First extract geometry from the same ordered training images, then build QA:

```bash
python data_preparation/da3/extract.py \
  --input raw/mindcube_train_10k.jsonl \
  --image-root raw/mindcube/images --image-prefix data/other_all_image_resize \
  --model-dir models/DA3NESTED-GIANT-LARGE-1.1 \
  --signals camera --output-dir datasets/mindcube_geometry

python data_preparation/stage2/generate.py \
  --recipe mindcube_da3_qwen25_3b --qa raw/mindcube_train_10k.jsonl \
  --image-root raw/mindcube/images --image-prefix data/other_all_image_resize \
  --geometry-index datasets/mindcube_geometry/groups.jsonl \
  --output-dir datasets/mindcube_stage2 --workers 4

python train_stage2.py --recipe mindcube_da3_qwen25_3b --method georeader \
  --base-model models/Qwen2.5-VL-3B-Instruct \
  --data-root raw/mindcube/images --data-file datasets/mindcube_stage2/qa.jsonl \
  --camera-reader outputs/mindcube-stage1/checkpoint-8440 \
  --output-dir outputs/mindcube-stage2
```

For the original input, the output contains 10,000 QA rows referencing 2,072 ordered image groups. The Stage 1 pair generator can use the same training geometry index. Evaluation images and QA stay separate from training; evaluation inputs must never be passed to this training generator. See the [DA3 README](../da3/README.md) for installation and the geometry format.

## Six benchmarks: QA construction and fixed training mixture

Download the prepared [GeoReader-MultiSource package](https://huggingface.co/datasets/RRRong/GeoReader-MultiSource) for the released training recipe. Extraction and training commands are in the [root README](../../README.md). Upstream datasets are listed in [SOURCES.md](../../SOURCES.md).

The released six-benchmark model uses 36 training files with 135,399 QA rows, built as follows:

```text
public training QA
  SpatialLadder / SPAR / Spatial-MM
       -> source-specific filtering, answer formatting and target markers

raw training RGB + annotations
  VSI / VLM-3R construction workflows
       -> static geometry, appearance order and route questions
  ViewSpatial construction workflows
       -> five balanced spatial task groups
  MMSI-style construction
       -> camera/object/region relations, motion, attributes and multi-step QA
  single-image distance construction
       -> near/far questions from training object geometry

all selected training QA
       -> source-specific split checks
       -> attach the matching geometry and preserve marker metadata
       -> 36 files in the fixed training order
```

The fixed mixture is:

```text
Files     QA family                         Rows
01--08    VSI-style / VLM-3R workflow        34,449
09        SpatialLadder                     20,680
10, 13    SPAR                              15,000
11        ViewSpatial                       30,000
12        Spatial-MM                         1,291
14--34    MMSI-style                        23,979
35--36    single-image relative distance    10,000
Total                                      135,399
```

MMSI-style questions are built only from training images and annotations; MMSI-Bench is used only for evaluation.

The package contains the selected QA, marker instructions and assembled geometry consumed by training. Use `--data-root` to train on its fixed manifest. The generic joiner below supports new annotation sources with matching geometry.

### Build training rows from new upstream annotations

Provide upstream annotations in this format:

```json
{"id":"scene1-question1","image_input":["scene1/0001.jpg","scene1/0005.jpg"],"question":"Question including its options, when applicable","answer":"A","metadata":{"task_type":"relative_distance"}}
```

`question` must include any answer options; `answer` is the exact supervised answer string, including a numeric answer when appropriate. `metadata` carries source/task labels, marker instructions and coordinate-rewriting fields needed by the corresponding training sample. Rows containing `text_output`, `pose_path`, `depth_path` or `geometry_provenance` are rejected by this interface.

For a multiple-choice question, `question` must already include its options in the intended order; the joiner does not create or shuffle options.

The geometry index uses the DA3 extractor's `groups.jsonl` interface:

```json
{"image_input":["/absolute/rgb/scene1/0001.jpg","/absolute/rgb/scene1/0005.jpg"],"has_pose":true,"pose_path":"geometry/000000/camera.npz","has_depth":true,"depth_path":"geometry/000000/depth.npz","geometry_mode":"ordered_multiview"}
```

Relative RGB paths in the index are resolved against `--image-root`; geometry paths are resolved against the index's directory. A missing signal must be explicitly represented by a false availability flag and a null path. The four availability combinations are supported. The files must already be aligned to the QA images, with `image_input` and width/height `image_sizes` stored in each NPZ. Keep annotated QA images and unmarked geometry inputs aligned; preserve target-marker instructions in `metadata` when the training collator draws boxes.

To reproduce the released results, use the packaged geometry instead of re-extracting it.

```bash
python data_preparation/stage2/generate.py \
  --recipe sixbench_qwen3_8b --qa raw/training_questions.jsonl \
  --image-root raw/rgb --geometry-index datasets/geometry/groups.jsonl \
  --output-dir datasets/stage2_generated --workers 4
```

The result has `image_input`, `image_sizes`, `n_images`, `text_input`, `answer`, `text_output`, `has_pose`, `pose_path`, `has_depth` and `depth_path`, plus the provided training metadata. `generation.json` records input/output counts, ordered groups, modality counts and whether `--max-rows` produced only a subset.

Pass generated files to the existing training entrypoint in the intended mixture order:

```bash
torchrun --standalone --nproc_per_node=4 train_stage2.py \
  --recipe sixbench_qwen3_8b --method georeader \
  --base-model models/Qwen3-VL-8B-Instruct --data-root raw/rgb \
  --data-file datasets/stage2_generated/qa.jsonl \
  --camera-reader outputs/stage1-camera/checkpoint-73630 \
  --depth-reader outputs/stage1-depth/checkpoint-88560 \
  --output-dir outputs/stage2-generated
```

`--data-file` accepts multiple six-benchmark JSONL files in mixture order. The six-benchmark recipe currently expects the full 135,399-row mixture; smaller inputs are accepted only with `--smoke`. Without `--data-file`, the 36 files in `--data-root` are used. The MindCube recipe accepts one 10,000-row file.
