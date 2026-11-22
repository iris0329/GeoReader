# MindCube Stage 2 inputs

To reproduce the released MindCube model, download the
[GeoReader-MindCube package](https://huggingface.co/datasets/RRRong/GeoReader-MindCube):

```bash
hf download RRRong/GeoReader-MindCube --repo-type dataset \
  --local-dir /path/to/datasets/GeoReader-MindCube
tar -xf /path/to/datasets/GeoReader-MindCube/archives/mindcube.tar \
  -C /path/to/datasets/GeoReader-MindCube
```

The package contains separate training and evaluation QA, RGB, DA3 geometry and
Stage 1 supervision. Pass it to the entrypoints with `--data-root`. See the [root README](../../../README.md) for training/eval commands.

To rebuild from original inputs, use the authors'
[MindCube repository](https://github.com/mll-lab-nu/MindCube) and
[official dataset](https://huggingface.co/datasets/MLL-Lab/MindCube). Obtain the
original 10K training JSONL and its referenced `other_all_image_resize` images;
obtain TinyBench QA/images separately for evaluation. The training joiner needs
`idx`, `text_input`, `answer` and the ordered `image_input` list. Follow the
[MindCube generation commands](../README.md): original training RGB → DA3
camera geometry → answer-only QA.
