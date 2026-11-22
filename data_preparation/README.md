# Data preparation

These tools generate geometry, Stage 1 supervision and Stage 2 QA inputs. Downloading the released training packages does not require running them.

```text
data_preparation/
├── data_utils.py          # shared ordered-JSONL and file helpers
├── stage1/
│   ├── README.md
│   ├── build_windows.py   # raw ScanNet / ScanNet++ sensors → GT windows
│   ├── build_instances.py # raw mesh annotations + GT → visible-instance facts
│   ├── generate.py        # MindCube DA3 or GT geometry → Stage 1 JSONL
│   ├── templates.py
│   ├── depth_helpers.py
│   ├── depth_facts.py
│   ├── excluded_scenes.json
│   └── requirements.txt
├── stage2/
│   ├── README.md
│   ├── generate.py
│   └── mindcube/             # original MindCube download instructions
├── evaluation/               # official benchmark conversion and packaging
└── da3/
    ├── README.md
    ├── extract.py         # ordered RGB → camera / depth NPZ
    └── geometry.py        # reference frame, intrinsics and depth resizing
```

The two recipes have different inputs:

```text
MindCube DA3 Stage 1:
raw training-QA image groups → DA3 geometry → reference–target descriptions

Six-benchmark Stage 1:
raw sensors + GT → ordered GT windows → GT instance facts → descriptions

Stage 2:
upstream training QA + matching ordered geometry → answer-only QA JSONL
```

See [Stage 1 generation](stage1/README.md), [Stage 2 preparation](stage2/README.md) and [DA3 extraction](da3/README.md) for commands and formats. Generated JSONLs can be passed to `train_stage1.py --data-file` or `train_stage2.py --data-file`; the downloaded packages use `--data-root`. Train and evaluate the released recipes with the prepared Hugging Face packages. Upstream dataset links are listed in [SOURCES.md](../SOURCES.md).

Sensor decoding, mesh projection and DA3 inference are compute-heavy. Keep their outputs separate from the downloaded datasets.
