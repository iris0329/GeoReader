# Download ScanNet++ for Stage 1

Register at the [official ScanNet++ portal](https://scannetpp.mlsg.cit.tum.de/scannetpp/)
and follow its [download documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation).
Use your own dataset access credentials with the upstream downloader.

Download the training scenes with their iPhone RGB, depth and pose streams,
plus aligned meshes and instance annotations. Preserve the official layout:

```text
raw/scannetpp/data/<scene>/
├── iphone/
│   ├── rgb.mkv
│   ├── depth.bin
│   └── pose_intrinsic_imu.json
└── scans/
    └── aligned mesh and semantic/instance annotations
```

Pass `raw/scannetpp/data` to `--scannetpp-root` in the
[Stage 1 generation commands](README.md#six-benchmark-gt-data).
Install FFmpeg and FFprobe for video decoding. The supplied
`excluded_scenes.json` controls held-out scene exclusions.

For training with the released data, download
[GeoReader-MultiSource](https://huggingface.co/datasets/RRRong/GeoReader-MultiSource)
and use the [training entrypoints](../../README.md).
