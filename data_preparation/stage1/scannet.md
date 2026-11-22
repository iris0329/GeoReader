# Download ScanNet for Stage 1

Request access through the [official ScanNet site](https://www.scan-net.org/ScanNet/)
and follow the [official download instructions](https://github.com/ScanNet/ScanNet).
Use the downloader provided after access is approved.

Download the training `.sens` recordings, scene metadata, meshes, segmentations
and instance aggregations. Keep the original scene IDs and filenames:

```text
raw/scannet/scans/<scene>/
├── <scene>.sens
├── <scene>.txt
├── <scene>_vh_clean_2.ply
├── <scene>_vh_clean_2.0.010000.segs.json
└── <scene>.aggregation.json
```

Pass `raw/scannet/scans` to `--scannet-root` in the
[Stage 1 generation commands](README.md#six-benchmark-gt-data).
The supplied `excluded_scenes.json` controls held-out scene exclusions.

For training with the released data, download
[GeoReader-MultiSource](https://huggingface.co/datasets/RRRong/GeoReader-MultiSource)
and use the [training entrypoints](../../README.md).
