"""Evaluate shared bounded RGB decoding; preserve input order, report errors for bad images directly to the main process."""

import atexit
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from PIL import Image

# Bound image-decoding threads per evaluation.
WORKERS = int(os.environ.get("EVAL_IMAGE_WORKERS", "1"))
if WORKERS < 1:
    raise ValueError("EVAL_IMAGE_WORKERS must be a positive integer")
# Spawn decoding workers so they do not inherit the parent CUDA context.
MODE = os.environ.get("EVAL_IMAGE_POOL", "thread")
if MODE not in {"thread", "process"}:
    raise ValueError("EVAL_IMAGE_POOL must be thread/process")
_POOL = None


def _read_rgb(item):
    """CPU subprocess only reads images, does not load models; exceptions carry samples and paths."""
    path, benchmark_name, sample_id = item
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Image not found: benchmark={benchmark_name}, sample={sample_id}, path={path}"
        )
    with Image.open(path) as image:
        return image.convert("RGB")


def load_rgb_images(paths, *, benchmark_name, sample_id):
    """Parallel image reading and decoding, map returns frames in original order, avoiding concurrent frame disorder."""
    global _POOL
    if _POOL is None and WORKERS > 1:
        if MODE == "process":
            _POOL = ProcessPoolExecutor(
                max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")
            )
        else:
            _POOL = ThreadPoolExecutor(max_workers=WORKERS)
        atexit.register(_POOL.shutdown)
        print(f"[RGB decode] pool={MODE}, workers={WORKERS}", flush=True)
    items = [(str(path), benchmark_name, sample_id) for path in paths]
    # map ensures output order remains unchanged, has a maximum process limit, and does not split 32 frames into 32 GPU models.
    return (
        list(_POOL.map(_read_rgb, items))
        if _POOL is not None
        else [_read_rgb(item) for item in items]
    )
