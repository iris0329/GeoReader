"""Continue using the already-accepted DA3 coordinates, intrinsics, and depth-to-dimension transformation."""

import numpy as np
from PIL import Image

# Same 3% aspect ratio threshold as the training reader.
DEPTH_RGB_ASPECT_RATIO_REL_TOL = 0.03


def target_depth_hw_for_rgb(
    image_size: np.ndarray, process_res: int
) -> tuple[int, int]:
    """Compute Depth dimensions with fixed long side according to original RGB aspect ratio."""

    width, height = map(int, image_size)
    if width <= 0 or height <= 0 or process_res <= 0:
        raise RuntimeError(
            f"Cannot compute Depth size: image_size={image_size.tolist()}, process_res={process_res}"
        )
    if width >= height:
        return max(1, round(process_res * height / width)), process_res
    return process_res, max(1, round(process_res * width / height))


def resize_depth_stack_to_rgb_aspect(
    depth_maps: np.ndarray,
    image_sizes: np.ndarray,
    *,
    process_res: int,
) -> np.ndarray:
    """Scale DA3's 14-multiple output back to original RGB ratio while preserving stackable (V,H,W)."""

    depth = np.asarray(depth_maps, dtype=np.float32)
    sizes = np.asarray(image_sizes, dtype=np.int32)
    if depth.ndim != 3 or sizes.shape != (depth.shape[0], 2):
        raise RuntimeError(
            f"Depth/RGB view mismatch: depth={depth.shape}, image_sizes={sizes.shape}"
        )
    target_shapes = [target_depth_hw_for_rgb(size, process_res) for size in sizes]
    if len(set(target_shapes)) != 1:
        raise RuntimeError(
            f"Different RGB aspect ratios on the same line cannot be saved as unified (V,H,W) Depth: image_sizes={sizes.tolist()}, target_shapes={target_shapes}"
        )
    target_h, target_w = target_shapes[0]
    if depth.shape[1:] == (target_h, target_w):
        return depth
    resized = [
        np.asarray(
            Image.fromarray(frame).resize(
                (target_w, target_h),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
        for frame in depth
    ]
    return np.stack(resized, axis=0)


def assert_depth_rgb_aspect(
    depth_maps: np.ndarray,
    image_sizes: np.ndarray,
    *,
    context: str,
) -> None:
    """Check Depth and RGB aspect ratios against the 3% threshold of Stage2 consumer."""

    depth = np.asarray(depth_maps)
    sizes = np.asarray(image_sizes, dtype=np.float64)
    if depth.ndim != 3 or sizes.shape != (depth.shape[0], 2):
        raise RuntimeError(
            f"Depth/RGB view mismatch: {context}, depth={depth.shape}, image_sizes={sizes.shape}"
        )
    depth_ratio = float(depth.shape[2]) / float(depth.shape[1])
    rgb_ratios = sizes[:, 0] / sizes[:, 1]
    errors = np.abs(rgb_ratios - depth_ratio) / np.maximum(rgb_ratios, depth_ratio)
    bad = np.where(errors > DEPTH_RGB_ASPECT_RATIO_REL_TOL)[0]
    if bad.size:
        view = int(bad[0])
        raise RuntimeError(
            f"Inconsistent aspect ratio between depth and RGB: {context}, view={view}, depth_hw={depth.shape[1:]}, rgb_wh={sizes[view].astype(int).tolist()}, rel_error={float(errors[view]):.6f}"
        )


def scale_intrinsics_to_original(
    intrinsics: np.ndarray,
    processed_sizes: np.ndarray,
    original_sizes: np.ndarray,
) -> np.ndarray:
    """Scale the DA3 processed pixel coordinates K back to each QA's original RGB coordinates."""

    result = np.asarray(intrinsics, dtype=np.float32).copy()
    processed = np.asarray(processed_sizes, dtype=np.float32)
    original = np.asarray(original_sizes, dtype=np.float32)
    if np.any(processed <= 0) or np.any(original <= 0):
        raise RuntimeError("intrinsics scaling encounters non-positive dimensions")
    scale_x = original[:, 0] / processed[:, 0]
    scale_y = original[:, 1] / processed[:, 1]
    result[:, 0, 0] *= scale_x
    result[:, 0, 2] *= scale_x
    result[:, 1, 1] *= scale_y
    result[:, 1, 2] *= scale_y
    expected_bottom = np.broadcast_to(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        result[:, 2, :].shape,
    )
    if not np.allclose(result[:, 2, :], expected_bottom, rtol=0.0, atol=1e-5):
        raise RuntimeError(
            "The homogeneous bottom row of DA3 intrinsics is not [0,0,1]."
        )
    return result


def canonicalize_w2c(extrinsics: np.ndarray) -> np.ndarray:
    """Rewrite the scene's global w2c as origin at the camera of frame 0 in this line."""

    values = np.asarray(extrinsics, dtype=np.float64)
    first_w2c = np.eye(4, dtype=np.float64)
    first_w2c[:3] = values[0]
    first_c2w = np.linalg.inv(first_w2c)
    result: list[np.ndarray] = []
    for value in values:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3] = value
        result.append((w2c @ first_c2w)[:3])
    output = np.stack(result).astype(np.float32)
    identity = np.concatenate(
        [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
        axis=1,
    )
    if not np.allclose(output[0], identity, rtol=0.0, atol=1e-5):
        raise RuntimeError(
            f"After canonicalizing frame 0, camera is not [I|0]: got={output[0]}"
        )
    return output
