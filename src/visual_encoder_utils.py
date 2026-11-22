#!/usr/bin/env python3
"""Qwen visual encoder shared utilities."""

from __future__ import annotations

import torch


def encode_visual_in_chunks(
    visual_model,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    visual_chunk_size: int,
    *,
    context: str,
) -> torch.Tensor:
    """Chunk Qwen visual encoder by number of images to avoid OOM when forwarding multiple images at once."""
    if pixel_values is None:
        raise RuntimeError(f"{context}: pixel_values cannot be empty.")
    if image_grid_thw is None:
        raise RuntimeError(f"{context}: image_grid_thw cannot be empty.")
    if image_grid_thw.dim() != 2 or int(image_grid_thw.shape[1]) != 3:
        raise RuntimeError(
            f"{context}: image_grid_thw must be [num_images,3], current shape={tuple(image_grid_thw.shape)}"
        )
    num_images = int(image_grid_thw.shape[0])
    if num_images <= 0:
        raise RuntimeError(f"{context}: image_grid_thw contains no images.")

    # The Qwen processor concatenates the visual patches from multiple images in image order to form pixel_values.
    # Each image has exactly T*H*W patches, so patches can be extracted precisely based on image_grid_thw.
    patch_counts = [int(thw.prod().item()) for thw in image_grid_thw]
    expected_patches = int(sum(patch_counts))
    actual_patches = int(pixel_values.shape[0])
    if expected_patches != actual_patches:
        raise RuntimeError(
            f"{context}: pixel_values patch number and image_grid_thw are inconsistent, expected={expected_patches}, actual={actual_patches}, image_grid_thw={image_grid_thw.detach().cpu().tolist()}"
        )

    pixel_values = pixel_values.type(visual_model.dtype)
    chunk_size = int(visual_chunk_size)
    if chunk_size <= 0 or num_images <= chunk_size:
        return visual_model(pixel_values, grid_thw=image_grid_thw).detach()

    chunks = []
    patch_start = 0
    for image_start in range(0, num_images, chunk_size):
        image_end = min(image_start + chunk_size, num_images)
        patch_end = patch_start + sum(patch_counts[image_start:image_end])
        chunks.append(
            visual_model(
                pixel_values[patch_start:patch_end],
                grid_thw=image_grid_thw[image_start:image_end],
            ).detach()
        )
        patch_start = patch_end

    if patch_start != actual_patches:
        raise RuntimeError(
            f"{context}: visual chunk sliced, patch count inconsistent, used={patch_start}, actual={actual_patches}"
        )
    return torch.cat(chunks, dim=0)
