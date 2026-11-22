"""Depth Interpreter — A depth token generation module independent of CameraInterpreter.

Structure and CameraInterpreter are symmetric:
  depth per-patch features → depth_proj → cross-attention × N → K depth token deltas

Input: depth map (per-patch depth statistics + optional frame/scene coordinates) + optional image tokens
Output: K depth tokens delta (hidden_size dimensional)

Independent <depth_start> [depth_pad×K] <depth_end> block injected into LLM.
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEPTH_ENCODER_TYPE_STATS_6D = "stats_6d"
DEPTH_ENCODER_TYPE_STATS_6D_LOG = "stats_6d_log"
DEPTH_ENCODER_TYPE_STATS_10D = "stats_10d"
DEPTH_ENCODER_TYPE_STATS_10D_LOG = "stats_10d_log"
DEPTH_ENCODER_TYPE_STATS_FOURIER = "stats_fourier"
DEPTH_ENCODER_TYPE_CENTER_FRAME_FOURIER = "center_frame_fourier"
DEPTH_ENCODER_TYPE_SCALAR_5D = "scalar_5d"
DEPTH_ENCODER_TYPE_CHOICES = (
    DEPTH_ENCODER_TYPE_STATS_6D,
    DEPTH_ENCODER_TYPE_STATS_6D_LOG,
    DEPTH_ENCODER_TYPE_STATS_10D,
    DEPTH_ENCODER_TYPE_STATS_10D_LOG,
    DEPTH_ENCODER_TYPE_STATS_FOURIER,
    DEPTH_ENCODER_TYPE_CENTER_FRAME_FOURIER,
    DEPTH_ENCODER_TYPE_SCALAR_5D,
)
OUTPUT_CONTRACT_GATED_DELTA = "gated_delta"
OUTPUT_CONTRACT_DIRECT_EMBEDDING = "direct_embedding"
OUTPUT_CONTRACT_CHOICES = (
    OUTPUT_CONTRACT_GATED_DELTA,
    OUTPUT_CONTRACT_DIRECT_EMBEDDING,
)
OUTPUT_GATE_TYPE_SCALAR = "scalar"
OUTPUT_GATE_TYPE_NONE = "none"
OUTPUT_GATE_TYPE_CHOICES = (OUTPUT_GATE_TYPE_SCALAR, OUTPUT_GATE_TYPE_NONE)

# ── stats_fourier encoding parameters (p9 kit 1)──
# Sincos wavelength (meters) for depth values: ranging from 0.25m (resolving adjacent depths) to ~31m (covering indoor range),
# Adjacent wavelength ratio of 5 times — fine ruler for phase, coarse ruler for counting turns, step-by-step ambiguity resolution (hour/minute/second hand logic).
# Angular frequency ω = 2π/λ, angle = ω·depth.
DEPTH_FOURIER_ANGULAR_FREQS = tuple(
    2.0 * math.pi / lam for lam in (0.25, 1.25, 6.25, 31.25)
)
# Frame number Fourier: plan original formula enc(i)=[sin(i/s),cos(i/s) for s in (1,4,16,64)], angular frequency = 1/s.
# Note that the 2π/λ convention cannot be applied: frame index is an integer, when λ=1, sin(2πi)≡0 and cos(2πi)≡1, causing high-frequency components to degenerate.
# Key: i is the absolute frame index within the sample (0-based, corresponding to image i+1 in CoT), not divided by the total number of frames—
# This is the entire secret behind why the 3rd frame's fingerprint matches exactly (no drift) in both the 4-frame and 32-frame samples.
FRAME_FOURIER_ANGULAR_FREQS = tuple(1.0 / s for s in (1.0, 4.0, 16.0, 64.0))
# 52 = u,v(2) + anchor(1) + frame_number Fourier(8) + valid_ratio(1) + 5 metric statistics × 8-dimensional sincos(40).
# valid_ratio / u / v are inherently bounded in [0,1], no need for sincos (sincos not relevant for LayerNorm imbalance issue), retain raw values to save dimensions.
DEPTH_FOURIER_GEOMETRY_DIM = (
    2
    + 1
    + 2 * len(FRAME_FOURIER_ANGULAR_FREQS)
    + 1
    + 5 * 2 * len(DEPTH_FOURIER_ANGULAR_FREQS)
)
# 12 = u,v(2) + absolute_frame_Fourier(8) + log1p_center_depth(1) + valid_ratio(1).
# frame_idx=0 already uniquely identifies image 1; since depth has no reference frame semantics, anchor_flag is not duplicated.
CENTER_FRAME_FOURIER_GEOMETRY_DIM = 2 + 2 * len(FRAME_FOURIER_ANGULAR_FREQS) + 1 + 1
# scalar_5d fixed to use frame_idx/31 to represent image 1..32; errors occur for frames beyond 32, and the denominator cannot be dynamically changed.
SCALAR_5D_MAX_FRAME_INDEX = 31
SCALAR_5D_GEOMETRY_DIM = 5
TOKEN_LAYOUT_GLOBAL = "global"
TOKEN_LAYOUT_PER_FRAME = "per_frame"
TOKEN_LAYOUT_CHOICES = (TOKEN_LAYOUT_GLOBAL, TOKEN_LAYOUT_PER_FRAME)


def depth_geometry_dim_for_encoder_type(encoder_type: str) -> int:
    if encoder_type in (DEPTH_ENCODER_TYPE_STATS_6D, DEPTH_ENCODER_TYPE_STATS_6D_LOG):
        return 6
    if encoder_type in (DEPTH_ENCODER_TYPE_STATS_10D, DEPTH_ENCODER_TYPE_STATS_10D_LOG):
        return 10
    if encoder_type == DEPTH_ENCODER_TYPE_STATS_FOURIER:
        return DEPTH_FOURIER_GEOMETRY_DIM
    if encoder_type == DEPTH_ENCODER_TYPE_CENTER_FRAME_FOURIER:
        return CENTER_FRAME_FOURIER_GEOMETRY_DIM
    if encoder_type == DEPTH_ENCODER_TYPE_SCALAR_5D:
        return SCALAR_5D_GEOMETRY_DIM
    raise RuntimeError(
        f"depth_encoder_type must be {DEPTH_ENCODER_TYPE_CHOICES}current ={encoder_type!r}"
    )


def fourier_encode_scalar(values: torch.Tensor, angular_freqs: tuple) -> torch.Tensor:
    """Encode a scalar as bounded sine/cosine features.

    Input shape (..., 1) becomes (..., 2 * len(angular_freqs)).
    Invalid depth patches retain valid_ratio=0 to distinguish missing depth."""
    parts = []
    for omega in angular_freqs:
        angle = values * float(omega)
        parts.append(torch.sin(angle))
        parts.append(torch.cos(angle))
    return torch.cat(parts, dim=-1)


@dataclass
class DepthInterpreterConfig:
    hidden_size: int = 2048
    num_depth_tokens: int = 8
    # global means the entire Window shares K tokens; per_frame means K tokens are output independently per frame.
    token_layout: str = TOKEN_LAYOUT_GLOBAL
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # stats_10d includes image coordinates and frame identity alongside depth statistics.
    depth_geometry_dim: int = 10
    depth_encoder_type: str = DEPTH_ENCODER_TYPE_STATS_10D
    geometry_rank: int = 32
    # The output contract distinguishes residual deltas from complete embedding vectors.
    output_contract: str = OUTPUT_CONTRACT_GATED_DELTA
    # direct_embedding must be none, so the module will not create the output_gate parameter.
    output_gate_type: str = OUTPUT_GATE_TYPE_SCALAR
    output_gate_init: float = 0.1
    geom_gate_init: float = 0.1
    # Only control the bias added to attention scores from raw geometry; the main path of geometry MLP -> memory K/V is always retained.
    use_geometry_attention_bias: bool = True
    patch_size: int = 14
    spatial_merge_size: int = 2
    use_image_tokens: bool = True
    # Depth Stage 1 is fixed as spatial block -> instruction -> assistant.
    depth_token_position: str = "before_question"

    def __post_init__(self):
        if self.token_layout not in TOKEN_LAYOUT_CHOICES:
            raise RuntimeError(
                f"token_layout must be {TOKEN_LAYOUT_CHOICES}current ={self.token_layout!r}"
            )
        if int(self.num_depth_tokens) <= 0:
            raise RuntimeError(
                f"num_depth_tokens must be > 0, current={self.num_depth_tokens}"
            )
        # The per-frame summary interface supports Kd=1 or Kd=2.
        if self.token_layout == TOKEN_LAYOUT_PER_FRAME and int(
            self.num_depth_tokens
        ) not in {1, 2}:
            raise RuntimeError(
                f"Depth per_frame only allows Kd=1 or Kd=2 per frame: num_depth_tokens={self.num_depth_tokens}"
            )
        if self.output_contract not in OUTPUT_CONTRACT_CHOICES:
            raise RuntimeError(
                f"output_contract must be {OUTPUT_CONTRACT_CHOICES}current ={self.output_contract!r}"
            )
        expected_gate_type = (
            OUTPUT_GATE_TYPE_NONE
            if self.output_contract == OUTPUT_CONTRACT_DIRECT_EMBEDDING
            else OUTPUT_GATE_TYPE_SCALAR
        )
        if self.output_gate_type != expected_gate_type:
            raise RuntimeError(
                f"Depth output_contract / output_gate_type combination is illegal: contract={self.output_contract!r}, gate_type={self.output_gate_type!r}, expected={expected_gate_type!r}"
            )
        if self.depth_token_position not in (
            "before_image",
            "before_question",
            "after_question",
        ):
            raise RuntimeError(
                f"depth_token_position must be before_image/before_question/after_question, current={self.depth_token_position!r}"
            )
        if not isinstance(self.use_geometry_attention_bias, bool):
            raise RuntimeError(
                f"use_geometry_attention_bias must be bool: actual={self.use_geometry_attention_bias!r}"
            )
        expected = depth_geometry_dim_for_encoder_type(self.depth_encoder_type)
        if int(self.depth_geometry_dim) != int(expected):
            raise RuntimeError(
                f"depth_geometry_dim and depth_encoder_type are inconsistent: type={self.depth_encoder_type!r}, dim={self.depth_geometry_dim}, expected={expected}"
            )

    def to_dict(self):
        return asdict(self)

    def output_token_count(self, num_frames: int) -> int:
        """Returns the number of Depth tokens actually output by the current Window based on the layout."""
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise RuntimeError(f"num_frames must be > 0, current={num_frames}")
        if self.token_layout == TOKEN_LAYOUT_GLOBAL:
            return int(self.num_depth_tokens)
        if self.token_layout == TOKEN_LAYOUT_PER_FRAME:
            return num_frames * int(self.num_depth_tokens)
        raise RuntimeError(f"Unknown token_layout={self.token_layout!r}")


def parse_depth_interpreter_config_dict(
    config_dict: dict, config_path: str
) -> DepthInterpreterConfig:
    """Strictly parse checkpoint-local Depth config, and directly report errors for missing or extra fields."""
    if not isinstance(config_dict, dict):
        raise RuntimeError(
            f"DepthInterpreterConfig must be a dict: {config_path}, type={type(config_dict)}"
        )
    config_dict = dict(config_dict)
    legacy_contract_fields = {"output_contract", "output_gate_type"}
    present_contract_fields = legacy_contract_fields.intersection(config_dict)
    legacy_contract = not present_contract_fields
    if legacy_contract:
        # Use only explicit mappings for missing scalar-gated-delta contract fields.
        config_dict["output_contract"] = OUTPUT_CONTRACT_GATED_DELTA
        config_dict["output_gate_type"] = OUTPUT_GATE_TYPE_SCALAR
        print(
            "[depth_config] missing output contract "
            f"-> gated_delta/scalar: {config_path}"
        )
    elif present_contract_fields != legacy_contract_fields:
        raise RuntimeError(
            f"Depth checkpoint's output_contract / output_gate_type must both exist: present={sorted(present_contract_fields)}, path={config_path}"
        )
    if "depth_token_position" not in config_dict and legacy_contract:
        # Report the before_question default when token placement is unspecified.
        config_dict["depth_token_position"] = "before_question"
        print(
            "[depth_config] missing depth_token_position "
            f"-> before_question: {config_path}"
        )
    known = {f.name for f in DepthInterpreterConfig.__dataclass_fields__.values()}
    extra = set(config_dict.keys()) - known
    missing = known - set(config_dict.keys())
    if extra:
        raise RuntimeError(
            f"Unknown field in DepthInterpreterConfig {extra}: {config_path}"
        )
    legacy_defaults = {
        "use_geometry_attention_bias": True,
        "token_layout": TOKEN_LAYOUT_GLOBAL,
    }
    if missing and missing.issubset(legacy_defaults):
        if (
            "token_layout" in missing
            and int(config_dict.get("num_depth_tokens", -1)) != 8
        ):
            # Only global K=8 configurations may omit token_layout.
            raise RuntimeError(
                f"Depth checkpoint must specify token_layout unless num_depth_tokens=8: num_depth_tokens={config_dict.get('num_depth_tokens')}, path={config_path}"
            )
        # Fill only missing fields with a uniquely defined configuration meaning.
        for key in sorted(missing):
            config_dict[key] = legacy_defaults[key]
            print(
                f"[depth_config] {config_path}: missing {key} "
                f"-> {legacy_defaults[key]!r}"
            )
        missing = set()
    if missing:
        raise RuntimeError(
            f"DepthInterpreterConfig missing field {missing}: {config_path}"
        )
    return DepthInterpreterConfig(**config_dict)


def build_depth_patch_stats(
    depth_maps: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    patch_size: int,
    image_sizes: torch.Tensor,
    encoder_type: str = DEPTH_ENCODER_TYPE_STATS_10D,
) -> torch.Tensor:
    """Construct per-patch depth geometry from depth maps, aligned with the patch grid of image_grid_thw.

    Args:
        depth_maps: (V, H_d, W_d) float32, units in meters (converted from uint16 mm)
        image_grid_thw: (V, 3) — (temporal_patches, grid_h, grid_w) for each view
        spatial_merge_size: Spatial merge for Qwen VL
        patch_size: Patch size for Qwen VL
        image_sizes: (V, 2) — [width, height] per view (original image dimensions)

    Returns:
        depth_geometry: (total_patches, D) float32
        stats_10d's total_patches aligns with Qwen LLM image token count:
          sum(grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size))
        stats_6d / stats_6d_log use the depth grid specified by their configuration.
        stats_6d / stats_6d_log:
          [mean, std, min, max, center_depth, valid_ratio]
        stats_10d / stats_10d_log:
          [u_norm, v_norm, view_scalar, anchor_flag,
           mean, std, min, max, center_depth, valid_ratio]
        stats_fourier (52-dimensional; grid and stats_10d both LLM-aligned):
          [u_norm, v_norm, anchor_flag,
           frame_fourier×8 (absolute frame number sincos, replaces view_scalar which drifts with total frame count)
           valid_ratio,
           mean/std/min/max/center each ×8 dimensions sincos (metric original values expanded, no log)
        center_frame_fourier (12-dimensional; grid and stats_fourier are the same):
          [u_norm, v_norm,
           frame_fourier×8 (absolute 0-based frame index, not divided by Window length),
           log1p(center_depth_m), valid_ratio]
        scalar_5d (5D; same grid as center_frame_fourier):
          [u_norm, v_norm, frame_idx/31.0, log1p(center_depth_m), valid_ratio]
        In stats_6d_log / stats_10d_log, depth stats use log(depth_m + 1.0), while valid_ratio still comes from raw depth.
    """
    device = depth_maps.device
    dtype = torch.float32
    depth_maps = depth_maps.to(device=device, dtype=dtype)

    num_views = int(depth_maps.shape[0])
    if int(image_grid_thw.shape[0]) != num_views:
        raise RuntimeError(
            f"depth_maps view count and image_grid_thw are inconsistent: depth={num_views}, grid={int(image_grid_thw.shape[0])}"
        )
    if (
        encoder_type == DEPTH_ENCODER_TYPE_SCALAR_5D
        and num_views > SCALAR_5D_MAX_FRAME_INDEX + 1
    ):
        raise RuntimeError(
            f"scalar_5d supports at most 32 frames; fixed frame scalar must be frame_idx/31.0: num_views={num_views}"
        )

    chunks = []
    for view_idx in range(num_views):
        grid_t, grid_h, grid_w = [int(x) for x in image_grid_thw[view_idx]]
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise RuntimeError(
                f"Invalid image_grid_thw: view={view_idx}, thw={[grid_t, grid_h, grid_w]}"
            )
        use_llm_aligned_grid = encoder_type in (
            DEPTH_ENCODER_TYPE_STATS_10D,
            DEPTH_ENCODER_TYPE_STATS_10D_LOG,
            DEPTH_ENCODER_TYPE_STATS_FOURIER,
            DEPTH_ENCODER_TYPE_CENTER_FRAME_FOURIER,
            DEPTH_ENCODER_TYPE_SCALAR_5D,
        )
        if use_llm_aligned_grid:
            if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
                raise RuntimeError(
                    f"stats_10d's depth patch grid must align with Qwen LLM image token grid, but image_grid_thw cannot be divided by spatial_merge_size: view={view_idx}, thw={[grid_t, grid_h, grid_w]}, spatial_merge_size={spatial_merge_size}"
                )
            # stats_10d/stats_10d_log is a new metric for image-depth fusion: aligns with the Qwen merged LLM image token grid.
            processed_h = grid_h * patch_size
            processed_w = grid_w * patch_size
        else:
            # stats_6d uses the depth patch grid defined by its checkpoint configuration.
            processed_h = grid_h * spatial_merge_size * patch_size
            processed_w = grid_w * spatial_merge_size * patch_size

        depth_frame = depth_maps[view_idx]
        H_d, W_d = depth_frame.shape

        resized_raw = (
            F.interpolate(
                depth_frame.unsqueeze(0).unsqueeze(0),
                size=(processed_h, processed_w),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )

        raw_valid_px = resized_raw > 0.01
        use_log_depth = encoder_type in (
            DEPTH_ENCODER_TYPE_STATS_6D_LOG,
            DEPTH_ENCODER_TYPE_STATS_10D_LOG,
        )
        if use_log_depth:
            # log only performs fixed metric transformation, without per-scene normalization; this compresses distant depths but preserves absolute scale.
            depth_for_stats = torch.where(
                raw_valid_px,
                torch.log(resized_raw + 1.0),
                torch.zeros_like(resized_raw),
            )
        else:
            depth_for_stats = resized_raw

        # stats_10d aligns depth patches with merged RGB tokens; stats_6d uses its configured grid.
        patches = depth_for_stats.unfold(
            0, patch_size * spatial_merge_size, patch_size * spatial_merge_size
        ).unfold(1, patch_size * spatial_merge_size, patch_size * spatial_merge_size)
        n_patches_h, n_patches_w = patches.shape[0], patches.shape[1]
        patches = patches.contiguous().view(n_patches_h * n_patches_w, -1)
        raw_valid_patches = raw_valid_px.unfold(
            0, patch_size * spatial_merge_size, patch_size * spatial_merge_size
        ).unfold(1, patch_size * spatial_merge_size, patch_size * spatial_merge_size)
        raw_valid_patches = raw_valid_patches.contiguous().view(
            n_patches_h * n_patches_w, -1
        )

        # The log version must use raw depth to validate effectively; otherwise, small values after logging may be incorrectly penalized by thresholds; valid_ratio also indicates the true sampling coverage.
        valid_mask = raw_valid_patches
        valid_ratio = valid_mask.float().mean(dim=1, keepdim=True)

        safe_patches = patches.clone()
        safe_patches[~valid_mask] = float("nan")

        patch_mean = torch.nanmean(safe_patches, dim=1, keepdim=True)
        patch_mean = torch.where(
            torch.isnan(patch_mean), torch.zeros_like(patch_mean), patch_mean
        )

        diff = safe_patches - patch_mean
        diff[~valid_mask] = 0.0
        valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        patch_std = (diff.pow(2).sum(dim=1, keepdim=True) / valid_count).sqrt()

        patch_min_vals = patches.clone()
        patch_min_vals[~valid_mask] = 1e6
        patch_min = patch_min_vals.min(dim=1, keepdim=True).values
        patch_min = torch.where(valid_ratio > 0, patch_min, torch.zeros_like(patch_min))

        patch_max_vals = patches.clone()
        patch_max_vals[~valid_mask] = 0.0
        patch_max = patch_max_vals.max(dim=1, keepdim=True).values

        ppw = patch_size * spatial_merge_size
        center_y = (
            torch.arange(n_patches_h, device=device, dtype=dtype) * ppw + ppw // 2
        )
        center_x = (
            torch.arange(n_patches_w, device=device, dtype=dtype) * ppw + ppw // 2
        )
        cy_grid, cx_grid = torch.meshgrid(center_y, center_x, indexing="ij")
        cy_flat = cy_grid.reshape(-1).long().clamp(0, processed_h - 1)
        cx_flat = cx_grid.reshape(-1).long().clamp(0, processed_w - 1)
        center_depth = depth_for_stats[cy_flat, cx_flat].unsqueeze(1)

        depth_stats = torch.cat(
            [
                patch_mean,
                patch_std,
                patch_min,
                patch_max,
                center_depth,
                valid_ratio,
            ],
            dim=1,
        )

        if encoder_type in (
            DEPTH_ENCODER_TYPE_STATS_10D,
            DEPTH_ENCODER_TYPE_STATS_10D_LOG,
        ):
            # Include image coordinates and frame identity to distinguish depth patches across views.
            yy, xx = torch.meshgrid(
                torch.arange(n_patches_h, device=device, dtype=dtype),
                torch.arange(n_patches_w, device=device, dtype=dtype),
                indexing="ij",
            )
            u_norm = ((xx.reshape(-1) + 0.5) / float(n_patches_w)).unsqueeze(1)
            v_norm = ((yy.reshape(-1) + 0.5) / float(n_patches_h)).unsqueeze(1)
            view_scalar = torch.full(
                (depth_stats.shape[0], 1),
                float(view_idx) / float(max(1, num_views - 1)),
                device=device,
                dtype=dtype,
            )
            anchor_flag = torch.full(
                (depth_stats.shape[0], 1),
                1.0 if view_idx == 0 else 0.0,
                device=device,
                dtype=dtype,
            )
            geom = torch.cat(
                [u_norm, v_norm, view_scalar, anchor_flag, depth_stats], dim=1
            )
        elif encoder_type == DEPTH_ENCODER_TYPE_STATS_FOURIER:
            # stats_fourier (p9 kit 1) fixes two breakpoints:
            #   Breakpoint 1: stats_10d's view_scalar = i/(V-1) drifts with total frame count — "3rd frame" from 4-frame sample
            #     The encoding 0.43 for 32-frame samples is 0.097, and the short-frame learned rules are completely invalid when applied to long frames.
            #     Here, replace with multi-frequency sincos fingerprint based on absolute frame numbers, independent of total frame count.
            #   LayerNorm Imbalance: Metric raw scalars (0.3~15m) dominate LayerNorm; after sincos, all values are bounded [-1,1].
            yy, xx = torch.meshgrid(
                torch.arange(n_patches_h, device=device, dtype=dtype),
                torch.arange(n_patches_w, device=device, dtype=dtype),
                indexing="ij",
            )
            u_norm = ((xx.reshape(-1) + 0.5) / float(n_patches_w)).unsqueeze(1)
            v_norm = ((yy.reshape(-1) + 0.5) / float(n_patches_h)).unsqueeze(1)
            anchor_flag = torch.full(
                (depth_stats.shape[0], 1),
                1.0 if view_idx == 0 else 0.0,
                device=device,
                dtype=dtype,
            )
            # Frame number fingerprint: i refers to the absolute frame number (0-based) within the sample, and corresponds to CoT's "image i+1".
            frame_scalar = torch.full(
                (depth_stats.shape[0], 1), float(view_idx), device=device, dtype=dtype
            )
            frame_enc = fourier_encode_scalar(frame_scalar, FRAME_FOURIER_ANGULAR_FREQS)
            # 5 metric statistics (mean/std/min/max/center) expanded individually via sincos; valid_ratio retains raw value
            metric_stats = depth_stats[:, :5]
            valid_ratio_col = depth_stats[:, 5:6]
            stats_enc = fourier_encode_scalar(
                metric_stats.unsqueeze(-1), DEPTH_FOURIER_ANGULAR_FREQS
            )
            stats_enc = stats_enc.reshape(depth_stats.shape[0], -1)
            geom = torch.cat(
                [u_norm, v_norm, anchor_flag, frame_enc, valid_ratio_col, stats_enc],
                dim=1,
            )
        elif encoder_type == DEPTH_ENCODER_TYPE_CENTER_FRAME_FOURIER:
            # Depth has no reference frame semantics; frame_idx=0 uniquely identifies image 1, so remove the duplicate anchor_flag.
            yy, xx = torch.meshgrid(
                torch.arange(n_patches_h, device=device, dtype=dtype),
                torch.arange(n_patches_w, device=device, dtype=dtype),
                indexing="ij",
            )
            u_norm = ((xx.reshape(-1) + 0.5) / float(n_patches_w)).unsqueeze(1)
            v_norm = ((yy.reshape(-1) + 0.5) / float(n_patches_h)).unsqueeze(1)
            # frame_idx continues to use absolute 0-based indexing within the sample; it cannot be divided by Window length, otherwise the same frame will drift with V.
            frame_scalar = torch.full(
                (depth_stats.shape[0], 1), float(view_idx), device=device, dtype=dtype
            )
            frame_enc = fourier_encode_scalar(frame_scalar, FRAME_FOURIER_ANGULAR_FREQS)
            center_valid = raw_valid_px[cy_flat, cx_flat].unsqueeze(1)
            center_depth_raw = resized_raw[cy_flat, cx_flat].unsqueeze(1)
            # Center values are first judged valid if >0.01m; invalid centers are strictly set to 0, while valid centers are only fixed using log1p in metric units.
            center_depth_log1p = torch.where(
                center_valid,
                torch.log1p(center_depth_raw),
                torch.zeros_like(center_depth_raw),
            )
            geom = torch.cat(
                [u_norm, v_norm, frame_enc, center_depth_log1p, valid_ratio],
                dim=1,
            )
        elif encoder_type == DEPTH_ENCODER_TYPE_SCALAR_5D:
            yy, xx = torch.meshgrid(
                torch.arange(n_patches_h, device=device, dtype=dtype),
                torch.arange(n_patches_w, device=device, dtype=dtype),
                indexing="ij",
            )
            u_norm = ((xx.reshape(-1) + 0.5) / float(n_patches_w)).unsqueeze(1)
            v_norm = ((yy.reshape(-1) + 0.5) / float(n_patches_h)).unsqueeze(1)
            # Divide by 31.0 to ensure that the same image i is encoded identically in both the 4-frame and 32-frame windows.
            frame_scalar = torch.full(
                (depth_stats.shape[0], 1),
                float(view_idx) / float(SCALAR_5D_MAX_FRAME_INDEX),
                device=device,
                dtype=dtype,
            )
            center_valid = raw_valid_px[cy_flat, cx_flat].unsqueeze(1)
            center_depth_raw = resized_raw[cy_flat, cx_flat].unsqueeze(1)
            # Same as 12D baseline: center pixel is 0 when invalid, and only applies fixed metric log1p when valid.
            center_depth_log1p = torch.where(
                center_valid,
                torch.log1p(center_depth_raw),
                torch.zeros_like(center_depth_raw),
            )
            geom = torch.cat(
                [u_norm, v_norm, frame_scalar, center_depth_log1p, valid_ratio],
                dim=1,
            )
        else:
            geom = depth_stats

        if grid_t > 1:
            geom = geom.unsqueeze(0).expand(grid_t, -1, -1).reshape(-1, geom.shape[-1])

        chunks.append(geom)

    return torch.cat(chunks, dim=0)


class DepthCrossAttentionBlock(nn.Module):
    """Symmetric in structure to GeometryCrossAttentionBlock, except for the config class."""

    def __init__(self, config: DepthInterpreterConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads: hidden={config.hidden_size}, heads={config.num_heads}"
            )
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.geometry_rank = int(config.geometry_rank)
        self.num_tokens = int(config.num_depth_tokens)
        self.use_geometry_attention_bias = bool(config.use_geometry_attention_bias)

        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        if self.use_geometry_attention_bias:
            self.geom_key = nn.Linear(
                config.depth_geometry_dim, self.num_heads * self.geometry_rank
            )
            self.geom_query = nn.Parameter(
                torch.empty(self.num_tokens, self.num_heads, self.geometry_rank)
            )
            self.geom_gate = nn.Parameter(
                torch.tensor(float(config.geom_gate_init), dtype=torch.float32)
            )

        self.self_norm = nn.LayerNorm(self.hidden_size)
        self.self_attn = nn.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        ffn_hidden = int(self.hidden_size * float(config.mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_hidden, self.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_geometry_attention_bias:
            nn.init.normal_(
                self.geom_query, mean=0.0, std=1.0 / math.sqrt(self.geometry_rank)
            )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        geometry: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len, _ = queries.shape
        memory_len = int(memory.shape[1])

        q = self.q_proj(self.query_norm(queries))
        k = self.k_proj(self.memory_norm(memory))
        v = self.v_proj(memory)

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, memory_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
        scores = scores * (1.0 / math.sqrt(float(self.head_dim)))

        if self.use_geometry_attention_bias:
            # true preserves the original raw-geometry score completely; false means these parameters do not exist at all, and only standard QK attention is used.
            geom_k = self.geom_key(geometry.float())
            geom_k = geom_k.view(
                batch_size, memory_len, self.num_heads, self.geometry_rank
            )
            geom_k = geom_k.permute(0, 2, 1, 3).contiguous()
            geom_scores = torch.einsum(
                "khr,bhnr->bhkn",
                self.geom_query.float(),
                geom_k.float(),
            )
            geom_scores = geom_scores * (1.0 / math.sqrt(float(self.geometry_rank)))
            scores = scores + torch.tanh(self.geom_gate.float()) * geom_scores

        scores = scores.masked_fill(
            ~memory_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attn = F.softmax(scores, dim=-1).to(dtype=v.dtype)
        context = torch.matmul(attn, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.hidden_size)
        )
        queries = queries + self.dropout(self.out_proj(context))

        self_attn_out, _ = self.self_attn(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            need_weights=False,
        )
        queries = queries + self.dropout(self_attn_out)
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        return queries


class DepthInterpreter(nn.Module):
    """depth per-patch features + image tokens → K depth tokens output.

    Symmetric structure to CameraInterpreter, as an independent module.
    """

    def __init__(self, config: DepthInterpreterConfig):
        super().__init__()
        self.config = config

        self.image_norm = nn.LayerNorm(config.hidden_size)
        self.image_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.depth_norm = nn.LayerNorm(config.depth_geometry_dim)
        self.depth_proj = nn.Sequential(
            nn.Linear(config.depth_geometry_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.query = nn.Parameter(
            torch.empty(config.num_depth_tokens, config.hidden_size)
        )
        self.blocks = nn.ModuleList(
            [DepthCrossAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        if config.output_gate_type == OUTPUT_GATE_TYPE_SCALAR:
            self.output_gate = nn.Parameter(
                torch.tensor(float(config.output_gate_init), dtype=torch.float32)
            )

        if not bool(config.use_image_tokens):
            for param in self.image_norm.parameters():
                param.requires_grad = False
            for param in self.image_proj.parameters():
                param.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(
            self.query, mean=0.0, std=1.0 / math.sqrt(self.config.hidden_size)
        )
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _per_frame_token_counts(
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int,
    ) -> List[int]:
        """Calculate the number of depth patches per frame from Qwen grid."""
        if image_grid_thw.dim() != 2 or int(image_grid_thw.shape[-1]) != 3:
            raise RuntimeError(
                f"image_grid_thw must be [V,3], current shape={tuple(image_grid_thw.shape)}"
            )
        counts = []
        merge = int(spatial_merge_size)
        for thw in image_grid_thw.detach().cpu().tolist():
            grid_t, grid_h, grid_w = [int(value) for value in thw]
            if grid_h % merge != 0 or grid_w % merge != 0:
                raise RuntimeError(
                    f"Depth grid cannot be divided by spatial_merge_size: thw={thw}, merge={merge}"
                )
            count = grid_t * (grid_h // merge) * (grid_w // merge)
            if count <= 0:
                raise RuntimeError(f"Depth patch count per frame must > 0: thw={thw}")
            counts.append(count)
        return counts

    @staticmethod
    def _pack_per_frame(
        values: torch.Tensor,
        token_counts: List[int],
        *,
        context: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad the sequentially concatenated patch tensor to `[V, Pmax, D]` and return the valid position mask."""
        if values.dim() != 3 or int(values.shape[0]) != 1:
            raise RuntimeError(
                f"{context} Must be [1,N,D], current shape={tuple(values.shape)}"
            )
        total = sum(int(count) for count in token_counts)
        if not token_counts or total != int(values.shape[1]):
            raise RuntimeError(
                f"{context} Frame-wise token count inconsistency: counts={token_counts}, sum={total}, actual={int(values.shape[1])}"
            )
        max_tokens = max(int(count) for count in token_counts)
        packed = values.new_zeros(
            (len(token_counts), max_tokens, int(values.shape[-1]))
        )
        mask = torch.zeros(
            (len(token_counts), max_tokens),
            device=values.device,
            dtype=torch.bool,
        )
        start = 0
        for frame_idx, count in enumerate(token_counts):
            count = int(count)
            packed[frame_idx, :count] = values[0, start : start + count]
            mask[frame_idx, :count] = True
            start += count
        return packed, mask

    def forward(
        self,
        image_tokens: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        depth_maps: torch.Tensor,
        image_sizes: torch.Tensor,
        apply_output_gate: bool = True,
        return_patch_memory: bool = False,
    ):
        """
        Args:
            image_tokens: (1, N, hidden_size) or None
            image_grid_thw: (V, 3)
            depth_maps: (V, H_d, W_d) float32, unit: meters
            image_sizes: (V, 2) [width, height]

        Returns:
            Default returns output: (1, K, hidden_size). Returns when explicitly requesting patch memory.
            strict contract for summary_tokens / patch_memory / patch_mask / patch_counts.
        """
        module_dtype = next(self.parameters()).dtype
        use_image = bool(self.config.use_image_tokens)

        if use_image:
            if image_tokens is None:
                raise RuntimeError(
                    "When use_image_tokens=True, image_tokens must be provided."
                )
            if image_tokens.dim() == 2:
                image_tokens = image_tokens.unsqueeze(0)
            if int(image_tokens.shape[0]) != 1:
                raise RuntimeError(
                    f"DepthInterpreter v1 only supports batch_size=1, current={image_tokens.shape[0]}"
                )
            image_tokens = image_tokens.to(dtype=module_dtype)
            geometry_device = image_tokens.device
        else:
            # Reject RGB inputs when the Reader is configured for geometry-only input.
            if image_tokens is not None:
                raise RuntimeError(
                    "When use_image_tokens=False, image_tokens are not passed to DepthInterpreter."
                )
            geometry_device = depth_maps.device

        depth_geometry = build_depth_patch_stats(
            depth_maps=depth_maps,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=int(self.config.spatial_merge_size),
            patch_size=int(self.config.patch_size),
            image_sizes=image_sizes,
            encoder_type=str(self.config.depth_encoder_type),
        ).to(device=geometry_device, dtype=module_dtype)

        if int(depth_geometry.shape[-1]) != int(self.config.depth_geometry_dim):
            raise RuntimeError(
                f"depth geometry dim inconsistency: got={int(depth_geometry.shape[-1])}, expected={int(self.config.depth_geometry_dim)}"
            )
        depth_geometry = depth_geometry.unsqueeze(0)
        per_frame_counts = self._per_frame_token_counts(
            image_grid_thw,
            int(self.config.spatial_merge_size),
        )

        if use_image and int(depth_geometry.shape[1]) != int(image_tokens.shape[1]):
            raise RuntimeError(
                f"depth geometry patches and image token counts are inconsistent: depth={int(depth_geometry.shape[1])}, image={int(image_tokens.shape[1])}"
            )

        if self.config.token_layout == TOKEN_LAYOUT_GLOBAL:
            memory_mask = torch.ones(
                (1, int(depth_geometry.shape[1])),
                device=depth_geometry.device,
                dtype=torch.bool,
            )
            memory = self.depth_proj(self.depth_norm(depth_geometry))
            if use_image:
                memory = memory + self.image_proj(self.image_norm(image_tokens))
            queries = self.query.unsqueeze(0).expand(1, -1, -1)
            attention_geometry = depth_geometry
        elif self.config.token_layout == TOKEN_LAYOUT_PER_FRAME:
            # Key isolation: frame is directly inserted into batch, query i cannot see other frames' depth patch.
            packed_geometry, memory_mask = self._pack_per_frame(
                depth_geometry,
                per_frame_counts,
                context="Depth geometry",
            )
            memory = self.depth_proj(self.depth_norm(packed_geometry))
            if use_image:
                packed_images, image_mask = self._pack_per_frame(
                    image_tokens,
                    per_frame_counts,
                    context="Depth image tokens",
                )
                if not torch.equal(memory_mask, image_mask):
                    raise RuntimeError(
                        "Frame-wise padding masks for depth geometry/image are inconsistent."
                    )
                memory = memory + self.image_proj(self.image_norm(packed_images))
            queries = self.query.unsqueeze(0).expand(len(per_frame_counts), -1, -1)
            attention_geometry = packed_geometry
        else:
            raise RuntimeError(
                f"Unknown Depth token_layout={self.config.token_layout!r}"
            )
        if bool(return_patch_memory):
            # The Stage 2 depth interface requires geometry-only per-frame memory.
            if use_image or self.config.token_layout != TOKEN_LAYOUT_PER_FRAME:
                raise RuntimeError(
                    f"return_patch_memory supports only use_image_tokens=false + per_frame: use_image={use_image}, layout={self.config.token_layout!r}"
                )
        queries = queries.to(device=depth_geometry.device, dtype=module_dtype)
        for block in self.blocks:
            queries = block(queries, memory, attention_geometry, memory_mask)

        output = self.output_proj(self.output_norm(queries))
        if self.config.token_layout == TOKEN_LAYOUT_PER_FRAME:
            # frame-major flattening, ensuring that the i-th Depth token corresponds to ordered image_input[i].
            output = output.reshape(
                1, len(per_frame_counts) * int(self.config.num_depth_tokens), -1
            )
        expected_shape = (
            1,
            self.config.output_token_count(len(per_frame_counts)),
            int(self.config.hidden_size),
        )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Depth Interpreter output shape inconsistency: expected={expected_shape}, actual={tuple(output.shape)}"
            )
        if self.config.output_contract == OUTPUT_CONTRACT_DIRECT_EMBEDDING:
            # The projection in direct_embedding already provides a complete 2048D input; no output gate is created or read here.
            final_output = output
        elif not bool(apply_output_gate):
            # Apply the token gate outside the module for gated_delta outputs.
            final_output = output
        else:
            gate = torch.tanh(self.output_gate).to(
                device=output.device, dtype=output.dtype
            )
            final_output = output * gate
        if not bool(return_patch_memory):
            return final_output

        # patch memory remains outside Qwen; batch-wise padding is aligned with the [B,V,P,d] format of Stage2 Reader after reconstruction.
        return {
            "summary_tokens": final_output,
            "patch_memory": memory.unsqueeze(0),
            "patch_mask": memory_mask.unsqueeze(0),
            "patch_counts": tuple(int(count) for count in per_frame_counts),
        }
