import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CAMERA_TRANSLATION_ENCODING_RAW_METERS = "raw_meters"
CAMERA_TRANSLATION_ENCODING_RAW_METERS_PLANAR = "raw_meters_planar"
CAMERA_TRANSLATION_ENCODING_NORMALIZED = "normalized"
CAMERA_TRANSLATION_ENCODING_NORMALIZED_LOG_SCALE = "normalized_log_scale"
CAMERA_TRANSLATION_ENCODING_ROTATION_ONLY = "rotation_only"
CAMERA_TRANSLATION_ENCODING_CHOICES = (
    CAMERA_TRANSLATION_ENCODING_RAW_METERS,
    CAMERA_TRANSLATION_ENCODING_RAW_METERS_PLANAR,
    CAMERA_TRANSLATION_ENCODING_NORMALIZED,
    CAMERA_TRANSLATION_ENCODING_NORMALIZED_LOG_SCALE,
    CAMERA_TRANSLATION_ENCODING_ROTATION_ONLY,
)
TOKEN_LAYOUT_GLOBAL = "global"
TOKEN_LAYOUT_PER_FRAME = "per_frame"
TOKEN_LAYOUT_CHOICES = (TOKEN_LAYOUT_GLOBAL, TOKEN_LAYOUT_PER_FRAME)
OUTPUT_CONTRACT_GATED_DELTA = "gated_delta"
OUTPUT_CONTRACT_DIRECT_EMBEDDING = "direct_embedding"
OUTPUT_CONTRACT_CHOICES = (
    OUTPUT_CONTRACT_GATED_DELTA,
    OUTPUT_CONTRACT_DIRECT_EMBEDDING,
)
OUTPUT_GATE_TYPE_SCALAR = "scalar"
OUTPUT_GATE_TYPE_NONE = "none"
OUTPUT_GATE_TYPE_CHOICES = (OUTPUT_GATE_TYPE_SCALAR, OUTPUT_GATE_TYPE_NONE)
CAMERA_FRAME_INDEX_RELATIVE = "relative_window"
CAMERA_FRAME_INDEX_FIXED31 = "fixed_31"
CAMERA_FRAME_INDEX_CHOICES = (CAMERA_FRAME_INDEX_RELATIVE, CAMERA_FRAME_INDEX_FIXED31)


def geometry_dim_for_camera_translation_encoding(
    camera_translation_encoding: str,
) -> int:
    """Return the geometry dimension based on the camera translation encoding method."""
    if camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_NORMALIZED_LOG_SCALE:
        return 22
    if camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_RAW_METERS_PLANAR:
        return 20
    if camera_translation_encoding in {
        CAMERA_TRANSLATION_ENCODING_RAW_METERS,
        CAMERA_TRANSLATION_ENCODING_NORMALIZED,
    }:
        return 21
    if camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_ROTATION_ONLY:
        return 18
    raise RuntimeError(
        f"camera_translation_encoding must be {CAMERA_TRANSLATION_ENCODING_CHOICES}current ={camera_translation_encoding!r}"
    )


@dataclass
class CameraInterpreterConfig:
    hidden_size: int = 2048
    num_camera_tokens: int = 8
    # global means the entire Window shares K tokens; per_frame means K tokens are output per frame, and in this round K is fixed to 1.
    token_layout: str = TOKEN_LAYOUT_GLOBAL
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    geometry_dim: int = 21
    # The translation encoding determines whether geometry retains metric scale.
    camera_translation_encoding: str = CAMERA_TRANSLATION_ENCODING_NORMALIZED
    # fixed_31 preserves the source window frame index independently of the selected views.
    camera_frame_index_encoding: str = CAMERA_FRAME_INDEX_RELATIVE
    geometry_rank: int = 32
    # The output contract distinguishes residual deltas from complete embedding vectors.
    output_contract: str = OUTPUT_CONTRACT_GATED_DELTA
    # direct_embedding must be none, so the module will not create the output_gate parameter.
    output_gate_type: str = OUTPUT_GATE_TYPE_SCALAR
    output_gate_init: float = 0.1
    geom_gate_init: float = 0.1
    # false only disables raw geometry additional attention score; the geometry MLP -> memory K/V main path remains unchanged.
    use_geometry_attention_bias: bool = True
    patch_size: int = 14
    # The upstream still refers to spatial_merge_size; only the camera name is saved in the camera configuration.
    camera_merge_size: int = 2
    # image_sizes is passed from the image_sizes field in the NPZ file, no longer using global constants
    use_image_tokens: bool = True
    camera_token_position: str = "before_question"

    def __post_init__(self):
        if self.token_layout not in TOKEN_LAYOUT_CHOICES:
            raise RuntimeError(
                f"token_layout must be {TOKEN_LAYOUT_CHOICES}current ={self.token_layout!r}"
            )
        if int(self.num_camera_tokens) <= 0:
            raise RuntimeError(
                f"num_camera_tokens must be > 0, current={self.num_camera_tokens}"
            )
        if (
            self.token_layout == TOKEN_LAYOUT_PER_FRAME
            and int(self.num_camera_tokens) != 1
        ):
            raise RuntimeError(
                f"Camera per_frame: fix one token per frame this round: num_camera_tokens={self.num_camera_tokens}"
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
                f"Camera output_contract / output_gate_type combination is illegal: contract={self.output_contract!r}, gate_type={self.output_gate_type!r}, expected={expected_gate_type!r}"
            )
        if not isinstance(self.use_geometry_attention_bias, bool):
            raise RuntimeError(
                f"use_geometry_attention_bias must be JSON bool: actual={self.use_geometry_attention_bias!r}"
            )
        _valid_positions = ("before_image", "before_question", "after_question")
        if self.camera_token_position not in _valid_positions:
            raise RuntimeError(
                f"camera_token_position must be {_valid_positions}current ={self.camera_token_position!r}"
            )
        expected_dim = geometry_dim_for_camera_translation_encoding(
            self.camera_translation_encoding
        )
        if self.camera_frame_index_encoding not in CAMERA_FRAME_INDEX_CHOICES:
            raise RuntimeError(
                f"camera_frame_index_encoding illegal: {self.camera_frame_index_encoding!r}"
            )
        if int(self.geometry_dim) != int(expected_dim):
            raise RuntimeError(
                f"geometry_dim and camera_translation_encoding are inconsistent: mode={self.camera_translation_encoding!r}, geometry_dim={self.geometry_dim}, expected={expected_dim}"
            )

    def to_dict(self):
        return asdict(self)

    def output_token_count(self, num_frames: int) -> int:
        """Returns the number of Camera tokens actually output by the current Window based on the layout."""
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise RuntimeError(f"num_frames must be > 0, current={num_frames}")
        if self.token_layout == TOKEN_LAYOUT_GLOBAL:
            return int(self.num_camera_tokens)
        if self.token_layout == TOKEN_LAYOUT_PER_FRAME:
            return num_frames * int(self.num_camera_tokens)
        raise RuntimeError(f"Unknown token_layout={self.token_layout!r}")


def parse_camera_interpreter_config_dict(
    config_dict: dict,
    config_path: str,
    *,
    stage2_spatial_position_is_external: bool = False,
) -> CameraInterpreterConfig:
    """Strictly parse Camera configuration; D1 may require position fields to be fully managed by Stage 2 meta."""
    if not isinstance(config_dict, dict):
        raise RuntimeError(f"Interpreter config must be a JSON object: {config_path}")
    cfg = dict(config_dict)
    required_cfg_keys = set(CameraInterpreterConfig.__dataclass_fields__.keys())
    if stage2_spatial_position_is_external:
        # Stage 2 owns token placement; omit placement fields from its Reader configuration.
        required_cfg_keys.remove("camera_token_position")
    legacy_contract_fields = {"output_contract", "output_gate_type"}
    present_contract_fields = legacy_contract_fields.intersection(cfg)
    if not present_contract_fields:
        # Report missing contract fields and use only the explicit scalar-gated-delta mapping.
        cfg["output_contract"] = OUTPUT_CONTRACT_GATED_DELTA
        cfg["output_gate_type"] = OUTPUT_GATE_TYPE_SCALAR
        print(
            "[camera_config] missing output contract "
            f"-> gated_delta/scalar: {config_path}"
        )
    elif present_contract_fields != legacy_contract_fields:
        raise RuntimeError(
            f"Camera checkpoint's output_contract / output_gate_type must both exist: present={sorted(present_contract_fields)}, path={config_path}"
        )
    actual_cfg_keys = set(cfg.keys())
    missing_cfg_keys = sorted(required_cfg_keys - actual_cfg_keys)
    extra_cfg_keys = sorted(actual_cfg_keys - required_cfg_keys)
    legacy_defaults = {
        "use_geometry_attention_bias": True,
        "token_layout": TOKEN_LAYOUT_GLOBAL,
        "camera_frame_index_encoding": CAMERA_FRAME_INDEX_RELATIVE,
    }
    if (
        missing_cfg_keys
        and set(missing_cfg_keys).issubset(legacy_defaults)
        and not extra_cfg_keys
    ):
        if (
            "token_layout" in missing_cfg_keys
            and int(cfg.get("num_camera_tokens", -1)) != 8
        ):
            # Only global K=8 configurations may omit token_layout; other layouts must be explicit.
            raise RuntimeError(
                f"Camera checkpoint must specify token_layout unless num_camera_tokens=8: num_camera_tokens={cfg.get('num_camera_tokens')}, path={config_path}"
            )
        # Fill only configuration fields whose semantics are unambiguous.
        for key in missing_cfg_keys:
            cfg[key] = legacy_defaults[key]
            print(
                f"[camera_config] missing {key} "
                f"-> {legacy_defaults[key]!r}: {config_path}"
            )
        actual_cfg_keys.update(missing_cfg_keys)
        missing_cfg_keys = []
    if missing_cfg_keys:
        raise RuntimeError(
            f"Interpreter config missing field: missing={missing_cfg_keys}, path={config_path}"
        )
    if extra_cfg_keys:
        raise RuntimeError(
            f"Interpreter config has unknown field: extra={extra_cfg_keys}, path={config_path}"
        )
    return CameraInterpreterConfig(**cfg)


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def get_llm_image_token_counts(
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> List[int]:
    if image_grid_thw.dim() != 2 or int(image_grid_thw.shape[-1]) != 3:
        raise RuntimeError(
            f"image_grid_thw must be [num_images,3], current shape={tuple(image_grid_thw.shape)}"
        )

    counts: List[int] = []
    for thw in image_grid_thw.detach().cpu().tolist():
        grid_t, grid_h, grid_w = [int(x) for x in thw]
        if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
            raise RuntimeError(
                f"image_grid_thw cannot be divided by spatial_merge_size: thw={thw}, spatial_merge_size={spatial_merge_size}"
            )
        counts.append(
            grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)
        )
    return counts


def _squeeze_single_batch_pose(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if extrinsics.dim() == 4:
        if int(extrinsics.shape[0]) != 1:
            raise RuntimeError(
                f"CameraInterpreter v1 only supports batch_size=1, extrinsics shape={tuple(extrinsics.shape)}"
            )
        extrinsics = extrinsics[0]
    if intrinsics.dim() == 4:
        if int(intrinsics.shape[0]) != 1:
            raise RuntimeError(
                f"CameraInterpreter v1 only supports batch_size=1, intrinsics shape={tuple(intrinsics.shape)}"
            )
        intrinsics = intrinsics[0]

    if extrinsics.dim() != 3 or tuple(extrinsics.shape[-2:]) != (3, 4):
        raise RuntimeError(
            f"extrinsics must be [V,3,4] or [1,V,3,4], current shape={tuple(extrinsics.shape)}"
        )
    if intrinsics.dim() != 3 or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise RuntimeError(
            f"intrinsics must be [V,3,3] or [1,V,3,3], current shape={tuple(intrinsics.shape)}"
        )
    if int(extrinsics.shape[0]) != int(intrinsics.shape[0]):
        raise RuntimeError(
            f"extrinsics/intrinsics view count mismatch: extrinsics={tuple(extrinsics.shape)}, intrinsics={tuple(intrinsics.shape)}"
        )
    return extrinsics, intrinsics


def build_prrope_style_geometry(
    image_grid_thw: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    spatial_merge_size: int,
    patch_size: int,
    image_sizes: torch.Tensor,
    camera_translation_encoding: str = CAMERA_TRANSLATION_ENCODING_NORMALIZED,
    camera_frame_index_encoding: str = CAMERA_FRAME_INDEX_RELATIVE,
    frame_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Construct lightweight geometric features for each Qwen LLM image token.

    VGGT's extrinsics, understood from existing CoT-generated code, represent world->camera.
      C = -R^T t
      camera_forward = R^T [0,0,1]

    Here, we do not directly modify Qwen's RoPE, but rather adjust the ray, camera center, and intrinsics.
    Compile attention bias into token geometry compatible with PRoPE-inspired, but not full PRoPE.
    """
    extrinsics, intrinsics = _squeeze_single_batch_pose(extrinsics, intrinsics)
    device = extrinsics.device
    dtype = torch.float32
    extrinsics = extrinsics.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    image_grid_thw = image_grid_thw.to(device=device)

    num_views = int(extrinsics.shape[0])
    if int(image_grid_thw.shape[0]) != num_views:
        raise RuntimeError(
            f"image_grid_thw={tuple(image_grid_thw.shape)}, extrinsics={tuple(extrinsics.shape)}"
        )
    if num_views <= 0:
        raise RuntimeError(f"Invalid view count: num_views={num_views}")
    if camera_frame_index_encoding not in CAMERA_FRAME_INDEX_CHOICES:
        raise RuntimeError(
            f"camera_frame_index_encoding illegal: {camera_frame_index_encoding!r}"
        )
    if frame_indices is None:
        frame_indices = torch.arange(num_views, device=device, dtype=torch.long)
    else:
        frame_indices = torch.as_tensor(
            frame_indices, device=device, dtype=torch.long
        ).reshape(-1)
        if int(frame_indices.numel()) != num_views or int(frame_indices[0]) != 0:
            raise RuntimeError(
                f"Camera original window frame_indices must be of length V and start with 0: {frame_indices.tolist()}"
            )
    if camera_frame_index_encoding == CAMERA_FRAME_INDEX_FIXED31 and (
        bool((frame_indices < 0).any().item())
        or bool((frame_indices > 31).any().item())
    ):
        raise RuntimeError(
            f"Camera frame_idx/31 supports 0..31 only: {frame_indices.tolist()}"
        )

    rotations = extrinsics[..., :3]
    translations = extrinsics[..., 3]
    centers = -(rotations.transpose(-1, -2) @ translations.unsqueeze(-1)).squeeze(-1)

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    forward = _safe_normalize(
        rotations.transpose(-1, -2) @ z_axis.view(1, 3, 1), dim=1
    ).squeeze(-1)
    camera_right = _safe_normalize(
        rotations.transpose(-1, -2) @ x_axis.view(1, 3, 1), dim=1
    ).squeeze(-1)

    ref_forward = forward[0]
    # Use image1-relative axes so pose features agree with description coordinates.
    ref_right = camera_right[0]
    ref_right = _safe_normalize(ref_right, dim=0)
    ref_up = _safe_normalize(torch.cross(ref_right, ref_forward, dim=0), dim=0)
    ref_basis = torch.stack([ref_right, ref_up, ref_forward], dim=0)

    rel_centers = centers - centers[:1]
    center_scale = rel_centers.norm(dim=-1).amax().clamp_min(1e-6)
    # Pred Pose has scale meaning only in the Sim(3) sense; OmniVGGT's segmented concatenation does not add metric anchors to the first segment.
    # normalized will perform a single global normalization after all segments have been assembled into the complete input trajectory; it does not read chunk boundaries,
    # Therefore, preserve the direction and shape of the entire trajectory while removing any global scale across different scenes.
    # CoT text supervision is still determined by the data generation pipeline; here, only the Stage1 geometry input is changed, without prematurely binding the target format.
    if camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_RAW_METERS:
        # Metric displacement requires a reliable metric anchor.
        encoded_centers = rel_centers
        log_center_scale = None
    elif camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_RAW_METERS_PLANAR:
        # The new GT mainline only retains the right/forward metric components from image 1 coordinates and does not pass the vertical component to the Interpreter.
        encoded_centers = rel_centers
        log_center_scale = None
    elif camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_NORMALIZED:
        # Pred Camera Mainline: Normalize the entire trajectory after stitching, preserving global orientation and shape, without exposing any scale of OmniVGGT.
        encoded_centers = rel_centers / center_scale
        log_center_scale = None
    elif (
        camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_NORMALIZED_LOG_SCALE
    ):
        # Log-scale translation encoding requires a reliable scale estimate.
        encoded_centers = rel_centers / center_scale
        log_center_scale = torch.log(center_scale)
    elif camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_ROTATION_ONLY:
        # D1 mainline abandons explicit translation because global normalization cannot fix scale drift between OmniVGGT segments.
        encoded_centers = torch.zeros_like(rel_centers)
        log_center_scale = None
    else:
        raise RuntimeError(
            f"camera_translation_encoding must be {CAMERA_TRANSLATION_ENCODING_CHOICES}current ={camera_translation_encoding!r}"
        )

    centers_ref = encoded_centers @ ref_basis.transpose(0, 1)
    forward_ref = forward @ ref_basis.transpose(0, 1)
    expected_geometry_dim = geometry_dim_for_camera_translation_encoding(
        camera_translation_encoding
    )

    geom_chunks = []
    for view_idx, thw in enumerate(image_grid_thw.detach().cpu().tolist()):
        grid_t, grid_h, grid_w = [int(x) for x in thw]
        llm_h = grid_h // int(spatial_merge_size)
        llm_w = grid_w // int(spatial_merge_size)
        if grid_t <= 0 or llm_h <= 0 or llm_w <= 0:
            raise RuntimeError(f"Invalid image_grid_thw: {thw}")

        tt, yy, xx = torch.meshgrid(
            torch.arange(grid_t, device=device, dtype=dtype),
            torch.arange(llm_h, device=device, dtype=dtype),
            torch.arange(llm_w, device=device, dtype=dtype),
            indexing="ij",
        )
        flat_t = tt.reshape(-1)
        flat_y = yy.reshape(-1)
        flat_x = xx.reshape(-1)

        u_norm = (flat_x + 0.5) / float(llm_w)
        v_norm = (flat_y + 0.5) / float(llm_h)
        t_norm = (flat_t + 0.5) / float(grid_t)

        processed_w = float(grid_w * patch_size)
        processed_h = float(grid_h * patch_size)
        pixel_u = (flat_x + 0.5) * float(spatial_merge_size * patch_size)
        pixel_v = (flat_y + 0.5) * float(spatial_merge_size * patch_size)

        K = intrinsics[view_idx]
        img_w = float(image_sizes[view_idx, 0])
        img_h = float(image_sizes[view_idx, 1])
        if img_w <= 0.0 or img_h <= 0.0:
            raise RuntimeError(
                f"image_sizes[{view_idx}must be > 0, current width={img_w}, height={img_h}"
            )
        scale_x = processed_w / img_w
        scale_y = processed_h / img_h
        fx = (K[0, 0] * scale_x).abs().clamp_min(1e-6)
        fy = (K[1, 1] * scale_y).abs().clamp_min(1e-6)
        cx = K[0, 2] * scale_x
        cy = K[1, 2] * scale_y

        ray_cam = torch.stack(
            [
                (pixel_u - cx) / fx,
                (pixel_v - cy) / fy,
                torch.ones_like(pixel_u),
            ],
            dim=-1,
        )
        ray_cam = _safe_normalize(ray_cam, dim=-1)
        ray_world = _safe_normalize(ray_cam @ rotations[view_idx], dim=-1)
        ray_ref = ray_world @ ref_basis.transpose(0, 1)

        intr_norm = (
            torch.stack(
                [
                    fx / processed_w,
                    fy / processed_h,
                    cx / processed_w,
                    cy / processed_h,
                ],
                dim=0,
            )
            .view(1, 4)
            .expand(ray_world.shape[0], -1)
        )

        # Normalize source frame indices consistently across pair training and multi-view QA.
        view_value = (
            float(frame_indices[view_idx]) / 31.0
            if camera_frame_index_encoding == CAMERA_FRAME_INDEX_FIXED31
            else float(view_idx) / float(max(1, num_views - 1))
        )
        view_scalar = torch.full(
            (ray_world.shape[0], 1),
            view_value,
            device=device,
            dtype=dtype,
        )
        anchor_flag = torch.full(
            (ray_world.shape[0], 1),
            1.0 if view_idx == 0 else 0.0,
            device=device,
            dtype=dtype,
        )

        geom_parts = [
            torch.stack([u_norm, v_norm], dim=-1),
            ray_world,
            ray_ref,
        ]
        if camera_translation_encoding == CAMERA_TRANSLATION_ENCODING_RAW_METERS_PLANAR:
            geom_parts.append(
                centers_ref[view_idx, [0, 2]].view(1, 2).expand(ray_world.shape[0], -1)
            )
        elif camera_translation_encoding != CAMERA_TRANSLATION_ENCODING_ROTATION_ONLY:
            geom_parts.append(
                centers_ref[view_idx].view(1, 3).expand(ray_world.shape[0], -1)
            )
        # rotation_only removes only the relative center; the forward direction explicitly originates from rotation and must be retained.
        geom_parts.extend(
            [
                forward_ref[view_idx].view(1, 3).expand(ray_world.shape[0], -1),
                intr_norm,
                view_scalar,
                anchor_flag,
                t_norm.view(-1, 1),
            ]
        )
        if log_center_scale is not None:
            geom_parts.append(log_center_scale.view(1, 1).expand(ray_world.shape[0], 1))
        geom = torch.cat(geom_parts, dim=-1)
        geom_chunks.append(geom)

    geometry = torch.cat(geom_chunks, dim=0)
    if int(geometry.shape[-1]) != int(expected_geometry_dim):
        raise RuntimeError(
            f"Inconsistent internal geometry_dim and camera_translation_encoding: mode={camera_translation_encoding}, geometry_dim={int(geometry.shape[-1])}, expected={expected_geometry_dim}"
        )
    return geometry


def build_pose_only_image_grid_thw(
    extrinsics: torch.Tensor,
    *,
    spatial_merge_size: int,
    patch_size: int,
    image_width: int,
    image_height: int,
    min_pixels: int,
    max_pixels: int,
) -> torch.Tensor:
    """Construct a geometry grid compatible with pose-only mode and aligned with the Qwen image processor."""
    if extrinsics.dim() == 4:
        if int(extrinsics.shape[0]) != 1:
            raise RuntimeError(
                f"CameraInterpreter v1 only supports batch_size=1, extrinsics shape={tuple(extrinsics.shape)}"
            )
        num_views = int(extrinsics.shape[1])
    elif extrinsics.dim() == 3:
        num_views = int(extrinsics.shape[0])
    else:
        raise RuntimeError(
            f"extrinsics must be [V,3,4] or [1,V,3,4], current shape={tuple(extrinsics.shape)}"
        )
    if num_views <= 0:
        raise RuntimeError(f"Invalid view count: num_views={num_views}")

    merge = int(spatial_merge_size)
    patch = int(patch_size)
    width = int(image_width)
    height = int(image_height)
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    if merge <= 0:
        raise RuntimeError(f"spatial_merge_size must be > 0, current={merge}")
    if patch <= 0:
        raise RuntimeError(f"patch_size must be > 0, current={patch}")
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"pose-only image_width/image_height must > 0, current={width}x{height}"
        )
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise RuntimeError(
            f"pose-only min/max pixels illegal: min_pixels={min_pixels}, max_pixels={max_pixels}"
        )

    # This is the smart_resize rule replicated from the Qwen image processor.
    # pose-only does not have a real image_grid_thw, so it must be derived using the same rules from the original image dimensions;
    # Cannot regenerate [1, merge, merge], otherwise each view would only have 1 center ray token, leading to unfair ablation.
    factor = patch * merge
    if height < factor or width < factor:
        raise RuntimeError(
            f"pose-only image size is too small to apply factor={factor} Construct grid: PIL(W,H)=(){width},{height})"
        )
    if max(height, width) / min(height, width) > 200:
        raise RuntimeError(
            f"pose-only image aspect ratio invalid: PIL(W,H)={width},{height})"
        )

    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = math.floor(height / beta / factor) * factor
        resized_w = math.floor(width / beta / factor) * factor
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor

    grid_h = int(resized_h // patch)
    grid_w = int(resized_w // patch)
    if grid_h <= 0 or grid_w <= 0:
        raise RuntimeError(
            f"pose-only grid invalid: resized_h={resized_h}, resized_w={resized_w}, patch={patch}"
        )
    if grid_h % merge != 0 or grid_w % merge != 0:
        raise RuntimeError(
            f"pose-only grid cannot be divided by spatial_merge_size: grid_h={grid_h}, grid_w={grid_w}, merge={merge}"
        )
    return torch.tensor(
        [[1, grid_h, grid_w] for _ in range(num_views)],
        device=extrinsics.device,
        dtype=torch.long,
    )


class GeometryCrossAttentionBlock(nn.Module):
    def __init__(self, config: CameraInterpreterConfig):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads: hidden={config.hidden_size}, heads={config.num_heads}"
            )
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.geometry_rank = int(config.geometry_rank)
        self.num_camera_tokens = int(config.num_camera_tokens)
        self.use_geometry_attention_bias = bool(config.use_geometry_attention_bias)

        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.memory_norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.geom_key = nn.Linear(
            config.geometry_dim, self.num_heads * self.geometry_rank
        )
        self.geom_query = nn.Parameter(
            torch.empty(self.num_camera_tokens, self.num_heads, self.geometry_rank)
        )
        self.geom_gate = nn.Parameter(
            torch.tensor(float(config.geom_gate_init), dtype=torch.float32)
        )
        if not self.use_geometry_attention_bias:
            # When bias is disabled, these parameters are not involved in forward; freezing them prevents DDP find_unused_parameters=False errors.
            for param in self.geom_key.parameters():
                param.requires_grad = False
            self.geom_query.requires_grad = False
            self.geom_gate.requires_grad = False

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
        if query_len != self.num_camera_tokens:
            raise RuntimeError(
                f"query count and num_camera_tokens are inconsistent: query_len={query_len}, expected={self.num_camera_tokens}"
            )
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
            # Only increase the raw geometry score when the switch is on; after turning it off, continue using memory K/V generated by the geometry MLP.
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

        if memory_mask.shape != (batch_size, memory_len):
            raise RuntimeError(
                f"memory_mask shape inconsistency: mask={tuple(memory_mask.shape)}, expected={(batch_size, memory_len)}"
            )
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


class CameraInterpreter(nn.Module):
    """
    image tokens + camera geometry -> K camera tokens output.
    If config.use_image_tokens=False, only use the VGGT camera geometry within the same grid, without reading image token content.

    This module only implements PRoPE-inspired geometric attention internally within Interpreter, without modifying the Qwen-VL backbone:

        Qwen image tokens + per-patch ray/camera geometry
                         |
                  K learned queries
                         |
             geometry-biased cross attention
                         |
                  K camera token output
    """

    def __init__(self, config: CameraInterpreterConfig):
        super().__init__()
        self.config = config
        if config.hidden_size <= 0 or config.num_camera_tokens <= 0:
            raise ValueError("hidden_size and num_camera_tokens must be > 0")

        self.image_norm = nn.LayerNorm(config.hidden_size)
        self.image_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.geometry_norm = nn.LayerNorm(config.geometry_dim)
        self.geometry_proj = nn.Sequential(
            nn.Linear(config.geometry_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.query = nn.Parameter(
            torch.empty(config.num_camera_tokens, config.hidden_size)
        )
        self.blocks = nn.ModuleList(
            [GeometryCrossAttentionBlock(config) for _ in range(config.num_layers)]
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
        if not token_counts or any(int(count) <= 0 for count in token_counts):
            raise RuntimeError(
                f"{context} Invalid per-frame token count: {token_counts}"
            )
        total = sum(int(count) for count in token_counts)
        if total != int(values.shape[1]):
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
        image_grid_thw: Optional[torch.Tensor],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        image_sizes: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        apply_output_gate: bool = True,
    ) -> torch.Tensor:
        module_dtype = next(self.parameters()).dtype
        use_image_tokens = bool(self.config.use_image_tokens)
        if use_image_tokens:
            if image_tokens is None:
                raise RuntimeError(
                    "When use_image_tokens=True, image_tokens must be provided."
                )
            if image_grid_thw is None:
                raise RuntimeError(
                    "When use_image_tokens=True, image_grid_thw must be provided."
                )
            if image_tokens.dim() == 2:
                image_tokens = image_tokens.unsqueeze(0)
            if image_tokens.dim() != 3:
                raise RuntimeError(
                    f"image_tokens must be [N,H] or [1,N,H], current shape={tuple(image_tokens.shape)}"
                )
            if int(image_tokens.shape[0]) != 1:
                raise RuntimeError(
                    f"CameraInterpreter v1 only supports batch_size=1, image_tokens shape={tuple(image_tokens.shape)}"
                )
            if int(image_tokens.shape[-1]) != int(self.config.hidden_size):
                raise RuntimeError(
                    f"image token hidden size inconsistency: tokens={int(image_tokens.shape[-1])}, config={self.config.hidden_size}"
                )
            image_tokens = image_tokens.to(dtype=module_dtype)
            geometry_device = image_tokens.device
        else:
            if image_tokens is not None:
                raise RuntimeError(
                    "When use_image_tokens=False, image_tokens should not be passed."
                )
            if image_grid_thw is None:
                raise RuntimeError(
                    "image_grid_thw must also be provided when use_image_tokens=False, to construct the per-patch geometry grid aligned with the image input."
                )
            geometry_device = extrinsics.device

        geometry = build_prrope_style_geometry(
            image_grid_thw=image_grid_thw,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            spatial_merge_size=int(self.config.camera_merge_size),
            patch_size=int(self.config.patch_size),
            image_sizes=image_sizes,
            camera_translation_encoding=str(self.config.camera_translation_encoding),
            camera_frame_index_encoding=str(self.config.camera_frame_index_encoding),
            frame_indices=frame_indices,
        ).to(device=geometry_device, dtype=module_dtype)
        if int(geometry.shape[-1]) != int(self.config.geometry_dim):
            raise RuntimeError(
                f"geometry last dimension does not match config.geometry_dim: geometry={int(geometry.shape[-1])}, config={int(self.config.geometry_dim)}, mode={self.config.camera_translation_encoding}"
            )
        geometry = geometry.unsqueeze(0)
        per_frame_counts = get_llm_image_token_counts(
            image_grid_thw,
            spatial_merge_size=int(self.config.camera_merge_size),
        )

        if use_image_tokens and int(geometry.shape[1]) != int(image_tokens.shape[1]):
            raise RuntimeError(
                f"geometry token count and image token count are inconsistent: geometry={int(geometry.shape[1])}, image={int(image_tokens.shape[1])}, per_image_counts={per_frame_counts}"
            )

        if self.config.token_layout == TOKEN_LAYOUT_GLOBAL:
            memory_mask = torch.ones(
                (1, int(geometry.shape[1])),
                device=geometry.device,
                dtype=torch.bool,
            )
            memory = self.geometry_proj(self.geometry_norm(geometry))
            if use_image_tokens:
                memory = memory + self.image_proj(self.image_norm(image_tokens))
            queries = self.query.unsqueeze(0).expand(1, -1, -1)
            attention_geometry = geometry
        elif self.config.token_layout == TOKEN_LAYOUT_PER_FRAME:
            # Key Isolation: Fold frame into batch dimension, each batch item contains only its own frame's patch;
            # No need for block-diagonal attention mask, and impossible to read across frames.
            packed_geometry, memory_mask = self._pack_per_frame(
                geometry,
                per_frame_counts,
                context="Camera geometry",
            )
            memory = self.geometry_proj(self.geometry_norm(packed_geometry))
            if use_image_tokens:
                packed_images, image_mask = self._pack_per_frame(
                    image_tokens,
                    per_frame_counts,
                    context="Camera image tokens",
                )
                if not torch.equal(memory_mask, image_mask):
                    raise RuntimeError(
                        "Frame-wise padding mask inconsistency in camera geometry/image."
                    )
                memory = memory + self.image_proj(self.image_norm(packed_images))
            queries = self.query.unsqueeze(0).expand(len(per_frame_counts), -1, -1)
            attention_geometry = packed_geometry
        else:
            raise RuntimeError(
                f"Unknown Camera token_layout={self.config.token_layout!r}"
            )
        queries = queries.to(device=geometry.device, dtype=module_dtype)
        for block in self.blocks:
            queries = block(queries, memory, attention_geometry, memory_mask)

        output = self.output_proj(self.output_norm(queries))
        if self.config.token_layout == TOKEN_LAYOUT_PER_FRAME:
            # [V,K,H] -> [1,V*K,H] preserves frame-major order; in this round K=1, so the i-th output corresponds to the i-th frame.
            output = output.reshape(
                1, len(per_frame_counts) * int(self.config.num_camera_tokens), -1
            )
        expected_shape = (
            1,
            self.config.output_token_count(len(per_frame_counts)),
            int(self.config.hidden_size),
        )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"Camera Interpreter output shape inconsistency: expected={expected_shape}, actual={tuple(output.shape)}"
            )
        if self.config.output_contract == OUTPUT_CONTRACT_DIRECT_EMBEDDING:
            # The projection in direct_embedding already provides a complete 2048D input; no output gate is created or read here.
            return output
        # Apply the token gate outside the module for the gated_delta contract.
        if not bool(apply_output_gate):
            return output
        gate = torch.tanh(self.output_gate).to(device=output.device, dtype=output.dtype)
        return output * gate
