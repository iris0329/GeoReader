"""Depth Stage1's patch-wise complete token generator."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


DEPTH_PATCH_OUTPUT_CONTRACT = "direct_patch_embedding"
DEPTH_PATCH_TOKEN_POSITION = "before_question"
DEPTH_PATCH_POSITION_IDS_CONTRACT = "copy_rgb_mrope"
DEPTH_PATCH_TOKENIZER_TYPE = "mlp"


@dataclass
class DepthPatchTokenizerConfig:
    """Save all reconstruction contracts for Direct Patch Token Stage1."""

    hidden_size: int = 2048
    depth_geometry_dim: int = 5
    depth_encoder_type: str = "scalar_5d"
    token_layout: str = "per_patch"
    output_contract: str = DEPTH_PATCH_OUTPUT_CONTRACT
    token_position: str = DEPTH_PATCH_TOKEN_POSITION
    position_ids_contract: str = DEPTH_PATCH_POSITION_IDS_CONTRACT
    tokenizer_type: str = DEPTH_PATCH_TOKENIZER_TYPE
    num_linear_layers: int = 3
    num_hidden_linear_layers: int = 2
    output_norm: bool = True
    output_projection: bool = True
    output_gate_type: str = "none"
    base_add: bool = False
    use_image_tokens: bool = False
    patch_size: int = 14
    spatial_merge_size: int = 2

    def __post_init__(self):
        """Validate Direct Patch settings against the patch-level input contract."""
        expected = {
            "depth_geometry_dim": 5,
            "depth_encoder_type": "scalar_5d",
            "token_layout": "per_patch",
            "output_contract": DEPTH_PATCH_OUTPUT_CONTRACT,
            "token_position": DEPTH_PATCH_TOKEN_POSITION,
            "position_ids_contract": DEPTH_PATCH_POSITION_IDS_CONTRACT,
            "tokenizer_type": DEPTH_PATCH_TOKENIZER_TYPE,
            "num_linear_layers": 3,
            "num_hidden_linear_layers": 2,
            "output_norm": True,
            "output_projection": True,
            "output_gate_type": "none",
            "base_add": False,
            "use_image_tokens": False,
        }
        actual = self.to_dict()
        mismatches = {
            key: (actual[key], value)
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise RuntimeError(
                f"DepthPatchTokenizerConfig does not conform to direct_patch_embedding contract: mismatches={mismatches}"
            )
        if (
            int(self.hidden_size) <= 0
            or int(self.patch_size) <= 0
            or int(self.spatial_merge_size) <= 0
        ):
            raise RuntimeError(
                f"DepthPatchTokenizerConfig hidden_size/patch_size/spatial_merge_size must be positive numbers: config={self.to_dict()}"
            )

    def to_dict(self):
        """Return a strict set of fields suitable for direct writing to a checkpoint JSON."""
        return asdict(self)


def parse_depth_patch_tokenizer_config_dict(
    config_dict: dict, config_path: str
) -> DepthPatchTokenizerConfig:
    """Strictly parse Direct Patch Tokenizer checkpoint configuration."""
    if not isinstance(config_dict, dict):
        raise RuntimeError(
            f"DepthPatchTokenizerConfig must be dict: path={config_path}, type={type(config_dict)}"
        )
    expected_keys = set(DepthPatchTokenizerConfig.__dataclass_fields__)
    actual_keys = set(config_dict)
    extra = sorted(actual_keys - expected_keys)
    missing = sorted(expected_keys - actual_keys)
    if extra or missing:
        raise RuntimeError(
            f"DepthPatchTokenizerConfig fields must strictly match: extra={extra}, missing={missing}, path={config_path}"
        )
    return DepthPatchTokenizerConfig(**dict(config_dict))


class DepthPatchTokenizer(nn.Module):
    """Map each 5D Depth geometry directly into a full Qwen embedding."""

    def __init__(self, config: DepthPatchTokenizerConfig):
        super().__init__()
        self.config = config
        d = int(config.hidden_size)
        # Here reuses PyTorch Linear's default initialization, consistent with the existing direct_embedding linear layer in the repository;
        # Do not use zero-init to avoid initializing all patch tokens identically and cutting off gradients in the first two layers.
        self.input_norm = nn.LayerNorm(int(config.depth_geometry_dim))
        self.hidden_in = nn.Linear(int(config.depth_geometry_dim), d)
        self.activation = nn.GELU()
        self.hidden_out = nn.Linear(d, d)
        self.output_norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, d)

    def forward(self, depth_geometry: torch.Tensor) -> torch.Tensor:
        """Convert [B,N,5] or [N,5] per-patch geometry into the same number of full embeddings."""
        if depth_geometry.dim() not in (2, 3):
            raise RuntimeError(
                f"DepthPatchTokenizer input must be [N,5] or [B,N,5]: shape={tuple(depth_geometry.shape)}"
            )
        if int(depth_geometry.shape[-1]) != int(self.config.depth_geometry_dim):
            raise RuntimeError(
                f"DepthPatchTokenizer geometry dim inconsistency: actual={tuple(depth_geometry.shape)}, expected_last={self.config.depth_geometry_dim}"
            )
        output = self.input_norm(depth_geometry)
        output = self.hidden_in(output)
        output = self.activation(output)
        output = self.hidden_out(output)
        output = self.output_norm(output)
        output = self.output_proj(output)
        if tuple(output.shape[:-1]) != tuple(depth_geometry.shape[:-1]) or int(
            output.shape[-1]
        ) != int(self.config.hidden_size):
            raise RuntimeError(
                f"DepthPatchTokenizer output shape does not conform to patch-by-patch contract: input={tuple(depth_geometry.shape)}, output={tuple(output.shape)}"
            )
        return output
