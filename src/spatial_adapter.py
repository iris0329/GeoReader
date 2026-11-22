"""Convert Interpreter raw delta into Dense, Identity, or Residual RMS candidate."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


SPATIAL_ADAPTER_TYPE_DENSE = "dense"
SPATIAL_ADAPTER_TYPE_IDENTITY = "identity"
# Identity-init Dense only serves linear remapping for complete Direct embedding QA; step 0 must strictly equal input.
SPATIAL_ADAPTER_TYPE_IDENTITY_INIT_DENSE = "identity_init_dense"
SPATIAL_ADAPTER_TYPE_RMS_ALIGNED_IDENTITY = "rms_aligned_identity"
SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS = "residual_rms"
SPATIAL_ADAPTER_TYPE_CHOICES = (
    SPATIAL_ADAPTER_TYPE_DENSE,
    SPATIAL_ADAPTER_TYPE_IDENTITY,
    SPATIAL_ADAPTER_TYPE_IDENTITY_INIT_DENSE,
    SPATIAL_ADAPTER_TYPE_RMS_ALIGNED_IDENTITY,
    SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS,
)
SPATIAL_ADAPTER_PARAMETERLESS_TYPES = (
    SPATIAL_ADAPTER_TYPE_IDENTITY,
    SPATIAL_ADAPTER_TYPE_RMS_ALIGNED_IDENTITY,
)
SPATIAL_ADAPTER_RMS_TYPES = (
    SPATIAL_ADAPTER_TYPE_RMS_ALIGNED_IDENTITY,
    SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS,
)
SPATIAL_ADAPTER_PARAMETERIZED_TYPES = (
    SPATIAL_ADAPTER_TYPE_DENSE,
    SPATIAL_ADAPTER_TYPE_IDENTITY_INIT_DENSE,
    SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS,
)


@dataclass(frozen=True)
class SpatialAdapterConfig:
    """Save the minimal structural contract for Spatial Adapter."""

    hidden_size: int
    # adapter_type selects the candidate transformation; dense is the default.
    adapter_type: str = SPATIAL_ADAPTER_TYPE_DENSE
    # bottleneck only controls the width of the Residual RMS correction MLP; the larger the value, the higher the capacity and parameter count.
    bottleneck: int | None = None

    def __post_init__(self) -> None:
        if int(self.hidden_size) <= 0:
            raise RuntimeError(f"hidden_size must be > 0, current={self.hidden_size}")
        if self.adapter_type not in SPATIAL_ADAPTER_TYPE_CHOICES:
            raise RuntimeError(
                f"adapter_type illegal: {self.adapter_type!r}, choices={SPATIAL_ADAPTER_TYPE_CHOICES}"
            )
        if self.adapter_type == SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS:
            if self.bottleneck is None or int(self.bottleneck) <= 0:
                raise RuntimeError(
                    f"Residual RMS bottleneck must be a positive integer, current={self.bottleneck}"
                )
        elif self.bottleneck is not None:
            raise RuntimeError(
                f"Only residual_rms allows setting bottleneck; current adapter_type={self.adapter_type!r}"
            )

    def to_dict(self) -> dict:
        """Return a JSON-compatible Adapter configuration."""
        # Store optional adapter fields only when they apply to the selected adapter type.
        return {key: value for key, value in asdict(self).items() if value is not None}


def parse_spatial_adapter_config_dict(
    config_dict: dict, config_path: str
) -> SpatialAdapterConfig:
    """Strictly parse the Spatial Adapter configuration within the checkpoint."""
    if not isinstance(config_dict, dict):
        raise RuntimeError(
            f"Spatial Adapter config must be a JSON object: {config_path}"
        )
    required = {"hidden_size"}
    allowed = set(SpatialAdapterConfig.__dataclass_fields__)
    actual = set(config_dict)
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    if missing or extra:
        raise RuntimeError(
            f"Spatial Adapter config field incomplete: missing={missing}, extra={extra}, path={config_path}"
        )
    parsed = dict(config_dict)
    if "adapter_type" not in parsed:
        # A configuration containing only hidden_size selects the dense adapter.
        parsed["adapter_type"] = SPATIAL_ADAPTER_TYPE_DENSE
    return SpatialAdapterConfig(**parsed)


class SpatialAdapter(nn.Module):
    """Generate Dense, Identity, or Residual RMS candidates with the same shape as specified in the configuration."""

    def __init__(self, config: SpatialAdapterConfig):
        super().__init__()
        self.config = config
        self.proj = None
        self.correction_norm = None
        self.correction_input = None
        self.correction_activation = None
        self.correction_output = None
        if config.adapter_type == SPATIAL_ADAPTER_TYPE_DENSE:
            self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            # Dense step 0 must start from strictly zero perturbation; missing zero initialization for either weight or bias violates the contract.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif config.adapter_type == SPATIAL_ADAPTER_TYPE_IDENTITY_INIT_DENSE:
            self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            # Two passes of Direct require that the Linear step 0 does not alter Stage1 embedding; differences after training arise solely from QA adaptation.
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif config.adapter_type == SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS:
            bottleneck = int(config.bottleneck)
            self.correction_norm = nn.LayerNorm(config.hidden_size)
            self.correction_input = nn.Linear(config.hidden_size, bottleneck, bias=True)
            self.correction_activation = nn.GELU()
            self.correction_output = nn.Linear(
                bottleneck, config.hidden_size, bias=True
            )
            # Residual RMS must degenerate into pure RMS Identity at the start; only clear the last layer, the raw identity branch is always retained.
            nn.init.zeros_(self.correction_output.weight)
            nn.init.zeros_(self.correction_output.bias)

    def compute_correction(self, raw_delta: torch.Tensor) -> torch.Tensor:
        """Calculate MLP correction amount for Residual RMS; report error for other Adapter types."""
        if self.config.adapter_type != SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS:
            raise RuntimeError(
                f"Only residual_rms can compute correction, current={self.config.adapter_type!r}"
            )
        modules = (
            self.correction_norm,
            self.correction_input,
            self.correction_activation,
            self.correction_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("Incomplete residual RMS correction MLP structure.")
        normalized = self.correction_norm(
            raw_delta.to(dtype=self.correction_norm.weight.dtype)
        )
        hidden = self.correction_activation(self.correction_input(normalized))
        return self.correction_output(hidden)

    def _align_to_reference_rms(
        self,
        value: torch.Tensor,
        reference_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        """Align the K and H dimensional RMS using a common scalar per sample to the spatial slot embedding."""
        if reference_embeddings is None:
            raise RuntimeError(
                f"{self.config.adapter_type} Spatial slot reference embeddings must be provided."
            )
        if tuple(reference_embeddings.shape) != tuple(value.shape):
            raise RuntimeError(
                f"{self.config.adapter_type} The reference shape must match the tensor to be aligned: value={tuple(value.shape)}, reference={tuple(reference_embeddings.shape)}"
            )
        if not bool(torch.isfinite(reference_embeddings).all().item()):
            raise RuntimeError(
                f"{self.config.adapter_type} The reference embeddings contain non-finite values."
            )
        # Camera/Depth each computes a scalar per sample, and cannot modify direction token by token or channel by channel.
        value_rms = torch.sqrt(
            torch.mean(value.float().square(), dim=(1, 2), keepdim=True)
        )
        reference_rms = torch.sqrt(
            torch.mean(
                reference_embeddings.detach().float().square(), dim=(1, 2), keepdim=True
            )
        )
        if bool(torch.any(value_rms <= 0).item()):
            source_name = (
                "raw delta"
                if self.config.adapter_type == SPATIAL_ADAPTER_TYPE_RMS_ALIGNED_IDENTITY
                else "corrected delta"
            )
            raise RuntimeError(
                f"{self.config.adapter_type} valid {source_name} RMS must be > 0."
            )
        if bool(torch.any(reference_rms <= 0).item()):
            raise RuntimeError(
                f"{self.config.adapter_type} The reference embedding RMS must be > 0."
            )
        return (value.float() * (reference_rms / value_rms)).to(dtype=value.dtype)

    def forward(
        self,
        raw_delta: torch.Tensor,
        reference_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert raw delta to candidate with the same shape; read spatial slot embeddings in both RMS modes."""
        hidden_size = int(self.config.hidden_size)
        if raw_delta.dim() != 3 or int(raw_delta.shape[-1]) != hidden_size:
            raise RuntimeError(
                f"raw_delta must be [B,K,{hidden_size}], current={tuple(raw_delta.shape)}"
            )
        if not bool(torch.isfinite(raw_delta).all().item()):
            raise RuntimeError("Spatial Adapter input contains non-finite values.")
        if self.config.adapter_type == SPATIAL_ADAPTER_TYPE_IDENTITY:
            # Identity registers no parameters and does not perform scale or normalization; state_dict must remain empty.
            candidate = raw_delta
        elif self.config.adapter_type == SPATIAL_ADAPTER_TYPE_RMS_ALIGNED_IDENTITY:
            candidate = self._align_to_reference_rms(raw_delta, reference_embeddings)
        elif self.config.adapter_type == SPATIAL_ADAPTER_TYPE_RESIDUAL_RMS:
            correction = self.compute_correction(raw_delta)
            # raw main path does not go through MLP; MLP only learns correction, and finally applies RMS volume control uniformly.
            corrected_delta = raw_delta.to(dtype=correction.dtype) + correction
            candidate = self._align_to_reference_rms(
                corrected_delta, reference_embeddings
            )
        else:
            if self.proj is None:
                raise RuntimeError("Dense Spatial Adapter is missing proj.")
            candidate = self.proj(raw_delta.to(dtype=self.proj.weight.dtype))
        if not bool(torch.isfinite(candidate).all().item()):
            raise RuntimeError("Spatial Adapter outputs contain non-finite values.")
        return candidate
