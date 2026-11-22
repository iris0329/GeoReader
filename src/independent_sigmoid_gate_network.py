"""Use H36 question hidden to independently control the intensity of Camera and Depth residual."""

from dataclasses import asdict, dataclass
from typing import Tuple

import torch
import torch.nn as nn


INDEPENDENT_SIGMOID_BRANCHES = ("camera", "depth")
INDEPENDENT_SIGMOID_AVAILABILITY = ("has_pose", "has_depth")


@dataclass(frozen=True)
class IndependentSigmoidGateConfig:
    """Save strict checkpoint structure contract for dual-path independent Sigmoid Gate."""

    hidden_size: int
    # bottleneck controls the gate small network capacity; P0 is fixed at 256 to avoid making capacity a second variable.
    bottleneck: int = 256
    branches: Tuple[str, ...] = INDEPENDENT_SIGMOID_BRANCHES
    availability: Tuple[str, ...] = INDEPENDENT_SIGMOID_AVAILABILITY

    def __post_init__(self) -> None:
        if int(self.hidden_size) <= 0:
            raise RuntimeError(f"hidden_size must be > 0, current={self.hidden_size}")
        if int(self.bottleneck) <= 0:
            raise RuntimeError(f"bottleneck must be > 0, current={self.bottleneck}")
        # MindCube uses the original single-branch implementation; the six-benchmark recipe retains both output heads without adding untrained branches.
        supported = {
            ("camera",): ("has_pose",),
            INDEPENDENT_SIGMOID_BRANCHES: INDEPENDENT_SIGMOID_AVAILABILITY,
        }
        if (
            tuple(self.branches) not in supported
            or tuple(self.availability) != supported[tuple(self.branches)]
        ):
            raise RuntimeError(
                f"Unsupported Gate branch/availability: {self.branches}/{self.availability}"
            )

    def to_dict(self) -> dict:
        """{
          "type": "independent_sigmoid_gate",
          "config": {
            "gate": "sigmoid",
            "frozen": false,
            "checkpoint": null
          }
        }"""
        result = asdict(self)
        result["branches"] = list(self.branches)
        result["availability"] = list(self.availability)
        return result


def parse_independent_sigmoid_gate_config_dict(
    config_dict: dict,
    config_path: str,
) -> IndependentSigmoidGateConfig:
    """Strictly parse the Independent Sigmoid Gate configuration within the checkpoint."""
    if not isinstance(config_dict, dict):
        raise RuntimeError(
            f"Independent Sigmoid Gate config must be a JSON object: {config_path}"
        )
    required = set(IndependentSigmoidGateConfig.__dataclass_fields__)
    actual = set(config_dict)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise RuntimeError(
            f"Independent Sigmoid Gate config field incomplete: missing={missing}, extra={extra}, path={config_path}"
        )
    parsed = dict(config_dict)
    parsed["branches"] = tuple(parsed["branches"])
    parsed["availability"] = tuple(parsed["availability"])
    return IndependentSigmoidGateConfig(**parsed)


class IndependentSigmoidGateNetwork(nn.Module):
    """Output Camera/Depth Gate from H36 question hidden, unnormalized."""

    def __init__(self, config: IndependentSigmoidGateConfig):
        super().__init__()
        self.config = config
        self.question_norm = nn.LayerNorm(config.hidden_size)
        self.input_proj = nn.Linear(config.hidden_size, config.bottleneck)
        self.activation = nn.GELU()
        self.output_proj = nn.Linear(config.bottleneck, len(config.branches))
        # Normal zero logits initialization strictly corresponds to sigmoid(0)=0.5; this round does not add 1/3 bias or perform random sweep.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        question_context: torch.Tensor,
        availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `[B,1]` Camera/Depth Gate and unnormalized `[B,2]` dual-path Gate."""
        hidden_size = int(self.config.hidden_size)
        if (
            question_context.dim() != 2
            or int(question_context.shape[-1]) != hidden_size
        ):
            raise RuntimeError(
                f"question_context must be [B,{hidden_size}], current={tuple(question_context.shape)}"
            )
        batch_size = int(question_context.shape[0])
        expected_availability = len(self.config.availability)
        if tuple(availability.shape) != (batch_size, expected_availability):
            raise RuntimeError(
                f"availability must be [B,{expected_availability}], current={tuple(availability.shape)}"
            )
        if not bool(torch.isfinite(question_context).all().item()):
            raise RuntimeError("question_context contains non-finite values.")
        if not bool(((availability == 0) | (availability == 1)).all().item()):
            raise RuntimeError(
                "availability can only contain 0/1 for has_pose/has_depth."
            )

        module_dtype = self.input_proj.weight.dtype
        question_context = question_context.to(dtype=module_dtype)
        availability = availability.to(
            device=question_context.device, dtype=module_dtype
        )
        logits = self.output_proj(
            self.activation(self.input_proj(self.question_norm(question_context)))
        )
        # Two paths each apply Sigmoid without normalization; values close to 0 naturally represent none.
        branch_gates = torch.sigmoid(logits.float()).to(dtype=logits.dtype)
        # Missing modalities cannot be handed to Gate for guessing; after Sigmoid, apply hard masking to ensure corresponding residual is strictly zero.
        branch_gates = branch_gates * availability
        if not bool(torch.isfinite(branch_gates).all().item()):
            raise RuntimeError(
                "Independent Sigmoid Gate outputs contain non-finite values."
            )
        camera_gate = branch_gates[:, 0:1]
        depth_gate = (
            branch_gates[:, 1:2]
            if len(self.config.branches) == 2
            else torch.zeros_like(camera_gate)
        )
        return camera_gate, depth_gate, branch_gates
