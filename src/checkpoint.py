"""Load configuration and full state_dict strictly from the release package, without reading the original training machine directory."""

import json
from pathlib import Path

from accelerate import init_empty_weights
import torch
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen3VLForConditionalGeneration,
)

from main_camera_interpreter import _set_image_processor_pixel_budget
from main_camera_stage2_qwen3 import Qwen3CameraDepthStage2


def load_georeader(checkpoint, device="cuda"):
    """Only build structure based on two JSON files within the package; all model parameters must be provided by a complete checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    for name in (
        "pytorch_model.bin",
        "backbone_config.json",
        "stage2_meta.json",
        "processor_config.json",
        "tokenizer.json",
    ):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    metadata = json.loads((checkpoint / "stage2_meta.json").read_text())
    backbone = json.loads((checkpoint / "backbone_config.json").read_text())
    recipe = metadata.get("recipe", "sixbench_qwen3_8b")
    if recipe not in ("sixbench_qwen3_8b", "mindcube_da3_qwen25_3b"):
        raise RuntimeError(f"Unknown recipe in checkpoint: {recipe}")
    sixbench = recipe == "sixbench_qwen3_8b"
    expected = {
        "model_type": "qwen3_vl" if sixbench else "qwen2_5_vl",
        "stage2_mode": "camera_depth" if sixbench else "camera_only",
        "fusion": "two_pass_last_question_independent_sigmoid_dense_residual_add_input_layer",
        "spatial_token_position": "after_question",
        "hidden_size": 4096 if sixbench else 2048,
        "llm_layer_count": 36,
    }
    if not sixbench:
        expected.update(
            independent_sigmoid_gate_config={
                "hidden_size": 2048,
                "bottleneck": 256,
                "branches": ["camera"],
                "availability": ["has_pose"],
            },
            spatial_adapter_config={"hidden_size": 2048, "adapter_type": "dense"},
        )
    bad = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if bad:
        raise RuntimeError(f"Unexpected checkpoint structure: {bad}")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    pixels = _set_image_processor_pixel_budget(
        processor.image_processor,
        min_pixels=int(metadata["min_pixels"]),
        max_pixels=int(metadata["max_pixels"]),
    )
    if pixels != (int(metadata["min_pixels"]), int(metadata["max_pixels"])):
        raise RuntimeError("Processor pixel limits do not match the checkpoint")
    # Empty parameters only define shape, avoiding allocating a random 8B model first and then loading the entire weight set, reducing CPU memory peak.
    with init_empty_weights():
        model = Qwen3CameraDepthStage2(
            "",
            processor,
            None,
            None,
            checkpoint_config={"backbone": backbone, "metadata": metadata},
            recipe=recipe,
        )
    state = torch.load(
        checkpoint / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # After assign=True, restore native shared word embeddings, maintaining the same parameter bindings as the original training wrapper.
    model.qwen.tie_weights()
    model.qwen.generation_config = GenerationConfig.from_pretrained(
        checkpoint, local_files_only=True
    )
    if (model.camera_id, model.depth_id) != (
        metadata["camera_token_id"],
        metadata["depth_token_id"],
    ):
        raise RuntimeError(
            "Geometry token IDs in the tokenizer do not match the checkpoint."
        )
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return model, processor, metadata


def load_sft(checkpoint, device="cuda"):
    """Load a complete trained RGB-only checkpoint strictly."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    metadata = json.loads((checkpoint / "stage2_meta.json").read_text())
    expected = {
        "backbone_family": "qwen3_vl",
        "vision_trainable": False,
        "full_sft": True,
        "loss_contract": "answer_causal_ce_only",
        "interpreter_used": False,
    }
    bad = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if bad:
        raise RuntimeError(f"Not a Vanilla SFT checkpoint: {bad}")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    _set_image_processor_pixel_budget(
        processor.image_processor,
        min_pixels=int(metadata["min_pixels"]),
        max_pixels=int(metadata["max_pixels"]),
    )
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        output_loading_info=True,
    )
    # Reject incomplete SFT weights instead of accepting missing parameters.
    errors = {
        key: loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading.get(key)
    }
    if errors:
        raise RuntimeError(f"SFT weights do not match the model: {errors}")
    if (
        model.config.model_type != "qwen3_vl"
        or model.config.text_config.hidden_size != 4096
    ):
        raise RuntimeError("SFT checkpoint is not a Qwen3-VL-8B model")
    model.to(device=device).eval().requires_grad_(False)
    return model, processor, metadata
