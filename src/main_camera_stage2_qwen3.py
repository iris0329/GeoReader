#!/usr/bin/env python3
"""Qwen3-VL native visual pathway Camera+Depth two-pass QA SFT."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import SequentialSampler
from transformers import (
    AutoProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from camera_interpreter import CameraInterpreter, parse_camera_interpreter_config_dict
from depth_interpreter import DepthInterpreter, parse_depth_interpreter_config_dict
from independent_sigmoid_gate_network import (
    IndependentSigmoidGateConfig,
    IndependentSigmoidGateNetwork,
)
from main_camera_interpreter import (
    CAMERA_END,
    CAMERA_PAD,
    CAMERA_START,
    DEPTH_END,
    DEPTH_PAD,
    DEPTH_START,
    _init_new_special_token_embeddings,
    _resolve_pixel_budget_from_checkpoint_configs,
    _set_image_processor_pixel_budget,
)
from n_images_length_sampler import build_n_images_length_grouped_sampler
from spatial_adapter import SpatialAdapter, SpatialAdapterConfig
from stage2_dataset_collator import CameraStage2Collator, MindCubeQADataset


SPECIAL_TOKENS = (
    CAMERA_START,
    CAMERA_PAD,
    CAMERA_END,
    DEPTH_START,
    DEPTH_PAD,
    DEPTH_END,
)


def load_stage1(checkpoint: Path, branch: str, hidden_size: int):
    """Restore frozen Interpreter strictly from checkpoint-local JSON and pt payload."""
    # The fully released GeoReader includes a frozen Reader; this is the explicitly supported complete checkpoint format.
    release = checkpoint / "release.json"
    if release.is_file():
        manifest = json.loads(release.read_text())
        if manifest.get("format") != "georeader-qwen3-full-state-dict":
            raise RuntimeError(f"Unknown checkpoint format: {release}")
        metadata = json.loads((checkpoint / "stage2_meta.json").read_text())
        values = metadata[f"{branch}_interpreter_config"]
        if branch == "camera":
            config = parse_camera_interpreter_config_dict(values, str(checkpoint))
            module = CameraInterpreter(config)
        elif branch == "depth":
            config = parse_depth_interpreter_config_dict(values, str(checkpoint))
            module = DepthInterpreter(config)
        else:
            raise RuntimeError(f"Unknown Reader: {branch}")
        if config.hidden_size != hidden_size or config.token_layout != "per_frame":
            raise RuntimeError(
                "The Reader structure of the complete checkpoint is inconsistent with the training foundation."
            )
        state = torch.load(
            checkpoint / manifest["weights"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        prefix = branch + "."
        module.load_state_dict(
            {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            },
            strict=True,
        )
        module.eval().requires_grad_(False)
        return module
    config_path = checkpoint / f"{branch}_interpreter_config.json"
    weights_path = checkpoint / f"{branch}_interpreter.pt"
    if branch == "camera":
        values = json.loads(config_path.read_text())
        if "camera_token_position" not in values:
            # Read missing placement metadata from the checkpoint-local Stage 1 contract.
            stage1 = json.loads((checkpoint / "camera_checkpoint.json").read_text())
            values["camera_token_position"] = stage1["interpreter"]["config"][
                "camera_token_position"
            ]
        config = parse_camera_interpreter_config_dict(values, str(config_path))
        module = CameraInterpreter(config)
    elif branch == "depth":
        config = parse_depth_interpreter_config_dict(
            json.loads(config_path.read_text()), str(config_path)
        )
        module = DepthInterpreter(config)
    else:
        raise RuntimeError(f"Unknown Interpreter: {branch}")
    if int(config.hidden_size) != hidden_size or config.token_layout != "per_frame":
        raise RuntimeError(
            f"{branch} checkpoint's hidden/layout does not match Qwen3: {config_path}"
        )
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Normalize tensor and JSON configurations through the same parser before strict comparison.
    parser = (
        parse_camera_interpreter_config_dict
        if branch == "camera"
        else parse_depth_interpreter_config_dict
    )
    saved_values = dict(payload[f"{branch}_interpreter_config"])
    if branch == "camera" and "camera_token_position" not in saved_values:
        stage1 = json.loads((checkpoint / "camera_checkpoint.json").read_text())
        saved_values["camera_token_position"] = stage1["interpreter"]["config"][
            "camera_token_position"
        ]
    saved_config = parser(saved_values, str(weights_path))
    if saved_config.to_dict() != config.to_dict():
        raise RuntimeError(f"{branch} pt/JSON config inconsistency: {weights_path}")
    module.load_state_dict(payload[f"{branch}_interpreter"], strict=True)
    module.eval().requires_grad_(False)
    return module


class Qwen3CameraDepthStage2(nn.Module):
    """Keep the original Gate+Dense formula, run twice with Qwen3 native vision and DeepStack."""

    def __init__(
        self,
        base_model: str,
        processor,
        camera_checkpoint: Path,
        depth_checkpoint: Path,
        checkpoint_config=None,
        recipe="sixbench_qwen3_8b",
    ):
        """Training starts from the base model and Reader initialization, and inference publishing only constructs the same structure as specified within the package."""
        super().__init__()
        if recipe not in ("sixbench_qwen3_8b", "mindcube_da3_qwen25_3b"):
            raise RuntimeError(f"Unsupported recipe: {recipe}")
        self.use_depth = recipe == "sixbench_qwen3_8b"
        # The original MindCube uses single-stream geometry; both baselines retain their native vision and MRoPE implementations.
        native_model = (
            Qwen3VLForConditionalGeneration
            if self.use_depth
            else Qwen2_5_VLForConditionalGeneration
        )
        native_config = Qwen3VLConfig if self.use_depth else Qwen2_5_VLConfig
        tokens = SPECIAL_TOKENS if self.use_depth else SPECIAL_TOKENS[:3]
        if checkpoint_config is None:
            self.qwen = native_model.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
            )
            self.qwen.resize_token_embeddings(len(processor.tokenizer))
            _init_new_special_token_embeddings(
                self.qwen, processor.tokenizer, list(tokens)
            )
        else:
            # Build the complete model from checkpoint-local configuration and load all weights strictly.
            self.qwen = native_model(native_config(**checkpoint_config["backbone"]))
            if self.qwen.config.text_config.vocab_size != len(processor.tokenizer):
                raise RuntimeError(
                    "Model vocabulary within the package is inconsistent with the tokenizer."
                )
        self.qwen.model.visual.requires_grad_(False)
        self.hidden_size = int(self.qwen.config.text_config.hidden_size)
        self.layer_count = int(self.qwen.config.text_config.num_hidden_layers)
        if checkpoint_config is None:
            self.camera = load_stage1(camera_checkpoint, "camera", self.hidden_size)
            self.depth = (
                load_stage1(depth_checkpoint, "depth", self.hidden_size)
                if self.use_depth
                else None
            )
        else:
            metadata = checkpoint_config["metadata"]
            self.camera = CameraInterpreter(
                parse_camera_interpreter_config_dict(
                    metadata["camera_interpreter_config"], "stage2_meta.json"
                )
            )
            self.depth = (
                DepthInterpreter(
                    parse_depth_interpreter_config_dict(
                        metadata["depth_interpreter_config"], "stage2_meta.json"
                    )
                )
                if self.use_depth
                else None
            )
            self.camera.eval().requires_grad_(False)
            if self.depth is not None:
                self.depth.eval().requires_grad_(False)
        self.gate = IndependentSigmoidGateNetwork(
            IndependentSigmoidGateConfig(
                hidden_size=self.hidden_size,
                branches=("camera", "depth") if self.use_depth else ("camera",),
                availability=("has_pose", "has_depth")
                if self.use_depth
                else ("has_pose",),
            )
        )
        self.camera_adapter = SpatialAdapter(
            SpatialAdapterConfig(hidden_size=self.hidden_size, adapter_type="dense")
        )
        self.depth_adapter = (
            SpatialAdapter(
                SpatialAdapterConfig(hidden_size=self.hidden_size, adapter_type="dense")
            )
            if self.use_depth
            else None
        )
        self.image_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.camera_id = int(processor.tokenizer.convert_tokens_to_ids(CAMERA_PAD))
        self.depth_id = (
            int(processor.tokenizer.convert_tokens_to_ids(DEPTH_PAD))
            if self.use_depth
            else -1
        )

    def gradient_checkpointing_enable(self, **kwargs):
        """Let the Trainer's recompute switch be controlled only by the native Qwen3 model."""
        self.qwen.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Disable native Qwen3 recompute."""
        self.qwen.gradient_checkpointing_disable()

    def _positions(self, ids, attention, grid):
        """Use the current Qwen3 native MRoPE, cannot apply Qwen2.5's position interface."""
        positions, _ = self.qwen.model.get_rope_index(
            ids,
            mm_token_type_ids=(ids == self.image_id).long(),
            image_grid_thw=grid,
            attention_mask=attention,
        )
        return positions

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        image_mask,
        question_mask,
        extrinsics,
        intrinsics,
        image_sizes,
        has_pose,
        has_depth,
        depth_modality_keep,
        labels,
        idx,
        n_images,
        depth_maps=None,
        num_items_in_batch=None,
    ):
        """Pass 1: Only read RGB + Question; Pass 2: Add two frame-wise residual streams and compute Answer CE."""
        del idx, num_items_in_batch
        if int(input_ids.shape[0]) != 1:
            raise RuntimeError(
                "Per-sample geometry requires a per-GPU batch size of 1."
            )
        if int(depth_modality_keep.item()) != 1:
            raise RuntimeError(
                "In this experiment, depth modality dropout is fixed at 0."
            )
        if not torch.equal(image_mask.bool(), input_ids == self.image_id):
            raise RuntimeError("image_mask does not match the true RGB token.")
        views = int(image_grid_thw.shape[0])
        if views != int(n_images.item()):
            raise RuntimeError("Number of QA images does not match the visual mesh.")
        has_camera = int(has_pose.item())
        has_depth_value = int(has_depth.item())
        if has_depth_value and not self.use_depth:
            raise RuntimeError("MindCube recipe does not accept Depth input")
        if has_camera not in (0, 1) or has_depth_value not in (0, 1):
            raise RuntimeError("Modal availability must be 0 or 1.")
        # Require depth_maps only for samples that provide depth.
        if has_depth_value and depth_maps is None:
            raise RuntimeError("has_depth=1 but missing depth_maps.")
        positions = question_mask[0].nonzero().flatten()
        if positions.numel() == 0:
            raise RuntimeError("Sample missing Question token.")
        prefix_length = int(positions[-1].item()) + 1
        if bool((labels[:, :prefix_length] != -100).any()):
            raise RuntimeError("First read obtains Answer supervision.")
        if bool(
            (
                (input_ids[:, :prefix_length] == self.camera_id)
                | (input_ids[:, :prefix_length] == self.depth_id)
            ).any()
        ):
            raise RuntimeError("First reads the Camera/Depth token.")
        embeddings = self.qwen.model.language_model.embed_tokens(input_ids)
        self.qwen.model.visual.eval()
        self.camera.eval()
        if self.depth is not None:
            self.depth.eval()
        # Pass 1: Freeze the backward graph, but still use the native Qwen3 visual path to retain DeepStack.
        with torch.no_grad():
            prefix_output = self.qwen.model(
                inputs_embeds=embeddings[:, :prefix_length],
                attention_mask=attention_mask[:, :prefix_length],
                position_ids=self._positions(
                    input_ids[:, :prefix_length],
                    attention_mask[:, :prefix_length],
                    image_grid_thw,
                ),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                return_dict=True,
            )
            question_hidden = prefix_output.last_hidden_state[:, -1].detach()
        # First pass only retains the hidden problem; release 32 frames of visual-language output promptly to reduce peak GPU memory during the second pass backpropagation.
        del prefix_output
        availability = torch.tensor(
            [[has_camera, has_depth_value] if self.use_depth else [has_camera]],
            device=input_ids.device,
            dtype=question_hidden.dtype,
        )
        camera_gate, depth_gate, _ = self.gate(question_hidden, availability)
        with torch.no_grad():
            camera_delta = (
                self.camera(
                    image_tokens=None,
                    image_grid_thw=image_grid_thw,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    image_sizes=image_sizes,
                    apply_output_gate=False,
                )
                if has_camera
                else embeddings.new_zeros((1, views, self.hidden_size))
            )
            depth_delta = (
                self.depth(
                    image_tokens=None,
                    image_grid_thw=image_grid_thw,
                    depth_maps=depth_maps,
                    image_sizes=image_sizes,
                    apply_output_gate=False,
                )
                if has_depth_value
                else embeddings.new_zeros((1, views, self.hidden_size))
            )
        camera_mask = input_ids == self.camera_id
        depth_mask = input_ids == self.depth_id
        if int(camera_mask.sum()) != views or int(depth_mask.sum()) != (
            views if self.use_depth else 0
        ):
            raise RuntimeError(
                "The number of Camera/Depth slots per frame does not match the number of images."
            )
        if tuple(camera_delta.shape) != (1, views, self.hidden_size) or tuple(
            depth_delta.shape
        ) != (1, views, self.hidden_size):
            raise RuntimeError("Interpreter output is not [1,V,d].")
        injected = embeddings.clone()
        # Fuse the base slot with an independently gated, zero-initialized residual adapter.
        injected[camera_mask] = (
            injected[camera_mask]
            + (camera_gate.unsqueeze(-1) * self.camera_adapter(camera_delta)).reshape(
                views, self.hidden_size
            )
        ).to(dtype=injected.dtype)
        if self.use_depth:
            injected[depth_mask] = (
                injected[depth_mask]
                + (depth_gate.unsqueeze(-1) * self.depth_adapter(depth_delta)).reshape(
                    views, self.hidden_size
                )
            ).to(dtype=injected.dtype)
        # Run native vision and all language layers again for the 2nd pass; compute CE only on Assistant Answer tokens.
        output = self.qwen.model(
            inputs_embeds=injected,
            attention_mask=attention_mask,
            position_ids=self._positions(input_ids, attention_mask, image_grid_thw),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        # Compute answer-only cross-entropy at supervised positions to avoid full-sequence logits.
        answer_mask = labels[:, 1:] != -100
        if not bool(answer_mask.any()):
            raise RuntimeError("Stage2 samples have no supervised answer tokens.")
        answer_hidden = output.last_hidden_state[:, :-1, :][answer_mask]
        answer_labels = labels[:, 1:][answer_mask]
        answer_logits = self.qwen.lm_head(answer_hidden)
        loss = F.cross_entropy(answer_logits.float(), answer_labels)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Stage2 Answer loss is infinite.")
        return {"loss": loss}


class Stage2Trainer(Trainer):
    """Reuse the same image-length-grouped global sampling as with 3B."""

    def _save(self, output_dir=None, state_dict=None):
        """Save the wrapped model with shared embeddings in PyTorch format."""
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Use torch.save to preserve tied embedding and output-head weights.
        torch.save(
            state_dict if state_dict is not None else self.model.state_dict(),
            Path(output_dir) / "pytorch_model.bin",
        )
        torch.save(self.args, Path(output_dir) / "training_args.bin")

    def _get_train_sampler(self, *args, **kwargs):
        """Smoke fixes the four-modal sequence; formal training is grouped in n_images parts."""
        if getattr(self.args, "stage2_smoke", False):
            return SequentialSampler(self.train_dataset)
        return build_n_images_length_grouped_sampler(
            self.train_dataset,
            self.args,
            log_prefix="[qwen3-stage2-sampler]",
        )


class MetadataCallback(TrainerCallback):
    """Each checkpoint leaves behind a JSON file that can be reconstructed for forward passes and the native processor."""

    def __init__(self, processor, metadata):
        self.processor = processor
        self.metadata = metadata

    def on_save(self, args, state, control, **kwargs):
        """After the Trainer completes weight saving, it writes the interface contract into the same directory."""
        if state.is_world_process_zero:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            self.processor.save_pretrained(checkpoint)
            (checkpoint / "stage2_meta.json").write_text(
                json.dumps(self.metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            model = kwargs["model"]
            model = model.module if hasattr(model, "module") else model
            model.qwen.config.to_json_file(checkpoint / "backbone_config.json")
            model.qwen.generation_config.save_pretrained(checkpoint)
        return control


def main():
    """Launch four-card matched Stage2 from Qwen3 base with two-stage Stage1 weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--qa-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows")
    parser.add_argument("--resume-from-checkpoint")
    # Save optimizer and scheduler states to support training resumption.
    parser.add_argument(
        "--keep-resume-state",
        action="store_true",
        help="Keep optimizer and scheduler state so that training can be resumed.",
    )
    parser.add_argument("--smoke-resume-no-save", action="store_true")
    parser.add_argument("--smoke-no-save", action="store_true")
    args = parser.parse_args()
    recipe_config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/sixbench_qwen3_8b/recipe.json"
        ).read_text()
    )
    cfg = recipe_config["stage2"]
    # Set the seed before constructing trainable modules so their initialization is reproducible.
    set_seed(cfg["seed"])
    print(f"[initialization] seed={cfg['seed']} before_model=1", flush=True)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.smoke and world_size != cfg["world_size"]:
        raise RuntimeError(
            "Stage2 must be run with torchrun using 4 GPU(s), batch size 1 per GPU, gradient accumulation 2; single-card runs are only allowed with explicit Smoke."
        )
    if args.smoke_resume_no_save and not (args.smoke and args.resume_from_checkpoint):
        raise RuntimeError("Skipping the second save is only allowed with --smoke.")
    if args.smoke_no_save and not args.smoke:
        raise RuntimeError("Skipping saves is only allowed with --smoke.")
    if args.smoke != bool(args.smoke_rows):
        raise RuntimeError("--smoke and --smoke-rows must be given together.")
    qa_paths = (
        [Path(args.smoke_rows)]
        if args.smoke
        else [
            Path(line.strip())
            for line in Path(args.qa_manifest).read_text().splitlines()
            if line.strip()
        ]
    )
    if not args.smoke and len(qa_paths) != 36:
        raise RuntimeError(f"Expected the 36 Stage 2 QA files, got {len(qa_paths)}")
    for path in qa_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    processor = AutoProcessor.from_pretrained(args.model)
    # Require matching Reader pixel budgets and apply them to the Stage 2 processor.
    budgets = []
    for checkpoint in (Path(args.camera), Path(args.depth)):
        if (checkpoint / "release.json").is_file():
            # The full release model uses the native processor_config; pixel budget is explicitly recorded in the Stage2 contract.
            saved = json.loads((checkpoint / "stage2_meta.json").read_text())
            budgets.append((int(saved["min_pixels"]), int(saved["max_pixels"])))
        else:
            values = _resolve_pixel_budget_from_checkpoint_configs(
                min_pixels=None,
                max_pixels=None,
                checkpoint_dirs=[str(checkpoint)],
                context="Qwen3 Stage2 train",
            )
            budgets.append((values[0], values[1]))
    if len(set(budgets)) != 1:
        raise RuntimeError(f"Inconsistent pixel budget between two Readers: {budgets}")
    min_pixels, max_pixels = budgets[0]
    pixel_budget_source = "checkpoint-local JSON"
    actual_min_pixels, actual_max_pixels = _set_image_processor_pixel_budget(
        processor.image_processor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (actual_min_pixels, actual_max_pixels) != (min_pixels, max_pixels):
        raise RuntimeError(
            "Qwen3 Stage2 image processor pixel budget setting is not taking effect."
        )
    print(
        f"[image_processor] min_pixels={min_pixels} max_pixels={max_pixels} source={pixel_budget_source}",
        flush=True,
    )
    for token in SPECIAL_TOKENS:
        processor.tokenizer.add_tokens(token, special_tokens=True)
    dataset = MindCubeQADataset(
        data_path=[str(path) for path in qa_paths],
        image_folder=str(Path(args.data_root).resolve()),
        expected_image_size="mixed",
        draw_spar_marker_specs=True,
    )
    if len(dataset) != (8 if args.smoke else cfg["rows"]):
        raise RuntimeError(f"Unexpected number of QA rows: {len(dataset)}")
    collator = CameraStage2Collator(
        processor=processor,
        camera_latent_size=1,
        spatial_token_position="after_question",
        use_depth_interpreter=True,
        depth_latent_size=1,
        camera_token_layout="per_frame",
        depth_token_layout="per_frame",
        missing_pose_token_policy="keep_zero_delta",
        output_gate_type="independent_sigmoid",
        spatial_fusion_type="latent_tokens",
    )
    model = Qwen3CameraDepthStage2(
        args.model,
        processor,
        Path(args.camera),
        Path(args.depth),
    )
    model.qwen.config.use_cache = False
    metadata = {
        "model_type": "qwen3_vl",
        "base_model_name_or_path": str(Path(args.model).resolve()),
        "camera_stage1_checkpoint": str(Path(args.camera).resolve()),
        "depth_stage1_checkpoint": str(Path(args.depth).resolve()),
        "hidden_size": model.hidden_size,
        "llm_layer_count": model.layer_count,
        "camera_interpreter_config": model.camera.config.to_dict(),
        "depth_interpreter_config": model.depth.config.to_dict(),
        "camera_token_id": model.camera_id,
        "depth_token_id": model.depth_id,
        "fusion": "two_pass_last_question_independent_sigmoid_dense_residual_add_input_layer",
        "spatial_token_position": "after_question",
        "stage2_mode": "camera_depth",
        "qa_files": [str(path.resolve()) for path in qa_paths],
        "qa_rows": len(dataset),
        "gpu_count": world_size,
        "per_device_train_batch_size": 1,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "gradient_accumulation_steps": cfg["gradient_accumulation"],
        "effective_batch_size": world_size * cfg["gradient_accumulation"],
        "epochs": cfg["epochs"],
        "max_steps": 4 if args.smoke else cfg["max_steps"],
        "learning_rate": cfg["learning_rate"],
        "warmup_steps": cfg["warmup_steps"],
        "weight_decay": cfg["weight_decay"],
        "save_steps": 4 if args.smoke else cfg["save_steps"],
        "train_sampler_mode": "length_grouped",
        "save_only_model": args.smoke or not args.keep_resume_state,
        "initialization_seed": cfg["seed"],
        "seed": cfg["seed"],
        "data_seed": cfg["seed"],
    }
    training = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        num_train_epochs=cfg["epochs"],
        max_steps=4 if args.smoke else cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0 if args.smoke else 8,
        logging_steps=1 if args.smoke else cfg["logging_steps"],
        save_steps=4 if args.smoke else cfg["save_steps"],
        # Two complete Smoke states of 8B will exceed the current scratch space; restore rounds only validate strict load and the second-step update.
        save_strategy="no"
        if args.smoke_resume_no_save or args.smoke_no_save
        else "steps",
        # Use the save format supported by the installed Transformers version.
        save_total_limit=2 if args.smoke else 1,
        # Keep the latest complete checkpoint state for resuming training.
        save_only_model=args.smoke or not args.keep_resume_state,
        report_to=[],
        disable_tqdm=True,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )
    training.stage2_smoke = args.smoke
    trainer = Stage2Trainer(
        model=model,
        args=training,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[MetadataCallback(processor, metadata)],
    )
    if trainer.is_world_process_zero():
        print(
            "[stage2-config] "
            + json.dumps(
                {
                    "model": args.model,
                    "environment": os.environ.get("CONDA_PREFIX"),
                    "transformers": __import__("transformers").__version__,
                    "output_dir": args.output_dir,
                    "qa_rows": len(dataset),
                    "hidden_size": model.hidden_size,
                    "llm_layers": model.layer_count,
                    "smoke": args.smoke,
                    "training_setting": f"g{world_size} b1a2",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        print(f"[stage2-complete] global_step={trainer.state.global_step}", flush=True)


if __name__ == "__main__":
    main()
