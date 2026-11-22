#!/usr/bin/env python3
"""RGB-only Stage2 QA SFT independent entry for Qwen3-VL-8B-Instruct.

This line only uses images, Question, and Assistant Answer. It does not load the Stage1 checkpoint.
Do not create Camera/Depth tokens, nor read any geometric fields; native Qwen3-VL template and forward
This is the only model interface in this file, completely isolated from the Qwen3.5-4B entry point and checkpoint.
"""

from __future__ import annotations

import gc
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME


import main_stage2_imageqa_sft as imageqa_base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"
EXPECTED_MODEL_TYPE = "qwen3_vl"
EXPECTED_HIDDEN_SIZE = 4096
EXPECTED_NUM_LAYERS = 36


def get_args():
    """Expose the parameters used by the RGB-only 8B training recipe."""
    parser = argparse.ArgumentParser()
    for name in ("model", "image_folder", "save_model_path"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--qa_data_path", nargs="+", required=True)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--expected_final_step", type=int, required=True)
    parser.add_argument(
        "--wrapper_script", default=str(PROJECT_ROOT / "train_stage2.py")
    )
    parser.add_argument("--wandb_name", default="georeader-8b-vanilla-sft")
    # Each parameter meaning: epochs/steps control QA exposure; b1a2 together with GPU count controls effective batch.
    for name, default in {
        "epochs": 2.0,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
    }.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    # Pixel budget controls visual overhead, warmup controls initial updates; save/log frequency does not alter supervision.
    for name, default in {
        "max_steps": 33850,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "warmup_steps": 100,
        "min_pixels": 3136,
        "max_pixels": 150528,
        "logging_steps": 100,
        "save_steps": 16925,
        "save_total_limit": 1,
        "dataloader_num_workers": 8,
    }.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in {
        "save_only_model": "0",
        "disable_tqdm": "1",
        "gradient_checkpointing": "1",
        "draw_spar_marker_specs": "1",
    }.items():
        parser.add_argument(f"--{name}", choices=("0", "1"), default=default)
    parser.add_argument("--expected_image_size", default="mixed")
    parser.add_argument("--train_sampler_mode", choices=("random",), default="random")
    parser.set_defaults(dry_run_data=False, max_samples=None)
    return parser.parse_args()


def _template(processor, messages: list, *, add_generation_prompt: bool) -> str:
    """Expand the native chat template using Qwen3-VL's built-in tokenizer, without manually writing role/image tokens."""
    return processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def _set_native_pixel_budget(
    image_processor, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Set the minimum and maximum pixel area for smart-resize in Qwen3-VL's official SizeDict."""
    if image_processor is None or not hasattr(image_processor, "size"):
        raise RuntimeError("Qwen3-VL processor is missing image_processor.size.")
    min_pixels, max_pixels = int(min_pixels), int(max_pixels)
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise RuntimeError(
            f"Pixel budget illegal for Qwen3-VL: min={min_pixels}, max={max_pixels}"
        )
    # Convert indexable processor size settings to a plain mapping.
    image_processor.size["shortest_edge"] = min_pixels
    image_processor.size["longest_edge"] = max_pixels
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    return int(image_processor.size["shortest_edge"]), int(
        image_processor.size["longest_edge"]
    )


class Qwen3VLRGBOnlyCollator:
    """Convert a single multi-image QA data sample into a native Answer-only CE batch for Qwen3-VL."""

    def __init__(self, processor):
        """Save processor and native image pad id; train contract fixes one sample."""
        self.processor = processor
        self.image_token_id = int(
            processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        )
        if self.image_token_id < 0:
            raise RuntimeError("Qwen3-VL tokenizer is missing <|image_pad|>.")

    def _grid(self, images: List[object]) -> torch.Tensor:
        """Obtain the actual resize grid for each image, ensuring that the coordinate text and visual input correspond to the same processed result."""
        grids = []
        for image_idx, image in enumerate(images):
            encoded = self.processor.image_processor(
                images=[image], return_tensors="pt"
            )
            grid = encoded.get("image_grid_thw")
            if grid is None or tuple(grid.shape) != (1, 3):
                raise RuntimeError(
                    f"Qwen3-VL image_grid_thw illegal: image={image_idx}, shape={(None if grid is None else tuple(grid.shape))}"
                )
            grids.append(grid)
        return torch.cat(grids, dim=0)

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        """native generation prefix as the sole Answer starting point, prohibit inferring supervision scope by string guessing."""
        if len(examples) != 1:
            raise RuntimeError(
                f"Qwen3-VL RGB-only fixed per-device batch=1, actual={len(examples)}"
            )
        ex = examples[0]
        images = ex["images"]
        try:
            rewritten = imageqa_base.rewrite_sample_coordinates_in_texts(
                ex,
                {"text_input": ex["text_input"], "text_output": ex["text_output"]},
                images,
                self._grid(images),
                context=f"Qwen3-VL RGB-only idx={ex['idx']}",
            )
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": rewritten["text_input"]})
            user = [{"role": "user", "content": content}]
            full = user + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": rewritten["text_output"]}],
                }
            ]
            prefix = self.processor(
                images=images,
                text=[_template(self.processor, user, add_generation_prompt=True)],
                return_tensors="pt",
            )
            batch = self.processor(
                images=images,
                text=[_template(self.processor, full, add_generation_prompt=False)],
                return_tensors="pt",
            )
            prefix_len = int(prefix["input_ids"].shape[1])
            ids = batch["input_ids"][0]
            if prefix_len >= int(ids.numel()) or not torch.equal(
                prefix["input_ids"][0], ids[:prefix_len]
            ):
                raise RuntimeError(
                    "Qwen3-VL native generation prefix is not a training sequence prefix, and cannot safely construct Answer CE."
                )
            labels = ids.clone()
            labels[:prefix_len] = -100
            labels[labels == self.image_token_id] = -100
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == int(self.processor.tokenizer.pad_token_id)] = -100
            if not bool((labels != -100).any().item()):
                raise RuntimeError(
                    f"Qwen3-VL sample has no Assistant Answer supervised token: idx={ex['idx']}"
                )
            batch["labels"] = labels.unsqueeze(0)
            if int(batch["image_grid_thw"].shape[0]) != int(ex["n_images"]):
                raise RuntimeError(f"Qwen3-VL image count mismatch: idx={ex['idx']}")
            return dict(batch)
        finally:
            for image in images:
                image.close()


class Qwen3VLRGBOnlyModel(nn.Module):
    """Light wrapper for native Qwen3-VL RGB forward, responsible only for freezing the visual tower."""

    def __init__(self, qwen_model: Qwen3VLForConditionalGeneration):
        """Save the official model and freeze the vision tower; do not add any custom layers."""
        super().__init__()
        self.qwen_model = qwen_model
        self.config = qwen_model.config
        self.qwen_model.model.visual.requires_grad_(False)
        self.qwen_model.model.visual.eval()
        if any(p.requires_grad for p in self.qwen_model.model.visual.parameters()):
            raise RuntimeError("Qwen3-VL Vision Encoder freezing failed.")

    def train(self, mode: bool = True):
        """Keep the visual tower in eval mode to prevent Trainer's train() from altering the dropout behavior in the frozen tower."""
        super().train(mode)
        self.qwen_model.model.visual.eval()
        return self

    def gradient_checkpointing_enable(self, **kwargs):
        """Hand over gradient checkpointing to the official Qwen3-VL implementation."""
        self.qwen_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the official Qwen3-VL."""
        self.qwen_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        """Call native multi-image forward; no Camera, Depth, Gate, or additional loss inputs."""
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError(
                "Qwen3-VL RGB-only forward is missing pixel_values or image_grid_thw."
            )
        return self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            return_dict=True,
        )


class Qwen3VLRGBOnlyTrainer(Trainer):
    """Save independent checkpoint metadata, and validate RGB-only gradient boundaries during the first backward pass."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Require finite Answer CE, zero Vision gradient, and at least one non-zero finite LLM gradient."""
        loss = super().training_step(
            model, inputs, num_items_in_batch=num_items_in_batch
        )
        if getattr(self, "gradient_contract", None) is not None:
            return loss
        unwrapped = model.module if hasattr(model, "module") else model
        if not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError(f"Qwen3-VL Answer CE non-finite: {loss}")
        if [
            n
            for n, p in unwrapped.qwen_model.model.visual.named_parameters()
            if p.grad is not None
        ]:
            raise RuntimeError(
                "The vision encoder should be frozen but received gradients."
            )
        for name, p in unwrapped.qwen_model.model.language_model.named_parameters():
            if (
                p.requires_grad
                and p.grad is not None
                and float(p.grad.detach().abs().max()) > 0
            ):
                if not bool(torch.isfinite(p.grad).all().item()):
                    raise RuntimeError(f"Qwen3-VL LLM gradient is not finite: {name}")
                self.gradient_contract = {
                    "answer_ce_finite": True,
                    "vision_gradients_absent": True,
                    "llm_gradient": name,
                }
                print(
                    f"[smoke-gradient-check] {json.dumps(self.gradient_contract, ensure_ascii=False)}",
                    flush=True,
                )
                return loss
        raise RuntimeError("No non-zero LLM gradients found during --smoke.")

    def _save(self, output_dir=None, state_dict=None):
        """Write the model, processor, TrainingArguments, and unique stage2_meta simultaneously at each checkpoint."""
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.qwen_model.save_pretrained(output_dir)
        self.processing_class.save_pretrained(output_dir)
        if not (Path(output_dir) / "tokenizer_config.json").is_file() or not any(
            (Path(output_dir) / name).is_file()
            for name in ("processor_config.json", "preprocessor_config.json")
        ):
            raise RuntimeError(
                f"Qwen3-VL checkpoint is missing official processor/tokenizer files: {output_dir}"
            )
        torch.save(self.args, Path(output_dir) / TRAINING_ARGS_NAME)
        (Path(output_dir) / "stage2_meta.json").write_text(
            json.dumps(dict(self.args.stage2_meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _validate_checkpoint(path: str) -> None:
    """Check if checkpoint contains the minimum files required to rebuild Qwen3-VL RGB-only eval."""
    root = Path(path)
    required = ["config.json", "stage2_meta.json", "tokenizer_config.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if not (
        (root / "model.safetensors").is_file()
        or (root / "model.safetensors.index.json").is_file()
    ):
        missing.append("model.safetensors or model.safetensors.index.json")
    if not any(
        (root / name).is_file()
        for name in ("processor_config.json", "preprocessor_config.json")
    ):
        missing.append("processor_config.json or preprocessor_config.json")
    if missing:
        raise RuntimeError(
            f"Qwen3-VL checkpoint is incomplete: {root}, missing={missing}"
        )
    meta = json.loads((root / "stage2_meta.json").read_text(encoding="utf-8"))
    expected = {
        "backbone_family": EXPECTED_MODEL_TYPE,
        "vision_trainable": False,
        "full_sft": True,
        "loss_contract": "answer_causal_ce_only",
    }
    if {key: meta.get(key) for key in expected} != expected:
        raise RuntimeError(f"Qwen3-VL checkpoint metadata mismatch: {meta}")


def _strict_reload(path: str) -> dict:
    """Reload Qwen3-VL and processor strictly from smoke checkpoint, reject rolling back to initial model directory."""
    _validate_checkpoint(path)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, local_files_only=True
    )
    result = {
        "strict_reload": True,
        "processor": type(processor).__name__,
        "model": type(model).__name__,
        "model_type": model.config.model_type,
    }
    if result["model_type"] != EXPECTED_MODEL_TYPE:
        raise RuntimeError(f"Qwen3-VL strict reload model type error: {result}")
    del processor, model
    gc.collect()
    return result


def _barrier() -> None:
    """Synchronize all ranks in DDP to ensure that during save and strict reload, other processes do not prematurely release while rank 0 is saving."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    """Train Qwen3-VL-8B RGB-only SFT with frozen Design's ep2 / 4 GPU(s), batch size 1 per GPU, gradient accumulation 2 parameters."""
    args = get_args()
    if args.dry_run_data:
        raise RuntimeError(
            "Dry runs without a GPU are not supported; use --smoke on a GPU instead."
        )
    imageqa_base.assert_cpu_affinity_for_training()
    imageqa_base.seed_everything(42)
    rank, local_rank = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL RGB-only SFT must run on GPU compute nodes.")
    torch.cuda.set_device(local_rank)
    torch.cuda.reset_peak_memory_stats(local_rank)
    # The release entry must provide a clear QA recipe and cannot automatically switch to other training sets.
    recipe_path = os.environ.get("RGB_TRAIN_RECIPE")
    if not recipe_path:
        raise RuntimeError(
            "RGB_TRAIN_RECIPE is missing, please use train_stage2.py as entry point"
        )
    recipe = json.loads(Path(recipe_path).read_text())
    expected_paths = [Path(p) for p in recipe["qa_data_paths"]]
    if [Path(p).resolve() for p in args.qa_data_path] != [
        p.resolve() for p in expected_paths
    ]:
        raise RuntimeError("Stage 2 QA files or their order differ from the recipe.")
    if (
        float(args.epochs) != 2.0
        or int(args.per_device_train_batch_size) != 1
        or int(args.gradient_accumulation_steps) != 2
    ):
        raise RuntimeError(
            "Stage 2 SFT settings differ from the recipe (2 epochs, batch size 1 per GPU); --smoke may only reduce the number of samples."
        )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    min_pixels, max_pixels = _set_native_pixel_budget(
        processor.image_processor, args.min_pixels, args.max_pixels
    )
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if (
        config.model_type != EXPECTED_MODEL_TYPE
        or config.text_config.hidden_size != EXPECTED_HIDDEN_SIZE
        or config.text_config.num_hidden_layers != EXPECTED_NUM_LAYERS
    ):
        raise RuntimeError(f"Unsupported backbone config: {config.model_type}")
    dataset = imageqa_base.ImageQAJsonlDataset(
        args.qa_data_path,
        args.image_folder,
        args.max_samples,
        imageqa_base._parse_expected_image_size(args.expected_image_size),
        True,
        args.draw_spar_marker_specs == "1",
    )
    if recipe:
        expected_counts = {
            str(Path(p).resolve()): n
            for p, n in zip(recipe["qa_data_paths"], recipe["qa_data_rows"])
        }
        if (
            dataset.rows_per_file != expected_counts
            or dataset.total_rows != recipe["qa_rows_total"]
        ):
            raise RuntimeError("Training data does not match the recipe")
    model = Qwen3VLRGBOnlyModel(
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    meta = {
        "stage": "stage2_rgb_only_sft",
        "backbone_family": EXPECTED_MODEL_TYPE,
        "backbone_model_name": "Qwen3-VL-8B-Instruct",
        "backbone_source_path": str(Path(args.model).resolve()),
        "model_type": config.model_type,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": EXPECTED_NUM_LAYERS,
        "processor_class": type(processor).__name__,
        "transformers_version": transformers.__version__,
        "thinking_enabled": None,
        "vision_trainable": False,
        "full_sft": True,
        "loss_contract": "answer_causal_ce_only",
        "logical_sequence_contract": "images_then_question_then_assistant_answer",
        "inference_prompt_contract": "images_then_question_only_native_template",
        "interpreter_used": False,
        "camera_depth_reader_gate_router_used": False,
        "auxiliary_loss_used": False,
        "qa_data_paths": list(dataset.qa_data_paths),
        "qa_rows_total": dataset.total_rows,
        "qa_data_rows": [dataset.rows_per_file[p] for p in dataset.qa_data_paths],
        "expected_train_rows": recipe["qa_rows_total"],
        "expected_final_step": args.expected_final_step,
        "wrapper_script": args.wrapper_script,
        "qa_rows_used": len(dataset),
        "epochs": float(args.epochs),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "world_size": world_size,
        "effective_train_batch_size": world_size
        * int(args.per_device_train_batch_size)
        * int(args.gradient_accumulation_steps),
        "stage2_training_setting": f"g{world_size} b1a2",
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "weight_decay": float(args.weight_decay),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "gradient_checkpointing": args.gradient_checkpointing == "1",
        "train_sampler_mode": args.train_sampler_mode,
    }
    # Wandb is not installed in the environment; no callback is created, and all traceable training logs are still written independently to Slurm stdout, with WANDB_MODE=offline.
    training_args = TrainingArguments(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch_fused",
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
        run_name=args.wandb_name,
        disable_tqdm=args.disable_tqdm == "1",
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.gradient_checkpointing == "1",
        save_only_model=args.save_only_model == "1",
    )
    training_args.stage2_meta = meta
    training_args.train_sampler_mode = args.train_sampler_mode
    trainer = Qwen3VLRGBOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Qwen3VLRGBOnlyCollator(processor),
        processing_class=processor,
        callbacks=[],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if (
        args.expected_final_step is not None
        and trainer.state.global_step != args.expected_final_step
    ):
        raise RuntimeError(
            "The final training steps for Qwen3-VL do not match the recipe."
        )
    if trainer.gradient_contract is None or trainer.state.global_step <= 0:
        raise RuntimeError("Qwen3-VL training failed the first gradient gate.")
    if rank == 0:
        imageqa_base.save_final_checkpoint_with_latest_resume_state(
            trainer, _validate_checkpoint, 1, "[qwen3vl-rgb]"
        )
    result = {
        "rank": rank,
        "global_step": int(trainer.state.global_step),
        "gradient_contract": trainer.gradient_contract,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(local_rank)
        / 1024**3,
        "memory_headroom_gib": (
            torch.cuda.get_device_properties(local_rank).total_memory
            - torch.cuda.max_memory_reserved(local_rank)
        )
        / 1024**3,
    }
    result_path = Path(args.save_model_path) / f"smoke_result_rank{rank}.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _barrier()
    step = int(trainer.state.global_step)
    del trainer, model, processor, dataset
    gc.collect()
    torch.cuda.empty_cache()
    _barrier()
    if rank == 0 and os.environ.get("STRICT_RELOAD", "0") == "1":
        result.update(
            _strict_reload(str(Path(args.save_model_path) / f"checkpoint-{step}"))
        )
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    _barrier()


if __name__ == "__main__":
    main()
