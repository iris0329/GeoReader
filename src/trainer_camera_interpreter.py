"""Camera Interpreter Custom Trainer: Inherits HF Trainer, saves/restores only Interpreter weights."""

import hashlib
import json
import os
import glob

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from n_images_length_sampler import build_n_images_length_grouped_sampler


def _sha256_file(path: str) -> str:
    """Compute the SHA256 of the checkpoint small weights/config."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage1_checkpoint_metadata(metadata: dict, checkpoint_dir: str) -> None:
    """Strictly verify required fields and artifact hashes in the new Stage1 meta."""
    if int(metadata.get("stage1_meta_schema_version", -1)) != 1:
        raise RuntimeError(
            f"Stage1 meta schema does not support: {checkpoint_dir}, actual={metadata.get('stage1_meta_schema_version')!r}"
        )
    required = {
        "stage1_branch",
        "base_model",
        "data",
        "code_snapshot_identity",
        "input_recipe",
        "frame_order",
        "processor",
        "interpreter",
        "training_target",
        "optimization",
        "artifacts_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise RuntimeError(
            f"Stage1 checkpoint meta missing fields {missing}: {checkpoint_dir}"
        )
    branch = metadata["stage1_branch"]
    if branch not in {"Camera", "Depth"}:
        raise RuntimeError(
            f"Stage1 checkpoint branch illegal: {branch!r}, dir={checkpoint_dir}"
        )
    target = metadata["training_target"]
    if (
        not isinstance(target, dict)
        or not target.get("loss_mode")
        or not target.get("cot_version")
    ):
        raise RuntimeError(
            f"Stage1 checkpoint training_target missing loss_mode/cot_version: {checkpoint_dir}"
        )
    if target.get("stage1_task") == "camera_reference_target_pairs":
        config = metadata.get("interpreter", {}).get("config", {})
        expected = {
            "loss_mode": "motion_content",
            "pair_scope": "image_1_plus_original_target_frame",
            "frame_index_encoding": "fixed_31",
        }
        mismatches = {
            key: (target.get(key), value)
            for key, value in expected.items()
            if target.get(key) != value
        }
        if (
            mismatches
            or config.get("camera_frame_index_encoding") != "fixed_31"
            or config.get("token_layout") != "per_frame"
        ):
            raise RuntimeError(
                f"Camera pair checkpoint inconsistency: target={mismatches}, config={config}"
            )
        # A two-frame window produces exactly one reference-target pair.
        if int(target.get("expanded_pair_rows", 0)) < int(
            target.get("source_window_rows", 0)
        ):
            raise RuntimeError(
                "The number of camera pairing rows cannot be less than the original Window number."
            )
    if target.get("loss_mode") == "caption_pair_delta":
        is_crossframe = (
            target.get("stage1_task") == "depth_caption_pair_delta_crossframe"
        )
        required_target = {
            "stage1_task": (
                "depth_caption_pair_delta_crossframe"
                if is_crossframe
                else "depth_caption_pair_delta"
            ),
            "assistant_target_field": "depth_pair_answer",
            "relation_mask_source_field": "depth_pair_relation_answer",
            "caption_mask_source_field": "depth_scene_cot",
            "relation_loss_weight": 0.5,
            "caption_loss_weight": 0.5,
            "relation_pair_scope": "same_and_cross_frame"
            if is_crossframe
            else "same_frame",
        }
        mismatches = {
            key: (target.get(key), expected)
            for key, expected in required_target.items()
            if target.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"caption_pair_delta checkpoint training_target incomplete or inconsistent: mismatches={mismatches}, dir={checkpoint_dir}"
            )
        initialization = metadata.get("initialization")
        if (
            not isinstance(initialization, dict)
            or initialization.get("mode") != "weights_only_fresh_optimizer_scheduler"
            or initialization.get("optimizer_loaded") is not False
            or initialization.get("scheduler_loaded") is not False
            or initialization.get("trainer_state_loaded") is not False
            or not initialization.get("source_checkpoint")
            or not initialization.get("artifacts_sha256")
        ):
            raise RuntimeError(
                f"caption_pair_delta checkpoint missing weights-only fresh optimizer initialization contract: initialization={initialization!r}, dir={checkpoint_dir}"
            )
    if target.get("stage1_task") == "depth_perframe_caption_microbatch":
        prompt_source = target.get("prompt_source")
        prompt = target.get("prompt")
        loss_mode = target.get("loss_mode")
        if loss_mode not in {"full_caption", "number_only"}:
            raise RuntimeError(
                f"Frame-by-frame Caption loss_mode invalid: {loss_mode!r}"
            )
        # Save and validate the prompt actually used for frame-wise supervision.
        if (
            prompt_source not in {"none", "fixed"}
            or (prompt_source == "none" and prompt != "")
            or (
                prompt_source == "fixed"
                and (not isinstance(prompt, str) or not prompt.strip())
            )
        ):
            raise RuntimeError(
                f"Frame-by-frame Caption prompt contract invalid: source={prompt_source!r}, prompt={prompt!r}"
            )
        required_target = {
            "loss_mode": loss_mode,
            "cot_version": "depth_layout_perframe_caption_microbatch",
            "prompt_source": prompt_source,
            "prompt": prompt,
            "assistant_target_field": "depth_single_frame_caption",
            "frame_supervision_scope": "own_frame_only",
            "frame_microbatch_mode": "window_to_rows",
            "frame_loss_reduction": "mean_per_source_window",
            "qwen_batch_semantics": "one_row_per_source_frame_no_cross_row_attention",
        }
        mismatches = {
            key: (target.get(key), expected)
            for key, expected in required_target.items()
            if target.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Frame-by-frame caption checkpoint training_target is incomplete or inconsistent: mismatches={mismatches}, dir={checkpoint_dir}"
            )
        initialization = metadata.get("initialization")
        policy = target.get("initialization_policy", "weights_only")
        if policy == "from_scratch":
            # Record random initialization explicitly when training Depth from scratch.
            if (
                not isinstance(initialization, dict)
                or initialization.get("mode") != "random"
            ):
                raise RuntimeError(
                    f"Frame-by-frame depth training from scratch initialization mismatch: initialization={initialization!r}, dir={checkpoint_dir}"
                )
        elif policy == "weights_only":
            if (
                not isinstance(initialization, dict)
                or initialization.get("mode")
                != "weights_only_fresh_optimizer_scheduler"
                or initialization.get("optimizer_loaded") is not False
                or initialization.get("scheduler_loaded") is not False
                or initialization.get("trainer_state_loaded") is not False
                or not initialization.get("source_checkpoint")
                or not initialization.get("artifacts_sha256")
            ):
                raise RuntimeError(
                    f"Frame-by-frame Caption checkpoint missing initial depth prior initialization contract: initialization={initialization!r}, dir={checkpoint_dir}"
                )
        else:
            raise RuntimeError(
                f"Unknown frame-by-frame depth initialization constraint: {policy!r}"
            )
        data_contract = metadata.get("data")
        if (
            not isinstance(data_contract, dict)
            or int(data_contract.get("source_window_rows", -1))
            != int(target.get("source_window_rows", -2))
            or int(data_contract.get("expanded_rows", -1))
            != int(target.get("expanded_rows", -2))
        ):
            raise RuntimeError(
                f"Frame-by-frame Caption checkpoint source/expanded line counts inconsistent: data={data_contract}, target={target}, dir={checkpoint_dir}"
            )
    data = metadata["data"]
    if (
        not isinstance(data, dict)
        or not data.get("paths")
        or int(data.get("training_rows", 0)) <= 0
    ):
        raise RuntimeError(
            f"Stage1 checkpoint data meta illegal: {checkpoint_dir}, data={data!r}"
        )
    hashes = metadata["artifacts_sha256"]
    if branch == "Camera":
        expected_files = ("camera_interpreter.pt", "camera_interpreter_config.json")
    else:
        interpreter_config = metadata.get("interpreter", {}).get("config", {})
        if interpreter_config.get("output_contract") == "direct_patch_embedding":
            expected_files = (
                "depth_patch_tokenizer.pt",
                "depth_patch_tokenizer_config.json",
            )
        else:
            expected_files = ("depth_interpreter.pt", "depth_interpreter_config.json")
    if set(hashes) != set(expected_files):
        raise RuntimeError(
            f"Stage1 artifact hash file set mismatch: branch={branch}, actual={sorted(hashes)}, expected={sorted(expected_files)}"
        )
    for filename in expected_files:
        path = os.path.join(checkpoint_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Stage1 checkpoint missing artifact: {path}")
        actual = _sha256_file(path)
        if str(hashes[filename]) != actual:
            raise RuntimeError(
                f"Stage1 checkpoint artifact SHA256 inconsistency: {path}, saved={hashes[filename]}, actual={actual}"
            )


def _save_processor_artifacts(processor, output_dir: str):
    """Save Stage1 processor and verify that Stage2 can inherit the same resize parameters."""
    if processor is None:
        raise RuntimeError(
            "Camera Interpreter checkpoint requires a complete processor; it cannot save only the Interpreter weights."
        )
    processor.save_pretrained(output_dir)
    preprocessor_path = os.path.join(output_dir, "preprocessor_config.json")
    if not os.path.isfile(preprocessor_path):
        # Save processor settings separately so subsequent stages can enforce the same pixel budget.
        processor_path = os.path.join(output_dir, "processor_config.json")
        if not os.path.isfile(processor_path):
            raise RuntimeError(
                f"checkpoint missing processor_config.json: {output_dir}"
            )
        with open(processor_path, encoding="utf-8") as handle:
            image_config = json.load(handle).get("image_processor")
        if not isinstance(image_config, dict):
            raise RuntimeError(
                f"processor_config.json is missing image_processor: {processor_path}"
            )
        with open(preprocessor_path, "w", encoding="utf-8") as handle:
            json.dump(image_config, handle, ensure_ascii=False, indent=2)
    required = ["preprocessor_config.json", "tokenizer_config.json"]
    missing = [
        name for name in required if not os.path.isfile(os.path.join(output_dir, name))
    ]
    if missing:
        raise RuntimeError(
            f"Camera Interpreter checkpoint is missing processor/tokenizer files: {missing}, dir={output_dir}"
        )


class CustomTrainerCameraInterpreter(Trainer):
    """Stage1 Camera Interpreter Dedicated Trainer.

    Differences from standard HF Trainer:
      - _save() only saves the Interpreter weights (~a few MB), not the entire Qwen-VL base model (~7GB)
      - _load_from_checkpoint() only restores Interpreter weights
      - log() ensures every log entry includes a step field
    """

    _DEPTH_COMPONENT_SETS = (
        ("ce_loss", "meter_loss", "meter_mae"),
        ("relation_loss", "caption_loss", "relation_eligible_fraction"),
        ("frame_caption_loss", "frame_count"),
    )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """Keep the model loss component and pass it to training_step for aggregation within the same window as total loss."""
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        self._last_depth_components = None
        if isinstance(outputs, dict):
            matched_sets = [
                names
                for names in self._DEPTH_COMPONENT_SETS
                if all(name in outputs for name in names)
            ]
            if len(matched_sets) > 1:
                raise RuntimeError(
                    f"A forward function returns multiple depth loss components simultaneously: {matched_sets}"
                )
            if matched_sets:
                self._last_depth_components = {
                    name: outputs[name].detach().float().mean()
                    for name in matched_sets[0]
                }
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """With the same total loss, micro-batch, gradient accumulation scaling, and DDP weight accumulation components."""
        self._last_depth_components = None
        loss = super().training_step(
            model,
            inputs,
            num_items_in_batch=num_items_in_batch,
        )
        components = self._last_depth_components
        if components is None:
            return loss

        # Same as current HF Trainer.training_step: gradient accumulation is only applied when the model does not receive loss kwargs.
        component_scale = 1.0
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            component_scale /= float(self.args.gradient_accumulation_steps)
        if not hasattr(self, "_depth_component_window_sums"):
            self._depth_component_window_sums = {
                name: torch.zeros_like(value) for name, value in components.items()
            }
            # Measure logging intervals from the resumed global step.
            self._depth_component_window_start_step = int(self.state.global_step)
        for name, value in components.items():
            self._depth_component_window_sums[name] += value * component_scale
        return loss

    def log(self, logs, *args, **kwargs):
        """Add CE, meter loss, and meter MAE to the total loss log in the same window."""
        logs = dict(logs)
        if "loss" in logs and hasattr(self, "_depth_component_window_sums"):
            step_count = int(self.state.global_step) - int(
                self._depth_component_window_start_step
            )
            if step_count <= 0:
                raise RuntimeError(
                    f"meter component log window has no optimizer step: start={self._depth_component_window_start_step}, current={self.state.global_step}"
                )
            for name, local_sum in self._depth_component_window_sums.items():
                # Gather scalar logging values through Accelerate across training ranks.
                gathered_sum = (
                    self.accelerator.gather(local_sum.reshape(1)).mean().item()
                )
                logs[name] = round(float(gathered_sum) / float(step_count), 4)
            del self._depth_component_window_sums
            del self._depth_component_window_start_step
        logs.setdefault("step", int(self.state.global_step))
        return super().log(logs, *args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        """Return the original random sampler on switch, or group by local length of n_images."""
        if not bool(getattr(self.args, "group_by_length", False)):
            # Match the sampler-hook signature supported by the installed Trainer.
            return (
                super()._get_train_sampler(dataset)
                if dataset is not None
                else super()._get_train_sampler()
            )
        return build_n_images_length_grouped_sampler(
            dataset if dataset is not None else self.train_dataset,
            self.args,
            log_prefix="[train_sampler]",
        )

    def _save(self, output_dir=None, state_dict=None):
        """Save checkpoint: only save Interpreter weights + processor/tokenizer + training_args.

        Save content:
          - camera_interpreter.pt — state_dict of the Interpreter module
          - camera_interpreter_config.json — Interpreter structure parameters
          - processor/tokenizer file — contains newly added special tokens and training resize scale
          - training_args.bin — Training hyperparameters
          - camera_checkpoint.json — checkpoint metadata
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Get the original model through the DDP wrapper
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        if not hasattr(model, "save_camera_interpreter"):
            raise RuntimeError(
                "model is missing save_camera_interpreter(), unable to save Camera Interpreter."
            )

        model.save_camera_interpreter(output_dir)
        # Save and validate the complete processor for consistent resizing in subsequent stages.
        _save_processor_artifacts(self.processing_class, output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        raw_stage1_meta = getattr(model, "stage1_checkpoint_meta", None)
        if not isinstance(raw_stage1_meta, dict):
            raise RuntimeError(
                "The new Stage1 checkpoint is missing model.stage1_checkpoint_meta, which prevents saving weights alone without retaining training semantics."
            )
        # JSON round-trip performs deep copy and serializability check simultaneously to prevent subsequent modifications during training from contaminating the saved meta.
        metadata = json.loads(json.dumps(raw_stage1_meta, ensure_ascii=False))
        metadata.update(
            {
                "checkpoint_type": "stage1_camera_interpreter",
                "base_model": getattr(model, "base_model_name_or_path", None),
                # Camera loss definition may alter labels; after saving, resume must be verified, not guessed by directory name.
                "camera_loss_mode": getattr(model, "camera_loss_mode", "full_caption"),
                "camera_cot_version": getattr(model, "camera_cot_version", None),
                # Depth loss definition alters labels and must be saved along with checkpoint; resume cannot rely on wrapper to guess.
                "depth_loss_mode": getattr(model, "depth_loss_mode", "full_caption"),
                "depth_cot_version": getattr(model, "depth_cot_version", None),
                # Three common factors determine forward and loss for frame-by-frame captioning; resume should not only check the full_caption name.
                "depth_frame_supervision_scope": getattr(
                    model, "depth_frame_supervision_scope", "window_all_frames"
                ),
                "depth_frame_microbatch_mode": getattr(
                    model, "depth_frame_microbatch_mode", "disabled"
                ),
                "depth_frame_loss_reduction": getattr(
                    model, "depth_frame_loss_reduction", "token_mean"
                ),
                # meter loss does not modify the Interpreter architecture, but resume must strictly reproduce the same training objective.
                "depth_meter_loss_weight": float(
                    getattr(model, "depth_meter_loss_weight", 0.0)
                ),
                "depth_meter_loss_beta_m": float(
                    getattr(model, "depth_meter_loss_beta_m", 0.1)
                ),
                # Pair Delta: The weights of the two CE groups will change the training objective; after saving, resume must be strictly equal.
                "depth_relation_loss_weight": float(
                    getattr(model, "depth_relation_loss_weight", 0.5)
                ),
                # The sampler changes the sample combination at each step; auditing resume requires saving the checkpoint to determine whether it switches from an epoch boundary.
                "train_sampler_mode": getattr(
                    self.args, "train_sampler_mode", "random"
                ),
            }
        )
        branch = metadata.get("stage1_branch")
        if branch not in {"Camera", "Depth"}:
            raise RuntimeError(f"Stage1 checkpoint branch illegal: {branch!r}")
        if not hasattr(model, "stage1_artifact_names"):
            raise RuntimeError(
                "model is missing stage1_artifact_names(), preventing strict artifact hash saving."
            )
        artifact_names = tuple(model.stage1_artifact_names())
        metadata["artifacts_sha256"] = {
            filename: _sha256_file(os.path.join(output_dir, filename))
            for filename in artifact_names
        }
        metadata["strict_reload_verified"] = False
        with open(
            os.path.join(output_dir, "camera_checkpoint.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        _validate_stage1_checkpoint_metadata(metadata, output_dir)
        # Save immediately and walk the same config + state_dict strict=True path as resume, so downstream tasks won't discover bad checkpoints later.
        model.load_camera_interpreter(output_dir)
        metadata["strict_reload_verified"] = True
        with open(
            os.path.join(output_dir, "camera_checkpoint.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        _validate_stage1_checkpoint_metadata(metadata, output_dir)
        print(
            "[stage1-checkpoint-contract] "
            f"branch={branch} strict_reload=ok artifacts={metadata['artifacts_sha256']} "
            f"dir={output_dir}",
            flush=True,
        )

    def _save_checkpoint(self, model, trial):
        """Save checkpoint and retain only the optimizer/scheduler resume state of the latest checkpoint."""
        super()._save_checkpoint(model, trial)
        if self.args.save_only_model or not self.args.should_save:
            return
        # Retain full resume state only in the latest checkpoint to limit storage use.
        run_dir = self._get_output_dir(trial=trial)
        # Sort checkpoints by step before pruning resume states.
        checkpoints = sorted(
            (
                path
                for path in glob.glob(os.path.join(run_dir, "checkpoint-*"))
                if os.path.isdir(path)
                and os.path.basename(path).split("checkpoint-", 1)[-1].isdigit()
            ),
            key=lambda path: int(os.path.basename(path).split("checkpoint-", 1)[-1]),
        )
        if len(checkpoints) <= 1:
            return
        for checkpoint_dir in checkpoints[:-1]:
            self._drop_resume_state_files(checkpoint_dir)

    @staticmethod
    def _drop_resume_state_files(checkpoint_dir: str):
        """Remove resume states while retaining evaluable Reader weights."""
        exact_files = [
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        glob_patterns = [
            "rng_state*.pth",
        ]
        for filename in exact_files:
            path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(path):
                os.remove(path)
        for pattern in glob_patterns:
            for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
                if os.path.isfile(path):
                    os.remove(path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore Interpreter weights from checkpoint (for resuming training)."""
        model = model if model is not None else self.model
        while hasattr(model, "module"):
            model = model.module
        # Restore the model training mode after reloading Reader weights.
        was_training = model.training
        if not hasattr(model, "load_camera_interpreter"):
            raise RuntimeError(
                "model is missing load_camera_interpreter(), unable to resume Camera Interpreter."
            )
        self._validate_train_sampler_resume(resume_from_checkpoint)
        self._validate_camera_training_metadata(resume_from_checkpoint, model)
        self._validate_depth_training_metadata(resume_from_checkpoint, model)
        model.load_camera_interpreter(resume_from_checkpoint)

        # Resume training mode
        if was_training:
            model.train()

    def _validate_train_sampler_resume(self, checkpoint_dir: str) -> None:
        """Random/length_grouped can only be explicitly switched at integer epoch checkpoints."""
        metadata_path = os.path.join(checkpoint_dir, "camera_checkpoint.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Stage1 resume checkpoint missing training metadata: {metadata_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        saved_mode = str(metadata.get("train_sampler_mode", "random"))
        expected_mode = str(getattr(self.args, "train_sampler_mode", "random"))
        allowed_modes = {"random", "length_grouped"}
        if saved_mode not in allowed_modes or expected_mode not in allowed_modes:
            raise RuntimeError(
                f"Stage1 sampler mode illegal: saved={saved_mode!r}, expected={expected_mode!r}, path={metadata_path}"
            )
        if saved_mode == expected_mode:
            return

        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.isfile(trainer_state_path):
            raise FileNotFoundError(
                f"Switching Stage1 sampler requires verifying epoch boundaries in trainer_state.json: {trainer_state_path}"
            )
        with open(trainer_state_path, "r", encoding="utf-8") as handle:
            trainer_state = json.load(handle)
        epoch = trainer_state.get("epoch")
        if epoch is None or abs(float(epoch) - round(float(epoch))) > 1e-9:
            raise RuntimeError(
                f"Can only switch Stage1 sampler at integer epoch checkpoints to avoid mid-epoch skip ordering issues: saved={saved_mode!r}, expected={expected_mode!r}, epoch={epoch!r}, path={trainer_state_path}"
            )
        print(
            "[train_sampler_resume] epoch-boundary transition "
            f"{saved_mode}->{expected_mode}: epoch={float(epoch):.0f}, checkpoint={checkpoint_dir}"
        )

    @staticmethod
    def _validate_camera_training_metadata(checkpoint_dir: str, model) -> None:
        """Validate camera loss and description settings before resuming training."""
        if getattr(model, "camera_interpreter", None) is None:
            return
        metadata_path = os.path.join(checkpoint_dir, "camera_checkpoint.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Camera resume checkpoint missing training metadata: {metadata_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "stage1_meta_schema_version" in metadata:
            _validate_stage1_checkpoint_metadata(metadata, checkpoint_dir)
            if metadata.get("strict_reload_verified") is not True:
                raise RuntimeError(
                    f"Camera resume checkpoint failed strict reload after save: {metadata_path}"
                )
        expected_loss_mode = str(getattr(model, "camera_loss_mode", "full_caption"))
        expected_cot_version = getattr(model, "camera_cot_version", None)
        saved_loss_mode = metadata.get("camera_loss_mode")
        saved_cot_version = metadata.get("camera_cot_version")
        if saved_loss_mode is None:
            if expected_loss_mode != "full_caption":
                raise RuntimeError(
                    f"Checkpoint must specify camera_loss_mode to resume with motion_content: path={metadata_path}, expected={expected_loss_mode!r}"
                )
            print(
                "[camera_resume] missing camera_loss_mode "
                f"-> full_caption: {metadata_path}"
            )
            return
        if str(saved_loss_mode) != expected_loss_mode:
            raise RuntimeError(
                f"Camera resume loss mode inconsistent: saved={saved_loss_mode!r}, expected={expected_loss_mode!r}, path={metadata_path}"
            )
        if saved_cot_version != expected_cot_version:
            raise RuntimeError(
                f"Camera resume cot_version inconsistent: saved={saved_cot_version!r}, expected={expected_cot_version!r}, path={metadata_path}"
            )

    @staticmethod
    def _validate_depth_training_metadata(checkpoint_dir: str, model) -> None:
        """Validate depth loss and description settings before resuming training."""
        if (
            getattr(model, "depth_interpreter", None) is None
            and getattr(model, "depth_patch_tokenizer", None) is None
        ):
            return
        metadata_path = os.path.join(checkpoint_dir, "camera_checkpoint.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Depth resume checkpoint missing training metadata: {metadata_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "stage1_meta_schema_version" in metadata:
            _validate_stage1_checkpoint_metadata(metadata, checkpoint_dir)
            if metadata.get("strict_reload_verified") is not True:
                raise RuntimeError(
                    f"Depth resume checkpoint failed to pass strict reload after saving: {metadata_path}"
                )
        expected_loss_mode = str(getattr(model, "depth_loss_mode", "full_caption"))
        expected_cot_version = getattr(model, "depth_cot_version", None)
        saved_loss_mode = metadata.get("depth_loss_mode")
        saved_cot_version = metadata.get("depth_cot_version")
        if saved_loss_mode is None:
            if expected_loss_mode != "full_caption":
                raise RuntimeError(
                    f"Checkpoint must specify depth_loss_mode to resume with number_only: path={metadata_path}, expected={expected_loss_mode!r}"
                )
            print(
                "[depth_resume] missing depth_loss_mode "
                f"-> full_caption: {metadata_path}"
            )
            return
        if str(saved_loss_mode) != expected_loss_mode:
            raise RuntimeError(
                f"Depth resume loss mode inconsistency: saved={saved_loss_mode!r}, expected={expected_loss_mode!r}, path={metadata_path}"
            )
        if saved_cot_version != expected_cot_version:
            raise RuntimeError(
                f"Depth resume cot_version inconsistency: saved={saved_cot_version!r}, expected={expected_cot_version!r}, path={metadata_path}"
            )
        expected_frame_contract = (
            str(getattr(model, "depth_frame_supervision_scope", "window_all_frames")),
            str(getattr(model, "depth_frame_microbatch_mode", "disabled")),
            str(getattr(model, "depth_frame_loss_reduction", "token_mean")),
        )
        saved_frame_contract = (
            str(metadata.get("depth_frame_supervision_scope", "window_all_frames")),
            str(metadata.get("depth_frame_microbatch_mode", "disabled")),
            str(metadata.get("depth_frame_loss_reduction", "token_mean")),
        )
        if saved_frame_contract != expected_frame_contract:
            raise RuntimeError(
                f"Depth resume frame supervision/microbatch/reduction inconsistency: saved={saved_frame_contract}, expected={expected_frame_contract}, path={metadata_path}"
            )
        expected_relation_weight = float(
            getattr(model, "depth_relation_loss_weight", 0.5)
        )
        saved_relation_weight = metadata.get("depth_relation_loss_weight", 0.5)
        if float(saved_relation_weight) != expected_relation_weight:
            raise RuntimeError(
                f"Depth resume relation loss weight inconsistency: saved={saved_relation_weight}, expected={expected_relation_weight}, path={metadata_path}"
            )
        expected_meter_weight = float(getattr(model, "depth_meter_loss_weight", 0.0))
        expected_meter_beta = float(getattr(model, "depth_meter_loss_beta_m", 0.1))
        saved_meter_weight = metadata.get("depth_meter_loss_weight")
        saved_meter_beta = metadata.get("depth_meter_loss_beta_m")
        if saved_meter_weight is None:
            # A missing depth_meter_loss_weight explicitly maps to zero.
            if expected_meter_weight != 0.0:
                raise RuntimeError(
                    f"Checkpoint must specify depth_meter_loss_weight to resume with meter loss: path={metadata_path}, expected={expected_meter_weight}"
                )
            print(f"[depth_resume] missing meter fields -> weight=0.0: {metadata_path}")
            return
        try:
            normalized_saved_weight = float(saved_meter_weight)
            normalized_saved_beta = float(saved_meter_beta)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Depth resume meter metadata is not a valid floating-point number: weight={saved_meter_weight!r}, beta={saved_meter_beta!r}, path={metadata_path}"
            ) from error
        if normalized_saved_weight != expected_meter_weight:
            raise RuntimeError(
                f"Depth resume meter weight inconsistency: saved={normalized_saved_weight}, expected={expected_meter_weight}, path={metadata_path}"
            )
        if normalized_saved_beta != expected_meter_beta:
            raise RuntimeError(
                f"Depth resume meter beta inconsistent: saved={normalized_saved_beta}, expected={expected_meter_beta}, path={metadata_path}"
            )
