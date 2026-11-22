"""Retain evaluation weights for saved checkpoints and resume state for the latest one."""

import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable, List, Tuple

from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerCallback


RESUME_STATE_FILE_PATTERNS = (
    TRAINER_STATE_NAME,
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "rng_state*.pt",
    "lora_adapter",
)


def hf_save_only_model_for_latest_resume(
    save_only_model_value, log_prefix: str = "[ckpt-prune]"
) -> bool:
    """Save complete Trainer state before applying shared checkpoint retention."""
    if isinstance(save_only_model_value, str):
        requested_model_only = save_only_model_value == "1"
    else:
        requested_model_only = bool(save_only_model_value)
    if requested_model_only:
        # Prune completed checkpoints while retaining the latest optimizer, scheduler, and RNG state.
        print(
            f"{log_prefix} SAVE_ONLY_MODEL=1 prunes completed checkpoints; "
            "HF TrainingArguments.save_only_model is forced to False so the latest checkpoint can resume."
        )
    return False


def _list_step_checkpoints(output_dir: str) -> List[Tuple[int, Path]]:
    """List the checkpoint-* directories in output_dir in ascending order of step."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def remove_resume_state_files(checkpoint_dir: str) -> int:
    """Delete the resume state file within a single checkpoint, retaining only the model/processor/meta files for eval use."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise RuntimeError(
            f"Checkpoint directory does not exist, cannot clear resume state: {root}"
        )
    removed = 0
    for pattern in RESUME_STATE_FILE_PATTERNS:
        for path_str in glob.glob(str(root / pattern)):
            path = Path(path_str)
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                shutil.rmtree(path)
                removed += 1
    return removed


def prune_old_resume_states(
    output_dir: str, keep_latest_n: int = 1, log_prefix: str = "[ckpt-prune]"
) -> None:
    """Only retain resume state for the latest N checkpoints; when N=0, only eval files are retained."""
    keep_latest_n = int(keep_latest_n)
    if keep_latest_n < 0:
        raise RuntimeError(f"keep_latest_n must be >= 0, current={keep_latest_n}")
    checkpoints = _list_step_checkpoints(output_dir)
    if len(checkpoints) <= keep_latest_n:
        return
    # keep_latest_n=0 selects every checkpoint for resume-state cleanup.
    old_checkpoints = (
        checkpoints if keep_latest_n == 0 else checkpoints[:-keep_latest_n]
    )
    removed_total = 0
    stripped = 0
    for _, checkpoint_dir in old_checkpoints:
        removed = remove_resume_state_files(str(checkpoint_dir))
        if removed > 0:
            stripped += 1
            removed_total += removed
    if removed_total > 0:
        latest_steps = (
            []
            if keep_latest_n == 0
            else [str(step) for step, _ in checkpoints[-keep_latest_n:]]
        )
        print(
            f"{log_prefix} old_eval_checkpoints={stripped} removed_resume_files={removed_total} "
            f"kept_latest_steps={','.join(latest_steps) if latest_steps else 'none'}"
        )


def checkpoint_has_resume_state(checkpoint_dir: str) -> bool:
    """Determine whether the checkpoint contains the minimum state required for continued training: trainer_state + optimizer + scheduler."""
    root = Path(checkpoint_dir)
    has_trainer_state = (root / TRAINER_STATE_NAME).is_file()
    has_optimizer = (root / "optimizer.pt").is_file() or (
        root / "optimizer.bin"
    ).is_file()
    has_scheduler = (root / "scheduler.pt").is_file()
    if not bool(has_trainer_state and has_optimizer and has_scheduler):
        return False

    lora_meta_path = root / "lora_meta.json"
    if lora_meta_path.is_file():
        meta = json.loads(lora_meta_path.read_text(encoding="utf-8"))
        if bool(meta.get("use_lora")) and not (root / "lora_adapter").is_dir():
            return False
    return True


def load_qwen_model_state_from_checkpoint(
    qwen_model, checkpoint_dir: str, log_prefix: str = "[stage2]"
) -> None:
    """Restore Qwen weights from Stage2 checkpoint; supports single file and sharded safetensors."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise RuntimeError(
            f"Checkpoint directory does not exist, cannot restore model weights: {root}"
        )

    if (root / "model.safetensors.index.json").is_file() or (
        root / "pytorch_model.bin.index.json"
    ).is_file():
        # Preserve every weight shard and its index so saved checkpoints remain loadable.
        from transformers.modeling_utils import load_sharded_checkpoint

        load_sharded_checkpoint(qwen_model, str(root), strict=False, prefer_safe=True)
        print(f"{log_prefix} LLM weights restored from sharded checkpoint: {root}")
        return

    ckpt_path = root / "model.safetensors"
    if not ckpt_path.exists():
        ckpt_path = root / "pytorch_model.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find LLM checkpoint weights: {root}")

    from transformers.modeling_utils import load_state_dict

    state = load_state_dict(str(ckpt_path))
    qwen_model.load_state_dict(state, strict=False)
    print(f"{log_prefix} Loaded LLM weights from checkpoint: {ckpt_path}")


class KeepLatestResumeStateCallback(TrainerCallback):
    """Retain resume state only for the latest checkpoint after each save."""

    def __init__(self, keep_latest_n: int = 1, log_prefix: str = "[ckpt-prune]"):
        keep_latest_n = int(keep_latest_n)
        if keep_latest_n <= 0:
            raise RuntimeError(f"keep_latest_n must be > 0, current={keep_latest_n}")
        self.keep_latest_n = keep_latest_n
        self.log_prefix = log_prefix

    def on_save(self, args, state, control, **kwargs):
        # Only allow the should_save process to clean up the shared directory to prevent multiple GPUs from deleting the same batch of files simultaneously.
        if not bool(getattr(args, "should_save", False)):
            return control
        prune_old_resume_states(
            output_dir=args.output_dir,
            keep_latest_n=self.keep_latest_n,
            log_prefix=self.log_prefix,
        )
        return control


class DropAllResumeStateCallback(TrainerCallback):
    """After each save, delete optimizer/scheduler/trainer_state, leaving only the model and eval files."""

    def __init__(self, log_prefix: str = "[ckpt-drop-resume]"):
        self.log_prefix = log_prefix

    def on_save(self, args, state, control, **kwargs):
        # Only allow the should_save process to clean up the shared checkpoint to prevent multiple GPUs from deleting the same batch of files simultaneously.
        if not bool(getattr(args, "should_save", False)):
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        removed = remove_resume_state_files(str(checkpoint_dir))
        print(
            f"{self.log_prefix} checkpoint={checkpoint_dir} removed_resume_files={removed}",
            flush=True,
        )
        return control


def save_final_checkpoint_with_latest_resume_state(
    trainer,
    validate_checkpoint: Callable[[str], None],
    keep_latest_n: int = 1,
    log_prefix: str = "[stage2]",
) -> None:
    """Save final checkpoint; when save_only_model=0, the final checkpoint can also be used for resuming."""
    global_step = int(trainer.state.global_step)
    if global_step <= 0:
        raise RuntimeError(
            f"Invalid global_step at training end, cannot save final checkpoint: {global_step}"
        )

    final_checkpoint_dir = Path(trainer.args.output_dir) / f"checkpoint-{global_step}"
    require_resume_state = not bool(getattr(trainer.args, "save_only_model", False))
    should_resave = True
    if final_checkpoint_dir.is_dir():
        try:
            validate_checkpoint(str(final_checkpoint_dir))
        except RuntimeError as exc:
            print(
                f"{log_prefix} Final checkpoint already exists but is incomplete; will be saved again: {final_checkpoint_dir}; reason={exc}"
            )
        else:
            if require_resume_state and not checkpoint_has_resume_state(
                str(final_checkpoint_dir)
            ):
                print(
                    f"{log_prefix} Final checkpoint is missing resume state and will be saved again: {final_checkpoint_dir}"
                )
            else:
                print(
                    f"{log_prefix} final checkpoint already exists: {final_checkpoint_dir}"
                )
                should_resave = False

    if should_resave:
        if require_resume_state:
            # Use the Trainer checkpoint path for the final save so training can resume.
            trainer._save_checkpoint(trainer.model, trial=None)
        else:
            trainer.save_model(str(final_checkpoint_dir))
        validate_checkpoint(str(final_checkpoint_dir))
        print(f"{log_prefix} saved final checkpoint: {final_checkpoint_dir}")

    prune_old_resume_states(
        output_dir=trainer.args.output_dir,
        keep_latest_n=keep_latest_n,
        log_prefix="[ckpt-prune]",
    )
