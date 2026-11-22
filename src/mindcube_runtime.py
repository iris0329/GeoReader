"""MindCube DA3 shared root directory entry, continuing the original 3B single-stream Reader and two-pass injection."""

import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoConfig, AutoProcessor, TrainingArguments, set_seed

from checkpoint import load_georeader
from georeader_data import iter_jsonl
from inference import generate_answer
from main_camera_interpreter import (
    CAMERA_START,
    CAMERA_PAD,
    CAMERA_END,
    _load_camera_pose_from_path,
    _set_image_processor_pixel_budget,
)
from main_camera_stage2_qwen3 import (
    Qwen3CameraDepthStage2,
    Stage2Trainer,
    MetadataCallback,
)
from mindcube_answer import extract_mindcube_answer
from preparation import validate_output_dir
from stage2_dataset_collator import CameraStage2Collator, MindCubeQADataset


RECIPE = "mindcube_da3_qwen25_3b"
ROOT = Path(__file__).resolve().parents[1]


def configuration():
    """Recipes are saved in the configs directory and are not inferred from directory names for training budget."""
    return json.loads((ROOT / "configs" / RECIPE / "recipe.json").read_text())


def dataset_path(data_root, phase):
    """Accept only the official single list for this recipe; does not automatically search for other versions."""
    root = Path(data_root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "georeader-mindcube":
        raise RuntimeError(
            "The MindCube recipe needs the GeoReader-MindCube data package"
        )
    entries = [row for row in manifest["datasets"] if row["phase"] == phase]
    if len(entries) != 1:
        raise RuntimeError(
            f"{phase}: expected exactly one dataset entry in manifest.json"
        )
    path = root / entries[0]["path"]
    if not path.is_file() or not path.is_relative_to(root):
        raise FileNotFoundError(path)
    return path


def validate_base(path):
    """Before reading weights, verify the 3B structure; refuse connecting the 8B Reader to the 3B base."""
    # Use AutoConfig to normalize Qwen2.5 configuration fields.
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    text = config.text_config
    if (config.model_type, text.hidden_size, text.num_hidden_layers) != (
        "qwen2_5_vl",
        2048,
        36,
    ):
        raise RuntimeError("MindCube DA3 recipe only supports Qwen2.5-VL-3B-Instruct")


def run_stage1(args):
    """Directly reuse the original Stage1 trainer, only switching MindCube's input, prompt, and parameters."""
    if args.branch != "camera":
        raise RuntimeError("MindCube DA3 recipe has no Depth branch")
    validate_base(args.base_model)
    validate_output_dir(args.output_dir, args.resume_from_checkpoint)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "MindCube Stage 1 runs on 1 GPU with batch size 1 and gradient accumulation 4"
        )
    os.environ["WANDB_MODE"] = "offline"
    values = json.loads((ROOT / "configs" / RECIPE / "stage1_camera.json").read_text())[
        "arguments"
    ]
    source = (
        Path(args.data_file).expanduser().resolve()
        if args.data_file
        else dataset_path(args.data_root, "stage1")
    )
    if not args.smoke and sum(1 for _ in iter_jsonl(source)) != 3375:
        raise RuntimeError("MindCube Stage 1 expects 3,375 pairs")
    values.update(
        model=args.base_model,
        data_path=str(source),
        image_folder=args.data_root,
        save_model_path=args.output_dir,
        cache_dir=str(Path(args.output_dir) / "cache"),
    )
    if args.smoke:
        # Engineering verification only proceeds through the first update/log/save; the official recipe remains at 8440 updates.
        values.update(
            max_samples=2,
            max_steps=1,
            warmup_steps=0,
            logging_steps=1,
            save_steps=1,
            save_total_limit=1,
            save_only_model="1",
            skip_final_save="1",
        )
    if args.resume_from_checkpoint:
        values["resume_from_checkpoint"] = args.resume_from_checkpoint
    sys.argv = [sys.argv[0]] + [
        part for key, value in values.items() for part in (f"--{key}", str(value))
    ]
    from main_camera_interpreter import main

    return main()


def make_collator(processor):
    """One geometric input per frame per stream; no inserted untrained Depth slot."""
    return CameraStage2Collator(
        processor=processor,
        camera_latent_size=1,
        camera_token_layout="per_frame",
        spatial_token_position="after_question",
        use_depth_interpreter=False,
        missing_pose_token_policy="keep_zero_delta",
        output_gate_type="independent_sigmoid",
        spatial_fusion_type="latent_tokens",
    )


def run_stage2(args):
    """Keep the original MindCube 1 GPU(s), batch size 1 per GPU, gradient accumulation 1, 10K QA, and 10-round recipe unchanged, and keep the Six-List 8B configuration independent."""
    if args.method != "georeader" or not args.camera_reader or args.depth_reader:
        raise RuntimeError(
            "The MindCube recipe supports only --method georeader with a single --camera-reader."
        )
    validate_base(args.base_model)
    validate_output_dir(args.output_dir, args.resume_from_checkpoint)
    recipe = configuration()
    cfg = recipe["stage2"]
    # Seed model and adapter initialization before constructing the Trainer.
    set_seed(cfg["seed"])
    print(f"[initialization] seed={cfg['seed']} before_model=1", flush=True)
    if int(os.environ.get("WORLD_SIZE", "1")) != cfg["world_size"]:
        raise RuntimeError(
            "MindCube Stage 2 runs on 1 GPU with batch size 1 and gradient accumulation 1"
        )
    os.environ["WANDB_MODE"] = "offline"
    source = (
        Path(args.data_file[0]).expanduser().resolve()
        if args.data_file
        else dataset_path(args.data_root, "stage2")
    )
    records = list(iter_jsonl(source))
    if len(records) != cfg["rows"] or any(
        not row["has_pose"] or row["has_depth"] for row in records
    ):
        raise RuntimeError(
            "MindCube Stage 2 expects 10,000 QA rows with camera geometry."
        )
    if args.smoke:
        # Validate on real 2/3/4 image samples, without fabricating missing modality samples or altering answers.
        selected = [
            next(row for row in records if row["n_images"] == views)
            for views in (2, 3, 4)
        ]
        source = Path(args.output_dir).resolve() / "prepared_data/smoke.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        )
    reader = Path(args.camera_reader).resolve()
    processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
    saved = json.loads((reader / "preprocessor_config.json").read_text())
    pixels = int(saved["min_pixels"]), int(saved["max_pixels"])
    if pixels != (3136, 150528):
        raise RuntimeError(
            f"MindCube Reader pixel limits do not match the recipe: {pixels}"
        )
    _set_image_processor_pixel_budget(
        processor.image_processor, min_pixels=pixels[0], max_pixels=pixels[1]
    )
    for token in (CAMERA_START, CAMERA_PAD, CAMERA_END):
        processor.tokenizer.add_tokens(token, special_tokens=True)
    # Dataset API accepts PIL (W, H) tuples; CLI string arguments are treated as character sequences.
    dataset = MindCubeQADataset(
        data_path=[str(source)],
        image_folder=args.data_root,
        expected_image_size=(480, 640),
        draw_spar_marker_specs=False,
    )
    model = Qwen3CameraDepthStage2(
        args.base_model, processor, reader, None, recipe=RECIPE
    )
    model.qwen.config.use_cache = False
    # steps control the update budget, save_steps only controls the save frequency; Smoke overrides the first log and save.
    steps = 3 if args.smoke else cfg["max_steps"]
    save_steps = steps if args.smoke else cfg["save_steps"]
    metadata = dict(
        recipe=RECIPE,
        model_type=recipe["model_type"],
        stage2_mode="camera_only",
        fusion="two_pass_last_question_independent_sigmoid_dense_residual_add_input_layer",
        spatial_token_position="after_question",
        hidden_size=model.hidden_size,
        llm_layer_count=model.layer_count,
        camera_interpreter_config=model.camera.config.to_dict(),
        depth_interpreter_config=None,
        independent_sigmoid_gate_config=model.gate.config.to_dict(),
        spatial_adapter_config=model.camera_adapter.config.to_dict(),
        camera_token_id=model.camera_id,
        depth_token_id=-1,
        min_pixels=pixels[0],
        max_pixels=pixels[1],
        qa_files=[str(source)],
        qa_rows=len(dataset),
        camera_stage1_checkpoint=str(reader),
        depth_stage1_checkpoint=None,
        epochs=cfg["epochs"],
        max_steps=steps,
        gpu_count=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        effective_batch_size=1,
        stage2_training_setting="g1 b1a1",
        learning_rate=cfg["learning_rate"],
        warmup_steps=0 if args.smoke else cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        save_steps=save_steps,
        save_only_model=args.smoke,
        train_sampler_mode="length_grouped",
        mindcube_prompt_style="question_only",
        geometry_status="da3_training_geometry",
        initialization_seed=cfg["seed"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )
    training = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=cfg["epochs"],
        max_steps=steps,
        learning_rate=cfg["learning_rate"],
        warmup_steps=0 if args.smoke else cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        logging_steps=1 if args.smoke else cfg["logging_steps"],
        save_steps=save_steps,
        save_total_limit=1,
        save_only_model=args.smoke,
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
        data_collator=make_collator(processor),
        callbacks=[MetadataCallback(processor, metadata)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"[stage2-complete] global_step={trainer.state.global_step}", flush=True)


def predict(model, processor, record, max_new_tokens):
    """Generate with the original question-only template and retain the complete response for scoring."""
    from PIL import Image

    paths = record["image_input"]
    images = []
    try:
        for path in paths:
            with Image.open(path) as original:
                images.append(original.convert("RGB"))
        question = str(record["question"]).strip()
        slots = CAMERA_START + CAMERA_PAD * len(images) + CAMERA_END
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": question + "\n" + slots})
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = make_collator(processor)._process_images_individually(
            rendered, images, question_text=question
        )
        batch["image_sizes"] = torch.tensor(
            [[im.width, im.height] for im in images], dtype=torch.float32
        )
        batch.update(
            _load_camera_pose_from_path(
                record["pose_path"],
                expected_views=len(images),
                expected_image_paths=paths,
                expected_image_sizes=record.get("image_sizes"),
            )
        )
        device = next(model.parameters()).device
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        ids, _ = generate_answer(model, batch, True, False, max_new_tokens)
        return processor.tokenizer.decode(ids, skip_special_tokens=False)
    finally:
        for image in images:
            image.close()


def load_inputs(args):
    """Bind the recipe, checkpoint and benchmark, and reject incompatible six-benchmark or SFT models."""
    if (
        args.method != "georeader"
        or args.benchmark not in ("mindcube", "all")
        or args.max_new_tokens <= 0
    ):
        raise RuntimeError(
            "MindCube evaluation supports only --method georeader, and --max-new-tokens must be positive."
        )
    source = dataset_path(args.data_root, "eval")
    records = list(iter_jsonl(source))
    if len(records) != 1050:
        raise RuntimeError("MindCube-Tiny expects 1,050 questions.")
    metadata = json.loads((Path(args.checkpoint) / "stage2_meta.json").read_text())
    if (
        metadata.get("recipe") != RECIPE
        or metadata.get("mindcube_prompt_style") != "question_only"
    ):
        raise RuntimeError("Checkpoint recipe or MindCube prompt does not match")
    model, processor, _ = load_georeader(args.checkpoint)
    return model, processor, records


def run_infer(args):
    """Single-question inference and full evaluation use the same predict function."""
    model, processor, records = load_inputs(args)
    if not 0 <= args.index < len(records):
        raise IndexError(args.index)
    record = records[args.index]
    answer = predict(model, processor, record, args.max_new_tokens)
    print(
        json.dumps(
            dict(
                id=record["id"],
                question=record["question"],
                answer=answer,
                ground_truth=record["gt_answer"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def run_eval(args):
    """Save responses and score them with the original MindCube answer extractor; mark partial runs as non-final."""
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    model, processor, records = load_inputs(args)
    selected = records if args.max_samples is None else records[: args.max_samples]
    output.mkdir(parents=True)
    correct = 0
    started = time.time()
    with (output / "eval_results.jsonl").open("w") as handle:
        for row in selected:
            answer = predict(model, processor, row, args.max_new_tokens)
            letter, reason = extract_mindcube_answer(answer)
            good = letter is not None and letter == row["gt_answer"]
            correct += int(good)
            result = dict(
                row,
                answer=answer,
                generated=answer,
                prediction=letter,
                extract_reason=reason,
                correct=good,
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    result = dict(
        benchmark="mindcube",
        recipe=RECIPE,
        method="georeader",
        checkpoint=str(Path(args.checkpoint).resolve()),
        total=len(selected),
        correct=correct,
        accuracy=correct / len(selected),
        total_records=len(records),
        is_final=args.max_samples is None,
        max_new_tokens=args.max_new_tokens,
        elapsed_seconds=round(time.time() - started, 1),
    )
    (output / "eval_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
