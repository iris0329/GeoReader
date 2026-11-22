"""Continue using the native 8B response implementation, unify data paths for single-question inference and six-ranking evaluations."""

import torch

from benchmark import (
    _load_rgb_images,
    _resolve_image_path,
    _build_user_content,
    _spar_coordinate_rewrite_plan,
    _rewrite_spar_prompt_record,
)
from main_camera_interpreter import (
    _load_camera_pose_from_path,
    _load_depth_maps_from_path,
)
from stage2_dataset_collator import CameraStage2Collator


def make_collator(processor):
    """Run inference with the geometric interface specified by the checkpoint."""
    return CameraStage2Collator(
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


def content_for_record(record, benchmark, images, paths, geometry):
    """Reuse the original evaluation problem statement, answering instructions, and coordinate processing, without mixing path migration with prompt changes."""
    return _build_user_content(
        record,
        benchmark=benchmark,
        images=images,
        image_paths=paths,
        raw_video_path=None,
        camera_latent_size=1,
        use_camera_tokens=geometry,
        input_mode="multi_image",
        vsi_video_source="frames",
        vsi_num_frames=32,
        cv_prompt_style="official",
        camera_token_position="after_question",
        depth_latent_size=1,
        use_depth_tokens=geometry,
        camera_token_layout="per_frame",
        depth_token_layout="per_frame",
        num_frames=len(images),
    )


def predict(model, processor, record, benchmark, *, method, max_new_tokens=128):
    """Both methods use their native generation entry points and return the original decoded count and actual image count."""
    raw = record.get("images") or record.get("image_input")
    paths = [_resolve_image_path(benchmark.image_root, path) for path in raw]
    images = _load_rgb_images(
        paths, benchmark_name=benchmark.name, sample_id=str(record.get("id"))
    )
    try:
        # The SPAR title must be rewritten according to the processed mesh; retain the original question for scoring, without directly modifying the record.
        prompt_record = record
        coordinate_plan = _spar_coordinate_rewrite_plan(record, benchmark)
        if coordinate_plan is not None:
            if method == "georeader":
                _, grid = make_collator(processor)._process_images_only(images)
            else:
                grid = processor.image_processor(images=images, return_tensors="pt")[
                    "image_grid_thw"
                ]
            prompt_record = _rewrite_spar_prompt_record(
                record, images, grid, rewrite_plan=coordinate_plan
            )
        content = content_for_record(
            prompt_record, benchmark, images, paths, method == "georeader"
        )
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if method == "georeader":
            collator = make_collator(processor)
            batch = collator._process_images_individually(
                rendered, images, question_text=str(prompt_record["question"]).strip()
            )
            batch["image_sizes"] = torch.tensor(
                [[image.width, image.height] for image in images], dtype=torch.float32
            )
            has_pose, has_depth = bool(record["_has_pose"]), bool(record["_has_depth"])
            if has_pose:
                batch.update(
                    _load_camera_pose_from_path(
                        record["_pose_path"],
                        expected_views=len(images),
                        expected_image_paths=paths,
                        expected_image_sizes=record.get("image_sizes"),
                        expected_frame_ids=record.get("window_frame_ids"),
                    )
                )
            if has_depth:
                batch["depth_maps"] = _load_depth_maps_from_path(
                    record["_depth_path"],
                    expected_views=len(images),
                    image_sizes=batch["image_sizes"],
                    expected_image_paths=paths,
                    expected_frame_ids=record.get("window_frame_ids"),
                )
            device = next(model.parameters()).device
            batch = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            ids, views = generate_answer(
                model, batch, bool(has_pose), bool(has_depth), max_new_tokens
            )
        elif method == "sft":
            batch = processor(images=images, text=[rendered], return_tensors="pt")
            batch = batch.to(next(model.parameters()).device)
            with torch.inference_mode():
                generated = model.generate(
                    **batch, do_sample=False, max_new_tokens=max_new_tokens
                )
            ids = generated[0, batch["input_ids"].shape[1] :]
            views = len(images)
        else:
            raise ValueError(f"Unknown method: {method}")
        extra = {}
        if coordinate_plan is not None:
            extra = {
                "model_prompt_question": str(prompt_record["question"]),
                "coordinate_rewrite_applied": True,
            }
        return processor.tokenizer.decode(ids, skip_special_tokens=False), views, extra
    finally:
        for image in images:
            image.close()


@torch.inference_mode()
def generate_answer(model, batch, has_pose: bool, has_depth: bool, max_new_tokens: int):
    """Replicate the first pass of training with Question hidden, dual-path Gate, and second pass input layer residual."""
    ids = batch["input_ids"]
    attention = batch["attention_mask"]
    grid = batch["image_grid_thw"]
    views = int(grid.shape[0])
    positions = batch["question_mask"][0].nonzero().flatten()
    if positions.numel() == 0:
        raise RuntimeError("Question token mask is empty")
    prefix_length = int(positions[-1].item()) + 1
    if bool(
        (
            (ids[:, :prefix_length] == model.camera_id)
            | (ids[:, :prefix_length] == model.depth_id)
        ).any()
    ):
        raise RuntimeError("The first prefix must not contain Camera/Depth tokens.")
    embeddings = model.qwen.model.language_model.embed_tokens(ids)
    first = model.qwen.model(
        inputs_embeds=embeddings[:, :prefix_length],
        attention_mask=attention[:, :prefix_length],
        position_ids=model._positions(
            ids[:, :prefix_length], attention[:, :prefix_length], grid
        ),
        pixel_values=batch["pixel_values"],
        image_grid_thw=grid,
        use_cache=False,
        return_dict=True,
    )
    question_hidden = first.last_hidden_state[:, -1]
    del first
    availability = torch.tensor(
        [[int(has_pose), int(has_depth)] if model.use_depth else [int(has_pose)]],
        device=ids.device,
        dtype=question_hidden.dtype,
    )
    camera_gate, depth_gate, _ = model.gate(question_hidden, availability)
    camera_delta = (
        model.camera(
            image_tokens=None,
            image_grid_thw=grid,
            extrinsics=batch["extrinsics"],
            intrinsics=batch["intrinsics"],
            image_sizes=batch["image_sizes"],
            apply_output_gate=False,
        )
        if has_pose
        else embeddings.new_zeros((1, views, model.hidden_size))
    )
    depth_delta = (
        model.depth(
            image_tokens=None,
            image_grid_thw=grid,
            depth_maps=batch["depth_maps"],
            image_sizes=batch["image_sizes"],
            apply_output_gate=False,
        )
        if has_depth
        else embeddings.new_zeros((1, views, model.hidden_size))
    )
    camera_mask, depth_mask = ids == model.camera_id, ids == model.depth_id
    if int(camera_mask.sum()) != views or int(depth_mask.sum()) != (
        views if model.use_depth else 0
    ):
        raise RuntimeError(
            f"Expected one Camera token and one Depth token per frame (V={views})"
        )
    injected = embeddings.clone()
    injected[camera_mask] = (
        injected[camera_mask]
        + (camera_gate.unsqueeze(-1) * model.camera_adapter(camera_delta)).reshape(
            views, model.hidden_size
        )
    ).to(injected.dtype)
    if model.use_depth:
        injected[depth_mask] = (
            injected[depth_mask]
            + (depth_gate.unsqueeze(-1) * model.depth_adapter(depth_delta)).reshape(
                views, model.hidden_size
            )
        ).to(injected.dtype)
    # Pass image token types so generation uses the same multimodal positions as training.
    mm_token_type_ids = (ids == model.image_id).long()
    # Second pass retains native pixel_values, aligning Qwen3's visual pathway with DeepStack and training consistency.
    generated = model.qwen.generate(
        input_ids=ids,
        inputs_embeds=injected,
        attention_mask=attention,
        pixel_values=batch["pixel_values"],
        image_grid_thw=grid,
        mm_token_type_ids=mm_token_type_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return generated[0, ids.shape[1] :], views
