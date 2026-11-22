"""Stage2 QA data reading and single-sample batch assembly."""

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from georeader_data import bundle_root, resolve_records

from coordinate_rewriter import (
    copy_coordinate_rewrite_metadata,
    rewrite_sample_coordinates_in_texts,
)
from main_camera_interpreter import (
    CAMERA_PAD,
    DEPTH_END,
    DEPTH_PAD,
    DEPTH_START,
    _assert_pil_image_size,
    _assemble_camera_prompt,
    _assert_depth_token_layout,
    _build_camera_token_block,
    _build_depth_token_block,
    _build_labels_after_assistant,
    _coerce_modality_flag,
    _geometry_prompt_token_count,
    _get_record_image_sizes,
    _load_camera_pose_from_path,
    _load_depth_maps_from_path,
    _resolve_record_depth_path,
    _resolve_record_pose_path,
    _should_insert_camera_tokens,
    _single_token_id,
)
from spar_marker_utils import (
    draw_spar_marker_specs,
    get_optional_spar_marker_specs,
    validate_spar_marker_specs,
)


def _has_nonempty_think(text: str) -> bool:
    """Determine if the assistant target contains a non-empty Think, for checkpoint data contract."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    return match is not None and bool(match.group(1).strip())


from stage2_config import (
    MISSING_POSE_TOKEN_POLICY_CHOICES,
    MISSING_POSE_TOKEN_POLICY_DROP,
    MISSING_POSE_TOKEN_POLICY_KEEP_ZERO_DELTA,
    OUTPUT_GATE_TYPE_CHOICES,
    OUTPUT_GATE_TYPE_FIXED_SCALE,
    OUTPUT_GATE_TYPE_SCALAR,
    OUTPUT_GATE_TYPE_SPATIAL_ADAPTER_FAMILY,
    SPATIAL_FUSION_TYPE_CHOICES,
    SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TOKENS,
    SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS,
    SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS_PERFRAME_READER,
    SPATIAL_FUSION_TYPE_IMAGE_RESIDUAL,
    SPATIAL_FUSION_TYPE_DEPTH_PATCH_RESIDUAL,
    SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_AFTER_QUESTION,
    SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION,
    SPATIAL_FUSION_TYPE_LATENT_TOKENS,
    SPATIAL_TOKEN_POSITION_CHOICES,
)

# These categories have completed source-level full quantitative acceptance; other sources still perform complete depth value range checks each time they are read.
PUBLISHED_DEPTH_SOURCES = frozenset(
    {"scannet", "scannetpp", "arkitscenes", "mp3d", "sunrgbd"}
)


def _load_json_or_jsonl(path: str) -> List[dict]:
    """Read QA and automatically parse the release package path without modifying the downloaded files."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]
    if not isinstance(records, list):
        raise RuntimeError(f"The data file must be a JSON array or JSONL: {path}")
    return resolve_records(records, path)


def _assign_static_depth_dropout(
    records: List[dict],
    probability: float,
    seed: int,
) -> tuple[int, int]:
    """Select a fixed proportion of Depth-off samples stably based on sample_id."""
    probability = float(probability)
    if not math.isfinite(probability) or probability < 0.0 or probability >= 1.0:
        raise RuntimeError(
            f"depth_modality_dropout_probability must be in [0,1), current={probability}"
        )
    for record in records:
        record["_depth_modality_keep"] = 1
    if probability == 0.0:
        return 0, sum(int(record["_has_depth"]) for record in records)

    eligible = [record for record in records if int(record["_has_depth"]) == 1]
    sample_ids = []
    for record in eligible:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise RuntimeError(
                f"When Depth modality dropout is enabled, every eligible sample must have a non-empty sample_id: idx={record.get('idx')!r}"
            )
        sample_ids.append(sample_id)
    duplicate_ids = [
        sample_id for sample_id, count in Counter(sample_ids).items() if count != 1
    ]
    if duplicate_ids:
        raise RuntimeError(
            f"Depth modality dropout requires sample_id to be globally unique; duplicate found: count={len(duplicate_ids)}, examples={sorted(duplicate_ids)[:5]}"
        )

    # Key change: Sorting depends only on seed and sample_id, not on worker, rank, epoch, or sample reading order.
    # Therefore, DDP, resume, and two epochs will reuse the same mask for the same sample.
    ranked = sorted(
        eligible,
        key=lambda record: (
            hashlib.sha256(
                f"{int(seed)}\0{record['sample_id']}".encode("utf-8")
            ).digest(),
            record["sample_id"],
        ),
    )
    dropped_count = int(math.floor(len(ranked) * probability + 0.5))
    for record in ranked[:dropped_count]:
        record["_depth_modality_keep"] = 0
    return dropped_count, len(ranked)


def _build_text_span_mask(
    input_ids: torch.Tensor,
    tokenizer,
    rendered_text: str,
    target_text: str,
    *,
    context: str,
) -> torch.Tensor:
    """Annotate the rendered text with the unique target token(s) precisely according to tokenizer offset."""
    target_text = str(target_text)
    if not target_text.strip():
        raise RuntimeError(f"Target body is empty, cannot construct mask: {context}")
    starts = []
    search_from = 0
    while True:
        start = rendered_text.find(target_text, search_from)
        if start < 0:
            break
        starts.append(start)
        search_from = start + max(1, len(target_text))
    if len(starts) != 1:
        raise RuntimeError(
            f"The target text must appear exactly once in rendered text: count={len(starts)}, {context}"
        )
    char_start = starts[0]
    char_end = char_start + len(target_text)
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            f"Tokenizer does not support offset_mapping, cannot construct question_mask: {context}"
        )
    if input_ids.dim() != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError(
            f"text span mask only supports input_ids in range [1,L], current={tuple(input_ids.shape)}"
        )
    if token_ids != input_ids[0].detach().cpu().tolist():
        raise RuntimeError(
            f"Inconsistent input_ids between processor/tokenizer, unable to safely construct question_mask: {context}"
        )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_idx, (start, end) in enumerate(offsets):
        # The offset of special tokens is often (0,0) and should not be mistakenly included in the question.
        if int(end) > int(start) and int(start) < char_end and int(end) > char_start:
            mask[0, token_idx] = True
    if not bool(mask.any().item()):
        raise RuntimeError(f"question_mask is empty: {context}")
    return mask


def _should_keep_camera_token_slot(
    has_usable_pose: bool,
    camera_latent_size: int,
    missing_pose_token_policy: str,
) -> bool:
    """Determine whether to retain the Camera block in the prompt; when pose is missing, only retain an empty slot, not invoking the Interpreter."""
    if missing_pose_token_policy not in MISSING_POSE_TOKEN_POLICY_CHOICES:
        raise RuntimeError(
            f"missing_pose_token_policy illegal: {missing_pose_token_policy!r}"
        )
    if bool(has_usable_pose):
        if int(camera_latent_size) <= 0:
            raise RuntimeError(
                "When poses are available, camera_latent_size must be > 0."
            )
        return True
    return bool(
        int(camera_latent_size) > 0
        and missing_pose_token_policy == MISSING_POSE_TOKEN_POLICY_KEEP_ZERO_DELTA
    )


def _pose_cache_key(pose_path: str) -> str:
    """Generate training pose cache key; the full file path is the unique identity, no longer concatenating directory numbers."""
    return os.path.abspath(os.path.expanduser(str(pose_path)))


def _rgb_migration_paths(record: dict, image_paths, *, context: str):
    """Read declared image sequences for QA records with an explicit migration contract."""
    provenance = record.get("geometry_provenance") or {}
    if not isinstance(provenance, dict):
        raise RuntimeError(f"geometry_provenance is not an object: {context}")
    contract = provenance.get("rgb_migration_contract")
    if contract not in {
        "qa_path_changed_npz_original_audit",
        "qa_path_changed_npz_original_audit_unmarked",
    }:
        return None, None
    original = provenance.get("original_image_input")
    geometry = provenance.get("geometry_image_input")
    qa_paths = provenance.get("qa_rgb_paths")
    if (
        not isinstance(original, list)
        or not isinstance(geometry, list)
        or not isinstance(qa_paths, list)
        or qa_paths != list(image_paths)
        or len(original) != len(image_paths)
        or len(geometry) != len(image_paths)
    ):
        raise RuntimeError(
            f"Image lists in geometry_provenance do not match the QA images: {context}"
        )
    if (
        contract == "qa_path_changed_npz_original_audit_unmarked"
        and geometry != qa_paths
    ):
        raise RuntimeError(
            f"Geometry image list must match the QA image list for this sample: {context}"
        )
    return tuple(original), tuple(geometry)


def _build_pose_cache(
    pose_requests: List[tuple],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Preload the camera poses used in the training set to avoid repeatedly opening small npz files per step."""
    specs_by_key = {}
    for (
        pose_path,
        expected_views,
        image_paths,
        image_sizes,
        frame_ids,
        original_paths,
        geometry_paths,
        unmarked,
    ) in pose_requests:
        key = _pose_cache_key(pose_path)
        spec = (
            int(expected_views),
            tuple(image_paths),
            tuple(image_sizes),
            tuple(frame_ids),
            tuple(original_paths) if original_paths is not None else None,
            tuple(geometry_paths) if geometry_paths is not None else None,
            bool(unmarked),
        )
        old_spec = specs_by_key.get(key)
        if old_spec is not None and old_spec != spec:
            raise RuntimeError(
                f"The same pose_path is reused with different RGB orders/frames, cannot safely cache: key={key}, existing={old_spec}, new={spec}"
            )
        specs_by_key[key] = spec

    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    sorted_keys = sorted(specs_by_key.keys())
    rank = int(os.environ.get("RANK", "0"))
    # Only allow rank0 to display tqdm, to avoid 8 cards outputting 8 progress bars simultaneously, which would mess up train.log; other ranks still normally build cache.
    iterator = tqdm(
        sorted_keys,
        desc="[dataset] cache camera pose",
        dynamic_ncols=True,
        mininterval=5,
        disable=(rank != 0),
    )
    for pose_path in iterator:
        (
            expected_views,
            image_paths,
            image_sizes,
            frame_ids,
            original_paths,
            geometry_paths,
            unmarked,
        ) = specs_by_key[pose_path]
        cache[pose_path] = _load_camera_pose_from_path(
            pose_path,
            expected_views=expected_views,
            expected_image_paths=image_paths,
            expected_image_sizes=image_sizes or None,
            expected_frame_ids=frame_ids or None,
            expected_original_image_paths=original_paths,
            expected_geometry_image_paths=geometry_paths,
            unmarked_rgb_migration=unmarked,
        )
    return cache


class MindCubeQADataset(Dataset):
    """Read Stage2 QA; Depth-only mode ignores all Camera fields in the JSONL."""

    def __init__(
        self,
        data_path: List[str],
        image_folder: str,
        max_samples: Optional[int] = None,
        expected_image_size: Optional[tuple] = (480, 640),
        draw_spar_marker_specs: bool = False,
        pose_cache: bool = True,
        depth_only: bool = False,
        depth_modality_dropout_probability: float = 0.0,
        depth_modality_dropout_seed: int = 42,
    ):
        if not data_path:
            raise RuntimeError("--qa_data_path must specify at least one JSONL file.")
        self.qa_data_paths = [os.path.abspath(os.path.expanduser(p)) for p in data_path]
        for p in self.qa_data_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"QA data file not found: {p}")
        self.data_path = ",".join(self.qa_data_paths)
        self.image_folder = os.path.abspath(os.path.expanduser(image_folder))
        self.expected_image_size = expected_image_size
        self.draw_spar_marker_specs = bool(draw_spar_marker_specs)
        self.depth_only = bool(depth_only)
        # Depth-only prohibits opening camera.npz; even if the caller accidentally enables POSE_CACHE, it must be forcibly disabled here.
        self.pose_cache_enabled = bool(pose_cache) and not self.depth_only

        kept = []
        pose_requests: List[
            Tuple[
                str, int, tuple[str, ...], tuple[tuple[int, int], ...], tuple[int, ...]
            ]
        ] = []
        pose_rows_per_dir = Counter()
        spar_marker_kind_stats = Counter()
        spar_marker_task_stats = Counter()
        spar_marker_rows = 0
        spar_marker_specs = 0
        total_rows = 0
        rows_per_file = {}
        no_pose_rows_per_file = Counter()
        pose_rows_per_file = Counter()
        depth_rows_per_file = Counter()
        modality_rows = Counter()
        modality_rows_per_file: Dict[str, Counter] = {}
        ignored_pose_rows = 0
        coldstart_cot_rows = 0
        for qa_path in self.qa_data_paths:
            records = _load_json_or_jsonl(qa_path)
            # After publishing the package, the accepted geometry also changes its root directory; retain the original training's duplicate check strategy.
            data_root = bundle_root(qa_path)
            published_geometry_root = (
                data_root / "train/stage2/qa_geometry"
                if data_root is not None
                else None
            )
            rows_per_file[qa_path] = len(records)
            modality_rows_per_file[qa_path] = Counter()
            total_rows += len(records)
            for source_row_idx, record in enumerate(records):
                row_label = f"path={qa_path}, row={source_row_idx}"
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"Samples must be dict: {row_label}, type={type(record)}"
                    )
                for field in ("text_input", "image_input", "text_output", "idx"):
                    if field not in record:
                        raise RuntimeError(
                            f"Sample missing field {field!r}: {row_label}, keys={sorted(record.keys())}"
                        )
                if (
                    not isinstance(record["image_input"], list)
                    or not record["image_input"]
                ):
                    raise RuntimeError(
                        f"image_input must be a non-empty list: {row_label}, idx={record.get('idx')}"
                    )
                n_images = len(record["image_input"])
                if "n_images" in record and int(record["n_images"]) != n_images:
                    raise RuntimeError(
                        f"n_images and image_input count do not match: {row_label}, n_images={record['n_images']}, image_input={n_images}"
                    )
                record_image_sizes = _get_record_image_sizes(
                    record, row_idx=source_row_idx, required=False
                )
                if record_image_sizes is not None and len(record_image_sizes) != len(
                    record["image_input"]
                ):
                    raise RuntimeError(
                        f"image_sizes and image_input quantities do not match: {row_label}, idx={record.get('idx')}, image_sizes={len(record_image_sizes)}, image_input={len(record['image_input'])}"
                    )
                if not str(record["text_output"]).strip():
                    raise RuntimeError(
                        f"text_output cannot be empty: {row_label}, idx={record.get('idx')}"
                    )
                if self.draw_spar_marker_specs:
                    marker_specs = validate_spar_marker_specs(
                        record, row_label=row_label, n_images=n_images
                    )
                else:
                    marker_specs = get_optional_spar_marker_specs(record)
                if marker_specs:
                    spar_marker_rows += 1
                    spar_marker_specs += len(marker_specs)
                    if self.draw_spar_marker_specs:
                        spar_marker_kind_stats.update(
                            str(spec["kind"]) for spec in marker_specs
                        )
                    spar_marker_task_stats[
                        str(
                            record.get(
                                "task_family", record.get("source_dataset", "unknown")
                            )
                        )
                    ] += 1

                context = f"Stage2 sample {row_label}, idx={record.get('idx')}"
                if self.depth_only:
                    # 0716 Same JSONL file contains both pose/depth; C line only reads depth, cannot parse or open pose_path.
                    if "has_pose" in record:
                        _coerce_modality_flag(
                            record["has_pose"], field_name="has_pose", context=context
                        )
                    if record.get("pose_path") is not None or bool(
                        record.get("has_pose", 0)
                    ):
                        ignored_pose_rows += 1
                    pose_path = None
                else:
                    pose_path = _resolve_record_pose_path(record, context=context)
                depth_path = _resolve_record_depth_path(record, context=context)
                has_pose = pose_path is not None
                has_depth = depth_path is not None
                if self.depth_only and not has_depth:
                    raise RuntimeError(
                        f"Depth-only training requires each line to have has_depth=1 and depth_path to exist: {context}"
                    )
                # Joint training retains two modalities independently per QA; four-cell statistics prevent misclassifying depth-only rows as dual-modality rows.
                if has_pose and has_depth:
                    modality_name = "both"
                elif has_pose:
                    modality_name = "camera_only"
                elif has_depth:
                    modality_name = "depth_only"
                else:
                    modality_name = "neither"
                modality_rows[modality_name] += 1
                modality_rows_per_file[qa_path][modality_name] += 1
                if has_pose and n_images <= 1:
                    raise RuntimeError(
                        f"Single-image Stage2 samples cannot write pose_path: {context}"
                    )
                resolved_image_paths = tuple(
                    self._resolve_image_path(str(path))
                    for path in record["image_input"]
                )
                original_paths, geometry_paths = _rgb_migration_paths(
                    record,
                    resolved_image_paths,
                    context=context,
                )
                frame_ids = tuple(
                    int(item) for item in record.get("window_frame_ids", [])
                )
                if frame_ids and len(frame_ids) != n_images:
                    raise RuntimeError(
                        f"window_frame_ids and image_input quantities do not match: {context}"
                    )
                image_sizes_tuple = tuple(record_image_sizes or ())
                if has_pose:
                    # Validate pose shape and image order during dataset initialization using pose_path.
                    unmarked = (record.get("geometry_provenance") or {}).get(
                        "rgb_migration_contract"
                    ) == "qa_path_changed_npz_original_audit_unmarked"
                    request = (
                        pose_path,
                        n_images,
                        resolved_image_paths,
                        image_sizes_tuple,
                        frame_ids,
                        original_paths,
                        geometry_paths,
                        unmarked,
                    )
                    if self.pose_cache_enabled:
                        pose_requests.append(request)
                    else:
                        _load_camera_pose_from_path(
                            pose_path,
                            n_images,
                            expected_image_paths=resolved_image_paths,
                            expected_image_sizes=record_image_sizes,
                            expected_frame_ids=frame_ids or None,
                            expected_original_image_paths=original_paths,
                            expected_geometry_image_paths=geometry_paths,
                            unmarked_rgb_migration=unmarked,
                        )
                    pose_rows_per_dir[str(Path(pose_path).parent.parent)] += 1
                    pose_rows_per_file[qa_path] += 1
                else:
                    no_pose_rows_per_file[qa_path] += 1
                if has_depth:
                    # Even if the current model is not enabled for Depth, first verify that the path written in the JSONL file exists; bad paths cannot silently become no-depth.
                    depth_rows_per_file[qa_path] += 1
                kept_record = dict(record)
                kept_record["_qa_data_path"] = qa_path
                kept_record["_pose_path"] = pose_path
                kept_record["_depth_path"] = depth_path
                provenance = record.get("geometry_provenance") or {}
                source = (
                    provenance.get("rgb_source")
                    if isinstance(provenance, dict)
                    else None
                )
                # Identify published geometry sources from their package-relative locations.
                if (
                    has_depth
                    and published_geometry_root is not None
                    and Path(depth_path).is_relative_to(published_geometry_root)
                ):
                    source = (
                        Path(depth_path).relative_to(published_geometry_root).parts[0]
                    )
                # Reuse source-level validation only for published geometry in the expected package directory.
                kept_record["_depth_prevalidated"] = bool(
                    has_depth
                    and source in PUBLISHED_DEPTH_SOURCES
                    and published_geometry_root is not None
                    and Path(depth_path).is_relative_to(
                        published_geometry_root / source
                    )
                )
                kept_record["_has_pose"] = int(has_pose)
                kept_record["_has_depth"] = int(has_depth)
                # The sampler cannot call __getitem__ to load images; write back the number of verified images into the lightweight records for grouping use.
                kept_record["n_images"] = n_images
                kept.append(kept_record)
                if _has_nonempty_think(str(record["text_output"])):
                    coldstart_cot_rows += 1
                if max_samples is not None and len(kept) >= int(max_samples):
                    break
            if max_samples is not None and len(kept) >= int(max_samples):
                break
        if not kept:
            raise RuntimeError(f"No trainable samples: {self.data_path}")
        self.records = kept
        self.depth_modality_dropout_probability = float(
            depth_modality_dropout_probability
        )
        self.depth_modality_dropout_seed = int(depth_modality_dropout_seed)
        (
            self.depth_modality_dropout_count,
            self.depth_modality_dropout_eligible_count,
        ) = _assign_static_depth_dropout(
            self.records,
            self.depth_modality_dropout_probability,
            self.depth_modality_dropout_seed,
        )
        self.coldstart_cot_rows = coldstart_cot_rows
        self.pose_cache = (
            _build_pose_cache(pose_requests) if self.pose_cache_enabled else {}
        )
        print(f"[dataset] qa_files={self.qa_data_paths}")
        print(f"[dataset] qa_rows_per_file={rows_per_file}")
        print(f"[dataset] qa_rows_total={total_rows}")
        print(f"[dataset] qa_rows_used={len(self.records)}")
        print(f"[dataset] coldstart_cot_rows={self.coldstart_cot_rows}")
        print(
            f"[dataset] no_pose_rows_per_file={dict(sorted(no_pose_rows_per_file.items()))}"
        )
        print(
            f"[dataset] pose_rows_per_file={dict(sorted(pose_rows_per_file.items()))}"
        )
        print(
            f"[dataset] depth_rows_per_file={dict(sorted(depth_rows_per_file.items()))}"
        )
        print(f"[dataset] modality_rows={dict(sorted(modality_rows.items()))}")
        print(
            "[dataset] modality_rows_per_file="
            f"{dict((path, dict(sorted(stats.items()))) for path, stats in modality_rows_per_file.items())}"
        )
        print(f"[dataset] pose_rows_per_dir={dict(sorted(pose_rows_per_dir.items()))}")
        print(f"[dataset] draw_spar_marker_specs={int(self.draw_spar_marker_specs)}")
        print(f"[dataset] pose_cache={int(self.pose_cache_enabled)}")
        print(f"[dataset] pose_cache_entries={len(self.pose_cache)}")
        print(f"[dataset] depth_only={int(self.depth_only)}")
        print(f"[dataset] ignored_pose_rows={ignored_pose_rows}")
        print(
            "[depth-modality-dropout-dataset] "
            f"probability={self.depth_modality_dropout_probability:.6f} "
            f"seed={self.depth_modality_dropout_seed} "
            f"dropped={self.depth_modality_dropout_count} "
            f"eligible={self.depth_modality_dropout_eligible_count} "
            "scope=sample_all_frames schedule=constant eval=off "
            "identity=sample_id_static",
            flush=True,
        )
        print(f"[dataset] spar_marker_rows={spar_marker_rows}")
        print(f"[dataset] spar_marker_specs={spar_marker_specs}")
        print(
            f"[dataset] spar_marker_kind={dict(sorted(spar_marker_kind_stats.items()))}"
        )
        print(
            f"[dataset] spar_marker_task_rows={dict(sorted(spar_marker_task_stats.items()))}"
        )

    def __len__(self):
        return len(self.records)

    def _resolve_image_path(self, raw_path: str) -> str:
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(self.image_folder, raw_path)

    def __getitem__(self, index):
        record = dict(self.records[int(index)])
        image_paths = [self._resolve_image_path(p) for p in record["image_input"]]
        record_image_sizes = _get_record_image_sizes(record, required=False)
        images = []
        for image_idx, path in enumerate(image_paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Cannot find input image: {path}")
            image = Image.open(path).convert("RGB")
            if record_image_sizes is not None:
                expected_record_size = tuple(record_image_sizes[image_idx])
                if tuple(image.size) != expected_record_size:
                    raise RuntimeError(
                        f"image_sizes metadata and actual image dimensions do not match: path={path}, actual PIL(W,H)={tuple(image.size)}, metadata={expected_record_size}"
                    )
            _assert_pil_image_size(
                image, path, self.expected_image_size, "Stage2 Camera QA"
            )
            images.append(image)
        marker_specs = record.get("spar_marker_specs", [])
        if self.draw_spar_marker_specs and marker_specs:
            # Key point: In SPAR with-marker JSONL, the image_input still points to the original image;
            # The training loader draws markers online here, while the eval entry does not redraw them.
            draw_spar_marker_specs(
                images,
                marker_specs,
                sample_id=str(record.get("sample_id", record.get("idx"))),
            )
        n_images = len(image_paths)
        result = {
            "idx": int(record["idx"]),
            "n_images": n_images,
            "images": images,
            "image_input": image_paths,
            "text_input": str(record["text_input"]).strip(),
            "text_output": str(record["text_output"]).strip(),
            "has_pose": int(record["_has_pose"]),
            "has_depth": int(record["_has_depth"]),
            "depth_modality_keep": int(record["_depth_modality_keep"]),
        }
        original_paths, geometry_paths = _rgb_migration_paths(
            record,
            image_paths,
            context=f"Stage2 idx={record['idx']}",
        )
        if original_paths is not None:
            result["rgb_migration_original_image_input"] = original_paths
            result["rgb_migration_geometry_image_input"] = geometry_paths
            result["rgb_migration_unmarked"] = (
                record["geometry_provenance"]["rgb_migration_contract"]
                == "qa_path_changed_npz_original_audit_unmarked"
            )
        # Preserve task metadata so coordinate rewriting uses the correct image and coordinate space.
        copy_coordinate_rewrite_metadata(record, result)
        if record_image_sizes is not None:
            result["image_sizes"] = record_image_sizes
        if record["_pose_path"] is not None:
            result["pose_path"] = str(record["_pose_path"])
        if record["_depth_path"] is not None:
            result["depth_path"] = str(record["_depth_path"])
            result["depth_prevalidated"] = bool(record["_depth_prevalidated"])
        if "window_frame_ids" in record:
            result["window_frame_ids"] = [
                int(item) for item in record["window_frame_ids"]
            ]
        if result["has_pose"]:
            if self.pose_cache_enabled:
                cache_key = _pose_cache_key(result["pose_path"])
                if cache_key not in self.pose_cache:
                    raise RuntimeError(f"pose cache missing samples: key={cache_key}")
                # The cache only stores read-only pose tensors; the collator does not modify them in place, and the DataLoader will handle moving them to the GPU subsequently.
                result["camera"] = self.pose_cache[cache_key]
        return result


class CameraStage2Collator:
    def __init__(
        self,
        processor,
        camera_latent_size: int,
        spatial_token_position: str = "before_question",
        use_depth_interpreter: bool = False,
        depth_latent_size: int = 8,
        camera_token_layout: str = "global",
        depth_token_layout: str = "global",
        missing_pose_token_policy: str = MISSING_POSE_TOKEN_POLICY_DROP,
        output_gate_type: str = OUTPUT_GATE_TYPE_SCALAR,
        spatial_fusion_type: str = SPATIAL_FUSION_TYPE_LATENT_TOKENS,
        enable_thinking: Optional[bool] = None,
    ):
        """Initialize Stage2 collator; read max_pixels from image_processor for online coordinate scaling."""
        self.processor = processor
        # Qwen3.5 native templates default to enabling thinking; only new backbones explicitly pass False, while older models retain their original behavior.
        self.enable_thinking = enable_thinking
        self.camera_latent_size = int(camera_latent_size)
        self.spatial_token_position = str(spatial_token_position)
        if self.spatial_token_position not in SPATIAL_TOKEN_POSITION_CHOICES:
            raise RuntimeError(
                f"spatial_token_position illegal: {self.spatial_token_position!r}, choices={SPATIAL_TOKEN_POSITION_CHOICES}"
            )
        self.missing_pose_token_policy = str(missing_pose_token_policy)
        if self.missing_pose_token_policy not in MISSING_POSE_TOKEN_POLICY_CHOICES:
            raise RuntimeError(
                f"missing_pose_token_policy illegal: {self.missing_pose_token_policy!r}"
            )
        self.output_gate_type = str(output_gate_type)
        if self.output_gate_type not in OUTPUT_GATE_TYPE_CHOICES:
            raise RuntimeError(f"output_gate_type illegal: {self.output_gate_type!r}")
        self.spatial_fusion_type = str(spatial_fusion_type)
        if self.spatial_fusion_type not in SPATIAL_FUSION_TYPE_CHOICES:
            raise RuntimeError(
                f"spatial_fusion_type illegal: {self.spatial_fusion_type!r}"
            )
        self.image_residual = (
            self.spatial_fusion_type == SPATIAL_FUSION_TYPE_IMAGE_RESIDUAL
        )
        self.depth_patch_residual = (
            self.spatial_fusion_type == SPATIAL_FUSION_TYPE_DEPTH_PATCH_RESIDUAL
        )
        # Direct Patch: Two groups only change token order; each <|depth_pad|> is replaced in the model by the full tokenizer embedding frozen for the tokenizer.
        self.direct_patch_depth = self.spatial_fusion_type in {
            SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION,
            SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_AFTER_QUESTION,
        }
        self.direct_embedding_tokens = self.spatial_fusion_type in {
            SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TOKENS,
            SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS,
            SPATIAL_FUSION_TYPE_DIRECT_EMBEDDING_TWOPASS_PERFRAME_READER,
        }
        # Image Residual uses the RGB-only tokenizer without additional geometry tokens.
        self.camera_token_id = (
            -1
            if self.image_residual
            else _single_token_id(processor.tokenizer, CAMERA_PAD)
        )
        self.image_token_id = _single_token_id(processor.tokenizer, "<|image_pad|>")
        self.use_depth_interpreter = bool(use_depth_interpreter)
        self.use_depth_geometry = bool(
            self.use_depth_interpreter or self.image_residual
        )
        self.depth_latent_size = int(depth_latent_size)
        self.camera_token_layout = str(camera_token_layout)
        self.depth_token_layout = str(depth_token_layout)
        for name, value in (
            ("camera_token_layout", self.camera_token_layout),
            ("depth_token_layout", self.depth_token_layout),
        ):
            if value not in ("global", "per_frame"):
                raise RuntimeError(
                    f"{name} Must be global/per_frame, current={value!r}"
                )
        self.depth_token_id = (
            _single_token_id(processor.tokenizer, DEPTH_PAD)
            if self.use_depth_interpreter
            and not self.image_residual
            and not self.depth_patch_residual
            else -1
        )
        self.depth_start_id = (
            _single_token_id(processor.tokenizer, DEPTH_START)
            if self.direct_patch_depth
            else -1
        )
        self.depth_end_id = (
            _single_token_id(processor.tokenizer, DEPTH_END)
            if self.direct_patch_depth
            else -1
        )
        self._logged_modality_patterns = set()
        if processor.tokenizer.pad_token_id is None:
            raise RuntimeError("processor.tokenizer.pad_token_id is empty.")
        self.pad_token_id = int(processor.tokenizer.pad_token_id)
        self.answer_start_pattern = torch.tensor(
            processor.tokenizer("<|im_start|>assistant", add_special_tokens=False)[
                "input_ids"
            ],
            dtype=torch.long,
        )
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError(
                "processor is missing image_processor, Stage2 cannot process images."
            )
        self.image_processor_merge_size = int(
            getattr(
                image_processor,
                "merge_size",
                getattr(image_processor, "spatial_merge_size", 2),
            )
        )

    def _expand_image_tokens(self, text: str, image_grid_thw: torch.Tensor) -> str:
        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        merge_length = int(self.image_processor_merge_size) ** 2
        expanded = str(text)
        for idx, thw in enumerate(image_grid_thw):
            if image_token not in expanded:
                raise RuntimeError(
                    f"Stage2 chat template has fewer image tokens than the number of images: missing index={idx}, token={image_token!r}"
                )
            repeat = int(thw.prod().item()) // merge_length
            if repeat <= 0:
                raise RuntimeError(
                    f"Stage2 image_grid_thw derived image token count is invalid: thw={thw.tolist()}, repeat={repeat}"
                )
            expanded = expanded.replace(image_token, "<|placeholder|>" * repeat, 1)
        if image_token in expanded:
            raise RuntimeError(
                f"Stage2 chat template has more image tokens than the number of images: token={image_token!r}"
            )
        return expanded.replace("<|placeholder|>", image_token)

    def _process_images_only(self, images: List[Image.Image]) -> tuple:
        """Process each image through image_processor, returning (pixel_values_list, image_grid_thw).

        Do not perform tokenization — coordinate rewriting must occur before tokenization, so first extract image_grid_thw separately.
        """
        if not images:
            raise RuntimeError("Stage2 training must include images.")
        pixel_values = []
        grids = []
        for image_idx, image in enumerate(images):
            # Process image by image: In Ego3D samples, different cameras have varying widths and heights, so they cannot be passed to the Qwen processor at once.
            image_inputs = self.processor.image_processor(
                images=[image], return_tensors="pt"
            )
            if (
                "pixel_values" not in image_inputs
                or "image_grid_thw" not in image_inputs
            ):
                raise RuntimeError(
                    f"Stage2 processor.image_processor output missing field: image_idx={image_idx}, keys={list(image_inputs.keys())}"
                )
            grid = image_inputs["image_grid_thw"]
            if tuple(grid.shape) != (1, 3):
                raise RuntimeError(
                    f"Stage2 single image image_grid_thw shape should be (1,3): image_idx={image_idx}, shape={tuple(grid.shape)}"
                )
            pixel_values.append(image_inputs["pixel_values"])
            grids.append(grid)
        image_grid_thw = torch.cat(grids, dim=0)
        return pixel_values, image_grid_thw

    def _build_batch_from_processed(
        self,
        text: str,
        pixel_values_list: list,
        image_grid_thw: torch.Tensor,
        question_text: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Construct the final batch using preprocessed image data; coordinates have been rewritten prior to calling."""
        expanded_text = self._expand_image_tokens(text, image_grid_thw)
        batch = self.processor.tokenizer(
            text=[expanded_text], return_tensors="pt", padding=True
        )
        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["image_grid_thw"] = image_grid_thw
        self._add_context_masks(batch, expanded_text, question_text=question_text)
        return batch

    def _add_context_masks(
        self,
        batch: Dict[str, torch.Tensor],
        expanded_text: str,
        question_text: Optional[str] = None,
    ) -> None:
        """Add a unified question/image mask to the eval batch for training and reusing visual cache."""
        batch["image_mask"] = batch["input_ids"] == self.image_token_id
        if not bool(batch["image_mask"].any().item()):
            raise RuntimeError("Stage2 batch has an empty image_mask.")
        if question_text is not None:
            batch["question_mask"] = _build_text_span_mask(
                batch["input_ids"],
                self.processor.tokenizer,
                expanded_text,
                question_text,
                context="Stage2 rendered prompt",
            )

    def _process_images_individually(
        self,
        text: str,
        images: List[Image.Image],
        question_text: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Batch processed images using an already constructed evaluation prompt."""
        # Evaluation can batch an already constructed prompt without training-specific rewriting.
        pixel_values_list, image_grid_thw = self._process_images_only(images)
        return self._build_batch_from_processed(
            text,
            pixel_values_list,
            image_grid_thw,
            question_text=question_text,
        )

    def __call__(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        """Assemble a single QA sample into a model batch, with coordinates precisely rewritten from image_grid_thw."""
        if len(examples) != 1:
            raise RuntimeError(
                f"Stage 2 v1 only supports per_device_train_batch_size=1, current={len(examples)}"
            )
        ex = examples[0]

        # Step 1: Process each image individually to obtain the precise image_grid_thw
        pixel_values_list, image_grid_thw = self._process_images_only(ex["images"])

        # Rewrite coordinates using the processor grid and the image selected by the task.
        rewritten_texts = rewrite_sample_coordinates_in_texts(
            ex,
            {"text_input": ex["text_input"], "text_output": ex["text_output"]},
            ex["images"],
            image_grid_thw,
            context=f"Stage2 batch idx={ex.get('idx')}",
        )
        text_input = rewritten_texts["text_input"]
        text_output = rewritten_texts["text_output"]

        has_pose = _coerce_modality_flag(
            ex.get("has_pose"),
            field_name="has_pose",
            context=f"Stage2 batch idx={ex.get('idx')}",
        )
        has_usable_pose = _should_insert_camera_tokens(
            has_pose,
            int(ex["n_images"]),
            context=f"Stage2 batch idx={ex.get('idx')}",
        )
        if self.image_residual:
            # Image Residual uses no geometry-token slots.
            use_camera_tokens = False
        else:
            # When D1 has no pose, reserve Kc camera slots, but subsequent pose reading or Camera Interpreter invocation will not occur.
            use_camera_tokens = _should_keep_camera_token_slot(
                has_usable_pose,
                self.camera_latent_size,
                self.missing_pose_token_policy,
            )

        # Step 3: Build messages → apply_chat_template → tokenize
        # camera tokens: Inserted only when the JSONL file contains a valid pose_path (has_pose=1).
        # depth tokens: Insert if depth.npz exists (even for single images with depth information)
        num_frames = int(ex["n_images"])
        camera_tokens = _build_camera_token_block(
            self.camera_latent_size,
            use_camera_tokens,
            token_layout=self.camera_token_layout,
            num_frames=num_frames,
        )
        source_has_depth = _coerce_modality_flag(
            ex.get("has_depth"),
            field_name="has_depth",
            context=f"Stage2 batch idx={ex.get('idx')}",
        )
        has_depth = bool(self.use_depth_geometry and source_has_depth)
        depth_path = None
        if has_depth:
            depth_path = ex.get("depth_path")
            if depth_path is None:
                raise RuntimeError(
                    f"has_depth=1 but batch missing depth_path: idx={ex.get('idx')}"
                )
        # Spatial Router retains two stable slot blocks; global uses a fixed K, per-frame uses current frame × K.
        # Keep null slots when missing Depth, but availability must remain 0.
        keep_depth_token_slot = bool(
            self.use_depth_interpreter
            and not self.image_residual
            and not self.depth_patch_residual
            # Direct Patch depth slots are constructed from the RGB patch grid.
            and not self.direct_patch_depth
            and (
                has_depth
                or self.output_gate_type in OUTPUT_GATE_TYPE_SPATIAL_ADAPTER_FAMILY
            )
        )
        if keep_depth_token_slot:
            depth_tokens = _build_depth_token_block(
                self.depth_latent_size,
                True,
                token_layout=self.depth_token_layout,
                num_frames=num_frames,
            )
            camera_tokens = camera_tokens + depth_tokens
        direct_patch_token_count = 0
        if self.direct_patch_depth:
            if not has_depth:
                raise RuntimeError(
                    "Direct Patch Depth training data must have has_depth=1 on every row."
                )
            # Remove summary placeholders before constructing grid-aligned Direct Patch positions.
            text_input = (
                text_input.replace(DEPTH_START, "")
                .replace(DEPTH_PAD, "")
                .replace(DEPTH_END, "")
            )
            # Frame counts must come from the actual output of this processor, not guessed from original image dimensions or fixed patch numbers.
            direct_patch_token_count = sum(
                int(row.prod().item()) // int(self.image_processor_merge_size) ** 2
                for row in image_grid_thw
            )
            depth_tokens = _build_depth_token_block(
                depth_latent_size=1,
                use_depth_tokens=True,
                patch_token_count=direct_patch_token_count,
            )
            if (
                self.spatial_fusion_type
                == SPATIAL_FUSION_TYPE_DIRECT_PATCH_DEPTH_TWOPASS_BEFORE_QUESTION
            ):
                # A: [RGB] [Depth patches] [Question] [Camera] [Answer]。
                camera_prefix, prompt_text = (
                    None,
                    f"{depth_tokens}\n{text_input}\n{camera_tokens}",
                )
            else:
                # B: [RGB] [Question] [Camera] [Depth patches] [Answer]。
                camera_prefix, prompt_text = (
                    None,
                    f"{text_input}\n{camera_tokens}\n{depth_tokens}",
                )
        else:
            camera_prefix, prompt_text = _assemble_camera_prompt(
                camera_tokens,
                text_input,
                self.spatial_token_position,
            )
        user_content = []
        if camera_prefix:
            user_content.append({"type": "text", "text": camera_prefix})
        user_content.extend({"type": "image", "image": image} for image in ex["images"])
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": text_output}]},
        ]
        template_kwargs = (
            {}
            if self.enable_thinking is None
            else {"enable_thinking": self.enable_thinking}
        )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, **template_kwargs
        )
        rendered_depth_pad_count = (
            int(text.count(DEPTH_PAD)) if self.direct_patch_depth else 0
        )
        batch = self._build_batch_from_processed(
            text,
            pixel_values_list,
            image_grid_thw,
            # E1's fixed fusion does not construct question_mask to prevent accidentally reconnecting it to the H36 conditional chain in the future.
            question_text=(
                None
                if self.output_gate_type == OUTPUT_GATE_TYPE_FIXED_SCALE
                else text_input
            ),
        )

        if has_usable_pose:
            if "camera" in ex:
                camera = ex["camera"]
            else:
                camera = _load_camera_pose_from_path(
                    ex["pose_path"],
                    expected_views=int(ex["n_images"]),
                    expected_image_paths=ex.get("image_input"),
                    expected_image_sizes=ex.get("image_sizes"),
                    expected_frame_ids=ex.get("window_frame_ids"),
                    expected_original_image_paths=ex.get(
                        "rgb_migration_original_image_input"
                    ),
                    expected_geometry_image_paths=ex.get(
                        "rgb_migration_geometry_image_input"
                    ),
                    unmarked_rgb_migration=ex.get("rgb_migration_unmarked", False),
                )
            batch.update(camera)
        else:
            n = int(ex["n_images"])
            batch["extrinsics"] = torch.zeros(1, n, 3, 4)
            batch["intrinsics"] = torch.zeros(1, n, 3, 3)
            # depth-only / single-image samples also require real RGB width and height for checking the depth/RGB aspect ratio; zero-filled placeholders are no longer allowed.
            batch["image_sizes"] = torch.tensor(
                [[int(image.width), int(image.height)] for image in ex["images"]],
                dtype=torch.float32,
            )
        if has_depth:
            depth_maps = _load_depth_maps_from_path(
                depth_path,
                int(ex["n_images"]),
                image_sizes=batch.get("image_sizes"),
                expected_image_paths=ex.get("image_input"),
                expected_frame_ids=ex.get("window_frame_ids"),
                expected_original_image_paths=ex.get(
                    "rgb_migration_original_image_input"
                ),
                expected_geometry_image_paths=ex.get(
                    "rgb_migration_geometry_image_input"
                ),
                check_distribution=not ex.get("depth_prevalidated", False),
            )
            batch["depth_maps"] = depth_maps
        batch["has_pose"] = torch.tensor([has_pose], dtype=torch.long)
        batch["has_depth"] = torch.tensor([int(has_depth)], dtype=torch.long)
        batch["depth_modality_keep"] = torch.tensor(
            [int(ex["depth_modality_keep"])], dtype=torch.long
        )

        # Print the true token count when each of the four-panel modalities first appears; also intercept slot missing during the collator stage.
        camera_pad_count = int(
            (batch["input_ids"] == self.camera_token_id).sum().item()
        )
        depth_pad_count = (
            int((batch["input_ids"] == self.depth_token_id).sum().item())
            if self.use_depth_interpreter and self.depth_token_id >= 0
            else 0
        )
        expected_camera_count = (
            _geometry_prompt_token_count(
                self.camera_latent_size,
                self.camera_token_layout,
                num_frames,
            )
            if use_camera_tokens
            else 0
        )
        expected_depth_count = (
            direct_patch_token_count
            if self.direct_patch_depth
            else (
                _geometry_prompt_token_count(
                    self.depth_latent_size,
                    self.depth_token_layout,
                    num_frames,
                )
                if keep_depth_token_slot
                else 0
            )
        )
        if (
            camera_pad_count != expected_camera_count
            or depth_pad_count != expected_depth_count
        ):
            raise RuntimeError(
                f"Stage2 modality token count anomaly: idx={ex.get('idx')}, camera={camera_pad_count}/{expected_camera_count}, depth={depth_pad_count}/{expected_depth_count}, rendered_depth={rendered_depth_pad_count}/{expected_depth_count}"
            )
        if self.direct_patch_depth:
            _assert_depth_token_layout(
                batch["input_ids"],
                depth_start_id=self.depth_start_id,
                depth_pad_id=self.depth_token_id,
                depth_end_id=self.depth_end_id,
                expected_pad_count=direct_patch_token_count,
                context=f"Direct Patch Stage2 idx={ex.get('idx')}",
            )
        modality_pattern = (int(has_pose), int(has_depth))
        if modality_pattern not in self._logged_modality_patterns:
            self._logged_modality_patterns.add(modality_pattern)
            print(
                f"[batch-modality] idx={ex.get('idx')} has_pose={int(has_pose)} "
                f"has_depth={int(has_depth)} camera_pad_tokens={camera_pad_count} "
                f"depth_pad_tokens={depth_pad_count}",
                flush=True,
            )

        batch["labels"] = _build_labels_after_assistant(
            batch["input_ids"],
            self.answer_start_pattern,
            pad_token_id=self.pad_token_id,
            image_token_id=self.image_token_id,
            camera_token_id=self.camera_token_id,
        )
        if (
            self.use_depth_interpreter
            and not self.image_residual
            and self.depth_token_id >= 0
        ):
            depth_mask = batch["input_ids"] == self.depth_token_id
            batch["labels"][depth_mask] = -100
        if not bool((batch["labels"] != -100).any().item()):
            raise RuntimeError(f"No supervised token for sample: idx={ex['idx']}")
        if int(batch["image_grid_thw"].shape[0]) != int(ex["n_images"]):
            raise RuntimeError(
                f"processor image_grid_thw count and n_images are inconsistent: idx={ex['idx']}, grid={tuple(batch['image_grid_thw'].shape)}, n_images={ex['n_images']}"
            )
        batch["idx"] = torch.tensor([int(ex["idx"])], dtype=torch.long)
        batch["n_images"] = torch.tensor([int(ex["n_images"])], dtype=torch.long)
        return batch
