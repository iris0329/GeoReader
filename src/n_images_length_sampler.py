"""Construct a multi-card length-grouped sampler based on image count, shared by Stage 1 and Stage 2."""

from transformers.trainer_pt_utils import LengthGroupedSampler


TRAIN_SAMPLER_MODE_RANDOM = "random"
TRAIN_SAMPLER_MODE_LENGTH_GROUPED = "length_grouped"
TRAIN_SAMPLER_MODE_CHOICES = (
    TRAIN_SAMPLER_MODE_RANDOM,
    TRAIN_SAMPLER_MODE_LENGTH_GROUPED,
)


def build_n_images_length_grouped_sampler(dataset, training_args, *, log_prefix: str):
    """Align multi-card samples within the same DDP micro-step according to Dataset.records[*].n_images."""
    records = getattr(dataset, "records", None)
    if not isinstance(records, list) or len(records) != len(dataset):
        raise RuntimeError(
            f"n_images length-grouped sampler requires the Dataset to expose records of equal length, records={type(records).__name__}, dataset_len={len(dataset)}"
        )

    lengths = []
    for row_idx, record in enumerate(records):
        if not isinstance(record, dict) or "n_images" not in record:
            raise RuntimeError(
                f"n_images length-grouped sampler requires each record to have n_images: row={row_idx}, type={type(record).__name__}"
            )
        n_images = int(record["n_images"])
        if n_images <= 0:
            raise RuntimeError(
                f"n_images length-grouped sampler encountered invalid image count: row={row_idx}, value={n_images}"
            )
        lengths.append(n_images)

    local_batch_size = int(training_args.train_batch_size)
    world_size = int(training_args.world_size)
    if local_batch_size <= 0 or world_size <= 0:
        raise RuntimeError(
            f"The batch/world_size for the n_images length-grouped sampler must be positive: local_batch={local_batch_size}, world_size={world_size}"
        )
    # Accelerate distributes consecutive samples across different ranks; samples are grouped by global micro-batch, and only then do consecutive indices correspond to a single forward pass.
    global_micro_batch_size = local_batch_size * world_size
    if int(training_args.process_index) == 0:
        print(
            f"{log_prefix} mode=length_grouped key=n_images rows={len(lengths)} "
            f"local_batch={local_batch_size} world_size={world_size} "
            f"global_micro_batch={global_micro_batch_size}",
            flush=True,
        )
    # HF sortish sampler only sorts within random mega-batches, without fixing the entire epoch into a curriculum from short to long images.
    return LengthGroupedSampler(global_micro_batch_size, lengths=lengths)
