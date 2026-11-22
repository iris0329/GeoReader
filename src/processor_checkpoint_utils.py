"""checkpoint processor save tool."""

from __future__ import annotations

import os


def save_processor_artifacts(
    processor, output_dir: str, *, checkpoint_label: str
) -> None:
    """Save the complete processor and enforce strict validation of the config file required for eval to read max_pixels."""
    if processor is None:
        raise RuntimeError(
            f"{checkpoint_label} Saving checkpoint requires the full processor, not just the tokenizer."
        )
    processor.save_pretrained(output_dir)
    # Require checkpoint-local processor settings to preserve the training pixel budget.
    required = ["preprocessor_config.json", "tokenizer_config.json"]
    missing = [
        name for name in required if not os.path.isfile(os.path.join(output_dir, name))
    ]
    if missing:
        raise RuntimeError(
            f"{checkpoint_label} checkpoint is missing processor/tokenizer files: {missing}, dir={output_dir}"
        )
