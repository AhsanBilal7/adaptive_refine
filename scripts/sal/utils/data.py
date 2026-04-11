"""Dataset loading and saving utilities."""

import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def get_dataset(config):
    """Load dataset from HuggingFace hub or local path."""
    logger.info(f"Loading dataset: {config.dataset_name} split={config.dataset_split}")

    kwargs = {}
    if config.dataset_config is not None:
        kwargs["name"] = config.dataset_config

    dataset = load_dataset(config.dataset_name, split=config.dataset_split, **kwargs)

    if config.num_samples is not None:
        dataset = dataset.select(range(min(config.num_samples, len(dataset))))

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def save_dataset(dataset, config):
    """Save dataset to jsonl file."""
    if config.output_file is not None:
        dataset.to_json(config.output_file, orient="records", lines=True)
