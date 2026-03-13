"""
Split utilities for collection experiments. Thin wrapper over benchmark_utils.
"""

from .benchmark_utils import (
    apply_split_to_features,
    apply_split_to_labels,
    create_composite_index,
    get_split_info_paths_for_dataset,
    load_split_info,
    save_split_info,
)

SPLIT_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.2,
    "random_state": 42,
}

__all__ = [
    "SPLIT_CONFIG",
    "create_composite_index",
    "load_split_info",
    "save_split_info",
    "apply_split_to_features",
    "apply_split_to_labels",
    "get_split_info_paths_for_dataset",
]
