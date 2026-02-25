"""
Split utilities for benchmark experiments. Thin wrapper over benchmark_utils.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .benchmark_utils import (
    create_composite_index,
    load_split_info,
    save_split_info,
    apply_split_to_features,
    apply_split_to_labels,
    get_split_info_paths_for_dataset,
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
