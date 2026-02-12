"""
Simple data loading utilities for the eye-tracking benchmark.
"""

from eyefeatures.data.benchmark import (
    DEFAULT_BENCHMARK_DIR,
    get_labels,
    get_meta,
    get_pk,
    list_datasets,
    load_dataset,
)

__all__ = [
    "DEFAULT_BENCHMARK_DIR",
    "list_datasets",
    "load_dataset",
    "get_pk",
    "get_labels",
    "get_meta",
]
