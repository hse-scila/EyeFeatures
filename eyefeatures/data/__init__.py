"""
Simple data loading utilities for the eye-tracking collection.
"""

from eyefeatures.data.utils import (
    DEFAULT_COLLECTION_DIR,
    get_labels,
    get_meta,
    get_pk,
    list_datasets,
    load_dataset,
)

__all__ = [
    "DEFAULT_COLLECTION_DIR",
    "list_datasets",
    "load_dataset",
    "get_pk",
    "get_labels",
    "get_meta",
]
