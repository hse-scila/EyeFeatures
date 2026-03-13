"""
Shared constants and helpers for FLAML and DL training (no cross-import between them).
"""

import pandas as pd

REGRESSION_DATASET_PREFIXES = ("Cognitive_load", "Emotions", "Surgical")

# Dataset/split names containing any of these substrings are skipped in training.
SKIP_DATASET_SUBSTRINGS = ("label_Anger",)


def get_task_type(y: pd.Series, label_name: str | None = None) -> str:
    """Determine task type (classification or regression) from target variable."""
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        n_samples = len(y)
        if pd.api.types.is_integer_dtype(y):
            if n_unique < 20 and n_unique < n_samples * 0.1:
                return "classification"
            if n_unique <= 10:
                return "classification"
        else:
            if n_unique < 10:
                return "classification"
        return "regression"
    return "classification"


def get_task_type_for_dataset_label(
    dataset_name: str,
    label_column: str,
    y: pd.Series,
) -> str:
    """
    Effective task type for a (dataset, label) pair.
    Surgical, Cognitive load, and Emotion datasets use regression for all labels
    except group_task_label, which remains classification.
    """
    if label_column == "group_task_label":
        return get_task_type(y)
    full_name = (
        dataset_name
        if (label_column and dataset_name.endswith("_" + label_column))
        else (dataset_name + "_" + label_column) if label_column else dataset_name
    )
    if any(full_name.startswith(prefix) for prefix in REGRESSION_DATASET_PREFIXES):
        return "regression"
    return get_task_type(y)
