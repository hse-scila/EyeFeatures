"""
Shared collection utilities using eyefeatures.data.

- Load data via eyefeatures.data.load_dataset / list_datasets
- Resolve split groups from data/collection/meta.json (labels[*].splitting_column)
- Create and apply train/val/test splits by group
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from eyefeatures.data import (
    list_datasets as _list_datasets,
    load_dataset,
)


def get_collection_dir() -> Path:
    """Return collection root"""
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "collection"


def list_datasets(
    *,
    dataset_type: str | None = "fixation",
    include_extensive_collection: bool = True,
    extracted_fixations_only: bool = False,
    extensive_collection_only: bool = False,
    subdir: str | None = None,
) -> list[str]:
    """
    List dataset names from the benchmark (Parquet). Uses eyefeatures.data.list_datasets.
    Default: fixation datasets only.
    subdir: optional subfolder of benchmark root (e.g. 'extracted_fixations').
    """
    bdir = get_collection_dir() if subdir is None else get_collection_dir() / subdir
    return _list_datasets(
        bdir,
        include_extensive_collection=include_extensive_collection,
        extensive_collection_only=extensive_collection_only,
        extracted_fixations_only=extracted_fixations_only,
        include_extracted_fixations=not extensive_collection_only,
        dataset_type=dataset_type,
    )


def load_dataset_with_meta(
    dataset_name: str,
    *,
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load one dataset and its meta using eyefeatures.data.
    Returns (df, meta_info) with meta_info['pk'], 'labels', 'meta', 'info' (from meta.json).
    """
    bdir = get_collection_dir()
    df, meta_info = load_dataset(dataset_name, collection_dir=bdir, normalize=normalize)
    df = ensure_duration(df)
    return df, meta_info


def ensure_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'duration' column if missing but timestamp_start/timestamp_end or start_time/end_time exist."""
    if "duration" in df.columns:
        return df
    df = df.copy()
    if "timestamp_start" in df.columns and "timestamp_end" in df.columns:
        df["duration"] = df["timestamp_end"] - df["timestamp_start"]
    elif "start_time" in df.columns and "end_time" in df.columns:
        df["duration"] = df["end_time"] - df["start_time"]
    return df


def col_info_from_meta(df: pd.DataFrame, meta_info: dict[str, Any]) -> dict[str, Any]:
    """Build extractor-friendly col_info from loaded df and meta_info (x, y, t, duration, pk)."""
    pk = meta_info.get("pk") or [c for c in df.columns if c.startswith("group_")]
    x_col = (
        "norm_pos_x"
        if "norm_pos_x" in df.columns
        else ("x" if "x" in df.columns else None)
    )
    y_col = (
        "norm_pos_y"
        if "norm_pos_y" in df.columns
        else ("y" if "y" in df.columns else None)
    )
    t_col = (
        "timestamp"
        if "timestamp" in df.columns
        else ("timestamp_start" if "timestamp_start" in df.columns else None)
    )
    duration_col = "duration" if "duration" in df.columns else None
    return {
        "x_col": x_col,
        "y_col": y_col,
        "t_col": t_col,
        "duration": duration_col,
        "group_cols": pk,
        "has_duration": duration_col is not None,
    }


def get_split_group_cols_from_meta(
    meta_info: dict[str, Any],
    label_col: str,
) -> list[str] | None:
    """
    Get the list of group columns to use for splitting for a given label.
    Reads from meta_info['info']['labels'][label_col]['splitting_column'].
    Meta stores a single column name; returns [splitting_column].
    Returns None if no meta or label/splitting_column missing.
    """
    info = meta_info.get("info") or {}
    labels = info.get("labels") or {}
    label_meta = labels.get(label_col)
    if not label_meta:
        return None
    col = label_meta.get("splitting_column")
    if not col:
        return None
    return [col] if isinstance(col, str) else list(col)


def create_composite_index(df: pd.DataFrame, pk_cols: list[str]) -> pd.Series:
    """Composite index from pk columns (e.g. group_subject_group_trial -> 's1_t1')."""
    if not pk_cols:
        raise ValueError("pk_cols must be non-empty")
    if len(pk_cols) == 1:
        return df[pk_cols[0]].astype(str)
    return df[pk_cols].astype(str).agg("_".join, axis=1)


def create_split_info(
    df: pd.DataFrame,
    pk_cols: list[str],
    split_group_cols: list[str],
    label_col: str | None = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Create train/val/test split info by splitting at group level, then mapping to scanpath indexes.
    split_group_cols: columns that define the split unit (e.g. [group_subject]).
    Returns dict with 'train', 'val', 'test' (lists of composite pk strings), 'split_pk', 'label_col'.
    """
    if not split_group_cols or not pk_cols:
        raise ValueError("pk_cols and split_group_cols must be non-empty")
    # Ensure stable, duplicate-free pk columns (meta can contain duplicates).
    pk_cols = list(dict.fromkeys(pk_cols))
    split_group_cols = list(dict.fromkeys(split_group_cols))
    # Group-level id
    df = df.copy()
    df["_split_group_"] = create_composite_index(df, split_group_cols)
    groups = df["_split_group_"].unique().tolist()
    # First split: test
    groups_train_val, groups_test = train_test_split(
        groups, test_size=test_size, random_state=random_state
    )
    # Second split: train / val
    val_ratio = val_size / (1 - test_size) if test_size < 1 else 0.0
    groups_train, groups_val = train_test_split(
        groups_train_val, test_size=val_ratio, random_state=random_state
    )
    # Scanpath-level composite index
    full_index = create_composite_index(df, pk_cols)
    train_idx = set(
        full_index[df["_split_group_"].isin(groups_train)].unique().tolist()
    )
    val_idx = set(full_index[df["_split_group_"].isin(groups_val)].unique().tolist())
    test_idx = set(full_index[df["_split_group_"].isin(groups_test)].unique().tolist())
    split_pk = (
        split_group_cols[0]
        if len(split_group_cols) == 1
        else "_".join(split_group_cols)
    )
    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
        "split_pk": split_pk,
        "label_col": label_col,
        "pk_cols": pk_cols,
        "split_group_cols": split_group_cols,
    }


def create_and_save_splits_for_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    meta_info: dict[str, Any],
    splits_dir: Path,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    overwrite: bool = False,
) -> tuple[Path, list[tuple[str, Path]]]:
    """
    Create per-label train/val/test split info from meta and save to splits_dir.
    Saves {dataset_name}_labels.csv (full) and {dataset_name}_{label}_split_info.json per label.
    Returns (labels_path, [(label_col, split_info_path), ...]).
    """
    pk_cols = meta_info.get("pk") or []
    label_cols = meta_info.get("labels") or []
    if not pk_cols:
        pk_cols = [c for c in df.columns if c.startswith("group_")]
    # Deduplicate pk columns (meta can contain duplicates).
    pk_cols = list(dict.fromkeys(pk_cols))
    if not pk_cols:
        raise ValueError(f"No pk columns for dataset {dataset_name}")

    info = meta_info.get("info") or {}
    meta_labels = (info.get("labels") or {}).keys()
    labels_to_use = [c for c in label_cols if c in meta_labels] or label_cols

    split_info_paths: list[tuple[str, Path]] = []
    for label_col in labels_to_use:
        split_info_path = splits_dir / f"{dataset_name}_{label_col}_split_info.json"
        if split_info_path.exists() and not overwrite:
            # If split info is cached, prefer its pk_cols so labels index matches cached split ids.
            try:
                cached = load_split_info(split_info_path)
                cached_pk_cols = cached.get("pk_cols")
                if isinstance(cached_pk_cols, list) and cached_pk_cols:
                    pk_cols = list(dict.fromkeys([str(c) for c in cached_pk_cols]))
            except Exception:
                # Fall back to meta-derived pk_cols.
                pass

        # Full labels CSV (pk + label columns) — written/updated per label so pk_cols stays consistent.
        cols_to_save = [c for c in pk_cols + label_cols if c in df.columns]
        cols_to_save = list(dict.fromkeys(cols_to_save))
        labels_df = df[cols_to_save].drop_duplicates()
        labels_df["index"] = create_composite_index(labels_df, pk_cols)
        labels_path = splits_dir / f"{dataset_name}_labels.csv"
        labels_df.to_csv(labels_path, index=False)

        # Per-label labels file so FLAML can find {dataset}_{label}_labels.csv
        if label_col in labels_df.columns:
            per_label_path = splits_dir / f"{dataset_name}_{label_col}_labels.csv"
            labels_df[["index", label_col]].drop_duplicates().to_csv(
                per_label_path, index=False
            )
        split_group_cols = get_split_group_cols_from_meta(meta_info, label_col)
        if not split_group_cols:
            split_group_cols = pk_cols
        if split_info_path.exists() and not overwrite:
            split_info = load_split_info(split_info_path)
            split_info_paths.append((label_col, split_info_path))
        else:
            split_info = create_split_info(
                df,
                pk_cols,
                split_group_cols,
                label_col=label_col,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
            )
            split_info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_info_path, "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2)
            split_info_paths.append((label_col, split_info_path))

        # Always create split label CSVs for this label_col.
        # These are used directly by downstream training scripts and are convenient for inspection.
        train_l, val_l, test_l = apply_split_to_labels(
            labels_df, split_info, index_column="index"
        )
        for name, data in (("train", train_l), ("val", val_l), ("test", test_l)):
            data.to_csv(
                splits_dir / f"{dataset_name}_{label_col}_labels_{name}.csv",
                index=False,
            )
    return labels_path, split_info_paths


def load_split_info(split_info_path: str | Path) -> dict[str, Any]:
    """Load split info from JSON."""
    with open(split_info_path, encoding="utf-8") as f:
        return json.load(f)


def save_split_info(split_info: dict[str, Any], output_path: str | Path) -> None:
    """Save split info to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, default=str)


def split_dataframe_by_split_info(
    df: pd.DataFrame,
    pk_cols: list[str],
    split_info: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a raw DataFrame into train/val/test by composite index (e.g. for distance features:
    fit on train, then transform train/val/test). Returns (train_df, val_df, test_df).
    """
    df = df.copy()
    df["_idx_"] = create_composite_index(df, pk_cols)
    train_set = set(split_info["train"])
    val_set = set(split_info["val"])
    test_set = set(split_info["test"])
    train_df = df[df["_idx_"].isin(train_set)].drop(columns=["_idx_"], errors="ignore")
    val_df = df[df["_idx_"].isin(val_set)].drop(columns=["_idx_"], errors="ignore")
    test_df = df[df["_idx_"].isin(test_set)].drop(columns=["_idx_"], errors="ignore")
    return train_df, val_df, test_df


def get_path_pk_for_split_id(
    split_id: str,
    pk_cols: list[str],
    path_pk_per_label: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Return path_pk (group columns for expected/reference path) for this split_id.
    From old notebook: PATH_PK_PER_LABEL[split_id]; default = full pk. Independent of split.
    """
    if path_pk_per_label and split_id in path_pk_per_label:
        return list(path_pk_per_label[split_id])
    return list(pk_cols)


def get_split_info_paths_for_dataset(splits_dir: Path, dataset_name: str) -> list[Path]:
    """Return list of split info JSON paths for this dataset (exact or {dataset}_*_split_info.json)."""
    exact = splits_dir / f"{dataset_name}_split_info.json"
    if exact.exists():
        return [exact]
    return sorted(splits_dir.glob(f"{dataset_name}_*_split_info.json"))


def apply_split_to_features(
    features_df: pd.DataFrame,
    split_info: dict[str, Any],
    index_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply pre-defined split to features DataFrame. split_info has 'train', 'val', 'test' lists.
    Returns (train_features, val_features, test_features).
    """
    for key in ("train", "val", "test"):
        if key not in split_info:
            raise ValueError(
                f"split_info must contain 'train', 'val', 'test'; missing '{key}'"
            )
    train_indexes = set(split_info["train"])
    val_indexes = set(split_info["val"])
    test_indexes = set(split_info["test"])
    if index_column is not None:
        if index_column not in features_df.columns:
            raise ValueError(f"Index column '{index_column}' not found")
        match_values = features_df[index_column].astype(str)
    elif "index" in features_df.columns:
        match_values = features_df["index"].astype(str)
    else:
        match_values = features_df.index.astype(str)
    train_mask = match_values.isin(train_indexes)
    val_mask = match_values.isin(val_indexes)
    test_mask = match_values.isin(test_indexes)
    n_matched = train_mask.sum() + val_mask.sum() + test_mask.sum()
    if n_matched == 0:
        raise ValueError(
            "No rows matched split indexes. "
            f"Sample split: {list(train_indexes)[:3]}, sample df: {match_values.head(3).tolist()}"
        )
    return (
        features_df[train_mask].copy(),
        features_df[val_mask].copy(),
        features_df[test_mask].copy(),
    )


def apply_split_to_labels(
    labels_df: pd.DataFrame,
    split_info: dict[str, Any],
    index_column: str = "index",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply pre-defined split to labels DataFrame. Returns (train_labels, val_labels, test_labels)."""
    if index_column not in labels_df.columns:
        raise ValueError(
            f"Index column '{index_column}' not found. Columns: {list(labels_df.columns)}"
        )
    train_indexes = set(split_info["train"])
    val_indexes = set(split_info["val"])
    test_indexes = set(split_info["test"])
    match_values = labels_df[index_column].astype(str)
    return (
        labels_df[match_values.isin(train_indexes)].copy(),
        labels_df[match_values.isin(val_indexes)].copy(),
        labels_df[match_values.isin(test_indexes)].copy(),
    )


# ---------------------------------------------------------------------------
# DL training adapters (Parquet; for run_dl_training_battery find/load_func)
# ---------------------------------------------------------------------------


def find_datasets_parquet(
    include_extensive_collection: bool = True,
    subdir: str | None = None,
    **kwargs,
) -> dict[str, list[Path]]:
    """
    Return structure {'fixation': [Path(...), ...], ...} for DL training battery.
    Paths are dummy (stem = dataset name) so load_dataset_parquet can use path.stem.
    subdir: optional subfolder of benchmark root (e.g. 'extracted_fixations').
    """
    names = list_datasets(
        dataset_type="fixation",
        include_extensive_collection=include_extensive_collection,
        subdir=subdir,
        **kwargs,
    )
    prefix = f"/x/{subdir}/" if subdir else "/x/"
    return {
        "fixation": [Path(f"{prefix}{n}.parquet") for n in names],
        "unknown": [],
        "skip": [],
        "gaze": [],
        "saccade": [],
    }


def load_dataset_parquet(
    dataset_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """
    Load one dataset by name (dataset_path.stem) using eyefeatures.data.
    Returns (df, col_info, 'fixation') for compatibility with run_dl_training_battery.
    """
    name = dataset_path.stem
    parts = getattr(dataset_path, "parts", ()) or ()
    if len(parts) >= 2 and parts[1] == "extracted_fixations":
        bdir = get_collection_dir() / "extracted_fixations"
        df, meta_info = load_dataset(name, collection_dir=bdir, normalize=True)
        df = ensure_duration(df)
    else:
        df, meta_info = load_dataset_with_meta(name)
    col_info = col_info_from_meta(df, meta_info)
    return df, col_info, "fixation"
