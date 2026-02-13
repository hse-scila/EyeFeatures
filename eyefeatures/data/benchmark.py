"""
Simple data loading utilities for the eye-tracking benchmark.

Benchmark data lives in this repo at ``data/benchmark`` as Parquet files
(tracked with Git LFS). You can pass a custom path or use the default.

Column conventions:
- Primary key (pk): columns starting with ``group_``
- Labels: columns ending with ``_label``
- Meta: columns starting with ``meta_``
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

#: Default root directory for benchmark Parquet files (``data/benchmark`` in the repo, Git LFS).
DEFAULT_BENCHMARK_DIR = Path("data/benchmark")


def _classify_dataset_type(dataset_name: str) -> str:
    """Classify dataset type by name suffix: 'gaze' or 'fixation'."""
    if dataset_name.endswith("_gaze") or dataset_name.endswith("_gazes"):
        return "gaze"
    if dataset_name.endswith("_fixations") or dataset_name.endswith("_fixation"):
        return "fixation"
    # Default: treat as fixation
    return "fixation"


def list_datasets(
    benchmark_dir: str | Path | None = None,
    *,
    include_extensive_collection: bool = True,
    extensive_collection_only: bool = False,
    dataset_type: str | None = None,
) -> list[str]:
    """List available dataset names in the benchmark directory.

    Parameters
    ----------
    benchmark_dir : path, optional
        Root directory containing benchmark Parquet files.
        Defaults to ``data/benchmark`` (repo data tracked with Git LFS).
    include_extensive_collection : bool, default True
        If True, also search in extensive_collection subfolder.
        Ignored when extensive_collection_only is True.
    extensive_collection_only : bool, default False
        If True, list only datasets from extensive_collection subfolder
        (main directory is not scanned).
    dataset_type : str, optional
        If "gaze", return only gaze datasets (names ending with _gaze/_gazes).
        If "fixation", return only fixation datasets (names ending with
        _fixations/_fixation or default). If None, return all.

    Returns
    -------
    list of str
        Sorted list of dataset names (without .parquet extension).
    """
    benchmark_path = (
        Path(benchmark_dir) if benchmark_dir is not None else DEFAULT_BENCHMARK_DIR
    )
    dataset_names = set()

    if extensive_collection_only:
        extensive_dir = benchmark_path / "extensive_collection"
        if extensive_dir.exists():
            for f in extensive_dir.glob("*.parquet"):
                dataset_names.add(f.stem)
    else:
        for f in benchmark_path.glob("*.parquet"):
            dataset_names.add(f.stem)
        if include_extensive_collection:
            extensive_dir = benchmark_path / "extensive_collection"
            if extensive_dir.exists():
                for f in extensive_dir.glob("*.parquet"):
                    dataset_names.add(f.stem)

    if dataset_type is not None:
        dataset_names = {
            name
            for name in dataset_names
            if _classify_dataset_type(name) == dataset_type
        }

    return sorted(dataset_names)


def load_dataset(
    dataset_name: str,
    benchmark_dir: str | Path | None = None,
    *,
    normalize: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load a benchmark dataset by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. "ASD_ready_data_fixations").
        Will search for {dataset_name}.parquet in benchmark_dir.
    benchmark_dir : path, optional
        Root directory containing benchmark Parquet files.
        Defaults to ``data/benchmark`` (repo data tracked with Git LFS).
    normalize : bool, default True
        If True and dataset has unnormalized x/y columns, normalize them
        and rename to norm_pos_x/norm_pos_y.

    Returns
    -------
    tuple (DataFrame, meta_info)
        - DataFrame: loaded and optionally normalized data
        - meta_info: dict with 'pk', 'labels', 'meta' column lists and 'info'
          (from benchmark_dir/meta.json under key dataset_name, if present).
    """
    benchmark_path = (
        Path(benchmark_dir) if benchmark_dir is not None else DEFAULT_BENCHMARK_DIR
    )
    dataset_path = benchmark_path / f"{dataset_name}.parquet"

    if not dataset_path.exists():
        # Try in extensive_collection
        extensive_path = (
            benchmark_path / "extensive_collection" / f"{dataset_name}.parquet"
        )
        if extensive_path.exists():
            dataset_path = extensive_path
        else:
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found in {benchmark_path} "
                f"or {benchmark_path / 'extensive_collection'}"
            )

    df = pd.read_parquet(dataset_path)

    # Parquet preserves types; ensure numeric for x/y if present (e.g. from older exports)
    if "x" in df.columns and not pd.api.types.is_numeric_dtype(df["x"]):
        df["x"] = pd.to_numeric(
            df["x"].astype(str).str.replace(",", "."), errors="coerce"
        )
    if "y" in df.columns and not pd.api.types.is_numeric_dtype(df["y"]):
        df["y"] = pd.to_numeric(
            df["y"].astype(str).str.replace(",", "."), errors="coerce"
        )

    # Handle left/right eye columns
    if "x_left" in df.columns and "x_right" in df.columns:
        if "x" not in df.columns:
            df["x"] = (df["x_left"] + df["x_right"]) / 2
        if "y" not in df.columns:
            df["y"] = (df["y_left"] + df["y_right"]) / 2

    # Normalize if requested and needed
    if normalize and "x" in df.columns and "y" in df.columns:
        if "norm_pos_x" not in df.columns:
            max_x = df["x"].max()
            max_y = df["y"].max()
            df["norm_pos_x"] = df["x"] / max_x if max_x > 0 else df["x"]
            df["norm_pos_y"] = df["y"] / max_y if max_y > 0 else df["y"]

    # Build meta info
    meta_info = {
        "pk": get_pk(df),
        "labels": get_labels(df),
        "meta": get_meta(df),
        "info": _load_meta_info(benchmark_path, dataset_name),
    }

    return df, meta_info


def _load_meta_info(benchmark_path: Path, dataset_name: str) -> Any | None:
    """Load meta.json from benchmark dir and return value for dataset_name key."""
    meta_path = benchmark_path / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get(dataset_name)
    except (json.JSONDecodeError, OSError):
        return None


def get_pk(df: pd.DataFrame) -> list[str]:
    r"""Get primary key column names (columns starting with ``group\_``).

    Parameters
    ----------
    df : DataFrame
        Benchmark dataset DataFrame.

    Returns
    -------
    list of str
        Primary key column names.
    """
    return [col for col in df.columns if col.startswith("group_")]


def get_labels(df: pd.DataFrame) -> list[str]:
    """Get label column names (columns ending with _label).

    Parameters
    ----------
    df : DataFrame
        Benchmark dataset DataFrame.

    Returns
    -------
    list of str
        Label column names.
    """
    return [col for col in df.columns if col.endswith("_label")]


def get_meta(df: pd.DataFrame) -> list[str]:
    r"""Get meta column names (columns starting with ``meta\_``).

    Parameters
    ----------
    df : DataFrame
        Benchmark dataset DataFrame.

    Returns
    -------
    list of str
        Meta column names.
    """
    return [col for col in df.columns if col.startswith("meta_")]
