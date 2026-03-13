"""
Common utilities for feature extraction notebooks. Uses eyefeatures.data and benchmark_utils only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .benchmark_utils import (
    apply_split_to_features,
    apply_split_to_labels,
    get_collection_dir,
    get_split_info_paths_for_dataset,
    load_split_info,
)


def setup_paths(
    output_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Set up paths for feature extraction.
    output_dir default 'features_output'; splits_dir default output_dir / 'splits'.
    """
    collection_path = get_collection_dir()
    if output_dir is None:
        output_dir = Path("features_output")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if splits_dir is None:
        splits_dir = output_dir / "splits"
    else:
        splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return {
        "collection_dir": collection_path,
        "output_dir": output_dir,
        "splits_dir": splits_dir,
    }


def apply_splits_and_save(
    features_df: pd.DataFrame,
    dataset_name: str,
    feature_type: str,
    paths: dict[str, Path],
    split_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply splits to features and save train/val/test files. When multiple label-based
    splits exist, saves one set per label. Optionally saves labels per split.
    """
    output_dir = paths["output_dir"]
    splits_dir = paths["splits_dir"]
    features_df = features_df.copy()
    features_df["index"] = features_df.index.astype(str)

    if split_info is not None:
        split_infos = [(None, split_info)]
    else:
        split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
        if not split_paths:
            out_path = output_dir / f"{dataset_name}_{feature_type}.csv"
            features_df.set_index("index").to_csv(out_path)
            return {
                "status": "success",
                "num_scanpaths": len(features_df),
                "num_features": features_df.shape[1] - 1,
                "output_path": str(out_path),
                "note": "No split info",
            }
        split_infos = [(p, load_split_info(p)) for p in split_paths]

    features_indexed = features_df.set_index("index")
    labels_path = splits_dir / f"{dataset_name}_labels.csv"
    labels_df = pd.read_csv(labels_path) if labels_path.exists() else None
    split_results = []

    for idx, (split_path, si) in enumerate(split_infos):
        split_id = (
            split_path.stem.replace("_split_info", "")
            if split_path is not None
            else dataset_name
        )
        n = len(split_infos)
        if n > 1:
            print(f"  Split [{idx + 1}/{n}] {split_id} (by: {si.get('split_pk', '?')})")
        train_f, val_f, test_f = apply_split_to_features(
            features_indexed, si, index_column=None
        )
        print(f"  Train {len(train_f)}, Val {len(val_f)}, Test {len(test_f)}")
        for name, data in (
            ("train", train_f),
            ("val", val_f),
            ("test", test_f),
        ):
            p = output_dir / f"{split_id}_{feature_type}_{name}.csv"
            data.to_csv(p)
        if labels_df is not None:
            train_l, val_l, test_l = apply_split_to_labels(
                labels_df, si, index_column="index"
            )
            for name, data in (("train", train_l), ("val", val_l), ("test", test_l)):
                (splits_dir / f"{split_id}_labels_{name}.csv").parent.mkdir(
                    parents=True, exist_ok=True
                )
                data.to_csv(splits_dir / f"{split_id}_labels_{name}.csv", index=False)
        split_results.append(
            {
                "split_id": split_id,
                "n_train_scanpaths": len(train_f),
                "n_val_scanpaths": len(val_f),
                "n_test_scanpaths": len(test_f),
                "num_features": train_f.shape[1],
            }
        )

    r = split_results[0]
    return {
        "status": "success",
        "n_train_scanpaths": r["n_train_scanpaths"],
        "n_val_scanpaths": r["n_val_scanpaths"],
        "n_test_scanpaths": r["n_test_scanpaths"],
        "num_features": r["num_features"],
        "train_output_path": str(
            output_dir / f"{split_results[0]['split_id']}_{feature_type}_train.csv"
        ),
        "val_output_path": str(
            output_dir / f"{split_results[0]['split_id']}_{feature_type}_val.csv"
        ),
        "test_output_path": str(
            output_dir / f"{split_results[0]['split_id']}_{feature_type}_test.csv"
        ),
    }


def check_cache(
    dataset_name: str,
    feature_type: str,
    paths: dict[str, Path],
) -> dict[str, Any] | None:
    """Return cached result dict if train/val/test files exist for this dataset and feature_type."""
    output_dir = paths["output_dir"]
    splits_dir = paths["splits_dir"]
    split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
    if split_paths:
        split_ids = [p.stem.replace("_split_info", "") for p in split_paths]
        for split_id in split_ids:
            t = output_dir / f"{split_id}_{feature_type}_train.csv"
            v = output_dir / f"{split_id}_{feature_type}_val.csv"
            s = output_dir / f"{split_id}_{feature_type}_test.csv"
            if not (t.exists() and v.exists() and s.exists()):
                return None
        sid = split_ids[0]
        train_df = pd.read_csv(
            output_dir / f"{sid}_{feature_type}_train.csv", index_col=0
        )
        val_df = pd.read_csv(output_dir / f"{sid}_{feature_type}_val.csv", index_col=0)
        test_df = pd.read_csv(
            output_dir / f"{sid}_{feature_type}_test.csv", index_col=0
        )
        return {
            "status": "cached",
            "n_train_scanpaths": len(train_df),
            "n_val_scanpaths": len(val_df),
            "n_test_scanpaths": len(test_df),
            "num_features": train_df.shape[1],
            "train_output_path": str(output_dir / f"{sid}_{feature_type}_train.csv"),
            "val_output_path": str(output_dir / f"{sid}_{feature_type}_val.csv"),
            "test_output_path": str(output_dir / f"{sid}_{feature_type}_test.csv"),
        }
    single = output_dir / f"{dataset_name}_{feature_type}.csv"
    if single.exists():
        df = pd.read_csv(single, index_col=0)
        return {
            "status": "cached",
            "num_scanpaths": len(df),
            "num_features": df.shape[1],
            "output_path": str(single),
        }
    return None


def print_summary(
    results: list[dict[str, Any]],
    feature_type: str = "features",
) -> None:
    """Print summary of extraction results."""
    ok = [r for r in results if r.get("status") in ("success", "cached")]
    failed = [r for r in results if r.get("status") == "error"]
    print("\n" + "=" * 80)
    print(f"{feature_type.upper()} EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"\nProcessed/cached: {len(ok)}, Failed: {len(failed)}")
    if ok:
        print(f"\n{'Dataset':<45} {'Train':>8} {'Val':>8} {'Test':>8} {'Features':>10}")
        print("-" * 85)
        for r in ok:
            if "n_train_scanpaths" in r:
                print(
                    f"  {r.get('dataset', '?'):<43} "
                    f"{r.get('n_train_scanpaths', 0):>8} "
                    f"{r.get('n_val_scanpaths', 0):>8} "
                    f"{r.get('n_test_scanpaths', 0):>8} "
                    f"{r.get('num_features', 0):>10}"
                )
            else:
                print(
                    f"  {r.get('dataset', '?'):<43} {r.get('num_scanpaths', 0):>8} scanpaths, {r.get('num_features', 0)} features"
                )
    for r in failed:
        print(f"  FAILED {r.get('dataset', '?')}: {r.get('error', '')}")


def extract_and_save_features(
    df: pd.DataFrame,
    dataset_name: str,
    feature_type: str,
    extractor,
    meta_info: dict[str, Any],
    paths: dict[str, Path],
    check_cache_first: bool = True,
) -> dict[str, Any]:
    """
    Optionally load from cache; else run extractor.fit_transform, then apply splits and save.
    meta_info is the dict returned by load_dataset_with_meta (used only for split resolution).
    """
    if check_cache_first:
        cached = check_cache(dataset_name, feature_type, paths)
        if cached:
            cached["dataset"] = dataset_name
            return cached
    features_df = extractor.fit_transform(df)
    result = apply_splits_and_save(
        features_df, dataset_name, feature_type, paths, split_info=None
    )
    result["dataset"] = dataset_name
    return result
