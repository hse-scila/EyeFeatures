"""
Distance-feature pipeline: split data first, then fit on train only and transform train/val/test.
Uses path_pk per split for expected-path computation. Saves one train/val/test set per split_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from eyefeatures.features.dist import (
    MultiMatchDist,
    SimpleDistances,
    TDEDist,
)

from .benchmark_utils import (
    create_composite_index,
    get_path_pk_for_split_id,
    get_split_info_paths_for_dataset,
    load_split_info,
    split_dataframe_by_split_info,
)

SIMPLE_DISTANCE_METHODS = ["euc", "hau", "dtw", "man", "eye", "dfr"]
ADVANCED_DISTANCE_METHODS = ["tde", "multimatch"]
EXPECTED_PATH_METHODS = ["mean", "fwp"]


def _create_composite_index_for_features(source_df: pd.DataFrame, pk: list[str]):
    """Composite index for feature rows: one per unique group in source_df (order preserved)."""
    comp = create_composite_index(source_df, pk)
    return comp.drop_duplicates(keep="first").values


def run_distance_extraction_for_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_id: str,
    path_pk: list[str],
    pk: list[str],
    col_info: dict[str, Any],
    output_dir: Path,
    *,
    simple_methods: list[str] = None,
    advanced_methods: list[str] = None,
    expected_path_methods: list[str] = None,
    tde_k: int = 1,
) -> dict[str, Any]:
    """
    Fit distance transformers on train only, transform train/val/test. Combine all
    method×expected_path combinations, add index, save to output_dir. Returns result dict.
    """
    if simple_methods is None:
        simple_methods = SIMPLE_DISTANCE_METHODS
    if advanced_methods is None:
        advanced_methods = ADVANCED_DISTANCE_METHODS
    if expected_path_methods is None:
        expected_path_methods = EXPECTED_PATH_METHODS

    x_col = col_info["x_col"]
    y_col = col_info["y_col"]
    has_duration = (
        col_info.get("has_duration", False) and "duration" in train_df.columns
    )

    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    successful: list[str] = []

    for ep_method in expected_path_methods:
        for d_method in simple_methods:
            try:
                trans = SimpleDistances(
                    methods=[d_method],
                    x=x_col,
                    y=y_col,
                    pk=pk,
                    path_pk=path_pk,
                    expected_paths_method=ep_method,
                    return_df=True,
                )
                trans.fit(train_df)
                t_f = trans.transform(train_df)
                v_f = trans.transform(val_df)
                s_f = trans.transform(test_df)
                if t_f is not None and len(t_f) > 0:
                    successful.append(f"{d_method}_{ep_method}")
                    train_frames.append(t_f)
                    val_frames.append(v_f)
                    test_frames.append(s_f)
            except Exception:
                pass

    for ep_method in expected_path_methods:
        for d_method in advanced_methods:
            if d_method in ("scanmatch", "multimatch") and not has_duration:
                continue
            try:
                if d_method == "tde":
                    trans = TDEDist(
                        k=tde_k,
                        x=x_col,
                        y=y_col,
                        pk=pk,
                        path_pk=path_pk,
                        expected_paths_method=ep_method,
                        return_df=True,
                    )
                elif d_method == "multimatch":
                    trans = MultiMatchDist(
                        x=x_col,
                        y=y_col,
                        duration="duration",
                        pk=pk,
                        path_pk=path_pk,
                        expected_paths_method=ep_method,
                        return_df=True,
                    )
                else:
                    continue
                trans.fit(train_df)
                t_f = trans.transform(train_df)
                v_f = trans.transform(val_df)
                s_f = trans.transform(test_df)
                if t_f is not None and len(t_f) > 0:
                    successful.append(f"{d_method}_{ep_method}")
                    train_frames.append(t_f)
                    val_frames.append(v_f)
                    test_frames.append(s_f)
            except Exception:
                pass

    if not train_frames:
        return {
            "status": "skipped",
            "reason": "No distance features computed",
            "split_id": split_id,
        }

    train_out = pd.concat(train_frames, axis=1)
    val_out = pd.concat(val_frames, axis=1)
    test_out = pd.concat(test_frames, axis=1)

    train_out["index"] = _create_composite_index_for_features(train_df, pk)
    val_out["index"] = _create_composite_index_for_features(val_df, pk)
    test_out["index"] = _create_composite_index_for_features(test_df, pk)

    train_path = output_dir / f"{split_id}_distance_features_train.csv"
    val_path = output_dir / f"{split_id}_distance_features_val.csv"
    test_path = output_dir / f"{split_id}_distance_features_test.csv"
    train_out.to_csv(train_path, index=True)
    val_out.to_csv(val_path, index=True)
    test_out.to_csv(test_path, index=True)

    return {
        "status": "success",
        "split_id": split_id,
        "n_train_scanpaths": len(train_out),
        "n_val_scanpaths": len(val_out),
        "n_test_scanpaths": len(test_out),
        "num_features": train_out.shape[1] - 1,
        "combinations": successful,
        "train_output_path": str(train_path),
        "val_output_path": str(val_path),
        "test_output_path": str(test_path),
    }


def extract_and_save_distance_features(
    df: pd.DataFrame,
    dataset_name: str,
    meta_info: dict[str, Any],
    col_info: dict[str, Any],
    paths: dict[str, Path],
    *,
    simple_methods: list[str] = None,
    advanced_methods: list[str] = None,
    expected_path_methods: list[str] = None,
    path_pk_per_label: dict[str, list[str]] | None = None,
    check_cache_per_split: bool = True,
) -> list[dict[str, Any]]:
    """
    For each split: split df into train/val/test; path_pk (reference path grouping) from
    path_pk_per_label[split_id] (old notebook PATH_PK_PER_LABEL), default = full pk.
    Fit on train, transform all. Returns list of result dicts (one per split_id).
    """
    pk = col_info["group_cols"]
    splits_dir = paths["splits_dir"]
    output_dir = paths["output_dir"]
    split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
    if not split_paths:
        return [{"dataset": dataset_name, "status": "error", "error": "No split info"}]

    results: list[dict[str, Any]] = []
    for split_path in split_paths:
        split_id = split_path.stem.replace("_split_info", "")
        path_pk = get_path_pk_for_split_id(split_id, pk, path_pk_per_label)
        missing = [c for c in path_pk if c not in df.columns]
        if missing:
            results.append(
                {
                    "dataset": dataset_name,
                    "split_id": split_id,
                    "status": "error",
                    "error": f"path_pk columns missing: {missing}",
                }
            )
            continue

        if check_cache_per_split:
            t_p = output_dir / f"{split_id}_distance_features_train.csv"
            v_p = output_dir / f"{split_id}_distance_features_val.csv"
            s_p = output_dir / f"{split_id}_distance_features_test.csv"
            if t_p.exists() and v_p.exists() and s_p.exists():
                tr = pd.read_csv(t_p, index_col=0)
                results.append(
                    {
                        "dataset": dataset_name,
                        "status": "cached",
                        "split_id": split_id,
                        "n_train_scanpaths": len(tr),
                        "n_val_scanpaths": len(pd.read_csv(v_p, index_col=0)),
                        "n_test_scanpaths": len(pd.read_csv(s_p, index_col=0)),
                        "num_features": tr.shape[1] - 1,
                    }
                )
                print(f"  Cached: {split_id}")
                continue

        split_info = load_split_info(split_path)
        train_df, val_df, test_df = split_dataframe_by_split_info(df, pk, split_info)
        print(f"  Split: {split_id} (path_pk: {path_pk})")
        res = run_distance_extraction_for_split(
            train_df,
            val_df,
            test_df,
            split_id,
            path_pk,
            pk,
            col_info,
            output_dir,
            simple_methods=simple_methods,
            advanced_methods=advanced_methods,
            expected_path_methods=expected_path_methods,
        )
        res["dataset"] = dataset_name
        results.append(res)
    return results
