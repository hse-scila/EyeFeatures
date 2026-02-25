"""
FLAML Training Module for Eye Features Benchmark.

Expects benchmark_experiments layout produced by create_splits + feature_extraction_all:
- Splits: splits_dir with {dataset}_{label_col}_split_info.json and {dataset}_{label_col}_labels.csv
  (or per-split labels: {split_id}_labels_train/val/test.csv from apply_splits_and_save).
- Features: features_dir with {split_id}_{battery}_train/val/test.csv (split_id = dataset_label).
- No Parquet loading here; works on pre-extracted feature CSVs and label CSVs.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    from flaml import AutoML
    from flaml.automl.data import DataFrame
except ImportError:
    raise ImportError(
        "FLAML is not installed. Install it with: pip install flaml[notebook]"
    )

from .benchmark_utils import load_split_info
from .training_common import (
    REGRESSION_DATASET_PREFIXES,
    SKIP_DATASET_SUBSTRINGS,
    get_task_type,
    get_task_type_for_dataset_label,
)


def get_labels_path(splits_dir: Union[str, Path], dataset_name: str) -> Optional[Path]:
    """
    Return path to labels file for a dataset.
    Supports both layouts:
    - Single file: {dataset_name}_labels.csv
    - Proper splits: {dataset_name}_labels_train.csv (train split only; used to discover columns)
    """
    splits_dir = Path(splits_dir)
    combined = splits_dir / f"{dataset_name}_labels.csv"
    if combined.exists():
        return combined
    train = splits_dir / f"{dataset_name}_labels_train.csv"
    if train.exists():
        return train
    return None


def identify_label_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the label column in the dataframe."""
    exclude_cols = {'index', 'Unnamed: 0'}
    
    label_cols = [
        col for col in df.columns 
        if col not in exclude_cols
        and not col.startswith('group_')
        and not ('.' in col and col.split('.')[-1].isdigit())
        and (col.endswith('_label') or 'label' in col.lower())
    ]
    
    if label_cols:
        return label_cols[0]
    
    common_names = ['label', 'target', 'y', 'class', 'category']
    for name in common_names:
        if name in df.columns and name not in exclude_cols and not name.startswith('group_'):
            return name
    
    return None


def get_available_labels(features_path: Union[str, Path]) -> List[str]:
    """Get list of available label columns for a dataset."""
    features_path = Path(features_path)
    
    stem = features_path.stem
    for split_suffix in ['_train', '_val', '_test']:
        if stem.endswith(split_suffix):
            stem = stem[:-len(split_suffix)]
            break
    
    dataset_name = stem
    for suffix in ['_simple_features', '_extended_features', '_complex_features', '_distance_features', '_all_features']:
        if dataset_name.endswith(suffix):
            dataset_name = dataset_name[:-len(suffix)]
            break
    
    splits_dir = features_path.parent / 'splits'
    labels_path = get_labels_path(splits_dir, dataset_name)
    
    if labels_path is not None:
        labels_df = pd.read_csv(labels_path)
        label_columns = []
        for col in labels_df.columns:
            if col == 'index':
                continue
            if col.startswith('group_'):
                continue
            if '.' in col and col.split('.')[-1].isdigit():
                continue
            if col.endswith('_label'):
                label_columns.append(col)
        return label_columns
    
    return []


def filter_valid_labels(
    features_path: Union[str, Path],
    label_columns: List[str],
    split_info: Optional[Dict[str, Any]] = None,
    min_unique_values: int = 2
) -> Tuple[List[str], List[str]]:
    """Filter label columns to only include those with enough unique values in the training set."""
    features_path = Path(features_path)
    
    stem = features_path.stem
    for split_suffix in ['_train', '_val', '_test']:
        if stem.endswith(split_suffix):
            stem = stem[:-len(split_suffix)]
            break
    
    dataset_name = stem
    for suffix in ['_simple_features', '_extended_features', '_complex_features', '_distance_features', '_all_features']:
        if dataset_name.endswith(suffix):
            dataset_name = dataset_name[:-len(suffix)]
            break
    
    splits_dir = features_path.parent / 'splits'
    labels_path = get_labels_path(splits_dir, dataset_name)
    
    if labels_path is None:
        return label_columns, []
    
    labels_df = pd.read_csv(labels_path)
    labels_df['index'] = labels_df['index'].astype(str)
    
    # If we have the combined file (not _labels_train), filter to train indexes when split_info is available
    is_train_only = '_labels_train.csv' in str(labels_path)
    if not is_train_only and split_info is not None:
        if 'train' in split_info:
            train_indexes = set(split_info['train'])
        elif 'train_indexes' in split_info:
            train_indexes = set(split_info['train_indexes'])
        else:
            train_indexes = None
        
        if train_indexes is not None:
            train_mask = labels_df['index'].isin(train_indexes)
            labels_df = labels_df[train_mask]
    
    valid_labels = []
    skipped_labels = []
    
    for label_col in label_columns:
        if label_col not in labels_df.columns:
            skipped_labels.append(label_col)
            continue
        
        n_unique = labels_df[label_col].nunique()
        if n_unique >= min_unique_values:
            valid_labels.append(label_col)
        else:
            skipped_labels.append(label_col)
    
    return valid_labels, skipped_labels


def annotate_label_task_types(
    labels_df: pd.DataFrame,
    label_columns: Optional[List[str]] = None,
    task_type_overrides: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Annotate label columns with their task types (classification or regression)."""
    if label_columns is None:
        exclude_cols = {'index', 'Unnamed: 0'}
        label_columns = [col for col in labels_df.columns if col not in exclude_cols]
    
    if task_type_overrides is None:
        task_type_overrides = {}
    
    label_task_types = {}
    
    for col in label_columns:
        if col not in labels_df.columns:
            continue
        
        if col in task_type_overrides:
            label_task_types[col] = task_type_overrides[col]
        else:
            y = labels_df[col].dropna()
            if len(y) == 0:
                continue
            label_task_types[col] = get_task_type(y)
    
    return label_task_types


def get_hyperparameter_search_space(task_type: str) -> Dict[str, Any]:
    """Get hyperparameter search space for FLAML based on task type."""
    if task_type == 'classification':
        return {
            "time_budget": 300,
            "metric": 'macro_f1',
            "task": 'classification',
            "estimator_list": ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree'],
            "n_jobs": -1,
            "verbose": 1,
        }
    else:
        return {
            "time_budget": 300,
            "metric": 'r2',
            "task": 'regression',
            "estimator_list": ['lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree'],
            "n_jobs": -1,
            "verbose": 1,
        }


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    task_type: str
) -> Dict[str, float]:
    """Compute evaluation metrics based on task type."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
    )
    
    metrics = {}
    
    if task_type == 'classification':
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
        except Exception:
            metrics['roc_auc'] = np.nan
        
        metrics['n_classes'] = len(np.unique(y_true))
        metrics['class_distribution'] = str(dict(zip(*np.unique(y_true, return_counts=True))))
    else:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mean_target'] = np.mean(y_true)
        metrics['std_target'] = np.std(y_true)
    
    return metrics


def train_flaml(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    task_type: Optional[str] = None,
    time_budget: int = 300,
    metric: Optional[str] = None,
    estimator_list: Optional[List[str]] = None,
    n_jobs: int = -1,
    verbose: int = 1,
    **kwargs
) -> Tuple[AutoML, Dict[str, Any]]:
    """Train FLAML AutoML model."""
    if task_type is None:
        task_type = get_task_type(y_train)
    
    search_space = get_hyperparameter_search_space(task_type)
    
    if metric is not None:
        search_space['metric'] = metric
    if estimator_list is not None:
        search_space['estimator_list'] = estimator_list
    search_space['time_budget'] = time_budget
    search_space['n_jobs'] = n_jobs
    search_space['verbose'] = verbose
    search_space.update(kwargs)
    
    automl = AutoML()
    
    if X_val is not None and y_val is not None:
        X_train_full = pd.concat([X_train, X_val], ignore_index=True)
        y_train_full = pd.concat([y_train, y_val], ignore_index=True)
        search_space['eval_method'] = 'holdout'
        search_space['split_ratio'] = len(X_train) / len(X_train_full)
    else:
        X_train_full = X_train
        y_train_full = y_train
        search_space['eval_method'] = 'holdout'
    
    automl.fit(
        X_train=X_train_full,
        y_train=y_train_full,
        **search_space
    )
    
    training_info = {
        'task_type': task_type,
        'best_model': automl.model.estimator.__class__.__name__,
        'best_config': automl.best_config,
        'best_loss': automl.best_loss,
        'time_budget': time_budget,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val) if X_val is not None else 0,
    }
    
    return automl, training_info


def evaluate_flaml(
    model: AutoML,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Evaluate FLAML model on test set."""
    if task_type is None:
        task_type = get_task_type(y_test)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, task_type)
    
    return y_pred, metrics


def run_flaml_pipeline_presplit(
    train_features_path: Union[str, Path],
    val_features_path: Union[str, Path],
    test_features_path: Union[str, Path],
    split_info_path: Optional[Union[str, Path]] = None,
    label_column: Optional[str] = None,
    time_budget: int = 300,
    **flaml_kwargs
) -> Dict[str, Any]:
    """
    Run FLAML pipeline with pre-split data.
    
    Labels are loaded from separate CSV files in the splits directory.
    """
    train_features_path = Path(train_features_path)
    val_features_path = Path(val_features_path)
    test_features_path = Path(test_features_path)
    
    print(f"Loading features...")
    df_train = pd.read_csv(train_features_path)
    df_val = pd.read_csv(val_features_path)
    df_test = pd.read_csv(test_features_path)
    print(f"  Train: {len(df_train)} samples, {df_train.shape[1]} columns")
    print(f"  Val:   {len(df_val)} samples")
    print(f"  Test:  {len(df_test)} samples")
    
    split_info = None
    if split_info_path is not None:
        split_info = load_split_info(split_info_path)
        print(f"  Split by: {split_info.get('split_pk', 'unknown')}")
    
    # Derive dataset name from features path
    stem = train_features_path.stem
    for split_suffix in ['_train', '_val', '_test']:
        if stem.endswith(split_suffix):
            stem = stem[:-len(split_suffix)]
            break
    dataset_name = stem
    for suffix in ['_simple_features', '_extended_features', '_complex_features', '_distance_features', '_all_features']:
        if dataset_name.endswith(suffix):
            dataset_name = dataset_name[:-len(suffix)]
            break
    
    # Load labels from separate CSV files
    splits_dir = train_features_path.parent / 'splits'
    train_labels_path = splits_dir / f"{dataset_name}_labels_train.csv"
    val_labels_path = splits_dir / f"{dataset_name}_labels_val.csv"
    test_labels_path = splits_dir / f"{dataset_name}_labels_test.csv"
    
    print(f"Loading labels from separate CSV files...")
    train_labels = pd.read_csv(train_labels_path)
    val_labels = pd.read_csv(val_labels_path)
    test_labels = pd.read_csv(test_labels_path)
    
    # Merge labels with features using index column
    index_col = 'index' if 'index' in df_train.columns else 'Unnamed: 0'
    if index_col not in df_train.columns:
        raise ValueError(f"Index column '{index_col}' not found in features. Cannot merge labels.")
    
    # Convert to string for matching
    for df in [df_train, df_val, df_test]:
        df[index_col] = df[index_col].astype(str)
    for labels_df in [train_labels, val_labels, test_labels]:
        labels_df['index'] = labels_df['index'].astype(str)
    
    # Merge labels
    train_labels_indexed = train_labels.set_index('index')
    val_labels_indexed = val_labels.set_index('index')
    test_labels_indexed = test_labels.set_index('index')
    
    df_train = df_train.merge(train_labels_indexed, left_on=index_col, right_index=True, how='left')
    df_val = df_val.merge(val_labels_indexed, left_on=index_col, right_index=True, how='left')
    df_test = df_test.merge(test_labels_indexed, left_on=index_col, right_index=True, how='left')
    
    # Validate alignment: ensure all features have matching labels
    train_features_indexes = set(df_train[index_col].astype(str))
    val_features_indexes = set(df_val[index_col].astype(str))
    test_features_indexes = set(df_test[index_col].astype(str))
    
    train_labels_indexes = set(train_labels['index'].astype(str))
    val_labels_indexes = set(val_labels['index'].astype(str))
    test_labels_indexes = set(test_labels['index'].astype(str))
    
    # Check for missing labels
    train_missing = train_features_indexes - train_labels_indexes
    val_missing = val_features_indexes - val_labels_indexes
    test_missing = test_features_indexes - test_labels_indexes
    
    if train_missing or val_missing or test_missing:
        error_msg = "Label alignment error: Some features do not have matching labels.\n"
        if train_missing:
            error_msg += f"  Train: {len(train_missing)} features without labels (e.g., {list(train_missing)[:3]})\n"
        if val_missing:
            error_msg += f"  Val: {len(val_missing)} features without labels (e.g., {list(val_missing)[:3]})\n"
        if test_missing:
            error_msg += f"  Test: {len(test_missing)} features without labels (e.g., {list(test_missing)[:3]})\n"
        raise ValueError(error_msg)
    
    # Check for extra labels (labels without features)
    train_extra = train_labels_indexes - train_features_indexes
    val_extra = val_labels_indexes - val_features_indexes
    test_extra = test_labels_indexes - test_features_indexes
    
    if train_extra or val_extra or test_extra:
        print(f"Warning: Some labels do not have matching features:")
        if train_extra:
            print(f"  Train: {len(train_extra)} labels without features")
        if val_extra:
            print(f"  Val: {len(val_extra)} labels without features")
        if test_extra:
            print(f"  Test: {len(test_extra)} labels without features")
    
    print(f"Label alignment verified: All features have matching labels")
    
    # Identify label column
    if label_column is None:
        label_column = identify_label_column(df_train)
    
    if label_column not in df_train.columns:
        raise ValueError(
            f"Label column '{label_column}' not found after merging labels. "
            f"Available columns: {[c for c in df_train.columns if c.endswith('_label')]}"
        )
    
    print(f"Using label column: {label_column}")
    
    # Validate that all features have non-null labels
    train_nan = df_train[label_column].isna().sum()
    val_nan = df_val[label_column].isna().sum()
    test_nan = df_test[label_column].isna().sum()
    
    if train_nan > 0 or val_nan > 0 or test_nan > 0:
        error_msg = f"Label alignment error: Found NaN values in label column '{label_column}' after merge.\n"
        if train_nan > 0:
            error_msg += f"  Train: {train_nan} missing labels\n"
        if val_nan > 0:
            error_msg += f"  Val: {val_nan} missing labels\n"
        if test_nan > 0:
            error_msg += f"  Test: {test_nan} missing labels\n"
        raise ValueError(error_msg)
    
    # Check if label has enough unique values for training
    n_unique_train = df_train[label_column].nunique()
    if n_unique_train < 2:
        raise ValueError(
            f"Label column '{label_column}' has only {n_unique_train} unique value(s) in training set. "
            "Need at least 2 unique values to train a model."
        )
    
    # Separate features and labels
    exclude_cols = {
        label_column,
        'index', 'Unnamed: 0', 'level_0'  # Index columns
    }
    exclude_cols.update([col for col in df_train.columns if col.startswith('group_')])
    
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    X_train = df_train[feature_cols].copy()
    y_train = pd.Series(df_train[label_column].squeeze())
    X_val = df_val[feature_cols].copy()
    y_val = pd.Series(df_val[label_column].squeeze())
    X_test = df_test[feature_cols].copy()
    y_test = pd.Series(df_test[label_column].squeeze())
    
    # Sanitize feature column names (remove special JSON characters for LightGBM/XGBoost)
    def sanitize_column_name(name):
        # Replace characters that cause issues with tree-based models
        for char in ['[', ']', '{', '}', '"', "'", ',', ':', ';', '<', '>', '\\', '/']:
            name = str(name).replace(char, '_')
        return name
    
    sanitized_cols = {col: sanitize_column_name(col) for col in X_train.columns}
    X_train = X_train.rename(columns=sanitized_cols)
    X_val = X_val.rename(columns=sanitized_cols)
    X_test = X_test.rename(columns=sanitized_cols)
    feature_cols = list(X_train.columns)
    
    # Handle missing values and infinity
    # Guard against empty DataFrames or DataFrames with no numeric columns
    if len(X_train) > 0:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Replace infinity values with NaN first
            X_train[numeric_cols] = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
            X_val[numeric_cols] = X_val[numeric_cols].replace([np.inf, -np.inf], np.nan)
            X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            # Fill numeric columns with mean
            train_mean = X_train[numeric_cols].mean()
            X_train[numeric_cols] = X_train[numeric_cols].fillna(train_mean)
            X_val[numeric_cols] = X_val[numeric_cols].fillna(train_mean)
            X_test[numeric_cols] = X_test[numeric_cols].fillna(train_mean)
        
        # Fill non-numeric columns with mode (most frequent value) or empty string
        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                mode_value = X_train[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else ''
                X_train[col] = X_train[col].fillna(fill_value)
                X_val[col] = X_val[col].fillna(fill_value)
                X_test[col] = X_test[col].fillna(fill_value)
    else:
        # If empty DataFrame, just fill with 0 for numeric, empty string for non-numeric
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(numeric_cols) > 0:
            # Replace infinity values with NaN first, then fill with 0
            X_train[numeric_cols] = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_val[numeric_cols] = X_val[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_test[numeric_cols] = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        if len(non_numeric_cols) > 0:
            X_train[non_numeric_cols] = X_train[non_numeric_cols].fillna('')
            X_val[non_numeric_cols] = X_val[non_numeric_cols].fillna('')
            X_test[non_numeric_cols] = X_test[non_numeric_cols].fillna('')
    
    # Determine task type and encode labels if needed (surgical/cognitive load/emotion → regression except group_task_label)
    task_type = get_task_type_for_dataset_label(dataset_name, label_column, y_train)
    label_encoder = None
    
    if task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_train):
        label_encoder = LabelEncoder()
        label_encoder.fit(pd.concat([y_train, y_val, y_test]))
        y_train = pd.Series(label_encoder.transform(y_train), index=y_train.index)
        y_val = pd.Series(label_encoder.transform(y_val), index=y_val.index)
        y_test = pd.Series(label_encoder.transform(y_test), index=y_test.index)
        print(f"Encoded {len(label_encoder.classes_)} classes")
    
    print(f"Training set: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train model
    print("Training FLAML model...")
    model, training_info = train_flaml(
        X_train, y_train, X_val, y_val,
        task_type=task_type,
        time_budget=time_budget,
        **flaml_kwargs
    )
    
    # Evaluate on training set
    print("Evaluating on training set...")
    y_train_pred, train_metrics_eval = evaluate_flaml(model, X_train, y_train, task_type)
    train_metrics = {
        'best_loss': training_info['best_loss'],
        'best_model': training_info['best_model'],
        **train_metrics_eval
    }
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_val_pred, val_metrics = evaluate_flaml(model, X_val, y_val, task_type)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_pred, test_metrics = evaluate_flaml(model, X_test, y_test, task_type)
    
    results = {
        'model': model,
        'task_type': task_type,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_info': training_info,
        'predictions': {
            'y_train': y_train.values,
            'y_train_pred': y_train_pred,
            'y_test': y_test.values,
            'y_test_pred': y_test_pred,
            'y_val': y_val.values,
            'y_val_pred': y_val_pred
        },
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        'split_info': split_info
    }
    
    return results


def find_all_datasets_with_splits(splits_dir: Union[str, Path]) -> List[str]:
    """Find all datasets that have split info files. Skips datasets whose name contains any SKIP_DATASET_SUBSTRINGS (e.g. label_Anger)."""
    splits_dir = Path(splits_dir)
    split_files = list(splits_dir.glob('*_split_info.json'))
    datasets = []
    for split_file in split_files:
        dataset_name = split_file.stem.replace('_split_info', '')
        if any(skip in dataset_name for skip in SKIP_DATASET_SUBSTRINGS):
            continue
        datasets.append(dataset_name)
    return sorted(datasets)


def create_all_features_battery(
    dataset_name: str,
    features_dir: Union[str, Path],
    base_batteries: List[str] = ['simple_features', 'extended_features', 'complex_features', 'distance_features']
) -> bool:
    """
    Create concatenated 'all_features' battery by merging all base feature batteries on index.
    
    Returns:
        True if successfully created, False otherwise
    """
    features_dir = Path(features_dir)
    
    # Check if all base batteries exist
    all_exist = True
    for battery in base_batteries:
        train_file = features_dir / f"{dataset_name}_{battery}_train.csv"
        val_file = features_dir / f"{dataset_name}_{battery}_val.csv"
        test_file = features_dir / f"{dataset_name}_{battery}_test.csv"
        if not (train_file.exists() and val_file.exists() and test_file.exists()):
            all_exist = False
            break
    
    if not all_exist:
        return False
    
    # Check if already exists
    all_features_train = features_dir / f"{dataset_name}_all_features_train.csv"
    all_features_val = features_dir / f"{dataset_name}_all_features_val.csv"
    all_features_test = features_dir / f"{dataset_name}_all_features_test.csv"
    
    if all_features_train.exists() and all_features_val.exists() and all_features_test.exists():
        return True  # Already created
    
    try:
        # Load all feature batteries for each split
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for battery in base_batteries:
            train_file = features_dir / f"{dataset_name}_{battery}_train.csv"
            val_file = features_dir / f"{dataset_name}_{battery}_val.csv"
            test_file = features_dir / f"{dataset_name}_{battery}_test.csv"
            
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
            test_df = pd.read_csv(test_file)
            
            # Ensure index column exists
            index_col = 'index' if 'index' in train_df.columns else 'Unnamed: 0'
            if index_col not in train_df.columns:
                raise ValueError(f"Index column '{index_col}' not found in {train_file}")
            
            # Set index for merging
            train_df = train_df.set_index(index_col)
            val_df = val_df.set_index(index_col)
            test_df = test_df.set_index(index_col)
            
            # Remove group_ columns and index-like columns from feature columns
            feature_cols = [col for col in train_df.columns 
                          if not col.startswith('group_') and col != index_col]
            
            # Add battery prefix to feature columns to avoid conflicts
            train_df_features = train_df[feature_cols].copy()
            val_df_features = val_df[feature_cols].copy()
            test_df_features = test_df[feature_cols].copy()
            
            # Rename columns with battery prefix
            train_df_features.columns = [f"{battery}_{col}" for col in train_df_features.columns]
            val_df_features.columns = [f"{battery}_{col}" for col in val_df_features.columns]
            test_df_features.columns = [f"{battery}_{col}" for col in test_df_features.columns]
            
            train_dfs.append(train_df_features)
            val_dfs.append(val_df_features)
            test_dfs.append(test_df_features)
        
        # Concatenate all features on index
        train_combined = pd.concat(train_dfs, axis=1)
        val_combined = pd.concat(val_dfs, axis=1)
        test_combined = pd.concat(test_dfs, axis=1)
        
        # Reset index to make it a column
        # The index name should be the same as index_col, but we want it to be 'index'
        train_combined = train_combined.reset_index()
        val_combined = val_combined.reset_index()
        test_combined = test_combined.reset_index()
        
        # Ensure the index column is named 'index'
        # After reset_index, the index becomes a column with its original name (index_col)
        # We need to rename it to 'index' for consistency
        index_col_name = train_combined.columns[0]  # First column is the index
        if index_col_name != 'index':
            train_combined = train_combined.rename(columns={index_col_name: 'index'})
            val_combined = val_combined.rename(columns={index_col_name: 'index'})
            test_combined = test_combined.rename(columns={index_col_name: 'index'})
        
        # Save concatenated features
        train_combined.to_csv(all_features_train, index=False)
        val_combined.to_csv(all_features_val, index=False)
        test_combined.to_csv(all_features_test, index=False)
        
        return True
    except Exception as e:
        print(f"Warning: Failed to create all_features battery for {dataset_name}: {e}")
        return False


def find_available_feature_batteries(
    dataset_name: str,
    features_dir: Union[str, Path],
    feature_batteries: List[str]
) -> List[str]:
    """Find which feature batteries are available for a dataset."""
    features_dir = Path(features_dir)
    available = []
    for battery in feature_batteries:
        if battery == 'all_features':
            # For all_features, check if it exists or can be created
            train_file = features_dir / f"{dataset_name}_all_features_train.csv"
            val_file = features_dir / f"{dataset_name}_all_features_val.csv"
            test_file = features_dir / f"{dataset_name}_all_features_test.csv"
            
            if train_file.exists() and val_file.exists() and test_file.exists():
                available.append(battery)
            else:
                # Try to create it
                if create_all_features_battery(dataset_name, features_dir):
                    available.append(battery)
        else:
            # Check if train/val/test files exist
            train_file = features_dir / f"{dataset_name}_{battery}_train.csv"
            val_file = features_dir / f"{dataset_name}_{battery}_val.csv"
            test_file = features_dir / f"{dataset_name}_{battery}_test.csv"
            
            if train_file.exists() and val_file.exists() and test_file.exists():
                available.append(battery)
    return available


def train_flaml_for_dataset_label_battery(
    dataset_name: str,
    label_column: str,
    feature_battery: str,
    features_dir: Union[str, Path],
    splits_dir: Union[str, Path],
    time_budget: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Train FLAML model for a specific dataset, label, and feature battery.
    
    Returns:
        Dictionary with results or None if training failed
    """
    from datetime import datetime
    
    features_dir = Path(features_dir)
    splits_dir = Path(splits_dir)
    
    # Construct file paths
    train_features_path = features_dir / f"{dataset_name}_{feature_battery}_train.csv"
    val_features_path = features_dir / f"{dataset_name}_{feature_battery}_val.csv"
    test_features_path = features_dir / f"{dataset_name}_{feature_battery}_test.csv"
    split_info_path = splits_dir / f"{dataset_name}_split_info.json"
    
    # Check if all files exist
    if not all([train_features_path.exists(), val_features_path.exists(), 
                test_features_path.exists(), split_info_path.exists()]):
        return None
    
    try:
        # Train FLAML model
        results = run_flaml_pipeline_presplit(
            train_features_path=train_features_path,
            val_features_path=val_features_path,
            test_features_path=test_features_path,
            split_info_path=split_info_path,
            label_column=label_column,
            time_budget=time_budget
        )
        
        # Extract metrics and metadata
        result_dict = {
            'dataset': dataset_name,
            'label': label_column,
            'feature_battery': feature_battery,
            'task_type': results['task_type'],
            'best_model': results['training_info']['best_model'],
            'n_features': results['n_features'],
            'n_train': results['n_samples']['train'],
            'n_val': results['n_samples']['val'],
            'n_test': results['n_samples']['test'],
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add train metrics
        for metric_name, metric_value in results['train_metrics'].items():
            if isinstance(metric_value, (int, float, str)):
                result_dict[f'train_{metric_name}'] = metric_value
        
        # Add validation metrics
        for metric_name, metric_value in results['val_metrics'].items():
            if isinstance(metric_value, (int, float, str)):
                result_dict[f'val_{metric_name}'] = metric_value
        
        # Add test metrics
        for metric_name, metric_value in results['test_metrics'].items():
            if isinstance(metric_value, (int, float, str)):
                result_dict[f'test_{metric_name}'] = metric_value
        
        return result_dict
        
    except Exception as e:
        from datetime import datetime
        return {
            'dataset': dataset_name,
            'label': label_column,
            'feature_battery': feature_battery,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }


def save_results_incremental(
    new_result: Dict[str, Any],
    results_file: Union[str, Path]
) -> None:
    """
    Save a single result incrementally to the results file.
    This ensures results are saved after each experiment to prevent data loss.
    
    Args:
        new_result: Dictionary containing result data for one experiment
        results_file: Path to CSV file to save/load results
    """
    results_file = Path(results_file)
    
    # Ensure results directory exists
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert single result to DataFrame
    new_df = pd.DataFrame([new_result])
    
    # Load existing results if file exists
    if results_file.exists():
        existing_df = pd.read_csv(results_file)
        required_cols = ['dataset', 'label', 'feature_battery']
        use_aoi_key = 'aoi_column' in new_result and 'aoi_column' in existing_df.columns
        if use_aoi_key:
            required_cols = required_cols + ['aoi_column']
        
        if all(col in existing_df.columns for col in required_cols):
            # Check if this combination already exists
            mask = (
                (existing_df['dataset'] == new_result.get('dataset')) &
                (existing_df['label'] == new_result.get('label')) &
                (existing_df['feature_battery'] == new_result.get('feature_battery'))
            )
            if use_aoi_key:
                mask = mask & (existing_df['aoi_column'] == new_result.get('aoi_column'))
            
            if mask.any():
                # Update existing row
                existing_df.loc[mask] = new_df.iloc[0]
                results_df = existing_df
            else:
                # Append new row
                results_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # If required columns don't exist, just append
            results_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Create new file
        results_df = new_df
    
    # Sort by dataset, label, feature_battery (and aoi_column if present)
    sort_cols = ['dataset', 'label', 'feature_battery']
    if 'aoi_column' in results_df.columns:
        sort_cols.append('aoi_column')
    if all(col in results_df.columns for col in sort_cols):
        results_df = results_df.sort_values(sort_cols)
    
    # Save to CSV
    results_df.to_csv(results_file, index=False)


def run_training_battery(
    features_dir: Union[str, Path],
    splits_dir: Union[str, Path],
    results_file: Union[str, Path],
    feature_batteries: List[str],
    time_budget: int = 300,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    Run FLAML training for all datasets, labels, and feature batteries.
    
    Args:
        features_dir: Directory containing feature CSV files
        splits_dir: Directory containing split info and label files
        results_file: Path to CSV file to save/load results
        feature_batteries: List of feature battery names to process
        time_budget: Time budget per model in seconds
        skip_existing: If True, skip combinations that already have results
    
    Returns:
        DataFrame with all results
    """
    from tqdm import tqdm
    
    features_dir = Path(features_dir)
    splits_dir = Path(splits_dir)
    results_file = Path(results_file)
    
    all_results = []
    existing_keys = set()
    
    # Load existing results if file exists
    if results_file.exists() and skip_existing:
        existing_df = pd.read_csv(results_file)
        required_cols = ['dataset', 'label', 'feature_battery']
        if all(col in existing_df.columns for col in required_cols):
            existing_keys = set(zip(
                existing_df['dataset'],
                existing_df['label'],
                existing_df['feature_battery']
            ))
    
    # Find all datasets
    all_datasets = find_all_datasets_with_splits(splits_dir)

    # Pre-pass: count total FLAML runs (dataset × label × feature_battery)
    total_runs = 0
    for dataset_name in all_datasets:
        available_batteries = find_available_feature_batteries(
            dataset_name, features_dir, feature_batteries
        )
        if not available_batteries:
            continue
        labels_path = get_labels_path(splits_dir, dataset_name)
        if labels_path is None:
            continue
        labels_df = pd.read_csv(labels_path)
        all_label_columns = [col for col in labels_df.columns if col.endswith('_label')]
        if not all_label_columns:
            continue
        for lc in all_label_columns:
            if dataset_name.endswith("_" + lc):
                all_label_columns = [lc]
                break
        split_info_path = splits_dir / f"{dataset_name}_split_info.json"
        split_info = load_split_info(split_info_path) if split_info_path.exists() else None
        for feature_battery in available_batteries:
            train_features_path = features_dir / f"{dataset_name}_{feature_battery}_train.csv"
            if not train_features_path.exists():
                continue
            valid_labels, _ = filter_valid_labels(
                train_features_path,
                all_label_columns,
                split_info=split_info,
                min_unique_values=2
            )
            for label_column in valid_labels:
                key = (dataset_name, label_column, feature_battery)
                if key not in existing_keys:
                    total_runs += 1
    print(f"Total FLAML runs (trials): {total_runs}")

    # Iterate over all datasets
    for dataset_name in tqdm(all_datasets, desc="Datasets"):
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Find available feature batteries for this dataset
        available_batteries = find_available_feature_batteries(
            dataset_name, features_dir, feature_batteries
        )
        if not available_batteries:
            print(f"  ⚠️  No feature batteries found for {dataset_name}, skipping...")
            continue
        
        print(f"  Found {len(available_batteries)} feature batteries: {', '.join(available_batteries)}")
        
        # Load labels to find all available labels (supports both _labels.csv and _labels_train.csv)
        labels_path = get_labels_path(splits_dir, dataset_name)
        if labels_path is None:
            print(f"  ⚠️  No labels file found for {dataset_name}, skipping...")
            continue
        
        labels_df = pd.read_csv(labels_path)
        
        # Get all label columns (ending with '_label')
        all_label_columns = [col for col in labels_df.columns if col.endswith('_label')]

        if not all_label_columns:
            print(f"  ⚠️  No label columns found for {dataset_name}, skipping...")
            continue

        # If dataset name is per-label (e.g. ..._effort_label or ..._temporal_label), train only for that label
        for lc in all_label_columns:
            if dataset_name.endswith("_" + lc):
                all_label_columns = [lc]
                print(f"  Per-label dataset: training only for {lc}")
                break

        print(f"  Found {len(all_label_columns)} labels: {', '.join(all_label_columns)}")
        
        # Load split info to filter valid labels
        split_info_path = splits_dir / f"{dataset_name}_split_info.json"
        split_info = load_split_info(split_info_path) if split_info_path.exists() else None
        
        # Iterate over feature batteries
        for feature_battery in available_batteries:
            print(f"\n  Processing feature battery: {feature_battery}")
            
            # Find valid labels for this feature battery
            train_features_path = features_dir / f"{dataset_name}_{feature_battery}_train.csv"
            if not train_features_path.exists():
                continue
            
            # Filter labels that have enough unique values in training set
            valid_labels, skipped_labels = filter_valid_labels(
                train_features_path,
                all_label_columns,
                split_info=split_info,
                min_unique_values=2
            )
            
            if skipped_labels:
                print(f"    Skipped {len(skipped_labels)} labels (insufficient unique values): {', '.join(skipped_labels)}")
            
            if not valid_labels:
                print(f"    ⚠️  No valid labels found for {feature_battery}, skipping...")
                continue
            
            print(f"    Training on {len(valid_labels)} labels: {', '.join(valid_labels)}")
            
            # Iterate over labels
            for label_column in valid_labels:
                # Check if we already have results for this combination
                key = (dataset_name, label_column, feature_battery)
                if key in existing_keys:
                    print(f"    ⏭️  Skipping {label_column} (already trained)")
                    continue
                
                print(f"\n    Training: {dataset_name} / {label_column} / {feature_battery}")
                
                # Train model
                result = train_flaml_for_dataset_label_battery(
                    dataset_name=dataset_name,
                    label_column=label_column,
                    feature_battery=feature_battery,
                    features_dir=features_dir,
                    splits_dir=splits_dir,
                    time_budget=time_budget
                )
                
                if result:
                    all_results.append(result)
                    
                    # Save result immediately after each experiment
                    try:
                        save_results_incremental(result, results_file)
                        print(f"      💾 Result saved to {results_file}")
                    except Exception as e:
                        print(f"      ⚠️  Warning: Failed to save result: {e}")
                    
                    # Print key metrics
                    if 'error' not in result:
                        if result['task_type'] == 'classification':
                            test_accuracy = result.get('test_accuracy', 'N/A')
                            test_f1 = result.get('test_f1', 'N/A')
                            print(f"      ✅ Test Accuracy: {test_accuracy}, Macro F1: {test_f1}")
                        else:
                            test_metric = result.get('test_r2', 'N/A')
                            print(f"      ✅ Test R²: {test_metric}")
                    else:
                        print(f"      ❌ Training failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"  Total models trained: {len(all_results)}")
    print(f"{'='*80}")
    
    # Final consolidation: reload from file to get all results (including incrementally saved ones)
    # This ensures we have all results even if some were saved incrementally but not in all_results
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        
        # Sort by dataset, label, feature_battery
        if all(col in results_df.columns for col in ['dataset', 'label', 'feature_battery']):
            results_df = results_df.sort_values(['dataset', 'label', 'feature_battery'])
        
        # Remove duplicates (keep latest) in case of any duplicates
        results_df = results_df.drop_duplicates(
            subset=['dataset', 'label', 'feature_battery'],
            keep='last'
        )
        
        # Save consolidated results
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Final consolidation: {len(results_df)} total results in {results_file}")
        
        return results_df
    elif all_results:
        # If file doesn't exist but we have results, save them
        results_df = pd.DataFrame(all_results)
        
        # Sort by dataset, label, feature_battery
        if all(col in results_df.columns for col in ['dataset', 'label', 'feature_battery']):
            results_df = results_df.sort_values(['dataset', 'label', 'feature_battery'])
        
        # Save to CSV
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Saved {len(results_df)} results to {results_file}")
        
        return results_df
    else:
        # No results at all
        return pd.DataFrame()


def print_results_summary(results_df: pd.DataFrame):
    """Print summary statistics for results DataFrame."""
    if results_df.empty:
        print("⚠️  No results to summarize.")
        return
    
    print(f"\nSummary Statistics:")
    print(f"  Total results: {len(results_df)}")
    
    if 'dataset' in results_df.columns:
        print(f"  Datasets: {results_df['dataset'].nunique()}")
    if 'feature_battery' in results_df.columns:
        print(f"  Feature batteries: {results_df['feature_battery'].nunique()}")
    if 'label' in results_df.columns:
        print(f"  Labels: {results_df['label'].nunique()}")
    
    if 'task_type' in results_df.columns:
        print(f"\n  Task types:")
        print(results_df['task_type'].value_counts().to_string())
    
    if 'test_accuracy' in results_df.columns and 'task_type' in results_df.columns:
        classification_results = results_df[results_df['task_type'] == 'classification']
        if len(classification_results) > 0:
            print(f"\n  Classification (n={len(classification_results)}):")
            print(f"    Mean Test Accuracy: {classification_results['test_accuracy'].mean():.4f}")
            print(f"    Std Test Accuracy: {classification_results['test_accuracy'].std():.4f}")
    
    if 'test_r2' in results_df.columns and 'task_type' in results_df.columns:
        regression_results = results_df[results_df['task_type'] == 'regression']
        if len(regression_results) > 0:
            print(f"\n  Regression (n={len(regression_results)}):")
            print(f"    Mean Test R²: {regression_results['test_r2'].mean():.4f}")
            print(f"    Std Test R²: {regression_results['test_r2'].std():.4f}")
    
    # Display first few rows
    print(f"\nFirst 5 results:")
    display_cols = []
    for col in ['dataset', 'label', 'feature_battery', 'task_type', 'best_model']:
        if col in results_df.columns:
            display_cols.append(col)
    if 'test_accuracy' in results_df.columns:
        display_cols.append('test_accuracy')
    if 'test_r2' in results_df.columns:
        display_cols.append('test_r2')
    if display_cols:
        print(results_df[display_cols].head().to_string(index=False))
    else:
        print("  No displayable columns found")
