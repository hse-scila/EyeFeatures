"""
Deep Learning Training Utilities for Eye Features Benchmark.

Aligned with Parquet benchmark: use find_datasets_parquet and load_dataset_parquet from utils.benchmark_utils.
- benchmark_dir: benchmark root (data/benchmark with Parquet + meta.json); find_datasets_func lists by name.
- splits_dir: create_splits output; get_split_info_paths_for_dataset(splits_dir, dataset_name) finds
  {dataset_name}_split_info.json or {dataset_name}_*_split_info.json (per-label splits).
- load_dataset_func(dataset_path) receives Path with .stem = dataset name; returns (df, col_info, type).
No disk saving of intermediate datasets - everything is processed in memory.
"""

import gc
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent figure accumulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eyefeatures.deep.datasets import (
    Dataset2D, DatasetTimeSeries, TimeSeries_2D_Dataset
)
from eyefeatures.deep.models import (
    Classifier, Regressor, SimpleRNN, VitNet, create_simple_CNN
)
from eyefeatures.utils import _split_dataframe
from .benchmark_utils import get_benchmark_dir, get_split_info_paths_for_dataset
from .training_common import (
    REGRESSION_DATASET_PREFIXES,
    SKIP_DATASET_SUBSTRINGS,
    get_task_type,
    get_task_type_for_dataset_label,
)


# ============================================================================
# Configuration
# ============================================================================

# Default configurations
DEFAULT_IMAGE_SHAPE = (100, 100)
ALL_REPRESENTATIONS = [
    'heatmap_fixed', 'heatmap_zoomed', 'baseline_fixed', 'baseline_zoomed',
    'gaf_fixed', 'mtf_fixed',  # GAF/MTF have no zoom variant
]
DEFAULT_TIMESERIES_FEATURES = ['duration']
DEFAULT_MAX_LENGTH = 300

# CNN architecture options
CNN_ARCHITECTURES = {
    'large_resnet': {
        0: {'type': 'Resnet_block', 'params': {'out_channels': 32}},
        1: {'type': 'MaxPool2d', 'params': {'kernel_size': 2, 'stride': 2}},
        2: {'type': 'Resnet_block', 'params': {'out_channels': 64}},
        3: {'type': 'MaxPool2d', 'params': {'kernel_size': 2, 'stride': 2}},
        4: {'type': 'Resnet_block', 'params': {'out_channels': 128}},
        5: {'type': 'Resnet_block', 'params': {'out_channels': 256}},
        6: {'type': 'Resnet_block', 'params': {'out_channels': 256}}
    }
}


# ============================================================================
# Model Wrappers
# ============================================================================

class DictWrapperRNN(nn.Module):
    """Wrapper to make RNN models accept dict inputs."""
    
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
    
    def forward(self, sequences=None, lengths=None, **kwargs):
        if sequences is None:
            sequences = kwargs.get('sequences')
        if lengths is None:
            lengths = kwargs.get('lengths')
        # Ensure lengths is on CPU (required by pack_padded_sequence)
        if lengths is not None and hasattr(lengths, 'is_cuda') and lengths.is_cuda:
            lengths = lengths.cpu()
        return self.rnn(sequences, lengths)


class DictWrapperVitNet(nn.Module):
    """Wrapper to make VitNet models accept dict inputs."""
    
    def __init__(self, vitnet):
        super().__init__()
        self.vitnet = vitnet
    
    def forward(self, images=None, sequences=None, lengths=None, **kwargs):
        if images is None:
            images = kwargs.get('images')
        if sequences is None:
            sequences = kwargs.get('sequences')
        if lengths is None:
            lengths = kwargs.get('lengths')
        # Ensure lengths is on CPU (required by pack_padded_sequence)
        if lengths is not None and hasattr(lengths, 'is_cuda') and lengths.is_cuda:
            lengths = lengths.cpu()
        return self.vitnet(images, sequences, lengths)


# ============================================================================
# Label Preparation
# ============================================================================

def prepare_labels(df: pd.DataFrame, col_info: Dict, pk: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract labels and encode string labels to numeric class IDs.
    
    Args:
        df: Input dataframe
        col_info: Column information dictionary with 'label_cols'
        pk: Primary key columns for grouping
    
    Returns:
        Tuple of (Y dataframe with pk columns and 'label' column, pk columns list)
    
    Raises:
        ValueError: If label_cols or pk are not provided
    """
    label_cols = col_info.get('label_cols', [])
    if not label_cols:
        raise ValueError("label_cols must be provided")
    
    if not pk or len(pk) == 0:
        raise ValueError("pk (group_cols) must be provided")
    
    label_col = label_cols[0]
    cols_to_select = pk.copy()
    if label_col not in cols_to_select:
        cols_to_select.append(label_col)
    
    Y = df[cols_to_select].drop_duplicates()
    
    # Handle duplicate columns
    if Y.columns.tolist().count(label_col) > 1:
        Y = Y.loc[:, ~Y.columns.duplicated()].copy()
    if label_col != 'label':
        Y = Y.rename(columns={label_col: 'label'})
    
    # Handle DataFrame instead of Series for label column
    if isinstance(Y['label'], pd.DataFrame):
        label_series = Y['label'].iloc[:, 0]
        Y = Y.drop(columns='label')
        Y['label'] = label_series
    
    # Validate pk columns exist
    for col in pk:
        if col not in Y.columns:
            raise ValueError(f"Primary key column '{col}' is missing from Y")
    
    # Convert labels to numeric
    if Y['label'].dtype == 'object' or Y['label'].dtype.name == 'object':
        unique_labels = sorted(Y['label'].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        Y['label'] = Y['label'].map(label_to_id).astype(int)
    elif not pd.api.types.is_numeric_dtype(Y['label']):
        try:
            Y['label'] = pd.to_numeric(Y['label'], errors='coerce').astype(int)
        except Exception:
            unique_labels = sorted(Y['label'].unique())
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            Y['label'] = Y['label'].map(label_to_id).astype(int)
    else:
        Y['label'] = Y['label'].astype(int)
    
    return Y, pk


# ============================================================================
# Split Utilities
# ============================================================================

def get_split_indices(
    df: pd.DataFrame,
    pk_cols: List[str],
    split_info: Dict[str, Any]
) -> Tuple[List[int], List[int], List[int]]:
    """Map split info composite indexes to dataset indices.
    
    Args:
        df: Input DataFrame
        pk_cols: Primary key column names
        split_info: Dictionary with 'train', 'val', 'test' composite indexes
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Get split indexes from split_info
    train_indexes = set(split_info.get('train', split_info.get('train_indexes', [])))
    val_indexes = set(split_info.get('val', split_info.get('val_indexes', [])))
    test_indexes = set(split_info.get('test', split_info.get('test_indexes', [])))
    
    # Group DataFrame by pk to get scanpath-level mapping
    groups = list(_split_dataframe(df, pk_cols, encode=True))
    group_to_idx = {}
    for idx, (group_id, _) in enumerate(groups):
        if isinstance(group_id, str):
            composite_pk = group_id
        elif isinstance(group_id, tuple):
            composite_pk = '_'.join(str(g) for g in group_id)
        else:
            composite_pk = str(group_id)
        group_to_idx[composite_pk] = idx
    
    # Map split indexes to dataset indices
    train_indices = [group_to_idx[pk] for pk in train_indexes if pk in group_to_idx]
    val_indices = [group_to_idx[pk] for pk in val_indexes if pk in group_to_idx]
    test_indices = [group_to_idx[pk] for pk in test_indexes if pk in group_to_idx]
    
    return train_indices, val_indices, test_indices


# ============================================================================
# Dataset Creation
# ============================================================================

def create_2d_dataset(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    x_col: str,
    y_col: str,
    pk: List[str],
    rep_type: str,
    image_shape: Tuple[int, int] = DEFAULT_IMAGE_SHAPE
) -> Dataset2D:
    """Create a 2D dataset for a single representation type.
    
    Args:
        df: Input DataFrame with fixation data
        Y: Labels DataFrame
        x_col: X coordinate column name
        y_col: Y coordinate column name
        pk: Primary key columns
        rep_type: Representation type (e.g., 'heatmap_fixed')
        image_shape: Shape of output images
    
    Returns:
        Dataset2D instance
    """
    dataset = Dataset2D(
        df, Y, x=x_col, y=y_col, pk=pk,
        shape=image_shape, representations=[rep_type]
    )
    # Clear matplotlib figures to prevent memory accumulation
    plt.close('all')
    return dataset


def create_timeseries_dataset(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    x_col: str,
    y_col: str,
    pk: List[str],
    features: Optional[List[str]] = None,
    max_length: int = DEFAULT_MAX_LENGTH
) -> DatasetTimeSeries:
    """Create a TimeSeries dataset.
    
    Args:
        df: Input DataFrame with fixation data
        Y: Labels DataFrame
        x_col: X coordinate column name
        y_col: Y coordinate column name
        pk: Primary key columns
        features: List of additional features (e.g., ['duration'])
        max_length: Maximum sequence length
    
    Returns:
        DatasetTimeSeries instance
    """
    # Validate features exist
    if features is not None:
        valid_features = [f for f in features if f in df.columns]
        if len(valid_features) != len(features):
            missing = set(features) - set(valid_features)
            warnings.warn(f"Features {missing} not found in DataFrame, using only {valid_features or 'coordinates'}")
        features = valid_features if valid_features else None
    
    return DatasetTimeSeries(
        df, Y, x=x_col, y=y_col, pk=pk,
        features=features, max_length=max_length
    )


def create_merged_dataset(
    dataset_2d: Dataset2D,
    dataset_ts: DatasetTimeSeries
) -> TimeSeries_2D_Dataset:
    """Create a merged dataset combining 2D and TimeSeries.
    
    Args:
        dataset_2d: 2D dataset
        dataset_ts: TimeSeries dataset
    
    Returns:
        TimeSeries_2D_Dataset instance
    """
    return TimeSeries_2D_Dataset(dataset_2d, dataset_ts)


# ============================================================================
# Model Creation
# ============================================================================

def create_cnn_backbone(in_channels: int, cnn_architecture: str = 'small_vgg') -> nn.Module:
    """Create CNN backbone based on architecture configuration.
    
    Args:
        in_channels: Number of input channels
        cnn_architecture: Architecture name ('small_vgg', 'large_vgg', 'large_resnet')
    
    Returns:
        CNN model
    
    Raises:
        ValueError: If architecture is unknown
    """
    if cnn_architecture not in CNN_ARCHITECTURES:
        raise ValueError(
            f"Unknown CNN architecture: {cnn_architecture}. "
            f"Must be one of: {list(CNN_ARCHITECTURES.keys())}"
        )
    return create_simple_CNN(CNN_ARCHITECTURES[cnn_architecture], in_channels=in_channels)


def create_model(
    dataset_type: str,
    train_dataset,
    task_type: str,
    n_classes: Optional[int] = None,
    cnn_architecture: str = 'small_vgg'
) -> pl.LightningModule:
    """Create appropriate model based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('2d', 'timeseries', 'merged')
        train_dataset: Training dataset (for inferring input dimensions)
        task_type: 'classification' or 'regression'
        n_classes: Number of classes (for classification)
        cnn_architecture: CNN architecture name
    
    Returns:
        PyTorch Lightning model (Classifier or Regressor)
    
    Raises:
        ValueError: If dataset type is unknown or dataset is empty
    """
    if len(train_dataset) == 0:
        raise ValueError("Cannot create model from empty dataset")
    
    sample = train_dataset[0]
    if isinstance(sample, dict):
        sample_x = sample
    else:
        sample_x, _ = sample
    
    if dataset_type == '2d':
        # CNN for 2D images
        images = sample_x['images'] if isinstance(sample_x, dict) else sample_x
        in_channels = images.shape[0] if len(images.shape) == 3 else images.shape[1]
        cnn = create_cnn_backbone(in_channels, cnn_architecture=cnn_architecture)
        if task_type == 'classification':
            return Classifier(cnn, n_classes=n_classes, learning_rate=1e-3)
        else:
            return Regressor(cnn, output_dim=1, learning_rate=1e-3)
    
    elif dataset_type == 'timeseries':
        # RNN for TimeSeries
        sequences = sample_x['sequences'] if isinstance(sample_x, dict) else sample_x
        input_size = sequences.shape[-1] if len(sequences.shape) > 1 else 1
        rnn = SimpleRNN('LSTM', input_size=input_size, hidden_size=64, num_layers=2)
        rnn_wrapped = DictWrapperRNN(rnn)
        if task_type == 'classification':
            return Classifier(rnn_wrapped, n_classes=n_classes, learning_rate=1e-3)
        else:
            return Regressor(rnn_wrapped, output_dim=1, learning_rate=1e-3)
    
    elif dataset_type == 'merged':
        # VitNet for Merged (2D + TimeSeries)
        if 'images' not in sample_x or 'sequences' not in sample_x:
            raise ValueError(f"Merged dataset missing required keys. Got: {list(sample_x.keys())}")
        images = sample_x['images']
        sequences = sample_x['sequences']
        in_channels = images.shape[0] if len(images.shape) == 3 else images.shape[1]
        
        # Note: VitNet projects sequences to embed_dim BEFORE feeding to RNN
        # So RNN input_size should be embed_dim, not the raw sequence features
        embed_dim = 64
        cnn = create_cnn_backbone(in_channels, cnn_architecture=cnn_architecture)
        rnn = SimpleRNN('LSTM', input_size=embed_dim, hidden_size=64, num_layers=2)
        
        vitnet = VitNet(cnn, rnn, fusion_mode='concat', embed_dim=embed_dim)
        vitnet_wrapped = DictWrapperVitNet(vitnet)
        if task_type == 'classification':
            return Classifier(vitnet_wrapped, n_classes=n_classes, learning_rate=1e-3)
        else:
            return Regressor(vitnet_wrapped, output_dim=1, learning_rate=1e-3)
    
    raise ValueError(f"Unknown dataset type: {dataset_type}")


# ============================================================================
# Training Utilities
# ============================================================================

def get_collate_fn(dataset):
    """Get collate function from dataset."""
    if hasattr(dataset, 'collate_fn'):
        return dataset.collate_fn
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'collate_fn'):
        return dataset.dataset.collate_fn
    return None


def get_all_labels_from_dataset(dataset) -> List[int]:
    """Extract ALL labels from dataset.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        List of integer labels
    """
    if len(dataset) == 0:
        raise ValueError("Cannot extract labels from empty dataset")
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, dict):
            label = sample['y']
        else:
            _, label = sample
        labels.append(int(label.item()) if torch.is_tensor(label) else int(label))
    return labels


def compute_metrics(y_true, y_pred, task_type: str) -> Dict[str, Any]:
    """Compute metrics based on task type.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: 'classification' or 'regression'
    
    Returns:
        Dictionary of metric names to values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if task_type == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'n_classes': len(np.unique(y_true))
        }
    else:
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }


def create_collate_with_label_remap(base_collate_fn, label_encoder: Optional[LabelEncoder], task_type: str = 'classification'):
    """Create a collate function that remaps labels using the encoder.
    
    Args:
        base_collate_fn: Original collate function
        label_encoder: Fitted LabelEncoder (or None for no remapping)
        task_type: 'classification' or 'regression'; regression targets are cast to float
    
    Returns:
        Collate function
    """
    def collate_with_label_remap(batch):
        if base_collate_fn is not None:
            result = base_collate_fn(batch)
        else:
            # Default collation
            result = {}
            if 'images' in batch[0]:
                result['images'] = torch.stack([x['images'] for x in batch])
            if 'sequences' in batch[0]:
                result['sequences'] = torch.stack([x['sequences'] for x in batch])
            dtype = torch.float32 if task_type == 'regression' else None
            result['y'] = torch.tensor([
                x['y'].item() if torch.is_tensor(x['y']) else x['y'] 
                for x in batch
            ], dtype=dtype)
        
        # Ensure all float tensors are float32 (not float64/Double)
        # This fixes "mat1 and mat2 must have the same dtype" errors
        for key in ['images', 'sequences']:
            if key in result and result[key] is not None:
                if result[key].dtype == torch.float64:
                    result[key] = result[key].float()  # Convert to float32
        
        # Remap labels if needed (classification only)
        if label_encoder is not None and 'y' in result:
            y = result['y']
            if torch.is_tensor(y):
                y_np = y.cpu().numpy()
                y_remapped = label_encoder.transform(y_np)
                result['y'] = torch.tensor(y_remapped, dtype=torch.long)
        elif task_type == 'regression' and 'y' in result:
            # Regression: ensure target is float (MSELoss expects Float, not Long)
            if result['y'].dtype in (torch.long, torch.int, torch.int32, torch.int64):
                result['y'] = result['y'].float()
        
        return result
    
    return collate_with_label_remap


# ============================================================================
# Epoch-wise progress callback
# ============================================================================


class EpochTqdmCallback(pl.Callback):
    """Lightning callback that shows a tqdm progress bar over epochs."""

    def __init__(self):
        self._pbar: Optional[tqdm] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._pbar = tqdm(total=trainer.max_epochs, desc="Epoch", unit="epoch")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._pbar is not None:
            self._pbar.update(1)
            if trainer.current_epoch + 1 < trainer.max_epochs:
                self._pbar.set_postfix({"epoch": trainer.current_epoch + 1})

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# ============================================================================
# Main Training Function
# ============================================================================


def train_model(
    train_dataset,
    val_dataset,
    test_dataset,
    dataset_type: str,
    task_type: str,
    n_classes: Optional[int] = None,
    cnn_architecture: str = 'small_vgg',
    max_epochs: int = 50,
    batch_size: int = 32,
    label_encoder: Optional[LabelEncoder] = None
) -> Dict[str, Any]:
    """Train a model and return metrics.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        dataset_type: Type of dataset ('2d', 'timeseries', 'merged')
        task_type: 'classification' or 'regression'
        n_classes: Number of classes
        cnn_architecture: CNN architecture name
        max_epochs: Maximum training epochs
        batch_size: Batch size
        label_encoder: Label encoder for remapping
    
    Returns:
        Dictionary with metrics and training info
    """
    # Create model
    model = create_model(
        dataset_type, train_dataset, task_type, 
        n_classes, cnn_architecture=cnn_architecture
    )
    
    # Get collate function with label remapping
    base_collate_fn = get_collate_fn(train_dataset)
    collate_fn = create_collate_with_label_remap(base_collate_fn, label_encoder, task_type=task_type)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, collate_fn=collate_fn
    )
    
    # Train (epoch-wise tqdm)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        callbacks=[EpochTqdmCallback()],
    )
    trainer.fit(model, train_loader, val_loader)
    
    # Evaluate
    model.eval()
    y_true_train, y_pred_train = [], []
    y_true_val, y_pred_val = [], []
    y_true_test, y_pred_test = [], []
    
    with torch.no_grad():
        for loader, y_true_list, y_pred_list in [
            (train_loader, y_true_train, y_pred_train),
            (val_loader, y_true_val, y_pred_val),
            (test_loader, y_true_test, y_pred_test)
        ]:
            for batch in loader:
                y = batch.pop('y')
                pred = model(batch)
                if task_type == 'classification':
                    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
                y_true_list.extend(y.cpu().numpy())
                y_pred_list.extend(pred.cpu().numpy().flatten())
    
    # Compute metrics
    train_metrics = compute_metrics(y_true_train, y_pred_train, task_type)
    val_metrics = compute_metrics(y_true_val, y_pred_val, task_type)
    test_metrics = compute_metrics(y_true_test, y_pred_test, task_type)
    
    return {
        'task_type': task_type,
        'n_classes': n_classes,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


# ============================================================================
# Full Pipeline
# ============================================================================

def process_dataset_label(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    x_col: str,
    y_col: str,
    pk: List[str],
    split_info: Dict[str, Any],
    dataset_type: str,
    rep_type: Optional[str] = None,
    cnn_architecture: str = 'small_vgg',
    max_epochs: int = 50,
    batch_size: int = 32,
    image_shape: Tuple[int, int] = DEFAULT_IMAGE_SHAPE,
    timeseries_features: Optional[List[str]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Process a single dataset+label+type combination.
    
    Creates datasets in memory, trains model, and returns metrics.
    
    Args:
        df: Input DataFrame
        Y: Labels DataFrame
        x_col: X coordinate column
        y_col: Y coordinate column
        pk: Primary key columns
        split_info: Split information dictionary
        dataset_type: '2d', 'timeseries', or 'merged'
        rep_type: Representation type (for 2d/merged)
        cnn_architecture: CNN architecture name
        max_epochs: Maximum training epochs
        batch_size: Batch size
        image_shape: Image shape for 2D datasets
        timeseries_features: Features for TimeSeries dataset
        max_length: Max sequence length
        label_name: Label column name for task type inference
        dataset_name: Optional dataset name (base or full) for regression-override rule
            (surgical / cognitive load / emotion → regression except group_task_label)
    
    Returns:
        Dictionary with results or None on failure
    """
    try:
        # Create appropriate dataset
        if dataset_type == '2d':
            if rep_type is None:
                raise ValueError("rep_type required for 2d dataset")
            full_dataset = create_2d_dataset(df, Y, x_col, y_col, pk, rep_type, image_shape)
        
        elif dataset_type == 'timeseries':
            full_dataset = create_timeseries_dataset(
                df, Y, x_col, y_col, pk, timeseries_features, max_length
            )
        
        elif dataset_type == 'merged':
            if rep_type is None:
                raise ValueError("rep_type required for merged dataset")
            dataset_2d = create_2d_dataset(df, Y, x_col, y_col, pk, rep_type, image_shape)
            dataset_ts = create_timeseries_dataset(
                df, Y, x_col, y_col, pk, timeseries_features, max_length
            )
            full_dataset = create_merged_dataset(dataset_2d, dataset_ts)
        
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Get split indices
        train_indices, val_indices, test_indices = get_split_indices(df, pk, split_info)
        
        if not train_indices or not val_indices or not test_indices:
            print(f"  ⚠️  Empty split indices, skipping")
            return None
        
        # Create subset datasets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        # Get labels and determine task type
        train_labels = get_all_labels_from_dataset(train_dataset)
        val_labels = get_all_labels_from_dataset(val_dataset)
        test_labels = get_all_labels_from_dataset(test_dataset)
        
        all_labels = train_labels + val_labels + test_labels
        all_labels_series = pd.Series(all_labels)
        if dataset_name is not None and label_name is not None:
            task_type = get_task_type_for_dataset_label(
                dataset_name, label_name, all_labels_series
            )
        else:
            task_type = get_task_type(all_labels_series, label_name=label_name)
        
        # Handle label encoding for classification
        label_encoder = None
        n_classes = None
        if task_type == 'classification':
            unique_labels = sorted(set(all_labels))
            n_classes = len(unique_labels)
            
            if n_classes < 2:
                print(f"  ⚠️  Less than 2 classes, skipping")
                return None
            
            # Create label encoder
            label_encoder = LabelEncoder()
            label_encoder.fit(unique_labels)
            
            # Check if remapping is needed
            min_label = min(unique_labels)
            max_label = max(unique_labels)
            needs_remapping = min_label != 0 or max_label != n_classes - 1
            
            if needs_remapping:
                print(f"    Remapping labels: {unique_labels[:5]}... -> [0, {n_classes-1}]")
        
        # Train model
        results = train_model(
            train_dataset, val_dataset, test_dataset,
            dataset_type, task_type, n_classes,
            cnn_architecture=cnn_architecture,
            max_epochs=max_epochs,
            batch_size=batch_size,
            label_encoder=label_encoder
        )
        
        results['rep_type'] = rep_type or ''
        results['cnn_architecture'] = cnn_architecture if dataset_type in ['2d', 'merged'] else ''
        
        # Clean up
        del full_dataset, train_dataset, val_dataset, test_dataset
        gc.collect()
        plt.close('all')
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def _append_result_to_csv(results_file: Path, result_dict: Dict[str, Any]) -> None:
    """Append a single result row to the CSV and dedupe. Used for incremental saving."""
    results_file = Path(results_file)
    dedup_cols = ['dataset', 'label', 'dataset_type', 'rep_type']
    if 'cnn_architecture' in result_dict:
        dedup_cols.append('cnn_architecture')
    row_df = pd.DataFrame([result_dict])
    if results_file.exists():
        existing_df = pd.read_csv(results_file)
        combined_df = pd.concat([existing_df, row_df], ignore_index=True)
        # Only dedupe on columns that exist
        dedup_cols = [c for c in dedup_cols if c in combined_df.columns]
        combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep='last')
        combined_df.to_csv(results_file, index=False)
    else:
        row_df.to_csv(results_file, index=False)


def run_dl_training_battery(
    splits_dir: Union[str, Path],
    results_file: Union[str, Path],
    find_datasets_func,
    load_dataset_func,
    dataset_types: List[str] = ['2d', 'timeseries', 'merged'],
    representations: List[str] = None,
    cnn_architecture: str = 'small_vgg',
    max_epochs: int = 50,
    batch_size: int = 32,
    image_shape: Tuple[int, int] = DEFAULT_IMAGE_SHAPE,
    timeseries_features: Optional[List[str]] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    skip_existing: bool = True,
    test_mode: bool = False,
    test_max_samples: int = 100
) -> pd.DataFrame:
    """Run DL training battery for all datasets. Data is always read from repo data/benchmark.
    
    Args:
        splits_dir: Directory with split JSONs from create_splits ({dataset}_*_split_info.json)
        results_file: Path to save results CSV
        find_datasets_func: Function to find all datasets
        load_dataset_func: Function to load and preprocess a dataset
        dataset_types: List of dataset types to train
        representations: List of representation types (for 2d/merged)
        cnn_architecture: CNN architecture name
        max_epochs: Maximum training epochs
        batch_size: Batch size
        image_shape: Image shape for 2D datasets
        timeseries_features: Features for TimeSeries dataset
        max_length: Max sequence length
        skip_existing: Skip combinations that already have results
        test_mode: If True, use limited samples for faster testing
        test_max_samples: Maximum number of samples to use in test mode
    
    Returns:
        DataFrame with all results
    """
    import json
    
    benchmark_dir = get_benchmark_dir()
    splits_dir = Path(splits_dir)
    results_file = Path(results_file)
    
    if representations is None:
        representations = ALL_REPRESENTATIONS
    if timeseries_features is None:
        timeseries_features = DEFAULT_TIMESERIES_FEATURES
    
    # Load existing results
    existing_keys = set()
    if results_file.exists() and skip_existing:
        existing_df = pd.read_csv(results_file)
        if all(col in existing_df.columns for col in ['dataset', 'label', 'dataset_type', 'rep_type']):
            if 'cnn_architecture' in existing_df.columns:
                existing_keys = set(zip(
                    existing_df['dataset'], existing_df['label'],
                    existing_df['dataset_type'], existing_df['rep_type'].fillna(''),
                    existing_df['cnn_architecture'].fillna('')
                ))
            else:
                existing_keys = set(zip(
                    existing_df['dataset'], existing_df['label'],
                    existing_df['dataset_type'], existing_df['rep_type'].fillna('')
                ))
    
    # Find all datasets (main + extensive; include extracted_fixations if present)
    all_datasets = find_datasets_func(include_extensive_collection=True)
    datasets_to_process = all_datasets.get('fixation', []) + all_datasets.get('unknown', [])
    extracted_dir = benchmark_dir / 'extracted_fixations'
    if extracted_dir.exists():
        ext = find_datasets_func(include_extensive_collection=False, subdir='extracted_fixations')
        datasets_to_process = datasets_to_process + ext.get('fixation', []) + ext.get('saccade', [])

    # For Cognitive_load and Emotions, keep only 0.02 dispersion to reduce computation
    def _keep_dataset(path: Path) -> bool:
        stem = path.stem
        if stem.startswith("Cognitive_load_ready_data_gazes_") or stem.startswith("Emotions_ready_data_gazes_"):
            return "_0.02" in stem
        return True

    datasets_to_process = [p for p in datasets_to_process if _keep_dataset(p)]
    print(f"Found {len(datasets_to_process)} datasets to process")
    
    results = []
    
    for dataset_path in tqdm(datasets_to_process, desc="Datasets"):
        dataset_name = dataset_path.stem
        
        # Per-label splits: get all split files for this dataset
        split_paths = get_split_info_paths_for_dataset(splits_dir, dataset_name)
        if not split_paths:
            continue
        
        # Load and preprocess dataset once per dataset
        df, col_info, _ = load_dataset_func(dataset_path)
        if df is None or len(df) == 0:
            print(f"\n⚠️  Failed to load {dataset_name}, skipping...")
            continue
        
        pk = col_info.get('group_cols', [])
        x_col, y_col = col_info['x_col'], col_info['y_col']
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({len(df):,} rows, {len(split_paths)} split(s))")
        print(f"{'='*60}")
        
        # Process each label split (one split file = one label)
        for split_path in split_paths:
            split_id = split_path.stem.replace('_split_info', '')
            if any(skip in split_id for skip in SKIP_DATASET_SUBSTRINGS):
                continue
            with open(split_path, 'r') as f:
                split_info = json.load(f)
            
            # Label for this split (from split_info or inferred from split_id)
            label_col = split_info.get('label_col')
            if not label_col and len(split_id) > len(dataset_name) + 1:
                label_col = split_id[len(dataset_name) + 1:]
            if not label_col:
                print(f"\n⚠️  Could not infer label for {split_id}, skipping...")
                continue
            
            # In test mode, limit the number of samples in each split
            if test_mode:
                for split_key in ['train', 'val', 'test']:
                    if split_key in split_info and len(split_info[split_key]) > test_max_samples:
                        split_info[split_key] = split_info[split_key][:test_max_samples]
                print(f"  [TEST MODE] Limited to {test_max_samples} samples per split")
            
            # In test mode, filter DataFrame to only include samples in the limited splits
            if test_mode:
                all_split_samples = set(
                    split_info.get('train', []) +
                    split_info.get('val', []) +
                    split_info.get('test', [])
                )
                if len(pk) == 1:
                    df_composite_idx = df[pk[0]].astype(str)
                else:
                    df_composite_idx = df[pk].astype(str).agg('_'.join, axis=1)
                df_work = df[df_composite_idx.isin(all_split_samples)].copy()
                print(f"  [TEST MODE] Filtered DataFrame to {len(df_work):,} rows")
            else:
                df_work = df.copy()
            
            # Process this (dataset, label) with this split_info
            pk_work = pk.copy()
            
            # Handle label column conflicts
            if label_col in pk_work:
                label_col_copy = f"{label_col}_label"
                if label_col_copy not in df_work.columns:
                    df_work[label_col_copy] = df_work[label_col].copy()
                label_col_to_use = label_col_copy
            else:
                label_col_to_use = label_col
            
            col_info_single = col_info.copy()
            col_info_single['label_cols'] = [label_col_to_use]
            
            try:
                Y, pk_work = prepare_labels(df_work, col_info_single, pk_work)
            except Exception as e:
                print(f"  ⚠️  Can't prepare labels for '{label_col}': {e}")
                continue
            
            if Y['label'].nunique() < 2:
                print(f"  ⚠️  Constant label '{label_col}', skipping...")
                continue
            
            print(f"\n  Label: {label_col}")
            
            # Process each dataset type
            for dataset_type in dataset_types:
                if dataset_type in ['2d', 'merged']:
                    rep_types_to_use = representations
                else:
                    rep_types_to_use = [None]
                
                for rep_type in rep_types_to_use:
                    # Check if already exists
                    cnn_key = cnn_architecture if dataset_type in ['2d', 'merged'] else ''
                    if len(existing_keys) > 0 and len(next(iter(existing_keys))) == 5:
                        key = (dataset_name, label_col, dataset_type, rep_type or '', cnn_key)
                    else:
                        key = (dataset_name, label_col, dataset_type, rep_type or '')
                    
                    if key in existing_keys:
                        desc_skip = f"{dataset_type}" + (f"/{rep_type}" if rep_type else "")
                        print(f"Skipping {desc_skip} (already in results)")
                        continue
                    
                    desc = f"{dataset_type}"
                    if rep_type:
                        desc += f"/{rep_type}"
                    print(f"    Training {desc}...", end=' ')
                    
                    result = process_dataset_label(
                        df_work, Y, x_col, y_col, pk_work, split_info,
                        dataset_type, rep_type,
                        cnn_architecture=cnn_architecture,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        image_shape=image_shape,
                        timeseries_features=timeseries_features,
                        max_length=max_length,
                        label_name=label_col,
                        dataset_name=dataset_name,
                    )
                    
                    if result and 'error' not in result:
                        result_dict = {
                            'dataset': dataset_name,
                            'label': label_col,
                            'dataset_type': dataset_type,
                            **result
                        }
                        
                        # Flatten metrics
                        for split in ['train', 'val', 'test']:
                            metrics_key = f'{split}_metrics'
                            if metrics_key in result_dict:
                                for k, v in result_dict[metrics_key].items():
                                    result_dict[f'{split}_{k}'] = v
                                del result_dict[metrics_key]
                        
                        results.append(result_dict)
                        _append_result_to_csv(results_file, result_dict)
                        
                        if result.get('task_type') == 'classification':
                            acc = result_dict.get('test_accuracy', 0)
                            f1 = result_dict.get('test_f1', 0)
                            print(f"✅ acc={acc:.4f}, f1_macro={f1:.4f}")
                        else:
                            r2 = result_dict.get('test_r2', 0)
                            print(f"✅ r2={r2:.4f}")
                    elif result and 'error' in result:
                        print(f"❌ {result['error'][:50]}")
                        error_row = {
                            'dataset': dataset_name,
                            'label': label_col,
                            'dataset_type': dataset_type,
                            'rep_type': rep_type or '',
                            'error': result['error']
                        }
                        if dataset_type in ['2d', 'merged']:
                            error_row['cnn_architecture'] = cnn_key
                        results.append(error_row)
                        _append_result_to_csv(results_file, error_row)
                    else:
                        print("⚠️  Skipped")
            
            # Clean up after each label
            del df_work, Y
            gc.collect()
        
        # Clean up after each dataset
        del df, col_info
        gc.collect()
    
    # Results were saved incrementally after each iteration
    if results:
        print(f"\n✅ Saved {len(results)} results incrementally to {results_file}")
    else:
        print("\n✅ No new results to save")
    if results_file.exists():
        return pd.read_csv(results_file)
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
    if 'dataset_type' in results_df.columns:
        print(f"  Dataset types: {list(results_df['dataset_type'].unique())}")
    if 'label' in results_df.columns:
        print(f"  Labels: {results_df['label'].nunique()}")
    
    # Filter out error rows
    valid_results = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()]
    
    if 'task_type' in valid_results.columns:
        print(f"\n  Task types:")
        print(valid_results['task_type'].value_counts().to_string())
    
    if 'test_accuracy' in valid_results.columns and 'task_type' in valid_results.columns:
        classification_results = valid_results[valid_results['task_type'] == 'classification']
        if len(classification_results) > 0:
            print(f"\n  Classification (n={len(classification_results)}):")
            print(f"    Mean Test Accuracy: {classification_results['test_accuracy'].mean():.4f}")
            print(f"    Std Test Accuracy: {classification_results['test_accuracy'].std():.4f}")
    
    if 'test_r2' in valid_results.columns and 'task_type' in valid_results.columns:
        regression_results = valid_results[valid_results['task_type'] == 'regression']
        if len(regression_results) > 0:
            print(f"\n  Regression (n={len(regression_results)}):")
            print(f"    Mean Test R²: {regression_results['test_r2'].mean():.4f}")
            print(f"    Std Test R²: {regression_results['test_r2'].std():.4f}")
