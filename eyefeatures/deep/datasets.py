import warnings
from collections.abc import Callable
from copy import copy
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Data
from tqdm import tqdm

from eyefeatures.features.complex import get_gafs, get_heatmaps, get_mtfs
from eyefeatures.preprocessing.base import BaseFixationPreprocessor
from eyefeatures.utils import _split_dataframe
from eyefeatures.visualization.static_visualization import get_visualizations


def _coord_to_grid(coords: np.array, xlim: tuple, ylim: tuple, shape: tuple):
    """Maps 2D coordinates to grid indices based on the grid resolution.

    Args:
        coords (np.ndarray): Array of coordinates to map.
        xlim: The x-axis limits (x_min, x_max).
        ylim: The y-axis limits (y_min, y_max).
        shape: The shape of the grid (rows, cols).

    Returns:
        tuple(int, int): A tuple (i, j) - the grid indices
        corresponding to the coordinates.
    """

    i = (((coords[:, 0] - xlim[0]) / (xlim[1] - xlim[0])) * shape[0]).astype(int)
    j = (((coords[:, 1] - ylim[0]) / (ylim[1] - ylim[0])) * shape[1]).astype(int)
    return i, j


def _cell_index(i: int, j: int, shape: tuple[int, int]):
    """Maps grid indices (i, j) to a 1D cell index based on the grid shape.

    Args:
        i (int): Row index in the grid.
        j (int): Column index in the grid.
        shape (tuple(int,int)): The shape of the grid (rows, cols).

    Returns:
        int:  The 1D cell index.
    """

    return i * shape[1] + j


def _calculate_cell_center(i: int, j: int, xlim: tuple, ylim: tuple, shape: tuple):
    """Calculates the center coordinates of a grid cell.

    Args:
        i (int): Row index in the grid.
        j (int): Column index in the grid.
        xlim: The x-axis limits (x_min, x_max).
        ylim: The y-axis limits (y_min, y_max).
        shape: The shape of the grid (rows, cols).

    Returns:
        tuple(float,float):
    -------
    x_center, y_center: Tuple[float, float]
        Center coordinates of the grid cell.
    """
    cell_width = (xlim[1] - xlim[0]) / shape[0]
    cell_height = (ylim[1] - ylim[0]) / shape[1]
    x_center = xlim[0] + (i + 0.5) * cell_width
    y_center = ylim[0] + (j + 0.5) * cell_height
    return x_center, y_center


def _calculate_length_vectorized(coords: np.array):
    """
    Calculates the Euclidean distance between consecutive points in 2D space.

    :param coords: Array of coordinates with shape (n, 2).

    Returns
    -------
    lengths: np.array
        Euclidean distances between consecutive points.
    """
    # Calculate the difference between consecutive points
    dx = coords[1:, 0] - coords[:-1, 0]
    dy = coords[1:, 1] - coords[:-1, 1]

    # Calculate the Euclidean distance between consecutive points
    lengths = np.sqrt(dx**2 + dy**2)

    return lengths


def create_edge_list_and_cumulative_features(
    df, add_duration, x_col, y_col, xlim, ylim, shape, directed=True
):
    """Creates an edge list and computes cumulative node
        features (total duration, total saccade lengths, and cell
        center coordinates). These features are normalized by their
        respective maximum values. Also computes edge features based
        on the sum of edge lengths.

        Args:
            df: DataFrame containing the coordinates and other node features.
            x_col: Column name in df for the x coordinates.
            y_col: Column name in df for the y coordinates.
            add_duration: Column name in df for the duration between
                consecutive points (optional).
            xlim: Tuple (x_min, x_max) defining the bounds for the x-axis.
            ylim: Tuple (y_min, y_max) defining the bounds for the y-axis.
            shape: Tuple (x_res, y_res) defining the resolution of the grid.
            directed: If True, the graph is directional; if False, bidirectional
                edges are created.

    Returns:
        edge_list: List of edges as pairs of node indices.
        edge_features: Normalized edge features (sum of edge lengths).
        node_mapping: Mapping of old node indices to new compacted indices.
        cumulative_node_features: Normalized cumulative node features:

            * ``total_duration``: Total duration at each node, normalized.
            * ``total_saccade_length_to``: Total saccade length directed to
                each node, normalized.
            * ``total_saccade_length_from``: Total saccade length originating
                from each node, normalized.
            * ``cell_centers``: Coordinates of the center of each cell.

    """

    coords = df[[x_col, y_col]].values
    i, j = _coord_to_grid(coords, xlim, ylim, shape)
    grid_indices = _cell_index(i, j, shape)
    # print(grid_indices)
    unique_nodes = np.unique(grid_indices)
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    # Initialize cumulative feature arrays
    total_durations = np.zeros(num_nodes)
    total_saccade_length_to = np.zeros(num_nodes)
    total_saccade_length_from = np.zeros(num_nodes)
    cell_centers = np.zeros((num_nodes, 2))

    # Dictionary to accumulate edge lengths
    edge_length_sum = {}

    # Create edge list and calculate cumulative features
    edge_list = []
    lengths = _calculate_length_vectorized(coords)

    for k in range(len(df) - 1):
        src_node = node_mapping[grid_indices[k]]
        dst_node = node_mapping[grid_indices[k + 1]]

        # Handle self-loops: Only update duration
        if src_node == dst_node:
            if add_duration:
                total_durations[src_node] += df["duration"].iloc[k]
            continue  # Skip the rest of the loop, don't add an edge

        # Add edge if not a self-loop
        edge = (src_node, dst_node)
        edge_list.append(edge)

        if not directed:
            reverse_edge = (dst_node, src_node)
            edge_list.append(reverse_edge)
            edge_length_sum[reverse_edge] = (
                edge_length_sum.get(reverse_edge, 0) + lengths[k]
            )

        # Accumulate the length of the edge
        edge_length_sum[edge] = edge_length_sum.get(edge, 0) + lengths[k]

        # Update cumulative saccade features (only for non-self-loops)
        total_saccade_length_to[dst_node] += lengths[k]
        total_saccade_length_from[src_node] += lengths[k]

        # Calculate the center coordinates of the cell
        i_node, j_node = i[k], j[k]
        x_center, y_center = _calculate_cell_center(i_node, j_node, xlim, ylim, shape)
        cell_centers[src_node] = [x_center, y_center]

    # Normalize cumulative features by their respective maximum values
    if np.max(total_durations) > 0:
        total_durations /= np.max(total_durations)
    if np.max(total_saccade_length_to) > 0:
        total_saccade_length_to /= np.max(total_saccade_length_to)
    if np.max(total_saccade_length_from) > 0:
        total_saccade_length_from /= np.max(total_saccade_length_from)

    # Normalize edge features (sum of lengths) by their maximum value
    if edge_length_sum:  # Ensure there are edges to normalize
        max_edge_length_sum = np.max(list(edge_length_sum.values()))
        edge_features = [
            edge_length_sum[edge] / max_edge_length_sum for edge in edge_list
        ]
    else:
        edge_features = []

    # Combine cumulative features into a dictionary
    cumulative_node_features = {
        "total_duration": total_durations,
        "total_saccade_length_to": total_saccade_length_to,
        "total_saccade_length_from": total_saccade_length_from,
        "cell_centers": cell_centers,
    }

    return edge_list, edge_features, node_mapping, cumulative_node_features


def create_graph_data_from_dataframe(
    df, y, x_col, y_col, add_duration, xlim, ylim, shape, directed=True
):
    """Converts a DataFrame into a PyTorch Geometric Data object for GCN training.
    Includes cumulative node features (total duration, total saccade
    length to/from node, and cell center coordinates).
    Edge features are based on the sum of lengths of corresponding edges.

    Args:
        df: DataFrame containing the coordinates and other node features.
        x_col: Column name in df for the x coordinates.
        y_col: Column name in df for the y coordinates.
        add_duration: Column name in df for the duration between consecutive
            points (optional).
        xlim: Tuple (x_min, x_max) defining the bounds for the x-axis.
        ylim: Tuple (y_min, y_max) defining the bounds for the y-axis.
        shape: Tuple (x_res, y_res) defining the resolution of the grid.
        directed: If True, the graph is directional; if False, bidirectional
            edges are created.

    Returns:
        A PyTorch Geometric Data object containing the graph and its features.
    """

    # Get edge list and cumulative features
    (
        edge_list,
        edge_features,
        node_mapping,
        cumulative_node_features,
    ) = create_edge_list_and_cumulative_features(
        df, add_duration, x_col, y_col, xlim, ylim, shape, directed
    )

    # Combine cumulative features into a feature matrix
    node_features = np.hstack(
        [
            cumulative_node_features["cell_centers"],  # Cell center coordinates
            cumulative_node_features["total_duration"].reshape(-1, 1),
            cumulative_node_features["total_saccade_length_to"].reshape(-1, 1),
            cumulative_node_features["total_saccade_length_from"].reshape(-1, 1),
        ]
    )

    # Convert node features to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Convert edge list and edge features to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float).view(
        -1, 1
    )  # Shape [num_edges, 1]
    mapping = torch.tensor(sorted(node_mapping, key=node_mapping.get))

    # Create PyTorch Geometric Data object with edge weights
    data = Data(
        x=x,
        y=torch.tensor(y),
        edge_index=edge_index,
        edge_attr=edge_attr,
        mapping=mapping,
    )
    return data


# Representation types with zoom options:
# - *_fixed: uses fixed [0,1] coordinate space (shows absolute position)
# - *_zoomed: zooms to data range (fills the image with the scanpath region)
_representations = {
    # Heatmaps
    "heatmap": get_heatmaps,  # Default: fixed [0,1] space (backward compat)
    "heatmap_fixed": partial(get_heatmaps, zoom_to_data=False),
    "heatmap_zoomed": partial(get_heatmaps, zoom_to_data=True),
    # Baseline visualization
    "baseline_visualization": partial(get_visualizations, pattern="baseline"),  # Default: zoomed (backward compat)
    "baseline_fixed": partial(get_visualizations, pattern="baseline", zoom_to_data=False),
    "baseline_zoomed": partial(get_visualizations, pattern="baseline", zoom_to_data=True),
    # AOI visualization
    "aoi_visualization": partial(get_visualizations, pattern="aoi"),
    "aoi_fixed": partial(get_visualizations, pattern="aoi", zoom_to_data=False),
    "aoi_zoomed": partial(get_visualizations, pattern="aoi", zoom_to_data=True),
    # Saccade visualization (with sequential colormap)
    "saccade_visualization": partial(get_visualizations, pattern="saccades"),  # Default: zoomed
    "saccade_fixed": partial(get_visualizations, pattern="saccades", zoom_to_data=False),
    "saccade_zoomed": partial(get_visualizations, pattern="saccades", zoom_to_data=True),
    # GAF (Gramian Angular Field) and MTF (Markov Transition Field) 2D maps for DL (no zoom variant)
    "gaf_fixed": get_gafs,
    "mtf_fixed": get_mtfs,
}


class Dataset2D(Dataset):
    """Custom dataset for 2D image-based representations derived from gaze data.

    Args:
        X: Input data.
        Y: Labels for the data.
        pk: List of primary keys for grouping.
        shape: Shape of the images.
        representations: List of representation types.
        upload_to_cuda: If True, upload the data to the GPU. Default: False.
        transforms: Transformations to apply to the data.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        Y: ArrayLike,
        x: str,
        y: str,
        pk: list[str],
        shape: tuple[int] | int,
        representations: list[str],
        upload_to_cuda: bool = False,
        transforms=None,
    ):
        self.pmk = pk
        rep_tensors = []
        for i in representations:
            rep_data = _representations[i](X, x, y, pk=pk, shape=shape)
            rep_tensor = torch.tensor(rep_data, dtype=torch.float32)
            # Only add channel dimension if not already present
            # Heatmaps return (n, h, w), visualizations return (n, c, h, w)
            if rep_tensor.ndim == 3:
                rep_tensor = rep_tensor.unsqueeze(1)
            rep_tensors.append(rep_tensor)
        self.X = torch.cat(rep_tensors, dim=1)
        self.channels_dim = self.X.shape[1]
        print(f"Number of channels = {self.channels_dim}.")
        if not isinstance(Y, pd.Series):
            Y = Y.set_index(pk).squeeze(axis=0)
        self.y = Y.sort_index().values
        if np.issubdtype(self.y.dtype, np.integer):
            self.y = torch.tensor(self.y, dtype=torch.long)
        else:
            self.y = torch.tensor(self.y, dtype=torch.float)
        if upload_to_cuda:
            self.X = self.X.cuda()
            self.y = self.y.cuda()
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        X = self.X[idx, :, :, :]
        label = self.y[idx]
        
        if self.transforms is not None:
            X = self.transforms(X)

        return {
            "images": X,
            "y": label,
        }

    def collate_fn(self, batch):
        images = torch.stack([x["images"] for x in batch])
        y = torch.tensor([x["y"] for x in batch])
        return {"images": images, "y": y}


def _get_features(X, features, x, y, t, pk):
    # Handle None features case - return only x and y coordinates
    if features is None:
        output = []
        groups = _split_dataframe(X, pk)
        for group_id, group_X in tqdm(groups):
            output.append(group_X[[x, y]].values)
        return output
    
    preprocessor = BaseFixationPreprocessor(x, y, t, pk)
    features_to_get = copy(features)
    for i in features:
        if i in X.columns:
            features_to_get.remove(i)
    output = []
    groups = _split_dataframe(X, pk)
    for group_id, group_X in tqdm(groups):
        cur_X = preprocessor._compute_feats(group_X, features_to_get)
        output.append(cur_X[[x, y] + features].values)

    return output


class DatasetTimeSeries(Dataset):
    """Custom dataset for time-series data.

    Args:
        X: Input time-series data.
        Y: Labels for the data.
        pk: Primary keys for grouping.
        features: List of features to extract. If None, only x and y coordinates are used.
        transforms: Transformations to apply to the data.
        max_length: maximum length of scanpath.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        Y: ArrayLike,
        x: str,
        y: str,
        pk: list[str],
        features: Optional[list[str]] = None,
        transforms: Callable = None,
        max_length: int = 10,
    ):
        self.pmk = pk
        self.X = _get_features(X, features, x, y, t=None, pk=pk)
        if not isinstance(Y, pd.Series):
            Y = Y.set_index(pk).squeeze(axis=0)
        self.Y = Y.sort_index().values
        if np.issubdtype(self.Y.dtype, np.integer):
            self.Y = torch.tensor(self.Y, dtype=torch.long)
        else:
            self.Y = torch.tensor(self.Y, dtype=torch.float)

        self.n_features = 2 + (len(features) if features is not None else 0)
        self.transforms = transforms
        self.max_length = max_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        X = self.X[idx]
        label = self.Y[idx]

        if self.transforms:
            X = self.transforms(X)

        return {
            "sequences": torch.tensor(X, dtype=torch.float),
            "y": label,
        }

    def collate_fn(self, batch):
        if self.max_length is None:
            lengths = [x["sequences"].shape[0] for x in batch]
            max_len = max(lengths)
            padded_batch = [
                torch.cat(
                    [
                        x["sequences"],
                        torch.zeros(max_len - x["sequences"].shape[0], self.n_features),
                    ],
                    axis=0,
                )
                for x in batch
            ]
        else:
            max_len = self.max_length
            lengths = [min(x["sequences"].shape[0], max_len) for x in batch]
            padded_batch = [
                torch.cat(
                    [
                        x["sequences"][: self.max_length],
                        torch.zeros(
                            max_len - x["sequences"][: self.max_length].shape[0],
                            self.n_features,
                        ),
                    ],
                    axis=0,
                )
                for x in batch
            ]

        y = torch.tensor([x["y"] for x in batch])
        return {
            "sequences": torch.stack(padded_batch),
            "lengths": torch.tensor(lengths),
            "y": y,
        }


class TimeSeries_2D_Dataset(Dataset):
    """Composite dataset that combines image and time-series data.

    Args:
        image_dataset: Dataset containing image data.
        sequence_dataset: Dataset containing sequence data.
    """

    def __init__(self, image_dataset: Dataset, sequence_dataset: Dataset):
        # Ensure both datasets have the same length
        assert len(image_dataset) == len(
            sequence_dataset
        ), "Datasets must be of the same length"

        self.image_dataset = image_dataset
        self.sequence_dataset = sequence_dataset

    def __len__(self):
        # The length of the composite dataset is the same as either individual dataset
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Fetch the data from both datasets using the same index
        image = self.image_dataset.X[idx, :, :, :]
        sequence = self.sequence_dataset.X[idx]
        y = self.image_dataset.y[idx]
        # Use float32 so batch matches model parameters (avoid Double vs Float mismatch)
        return {
            "images": torch.as_tensor(image, dtype=torch.float32),
            "sequences": torch.as_tensor(sequence, dtype=torch.float32),
            "y": y,
        }

    def collate_fn(self, batch):
        lengths = [x["sequences"].shape[0] for x in batch]
        max_len = max(lengths)
        padded_batch = [
            torch.cat(
                [
                    x["sequences"],
                    torch.zeros(
                        max_len - x["sequences"].shape[0],
                        self.sequence_dataset.n_features,
                        dtype=torch.float32,
                    ),
                ],
                axis=0,
            )
            for x in batch
        ]

        y = torch.tensor([x["y"] for x in batch])

        return {
            "sequences": torch.stack(padded_batch),
            "lengths": torch.tensor(lengths),
            "images": torch.stack([x["images"] for x in batch]),
            "y": y,
        }


class GridGraphDataset(Dataset):
    """Custom dataset for generating grid-based graph
    representations from spatial coordinates.

    Args:
        X: Input dataframe.
        Y: Labels for the data.
        x: X coordinate column name.
        y: Y coordinate column name.
        pk: Primary keys for grouping.
        x_col, y_col: Column names for x and y coordinates.
        add_duration: Column name for time durations.
        xlim: Limits of the x-axis.
        ylim: Limits of the y-axis.
        shape: Shape of the grid.
        directed: Whether the graph is directed.
        transforms: Transformations to apply to the data.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        Y: ArrayLike,
        x: str,
        y: str,
        add_duration: str,
        pk: list[str],
        xlim: tuple[float, float] = (0, 1),
        ylim: tuple[float, float] = (0, 1),
        shape: tuple[int, int] = (10, 10),
        directed: bool = True,
        transforms: Callable = None,
    ):
        super().__init__()
        self.transform = transforms
        self.pk = pk
        self.directed = directed
        if not isinstance(Y, pd.Series):
            Y = Y.set_index(pk).sort_index().squeeze(axis=0)
        Y = Y.values
        self.X = self.get_graphs(X, Y, x, y, add_duration, xlim, ylim, shape)

    def get_graphs(self, X, Y, x_col, y_col, add_duration, xlim, ylim, shape):
        groups = _split_dataframe(X, pk=self.pk)
        graphs = []
        for i, (group_id, cur_X) in tqdm(enumerate(groups), desc="Getting graphs..."):
            graphs.append(
                create_graph_data_from_dataframe(
                    cur_X,
                    Y[i],
                    x_col,
                    y_col,
                    add_duration,
                    xlim,
                    ylim,
                    shape,
                    directed=self.directed,
                )
            )

        return graphs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

    def collate_fn(self, batch):
        return batch