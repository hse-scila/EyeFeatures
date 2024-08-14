import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike
from typing import Union, List, Tuple
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
import warnings
from eyetracking.features.complex import get_heatmaps
from skmultilearn.model_selection import IterativeStratification

def iterative_split(df: pd.pd.DataFrame, y: ArrayLike, test_size: float, stratify_columns: List[str]):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'

    From https://madewithml.com/courses/mlops/splitting/#stratified-split
    """
    # One-hot encode the stratify columns and concatenate them
    one_hot_cols = [pd.get_dummies(df[col]) for col in stratify_columns]
    one_hot_cols = pd.concat(one_hot_cols, axis=1).to_numpy()
    stratifier = IterativeStratification(
        n_splits=2, order=len(stratify_columns), sample_distribution_per_fold=[test_size, 1-test_size])
    train_indices, test_indices = next(stratifier.split(df.to_numpy(), one_hot_cols))
    # Return the train and test set dataframes
    X_train, X_test = df.iloc[train_indices], df.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def coord_to_grid(coords, xlim, ylim, shape):
    """Maps coordinates to grid indices."""
    i = (((coords[:, 0] - xlim[0]) / (xlim[1] - xlim[0]))*shape[0]).astype(int)
    j = (((coords[:, 0] - ylim[0]) / (ylim[1] - ylim[0]))*shape[1]).astype(int)
    return i, j

def cell_index(i, j, shape):
    """Maps grid indices to a 1D index."""
    return i * shape[1] + j

def calculate_cell_center(i, j, xlim, ylim, shape):
    """Calculates the center of a grid cell."""
    cell_width = (xlim[1] - xlim[0]) / shape[0]
    cell_height = (ylim[1] - ylim[0]) / shape[1]
    x_center = xlim[0] + (i + 0.5) * cell_width
    y_center = ylim[0] + (j + 0.5) * cell_height
    return x_center, y_center

def calculate_length_vectorized(coords):
    """
    Calculate the Euclidean distance (length) between consecutive points in a 2D space.
    
    Parameters:
    - coords: A NumPy array of shape (n, 2), where each row represents the (x, y) coordinates of a point.
    
    Returns:
    - lengths: A NumPy array of shape (n-1,), where each element is the Euclidean distance between consecutive points.
    """
    # Calculate the difference between consecutive points
    dx = coords[1:, 0] - coords[:-1, 0]
    dy = coords[1:, 1] - coords[:-1, 1]
    
    # Calculate the Euclidean distance between consecutive points
    lengths = np.sqrt(dx**2 + dy**2)
    
    return lengths

def coord_to_grid(coords, xlim, ylim, shape):
    """Maps coordinates to grid indices."""
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_res, y_res = shape
    i = ((coords[:, 0] - x_min) / (x_max - x_min) * x_res).astype(int)
    j = ((coords[:, 1] - y_min) / (y_max - y_min) * y_res).astype(int)
    return i, j

def cell_index(i, j, shape):
    """Maps grid indices to a 1D index."""
    x_res, y_res = shape
    return i * y_res + j

def calculate_cell_center(i, j, xlim, ylim, shape):
    """Calculates the center of a grid cell."""
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_res, y_res = shape
    cell_width = (x_max - x_min) / x_res
    cell_height = (y_max - y_min) / y_res
    x_center = x_min + (i + 0.5) * cell_width
    y_center = y_min + (j + 0.5) * cell_height
    return x_center, y_center

def calculate_length_vectorized(coords):
    """Calculate the Euclidean distance between consecutive points in a 2D space."""
    dx = coords[1:, 0] - coords[:-1, 0]
    dy = coords[1:, 1] - coords[:-1, 1]
    lengths = np.sqrt(dx**2 + dy**2)
    return lengths

def create_edge_list_and_cumulative_features(df, x_col, y_col, duration_col, xlim, ylim, shape, directed=True):
    """
    Creates an edge list and computes cumulative node features (total duration, total saccade lengths, and cell center coordinates).
    These features are normalized by their respective maximum values. Also computes edge features based on the sum of edge lengths.
    
    Parameters:
    - df: DataFrame containing the coordinates and other node features.
    - x_col: Column name in df for the x coordinates.
    - y_col: Column name in df for the y coordinates.
    - duration_col: Column name in df for the duration between consecutive points (optional).
    - xlim: Tuple (x_min, x_max) defining the bounds for the x-axis.
    - ylim: Tuple (y_min, y_max) defining the bounds for the y-axis.
    - shape: Tuple (x_res, y_res) defining the resolution of the grid.
    - directed: If True, the graph is directional; if False, bidirectional edges are created.
    
    Returns:
    - edge_list: List of edges as pairs of node indices.
    - edge_features: List of normalized edge features (sum of lengths of corresponding edges).
    - node_mapping: Mapping of old node indices to new compacted indices.
    - cumulative_node_features: A dictionary containing normalized cumulative features:
        - 'total_duration': Normalized total duration at each node.
        - 'total_saccade_length_to': Normalized total saccade length directed to each node.
        - 'total_saccade_length_from': Normalized total saccade length originating from each node.
        - 'cell_centers': Coordinates of the center of each cell as node features.
    """
    
    coords = df[[x_col, y_col]].values
    i, j = coord_to_grid(coords, xlim, ylim, shape)
    grid_indices = cell_index(i, j, shape)
    
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
    lengths = calculate_length_vectorized(coords)
    
    for k in range(len(df) - 1):
        src_node = node_mapping[grid_indices[k]]
        dst_node = node_mapping[grid_indices[k + 1]]
        
        # Handle self-loops: Only update duration
        if src_node == dst_node:
            if duration_col:
                total_durations[src_node] += df[duration_col].iloc[k]
            continue  # Skip the rest of the loop, don't add an edge
        
        # Add edge if not a self-loop
        edge = (src_node, dst_node)
        edge_list.append(edge)
        
        if not directed:
            reverse_edge = (dst_node, src_node)
            edge_list.append(reverse_edge)
            edge_length_sum[reverse_edge] = edge_length_sum.get(reverse_edge, 0) + lengths[k]
        
        # Accumulate the length of the edge
        edge_length_sum[edge] = edge_length_sum.get(edge, 0) + lengths[k]
        
        # Update cumulative saccade features (only for non-self-loops)
        total_saccade_length_to[dst_node] += lengths[k]
        total_saccade_length_from[src_node] += lengths[k]
        
        # Calculate the center coordinates of the cell
        i_node, j_node = i[k], j[k]
        x_center, y_center = calculate_cell_center(i_node, j_node, xlim, ylim, shape)
        cell_centers[src_node] = [x_center, y_center]
    
    # Normalize cumulative features by their respective maximum values
    total_durations /= np.max(total_durations)
    total_saccade_length_to /= np.max(total_saccade_length_to)
    total_saccade_length_from /= np.max(total_saccade_length_from)
    
    # Normalize edge features (sum of lengths) by their maximum value
    if edge_length_sum:  # Ensure there are edges to normalize
        max_edge_length_sum = np.max(list(edge_length_sum.values()))
        edge_features = [edge_length_sum[edge] / max_edge_length_sum for edge in edge_list]
    else:
        edge_features = []
    
    # Combine cumulative features into a dictionary
    cumulative_node_features = {
        'total_duration': total_durations,
        'total_saccade_length_to': total_saccade_length_to,
        'total_saccade_length_from': total_saccade_length_from,
        'cell_centers': cell_centers
    }
    
    return edge_list, edge_features, node_mapping, cumulative_node_features

def create_graph_data_from_dataframe(df, y, x_col, y_col, duration_col, xlim, ylim, shape, directed=True):
    """
    Converts a DataFrame into a PyTorch Geometric Data object for GCN training.
    Includes cumulative node features (total duration, total saccade length to/from node, and cell center coordinates).
    Edge features are based on the sum of lengths of corresponding edges.
    
    Parameters:
    - df: DataFrame containing the coordinates and other node features.
    - x_col: Column name in df for the x coordinates.
    - y_col: Column name in df for the y coordinates.
    - duration_col: Column name in df for the duration between consecutive points (optional).
    - xlim: Tuple (x_min, x_max) defining the bounds for the x-axis.
    - ylim: Tuple (y_min, y_max) defining the bounds for the y-axis.
    - shape: Tuple (x_res, y_res) defining the resolution of the grid.
    - directed: If True, the graph is directional; if False, bidirectional edges are created.
    
    Returns:
    - data: A PyTorch Geometric Data object containing the graph and its features.
    """
    
    # Get edge list and cumulative features
    edge_list, edge_features, node_mapping, cumulative_node_features = create_edge_list_and_cumulative_features(
        df, x_col, y_col, duration_col, xlim, ylim, shape, directed
    )
    
    # Combine cumulative features into a feature matrix
    node_features = np.hstack([
        cumulative_node_features['cell_centers'],  # Cell center coordinates
        cumulative_node_features['total_duration'].reshape(-1, 1),
        cumulative_node_features['total_saccade_length_to'].reshape(-1, 1),
        cumulative_node_features['total_saccade_length_from'].reshape(-1, 1)
    ])
    
    # Convert node features to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Convert edge list and edge features to PyTorch tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)  # Shape [num_edges, 1]
    mapping = torch.tensor(sorted(node_mapping, key=node_mapping.get))

    # Create PyTorch Geometric Data object with edge weights
    data = Data(x=x, y=torch.tensor(y), edge_index=edge_index, edge_attr=edge_attr, mapping = mapping)
    return data


_representations = {
    'heatmap': get_heatmaps
}

class Dataset2D(Dataset):
    def __init__(self, 
                 X: pd.DataFrame, 
                 y: ArrayLike, 
                 pk:List[str], 
                 shape: Union[Tuple[int], int],
                 representations: List[str], 
                 transforms=None):

        self.pmk = pk
        self.X = torch.cat((_representations[i](X, pk=pk, shape=shape, 
                return_tensors=True)[:,None, :, :] for i in representations),
                dim=1
                )
        self.channels_dim = self.X.shape[1]
        print(f'Number of channels = {self.channels_dim}.')
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):

        if self.transforms is None:
            X = self.X[idx, :, :, :]
            label = self.y.iloc[idx]

        return {
            "x": torch.tensor(X, dtype=torch.float),
            "y": torch.tensor(label, dtype=torch.long),
        }
        
get_features = ''

class DatasetTimeSeries(Dataset):
    def __init__(self, 
                 X: pd.DataFrame, 
                 y: ArrayLike, 
                 pk:List[str], 
                 shape: Union[Tuple[int], int],
                 features: List[str], 
                 transforms=None):

        self.pmk = pk
        self.X = get_features(self.features)
        self.channels_dim = self.X.shape[1]
        print(f'Number of channels = {self.channels_dim}.')
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):

        if self.transforms is None:
            X = self.X[idx, :, :,]
            label = self.y.iloc[idx]

        return {
            "x": torch.tensor(X, dtype=torch.float),
            "y": torch.tensor(label, dtype=torch.long),
        }
    
    def collate_fn(self, batch):
        lengths = [sequence.shape[0] for sequence in batch['x']]
        max_len = max(lengths)
        padded_batch = [torch.cat(sequence, torch.zeros(len(self.features),
                                                        max_len - sequence.shape[0]), axis=0) for sequence in batch]
        return torch.stack(padded_batch), torch.tensor(lengths)
    

class GridGraphDataset(Dataset):
    def __init__(self,
                 X: pd.DataFrame, 
                 y: ArrayLike, 
                 pk:List[str],
                 features: List[str], 
                 x_col, y_col, duration_col, xlim, ylim, shape, directed=True,
                 transforms=None):
        super(GridGraphDataset, self).__init__()
        self.y = y
        self.transform = transforms
        self.pk = pk
        self.X = self.X
        self.directed = directed
        self.X = self.get_graphs(x_col, y_col, duration_col, xlim, ylim, shape)

    def get_graphs(self, x_col, y_col, duration_col, xlim, ylim, shape):
        groups = self.X[self.pk].drop_duplicates().values
        graphs = []
        for i, group in enumerate(groups):
            cur_X = self.X[pd.DataFrame(self.X[self.pk] == group).all(axis=1)]
            graphs.append(create_graph_data_from_dataframe(cur_X, self.y[i], x_col, y_col, duration_col, 
                                                           xlim, ylim, shape, directed=self.directed))

        return graphs

    def len(self):
        return len(self.X)

    def get(self, idx):
        return self.X[idx]


class DatasetLightning(pl.LightningDataModule):
    def __init__(self,  
                 label_name, 
                 X: pd.DataFrame, 
                 y:ArrayLike, 
                 pk:List[str], 
                 shape: Union[Tuple[int], int],
                 representations: List[str],  
                 test_size: int, 
                 batch_size: int,
                 split_type = 'unique'):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        self.X = X
        self.y = y
        self.label_name = label_name
        self.pk = pk
        self.representations = representations
        self.test_size = test_size
        self.split_type = split_type

    def setup(self, stage=None):

        X_train, y_train, X_val, y_val = self.split_train_val()
        self.train_dataset = Dataset2D(X_train, y_train, pk=self.pk, 
                                       shape=self.shape, representations=self.representations)
        self.validation_dataset = Dataset2D(X_val, y_val, pk=self.pk, 
                                       shape=self.shape, representations=self.representations)

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        return train_loader

    def val_dataloader(self):

        valid_loader = DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        return valid_loader

    def split_train_val(self):

        pk = self.pk
        groups = self.X[self.pk].drop_duplicates()
        if len(self.pk) == 1 or self.split_type=='simple':
            if self.split_type != 'simple':
                warnings.warn('Ignoring split type. Split type cannot != "simple" if there is single primary key.')
            groups_train, groups_val = train_test_split(groups, test_size=self.test_size)
        elif self.split_type =='all_categories_unique':
            groups_train, groups_val = groups.copy(), groups.copy()
            for i in self.pk:
                gr = groups[i].drop_duplicates().sort_values().sample(frac=1-self.test_size).astype(str)
                groups_train = groups_train[groups_train[i].isin(gr)]
                groups_val = groups_val[~groups_val[i].isin(gr)]
        elif self.split_type == 'all_categories_repetetive':
            groups_train, groups_val = iterative_split(groups, self.test_size, self.pk)
        elif self.split_type == 'first_category_repetetive':
            groups_train, groups_val = train_test_split(groups, test_size=self.test_size, stratify = groups.iloc[:,0])
        elif self.split_type == 'first_category_unique':
            groups_train, groups_val = train_test_split(groups.iloc[:,0].drop_duplicates(), test_size=self.test_size)
            pk = groups.iloc[:,0]
        else:
            raise ValueError(f'''Invalid split type: {self.split_type}. 
                             Supported split types are: simple, first_category_unique, first_category_repetetive,
                             all_categories_unique, all_categories_repetetive.''')

        X_train = pd.concat(
            [
                self.X[pd.DataFrame(self.X[pk] == gr).all(axis=1)]
                for gr in groups_train
            ]
        )
        y_train = pd.concat(
            [
                self.y[pd.DataFrame(self.y[pk] == gr).all(axis=1)][self.label_name]
                for gr in groups_train
            ]
        )
        X_val = pd.concat(
            [
                self.X[pd.DataFrame(self.X[pk] == gr).all(axis=1)]
                for gr in groups_val
            ]
        )
        y_val = pd.concat(
            [
                self.y[pd.DataFrame(self.y[pk] == gr).all(axis=1)][self.label_name]
                for gr in groups_val
            ]
        )

        return X_train, y_train, X_val, y_val
