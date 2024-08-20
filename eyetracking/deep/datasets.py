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
from eyetracking.visualization.static_visualization import get_visualizations
from skmultilearn.model_selection import IterativeStratification
from functools import partial 
from eyetracking.preprocessing.base import BaseFixationPreprocessor
from eyetracking.utils import _split_dataframe
from tqdm import tqdm
from copy import copy

def iterative_split(df: pd.DataFrame, y: ArrayLike, test_size: float, stratify_columns: List[str]):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    
    Parameters:
    - df (pd.DataFrame): Input dataframe to split.
    - y (ArrayLike): Labels corresponding to the dataframe.
    - test_size (float): Proportion of the dataset to include in the test split.
    - stratify_columns (List[str]): List of column names to stratify by.

    Returns:
    - X_train (pd.DataFrame): Training split of the dataframe.
    - X_test (pd.DataFrame): Test split of the dataframe.
    - y_train (ArrayLike): Training labels.
    - y_test (ArrayLike): Test labels.

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


def _coord_to_grid(coords, xlim, ylim, shape):
    """
    Maps 2D coordinates to grid indices based on the grid resolution.

    Parameters:
    - coords (np.array): Array of coordinates to map.
    - xlim (Tuple): The x-axis limits (x_min, x_max).
    - ylim (Tuple): The y-axis limits (y_min, y_max).
    - shape (Tuple): The shape of the grid (rows, cols).

    Returns:
    - i, j (np.array): The grid indices corresponding to the coordinates.
    """
     
    i = (((coords[:, 0] - xlim[0]) / (xlim[1] - xlim[0]))*shape[0]).astype(int)
    j = (((coords[:, 1] - ylim[0]) / (ylim[1] - ylim[0]))*shape[1]).astype(int)
    return i, j

def _cell_index(i, j, shape):
    """
    Maps grid indices (i, j) to a 1D cell index based on the grid shape.
    
    Parameters:
    - i (int): Row index in the grid.
    - j (int): Column index in the grid.
    - shape (Tuple): The shape of the grid (rows, cols).

    Returns:
    - index (int): The 1D cell index.
    """
     
    return i * shape[1] + j

def _calculate_cell_center(i, j, xlim, ylim, shape):
    """
    Calculates the center coordinates of a grid cell.

    Parameters:
    - i (int): Row index of the cell.
    - j (int): Column index of the cell.
    - xlim (Tuple): Limits of the x-axis (x_min, x_max).
    - ylim (Tuple): Limits of the y-axis (y_min, y_max).
    - shape (Tuple): Shape of the grid (rows, cols).

    Returns:
    - x_center, y_center (float): Center coordinates of the grid cell.
    """
    cell_width = (xlim[1] - xlim[0]) / shape[0]
    cell_height = (ylim[1] - ylim[0]) / shape[1]
    x_center = xlim[0] + (i + 0.5) * cell_width
    y_center = ylim[0] + (j + 0.5) * cell_height
    return x_center, y_center

def _calculate_length_vectorized(coords):
    """
    Calculates the Euclidean distance between consecutive points in 2D space.

    Parameters:
    - coords (np.array): Array of coordinates with shape (n, 2).

    Returns:
    - lengths (np.array): Euclidean distances between consecutive points.
    """
    # Calculate the difference between consecutive points
    dx = coords[1:, 0] - coords[:-1, 0]
    dy = coords[1:, 1] - coords[:-1, 1]
    
    # Calculate the Euclidean distance between consecutive points
    lengths = np.sqrt(dx**2 + dy**2)
    
    return lengths

def create_edge_list_and_cumulative_features(df, add_duration, x_col, y_col, 
                                             xlim, ylim, shape, directed=True):
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
    i, j = _coord_to_grid(coords, xlim, ylim, shape)
    grid_indices = _cell_index(i, j, shape)
    #print(grid_indices)
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
                total_durations[src_node] += df['duration'].iloc[k]
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
        x_center, y_center = _calculate_cell_center(i_node, j_node, xlim, ylim, shape)
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

def create_graph_data_from_dataframe(df, y, x_col, y_col, add_duration,
                                     xlim, ylim, shape, directed=True):
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
        df, add_duration, x_col, y_col, xlim, ylim, shape, directed
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
    'heatmap': get_heatmaps,
    'baseline_visualization' : partial(get_visualizations, pattern='baseline'),
    'aoi_visualization' : partial(get_visualizations, pattern='aoi'),
    'saccade_visualization' : partial(get_visualizations, pattern='saccades')
 }

class Dataset2D(Dataset):

    """
    Custom dataset for 2D image-based representations derived from gaze data.

    Parameters:
    - X (pd.DataFrame): Input data.
    - y (ArrayLike): Labels for the data.
    - pk (List[str]): List of primary keys for grouping.
    - shape (Union[Tuple[int], int]): Shape of the images.
    - representations (List[str]): List of representation types.
    - upload_to_cuda (bool): If True, upload the data to the GPU. Default: False.
    - transforms (Optional): Transformations to apply to the data.
    """
    def __init__(self, 
                 X: pd.DataFrame,
                 x: str,
                 y: str,  
                 Y: ArrayLike, 
                 pk:List[str], 
                 shape: Union[Tuple[int], int],
                 representations: List[str], 
                 upload_to_cuda: bool = False,
                 transforms=None):

        self.pmk = pk
        self.X = torch.cat([torch.tensor(_representations[i](X, x, y, 
                            pk=pk, shape=shape), dtype=torch.float32) for i in representations], dim=1)
        self.channels_dim = self.X.shape[1]
        print(f'Number of channels = {self.channels_dim}.')
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

        if self.transforms is None:
            X = self.X[idx, :, :, :]
            label = self.y[idx]

        return {
            "images": X,
            "y": label, #
        }
    
    def collate_fn(self, batch):
        images = torch.stack([x["images"] for x in batch])
        y = torch.tensor([x['y'] for x in batch])
        return {"images" : images,
                'y': y}
        return batch
        
def _get_features(X, features, x, y, t, pk):
    preprocessor = BaseFixationPreprocessor(x, y, t, pk)
    features_to_get = copy(features)
    for i in features:
        if i in X.columns:
            features_to_get.remove(i)
    output = []
    groups = _split_dataframe(X, pk)
    for group_id, group_X in tqdm(groups):
        cur_X = preprocessor._compute_feats(group_X, features_to_get)
        output.append(cur_X[[x,y]+features].values)
    
    return output

class DatasetTimeSeries(Dataset):

    """
    Custom dataset for time-series data.

    Parameters:
    - X (pd.DataFrame): Input time-series data.
    - y (ArrayLike): Labels for the data.
    - pk (List[str]): Primary keys for grouping.
    - shape (Union[Tuple[int], int]): Shape of the data samples.
    - features (List[str]): List of features to extract.
    - upload_to_cuda (bool): If True, upload the data to the GPU. Default: False.
    - transforms (Optional): Transformations to apply to the data.
    """
    def __init__(self, 
                 X: pd.DataFrame, 
                 Y: ArrayLike, 
                 x: str,
                 y: str,
                 pk:List[str], 
                 features: List[str], 
                 transforms=None):

        self.pmk = pk
        self.X = _get_features(X, features, x, y, t=None, pk=pk)
        if not isinstance(Y, pd.Series):
            Y = Y.set_index(pk).squeeze(axis=0)
        self.Y = Y.sort_index().values
        if np.issubdtype(self.Y.dtype, np.integer):
            self.Y = torch.tensor(self.Y, dtype=torch.long)
        else:
            self.Y = torch.tensor(self.Y, dtype=torch.float)
        
        self.n_features = 2+len(features)
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        if self.transforms is None:
            X = self.X[idx]
            label = self.Y[idx]

        return {
            "sequences": torch.tensor(X, dtype=torch.float),
            "y": label,
        }
    
    def collate_fn(self, batch):
        lengths = [x["sequences"].shape[0] for x in batch]
        max_len = max(lengths)
        padded_batch = [torch.cat([x["sequences"], torch.zeros(max_len - x["sequences"].shape[0], 
                                                        self.n_features)], axis=0) for x in batch]
        
        y = torch.tensor([x['y'] for x in batch])
        return {"sequences": torch.stack(padded_batch),
                 "lengths" : torch.tensor(lengths), 
                 'y': y}
    
class TimeSeries_2D_Dataset(Dataset):

    """
    Composite dataset that combines image and time-series data.

    Parameters:
    - image_dataset (Dataset): Dataset containing image data.
    - sequence_dataset (Dataset): Dataset containing sequence data.
    """
    def __init__(self, image_dataset, sequence_dataset):
        # Ensure both datasets have the same length
        assert len(image_dataset) == len(sequence_dataset), "Datasets must be of the same length"
        
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
        
        return {
            'images': image,
            'sequences': torch.tensor(sequence),
            'y': y
        }
    def collate_fn(self, batch):
        lengths = [x["sequences"].shape[0] for x in batch]
        max_len = max(lengths)
        padded_batch = [torch.cat([x["sequences"], torch.zeros(max_len - x["sequences"].shape[0], 
                                                        self.n_features)], axis=0) for x in batch]
        
        y = torch.tensor([x['y'] for x in batch])

        return {"sequences": torch.stack(padded_batch),
                "lengths" : torch.tensor(lengths),
                "images":batch['images'], 
                'y': y}
    
    

class GridGraphDataset(Dataset):

    """
    Custom dataset for generating grid-based graph representations from spatial coordinates.

    Parameters:
    - X (pd.DataFrame): Input dataframe.
    - y (ArrayLike): Labels for the data.
    - pk (List[str]): Primary keys for grouping.
    - features (List[str]): List of features to extract.
    - x_col, y_col (str): Column names for x and y coordinates.
    - duration_col (str): Column name for time durations.
    - xlim (Tuple): Limits of the x-axis.
    - ylim (Tuple): Limits of the y-axis.
    - shape (Tuple): Shape of the grid.
    - directed (bool): Whether the graph is directed.
    - transforms (Optional): Transformations to apply to the data.
    """
    def __init__(self,
                 X: pd.DataFrame, 
                 Y: ArrayLike, 
                 x: str, 
                 y: str,
                 add_duration: bool,
                 pk:List[str],
                 xlim: Tuple[float, float] = (0,1), 
                 ylim: Tuple[float, float] = (0,1),
                 shape : Tuple[int, int] = (10,10),
                 directed=True,
                 transforms=None):
        super(GridGraphDataset, self).__init__()
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
        for i, (group_id, cur_X) in tqdm(enumerate(groups), desc='Getting graphs...'):
            graphs.append(create_graph_data_from_dataframe(cur_X, Y[i], x_col, y_col, add_duration,
                                                           xlim, ylim, shape, directed=self.directed))

        return graphs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    
    def collate_fn(self, batch):
        return batch


class DatasetLightningBase(pl.LightningDataModule):
    """
    Base PyTorch Lightning DataModule for managing datasets and dataloaders.

    Parameters:
    - X (pd.DataFrame): Input data.
    - y (ArrayLike): Labels for the data.
    - pk (List[str]): Primary keys for grouping.
    - test_size (int): Test size for the train-validation split.
    - batch_size (int): Batch size for the dataloaders.
    - split_type (str): Type of train-validation split.
    """
    def __init__(self,  
                 X: pd.DataFrame, 
                 Y: ArrayLike, 
                 x: str,
                 y:str,
                 pk: List[str], 
                 test_size: int, 
                 batch_size: int, 
                 split_type: str = 'simple'):
        super().__init__()
        self.X = X
        self.Y = Y

        self.x = x
        self.y = y
        self.pk = pk
        self.test_size = test_size
        self.batch_size = batch_size
        self.split_type = split_type

    def setup(self, stage=None):
        X_train, y_train, X_val, y_val = self.split_train_val()
        self.create_train_val_datasets(X_train, y_train, X_val, y_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, 
            shuffle=False, collate_fn=self.val_dataset.collate_fn
        )

    def split_train_val(self):
        pk = self.pk
        groups = self.X[self.pk].drop_duplicates()
        if len(self.pk) == 1 or self.split_type == 'simple':
            if self.split_type != 'simple':
                warnings.warn('''Ignoring split type. 
                              Split type cannot != "simple" if there is single primary key.''')
            groups_train, groups_val = train_test_split(groups, test_size=self.test_size)
        elif self.split_type == 'all_categories_unique':
            groups_train, groups_val = groups.copy(), groups.copy()
            for i in self.pk:
                gr = groups[i].drop_duplicates().sort_values().sample(frac=1-self.test_size).astype(str)
                groups_train = groups_train[groups_train[i].isin(gr)]
                groups_val = groups_val[~groups_val[i].isin(gr)]
        elif self.split_type == 'all_categories_repetetive':
            groups_train, groups_val = iterative_split(groups, self.test_size, self.pk)
        elif self.split_type == 'first_category_repetetive':
            groups_train, groups_val = train_test_split(groups, 
                                                        test_size=self.test_size, 
                                                        stratify=groups.iloc[:, 0])
        elif self.split_type == 'first_category_unique':
            groups_train, groups_val = train_test_split(groups.iloc[:, 0].drop_duplicates(), 
                                                        test_size=self.test_size)
            pk = groups.iloc[:, 0]
        else:
            raise ValueError(f'''Invalid split type: {self.split_type}. 
                             Supported split types are: simple, first_category_unique,
                             first_category_repetetive, 
                             all_categories_unique, all_categories_repetetive.''')

        X_train = self.X.merge(groups_train, on=pk)
        y_train = self.Y.merge(groups_train, on=pk)
        X_val = self.X.merge(groups_val, on=pk)
        y_val = self.Y.merge(groups_val, on=pk)

        return X_train, y_train, X_val, y_val

    def create_train_val_datasets(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    
class DatasetLightning2D(DatasetLightningBase):
    """
    PyTorch Lightning DataModule for 2D datasets and dataloaders.
    """

    def __init__(self,  
                 X: pd.DataFrame, 
                 Y: ArrayLike, 
                 x: str,
                 y: str,
                 pk: List[str], 
                 shape: Union[Tuple[int], int],
                 representations: List[str],  
                 test_size: int, 
                 batch_size: int, 
                 split_type: str = 'simple'):
        super().__init__(X, Y, x, y, pk, test_size, batch_size, split_type)
        self.shape = shape
        self.representations = representations

    def create_train_val_datasets(self, X_train, y_train, X_val, y_val):
        self.train_dataset = Dataset2D(X_train, self.x, self.y, y_train,
                                        pk=self.pk, shape=self.shape, representations=self.representations)
        self.val_dataset = Dataset2D(X_val, self.x, self.y, y_val, 
                                     pk=self.pk, shape=self.shape, representations=self.representations)


class DatasetLightningTimeSeries(DatasetLightningBase):
    """
    PyTorch Lightning DataModule for Time Series datasets and dataloaders.
    """

    def __init__(self,  
                X: pd.DataFrame, 
                 Y: ArrayLike, 
                 x: str,
                 y: str,
                 pk: List[str], 
                 features: List[str],  
                 test_size: int, 
                 batch_size: int, 
                 split_type: str = 'simple'):
        super().__init__(X, Y, x, y, pk, test_size, batch_size, split_type)
        self.features = features

    def create_train_val_datasets(self, X_train, y_train, X_val, y_val):
        self.train_dataset = DatasetTimeSeries(X_train, self.x, self.y, 
                                               y_train, pk=self.pk, features=self.features)
        self.val_dataset = DatasetTimeSeries(X_val, self.x, self.y,
                                              y_val, pk=self.pk, features=self.features)


class DatasetLightningTimeSeries2D(DatasetLightningBase):
    """
    PyTorch Lightning DataModule for Time Series 2D datasets and dataloaders.
    """

    def __init__(self,  
                 X: pd.DataFrame, 
                 Y: ArrayLike, 
                 x: str,
                 y: str,
                 pk: List[str], 
                 shape: Union[Tuple[int], int],
                 representations: List[str],
                 features: List[str],  
                 test_size: int, 
                 batch_size: int, 
                 split_type: str = 'simple'):
        super().__init__(X, Y, x, y, pk, test_size, batch_size, split_type)
        self.shape = shape
        self.representations = representations
        self.features = features

    def create_train_val_datasets(self, X_train, y_train, X_val, y_val):
        data2D = Dataset2D(X_train, self.x, self.y, y_train, pk=self.pk, 
                           shape=self.shape, representations=self.representations)
        dataTime = DatasetTimeSeries(X_train, self.x, self.y, 
                                     y_train, pk=self.pk, features=self.features)
        self.train_dataset = TimeSeries_2D_Dataset(data2D, dataTime)
