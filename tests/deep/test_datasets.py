"""Tests for eyefeatures/deep/datasets.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.deep.datasets import (
    Dataset2D,
    DatasetTimeSeries,
    GridGraphDataset,
    _calculate_cell_center,
    _cell_index,
    _coord_to_grid,
    create_graph_data_from_dataframe,
    iterative_split,
)


@pytest.fixture
def deep_sample_df():
    """Synthetic fixation data for deep learning tests."""
    df = pd.DataFrame(
        {
            "x": [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 120.0, 220.0],
            "y": [100.0, 150.0, 200.0, 100.0, 200.0, 150.0, 110.0, 160.0],
            "t": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "duration": [50, 60, 55, 70, 65, 55, 52, 58],
            "participant": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "stimulus": ["s1", "s1", "s1", "s1", "s1", "s1", "s1", "s1"],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    return df


@pytest.fixture
def y_labels(deep_sample_df):
    """Labels for stratification."""
    return deep_sample_df[["participant", "stimulus", "label"]].drop_duplicates()


def test_iterative_split():
    """Test iterative_split function."""
    df = pd.DataFrame({"a": [1, 1, 0, 0, 1, 1, 0, 0], "b": [1, 0, 1, 0, 1, 0, 1, 0]})
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

    X_train, X_test, y_train, y_test = iterative_split(
        df, y, test_size=0.5, stratify_columns=["a", "b"]
    )

    assert len(X_train) == 4
    assert len(X_test) == 4


def test_coord_helpers():
    """Test coordinate transformation helpers."""
    coords = np.array([[50, 50], [150, 150]])
    xlim = (0, 200)
    ylim = (0, 200)
    shape = (10, 10)

    i, j = _coord_to_grid(coords, xlim, ylim, shape)
    assert i[0] == 2  # (50/200)*10 = 2.5 -> 2
    assert j[0] == 2
    assert i[1] == 7  # (150/200)*10 = 7.5 -> 7
    assert j[1] == 7

    idx = _cell_index(2, 2, shape)
    assert idx == 2 * 10 + 2

    cx, cy = _calculate_cell_center(2, 2, xlim, ylim, shape)
    # cell_width = 20, cx = 0 + (2+0.5)*20 = 50
    assert cx == 50.0
    assert cy == 50.0


def test_create_graph_data(deep_sample_df):
    """Test graph data generation."""
    data_p1 = deep_sample_df[deep_sample_df["participant"] == "p1"]

    data = create_graph_data_from_dataframe(
        data_p1,
        y=0,
        x_col="x",
        y_col="y",
        add_duration="duration",
        xlim=(0, 500),
        ylim=(0, 500),
        shape=(10, 10),
    )

    assert data.x.shape[1] == 5  # cx, cy, dur, deg_to, deg_from
    assert data.y == 0
    assert data.edge_index.ndim == 2


def test_dataset_2d(deep_sample_df):
    """Test Dataset2D."""
    # Group labels
    Y = deep_sample_df[["participant", "stimulus", "label"]].drop_duplicates()

    ds = Dataset2D(
        deep_sample_df,
        Y,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        shape=(10, 10),
        representations=["heatmap"],
    )

    assert len(ds) == 2
    item = ds[0]
    assert "images" in item
    assert "y" in item
    assert item["images"].shape == (1, 10, 10)


def test_dataset_time_series(deep_sample_df):
    """Test DatasetTimeSeries."""
    Y = deep_sample_df[["participant", "stimulus", "label"]].drop_duplicates()

    # We need some dummy features like 'duration'
    ds = DatasetTimeSeries(
        deep_sample_df,
        Y,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        features=["duration"],
        max_length=10,
    )

    assert len(ds) == 2
    item = ds[0]
    assert "sequences" in item
    assert "y" in item
    # x, y + duration = 3 features
    assert item["sequences"].shape[1] == 3
    assert item["sequences"].shape[0] == 4  # 4 fixations for p1


def test_time_series_2d_dataset(deep_sample_df):
    """Test TimeSeries_2D_Dataset (composite)."""
    # Import inside to avoid issues if not needed elsewhere
    from eyefeatures.deep.datasets import TimeSeries_2D_Dataset

    Y = deep_sample_df[["participant", "stimulus", "label"]].drop_duplicates()
    ds2d = Dataset2D(
        deep_sample_df,
        Y,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        shape=(10, 10),
        representations=["heatmap"],
    )
    dsts = DatasetTimeSeries(
        deep_sample_df,
        Y,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        features=["duration"],
        max_length=10,
    )

    composite = TimeSeries_2D_Dataset(ds2d, dsts)
    assert len(composite) == 2
    item = composite[0]
    assert "images" in item
    assert "sequences" in item
    assert "y" in item


def test_grid_graph_dataset(deep_sample_df):
    """Test GridGraphDataset."""
    Y = deep_sample_df[["participant", "stimulus", "label"]].drop_duplicates()

    ds = GridGraphDataset(
        deep_sample_df,
        Y,
        x="x",
        y="y",
        add_duration="duration",
        pk=["participant", "stimulus"],
        xlim=(0, 500),
        ylim=(0, 500),
        shape=(5, 5),
    )

    assert len(ds) == 2
    graph = ds[0]
    assert hasattr(graph, "x")
    assert hasattr(graph, "edge_index")


def test_lightning_datamodules(deep_sample_df):
    """Test PyTorch Lightning DataModules."""
    from eyefeatures.deep.datasets import DatasetLightning2D, DatasetLightningTimeSeries

    Y = deep_sample_df[["participant", "stimulus", "label"]].drop_duplicates()
    # Need to group X and Y by PK to pass to DataModule split logic
    X_grouped = deep_sample_df
    Y_grouped = Y

    dm2d = DatasetLightning2D(
        X_grouped,
        Y_grouped,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        shape=(10, 10),
        representations=["heatmap"],
        test_size=0.5,
        batch_size=1,
    )
    dm2d.setup()
    assert len(dm2d.train_dataloader()) == 1
    assert len(dm2d.val_dataloader()) == 1

    dmts = DatasetLightningTimeSeries(
        X_grouped,
        Y_grouped,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        features=["duration"],
        test_size=0.5,
        batch_size=1,
    )
    dmts.setup()
    assert len(dmts.train_dataloader()) == 1

    # Test another split type
    dm2d_alt = DatasetLightning2D(
        X_grouped,
        Y_grouped,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        shape=(10, 10),
        representations=["heatmap"],
        test_size=0.5,
        batch_size=1,
        split_type="first_category_unique",
    )
    dm2d_alt.setup()
    assert dm2d_alt.train_dataset is not None
