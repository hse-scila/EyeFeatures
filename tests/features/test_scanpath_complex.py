"""Tests for eyefeatures/features/scanpath_complex.py - Scanpaths."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.features.scanpath_complex import (
    _get_fill_path,
    dimensionality_reduction_order,
    get_compromise_matrix,
    get_expected_path,
    get_sim_matrix,
    hierarchical_clustering_order,
    optimal_leaf_ordering_clustering,
    spectral_order,
)


@pytest.fixture
def simple_paths_df():
    """Simple paths for expected path calculation."""
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 1.0, 0.2, 1.2],  # Slight offset for second participant
            "duration": [100, 100, 100, 100],
            "participant": ["p1", "p1", "p2", "p2"],
            "stimulus": ["s1", "s1", "s1", "s1"],
        }
    )


def test_get_expected_path(simple_paths_df):
    """Test calculation of expected path (mean path)."""
    # Expected path for s1 should be mean of p1 and p2
    # p1: (0,0), (1,1)
    # p2: (0,0.2), (1,1.2)
    # Mean: (0, 0.1), (1, 1.1)

    ep = get_expected_path(
        data=simple_paths_df,
        x="x",
        y="y",
        pk=["participant", "stimulus"],
        path_pk=["stimulus"],
        method="mean",
    )

    assert isinstance(ep, dict)
    assert "s1" in ep
    assert isinstance(ep["s1"], pd.DataFrame)
    assert len(ep["s1"]) == 2

    # Check mean calculation values
    res = ep["s1"]
    assert np.allclose(res["x_est"].values, [0.0, 1.0])
    assert np.allclose(res["y_est"].values, [0.1, 1.1])


def test_get_expected_path_invalid_pk():
    """Test validation of pk and path_pk relationship."""
    df = pd.DataFrame({"x": [1], "y": [1], "p": ["1"]})
    with pytest.raises(ValueError, match="path_pk must be a subset of pk"):
        get_expected_path(data=df, x="x", y="y", pk=["p"], path_pk=["other"])


def test_get_fill_path():
    """Test fill path calculation (mean of expected paths)."""
    # Create two dataframes representing expected paths
    df1 = pd.DataFrame({"x": [10, 20], "y": [0, 1]})
    df2 = pd.DataFrame({"x": [0, 2], "y": [2, 3]})
    data = [df1, df2]

    fill = _get_fill_path(data, x="x", y="y", method="mean")

    assert isinstance(fill, pd.DataFrame)
    assert np.allclose(fill["x_est"].values, [5, 11])
    assert np.allclose(fill["y_est"].values, [1, 2])


def test_get_sim_matrix():
    """Test similarity matrix computation."""

    # define dummy metric: 1.0 for identical, lower for different
    def dummy_metric(p1, p2):
        dist = np.sum(np.abs(p1 - p2))
        return 1.0 / (1.0 + dist)

    # Two identical paths, one different
    p1 = np.array([[0, 0], [1, 1]])
    p2 = np.array([[0, 0], [1, 1]])
    p3 = np.array([[1, 1], [2, 2]])  # Diff from p1 is 4. Sim = 1/5 = 0.2

    scanpaths = [p1, p2, p3]

    sim_mat = get_sim_matrix(scanpaths, dummy_metric)

    assert sim_mat.shape == (3, 3)
    # p1, p2 identical. sim=1.
    # M[0,1]=1. M[1,0]=1. M+=T -> 2. Diag=2 (from init 1+1).
    # Norm by max=2. Result 1.0.
    assert np.isclose(sim_mat[0, 1], 1.0)

    assert np.allclose(sim_mat, sim_mat.T)


def test_clustering_orders():
    """Test matrix reordering functions."""
    # Create a simple symmetric matrix
    mat = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])

    # Hierarchical
    order = hierarchical_clustering_order(mat)
    # Functions return reordered MATRIX (np.ndarray), not list of indices!
    assert isinstance(order, np.ndarray)
    assert order.shape == mat.shape
    assert np.allclose(order, order.T)  # Symmetric

    # Optimal leaf
    order_opt = optimal_leaf_ordering_clustering(mat)
    assert order_opt.shape == mat.shape

    # Spectral
    order_spec = spectral_order(mat)
    assert order_spec.shape == mat.shape

    # MDS
    order_mds = dimensionality_reduction_order(mat)
    assert order_mds.shape == mat.shape


def test_compromise_matrix():
    """Test compromise matrix logic (RV coeff, etc)."""
    # 3x3 distance matrix
    D1 = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    D2 = np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]])  # D2 = 2*D1

    matrices = [D1, D2]

    comp_mat = get_compromise_matrix(matrices)

    # Result should be a cross-product matrix (n x n)
    assert comp_mat.shape == (3, 3)
