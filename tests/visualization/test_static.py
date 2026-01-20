"""Tests for eyefeatures/visualization/static_visualization.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.visualization.static_visualization import (
    Visualization,
    aoi_visualization,
    baseline_visualization,
    get_visualizations,
    saccade_visualization,
    scanpath_visualization,
)


@pytest.fixture
def viz_df():
    """Sample dataframe for visualization tests."""
    df = pd.DataFrame(
        {
            "x": [10, 20, 10, 100, 110, 100],
            "y": [10, 10, 20, 100, 100, 110],
            "AOI": ["aoi_1"] * 3 + ["aoi_2"] * 3,
            "participant": ["p1"] * 6,
            "stimulus": ["s1"] * 6,
            "duration": [100, 200, 300, 100, 200, 300],
        }
    )
    return df


class TestStaticVisualization:
    """Tests for static visualization functions and classes."""

    def test_scanpath_visualization_basic(self, viz_df):
        """Test basic scanpath visualization."""
        arr = scanpath_visualization(
            viz_df,
            x="x",
            y="y",
            show_plot=False,
            return_ndarray=True,
            fig_size=(5, 5),
            dpi=50,
        )
        assert arr is not None
        assert isinstance(arr, np.ndarray)
        # Check shape: (height, width, channels)
        # 5 inches * 50 dpi = 250px
        assert arr.shape[0] > 0
        assert arr.shape[1] > 0

    def test_scanpath_visualization_aoi(self, viz_df):
        """Test scanpath visualization with AOI and hulls."""
        arr = scanpath_visualization(
            viz_df,
            x="x",
            y="y",
            aoi="AOI",
            show_hull=True,
            show_plot=False,
            return_ndarray=True,
            fig_size=(5, 5),
            dpi=50,
        )
        assert arr is not None

    def test_visualization_class(self, viz_df):
        """Test the Visualization estimator class."""
        v = Visualization(x="x", y="y", aoi="AOI", fig_size=(5, 5))
        arr = v.fit_transform(viz_df)
        assert arr is not None
        assert isinstance(arr, np.ndarray)

    def test_get_visualizations_baseline(self, viz_df):
        """Test get_visualizations meta-function with baseline pattern."""
        res = get_visualizations(
            viz_df,
            x="x",
            y="y",
            shape=(5, 5),
            pattern="baseline",
            pk=["participant"],
            dpi=50,
        )
        # res shape should be [N_groups, C, H, W]
        assert res.ndim == 4
        assert res.shape[0] == 1  # One participant

    def test_get_visualizations_aoi(self, viz_df):
        """Test get_visualizations meta-function with aoi pattern."""
        res = get_visualizations(
            viz_df,
            x="x",
            y="y",
            shape=(5, 5),
            pattern="aoi",
            pk=["participant"],
            dpi=50,
        )
        assert res.ndim == 4

    def test_get_visualizations_no_pk(self, viz_df):
        """Test get_visualizations without primary keys."""
        res = get_visualizations(
            viz_df, x="x", y="y", shape=(5, 5), pattern="baseline", pk=None, dpi=50
        )
        assert res.ndim == 4
        assert res.shape[0] == 1

    def test_specialized_visualizations(self, viz_df):
        """Test baseline_visualization, aoi_visualization, saccade_visualization."""
        res_baseline = baseline_visualization(
            viz_df, x="x", y="y", shape=(5, 5), dpi=50
        )
        assert res_baseline is not None

        res_aoi = aoi_visualization(
            viz_df, x="x", y="y", shape=(5, 5), aoi="AOI", dpi=50
        )
        assert res_aoi is not None

        res_saccade = saccade_visualization(viz_df, x="x", y="y", shape=(5, 5), dpi=50)
        assert res_saccade is not None

    def test_scanpath_visualization_vectors(self, viz_df):
        """Test visualization with vectors (quiver)."""
        arr = scanpath_visualization(
            viz_df,
            x="x",
            y="y",
            is_vectors=True,
            show_plot=False,
            return_ndarray=True,
            fig_size=(5, 5),
            dpi=50,
        )
        assert arr is not None
