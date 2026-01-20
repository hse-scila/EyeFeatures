"""Tests for eyefeatures/visualization/dynamic_visualization.py."""

import os

import pandas as pd
import plotly.graph_objects as go
import pytest

from eyefeatures.visualization.dynamic_visualization import (
    scanpath_animation,
    tracker_animation,
)


@pytest.fixture
def dyn_viz_df():
    """Sample dataframe for dynamic visualization tests."""
    df = pd.DataFrame(
        {
            "x": [10, 20, 30, 40],
            "y": [10, 20, 30, 40],
            "participant": ["p1"] * 4,
            "AOI": ["A", "A", "B", "B"],
            "duration": [1, 2, 3, 4],
        }
    )
    return df


class TestDynamicVisualization:
    """Tests for dynamic visualization (animations)."""

    def test_tracker_animation_show(self, dyn_viz_df, monkeypatch):
        """Test tracker_animation calls show."""
        show_called = False

        def mock_show(*args, **kwargs):
            nonlocal show_called
            show_called = True

        monkeypatch.setattr(go.Figure, "show", mock_show)

        tracker_animation(dyn_viz_df, x="x", y="y", aoi="AOI")
        assert show_called

    def test_scanpath_animation_show(self, dyn_viz_df, monkeypatch):
        """Test scanpath_animation calls show."""
        show_called = False

        def mock_show(*args, **kwargs):
            nonlocal show_called
            show_called = True

        monkeypatch.setattr(go.Figure, "show", mock_show)

        scanpath_animation(dyn_viz_df, x="x", y="y")
        assert show_called

    def test_tracker_animation_save_gif(self, dyn_viz_df, monkeypatch, tmp_path):
        """Test tracker_animation saving to GIF."""
        # Mock show to avoid browser window
        monkeypatch.setattr(go.Figure, "show", lambda self: None)

        gif_path = tmp_path / "test_tracker.gif"
        # Using a small frames_count or small df to keep it fast
        tracker_animation(dyn_viz_df.iloc[:2], x="x", y="y", save_gif=str(gif_path))

        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0

    def test_scanpath_animation_save_gif(self, dyn_viz_df, monkeypatch, tmp_path):
        """Test scanpath_animation saving to GIF."""
        monkeypatch.setattr(go.Figure, "show", lambda self: None)

        gif_path = tmp_path / "test_scanpath.gif"
        scanpath_animation(dyn_viz_df.iloc[:2], x="x", y="y", save_gif=str(gif_path))

        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0

    def test_tracker_animation_with_metadata(self, dyn_viz_df, monkeypatch):
        """Test tracker_animation with meta_data."""
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        # Should not raise error
        tracker_animation(
            dyn_viz_df, x="x", y="y", meta_data=["duration", "participant"]
        )
