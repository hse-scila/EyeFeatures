"""Tests for eyefeatures/preprocessing/aoi_extraction.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.preprocessing.aoi_extraction import (
    AOIExtractor,
    GradientBased,
    OverlapClustering,
    ShapeBased,
    ThresholdBased,
)

NUM_POINTS_PER_CLUSTER = 20


@pytest.fixture
def aoi_test_df():
    """Synthetic fixation data for AOI tests.

    Contains enough points for KDE-based methods and necessary columns
    for OverlapClustering.
    """
    np.random.seed(42)
    # Cluster 1 around (10, 10)
    x1 = np.random.normal(10, 1, NUM_POINTS_PER_CLUSTER)
    y1 = np.random.normal(10, 1, NUM_POINTS_PER_CLUSTER)
    # Cluster 2 around (100, 100)
    x2 = np.random.normal(100, 1, NUM_POINTS_PER_CLUSTER)
    y2 = np.random.normal(100, 1, NUM_POINTS_PER_CLUSTER)
    # Cluster 3 around (200, 200)
    x3 = np.random.normal(200, 1, NUM_POINTS_PER_CLUSTER)
    y3 = np.random.normal(200, 1, NUM_POINTS_PER_CLUSTER)

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])

    total_points = NUM_POINTS_PER_CLUSTER * 3
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "participant": ["p1"] * total_points,
            "stimulus": ["s1"] * total_points,
            "diameters": [5.0] * total_points,
        }
    )

    # Add centers column for OverlapClustering
    df["centers"] = df.apply(lambda row: np.array([row["x"], row["y"]]), axis=1)

    return df


class TestAOIExtraction:
    """Tests for AOI extraction methods."""

    def test_shape_based(self, aoi_test_df):
        """Test ShapeBased AOI extraction."""
        # Define two rectangles
        shapes = [
            [
                ("r", (0, 0), (25, 25)),  # Rect 1 covers Cluster 1
                ("r", (90, 90), (115, 115)),  # Rect 2 covers Cluster 2
            ]
        ]
        sb = ShapeBased(x="x", y="y", shapes=shapes, aoi_name="AOI")
        result = sb.fit_transform(aoi_test_df)

        assert "AOI" in result.columns
        # First cluster of points should be around aoi_0
        assert (result.iloc[:NUM_POINTS_PER_CLUSTER]["AOI"] == "aoi_0").any()
        # Second cluster of points should be around aoi_1
        assert (
            result.iloc[NUM_POINTS_PER_CLUSTER : 2 * NUM_POINTS_PER_CLUSTER]["AOI"]
            == "aoi_1"
        ).any()

    def test_threshold_based(self, aoi_test_df):
        """Test ThresholdBased AOI extraction."""
        tb = ThresholdBased(
            x="x",
            y="y",
            window_size=5,
            threshold=0.0001,
            aoi_name="AOI",
            algorithm_type="kmeans",
        )
        result = tb.fit_transform(aoi_test_df)

        assert "AOI" in result.columns
        assert result["AOI"].nunique() >= 2

    def test_gradient_based(self, aoi_test_df):
        """Test GradientBased AOI extraction."""
        gb = GradientBased(
            x="x", y="y", window_size=5, threshold=0.0001, aoi_name="AOI"
        )
        result = gb.fit_transform(aoi_test_df)
        assert "AOI" in result.columns
        assert result["AOI"].nunique() >= 1

    def test_overlap_clustering(self, aoi_test_df):
        """Test OverlapClustering AOI extraction."""
        oc = OverlapClustering(
            x="x",
            y="y",
            diameters="diameters",
            centers="centers",
            pk=["participant"],
            aoi_name="AOI",
        )
        result = oc.fit_transform(aoi_test_df)
        assert "AOI" in result.columns
        assert result["AOI"].nunique() >= 2

    def test_aoi_extractor(self, aoi_test_df):
        """Test meta AOIExtractor."""
        methods = [
            ThresholdBased(window_size=5, threshold=0.0001, algorithm_type="kmeans"),
        ]

        extractor = AOIExtractor(
            methods=methods,
            x="x",
            y="y",
            pk=["participant"],
            instance_columns=["participant", "stimulus"],
            aoi_name="AOI_best",
        )

        result = extractor.fit_transform(aoi_test_df)
        assert "AOI_best" in result.columns
        assert result["AOI_best"].nunique() >= 1
