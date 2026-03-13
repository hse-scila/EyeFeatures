"""Tests for eyefeatures/features/scanpath_dist.py - Distance transformers."""

import pandas as pd
import pytest

from eyefeatures.features.scanpath_dist import (
    DistanceTransformer,
    MultiMatchDist,
    ScanMatchDist,
    SimpleDistances,
)


@pytest.fixture
def norm_df(sample_df):
    """Normalized sample DataFrame for distance calculations (requires 0-1 range)."""
    df = sample_df.copy()
    # sample_df values are ~100-400, so dividing by 1000 ensures 0-1 range
    df["x"] = df["x"] / 1000.0
    df["y"] = df["y"] / 1000.0
    return df


class TestDistanceTransformer:
    """Tests for base DistanceTransformer logic."""

    def test_fit_calculates_expected_paths(self, norm_df):
        """Test fit() calculates expected paths. pk must include path_pk."""
        dt = DistanceTransformer(
            x="x",
            y="y",
            duration="duration",
            pk=["participant", "stimulus"],
            path_pk=["stimulus"],
        )
        dt.fit(norm_df)

        assert dt.expected_paths is not None
        assert "s1" in dt.expected_paths
        # s2 is present in sample_df for p1 (even if p2 doesn't have it)
        assert "s2" in dt.expected_paths
        assert isinstance(dt.expected_paths["s1"], pd.DataFrame)


class TestSimpleDistances:
    """Tests for SimpleDistances transformer (Euclidean, DTW, etc.)."""

    def test_euclidean_distance(self, norm_df):
        """Test Euclidean distance calculation."""
        sd = SimpleDistances(
            methods=["euc"],
            x="x",
            y="y",
            pk=["participant", "stimulus"],
            path_pk=["stimulus"],
        )
        result = sd.fit(norm_df).transform(norm_df)

        assert isinstance(result, pd.DataFrame)
        assert "euc_dist_mean" in result.columns
        # sample_df have p1(s1, s2) and p2(s1). Total 3 groups.
        assert len(result) == 3
        assert (result["euc_dist_mean"] >= 0).all()

    def test_multiple_methods(self, norm_df):
        """Test multiple distance metrics at once."""
        sd = SimpleDistances(
            methods=["euc", "dtw", "man"],
            x="x",
            y="y",
            pk=["participant", "stimulus"],
            path_pk=["stimulus"],
        )
        result = sd.fit(norm_df).transform(norm_df)

        assert "euc_dist_mean" in result.columns
        assert "dtw_dist_mean" in result.columns
        assert "man_dist_mean" in result.columns


class TestScanMatchDist:
    """Tests for ScanMatchDist transformer."""

    def test_scanmatch_execution(self, norm_df):
        """Test ScanMatch distance execution."""
        sm = ScanMatchDist(
            x="x",
            y="y",
            duration="duration",
            pk=["participant", "stimulus"],
            path_pk=["stimulus"],
            t_bin=50,
        )
        result = sm.fit(norm_df).transform(norm_df)

        assert "scan_match_dist_mean" in result.columns
        assert (result["scan_match_dist_mean"] >= 0).all()

    def test_scanmatch_requires_duration(self, norm_df):
        """Test that validation fails if duration is missing."""
        sm = ScanMatchDist(
            x="x",
            y="y",
            pk=["participant", "stimulus"],
            path_pk=["stimulus"],
            duration=None,  # Missing duration
        )
        with pytest.raises(RuntimeError):
            # check_init is called in transform (and fit via DistanceTransformer)
            sm.fit(norm_df).transform(norm_df)


class TestMultiMatchDist:
    """Tests for MultiMatchDist transformer."""

    def test_multimatch_execution(self, norm_df):
        """Test MultiMatch distance execution."""
        mm = MultiMatchDist(
            x="x",
            y="y",
            duration="duration",
            pk=["participant", "stimulus"],
            path_pk=["stimulus"],
        )
        result = mm.fit(norm_df).transform(norm_df)

        expected_suffixes = ["shape", "angle", "len", "pos", "duration"]
        for suffix in expected_suffixes:
            col = f"mm_{suffix}_mean"
            assert col in result.columns

        assert len(result.columns) == 5
