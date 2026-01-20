"""Tests for eyefeatures/preprocessing/smoothing.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.preprocessing.smoothing import (
    FIRFilter,
    IIRFilter,
    SavGolFilter,
    WienerFilter,
)


@pytest.fixture
def raw_gaze_df():
    """Synthetic raw gaze data with some noise."""
    t = np.linspace(0, 1, 100)
    # Sine wave with noise
    x = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, 100)
    y = np.cos(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, 100)
    return pd.DataFrame({"x": x, "y": y, "t": t, "participant": ["p1"] * 100})


class TestSmoothingPreprocessors:
    """Tests for various smoothing preprocessors."""

    def test_savgol_filter(self, raw_gaze_df):
        """Test Savitzky-Golay filter."""
        # Test without pk
        filter_ = SavGolFilter(x="x", y="y", t="t", window_length=11, polyorder=2)
        result = filter_.fit_transform(raw_gaze_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == raw_gaze_df.shape
        assert not np.array_equal(result["x"].values, raw_gaze_df["x"].values)

        # Test with pk
        filter_pk = SavGolFilter(
            x="x", y="y", t="t", pk=["participant"], window_length=11, polyorder=2
        )
        result_pk = filter_pk.fit_transform(raw_gaze_df)
        assert "participant" in result_pk.columns
        assert len(result_pk) == 100

    def test_fir_filter(self, raw_gaze_df):
        """Test FIR filter."""
        # Using a small numtaps for the small synthetic dataset
        filter_ = FIRFilter(
            x="x", y="y", t="t", numtaps=11, fs=100, cutoff=10, mode="same"
        )
        result = filter_.fit_transform(raw_gaze_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == raw_gaze_df.shape
        assert not np.array_equal(result["x"].values, raw_gaze_df["x"].values)

        # Test valid mode (should reduce length)
        filter_valid = FIRFilter(
            x="x", y="y", t="t", numtaps=11, fs=100, cutoff=10, mode="valid"
        )
        result_valid = filter_valid.fit_transform(raw_gaze_df)
        assert len(result_valid) < len(raw_gaze_df)

    def test_iir_filter(self, raw_gaze_df):
        """Test IIR filter."""
        filter_ = IIRFilter(x="x", y="y", t="t", N=3, Wn=0.2, btype="lowpass")
        result = filter_.fit_transform(raw_gaze_df)

        assert isinstance(result, pd.DataFrame)
        # IIRFilter implementation slices the result based on kernel size
        assert len(result) < len(raw_gaze_df)
        assert not np.array_equal(
            result["x"].values, raw_gaze_df["x"].values[: len(result)]
        )

    def test_wiener_filter(self, raw_gaze_df):
        """Test Wiener filter."""
        filter_ = WienerFilter(x="x", y="y", t="t", K=1e-4)
        result = filter_.fit_transform(raw_gaze_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == raw_gaze_df.shape
        assert not np.array_equal(result["x"].values, raw_gaze_df["x"].values)

        # Test 'auto' K
        filter_auto = WienerFilter(x="x", y="y", t="t", K="auto")
        result_auto = filter_auto.fit_transform(raw_gaze_df)
        assert isinstance(result_auto, pd.DataFrame)
        assert filter_auto.K != "auto"  # Should be updated after fit
