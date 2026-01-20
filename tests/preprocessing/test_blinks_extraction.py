"""Tests for eyefeatures/preprocessing/blinks_extraction.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.preprocessing.blinks_extraction import (
    detect_blinks_eo,
    detect_blinks_pupil_missing,
    detect_blinks_pupil_vt,
)


@pytest.fixture
def blink_pupil_df():
    """Pupil signal with a NaN-based blink."""
    t = np.linspace(0, 1000, 500)  # 500Hz for 1s
    pupil = np.ones(500) * 4.0
    # Blink from 200ms to 400ms (indices 100 to 200)
    pupil[100:200] = np.nan
    # Add some monotonic decrease/increase for detect_blinks_pupil_missing boundaries
    pupil[90:100] = np.linspace(4.0, 3.5, 10)
    pupil[200:210] = np.linspace(3.5, 4.0, 10)

    return pd.DataFrame({"pupil": pupil, "t": t})


@pytest.fixture
def blink_eo_df():
    """Eye openness signal with a blink (dip in openness)."""
    np.random.seed(42)
    t = np.linspace(0, 2000, 1000)  # 500Hz for 2s
    # Eye is 1.0 open with some noise
    eo = 1.0 + np.random.normal(0, 0.01, 1000)
    # Blink dip from 400ms to 800ms (indices 200 to 400)
    # 200 samples = 400ms duration
    blink_range = np.arange(200, 400)
    # Deep dip: amplitude 0.8
    eo[blink_range] -= 0.8 * np.sin(np.pi * (blink_range - 200) / 200)

    return pd.DataFrame({"eo": eo, "t": t})


class TestBlinksExtraction:
    """Tests for various blink detection algorithms."""

    def test_detect_blinks_pupil_missing(self, blink_pupil_df):
        """Test detect_blinks_pupil_missing."""
        df = detect_blinks_pupil_missing(
            pupil_signal=blink_pupil_df["pupil"].values,
            timestamps=blink_pupil_df["t"].values,
            min_separation=50,
            min_dur=50,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["duration"] >= 200  # Original 100 indices @ 500Hz = 200ms
        assert df.iloc[0]["onset"] < 200  # Should catch the monotonic decrease

    def test_detect_blinks_pupil_vt(self, blink_pupil_df):
        """Test detect_blinks_pupil_vt."""
        # This one uses NaNs but also allows interpolation of small gaps
        df = detect_blinks_pupil_vt(
            pupil_signal=blink_pupil_df["pupil"].values,
            timestamps=blink_pupil_df["t"].values,
            Fs=500,
            gap_dur=20,
            min_dur=50,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["duration"] >= 190

    def test_detect_blinks_eo(self, blink_eo_df):
        """Test detect_blinks_eo."""
        df = detect_blinks_eo(
            eye_openness_signal=blink_eo_df["eo"].values,
            timestamps=blink_eo_df["t"].values,
            Fs=500,
            min_blink_length=50,
            min_amplitude=0.5,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "openness_at_peak" in df.columns
        # Theoretical dip is to 0.2, but noise and smoothing can push it slightly higher
        assert df.iloc[0]["openness_at_peak"] < 0.21
