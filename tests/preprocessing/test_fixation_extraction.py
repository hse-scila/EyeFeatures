"""Tests for eyefeatures/preprocessing/fixation_extraction.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.preprocessing.fixation_extraction import IDT, IHMM, IVT


@pytest.fixture
def synthetic_gaze_df():
    """Synthetic gaze data with clear fixations and saccades."""
    # Three fixations: (100, 100), (200, 200), (300, 300)
    # Each fixation is 20 samples long, with small noise.
    # Total 100 samples (including transitions)
    t = np.linspace(0, 1, 100)  # fs = 100Hz, dt=0.01
    x = np.zeros(100)
    y = np.zeros(100)

    # Fixation 1
    x[0:20] = 100 + np.random.normal(0, 0.5, 20)
    y[0:20] = 100 + np.random.normal(0, 0.5, 20)

    # Saccade 1
    x[20:30] = np.linspace(100, 200, 10)
    y[20:30] = np.linspace(100, 200, 10)

    # Fixation 2
    x[30:50] = 200 + np.random.normal(0, 0.5, 20)
    y[30:50] = 200 + np.random.normal(0, 0.5, 20)

    # Saccade 2
    x[50:60] = np.linspace(200, 300, 10)
    y[50:60] = np.linspace(200, 300, 10)

    # Fixation 3
    x[60:80] = 300 + np.random.normal(0, 0.5, 20)
    y[60:80] = 300 + np.random.normal(0, 0.5, 20)

    # Rest is noise/saccade
    x[80:] = 400 + np.random.normal(0, 5, 20)
    y[80:] = 400 + np.random.normal(0, 5, 20)

    return pd.DataFrame({"x": x, "y": y, "t": t, "participant": ["p1"] * 100})


class TestFixationExtraction:
    """Tests for various fixation extraction algorithms."""

    def test_ivt(self, synthetic_gaze_df):
        """Test Velocity Threshold Identification (IVT)."""
        # Increase threshold to be more robust against noise in synthetic data
        ivt = IVT(x="x", y="y", t="t", threshold=500, min_duration=0.1)
        fixations = ivt.fit_transform(synthetic_gaze_df)

        assert isinstance(fixations, pd.DataFrame)
        assert len(fixations) >= 3
        assert "duration" in fixations.columns
        assert "saccade_length" in fixations.columns
        assert (fixations["duration"] >= 0.1).all()

    def test_idt(self, synthetic_gaze_df):
        """Test Dispersion Threshold Identification (IDT)."""
        # max_dispersion ~ 10 should be enough for noise ~ 0.5 std
        idt = IDT(
            x="x", y="y", t="t", min_duration=0.1, max_duration=1.0, max_dispersion=20.0
        )
        fixations = idt.fit_transform(synthetic_gaze_df)

        assert isinstance(fixations, pd.DataFrame)
        assert len(fixations) >= 3
        assert "dispersion" in fixations.columns
        assert (fixations["dispersion"] <= 20.0).all()

    def test_ihmm(self, synthetic_gaze_df):
        """Test Hidden Markov Model Identification (IHMM)."""
        # IHMM might need careful tuning of distrib_params for synthetic data
        # Using custom params to match synthetic data scale
        dp = {
            "fixation": {"loc": 50, "scale": 50},
            "saccade": {"loc": 1000, "scale": 200},
        }
        ihmm = IHMM(x="x", y="y", t="t", distrib_params=dp)
        fixations = ihmm.fit_transform(synthetic_gaze_df)

        assert isinstance(fixations, pd.DataFrame)
        assert len(fixations) > 0
        assert "duration" in fixations.columns

    def test_ivt_pk(self, synthetic_gaze_df):
        """Test IVT with primary key grouping."""
        df2 = synthetic_gaze_df.copy()
        df2["participant"] = "p2"
        combined_df = pd.concat([synthetic_gaze_df, df2], ignore_index=True)

        ivt = IVT(
            x="x", y="y", t="t", threshold=500, min_duration=0.1, pk=["participant"]
        )
        fixations = ivt.fit_transform(combined_df)

        assert "participant" in fixations.columns
        assert set(fixations["participant"]) == {"p1", "p2"}
        assert len(fixations) >= 6  # At least 3 per participant
