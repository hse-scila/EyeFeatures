import os

os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def fixations_df():
    """Load sample fixation data from subset file."""
    return pd.read_csv(DATA_DIR / "fixations" / "fixations_subset.csv")


@pytest.fixture
def gazes_df():
    """Load sample gaze data from subset file."""
    return pd.read_csv(DATA_DIR / "gazes" / "gazes_subset.csv")


@pytest.fixture
def blinks_df():
    """Load sample blink data from subset file."""
    return pd.read_csv(DATA_DIR / "blinks" / "blinks_subset.csv")


@pytest.fixture
def sample_df():
    """Multi-group synthetic fixation data with all necessary columns.

    Contains multiple participants, stimuli, AOIs, timestamps, durations,
    and dispersions for comprehensive testing of all transformers.
    """
    return pd.DataFrame(
        {
            "x": [100.0, 200.0, 150.0, 300.0, 250.0, 400.0, 120.0, 220.0, 180.0, 280.0],
            "y": [100.0, 150.0, 200.0, 100.0, 200.0, 150.0, 110.0, 160.0, 190.0, 120.0],
            "t": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.1, 0.2, 0.3],
            "duration": [50, 60, 55, 70, 65, 55, 52, 58, 62, 68],
            "dispersion": [5.0, 6.0, 5.5, 7.0, 6.5, 5.5, 5.2, 5.8, 6.2, 6.8],
            "participant": ["p1", "p1", "p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "stimulus": ["s1", "s1", "s1", "s2", "s2", "s2", "s1", "s1", "s1", "s1"],
            "aoi": ["A", "A", "B", "B", "A", "A", "A", "B", "B", "A"],
        }
    )


@pytest.fixture
def extracted_features_df():
    """Sample extracted features DataFrame (output of stats transformers).

    Contains feature columns like sac_length_mean, fix_duration_mean that
    represent already computed features, useful for testing normalization.
    """
    return pd.DataFrame(
        {
            "sac_length_mean": [10.0, 20.0, 15.0, 25.0, 12.0, 22.0],
            "sac_length_std": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            "fix_duration_mean": [100.0, 200.0, 150.0, 250.0, 120.0, 220.0],
            "participant": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "stimulus": ["s1", "s1", "s2", "s1", "s1", "s2"],
        }
    )
