import numpy as np
import pandas as pd
import pytest

from eyefeatures.features.extractor import Extractor
from eyefeatures.features.measures import (
    CorrelationDimension,
    FractalDimension,
    FuzzyEntropy,
    HHTFeatures,
    HurstExponent,
    LyapunovExponent,
    PhaseEntropy,
    RQAMeasures,
    SaccadeUnlikelihood,
    ShannonEntropy,
    SpectralEntropy,
)
from eyefeatures.features.scanpath_dist import (
    DFDist,
    DTWDist,
    EucDist,
    EyeAnalysisDist,
    HauDist,
    MannanDist,
    MultiMatchDist,
    ScanMatchDist,
    TDEDist,
)
from eyefeatures.features.stats import (
    FixationFeatures,
    MicroSaccadeFeatures,
    RegressionFeatures,
    SaccadeFeatures,
)


@pytest.fixture
def fixations_data():
    """Generates a dummy dataset with all required fields."""
    n_samples = 300
    np.random.seed(42)

    # Generate random walk for x, y
    x = np.cumsum(np.random.randn(n_samples))
    y = np.cumsum(np.random.randn(n_samples))

    # Generate timestaps
    t = np.arange(n_samples) * 20  # 20ms intervals

    data = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "time": t,
            "duration": np.random.randint(50, 500, size=n_samples),
            "dispersion": np.random.rand(n_samples) * 2,
            "aoi": np.random.choice(["A", "B", "C"], size=n_samples),
            "participant_id": ["p1"] * n_samples,
        }
    )
    return data


def generate_fixations_data(n_samples=300):
    """Generates a dummy dataset with all required fields."""
    np.random.seed(42)

    # Generate random walk for x, y
    x = np.cumsum(np.random.randn(n_samples))
    y = np.cumsum(np.random.randn(n_samples))

    # Generate timestaps
    t = np.arange(n_samples) * 20  # 20ms intervals

    data = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "time": t,
            "duration": np.random.randint(50, 500, size=n_samples),
            "dispersion": np.random.rand(n_samples) * 2,
            "aoi": np.random.choice(["A", "B", "C"], size=n_samples),
            "participant_id": ["p1"] * n_samples,
        }
    )
    return data


def test_all_features_extractor():
    """
    Comprehensive test that calculates all features available in the library
    using the Extractor class.

    Assertions:
    1. All transformers work with default arguments.
    2. Extractor correctly propagates column names to all children.
    3. No errors are raised during the entire pipeline.
    """

    # generate sample data
    fixations_data = generate_fixations_data(n_samples=500)

    # Initialize transformers
    transformers = [
        # Stats Features
        FixationFeatures(),
        SaccadeFeatures(),
        RegressionFeatures(),
        MicroSaccadeFeatures(),
        # Measures Features
        HurstExponent(n_iters=8),  # Adjusted for test speed/data size
        ShannonEntropy(),
        SpectralEntropy(),
        FuzzyEntropy(),
        PhaseEntropy(),
        LyapunovExponent(T=2),
        FractalDimension(),
        CorrelationDimension(),
        RQAMeasures(),
        SaccadeUnlikelihood(),
        HHTFeatures(max_imfs=2),  # Adjusted for speed
        # Scanpath Features (Distance-based)
        EucDist(),
        HauDist(),
        DTWDist(),
        MannanDist(),
        EyeAnalysisDist(),
        DFDist(),
        TDEDist(),
        ScanMatchDist(),
        MultiMatchDist(),
    ]

    # Extractor handles data binding
    extractor = Extractor(
        features=transformers,
        x="x",
        y="y",
        t="time",
        duration="duration",
        dispersion="dispersion",
        aoi="aoi",
        pk=["participant_id"],
        path_pk=["participant_id"],  # Required for scanpath features
        return_df=True,
    )

    # Fit and Transform
    result = extractor.fit_transform(fixations_data)

    feature_count = result.shape[1]
    cols = list(result.columns)

    print(f"Extractor generated {feature_count} features.")

    assert feature_count > 0, "No features were calculated."

    # Verify specific new features exist
    assert any("direction_angle" in col for col in cols), "direction_angle missing"
    assert any("rotation_angle" in col for col in cols), "rotation_angle missing"


def test_extractor_feature_names_consistency():
    """
    Test that Extractor's feature_names_in_ matches the child transformers'
    output and the final DataFrame columns.
    """
    data = generate_fixations_data(n_samples=100)

    # Use a mix of transformers
    transformers = [
        FixationFeatures(features_stats={"duration": ["mean", "std"]}),
        SaccadeFeatures(features_stats={"length": ["mean"]}),
        EucDist(),
        RQAMeasures(measures=["rec", "det"]),
    ]

    extractor = Extractor(
        features=transformers,
        x="x",
        y="y",
        t="time",
        duration="duration",
        dispersion="dispersion",
        pk=["participant_id"],
        path_pk=["participant_id"],
        return_df=True,
    )

    # Fit and transform
    X_out = extractor.fit_transform(data)

    # 1. Check feature_names_in_ matches DataFrame columns
    assert list(X_out.columns) == extractor.feature_names_in_

    # 2. Check feature_names_in_ matches aggregation of children's get_feature_names_out
    expected_names = []
    for t in transformers:
        expected_names.extend(t.get_feature_names_out())

    assert extractor.feature_names_in_ == expected_names

    # 3. Check individual children's feature_names_in_ if they have it (like StatsTransformer)
    for t in transformers:
        if hasattr(t, "feature_names_in_") and t.feature_names_in_ is not None:
            assert t.feature_names_in_ == t.get_feature_names_out()
