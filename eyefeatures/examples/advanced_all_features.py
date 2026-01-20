"""
Advanced All Features Example
=============================

This example calculates ALL available features in the library.
It serves as a benchmark and a demonstration of the comprehensive capability of EyeFeatures.
"""

import pandas as pd

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
    SimpleDistances,
    TDEDist,
)
from eyefeatures.features.stats import (
    FixationFeatures,
    MicroSaccadeFeatures,
    RegressionFeatures,
    SaccadeFeatures,
)

# Load data
fixations_df = pd.read_csv("data/fixations/fixations_subset.csv")
print("Loaded fixations data:", fixations_df.shape)

# Leave several texts(for speedup)
fixations_df = fixations_df[fixations_df["tekst"].isin([1, 10, 17])]
print("Selected data (several texts):", fixations_df.shape)


# Initialize ALL transformers
transformers = [
    # Stats
    FixationFeatures(),
    SaccadeFeatures(),
    RegressionFeatures(),
    MicroSaccadeFeatures(),
    # Measures
    HurstExponent(n_iters=5),
    ShannonEntropy(),
    SpectralEntropy(),
    FuzzyEntropy(),
    PhaseEntropy(),
    LyapunovExponent(T=2),
    FractalDimension(),
    CorrelationDimension(),
    SaccadeUnlikelihood(),
    HHTFeatures(max_imfs=2),
    RQAMeasures(),
    # Distances
    SimpleDistances(methods=["euc", "hau"]),
    EucDist(),
    HauDist(),
    DTWDist(),
    ScanMatchDist(t_bin=50),
    MannanDist(),
    EyeAnalysisDist(),
    DFDist(),
    TDEDist(k=1),
    MultiMatchDist(),
]

extractor = Extractor(
    features=transformers,
    x="norm_pos_x",
    y="norm_pos_y",
    t="start_timestamp",
    duration="duration",
    dispersion="dispersion",
    aoi="AOI",
    pk=["Participant"],
    path_pk=["Participant"],
    return_df=True,
)

print("\nCalculating all features in the library (this may take a moment)...")
features_df = extractor.fit_transform(fixations_df)

for participant_id in (0, 1):
    assert features_df.iloc[participant_id, :].isnull().sum() == 0

print("\nCalculation complete.")
print(f"Feature Matrix Shape: {features_df.shape}")
print(f"Total Features Evaluated: {features_df.shape[1]}")
