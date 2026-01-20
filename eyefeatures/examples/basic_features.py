"""
Basic Feature Extraction Example
================================

This example shows how to use the ``Extractor`` class to calculate a mix of features
(statistics, measures, and distance-based) from fixation data.
"""

import pandas as pd

from eyefeatures.features.extractor import Extractor
from eyefeatures.features.measures import HurstExponent, SpectralEntropy
from eyefeatures.features.scanpath_dist import EucDist
from eyefeatures.features.stats import SaccadeFeatures

fixations_df = pd.read_csv("data/fixations/fixations_subset.csv")
print("Loaded fixations data:", fixations_df.shape)

# Initialize Transformers
# We select a few representative transformers from different categories
transformers = [
    # 1. Statistical Features: Saccade properties (length, speed, angles)
    SaccadeFeatures(),
    # 2. Measures: Complexity/Entropy measures
    SpectralEntropy(),
    HurstExponent(n_iters=5),
    # 3. Scanpath Distances: Euclidean distance to the 'mean' path of the group
    EucDist(),
]

# Initialize Extractor
extractor = Extractor(
    features=transformers,
    x="norm_pos_x",
    y="norm_pos_y",
    t="start_timestamp",
    duration="duration",
    dispersion="dispersion",
    aoi="AOI",
    pk=["Participant"],  # Group by participant
    path_pk=["Participant"],  # Compare each participant to mean path of participants
    return_df=True,
)

# Calculate Features
fixations_df.dropna(inplace=True)
features_df = extractor.fit_transform(fixations_df)

print("\nResulting Features DataFrame:")
print(features_df.head())
print(f"\nShape: {features_df.shape}")
print("\nColumns:", list(features_df.columns))
