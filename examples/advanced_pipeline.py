"""
Advanced Pipeline Example
=========================

This example demonstrates a complete end-to-end pipeline:
1.  **Preprocessing**: Convert raw gazes to fixations using a pipeline of Wiener/SavGol filters and IDT.
2.  **Feature Extraction**: Calculate stats and measures using Extractor.
3.  **Normalization**: Normalize features per participant using IndividualNormalization.
4.  **Drop Metadata**: Drop metadata columns.
5.  **ML Model**: Add a machine learning model (optional).
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from eyefeatures.features.extractor import Extractor
from eyefeatures.features.measures import HurstExponent, SpectralEntropy
from eyefeatures.features.shift import IndividualNormalization
from eyefeatures.features.stats import (
    FixationFeatures,
    MicroSaccadeFeatures,
    RegressionFeatures,
    SaccadeFeatures,
)
from eyefeatures.preprocessing.fixation_extraction import IDT
from eyefeatures.preprocessing.smoothing import WienerFilter
from eyefeatures.utils import ColumnDropper

# Column Names in Dataset
X = "norm_pos_x"  # x-coordinate
Y = "norm_pos_y"  # y-coordinate
T_GAZE = "gaze_timestamp"  # timestamp for gazes
T_FIX = "start_time"  # timestamp for fixations
PK = ["Participant", "tekst"]  # primary keys

# 1. Load Data.
gazes_df = pd.read_csv("data/gazes/gazes_subset.csv")
print("Loaded gazes data:", gazes_df.shape)

# 2. Preprocessing: Gazes -> Fixations.
smoother = WienerFilter(
    x=X,
    y=Y,
    t=T_GAZE,
    pk=PK,
)

fixation_extractor = IDT(
    min_duration=0.08,
    max_duration=2.0,
    max_dispersion=0.20,
    x=X,
    y=Y,
    t=T_GAZE,
    pk=PK,
)

# 3. Feature Extraction.
feature_extractor = Extractor(
    features=[
        FixationFeatures(),
        SaccadeFeatures(),
        MicroSaccadeFeatures(),
        RegressionFeatures(),
        HurstExponent(coordinate=X, n_iters=4),
        SpectralEntropy(),
    ],
    x=X,
    y=Y,
    t=T_FIX,
    duration="duration",
    dispersion="dispersion",
    pk=PK,
    leave_pk=True,
    return_df=True,
)

# 4. Normalization per-group (in this example, per-participant).
normalizer = IndividualNormalization(pk=[PK[0]], inplace=False)

# 5. Drop metadata columns.
meta_columns = PK + ["start_time", "end_time"]
dropper = ColumnDropper(columns=meta_columns)

# 6. Apply Pipeline
pipe = Pipeline(
    steps=[
        ("smoother", smoother),
        ("fixation_extractor", fixation_extractor),
        ("feature_extractor", feature_extractor),
        ("normalizer", normalizer),
        ("dropper", dropper),
        # ("model", LinearRegression())  <-- add ML model for supervised tasks
    ]
)

processed_df = pipe.fit_transform(gazes_df)

print("\nPipeline Complete.")
print("Processed DataFrame Head:")
print(processed_df.head())
