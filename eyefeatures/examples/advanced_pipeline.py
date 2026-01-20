"""
Advanced Pipeline Example
=========================

This example demonstrates a complete end-to-end pipeline:
1.  **Preprocessing**: Convert raw gazes to fixations using a pipeline of Wiener/SavGol filters and IDT.
2.  **Feature Extraction**: Calculate stats and measures using Extractor.
3.  **Normalization**: Normalize features per participant using IndividualNormalization.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from eyefeatures.features.extractor import Extractor
from eyefeatures.features.shift import IndividualNormalization
from eyefeatures.features.stats import FixationFeatures, SaccadeFeatures
from eyefeatures.preprocessing.fixation_extraction import IDT

# -----------------------------------------------------------------------------
# 1. Load / Generate Data
# -----------------------------------------------------------------------------
gazes_df = pd.read_csv("data/gazes/gazes_subset.csv")
print("Loaded gazes data:", gazes_df.shape)

# -----------------------------------------------------------------------------
# 2. Preprocessing: Gazes -> Fixations
# -----------------------------------------------------------------------------
print("\n[Step 1] Define Fixation Extractor.")
fixation_extractor = IDT(
    min_duration=0.08,
    max_duration=2.0,
    max_dispersion=0.05,
    x="norm_pos_x",
    y="norm_pos_y",
    t="gaze_timestamp",
    pk=["Participant", "tekst"],
)

# -----------------------------------------------------------------------------
# 3. Feature Extraction
# -----------------------------------------------------------------------------
print("\n[Step 2] Define Feature Extractor.")
feature_extractor = Extractor(
    features=[
        FixationFeatures(),
        SaccadeFeatures(),
    ],
    x="norm_pos_x",
    y="norm_pos_y",
    t="start_time",
    duration="duration",
    dispersion="dispersion",
    pk=["Participant", "tekst"],
    leave_pk=True,
    return_df=True,
)

# -----------------------------------------------------------------------------
# 4. Normalization
# -----------------------------------------------------------------------------
print("\n[Step 3] Define Normalization Transformer.")

# By default, IndividualNormalization will discover all numeric columns
# (features) and normalize them per the specified primary key.
normalizer = IndividualNormalization(pk=["Participant"], inplace=False)

# -----------------------------------------------------------------------------
# 5. Apply Pipeline
# -----------------------------------------------------------------------------
print("\n[Step 4] Apply Pipeline.")

pipe = Pipeline(
    [
        ("fixation_extractor", fixation_extractor),
        ("feature_extractor", feature_extractor),
        ("normalizer", normalizer),
    ]
)

# Fit_transform runs the whole chain
processed_df = pipe.fit_transform(gazes_df)

print("\nPipeline Complete.")
print("Processed DataFrame Head:")
sample_col = "sac_length_mean"
if sample_col in processed_df.columns:
    cols_to_show = [sample_col]
    norm_col = f"{sample_col}_norm"
    if norm_col in processed_df.columns:
        cols_to_show.append(norm_col)
    print(processed_df[cols_to_show].head())
else:
    print(processed_df.head())
