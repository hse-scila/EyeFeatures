"""
Basic Fixation Extraction Example
=================================

This example demonstrates how to extract fixations from raw gaze data using the I-DT
(Identification by Dispersion-Threshold) algorithm. We use a subset of gaze data provided
in the library.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from eyefeatures.preprocessing.fixation_extraction import IDT
from eyefeatures.preprocessing.smoothing import SavGolFilter, WienerFilter

# Load sample data
gazes_df = pd.read_csv("data/gazes/gazes_subset.csv")
print("Loaded gazes data:", gazes_df.shape)

# column names
x, y, t = "norm_pos_x", "norm_pos_y", "gaze_timestamp"
pk = ["Participant", "tekst"]

# Initialize IDT algorithm
fixation_extractor = IDT(
    x=x,
    y=y,
    t=t,
    min_duration=0.08,  # seconds
    max_duration=2.0,  # seconds
    max_dispersion=0.05,  # Distance units (normalized data)
    pk=pk,
)

# Initialize smoothing algorithms
w_filter = WienerFilter(x=x, y=y, t=t, pk=pk, K="auto")
sg_filter = SavGolFilter(x=x, y=y, t=t, pk=pk, window_length=11)

# Initialize a pipeline gazes -> fixations
pipe = Pipeline(
    steps=[
        ("w_filter", w_filter),  # Wiener
        ("sg_filter", sg_filter),  # Savitzkiy-Golay
        ("fixation_extractor", fixation_extractor),  # IDT
    ]
)

# Run the pipeline
fixations_smooth = pipe.fit_transform(gazes_df)

print("\nResulting Fixations DataFrame:")
print(fixations_smooth.head())
print(f"\nShape: {fixations_smooth.shape}")
print("\nColumns:", list(fixations_smooth.columns))
