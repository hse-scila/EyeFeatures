"""
Basic Scanpath Processing Example
=================================

This example demonstrates how to generate a Gramian Angular Field (GAF) from a scanpath.
GAF is a technique to encode time-series (or in this case, scanpath) into an image,
often used for Deep Learning input.
"""

import matplotlib.pyplot as plt
import pandas as pd

from eyefeatures.features.complex import get_gaf

# Load sample data
fixations_df = pd.read_csv("data/fixations/fixations_subset.csv")
print("Loaded fixations data:", fixations_df.shape)

# Leave several texts(for speedup)
fixations_df = fixations_df[fixations_df["tekst"].isin([1, 10, 17])]
print("Selected data (several texts):", fixations_df.shape)

# Calculate GAF
# field_type: "sum" or "difference"
# to_polar: "regular" or "cosine"
gaf_matrix = get_gaf(
    data=fixations_df,
    x="norm_pos_x",
    y="norm_pos_y",
    t="start_timestamp",
    field_type="difference",
    to_polar="cosine",
)

print(f"\nGAF Matrix Shape: {gaf_matrix.shape}")

# Visualize
plt.figure(figsize=(8, 8))
plt.imshow(gaf_matrix[0], cmap="viridis", origin="lower")
plt.title("Gramian Angular Field (GAF)")
plt.colorbar()
# plt.show()
