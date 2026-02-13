.. _overview:

Overview
========

The eyefeatures library provides tools for extracting, analyzing, and visualizing eye-tracking data. 
It follows scikit-learn's API design with ``fit``/``transform`` methods, making it compatible 
with scikit-learn pipelines.

Key Features
------------
- Extraction of common eye-tracking features (fixations, saccades, regressions).
- Blinks detection from pupil signal.
- Statistical analysis of eye movement patterns and direct usage for ML tasks.
- Algorithms like Markov Transition Field, Hilbert Curve calculation, Vietoris-Rips filtration for complex analysis and potential usage in Deep Learning architectures.
- Visualization tools for exploring gaze/fixations patterns.
- Benchmark data loading utilities (Parquet datasets, column conventions for keys/labels/meta).
- ``scikit-learn`` compatible transformers for ``Pipeline`` integration.