<p align="center">
  <img src="docs/images/eyefeatures_logo.png" width="400" alt="EyeFeatures Logo">
</p>

# EyeFeatures

[![PyPI version](https://img.shields.io/pypi/v/eyefeatures.svg)](https://pypi.org/project/eyefeatures/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/eyefeatures.svg)](https://pypi.org/project/eyefeatures/)
[![License](https://img.shields.io/github/license/hse-scila/EyeFeatures.svg)](https://github.com/hse-scila/EyeFeatures/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/eyefeatures-docs/badge/?version=latest)](https://eyefeatures-docs.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/hse-scila/EyeFeatures/actions/workflows/ci.yml/badge.svg)](https://github.com/hse-scila/EyeFeatures/actions/workflows/ci.yml)

**EyeFeatures** is an end-to-end Python library for eye-tracking data analysis, designed to make eye-tracking research accessible to developers and scientists alike. From raw gaze preprocessing to complex feature engineering and deep learning, `eyefeatures` provides a unified, production-ready framework.

## Key Features

- **Scikit-learn Integration**: All transformers follow the `fit`/`transform` API and work seamlessly with `sklearn.Pipeline`.
- **PyTorch-Ready**: Native PyTorch `Dataset` classes and neural network modules (CNN, LSTM, GNN, ViT) for gaze-based classification.
- **Scanpath Visualizations**: Static and animated scanpath plots, heatmaps, and AOI overlays.
- **50+ Methods**: Extensive library of preprocessing, statistical, complexity, and distance-based features.
- **Group Analysis**: Built-in support for individual normalization and group-level comparisons.

## Installation

**Note**: Latest version in PyPi is `v1.0.1`. Check [Contribution](https://eyefeatures-docs.readthedocs.io/en/latest/contribution.html) page in the documentation for installation with `poetry`.

```bash
pip install eyefeatures
```

## Documentation & Tutorials

Check out our [Full Documentation](https://eyefeatures-docs.readthedocs.io/) and the following interactive tutorials:

- 🚀 [Quickstart Examples](https://eyefeatures-docs.readthedocs.io/en/latest/quickstart/quickstart.html)
- 📊 [Simple Features](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/features_tutorial.ipynb)
- 🧠 [Complex Features & Timeseries](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/complex_tutorial.ipynb)
- 🛠️ [Preprocessing & Smoothing](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/preprocessing_tutorial.ipynb)
- 🧿 [AOI Definition](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/AOI_definition_tutorial.ipynb)
- 🎥 [Visualization](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/visualization_tutorial.ipynb)
- ⚡ [Deep Learning with Gaze](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/DL_tutorial.ipynb)

## Supported Methods

Check a comprehensive list of all methods.

<details>
<summary><b>🔧 Preprocessing</b></summary>

> <details>
> <summary>Fixation Extraction</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Velocity Threshold Identification (I-VT) | Velocity-based fixation detection. O(n) complexity. | [IVT](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.fixation_extraction.IVT) |
> | Dispersion Threshold Identification (I-DT) | Dispersion-based fixation detection. O(n log n) avg. | [IDT](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.fixation_extraction.IDT) |
> | Hidden Markov Model Identification (I-HMM) | Probabilistic fixation detection via Viterbi algorithm | [IHMM](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.fixation_extraction.IHMM) |
>
> </details>
>
> <details>
> <summary>Smoothing Filters</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Savitzky-Golay Filter | Polynomial smoothing filter | [SavGolFilter](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.smoothing.SavGolFilter) |
> | Finite Impulse Response Filter (FIR) | Convolution with FIR kernel | [FIRFilter](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.smoothing.FIRFilter) |
> | Infinite Impulse Response Filter (IIR) | Convolution with IIR kernel | [IIRFilter](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.smoothing.IIRFilter) |
> | Wiener Filter | Noise reduction filter using spectral estimation | [WienerFilter](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.smoothing.WienerFilter) |
>
> </details>
>
> <details>
> <summary>Blink Detection</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Pupil Missing Detection | Blink detection via missing pupil data | [detect_blinks_pupil_missing](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.blinks_extraction.detect_blinks_pupil_missing) |
> | Pupil Velocity Threshold | Blink detection via pupil size velocity | [detect_blinks_pupil_vt](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.blinks_extraction.detect_blinks_pupil_vt) |
> | Eye Openness Detection | Blink detection via eye openness signal | [detect_blinks_eo](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.blinks_extraction.detect_blinks_eo) |
>
> </details>
>
> <details>
> <summary>Area of Interest (AOI) Extraction</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Shape-Based AOI | Define AOI using predefined shapes (rect, circle, polygon) | [ShapeBased](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.aoi_extraction.ShapeBased) |
> | Threshold-Based AOI | Density-based AOI using local maxima and KMeans | [ThresholdBased](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.aoi_extraction.ThresholdBased) |
> | Gradient-Based AOI | AOI extraction via gradient magnitude | [GradientBased](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.aoi_extraction.GradientBased) |
> | Overlap Clustering AOI | AOI via overlapping fixation clusters | [OverlapClustering](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.aoi_extraction.OverlapClustering) |
> | AOI Extractor | Meta-extractor selecting lowest entropy partition | [AOIExtractor](https://eyefeatures-docs.readthedocs.io/en/latest/api/preprocessing.html#eyefeatures.preprocessing.aoi_extraction.AOIExtractor) |
>
> </details>

</details>

<details>
<summary><b>📊 Feature Engineering</b></summary>

> <details>
> <summary>Statistical Features</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Fixation Features | Duration, dispersion, VAD statistics | [FixationFeatures](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.stats.FixationFeatures) |
> | Saccade Features | Length, speed, direction/rotation angles | [SaccadeFeatures](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.stats.SaccadeFeatures) |
> | Regression Features | Features for regressive (backward) saccades | [RegressionFeatures](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.stats.RegressionFeatures) |
> | Micro-Saccade Features | Statistics for small fixation-related saccades | [MicroSaccadeFeatures](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.stats.MicroSaccadeFeatures) |
>
> </details>
>
> <details>
> <summary>Complexity & Entropy Measures</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Hurst Exponent | Long-term memory measure via R/S analysis | [HurstExponent](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.HurstExponent) |
> | Shannon Entropy | Gaze distribution uncertainty over AOIs | [ShannonEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.ShannonEntropy) |
> | Spectral Entropy | Frequency-domain complexity via PSD | [SpectralEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.SpectralEntropy) |
> | Fuzzy Entropy | Robust entropy with fuzzy membership | [FuzzyEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.FuzzyEntropy) |
> | Sample Entropy | Irregularity/complexity of scanpath | [SampleEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.SampleEntropy) |
> | Incremental Entropy | Average entropy as exploration evolves | [IncrementalEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.IncrementalEntropy) |
> | Gridded Distribution Entropy | Spatial entropy over a grid | [GriddedDistributionEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.GriddedDistributionEntropy) |
> | Phase Entropy | Phase space trajectory complexity | [PhaseEntropy](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.PhaseEntropy) |
> | Lyapunov Exponent | Chaos indicator via trajectory divergence | [LyapunovExponent](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.LyapunovExponent) |
> | Fractal Dimension | Box-counting dimension of scanpath | [FractalDimension](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.FractalDimension) |
> | Correlation Dimension | Attractor dimensionality measure | [CorrelationDimension](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.CorrelationDimension) |
> | RQA Measures | Recurrence Quantification (REC, DET, LAM, CORM) | [RQAMeasures](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.RQAMeasures) |
> | Saccade Unlikelihood | Negative log-likelihood of saccade transitions | [SaccadeUnlikelihood](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.SaccadeUnlikelihood) |
> | Hilbert-Huang Transform | Features from Empirical Mode Decomposition | [HHTFeatures](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.measures.HHTFeatures) |
>
> </details>
>
> <details>
> <summary>Scanpath Distance Metrics</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Euclidean Distance | Point-to-point distance | [EucDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.EucDist) |
> | Hausdorff Distance | Max distance between point sets | [HauDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.HauDist) |
> | Dynamic Time Warping | Time-invariant scanpath similarity | [DTWDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.DTWDist) |
> | Discrete Fréchet Distance | Shape-based curve similarity | [DFDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.DFDist) |
> | ScanMatch | String-based scanpath comparison | [ScanMatchDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.ScanMatchDist) |
> | MultiMatch | Multi-dimensional scanpath comparison | [MultiMatchDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.MultiMatchDist) |
> | Mannan Distance | Fixation position similarity | [MannanDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.MannanDist) |
> | EyeAnalysis Distance | Fixation-based scanpath comparison | [EyeAnalysisDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.EyeAnalysisDist) |
> | Time Delay Embedding Distance | Phase-space reconstruction similarity | [TDEDist](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.scanpath_dist.TDEDist) |
>
> </details>
>
> <details>
> <summary>Complex Representations</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Heatmap | Aggregated gaze density visualization | [get_heatmap](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.complex.get_heatmap) |
> | Markov Transition Field | Temporal dynamics as transition probabilities | [get_mtf](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.complex.get_mtf) |
> | Gramian Angular Field | Polar encoding of time series | [get_gaf](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.complex.get_gaf) |
> | Recurrence Plot | Visual representation of dynamical systems | [get_rqa](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.complex.get_rqa) |
> | Hilbert Curve Mapping | Space-filling curve for 2D→1D mapping | [get_hilbert_curve](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.complex.get_hilbert_curve) |
>
> </details>
>
> <details>
> <summary>Normalization</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Individual Normalization | Group-relative feature scaling | [IndividualNormalization](https://eyefeatures-docs.readthedocs.io/en/latest/api/features.html#eyefeatures.features.shift.IndividualNormalization) |
>
> </details>

</details>

<details>
<summary><b>🤖 Deep Learning</b></summary>

> <details>
> <summary>Datasets</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Gaze Dataset | PyTorch Dataset for gaze sequences | [GazeDataset](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.datasets.GazeDataset) |
> | Time Series Dataset | PyTorch Dataset for 2D time series | [TimeSeriesDataset](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.datasets.TimeSeriesDataset) |
> | Graph Dataset | PyTorch Geometric Dataset for scanpaths | [GraphDataset](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.datasets.GraphDataset) |
>
> </details>
>
> <details>
> <summary>Models</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | CNN Model | Convolutional Neural Network for gaze classification | [CNNModel](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.models.CNNModel) |
> | LSTM Model | Recurrent model for sequential gaze data | [LSTMModel](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.models.LSTMModel) |
> | GNN Model | Graph Neural Network for scanpath analysis | [GNNModel](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.models.GNNModel) |
> | Vision Transformer | Attention-based model for gaze images | [ViTModel](https://eyefeatures-docs.readthedocs.io/en/latest/api/deep.html#eyefeatures.deep.models.ViTModel) |
>
> </details>

</details>

<details>
<summary><b>🎨 Visualization</b></summary>

> <details>
> <summary>Static Plots</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Static Scanpath Plot | Static visualization of eye movements | [static_scanpath_plot](https://eyefeatures-docs.readthedocs.io/en/latest/api/visualization.html#eyefeatures.visualization.static.static_scanpath_plot) |
> | Heatmap Visualization | Gaze density heatmap plot | [heatmap_plot](https://eyefeatures-docs.readthedocs.io/en/latest/api/visualization.html#eyefeatures.visualization.static.heatmap_plot) |
> | AOI Visualization | Area of Interest overlay plot | [aoi_plot](https://eyefeatures-docs.readthedocs.io/en/latest/api/visualization.html#eyefeatures.visualization.static.aoi_plot) |
>
> </details>
>
> <details>
> <summary>Animated Plots</summary>
>
> | Method | Description | Docs |
> | :--- | :--- | :---: |
> | Dynamic Scanpath Animation | Animated visualization of scanpaths | [dynamic_scanpath_plot](https://eyefeatures-docs.readthedocs.io/en/latest/api/visualization.html#eyefeatures.visualization.dynamic.dynamic_scanpath_plot) |
>
> </details>

</details>

<details>
<summary><b>📁 Data</b></summary>

> Utilities to list and load benchmark datasets (Parquet), with column conventions for keys, labels, and meta. [API](https://eyefeatures-docs.readthedocs.io/en/latest/api/data.html)

</details>
