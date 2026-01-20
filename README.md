<p align="center">
  <img src="docs/images/eyefeatures_logo.png" width="400" alt="EyeFeatures Logo">
</p>

# EyeFeatures

[![PyPI version](https://img.shields.io/pypi/v/eyefeatures.svg)](https://pypi.org/project/eyefeatures/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/eyefeatures.svg)](https://pypi.org/project/eyefeatures/)
[![License](https://img.shields.io/github/license/hse-scila/EyeFeatures.svg)](https://github.com/hse-scila/EyeFeatures/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/eyefeatures-docs/badge/?version=latest)](https://eyefeatures-docs.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/hse-scila/EyeFeatures/actions/workflows/ci.yml/badge.svg)](https://github.com/hse-scila/EyeFeatures/actions/workflows/ci.yml)

**EyeFeatures** is a powerful, Scikit-learn-compatible Python library for advanced eye-tracking data analysis. From raw gaze preprocessing to complex topological feature engineering and deep learning, `eyefeatures` provides a unified, production-ready framework for any visual task.

## Why EyeFeatures?

- **Unified Pipeline**: Seamlessly integrate smoothing, fixation extraction, and feature calculation into `sklearn.Pipeline`.
- **Advanced Features**: Go beyond descriptive statistics with Hurst exponents, Chaotic measures, and Scanpath similarities.
- **Deep Learning Ready**: Native PyTorch datasets and models for gaze-based classification.
- **Visualization**: Stunning static and dynamic visualizations of scanpaths and heatmaps.
- **Group Analysis**: Built-in support for individual normalization and group-level comparisons.

## Installation

```bash
pip install eyefeatures
```

## API at a Glance

### Preprocessing
| Module | Components |
| :--- | :--- |
| **Fixation Extraction** | `IDT` (I-DT algorithm) |
| **Smoothing** | `WienerFilter`, `SavGolFilter` |
| **Blinks** | `BlinkExtractor` |
| **AOI Extraction** | `GridAOI`, `CircleAOI`, `ConvexHullAOI` |

### Feature Engineering
| Category | Key Transformers / Methods |
| :--- | :--- |
| **Statistical** | `FixationFeatures`, `SaccadeFeatures`, `RegressionFeatures`, `MicroSaccadeFeatures` |
| **Measures** | `HurstExponent`, `ShannonEntropy`, `SpectralEntropy`, `FuzzyEntropy`, `LyapunovExponent` |
| **Distances** | `EucDist`, `HauDist`, `DTWDist`, `ScanMatchDist`, `MannanDist`, `MultiMatchDist` |
| **Complex** | `get_heatmap`, `get_mtf` (Markov Transition Field), `get_gaf` (Gramian Angular Field), `RQAMeasures` |
| **Normalization**| `IndividualNormalization` (Auto-discovery of features for group-relative scaling) |

### Visualization & Deep Learning
| Module | Components |
| :--- | :--- |
| **Visualization** | `static_scanpath_plot`, `dynamic_scanpath_plot`, `heatmap_plot`, `aoi_plot` |
| **Deep Learning**| `GazeDataset`, `CNNModel`, `LSTMModel`, `GNNModel` |

## Documentation & Tutorials

Check out our [Full Documentation](https://eyefeatures-docs.readthedocs.io/) and the following interactive tutorials:

- 🚀 [Quickstart Examples](https://eyefeatures-docs.readthedocs.io/en/latest/quickstart/quickstart.html)
- 📊 [Simple Features](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/features_tutorial.ipynb)
- 🧠 [Complex Features & Timeseries](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/complex_tutorial.ipynb)
- 🛠️ [Preprocessing & Smoothing](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/preprocessing_tutorial.ipynb)
- 🧿 [AOI Definition](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/AOI_definition_tutorial.ipynb)
- 🎥 [Visualization](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/visualization_tutorial.ipynb)
- ⚡ [Deep Learning with Gaze](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/DL_tutorial.ipynb)
