# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-10

### Added
- **Comprehensive Library Testing**:
  - Implemented unit tests for all dataset classes (`Dataset2D`, `DatasetTimeSeries`, `DatasetLightningBase`, etc.) in `eyefeatures.deep.datasets`.
  - Implemented unit tests for model architectures (`VitNet`, `VitNetWithCrossAttention`, `SimpleRNN`, `GIN`, `Classifier`, `Regressor`) in `eyefeatures.deep.models`.
  - New tests for `Extractor`, `BaseTransformer`, `SaccadeFeatures`, `FixationFeatures`, and `IndividualNormalization` in the `features` module.
  - Advanced tests for `ShannonEntropy`, `RQAMeasures`, and `HHTFeatures` confirming multi-group and multi-feature support.
  - Scanpath-based tests for `EucDist`, `HauDist`, and expected path calculations.
- **Warning-Free Test Output**:
  - Forced Matplotlib `Agg` backend via `MPLBACKEND` environment variable to eliminate 70+ deprecation warnings from the Tk backend and Pillow.
  - Protected PyTorch Lightning `self.log` calls with safer internal trainer checks, preventing warnings during isolated model unit tests.
- **Infrastructure & Tools**:
  - Successfully achieved and verified **>80% project-wide code coverage**.
  - Established CI/CD pipeline via GitHub Actions for automated testing.
  - Added local developer tools including `pre-commit` hooks for code style and quality.
  - Centralized shared test fixtures in `tests/conftest.py`.

### Changed
- **Architectural Refinements**:
  - **Unified `MeasureTransformer` Architecture**: Refactored base class to natively support multiple feature outputs and centralized grouping logic (`pk`).
  - **Deep Model Standardization**: Refactored `VitNet` and `VitNetWithCrossAttention` for consistent projections; updated `SimpleRNN` for hidden state access.
  - Standardized dataset constructors to require explicit coordinate labels (`x`, `y`).
- **Configuration**: Moved Matplotlib backend initialization to the top of `tests/conftest.py` for consistent headless execution.

### Fixed
- **Deep Module Stability**: Resolved bugs in datasets; fixed division-by-zero in graph feature extraction and shadowing bugs in `GINConv`.
- **Mathematical & Topological Correctness**:
  - Fixed `RuntimeWarning` (log(0)) in `persistence_entropy_curve` and standardized `float` return types.
  - Fixed `Fill Path Calculation` logic in `scanpath_complex.py` to correctly calculate the expected path of expected paths.
  - Validated original implementations for `HurstExponent` and `SpectralEntropy` after architecture refactor.

### Removed
- Deprecated Jupyter Notebooks (`.ipynb`) from the `tests/` directory.
