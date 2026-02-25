# Benchmark experiments – folder structure

Structure of the `benchmark_experiments` folder.

## Python modules

| File | Role |
|------|------|
| `benchmark_utils.py` | Data via `eyefeatures.data` (Parquet + meta), split groups from meta, `create_and_save_splits_for_dataset`, split helpers, `col_info_from_meta`, `ensure_duration`; DL adapters `find_datasets_parquet`, `load_dataset_parquet`. |
| `feature_extraction_utils.py` | Shared feature extraction: `setup_paths`, `extract_and_save_features`, `apply_splits_and_save`, `print_summary`. |
| `distance_extraction_utils.py` | Distance-feature pipeline: fit on train only, transform train/val/test; simple/advanced/expected-path methods; `path_pk_per_label`. |
| `split_utils.py` | Re-exports split-related helpers from `benchmark_utils`. |
| `flaml_training.py` | FLAML AutoML on feature CSVs: `run_training_battery`, split/label loading from splits_dir and features_dir. |
| `dl_training_utils.py` | DL training battery: dataset creation, model wrappers (2D, TimeSeries, merged), `run_dl_training_battery`. |
| `training_common.py` | Shared for FLAML and DL: task type, `REGRESSION_DATASET_PREFIXES`, `SKIP_DATASET_SUBSTRINGS`, label helpers. |

## Notebooks

| Notebook | Purpose |
|----------|--------|
| `create_splits.ipynb` | Create train/val/test splits from meta; writes split info under `features_output/splits/`. |
| `feature_extraction_all.ipynb` | Single pipeline for all feature batteries: simple, extended, complex, distance. Run after `create_splits`. |
| `DL_training.ipynb` | Deep learning training on extracted features / raw data. |
| `ML_training.ipynb` | ML/FLAML training on feature CSVs. |
| `gaze_idt_fixation_extraction.ipynb` | Gaze / I-DT fixation extraction. |

## Output and result folders

- **`features_output/`** – Extracted feature CSVs and split-applied train/val/test files (created by `feature_extraction_all`).
- **`features_output/splits/`** – Split metadata and label CSVs (created by `create_splits`).
- **`plots/`** – Figures and plot data.
- **`tables/`** – Result tables (e.g. best ML/DL per task).
- **`dl_results_proper_fixed/`** – DL training result CSVs.
- **`flaml_results_proper_fixed/`** – FLAML/AutoML result CSVs.

Data is read from the repo `data/benchmark` (Parquet + `meta.json`).
