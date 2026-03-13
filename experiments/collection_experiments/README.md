# Collection experiments – folder structure

Structure of the `collection_experiments` folder.

## Python modules (`utils/`)

| File | Role |
|------|------|
| `benchmark_utils.py` | Data via `eyefeatures.data` (Parquet + meta), `get_collection_dir`, split groups from meta, `create_and_save_splits_for_dataset`, split helpers, `col_info_from_meta`, `ensure_duration`; DL adapters `find_datasets_parquet`, `load_dataset_parquet`. |
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
| `training.ipynb` | ML (FLAML AutoML) and DL training: feature CSVs for FLAML; Parquet + splits for DL. Run after `create_splits` and `feature_extraction_all`. |
| `gaze_idt_fixation_extraction.ipynb` | Gaze / I-DT fixation extraction from gaze-only datasets. |

## Output and result folders

- **`features_output/`** – Extracted feature CSVs and split-applied train/val/test files (created by `feature_extraction_all`).
- **`features_output/splits/`** – Split metadata and label CSVs (created by `create_splits`).
- **`results/`** – Result CSVs: FLAML (`flaml_results_all_batteries.csv`, `flaml_results_all_batteries_additional.csv`), DL (`dl_training_results_all_representations.csv`), best ML/DL per task (`best_ml_dl_per_task_table.csv`).
- **`plots/`** – Figures and plot data (e.g. radar plots, ML vs DL wins).

Data is read from the repo `data/collection` (Parquet + `meta.json`).
