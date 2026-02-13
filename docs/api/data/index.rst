Data
====

.. currentmodule:: eyefeatures.data

The ``data`` module provides simple data loading utilities for the eye-tracking
benchmark. Benchmark data lives in the repo at ``data/benchmark`` as Parquet
files (tracked with Git LFS). Column conventions: primary key (columns
starting with ``group_``), labels (columns ending with ``_label``), meta
(columns starting with ``meta_``).

.. autodata:: DEFAULT_BENCHMARK_DIR

.. autofunction:: list_datasets

.. autofunction:: load_dataset

.. autofunction:: get_pk

.. autofunction:: get_labels

.. autofunction:: get_meta
