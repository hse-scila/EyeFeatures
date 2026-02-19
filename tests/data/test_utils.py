"""Tests for eyefeatures.data.utils."""

from pathlib import Path

import pandas as pd
import pytest

from eyefeatures.data.utils import (
    get_labels,
    get_meta,
    get_pk,
    list_datasets,
    load_dataset,
)


class TestGetPk:
    """Tests for get_pk function."""

    def test_group_prefix_columns(self):
        """Test extraction of group_ prefixed columns."""
        df = pd.DataFrame(
            {
                "group_participant": ["p1", "p2"],
                "group_stimulus": ["s1", "s2"],
                "x": [1.0, 2.0],
            }
        )
        assert get_pk(df) == ["group_participant", "group_stimulus"]

    def test_no_pk_columns(self):
        """Test empty result when no group_ columns."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        assert get_pk(df) == []


class TestGetLabels:
    """Tests for get_labels function."""

    def test_label_suffix_columns(self):
        """Test extraction of _label suffixed columns."""
        df = pd.DataFrame(
            {
                "condition_label": [0, 1],
                "task_label": ["a", "b"],
                "x": [1.0, 2.0],
            }
        )
        assert get_labels(df) == ["condition_label", "task_label"]

    def test_no_label_columns(self):
        """Test empty result when no _label columns."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        assert get_labels(df) == []


class TestGetMeta:
    """Tests for get_meta function."""

    def test_meta_prefix_columns(self):
        """Test extraction of meta_ prefixed columns."""
        df = pd.DataFrame(
            {
                "meta_screen_w": [1920],
                "meta_screen_h": [1080],
                "x": [1.0],
            }
        )
        assert get_meta(df) == ["meta_screen_w", "meta_screen_h"]

    def test_no_meta_columns(self):
        """Test empty result when no meta_ columns."""
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        assert get_meta(df) == []


class TestListDatasets:
    """Tests for list_datasets function."""

    def test_list_returns_sorted_names(self):
        """Test listing datasets from benchmark dir returns sorted names."""
        benchmark_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "benchmark"
        )
        if not benchmark_dir.exists():
            pytest.skip("Benchmark data dir not found")
        result = list_datasets(benchmark_dir=benchmark_dir)
        assert isinstance(result, list)
        assert result == sorted(result)
        if result:
            assert all(isinstance(name, str) for name in result)


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_paris_experiment_fixations(self):
        """Test loading Paris_experiment_fixations dataset from benchmark."""
        benchmark_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "benchmark"
        )
        dataset_path = benchmark_dir / "Paris_experiment_fixations.parquet"
        if not dataset_path.exists():
            pytest.skip("Paris_experiment_fixations.parquet not found (Git LFS?)")
        df, meta = load_dataset(
            "Paris_experiment_fixations", benchmark_dir=benchmark_dir
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "info" in meta
        assert "labels" in meta
        assert "general_info" in meta["info"]
        assert "reference" in meta["info"]
        assert "source_url" in meta["info"]
