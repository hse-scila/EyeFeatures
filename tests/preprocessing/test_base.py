"""Tests for eyefeatures/preprocessing/base.py and _utils.py."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.preprocessing._utils import _get_distance, _get_MEC
from eyefeatures.preprocessing.base import BasePreprocessor


class MockPreprocessor(BasePreprocessor):
    """Minimal implementation of BasePreprocessor for testing."""

    def _check_params(self):
        pass

    def _preprocess(self, X):
        # Just return X with a dummy column
        res = X.copy()
        # Drop self.pk if it exists in res to avoid duplication by BasePreprocessor
        if self.pk:
            drop_cols = [c for c in self.pk if c in res.columns]
            res = res.drop(columns=drop_cols)
        res["processed"] = True
        return res


class TestBasePreprocessor:
    """Tests for BasePreprocessor functionality."""

    def test_transform_no_pk(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        mp = MockPreprocessor()
        result = mp.transform(df)
        assert "processed" in result.columns
        assert len(result) == 2

    def test_transform_with_pk(self):
        df = pd.DataFrame({"a": [1, 2, 10, 20], "pk": ["g1", "g1", "g2", "g2"]})
        mp = MockPreprocessor(pk=["pk"])
        result = mp.transform(df)
        assert "processed" in result.columns
        assert "pk" in result.columns
        # Check that it split and processed groups
        assert len(result) == 4
        assert set(result["pk"]) == {"g1", "g2"}


class TestPreprocessingUtils:
    """Tests for internal preprocessing utilities."""

    def test_get_distance_scalar(self):
        assert _get_distance(1, 4, "euc") == 3
        assert _get_distance(1, 4, "manhattan") == 3
        assert _get_distance(1, 4, "chebyshev") == 3

    def test_get_distance_vector(self):
        v = np.array([0, 0])
        u = np.array([3, 4])
        assert _get_distance(v, u, "euc") == 5
        assert _get_distance(v, u, "manhattan") == 7
        assert _get_distance(v, u, "chebyshev") == 4

    def test_get_distance_matrix(self):
        v = np.array([[0, 0], [1, 1]])
        u = np.array([[3, 4], [4, 5]])
        dist_euc = _get_distance(v, u, "euc")
        assert len(dist_euc) == 2
        assert dist_euc[0] == 5
        assert dist_euc[1] == 5

    def test_get_mec(self):
        # Points on a square
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        # MEC should have center (0.5, 0.5) and radius sqrt(0.5)
        res = _get_MEC(points)
        assert np.allclose(res[:2], [0.5, 0.5])
        assert np.isclose(res[2], np.sqrt(0.5))

    def test_get_distance_invalid(self):
        with pytest.raises(NotImplementedError):
            _get_distance(1, 2, "invalid")
