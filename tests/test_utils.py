"""Tests for eyefeatures/utils.py utility functions."""

import numpy as np
import pandas as pd
import pytest

from eyefeatures.utils import (
    _calc_dt,
    _check_angle_boundaries,
    _cut_matrix,
    _get_angle,
    _get_angle2,
    _get_angle3,
    _get_id,
    _get_objs,
    _normalize_angle,
    _rec2square,
    _select_regressions,
    _split_dataframe,
    _square2rec,
)


class TestGetAngle:
    """Tests for _get_angle function - angle from origin to point (dx, dy)."""

    @pytest.mark.parametrize(
        "dx,dy,expected",
        [
            (1, 0, 0.0),  # Right: 0 degrees
            (0, 1, 90.0),  # Up: 90 degrees
            (-1, 0, 180.0),  # Left: 180 degrees
            (0, -1, -90.0),  # Down: -90 degrees
            (1, 1, 45.0),  # Q1 diagonal
            (-1, 1, 135.0),  # Q2 diagonal
            (-1, -1, 225.0),  # Q3 diagonal
            (1, -1, 315.0),  # Q4 diagonal
            (0, 0, 0.0),  # Zero vector
        ],
    )
    def test_angle_degrees(self, dx, dy, expected):
        """Test angle calculation in degrees."""
        assert pytest.approx(_get_angle(dx, dy, degrees=True), abs=1e-6) == expected

    def test_angle_radians(self):
        """Test that degrees=False returns radians."""
        assert pytest.approx(_get_angle(1, 1, degrees=False), abs=1e-6) == np.pi / 4


class TestGetAngle2:
    """Tests for _get_angle2 function - angle between two vectors from origin."""

    @pytest.mark.parametrize(
        "x1,y1,x2,y2,expected",
        [
            (1, 0, 0, 1, 90.0),  # Perpendicular
            (1, 0, 1, 1, 45.0),  # 45 degrees
            (1, 0, 2, 0, 0.0),  # Parallel
            (1, 0, -1, 0, 180.0),  # Opposite
        ],
    )
    def test_angle_between_vectors(self, x1, y1, x2, y2, expected):
        """Test angle between two vectors."""
        assert (
            pytest.approx(_get_angle2(x1, y1, x2, y2, degrees=True), abs=1e-6)
            == expected
        )


class TestGetAngle3:
    """Tests for _get_angle3 function - angle at point (x0,y0)."""

    @pytest.mark.parametrize(
        "x0,y0,x1,y1,x2,y2,expected",
        [
            (0, 0, 1, 0, 0, 1, 90.0),  # Right angle at origin
            (1, 1, 2, 1, 1, 2, 90.0),  # Right angle at offset origin
        ],
    )
    def test_angle_at_point(self, x0, y0, x1, y1, x2, y2, expected):
        """Test angle at a given point."""
        assert (
            pytest.approx(_get_angle3(x0, y0, x1, y1, x2, y2, degrees=True), abs=1e-6)
            == expected
        )


class TestNormalizeAngle:
    """Tests for _normalize_angle function."""

    @pytest.mark.parametrize(
        "angle,expected",
        [
            (45, 45),
            (360, 0),
            (450, 90),
            (-45, 315),
            (-90, 270),
        ],
    )
    def test_normalize(self, angle, expected):
        """Test angle normalization to [0, 360)."""
        assert pytest.approx(_normalize_angle(angle), abs=1e-6) == expected


class TestCheckAngleBoundaries:
    """Tests for _check_angle_boundaries function."""

    @pytest.mark.parametrize(
        "angle,allowed,deviation,expected",
        [
            # Within bounds
            (45, 45, 10, True),
            (50, 45, 10, True),
            (40, 45, 10, True),
            # Outside bounds
            (60, 45, 10, False),
            (30, 45, 10, False),
            # Wrap around 0/360
            (5, 0, 20, True),
            (355, 0, 20, True),
            (340, 0, 20, True),
        ],
    )
    def test_boundary_check(self, angle, allowed, deviation, expected):
        """Test angle boundary checking."""
        assert _check_angle_boundaries(angle, allowed, deviation) is expected


class TestSelectRegressions:
    """Tests for _select_regressions function."""

    @pytest.mark.parametrize(
        "rule,deviation,expected",
        [
            ((1,), None, [True, False, False, False]),  # Q1 only
            ((1, 3), None, [True, False, False, True]),  # Q1 and Q3
            ((2,), None, [False, True, False, False]),  # Q2 only
        ],
    )
    def test_quadrant_selection(self, rule, deviation, expected):
        """Test quadrant-based selection."""
        dx = pd.Series([1, -1, 1, -1])
        dy = pd.Series([1, 1, -1, -1])
        np.testing.assert_array_equal(
            _select_regressions(dx, dy, rule, deviation), np.array(expected)
        )

    def test_angle_based_selection(self):
        """Test angle-based selection with deviation."""
        dx, dy = pd.Series([1, 0, -1, 0]), pd.Series([0, 1, 0, -1])
        np.testing.assert_array_equal(
            _select_regressions(dx, dy, rule=(0,), deviation=10),
            np.array([True, False, False, False]),
        )


class TestSplitDataframe:
    """Tests for _split_dataframe function."""

    @pytest.mark.parametrize(
        "pk,expected_len,expected_ids",
        [
            (["id"], 2, ["a", "b"]),
        ],
    )
    def test_split_single_column(self, pk, expected_len, expected_ids):
        """Test split by single primary key."""
        df = pd.DataFrame({"id": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})
        result = _split_dataframe(df, pk=pk, encode=True)
        assert len(result) == expected_len
        assert [r[0] for r in result] == expected_ids

    def test_split_composite_key(self):
        """Test split by composite primary key."""
        df = pd.DataFrame(
            {
                "p": ["p1", "p1", "p2", "p2"],
                "s": ["s1", "s2", "s1", "s2"],
                "value": [1, 2, 3, 4],
            }
        )
        result = _split_dataframe(df, pk=["p", "s"], encode=True)
        assert len(result) == 4
        ids = [r[0] for r in result]
        assert "p1_s1" in ids and "p2_s2" in ids


class TestGetIdAndObjs:
    """Tests for _get_id and _get_objs functions."""

    @pytest.mark.parametrize(
        "elements,expected_id",
        [
            (["p1", "s1", "t1"], "p1_s1_t1"),
            (["a", "b"], "a_b"),
        ],
    )
    def test_encoding(self, elements, expected_id):
        """Test encoding elements to ID."""
        assert _get_id(elements) == expected_id

    @pytest.mark.parametrize(
        "id_str,expected",
        [
            ("p1_s1_t1", ["p1", "s1", "t1"]),
            ("a_b", ["a", "b"]),
        ],
    )
    def test_decoding(self, id_str, expected):
        """Test decoding ID to elements."""
        assert list(_get_objs(id_str)) == expected


class TestCalcDt:
    """Tests for _calc_dt function."""

    def test_with_duration(self):
        """Calculate dt with duration column."""
        df = pd.DataFrame({"t": [0.0, 0.1, 0.2], "duration": [50, 60, 70]})
        result = _calc_dt(df, duration="duration", t="t")
        assert pytest.approx(result.iloc[1], abs=1e-6) == 0.05

    def test_without_duration(self):
        """Calculate dt without duration column."""
        df = pd.DataFrame({"t": [0.0, 0.1, 0.2]})
        assert len(_calc_dt(df, duration=None, t="t")) == 3


class TestMatrixUtils:
    """Tests for matrix utility functions."""

    @pytest.mark.parametrize(
        "shape,expected_shape",
        [
            ((5, 4), (4, 4)),  # Tall
            ((4, 5), (4, 4)),  # Wide
            ((4, 4), (4, 4)),  # Square
        ],
    )
    def test_rec2square(self, shape, expected_shape):
        """Test rectangular to square conversion."""
        mat = np.arange(shape[0] * shape[1]).reshape(shape)
        assert _rec2square(mat).shape == expected_shape

    def test_square2rec(self):
        """Test square to rectangle conversion."""
        mat = np.arange(25).reshape(5, 5)
        assert _square2rec(mat, h=3, w=4).shape == (3, 4)

    @pytest.mark.parametrize(
        "n,axis,expected_shape",
        [
            (3, 0, (3, 4)),  # Cut rows
            (3, 1, (4, 3)),  # Cut columns
        ],
    )
    def test_cut_matrix(self, n, axis, expected_shape):
        """Test matrix cutting along axis."""
        mat = np.arange(20).reshape(4, 5) if axis == 1 else np.arange(20).reshape(5, 4)
        assert _cut_matrix(mat, n=n, axis=axis).shape == expected_shape

    def test_cut_matrix_centers(self):
        """Test that cut is centered."""
        mat = np.arange(25).reshape(5, 5)
        result = _cut_matrix(mat, n=3, axis=0)
        np.testing.assert_array_equal(result[0, :], mat[1, :])
