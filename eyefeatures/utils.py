from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Types:
    """
    Partition: List of split pairs <pk, Dataframe>
    EncodedPartition: List of split pairs <join(pk), Dataframe>
    Data: either Dataframe or Partition
    """

    Partition = list[tuple[tuple, pd.DataFrame]]
    EncodedPartition = list[tuple[str, pd.DataFrame]]
    Data = pd.DataFrame | Partition
    Quadrants = tuple[int, ...]


def _split_dataframe(
    df: pd.DataFrame, pk: list[str], encode=True
) -> Types.Partition | Types.EncodedPartition:
    """
    :param df: DataFrame to split
    :param pk: primary key to split by
    :param encode: bool, whether to encode groups into single identifier
    """

    assert set(pk).issubset(set(df.columns)), "Some key columns in df are missing"
    grouped: Types.Partition = list(df.groupby(by=pk))
    if not encode:
        return grouped
    return [(_get_id(grouped[i][0]), grouped[i][1]) for i in range(len(grouped))]


def _get_id(elements: Iterable[Any]) -> str:
    """
    Mapping between list of objects and string.
    """
    return "_".join(str(e) for e in elements)


def _get_objs(id_: str) -> Iterable[Any]:
    """
    Mapping between string and list of objects.
    """
    return id_.split("_")


def _calc_dt(X: pd.DataFrame, duration: str, t: str) -> pd.Series:
    if duration is None:
        dur = X[t].diff().shift(-1).fillna(0)
        dt = X[t] - (X[t] + dur / 1000).shift(1)
    else:
        dt = X[t] - (X[t] + X[duration] / 1000).shift(1)
    return dt


def _get_angle(dx: float, dy: float, degrees: bool = True) -> float:
    """
    Calculates angle of movement from (0, 0) to (dx, dy). Returns an
    anticlockwise angle in cartesian system between x-axis and vector.
    """
    if dx == 0:
        angle = np.pi / 2 * np.sign(dy)  # if dy == 0, then angle is zero
    elif dx < 0:
        angle = np.arctan(dy / dx) + np.pi  # (90, 270) degrees
    else:  # dx > 0
        angle = np.arctan(dy / dx)  # (-90, 90) degrees
        if angle < 0:  # ( 0, 90) or (270, 360) degrees
            angle += 2 * np.pi

    return angle * 180 / np.pi if degrees else angle


def _get_angle2(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    degrees: bool = True,
    smallest: bool = False,
):
    """
    Calculates non-negative anticlockwise angle between two vectors
    (x1, y1) and (x2, y2).
    """
    angle1 = _get_angle(
        x1, y1, degrees=False
    )  # get positive angle between x-axis and (x1, y1)
    angle2 = _get_angle(
        x2, y2, degrees=False
    )  # get positive angle between x-axis and (x2, y2)
    diff = np.abs(angle2 - angle1)
    if smallest:
        angle = np.min(diff, 2 * np.pi - diff)
    else:
        angle = diff
    return angle * 180 / np.pi if degrees else angle  # difference of angles


def _get_angle3(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    degrees: bool = True,
    smallest: bool = False,
):
    """
    Get angle at (x0, y0) based on points defining vectors
    (x1 - x0, y1 - y0) and (x2 - x0, y2 - y0).
    """
    # shift coordinate system such that (x0, y0) becomes (0, 0) point.
    return _get_angle2(
        x1=x1 - x0,
        y1=y1 - y0,
        x2=x2 - x0,
        y2=y2 - y0,
        degrees=degrees,
        smallest=smallest,
    )


def _check_angle_in_range(angle: float, r: tuple[float, float]) -> bool:
    """
    Checks if angle is within range r = (start, end).
    If start < end, then start <= angle <= end.
    If start > end (crosses 0/360), then start <= angle <= 360 or 0 <= angle <= end.
    """
    start, end = r
    angle = _normalize_angle(angle)
    # Range is automatically normalized in usage if logic is correct,
    # but we assume r values are in [0, 360].

    if start <= end:
        return start <= angle <= end
    else:
        # Range crosses the 0/360 boundary (e.g., 350 to 10)
        return (start <= angle <= 360) or (0 <= angle <= end)


def _check_angle_boundaries(angle, allowed_angle, deviation):
    left = _normalize_angle(allowed_angle - deviation)
    right = _normalize_angle(allowed_angle + deviation)
    return _check_angle_in_range(angle, (left, right))


def _normalize_angle(angle):
    """
    Map angle to interval on [0, 360).
    """
    # commented code is the same as pure modulus.
    # a = abs(angle) % 360
    # return a if angle > 0 else 360 - a
    return angle % 360


def _select_regressions(
    dx: pd.Series,
    dy: pd.Series,
    rule: tuple[int, ...],
    deviation: None | int | tuple[int, ...] = None,
) -> NDArray:
    mask = np.zeros(len(dx))

    if deviation is None:  # selection by quadrants
        if 1 in rule:
            mask = mask | ((dx > 0) & (dy > 0))
        if 2 in rule:
            mask = mask | ((dx < 0) & (dy > 0))
        if 3 in rule:
            mask = mask | ((dx < 0) & (dy < 0))
        if 4 in rule:
            mask = mask | ((dx > 0) & (dy < 0))
    else:  # selection by angles
        dx, dy = dx.values, dy.values
        if isinstance(deviation, int):
            d = np.full(len(rule), deviation)
        else:
            d = np.array(deviation)

        for i in range(len(mask)):
            angle = _get_angle(dx[i], dy[i], degrees=True)
            for allowed_angle, dev in zip(rule, d, strict=False):
                if _check_angle_boundaries(angle, allowed_angle, dev):
                    mask[i] = 1
                    break

    return mask.astype(bool)


def _select_regressions_by_ranges(
    dx: pd.Series,
    dy: pd.Series,
    ranges: tuple[tuple[float, float], ...],
) -> NDArray:
    mask = np.zeros(len(dx))
    dx_val, dy_val = dx.values, dy.values

    for i in range(len(mask)):
        angle = _get_angle(dx_val[i], dy_val[i], degrees=True)
        for r in ranges:
            if _check_angle_in_range(angle, r):
                mask[i] = 1
                break

    return mask.astype(bool)


# =========================== MATRIX TOOLS ===========================
def _rec2square(mat: np.array) -> np.array:
    """
    Given rectangular matrix, cuts it into shape (n,n) evenly from the longest side,
    where n = min(height, width).
    """
    assert len(mat.shape) == 2
    h, w = mat.shape

    if h > w:
        return _cut_matrix(mat, n=w, axis=0)
    else:
        return _cut_matrix(mat, n=h, axis=1)


def _square2rec(mat: np.array, h: int, w: int) -> np.array:
    """
    Given square NxN matrix, cut rectangle of shape (h,w) evenly.
    """
    mat = _cut_matrix(mat, n=h, axis=0)  # cut height
    mat = _cut_matrix(mat, n=w, axis=1)  # cut width
    return mat


def _cut_matrix(mat: np.array, n: int, axis: int) -> np.array:
    """
    Given matrix of shape (h,w), cut it evenly along given axis to size n.
    """
    assert len(mat.shape) == 2
    assert axis < 2
    assert mat.shape[axis] >= n

    h, w = mat.shape
    d = n % 2
    if axis == 0:
        mat = mat[h // 2 - n // 2 : h // 2 + n // 2 + d, :]
    else:
        mat = mat[:, w // 2 - n // 2 : w // 2 + n // 2 + d]
    return mat


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified columns. Use this to remove metadata columns
    (e.g. primary keys) before passing features to a machine learning model.

    Args:
        columns: List of column names to drop.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # If input is not DataFrame, do nothing (or raise error)
            return X
        return X.drop(
            columns=[c for c in self.columns if c in X.columns], errors="ignore"
        )
