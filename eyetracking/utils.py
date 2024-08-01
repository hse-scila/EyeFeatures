from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Union
from numpy.typing import NDArray

import pandas as pd
import numpy as np


@dataclass
class Types:
    """
    Partition: List of split pairs <pk, Dataframe>
    Data: either Dataframe or Partition
    """

    Partition = List[Tuple[str, pd.DataFrame]]
    Data = Union[pd.DataFrame, Partition]


def _split_dataframe(
    df: pd.DataFrame, pk: List[str], encode=True
) -> Union[Types.Partition, List[Tuple[Tuple, pd.DataFrame]]]:
    """
    :param df: DataFrame to split
    :param pk: primary key to split by
    :param encode: bool, whether to encode groups into single identifier
    """

    assert set(pk).issubset(set(df.columns)), "Some key columns in df are missing"
    grouped: List[Tuple[Tuple, pd.DataFrame]] = list(df.groupby(by=pk))
    if not encode:
        return grouped
    return [(_get_id(grouped[i][0]), grouped[i][1]) for i in range(len(grouped))]


def _get_id(elements: Iterable[Any]) -> str:
    """
    Mapping between list of objects to string.
    """
    return "_".join(str(e) for e in elements)


def _calc_dt(X: pd.DataFrame, duration: str, t: str) -> pd.Series:
    if duration is None:
        dur = X[t].diff().shift(-1).fillna(0)
        dt = X[t] - (X[t] + dur / 1000).shift(1)
    else:
        dt = X[t] - (X[t] + X[duration] / 1000).shift(1)
    return dt


def _get_angle(dx, dy, degrees=True):
    """
    Method calculates non-negative angle between vectors dx and dy in R^2, such that ||dx||, ||dy|| <= 1.
    """
    if dx == 0:
        angle = np.pi / 2 * np.sign(dy)  # if dy == 0, then angle is zero
    elif dx < 0:
        angle = np.arctan(dy / dx) + np.pi  # (90, 270) degrees
    else:  # dx > 0
        angle = np.arctan(dy / dx)  # (-90, 90) degrees
        if angle < 0:  # (0, 90) or (270, 360) degrees
            angle += 2 * np.pi

    return angle * 180 / np.pi if degrees else angle


def _check_angle_boundaries(angle, allowed_angle, deviation):
    left = allowed_angle - deviation
    right = allowed_angle + deviation
    if left <= angle <= right:
        return True
    if left - 360 <= angle <= right - 360:
        return True
    if left + 360 <= angle <= right + 360:
        return True
    return False


def _select_regressions(
    dx: pd.Series,
    dy: pd.Series,
    rule: Tuple[int, ...],
    deviation: Union[int, Tuple[int, ...]] = None,
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
            for allowed_angle, dev in zip(rule, d):
                if _check_angle_boundaries(angle, allowed_angle, dev):
                    mask[i] = 1
                    break

    return mask.astype(bool)
