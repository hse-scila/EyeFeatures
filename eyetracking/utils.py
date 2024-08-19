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


def _get_angle(dx: float, dy: float, degrees: bool = True) -> float:
    """
    Method calculates an angle of movement from (0, 0) to (dx, dy). Formally, method returns
    a non-negative angle in anticlockwise direction in cartesian system between
    x-axis and vector (dx, dy).
    """
    if dx == 0:
        angle = np.pi / 2 * np.sign(dy)     # if dy == 0, then angle is zero
    elif dx < 0:
        angle = np.arctan(dy / dx) + np.pi  # (90, 270) degrees
    else:                                   # dx > 0
        angle = np.arctan(dy / dx)          # (-90, 90) degrees
        if angle < 0:                       # ( 0, 90) or (270, 360) degrees
            angle += 2 * np.pi

    return angle * 180 / np.pi if degrees else angle


def _get_angle2(x1: float, y1: float, x2: float, y2: float, degrees: bool = True, smallest: bool = False):
    """
    Method calculates a non-negative angle in anticlockwise direction based on 2 points (i.e. between two vectors)
    (x1, y1) and (x2, y2) in cartesian system.
    """
    angle1 = _get_angle(x1, y1, degrees=False)         # get positive angle between x-axis and (x1, y1)
    angle2 = _get_angle(x2, y2, degrees=False)         # get positive angle between x-axis and (x2, y2)
    diff = np.abs(angle2 - angle1)
    if smallest:
        angle = np.min(diff, 2 * np.pi - diff)
    else:
        angle = diff
    return angle * 180 / np.pi if degrees else angle   # difference of angles


def _get_angle3(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float,
                degrees: bool = True, smallest: bool = False):
    """
    Get angle at (x0, y0) based on 2 other points defining vectors (x1 - x0, y1 - y0) and (x2 - x0, y2 - y0).
    """
    # shift coordinate system such that (x0, y0) becomes (0, 0) point.
    return _get_angle2(x1=x1 - x0, y1=y1 - y0, x2=x2 - x0, y2=y2 - y0,
                       degrees=degrees, smallest=smallest)


def _check_angle_boundaries(angle, allowed_angle, deviation):
    left = _normalize_angle(allowed_angle - deviation)
    right = _normalize_angle(allowed_angle + deviation)
    angle = _normalize_angle(angle)
    if left > right:                                     # [-10, 10] -> [350, 10] -> left > right
        return (0 <= angle <= right) or (left <= angle <= 360)
    else:
        return left <= angle <= right


def _normalize_angle(angle):
    """
    Map angle to interval on [-360, 360]. Mapping
    """
    a = (abs(angle) % 360)
    return a if angle > 0 else 360 - a


def _select_regressions(
    dx: pd.Series,
    dy: pd.Series,
    rule: Tuple[int, ...],
    deviation: Union[int, Tuple[int, ...]] = None,
) -> NDArray:
    mask = np.zeros(len(dx))

    if deviation is None:                         # selection by quadrants
        if 1 in rule:
            mask = mask | ((dx > 0) & (dy > 0))
        if 2 in rule:
            mask = mask | ((dx < 0) & (dy > 0))
        if 3 in rule:
            mask = mask | ((dx < 0) & (dy < 0))
        if 4 in rule:
            mask = mask | ((dx > 0) & (dy < 0))
    else:                                         # selection by angles
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
