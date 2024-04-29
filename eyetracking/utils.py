from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Union

import pandas as pd


@dataclass
class Types:
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
