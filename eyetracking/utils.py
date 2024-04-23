import typing
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import pandas as pd


@dataclass
class Types:
    Partition = List[Tuple[str, pd.DataFrame]]
    Data = Union[pd.DataFrame, Partition]


def _split_dataframe(df: pd.DataFrame, pk: List[str]) -> Types.Partition:
    """
    :param df: DataFrame to split
    :param pk: primary key to split by
    """

    assert set(pk).issubset(set(df.columns)), "Some key columns in df are missing"
    grouped: List[Tuple[Tuple, pd.DataFrame]] = list(df.groupby(by=pk))
    return [
        ("_".join(str(v) for v in grouped[i][0]), grouped[i][1])
        for i in range(len(grouped))
    ]


def _get_id(elements: List[Any]) -> str:
    """
    Mapping between list of objects to string.
    """
    return "_".join([str(e) for e in elements])
