import typing

import pandas as pd

from dataclasses import dataclass

from typing import Union, List, Tuple


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
