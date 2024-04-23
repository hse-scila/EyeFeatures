import numpy as np
import pandas as pd

from typing import List, Tuple


def _split_dataframe(df: pd.DataFrame, pk):
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
