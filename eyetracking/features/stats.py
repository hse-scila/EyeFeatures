import numpy as np
import pandas as pd

from typing import List
from extractor import BaseTransformer


class SaccadeLength(BaseTransformer):
    def __init__(
        self,
        features: List[str],
        x: str = None,
        y: str = None,
        t: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, aoi, pk, return_df)
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.features is None or len(self.features) == 0:
            return X if self.return_df else X.values

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

        dx = X[self.x].diff()
        dy = X[self.y].diff()
        sac_len = np.sqrt(dx ** 2 + dy ** 2)

        ...
