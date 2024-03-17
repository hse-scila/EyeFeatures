import numpy as np
import pandas as pd

from numba import jit

from typing import List, Union
from extractor import BaseTransformer


class SaccadeLength(BaseTransformer):
    def __init__(
        self,
        stats: List[str],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, duration, aoi, pk, return_df)
        self.stats = stats

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            sac_len: pd.DataFrame = np.sqrt(dx ** 2 + dy ** 2)
            column_names = [f"sac_len_{stat}" for stat in self.stats]
            gathered_features = [[sac_len.apply(stat)] for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                sac_len: pd.DataFrame = np.sqrt(dx ** 2 + dy ** 2)
                for stat in self.stats:
                    column_names.append(
                        f'sac_len_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append([sac_len.apply(stat)])

        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )

        return features_df if self.return_df else features_df.values
