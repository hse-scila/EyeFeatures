from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, x: str, y: str, t: str, aoi: str = None, pk: List[str] = None):
        self.x = x
        self.y = y
        self.t = t
        self.aoi = aoi
        self.pk = pk

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        return X
