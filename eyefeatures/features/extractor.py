from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        expected_paths_method: str = "mean",
        return_df: bool = True,
    ):
        self.x = x
        self.y = y
        self.t = t
        self.duration = duration
        self.dispersion = dispersion
        self.path_pk = path_pk
        self.pk = pk
        self.aoi = aoi
        self.return_df = return_df
        self.expected_paths = expected_paths
        self.fill_path = fill_path
        self.expected_paths_method = expected_paths_method

    def _check_init(self, items: List[Tuple[Any, str]]):
        for value, nm in items:
            if value is None:
                raise RuntimeError(f"{nm} is not initialized")

    def set_data(
        self,
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        expected_paths_method: str = "mean",
        return_df: bool = True,
    ):
        self.x = x
        self.y = y
        self.t = t
        self.duration = duration
        self.dispersion = dispersion
        self.path_pk = path_pk
        self.pk = pk
        self.aoi = aoi
        self.return_df = return_df
        self.expected_paths = expected_paths
        self.fill_path = fill_path
        self.expected_paths_method = expected_paths_method

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        return X if self.return_df else X.values


class Extractor(BaseEstimator, TransformerMixin):  # TODO rename to FeatureExtractor
    def __init__(
        self,
        features: List[BaseTransformer] = None,
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths_method: str = "mean",
        extra: List[str] = None,
        aggr_extra: str = None,
        return_df: bool = True,
    ):
        self.features = features
        self.x = x
        self.y = y
        self.t = t
        self.duration = duration
        self.dispersion = dispersion
        self.aoi = aoi
        self.path_pk = path_pk
        self.pk = pk
        self.expected_paths_method = expected_paths_method
        self.extra = extra
        self.aggr_extra = aggr_extra
        self.return_df = return_df
        self.is_fitted = False

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted = True
        if self.features is not None:
            for feature in self.features:
                feature.set_data(
                    x=self.x,
                    y=self.y,
                    t=self.t,
                    duration=self.duration,
                    dispersion=self.dispersion,
                    aoi=self.aoi,
                    path_pk=self.path_pk,
                    pk=self.pk,
                    expected_paths_method=self.expected_paths_method,
                    return_df=self.return_df,
                )
                feature.fit(X)

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Class is not fitted")

        if self.features is None:
            return X if self.return_df else X.values

        gathered_features = []
        data_df: pd.DataFrame = X[
            [self.x, self.y, self.t, self.duration, self.dispersion]
        ]

        if self.pk is not None:
            data_df = pd.concat([data_df, X[self.pk]], axis=1)

        if self.aoi is not None:
            data_df = pd.concat([data_df, X[self.aoi]], axis=1)

        for feature in self.features:
            gathered_features.append(feature.transform(data_df))

        if self.extra is not None:
            gathered_features.append(
                X[self.extra].groupby(self.pk).apply(self.aggr_extra)
            )

        features_df = pd.concat(gathered_features, axis=1)

        return features_df if self.return_df else features_df.values
