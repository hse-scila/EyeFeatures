from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import warnings
from numba import jit
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from eyefeatures.utils import _split_dataframe, _get_objs, _get_id


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
        warn: bool = True,
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
        self.warn = warn
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
        warn: bool = True,
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
        self.warn = warn
        self.return_df = return_df
        self.expected_paths = expected_paths
        self.fill_path = fill_path
        self.expected_paths_method = expected_paths_method

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        return X if self.return_df else X.values


# TODO features_names_in_ прокинуть для всех базовых классов
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
        warn: bool = True,
        leave_pk: bool = False,
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
        self.warn = warn
        self.leave_pk = leave_pk
        self.return_df = return_df
        self.is_fitted = False

    def _process_input(self, X: pd.DataFrame, y=None):
        if self.pk is not None and X[self.pk].isnull().values.any():
            raise ValueError("Found missing values in pk.")
        elif X.isnull().values.any():
            groups: List[str, pd.DataFrame] = _split_dataframe(
                X, self.pk
            )  # split by pk
            for group_id, group_X in groups:
                if group_X.isnull().values.any() and self.warn:
                    warnings.warn(f"Group {group_id} has missing values. Dropping them.",
                                  stacklevel=5)
            X = X.dropna()
        return X, y

    def fit(self, X: pd.DataFrame, y=None):
        X, y = self._process_input(X, y)

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
                    warn=self.warn,
                    return_df=self.return_df,
                )
                feature.fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Class is not fitted")

        if self.features is None:
            return X if self.return_df else X.values

        gathered_features = []
        data_df: pd.DataFrame = X

        for feature in tqdm(self.features):
            gathered_features.append(feature.transform(data_df))

        if self.extra is not None:
            columns = self.pk + [col for col in self.extra if col not in self.pk]
            extra_df = data_df[columns].groupby(self.pk).apply(self.aggr_extra)
            extra_df.index = [_get_id(index) for index in extra_df.index]
            gathered_features.append(extra_df)

        features_df = pd.concat(gathered_features, axis=1)
        if self.leave_pk:
            index = features_df.index.values
            index_as_cols = [_get_objs(id_) for id_ in index]
            for index_i in range(len(self.pk)):
                features_df[self.pk[index_i]] = [objs[index_i] for objs in index_as_cols]

        return features_df if self.return_df else features_df.values
