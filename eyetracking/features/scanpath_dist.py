import numpy as np
import pandas as pd

from numba import jit

from typing import List, Union
from extractor import BaseTransformer


@jit(forceobj=True, looplift=True)
def get_expected_path(
    X: pd.DataFrame,
    x: str,
    y: str,
    t: str,
    pk: List[str],
    method: str,
    return_df: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Estimates expected path by a given method
    :param X: pd.Dataframe containing coordinates of fixations and its timestamps
    :param x: Column name of x-coordinate
    :param y: Column name of y-coordinate
    :param t: Column name of timestamps
    :param pk: List of column names used to split pd.Dataframe
    :param method: Method to estimate expected path (e.g. 'mean', 'fw')
    :param return_df: Return pd.Dataframe object else np.ndarray
    :return: pd.Dataframe or np.ndarray of form (x_est, y_est, duration_est)
    """
    ...


class EucDist(BaseTransformer):
    def __init__(
        self,
        method: str,
        x: str = None,
        y: str = None,
        t: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, aoi, pk, return_df)
        self.method = method
        self.expected_path = None

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from EucDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from EucDist"
        assert (
            self.t is not None
        ), "Error: provide t column before calling transform from EucDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EucDist"
        assert (
            self.method is not None
        ), "Error: provide method column before calling transform from EucDist"

        self.expected_path = get_expected_path(
            X=X, x=self.x, y=self.y, t=self.t, pk=self.pk, method=self.method
        )

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from EucDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from EucDist"
        assert (
            self.t is not None
        ), "Error: provide t column before calling transform from EucDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EucDist"
        assert (
            self.method is not None
        ), "Error: provide method column before calling transform from EucDist"

        groups = X[self.pk].drop_duplicates().values
        column_names = []
        gathered_features = []
        for group in groups:
            current_path = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
            path_length = min(len(self.expected_path), len(current_path))
            current_path = current_path.head(path_length)
            current_path.reset_index(inplace=True)
            expected_path: pd.DataFrame = self.expected_path.head(path_length)
            expected_path.reset_index(inplace=True)
            column_names.append(f'euc_{"_".join([str(g) for g in group])}')
            gathered_features.append(
                (
                    np.sqrt(
                        (current_path[self.x] - expected_path["x_est"]) ** 2
                        + (current_path[self.y] - expected_path["y_est"]) ** 2
                    )
                ).sum()
            )

        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )

        return features_df if self.return_df else features_df.values
