from abc import abstractmethod
from typing import Any, List, Union

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, TransformerMixin

from eyetracking.preprocessing._utils import _get_distance
from eyetracking.utils import _split_dataframe


class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, x: str, y: str, t: str, aoi: str = None, pk: List[str] = None):
        self.x = x
        self.y = y
        self.t = t
        self.aoi = aoi
        self.pk = pk

    @abstractmethod
    def _check_params(self):
        """
        Method checks that provided data is sufficient to conduct preprocessing.
        """
        ...

    @abstractmethod
    def _preprocess(self, X: pd.DataFrame):
        """
        Main method to preprocess fixations.
        """
        ...

    @staticmethod
    def _err_no_field(m, c):
        return f"Method {m} requires {c} for preprocessing."

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        self._check_params()
        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, NDArray]:
        self._check_params()
        if self.pk is None:
            fixations = self._preprocess(X)
        else:
            fixations = None
            groups: List[str, pd.DataFrame] = _split_dataframe(X, self.pk, encode=False)
            for group_ids, group_X in groups:
                cur_fixations = self._preprocess(group_X)

                for i in range(len(self.pk)):
                    cur_fixations.insert(loc=i, column=self.pk[i], value=group_ids[i])

                if fixations is None:
                    fixations = cur_fixations
                else:
                    fixations = pd.concat(
                        [fixations, cur_fixations], ignore_index=True, axis=0
                    )

        return fixations


class BaseFixationPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _squash_fixations(is_fixation: NDArray) -> NDArray:
        """
        :param is_fixation: 0/1 array, whether a gaze is part of fixation.
        :returns: array of same size, ones are replaced with fixation_id.
        """
        n = len(is_fixation)
        fixation_id = 1
        prev_is_fixation = False
        for i in range(n):
            if is_fixation[i] == 0:
                prev_is_fixation = False
                continue
            if not prev_is_fixation:
                fixation_id += 1
                prev_is_fixation = True

            is_fixation[i] = fixation_id

        return is_fixation

    @staticmethod
    def _get_distances(points: NDArray, distance):
        dist = np.zeros(len(points) - 1)
        for i in range(len(points) - 1):
            dist[i] = _get_distance(points[i, :], points[i + 1, :], distance=distance)
        return dist


class BaseAOIPreprocessor(BasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_fixation_density(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray[Any, np.dtype], Any, Any]:
        """
        Finds the fixation density of a given dataframe.
        :param data: DataFrame with fixations.
        :param x: x coordinate of fixation.
        :param y: y coordinate of fixation.
        :return: density for each point in [x_min, x_max] x [y_min, y_max] area
        """
        df = data[[self.x, self.y]]
        assert df.shape[0] != 0, "Error: there are no points"
        kde = gaussian_kde(df.values.T)
        X, Y = np.mgrid[
            df[self.x].min() : df[self.x].max() : 100j,
            df[self.y].min() : df[self.y].max() : 100j,
        ]  # is 100 enough?
        positions = np.vstack([X.ravel(), Y.ravel()])
        return np.reshape(kde(positions), X.shape), X, Y

    @staticmethod
    @jit(forceobj=True, looplift=True)
    def _build_local_max_coordinates(loc_max_matrix: np.ndarray) -> np.ndarray:
        for i in range(loc_max_matrix.shape[0]):
            for j in range(loc_max_matrix.shape[1]):
                if i == 0 and j != 0:
                    if loc_max_matrix[i][j - 1] == loc_max_matrix[i][j]:
                        loc_max_matrix[i][j - 1] = 0
                elif j == 0 and i != 0:
                    if loc_max_matrix[i - 1][j] == loc_max_matrix[i][j]:
                        loc_max_matrix[i - 1][j] = 0
                elif i != 0 and j != 0:
                    if loc_max_matrix[i - 1][j] == loc_max_matrix[i][j]:
                        loc_max_matrix[i - 1][j] = 0
                    if loc_max_matrix[i - 1][j - 1] == loc_max_matrix[i][j]:
                        loc_max_matrix[i - 1][j - 1] = 0
                    if loc_max_matrix[i][j - 1] == loc_max_matrix[i][j]:
                        loc_max_matrix[i][j - 1] = 0
        return np.transpose(np.nonzero(loc_max_matrix))
