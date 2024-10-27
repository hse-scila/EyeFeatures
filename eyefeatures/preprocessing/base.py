from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit, prange
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, TransformerMixin

from eyefeatures.preprocessing._utils import _get_distance
from eyefeatures.utils import _get_angle, _get_angle3, _split_dataframe


class BasePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, pk: List[str] = None):
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
        return f"Method '{m}' requires '{c}' for preprocessing."

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

                for i in prange(len(self.pk)):
                    cur_fixations.insert(loc=i, column=self.pk[i], value=group_ids[i])

                if fixations is None:
                    fixations = cur_fixations
                else:
                    fixations = pd.concat(
                        [fixations, cur_fixations], ignore_index=True, axis=0
                    )

        return fixations


class BaseFixationPreprocessor(BasePreprocessor, ABC):
    def __init__(self, x: str, y: str, t: str, pk: List[str] = None):
        super().__init__(pk=pk)
        self.x = x
        self.y = y
        self.t = t

    @staticmethod
    def _squash_fixations(is_fixation: NDArray) -> NDArray:
        """
        :param is_fixation: 0/1 array, whether a gaze is part of fixation.
        :returns: array of same size, ones are replaced with fixation_id.
        """
        n = len(is_fixation)
        fixation_id = 0
        prev_is_fixation = False
        for i in prange(n):
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
        for i in prange(len(points) - 1):
            dist[i] = _get_distance(points[i, :], points[i + 1, :], distance=distance)
        return dist

    def _compute_feats(
        self, fixations_df: pd.DataFrame, feats: Tuple[str, ...]
    ) -> pd.DataFrame:
        """
        Method computes list of required features.
        """
        n = len(fixations_df)

        # fixation duration
        if "duration" in feats:
            fixations_df["duration"] = fixations_df.end_time - fixations_df.start_time

        # saccade preceding the fixation
        if "saccade_duration" in feats:
            sd = fixations_df.start_time.values - fixations_df.end_time.shift(1).values
            sd[0] = 0  # no preceding saccade
            fixations_df["saccade_duration"] = sd

        # saccade preceding the fixation
        if "saccade_length" in feats:
            start_points = fixations_df[[self.x, self.y]].values
            end_points = fixations_df[[self.x, self.y]].shift(1).values
            sl = _get_distance(
                end_points,
                start_points,
                distance=self.distance,  # initialized by child class
            )
            sl[0] = 0
            fixations_df["saccade_length"] = sl

        # angle between x-axis and saccade preceding the fixation,
        # assuming that saccade moves from (0, 0) to current fixation
        if "saccade_angle" in feats:
            dx: pd.Series = fixations_df[self.x].diff().values
            dy: pd.Series = fixations_df[self.y].diff().values
            sa = np.zeros(shape=(n,))
            for i in prange(1, n):
                sa[i] = _get_angle(dx[i], dy[i], degrees=True)
            fixations_df["saccade_angle"] = sa

        # angle between preceding and succeeding saccades
        if "saccade2_angle" in feats:
            xx, yy = fixations_df[self.x].values, fixations_df[self.y].values
            sa2 = np.zeros(shape=(n,))
            for i in prange(1, n - 1):
                sa2[i] = _get_angle3(
                    x0=xx[i],
                    y0=yy[i],
                    x1=xx[i - 1],
                    y1=yy[i - 1],
                    x2=xx[i + 1],
                    y2=yy[i + 1],
                )
            fixations_df["saccade2_angle"] = sa2

        return fixations_df


class BaseAOIPreprocessor(BasePreprocessor, ABC):
    def __init__(self, x: str, y: str, t: str, aoi: str = None, pk: List[str] = None):
        super().__init__(pk=pk)
        self.x = x
        self.y = y
        self.t = t
        self.aoi = aoi

    def _get_fixation_density(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray[Any, np.dtype], Any, Any]:
        """
        Finds the fixation density of a given dataframe.
        :param data: DataFrame with fixations.
        :return: density for each point in [x_min, x_max] x [y_min, y_max] area
        """
        df = data[[self.x, self.y]]
        assert df.shape[0] > 2, "Not enough points"
        kde = gaussian_kde(df.values.T)
        X, Y = np.mgrid[
            df[self.x].min() : df[self.x].max() : 100j,
            df[self.y].min() : df[self.y].max() : 100j,
        ]  # is 100 enough?
        positions = np.vstack([X.ravel(), Y.ravel()])
        return np.reshape(kde(positions), X.shape), X, Y

    @staticmethod
    @jit(forceobj=True, looplift=True)
    def _find_local_max_coordinates(loc_max_matrix: np.ndarray) -> np.ndarray:
        """
        Finds the local max coordinates of a fixation density matrix.
        :param loc_max_matrix: matrix with maxima.
        """
        for i in prange(loc_max_matrix.shape[0]):  # TODO vectorize with numpy?
            for j in prange(loc_max_matrix.shape[1]):
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

    def _scale_coordinates(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.x] -= X[self.x].mean()
        X[self.y] -= X[self.y].mean()
        return X


class BaseSmoothingPreprocessor(BasePreprocessor, ABC):
    def __init__(self, x: str, y: str, t: str, pk: List[str] = None):
        super().__init__(pk=pk)
        self.x = x
        self.y = y
        self.t = t
