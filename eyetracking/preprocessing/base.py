from typing import List, Union
from abc import abstractmethod

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numba import jit
from sklearn.base import BaseEstimator, TransformerMixin
from eyetracking.utils import _split_dataframe
from eyetracking.preprocessing._utils import _get_distance


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
            dist[i] = _get_distance(points[i, :], points[i + 1, :],
                                    distance=distance)
        return dist

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
            groups: List[str, pd.DataFrame] = _split_dataframe(
                X, self.pk, encode=False
            )
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
