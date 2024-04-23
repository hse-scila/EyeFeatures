from typing import List, Union

import numpy as np
import pandas as pd
from numba import jit

from eyetracking.preprocessing.base import BasePreprocessor


class IVT(BasePreprocessor):
    """
    Velocity Threshold Identification.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str,
        threshold: float,
        pk: List[str] = None,
        eps: float = 1e-10,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.threshold = threshold
        self.eps = eps

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.pk is None:
            fixations = self._ivt(
                x=X[self.x].values, y=X[self.y].values, t=X[self.t].values
            )
        else:
            fixations = None
            groups = X[self.pk].drop_duplicates().values
            for group in groups:
                cur_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                cur_fixations = self._ivt(
                    x=cur_X[self.x].values,
                    y=cur_X[self.y].values,
                    t=cur_X[self.t].values,
                )
                for i in range(len(self.pk)):
                    cur_fixations.insert(loc=i, column=self.pk[i], value=group[i])

                if fixations is None:
                    fixations = cur_fixations
                else:
                    fixations = pd.concat(
                        [fixations, cur_fixations], ignore_index=True, axis=0
                    )

        return fixations

    def _ivt(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> pd.DataFrame:
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)

        dist = np.sqrt(dx**2 + dy**2)
        vel = dist / (dt + self.eps)

        fixations = np.zeros(len(vel))
        fixation_id = 1
        prev_is_fixation = False
        for i in range(len(vel)):
            is_fixation = 0 if vel[i] < self.threshold else 1
            if not is_fixation:
                prev_is_fixation = False
                continue
            if not prev_is_fixation:
                fixation_id += 1
                prev_is_fixation = True

            fixations[i] = fixation_id

        fixations_df = pd.DataFrame(
            data={
                "fixation_id": fixations,
                self.x: x[:-1],
                self.y: y[:-1],
                "start_time": t[:-1],
                "end_time": t[:-1],
                "distance_min": dist,
                "distance_max": dist,
            }
        )

        fixations_df = fixations_df[fixations_df["fixation_id"] != 0]

        fixations_df = fixations_df.groupby(by=["fixation_id"]).agg(
            {
                self.x: "mean",
                self.y: "mean",
                "start_time": "min",
                "end_time": "max",
                "distance_min": "min",
                "distance_max": "max",
            }
        )

        fixations_df["duration"] = fixations_df.end_time - fixations_df.start_time

        return fixations_df
