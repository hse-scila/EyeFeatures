import numpy as np
import pandas as pd

from numba import jit
from scipy.optimize import minimize

from typing import List, Union, Dict
from extractor import BaseTransformer


def _target_norm(fwp: np.ndarray, fixations: np.ndarray) -> float:
    return np.linalg.norm(fixations - fwp, axis=1).sum()


@jit(forceobj=True, looplift=True)
def get_expected_path(
    data: pd.DataFrame,
    x: str,
    y: str,
    path_pk: List[str],
    pk: List[str],
    duration: str = None,
    return_df: bool = True,
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """
    Estimates expected path by a given method
    :param data: pd.Dataframe containing coordinates of fixations and its timestamps
    :param x: Column name of x-coordinate
    :param y: Column name of y-coordinate
    :param path_pk: List of column names of groups to calculate expected path (must be a subset of pk)
    :param pk: List of column names used to split pd.Dataframe
    :param duration: Column name of fixations duration if needed
    :param return_df: Return pd.Dataframe object else np.ndarray
    :return: Dict of groups and pd.Dataframe or np.ndarray of form (x_est, y_est, duration_est [if duration is passed])
    """

    assert set(path_pk).issubset(set(pk)), "path_pk must be a subset of pk"

    columns = [x, y]
    if duration is not None:
        columns.append(duration)

    expected_paths = dict()
    path_groups = data[path_pk].drop_duplicates().values
    pk_dif = [col for col in pk if col not in path_pk]  # pk \ path_pk

    for path_group in path_groups:
        length = 0
        cur_data = data[pd.DataFrame(data[path_pk] == path_group).all(axis=1)]
        expected_path, cur_paths = [], []
        groups = cur_data[pk_dif].drop_duplicates().values
        for group in groups:
            mask = pd.DataFrame(cur_data[pk_dif] == group).all(axis=1)
            path_data = cur_data[mask]
            length = max(length, len(path_data))
            cur_paths.append(path_data[columns].values)

        for i in range(length):
            vector_coord = []
            cnt, total_duration = 0, 0
            for path in cur_paths:
                if path.shape[0] > i:
                    cnt += 1
                    vector_coord.append(path[i, :2])
                    if len(columns) == 3:
                        total_duration += path[i, 2]

            vector_coord = np.array(vector_coord)
            fwp_init = np.mean(vector_coord, axis=0)
            fwp_init = minimize(
                _target_norm, fwp_init, args=(vector_coord,), method="L-BFGS-B"
            )
            next_fixation = [fwp_init.x[0], fwp_init.x[1]]
            if len(columns) == 3:
                next_fixation.append(total_duration / cnt)
            expected_path.append(next_fixation)
        ret_columns = ["x_est", "y_est"]
        if len(columns) == 3:
            ret_columns.append("duration_est")
        path_df = pd.DataFrame(expected_path, columns=ret_columns)
        expected_paths["_".join([str(g) for g in path_group])] = (
            path_df if return_df else path_df.values
        )

    return expected_paths


class EucDist(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        """
        Calculates Euclidean distance between given and expected scanpaths.
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(x, y, None, None, None, None, pk, return_df)
        self.path_pk = path_pk
        self.fill_path = None
        self.expected_paths = None

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from EucDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from EucDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from EucDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EucDist"

        self.expected_paths = get_expected_path(
            data=X, x=self.x, y=self.y, path_pk=self.path_pk, pk=self.pk
        )
        self.fill_path = list(self.expected_paths.values())[0]

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
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from EucDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EucDist"

        groups = X[self.pk].drop_duplicates().values
        column_names = []
        gathered_features = []
        self.fill_path = list(
            get_expected_path(
                data=X, x=self.x, y=self.y, path_pk=self.path_pk, pk=self.path_pk
            ).values()
        )[-1]
        for group in groups:
            current_path = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
            path_group = "_".join(
                [str(g) for g in current_path[self.path_pk].values[0]]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            path_length = min(len(expected_path), len(current_path))
            current_path = current_path.head(path_length)
            current_path.reset_index(inplace=True)
            expected_path: pd.DataFrame = expected_path.head(path_length)
            expected_path.reset_index(inplace=True)
            column_names.append(f'euc_{"_".join([str(g) for g in group])}')

            dx = current_path[self.x] - expected_path["x_est"]
            dy = current_path[self.y] - expected_path["y_est"]
            dist = np.sqrt(dx ** 2 + dy ** 2).sum()
            gathered_features.append(dist)

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values
