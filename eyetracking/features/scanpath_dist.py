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


def get_fill_path(
    paths: List[pd.DataFrame], x: str, y: str, duration: str = None
) -> pd.DataFrame:
    all_paths = pd.concat(
        [path.assign(pid=k) for k, path in enumerate(paths)], ignore_index=True
    )
    all_paths["dummy"] = 1
    return list(
        get_expected_path(
            data=all_paths,
            x=x,
            y=y,
            path_pk=["dummy"],
            pk=["dummy", "pid"],
            duration=duration,
        ).values()
    )[0]


class EucDist(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        Calculates Euclidean distance between given and expected scanpaths.
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(x=x, y=y, path_pk=path_pk, pk=pk, return_df=return_df)
        self.fill_path = fill_path
        self.expected_paths = expected_paths

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

        if self.expected_paths is None:
            self.expected_paths = get_expected_path(
                data=X, x=self.x, y=self.y, path_pk=self.path_pk, pk=self.pk
            )
        if self.fill_path is None:
            self.fill_path = get_fill_path(
                list(self.expected_paths.values()), "x_est", "y_est"
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
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from EucDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EucDist"

        groups = X[self.pk].drop_duplicates().values
        column_names = []
        gathered_features = []
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

            column_names.append(f'euc_{"_".join([str(g) for g in group])}')

            path_length = min(len(expected_path), len(current_path))
            if path_length == 0:
                gathered_features.append(np.nan)
                continue
            current_path = current_path.head(path_length)
            current_path.reset_index(inplace=True)
            expected_path: pd.DataFrame = expected_path.head(path_length)
            expected_path.reset_index(inplace=True)

            dx = current_path[self.x] - expected_path["x_est"]
            dy = current_path[self.y] - expected_path["y_est"]
            dist = np.sqrt(dx ** 2 + dy ** 2).sum()
            gathered_features.append(dist)

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class HauDist(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        Calculates Hausdorff distance between given and expected scanpaths.
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(x=x, y=y, path_pk=path_pk, pk=pk, return_df=return_df)
        self.fill_path = fill_path
        self.expected_paths = expected_paths

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from HauDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from HauDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from HauDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from HauDist"

        if self.expected_paths is None:
            self.expected_paths = get_expected_path(
                data=X, x=self.x, y=self.y, path_pk=self.path_pk, pk=self.pk
            )
        if self.fill_path is None:
            self.fill_path = get_fill_path(
                list(self.expected_paths.values()), "x_est", "y_est"
            )

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from HauDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from HauDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from HauDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from HauDist"

        groups = X[self.pk].drop_duplicates().values
        column_names = []
        gathered_features = []
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

            column_names.append(f'hau_{"_".join([str(g) for g in group])}')

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                continue

            cmax = 0
            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values
            np.random.shuffle(current_path), np.random.shuffle(expected_path)
            for p_x, p_y in current_path:
                cmin = np.inf
                for q_x, q_y in expected_path:
                    cdist = (p_x - q_x) ** 2 + (p_y - q_y) ** 2
                    cmin = np.minimum(cmin, cdist)
                    if cmin < cmax:
                        break
                cmax = np.maximum(cmax, cmin)

            for q_x, q_y in expected_path:
                cmin = np.inf
                for p_x, p_y in current_path:
                    cdist = (p_x - q_x) ** 2 + (p_y - q_y) ** 2
                    cmin = np.minimum(cmin, cdist)
                    if cmin < cmax:
                        break
                cmax = np.maximum(cmax, cmin)

            gathered_features.append(np.sqrt(cmax))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class DTWDist(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        Calculates Dynamic Time Warp distance between given and expected scanpaths.
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(x=x, y=y, path_pk=path_pk, pk=pk, return_df=return_df)
        self.fill_path = fill_path
        self.expected_paths = expected_paths

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from DTWDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from DTWDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from DTWDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from DTWDist"

        if self.expected_paths is None:
            self.expected_paths = get_expected_path(
                data=X, x=self.x, y=self.y, path_pk=self.path_pk, pk=self.pk
            )
        if self.fill_path is None:
            self.fill_path = get_fill_path(
                list(self.expected_paths.values()), "x_est", "y_est"
            )

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from DTWDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from DTWDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from DTWDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from DTWDist"

        groups = X[self.pk].drop_duplicates().values
        column_names = []
        gathered_features = []
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

            column_names.append(f'dtw_{"_".join([str(g) for g in group])}')

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                continue

            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values

            dp = np.zeros((len(current_path) + 1, len(expected_path) + 1))
            dp[0, :] = np.inf
            dp[:, 0] = np.inf
            dp[0, 0] = 0
            for i in range(1, len(current_path) + 1):
                for j in range(len(expected_path) + 1):
                    p_x, p_y = current_path[i - 1]
                    q_x, q_y = expected_path[j - 1]
                    cdist = (p_x - q_x) ** 2 + (p_y - q_y) ** 2
                    dp[i, j] = cdist + np.minimum(dp[i - 1, j], dp[i, j - 1])
                    dp[i, j] = np.minimum(dp[i, j], cdist + dp[i - 1, j - 1])

            gathered_features.append(dp[-1, -1])

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


def _transform_fixation(x, y, duration, t_bin):
    assert (0 <= x <= 1) and (0 <= y <= 1), "Fixations domain must be [0, 1] x [0, 1]"
    character = chr(97 + int(100 * x) // 5) + chr(65 + int(100 * y) // 5)
    return character * int(duration // t_bin)


def _transform_path(path: pd.DataFrame, t_bin: int):
    path = path.values
    encoded_fixations = [
        _transform_fixation(x, y, duration, t_bin) for x, y, duration in path
    ]
    return "".join(encoded_fixations)


class ScanMatchDist(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        duration: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        sub_mat: np.ndarray = np.ones((20, 20)),
        t_bin: int = 200,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        Calculates ScanMatch distance between given and expected scanpaths.
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param sub_mat: Substitute costs matrix of size 20x20 (for AOI differentiating)
        :param t_bin: Temporal bin for quantifying fixation durations
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(
            x=x, y=y, duration=duration, path_pk=path_pk, pk=pk, return_df=return_df
        )
        assert sub_mat.shape == (
            20,
            20,
        ), f"Substitute matrix size must be of shape (20, 20), got {sub_mat.shape}"
        self.sub_mat = sub_mat
        self.t_bin = t_bin
        self.fill_path = fill_path
        self.expected_paths = expected_paths

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from ScanMatchDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from ScanMatchDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from ScanMatchDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from ScanMatchDist"
        assert (
            self.duration is not None
        ), "Error: provide duration column before calling transform from ScanMatchDist"

        if self.expected_paths is None:
            self.expected_paths = get_expected_path(
                data=X,
                x=self.x,
                duration=self.duration,
                y=self.y,
                path_pk=self.path_pk,
                pk=self.pk,
            )
        if self.fill_path is None:
            self.fill_path = get_fill_path(
                list(self.expected_paths.values()), "x_est", "y_est", "duration_est"
            )

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from ScanMatchDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from ScanMatchDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from ScanMatchDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from ScanMatchDist"
        assert (
            self.duration is not None
        ), "Error: provide duration column before calling transform from ScanMatchDist"

        groups = X[self.pk].drop_duplicates().values
        column_names = []
        gathered_features = []
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

            column_names.append(f'dtw_{"_".join([str(g) for g in group])}')

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                continue

            current_path = current_path[[self.x, self.y, self.duration]]
            query_filter = "0 <= x_est <= 1 and 0 <= y_est <= 1"
            expected_path = expected_path[["x_est", "y_est", "duration_est"]].query(
                query_filter
            )

            current_path = _transform_path(current_path, self.t_bin)
            expected_path = _transform_path(expected_path, self.t_bin)

            dp = np.zeros((len(current_path) // 2 + 1, len(expected_path) // 2 + 1))
            dp[0, :] = np.inf
            dp[:, 0] = np.inf
            dp[0, 0] = 0
            for i in range(1, len(current_path) // 2 + 1):
                for j in range(1, len(expected_path) // 2 + 1):
                    dp[i, j] = np.minimum(dp[i - 1, j] + 1, dp[i, j - 1] + 1)
                    p_x, p_y = current_path[2 * (i - 1)], current_path[2 * (i - 1) + 1]
                    q_x, q_y = (
                        expected_path[2 * (j - 1)],
                        expected_path[2 * (j - 1) + 1],
                    )
                    if p_x == q_x and p_y == q_y:
                        dp[i, j] = np.minimum(dp[i, j], dp[i - 1, j - 1])
                    else:
                        cost = self.sub_mat[ord(p_x) - 97][ord(p_y) - 65]
                        dp[i, j] = np.minimum(dp[i, j], dp[i - 1, j - 1] + cost)

            gathered_features.append(dp[-1, -1])

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values
