from typing import Dict, List, Union

import multimatch_gaze as mm
import numpy as np
import pandas as pd
from extractor import BaseTransformer
from numba import jit
from scanpath_complex import get_expected_path, get_fill_path


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

            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values

            dist = ((current_path - expected_path) ** 2).sum()
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
        :param duration: Column name of fixations duration
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

            column_names.append(f'sm_{"_".join([str(g) for g in group])}')

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


class MannanDist(BaseTransformer):
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
        Calculates Mannan distance between given and expected scanpaths.
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
        ), "Error: provide x column before calling transform from MannanDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from MannanDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from MannanDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from MannanDist"

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
        ), "Error: provide x column before calling transform from MannanDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from MannanDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from MannanDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from MannanDist"

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

            column_names.append(f'man_{"_".join([str(g) for g in group])}')

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                continue

            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values

            sp, sq = 0, 0
            for p in current_path:
                sp += ((expected_path - p) ** 2).sum(axis=1).min()
            for q in expected_path:
                sq += ((current_path - q) ** 2).sum(axis=1).min()

            dist = len(expected_path) * sp + len(current_path) * sq

            gathered_features.append(
                np.sqrt(dist / (4 * len(expected_path) * len(current_path)))
            )

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class EyeAnalysisDist(BaseTransformer):
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
        Calculates Eye Analysis distance between given and expected scanpaths.
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
        ), "Error: provide x column before calling transform from EyeAnalysisDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from EyeAnalysisDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from EyeAnalysisDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EyeAnalysisDist"

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
        ), "Error: provide x column before calling transform from EyeAnalysisDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from EyeAnalysisDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from EyeAnalysisDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from EyeAnalysisDist"

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

            column_names.append(f'ea_{"_".join([str(g) for g in group])}')

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                continue

            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values

            dist = 0
            for p in current_path:
                dist += ((expected_path - p) ** 2).sum(axis=1).min()
            for q in expected_path:
                dist += ((current_path - q) ** 2).sum(axis=1).min()

            gathered_features.append(dist / max(len(current_path), len(expected_path)))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class DFDist(BaseTransformer):
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
        Calculates Discrete Frechet distance between given and expected scanpaths.
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
        ), "Error: provide x column before calling transform from DFDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from DFDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from DFDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from DFDist"

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
        ), "Error: provide x column before calling transform from DFDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from DFDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from DFDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from DFDist"

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

            column_names.append(f'df_{"_".join([str(g) for g in group])}')

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                continue

            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values

            dp = np.ones((len(current_path), len(expected_path))) * (-1)
            dp[0, 0] = ((current_path[0] - expected_path[0]) ** 2).sum()
            for i in range(1, len(current_path)):
                dp[i, 0] = max(
                    dp[i - 1, 0], ((current_path[i] - expected_path[0]) ** 2).sum()
                )
            for j in range(1, len(expected_path)):
                dp[0, j] = max(
                    dp[0, j - 1], ((current_path[0] - expected_path[j]) ** 2).sum()
                )

            for i in range(1, len(current_path)):
                for j in range(1, len(expected_path)):
                    dp[i, j] = max(
                        ((current_path[i] - expected_path[j]) ** 2).sum(),
                        min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                    )

            gathered_features.append(dp[-1, -1])

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class TDEDist(BaseTransformer):
    def __init__(
        self,
        k: int = 1,
        x: str = None,
        y: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        Calculates Time Delay Embedding distance between given and expected scanpaths.
        :param k: Number of scanpath batches
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(x=x, y=y, path_pk=path_pk, pk=pk, return_df=return_df)
        assert k > 0, "Error: k must be positive in TDEDist"
        self.k = k
        self.fill_path = fill_path
        self.expected_paths = expected_paths

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from TDEDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from TDEDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from TDEDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from TDEDist"

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
        ), "Error: provide x column before calling transform from TDEDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from TDEDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from TDEDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from TDEDist"

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

            column_names.append(f'tde_{"_".join([str(g) for g in group])}')

            if self.k < min(len(current_path), len(expected_path)) == 0:
                gathered_features.append(np.nan)
                continue

            current_path = current_path[[self.x, self.y]].values
            expected_path = expected_path[["x_est", "y_est"]].values

            dist = 0
            for i in range(len(current_path) // self.k):
                for j in range(len(expected_path) // self.k):
                    P = current_path[i * self.k : (i + 1) * self.k]
                    Q = expected_path[j * self.k : (j + 1) * self.k]
                    if len(P) != len(Q):
                        break
                    dist += ((P - Q) ** 2).sum()

            gathered_features.append(
                dist / ((len(current_path) // self.k) * (len(expected_path) // self.k))
            )

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class MultiMatchDist(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        duration: str = None,
        path_pk: List[str] = None,
        pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        Calculates MultiMatch distance between given and expected scanpaths.
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param duration: Column name of fixations duration
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """
        super().__init__(
            x=x, y=y, duration=duration, path_pk=path_pk, pk=pk, return_df=return_df
        )
        self.fill_path = fill_path
        self.expected_paths = expected_paths

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling transform from MultiMatchDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from MultiMatchDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from MultiMatchDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from MultiMatchDist"
        assert (
            self.duration is not None
        ), "Error: provide duration column before calling transform from MultiMatchDist"

        if self.expected_paths is None:
            self.expected_paths = get_expected_path(
                data=X,
                x=self.x,
                y=self.y,
                duration=self.duration,
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
        ), "Error: provide x column before calling transform from MultiMatchDist"
        assert (
            self.y is not None
        ), "Error: provide y column before calling transform from MultiMatchDist"
        assert (
            self.path_pk is not None
        ), "Error: provide path_pk column before calling transform from MultiMatchDist"
        assert (
            self.pk is not None
        ), "Error: provide pk column before calling transform from MultiMatchDist"
        assert (
            self.duration is not None
        ), "Error: provide duration column before calling transform from MultiMatchDist"

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

            if len(current_path) * len(expected_path) == 0:
                gathered_features.append(np.nan)
                column_names.append(f'mm_{"_".join([str(g) for g in group])}')
                continue

            current_path = current_path[[self.x, self.y, self.duration]]
            expected_path = expected_path[["x_est", "y_est", "duration_est"]]

            current_path = current_path.rename(
                columns=dict(
                    [
                        (self.x, "start_x"),
                        (self.y, "start_y"),
                        (self.duration, "duration"),
                    ]
                )
            )
            expected_path = expected_path.rename(
                columns=dict(
                    [
                        ("x_est", "start_x"),
                        ("y_est", "start_y"),
                        ("duration_est", "duration"),
                    ]
                )
            )

            current_path = current_path.reset_index()
            expected_path = expected_path.reset_index()

            sim = mm.docomparison(
                fixation_vectors1=current_path,
                fixation_vectors2=expected_path,
                screensize=[1, 1],
            )

            column_names.append(f'mm_shape_{"_".join([str(g) for g in group])}')
            column_names.append(f'mm_angle_{"_".join([str(g) for g in group])}')
            column_names.append(f'mm_len_{"_".join([str(g) for g in group])}')
            column_names.append(f'mm_pos_{"_".join([str(g) for g in group])}')
            column_names.append(f'mm_duration_{"_".join([str(g) for g in group])}')

            for metric in sim:
                gathered_features.append(metric)

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values
