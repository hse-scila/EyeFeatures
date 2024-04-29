from typing import Any, Dict, List, Tuple, Union

import multimatch_gaze as mm
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm

from eyetracking.features.extractor import BaseTransformer
from eyetracking.features.scanpath_complex import (get_expected_path,
                                                   get_fill_path)
from eyetracking.utils import Types, _split_dataframe


class DistanceTransformer(BaseTransformer):
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
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param duration: Column name of fixation duration
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was obtained from method get_expected_path
        :param fill_path: pd.DataFrame path which was obtained from method get_fill_path
        :param return_df: Return pd.Dataframe object or np.ndarray
        """

        self.fill_path = fill_path
        self.requires_duration = False
        self.expected_paths = expected_paths
        super(DistanceTransformer, self).__init__(
            x=x, y=y, duration=duration, path_pk=path_pk, pk=pk, return_df=return_df
        )

    def _get_required(self) -> List[Tuple[Any, str]]:
        return [
            (self.x, "x"),
            (self.y, "y"),
            (self.pk, "pk"),
            (self.path_pk, "path_pk"),
        ]

    def _get_partition(self, df: Types.Data) -> Types.Partition:
        if not isinstance(df, List) and not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be DataFrame or Partition")
        return df if isinstance(df, List) else _split_dataframe(df=df, pk=self.pk)

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        # TODO: modify for Types.Data

        # check must-have attributes
        super(DistanceTransformer, self)._check_init(self._get_required())

        # calculate expected path if not given
        if self.expected_paths is None:
            duration = self.duration if self.requires_duration else None
            self.expected_paths = get_expected_path(
                data=X,
                x=self.x,
                y=self.y,
                duration=duration,
                path_pk=self.path_pk,
                pk=self.pk,
            )

        # calculate filling path if not given
        if self.fill_path is None:
            duration = "duration_est" if self.requires_duration else None
            self.fill_path = get_fill_path(
                paths=list(self.expected_paths.values()),
                x="x_est",
                y="y_est",
                duration=duration,
            )

        return self


class SimpleDistances(DistanceTransformer):
    """Calculates simple distances using given methods."""

    def __init__(
        self,
        methods: List[str],
        x: str = None,
        y: str = None,
        pk: List[str] = None,
        path_pk: List[str] = None,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        return_df: bool = True,
    ):
        """
        :param methods: list of methods to use (e.g. "euc", "hau")
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """

        super(SimpleDistances, self).__init__(
            x=x,
            y=y,
            pk=pk,
            path_pk=path_pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            return_df=return_df,
        )

        self.methods = methods
        self._methods_cls = {
            "dfr": DFDist(),
            "euc": EucDist(),
            "hau": HauDist(),
            "dtw": DTWDist(),
            "man": MannanDist(),
            "eye": EyeAnalysisDist(),
        }

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(SimpleDistances, self)._check_init(
            super(SimpleDistances, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(SimpleDistances, self)._get_partition(X)

        # calculate given distances for each group
        dataframes = []
        for method in self.methods:
            if method not in list(self._methods_cls.keys()):
                raise ValueError(f"Unknown method: {method}")

            self._methods_cls[method].set_data(
                x=self.x,
                y=self.y,
                pk=self.pk,
                path_pk=self.path_pk,
                expected_paths=self.expected_paths,
                fill_path=self.fill_path,
                return_df=self.return_df,
            )
            dataframes.append(self._methods_cls[method].transform(data_part))

        features_df = pd.concat(dataframes, axis=1)
        return features_df if self.return_df else features_df.values


class EucDist(DistanceTransformer):
    """Calculates Euclidean distance between given and expected scanpaths."""

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(EucDist, self)._check_init(
            super(EucDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(EucDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["euc_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_euc_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class HauDist(DistanceTransformer):
    """Calculates Hausdorff distance between given and expected scanpaths."""

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(HauDist, self)._check_init(
            super(HauDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(HauDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["hau_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_hau_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class DTWDist(DistanceTransformer):
    """Calculates Dynamic Time Warp distance between given and expected scanpaths."""

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(DTWDist, self)._check_init(
            super(DTWDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(DTWDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["dtw_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_dtw_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class ScanMatchDist(DistanceTransformer):
    """Calculates ScanMatch distance between given and expected scanpaths."""

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

        self.t_bin = t_bin
        self.sub_mat = sub_mat
        assert sub_mat.shape == (
            20,
            20,
        ), f"Sub matrix size must be of shape (20, 20), got {sub_mat.shape}"
        super(ScanMatchDist, self).__init__(
            x=x,
            y=y,
            duration=duration,
            path_pk=path_pk,
            pk=pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            return_df=return_df,
        )
        self.requires_duration = True

    def _get_required(self) -> List[Tuple[Any, str]]:
        return [
            (self.x, "x"),
            (self.y, "y"),
            (self.pk, "pk"),
            (self.path_pk, "path_pk"),
            (self.duration, "duration"),
        ]

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(ScanMatchDist, self)._check_init(
            super(ScanMatchDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(ScanMatchDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["scan_match_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_scan_match_dist(
                group_path[[self.x, self.y, self.duration]],
                expected_path,
                t_bin=self.t_bin,
                sub_mat=self.sub_mat,
            )
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values

    def __repr__(self, **kwargs):
        return f"ScanMatch()"


class MannanDist(DistanceTransformer):
    """Calculates Mannan distance between given and expected scanpaths."""

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(MannanDist, self)._check_init(
            super(MannanDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(MannanDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["man_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_man_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class EyeAnalysisDist(DistanceTransformer):
    """Calculates Eye Analysis distance between given and expected scanpaths."""

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(EyeAnalysisDist, self)._check_init(
            super(EyeAnalysisDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(EyeAnalysisDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["eye_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_eye_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class DFDist(DistanceTransformer):
    """Calculates Discrete Frechet distance between given and expected scanpaths."""

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(DFDist, self)._check_init(
            super(DFDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(DFDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["dfr_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_dfr_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class TDEDist(DistanceTransformer):
    """Calculates Time Delay Embedding distance between given and expected scanpaths."""

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
        :param k: Number of scanpath batches
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """

        self.k = k
        assert k > 0, "k must be strictly positive"
        super(TDEDist, self).__init__(
            x=x,
            y=y,
            path_pk=path_pk,
            pk=pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            return_df=return_df,
        )

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(TDEDist, self)._check_init(
            super(TDEDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(TDEDist, self)._get_partition(X)

        # calculate distances for each group
        columns, features = ["tde_dist"], []
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_tde_dist(group_path[[self.x, self.y]], expected_path, k=self.k)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


class MultiMatchDist(DistanceTransformer):
    """Calculates MultiMatch distance between given and expected scanpaths."""

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
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param duration: Column name of fixations duration
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param return_df: Return pd.Dataframe object else np.ndarray
        """

        super(MultiMatchDist, self).__init__(
            x=x,
            y=y,
            duration=duration,
            path_pk=path_pk,
            pk=pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            return_df=return_df,
        )
        self.requires_duration = True

    def _get_required(self) -> List[Tuple[Any, str]]:
        return [
            (self.x, "x"),
            (self.y, "y"),
            (self.pk, "pk"),
            (self.path_pk, "path_pk"),
            (self.duration, "duration"),
        ]

    @jit(forceobj=True, looplift=True)
    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(MultiMatchDist, self)._check_init(
            super(MultiMatchDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(MultiMatchDist, self)._get_partition(X)

        # calculate distances for each group
        features = []
        columns = ["mm_shape", "mm_angle", "mm_len", "mm_pos", "mm_duration"]
        for group_nm, group_path in tqdm(data_part):
            expected_path = (
                self.expected_paths[group_nm]
                if group_nm in self.expected_paths.keys()
                else self.fill_path
            )
            shape, angle, length, pos, duration = calc_mm_features(
                group_path[[self.x, self.y, self.duration]], expected_path
            )
            features.append([shape, angle, length, pos, duration])

        features_df = pd.DataFrame(data=features, columns=columns)
        return features_df if self.return_df else features_df.values


# ===================== FUNCTIONS =====================
def _transform_fixation(x, y, duration, t_bin):
    assert (0 <= x <= 1) and (0 <= y <= 1), "Fixations domain must be [0, 1] x [0, 1]"
    character = chr(97 + int(100 * x) // 5) + chr(65 + int(100 * y) // 5)
    return character * int(duration // t_bin)


def _transform_path(path: pd.DataFrame, t_bin: int) -> str:
    path = path.values
    encoded_fixations = [
        _transform_fixation(x, y, duration, t_bin) for x, y, duration in path
    ]
    return "".join(encoded_fixations)


@jit(forceobj=True, looplift=True)
def calc_euc_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates Euclidean distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    """

    dist = np.nan
    length = min(len(p), len(q))
    if length > 0:
        p_aligned = p.values[:length]
        q_aligned = q.values[:length]
        dist = ((p_aligned - q_aligned) ** 2).sum()

    return dist


@jit(forceobj=True, looplift=True)
def calc_hau_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates Hausdorff distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    """

    dist = np.nan
    if len(p) * len(q) > 0:
        p_data = p.values
        q_data = q.values
        np.random.shuffle(p_data)
        np.random.shuffle(q_data)

        cur_max = 0
        for p_x, p_y in p_data:
            cur_min = np.inf
            for q_x, q_y in q_data:
                cur_dist = (p_x - q_x) ** 2 + (p_y - q_y) ** 2
                cur_min = np.minimum(cur_min, cur_dist)
                if cur_min < cur_max:
                    break
            cur_max = np.maximum(cur_max, cur_min)
        for q_x, q_y in q_data:
            cur_min = np.inf
            for p_x, p_y in p_data:
                cur_dist = (p_x - q_x) ** 2 + (p_y - q_y) ** 2
                cur_min = np.minimum(cur_min, cur_dist)
                if cur_min < cur_max:
                    break
            cur_max = np.maximum(cur_max, cur_min)

        dist = np.sqrt(cur_max)

    return dist


@jit(forceobj=True, looplift=True)
def calc_dtw_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates Dynamic Time Warp distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    """

    dist = np.nan
    if len(p) * len(q) > 0:
        p_data = p.values
        q_data = q.values

        dp = np.zeros((len(p_data) + 1, len(q_data) + 1))
        dp[0, :] = np.inf
        dp[:, 0] = np.inf
        dp[0, 0] = 0
        for i in range(1, len(p_data) + 1):
            for j in range(len(q_data) + 1):
                p_x, p_y = p_data[i - 1]
                q_x, q_y = q_data[j - 1]
                cdist = (p_x - q_x) ** 2 + (p_y - q_y) ** 2
                dp[i, j] = cdist + np.minimum(dp[i - 1, j], dp[i, j - 1])
                dp[i, j] = np.minimum(dp[i, j], cdist + dp[i - 1, j - 1])

        dist = dp[-1, -1]

    return dist


@jit(forceobj=True, looplift=True)
def calc_scan_match_dist(
    p: pd.DataFrame,
    q: pd.DataFrame,
    sub_mat: np.ndarray = np.ones((20, 20)),
    t_bin: int = 200,
) -> float:
    """
    Calculates ScanMatch distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y, duration) only
    :param q: pd.DataFrame containing columns (x, y, duration) only
    :param sub_mat: substitute costs matrix of shape 20x20 used for AOI differentiating
    :param t_bin: temporal bin for quantifying fixation durations
    """

    dist = np.nan
    if len(p) * len(q) > 0:
        p_x, p_y, p_dur = p.columns
        q_x, q_y, q_dur = q.columns
        p_filtered = p.query(f"0 <= {p_x} <= 1 and 0 <= {p_y} <= 1")
        q_filtered = q.query(f"0 <= {q_x} <= 1 and 0 <= {q_y} <= 1")
        p_transformed = _transform_path(path=p_filtered, t_bin=t_bin)
        q_transformed = _transform_path(path=q_filtered, t_bin=t_bin)

        dp = np.zeros((len(p_transformed) // 2 + 1, len(q_transformed) // 2 + 1))
        dp[0, :] = np.inf
        dp[:, 0] = np.inf
        dp[0, 0] = 0
        for i in range(1, len(p_transformed) // 2 + 1):
            for j in range(1, len(q_transformed) // 2 + 1):
                dp[i, j] = np.minimum(dp[i - 1, j] + 1, dp[i, j - 1] + 1)
                p_x, p_y = p_transformed[2 * (i - 1)], p_transformed[2 * (i - 1) + 1]
                q_x, q_y = q_transformed[2 * (j - 1)], q_transformed[2 * (j - 1) + 1]
                if p_x == q_x and p_y == q_y:
                    dp[i, j] = np.minimum(dp[i, j], dp[i - 1, j - 1])
                else:
                    cost = sub_mat[ord(p_x) - 97][ord(p_y) - 65]
                    dp[i, j] = np.minimum(dp[i, j], dp[i - 1, j - 1] + cost)

        dist = dp[-1, -1]

    return dist


@jit(forceobj=True, looplift=True)
def calc_man_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates Mannan distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    """

    dist = np.nan
    if len(p) * len(q) > 0:
        p_data = p.values
        q_data = q.values

        sp, sq = 0, 0
        for p_dot in p_data:
            sp += ((q_data - p_dot) ** 2).sum(axis=1).min()
        for q_dot in q_data:
            sq += ((p_data - q_dot) ** 2).sum(axis=1).min()

        dist = len(q_data) * sp + len(p_data) * sq
        dist /= 4 * len(p_data) * len(q_data)

    return dist


@jit(forceobj=True, looplift=True)
def calc_eye_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates Mannan distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    """

    dist = np.nan
    if len(p) * len(q) > 0:
        p_data = p.values
        q_data = q.values

        dist = 0
        for p_dot in p_data:
            dist += ((q_data - p_dot) ** 2).sum(axis=1).min()
        for q_dot in q_data:
            dist += ((p_data - q_dot) ** 2).sum(axis=1).min()

        dist /= max(len(p_data), len(q_data))

    return dist


@jit(forceobj=True, looplift=True)
def calc_dfr_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates Discrete Frechet distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    """

    dist = np.nan
    if len(p) * len(q) > 0:
        p_data = p.values
        q_data = q.values

        dp = np.ones((len(p_data), len(q_data))) * (-1)
        dp[0, 0] = ((p_data[0] - q_data[0]) ** 2).sum()
        for i in range(1, len(p_data)):
            dp[i, 0] = max(dp[i - 1, 0], ((p_data[i] - q_data[0]) ** 2).sum())
        for j in range(1, len(q_data)):
            dp[0, j] = max(dp[0, j - 1], ((p_data[0] - q_data[j]) ** 2).sum())

        for i in range(1, len(p_data)):
            for j in range(1, len(q_data)):
                dp[i, j] = max(
                    ((p_data[i] - q_data[j]) ** 2).sum(),
                    min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                )

        dist = dp[-1, -1]

    return dist


@jit(forceobj=True, looplift=True)
def calc_tde_dist(p: pd.DataFrame, q: pd.DataFrame, k: int = 1) -> float:
    """
    Calculates Time Delay Embedding distance between paths p and q
    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    :param k: number of scanpath batches
    """

    dist = np.nan
    assert k > 0, "k must be strictly positive"
    if k <= min(len(p), len(q)):
        p_data = p.values
        q_data = q.values

        dist = 0
        for i in range(len(p_data) // k):
            for j in range(len(q_data) // k):
                p_batch = p_data[i * k : (i + 1) * k]
                q_batch = q_data[j * k : (j + 1) * k]
                if len(p_batch) == len(q_batch):
                    dist += ((p_batch - q_batch) ** 2).sum()

        dist /= (len(p_data) // k) * (len(q_data) // k)

    return dist


@jit(forceobj=True, looplift=True)
def calc_mm_features(
    p: pd.DataFrame, q: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
    """
    Calculates MultiMatch features between paths p and q
    :param p: pd.DataFrame containing columns (x, y, duration) only
    :param q: pd.DataFrame containing columns (x, y, duration) only
    """

    shape, angle = np.nan, np.nan
    length, pos, duration = np.nan, np.nan, np.nan

    if len(p) * len(q) > 0:
        p_x, p_y, p_dur = p.columns
        q_x, q_y, q_dur = q.columns
        p_prep = p.rename(
            columns=dict([(p_x, "start_x"), (p_y, "start_y"), (p_dur, "duration")])
        )
        q_prep = q.rename(
            columns=dict([(q_x, "start_x"), (q_y, "start_y"), (q_dur, "duration")])
        )

        p_prep = p_prep.reset_index()
        q_prep = q_prep.reset_index()
        sim = mm.docomparison(
            fixation_vectors1=p_prep,
            fixation_vectors2=q_prep,
            screensize=[1, 1],
        )

        shape, angle, length, pos, duration = sim

    return shape, angle, length, pos, duration
