from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from numba import jit
from tqdm import tqdm

from eyefeatures.features.extractor import BaseTransformer
from eyefeatures.features.scanpath_complex import (_get_fill_path,
                                                   get_expected_path)
from eyefeatures.utils import Types, _split_dataframe


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
        expected_paths_method: str = "mean",
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
        :param expected_paths_method: method to calculate expected path ("mean" or "fwp")
        :param return_df: Return pd.Dataframe object or np.ndarray
        """

        self.fill_path = fill_path
        self.requires_duration = False
        self.expected_paths = expected_paths
        self.expected_paths_method = expected_paths_method
        super(DistanceTransformer, self).__init__(
            x=x, y=y, duration=duration, path_pk=path_pk, pk=pk, return_df=return_df
        )

    def _get_required(self) -> List[Tuple[Any, str]]:
        return [
            (self.x, "x"),
            (self.y, "y"),
            (self.pk, "pk"),
            (self.path_pk, "path_pk"),
            (self.expected_paths_method, "expected_paths_method"),
        ]

    def _get_partition(self, df: Types.Data) -> Types.Partition:
        if not isinstance(df, List) and not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be DataFrame or Partition")
        return df if isinstance(df, List) else _split_dataframe(df=df, pk=self.pk)

    @staticmethod
    def _get_path_group(pg: np.ndarray) -> str:
        return "_".join(str(v) for v in pg)

    def get_expected_paths(self) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        return self.expected_paths

    def fit(self, X: pd.DataFrame, y=None):
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
                method=self.expected_paths_method,
            )

        # calculate filling path if not given
        if self.fill_path is None:
            duration = "duration_est" if self.requires_duration else None
            self.fill_path = _get_fill_path(
                data=list(self.expected_paths.values()),
                x="x_est",
                y="y_est",
                duration=duration,
                method=self.expected_paths_method,
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
        expected_paths_method: str = "mean",
        return_df: bool = True,
    ):
        """
        :param methods: list of methods to use ("euc", "hau", "dfr", "eye", "man", "dtw")
        :param x: Column name of x-coordinate
        :param y: Column name of y-coordinate
        :param path_pk: List of column names of groups to calculate expected path
        :param pk: List of column names used to split pd.Dataframe
        :param expected_paths: Dict which was returned from get_expected_path method with the same params
        :param fill_path: pd.DataFrame path which was returned from get_fill_path method for the same expected_paths
        :param expected_paths_method: method to calculate expected path ("mean" or "fwp")
        :param return_df: Return pd.Dataframe object else np.ndarray
        """

        super(SimpleDistances, self).__init__(
            x=x,
            y=y,
            pk=pk,
            path_pk=path_pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            expected_paths_method=expected_paths_method,
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
                expected_paths_method=self.expected_paths_method,
                return_df=self.return_df,
            )
            dataframes.append(self._methods_cls[method].transform(data_part))

        features_df = pd.concat(dataframes, axis=1).add_suffix(
            "_" + self.expected_paths_method
        )
        return features_df if self.return_df else features_df.values


class EucDist(DistanceTransformer):
    """Calculates Euclidean distance between given and expected scanpaths."""

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(EucDist, self)._check_init(
            super(EucDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(EucDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["euc_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(EucDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_euc_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns, index=group_names)
        return features_df if self.return_df else features_df.values


class HauDist(DistanceTransformer):
    """Calculates Hausdorff distance between given and expected scanpaths."""

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(HauDist, self)._check_init(
            super(HauDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(HauDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["hau_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(HauDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_hau_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns, index=group_names)
        return features_df if self.return_df else features_df.values


class DTWDist(DistanceTransformer):
    """Calculates Dynamic Time Warp distance between given and expected scanpaths."""

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(DTWDist, self)._check_init(
            super(DTWDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(DTWDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["dtw_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(DTWDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_dtw_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns, index=group_names)
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
        t_bin: int = 20,
        expected_paths: Dict[str, pd.DataFrame] = None,
        fill_path: pd.DataFrame = None,
        expected_paths_method: str = "mean",
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
        :param expected_paths_method: method to calculate expected path ("mean" or "fwp")
        :param return_df: Return pd.Dataframe object else np.ndarray
        """

        self.t_bin = t_bin
        self.sub_mat = sub_mat
        if sub_mat.shape != (20, 20):
            raise ValueError(
                f"Sub matrix size must be of shape (20, 20), got {sub_mat.shape}"
            )

        super(ScanMatchDist, self).__init__(
            x=x,
            y=y,
            duration=duration,
            path_pk=path_pk,
            pk=pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            expected_paths_method=expected_paths_method,
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
            (self.t_bin, "t_bin"),
            (self.sub_mat, "sub_mat"),
        ]

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(ScanMatchDist, self)._check_init(
            super(ScanMatchDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(ScanMatchDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["scan_match_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(ScanMatchDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_scan_match_dist(
                group_path[[self.x, self.y, self.duration]],
                expected_path,
                t_bin=self.t_bin,
                sub_mat=self.sub_mat,
            )
            features.append([dist])

        features_df = pd.DataFrame(
            data=features, columns=columns, index=group_names
        ).add_suffix("_" + self.expected_paths_method)
        return features_df if self.return_df else features_df.values

    def __repr__(self, **kwargs):
        return f"ScanMatch()"


class MannanDist(DistanceTransformer):
    """Calculates Mannan distance between given and expected scanpaths."""

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(MannanDist, self)._check_init(
            super(MannanDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(MannanDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["man_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(MannanDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_man_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns, index=group_names)
        return features_df if self.return_df else features_df.values


class EyeAnalysisDist(DistanceTransformer):
    """Calculates Eye Analysis distance between given and expected scanpaths."""

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(EyeAnalysisDist, self)._check_init(
            super(EyeAnalysisDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(EyeAnalysisDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["eye_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(EyeAnalysisDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_eye_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns, index=group_names)
        return features_df if self.return_df else features_df.values


class DFDist(DistanceTransformer):
    """Calculates Discrete Frechet distance between given and expected scanpaths."""

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(DFDist, self)._check_init(
            super(DFDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(DFDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["dfr_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(DFDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_dfr_dist(group_path[[self.x, self.y]], expected_path)
            features.append([dist])

        features_df = pd.DataFrame(data=features, columns=columns, index=group_names)
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
        expected_paths_method: str = "mean",
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
        :param expected_paths_method: method to calculate expected path ("mean" or "fwp")
        :param return_df: Return pd.Dataframe object else np.ndarray
        """

        self.k = k
        if self.k <= 0:
            raise ValueError("k must be positive")
        super(TDEDist, self).__init__(
            x=x,
            y=y,
            path_pk=path_pk,
            pk=pk,
            expected_paths=expected_paths,
            fill_path=fill_path,
            expected_paths_method=expected_paths_method,
            return_df=return_df,
        )

    def transform(self, X: Types.Data) -> Union[pd.DataFrame, np.ndarray]:
        # check must-have attributes
        super(TDEDist, self)._check_init(
            super(TDEDist, self)._get_required()
            + ([(self.expected_paths, "expected_paths"), (self.fill_path, "fill_path")])
        )

        # get partitioned dataframes
        data_part: Types.Partition = super(TDEDist, self)._get_partition(X)

        # calculate distances for each group
        group_names = []
        columns, features = ["tde_dist"], []
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(TDEDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            dist = calc_tde_dist(group_path[[self.x, self.y]], expected_path, k=self.k)
            features.append([dist])

        features_df = pd.DataFrame(
            data=features, columns=columns, index=group_names
        ).add_suffix("_" + self.expected_paths_method)
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
        expected_paths_method: str = "mean",
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
        :param expected_paths_method: method to calculate expected path ("mean" or "fwp")
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
            expected_paths_method=expected_paths_method,
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
        group_names = []
        columns = ["mm_shape", "mm_angle", "mm_len", "mm_pos", "mm_duration"]
        for group_nm, group_path in tqdm(data_part):
            group_names.append(group_nm)
            path_group = super(MultiMatchDist, self)._get_path_group(
                group_path.head(1)[self.path_pk].values[0]
            )
            expected_path = (
                self.expected_paths[path_group]
                if path_group in self.expected_paths.keys()
                else self.fill_path
            )
            shape, angle, length, pos, duration = calc_mm_features(
                group_path[[self.x, self.y, self.duration]], expected_path
            )
            features.append([shape, angle, length, pos, duration])

        features_df = pd.DataFrame(
            data=features, columns=columns, index=group_names
        ).add_suffix("_" + self.expected_paths_method)
        return features_df if self.return_df else features_df.values


# ===================== FUNCTIONS =====================
def _transform_fixation(x, y, duration, t_bin):
    if x < 0 or x > 1 or y < 0 or y > 1:
        raise ValueError(
            "Fixations domain must be from unit square (i.e. in [0, 1] x [0, 1])"
        )
    character = chr(97 + int(99 * x) // 5) + chr(65 + int(99 * y) // 5)
    return character * int(duration // t_bin)


def _transform_path(path: pd.DataFrame, t_bin: int) -> str:
    path = path.values
    encoded_fixations = [
        _transform_fixation(x, y, duration, t_bin) for x, y, duration in path
    ]
    return "".join(encoded_fixations)


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
    if dist == np.nan:
        print(length, p_aligned, q_aligned)

    return dist


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


def calc_scan_match_dist(
    p: pd.DataFrame,
    q: pd.DataFrame,
    sub_mat: np.ndarray = np.ones((20, 20)),
    t_bin: int = 20,
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
        p_x, p_y, _ = p.columns
        q_x, q_y, _ = q.columns
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


def calc_eye_dist(p: pd.DataFrame, q: pd.DataFrame) -> float:
    """
    Calculates EyeDist distance between paths p and q

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


def calc_tde_dist(p: pd.DataFrame, q: pd.DataFrame, k: int = 1) -> float:
    """
    Calculates Time Delay Embedding distance between paths p and q

    :param p: pd.DataFrame containing columns (x, y) only
    :param q: pd.DataFrame containing columns (x, y) only
    :param k: number of scanpath batches
    """

    dist = np.nan
    if k <= 0:
        raise ValueError("k must be positive")
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


def calc_mm_features(
    p: pd.DataFrame, q: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
    """
    Calculates MultiMatch features between paths p and q

    :param p: pd.DataFrame containing columns (x, y, duration) only
    :param q: pd.DataFrame containing columns (x, y, duration) only
    """

    n, m = len(p), len(q)
    if n < 2 or m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # calculate pairwise saccade difference matrix
    p_sac_x, p_sac_y = np.diff(p.values[:, 0]), np.diff(p.values[:, 1])
    q_sac_x, q_sac_y = np.diff(q.values[:, 0]), np.diff(q.values[:, 1])

    dist = []
    for i in range(n - 1):
        delta_x = p_sac_x[i] * np.ones(m - 1) - q_sac_x
        delta_y = p_sac_y[i] * np.ones(m - 1) - q_sac_y
        dist.append(np.sqrt(delta_x**2 + delta_y**2))

    # prepare input for scipy coo_matrix
    nodes_from, nodes_to, edge_weights = [], [], []
    for i in range(n - 2):
        for j in range(m - 2):
            v = i * (m - 1) + j
            r, d, r_d = v + 1, v + (m - 1), v + (m - 1) + 1
            nodes_from.extend([v] * 3)
            nodes_to.extend([r, d, r_d])
            edge_weights.extend([dist[i][j + 1], dist[i + 1][j], dist[i + 1][j + 1]])

    for j in range(m - 2):
        v = (n - 2) * (m - 1) + j
        nodes_from.append(v)
        nodes_to.append(v + 1)
        edge_weights.append(dist[n - 2][j + 1])

    for i in range(n - 2):
        v = i * (m - 1) + m - 2
        nodes_from.append(v)
        nodes_to.append(v + (m - 1))
        edge_weights.append(dist[i + 1][m - 2])

    n_nodes = (n - 1) * (m - 1)
    sparse_dist = scipy.sparse.coo_matrix(
        (edge_weights, (nodes_from, nodes_to)), shape=(n_nodes, n_nodes)
    ).tocsr()

    # find and restore the shortest path using dijkstra
    _, predecessors = scipy.sparse.csgraph.dijkstra(
        csgraph=sparse_dist, indices=0, return_predecessors=True, directed=True
    )

    cur = n_nodes - 1
    shortest_path = []
    while cur != -9999:
        shortest_path.append(cur)
        cur = predecessors[cur]

    shortest_path.reverse()

    # calculate features using the shortest path
    p_fix_x, p_fix_y = p.values[:, 0], p.values[:, 0]
    q_fix_x, q_fix_y = q.values[:, 0], q.values[:, 1]
    p_fix_dur, q_fix_dur = p.values[:, 2], q.values[:, 2]
    p_sac_rho, p_sac_phi = np.sqrt(p_sac_x**2 + p_sac_y**2), np.arctan2(
        p_sac_y, p_sac_x
    )
    q_sac_rho, q_sac_phi = np.sqrt(q_sac_x**2 + q_sac_y**2), np.arctan2(
        q_sac_y, q_sac_x
    )

    shape_diff, angle_diff = [], []
    length_diff, position_diff, duration_diff = [], [], []
    for v in shortest_path:
        # get corresponding saccades
        i, j = v // (m - 1), v % (m - 1)
        # calculate shape diff
        delta_x = p_sac_x[i] - q_sac_x[j]
        delta_y = p_sac_y[i] - q_sac_y[j]
        shape_diff.append(np.sqrt(delta_x**2 + delta_y**2))
        # calculate angle diff
        angle_diff.append(abs(p_sac_phi[i] - q_sac_phi[j]))
        # calculate length diff
        length_diff.append(abs(p_sac_rho[i] - q_sac_rho[j]))
        # calculate position diff
        fix_delta_x = p_fix_x[i] - q_fix_x[j]
        fix_delta_y = p_fix_y[i] - q_fix_y[j]
        position_diff.append(np.sqrt(fix_delta_x**2 + fix_delta_y**2))
        # calculate duration diff
        mx_dur = max(p_fix_dur[i], q_fix_dur[j])
        duration_diff.append(abs(p_fix_dur[i] - q_fix_dur[j]) / mx_dur)

    shape_m = np.median(shape_diff)
    angle_m = np.median(angle_diff)
    length_m = np.median(length_diff)
    position_m = np.median(position_diff)
    duration_m = np.median(duration_diff)

    # normalize features into [0, 1]
    shape_m /= np.sqrt(8)  # max_shape = 2 * len(diagonal)
    angle_m /= 2 * np.pi  # max_angle = 2 * pi
    length_m /= np.sqrt(2)  # max_length = len(diagonal)
    position_m /= np.sqrt(2)  # max_position = len(diagonal)

    # return similarities
    return 1 - shape_m, 1 - angle_m, 1 - length_m, 1 - position_m, 1 - duration_m
