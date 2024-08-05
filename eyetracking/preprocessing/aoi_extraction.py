import math
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy.ndimage import maximum_filter, prewitt, sobel
from sklearn.base import BaseEstimator, TransformerMixin

from eyetracking.features.measures import Entropy
from eyetracking.preprocessing.base import BaseAOIPreprocessor
from eyetracking.utils import _split_dataframe


# ======== AOI PREPROCESSORS ========
class ThresholdBased(BaseAOIPreprocessor):
    """
    Defines the AOI for each fixation using density maximum and Kmeans (Find local maximum, pre-threshold it, and use it as a center of aoi)
    :param data: DataFrame with fixations.
    :param x: x coordinate of fixation.
    :param y: y coordinate of fixation.
    :param W: size of search window.
    :param threshold: threshold density.
    :param pk: list of column names used to split pd.DataFrame.
    :param aoi_name: name of AOI column.
    :param inplace: whether to modify the DataFrame rather than creating a new one
    :return: DataFrame with AOI column or None if inplace=True
    """

    def __init__(
        self,
        x: str = None,
        y: str = None,
        window_size: int = None,
        threshold: float = None,
        pk: List[str] = None,
        aoi_name: str = None,
        algorithm_type: str = "kmeans",
        threshold_dist: float = None,
    ):
        super().__init__(x=x, y=y, t=None, aoi=aoi_name, pk=pk)
        self.window_size = window_size
        self.threshold = threshold
        self.algorithm_type = algorithm_type
        self.threshold_dist = threshold_dist

    def _check_params(self):
        m = "ThresholdBased"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.window_size is not None, self._err_no_field(m, "window_size")
        assert self.threshold is not None, self._err_no_field(m, "threshold")
        assert self.window_size > 0, "Error: window size must be greater than zero"
        assert (
            self.threshold >= 0
        ), "Error: threshold must be greater than zero or equal to zero"
        assert self.algorithm_type in [
            "kmeans",
            "basic",
        ], "Error: only 'kmeans' or 'basic' are supported"
        if self.algorithm_type == "basic":
            assert self.threshold_dist is not None, self._err_no_field(
                m, "threshold_dist"
            )

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:

        assert X.shape[0] != 0, "Error: there are no points"
        X.drop(columns=self.pk, inplace=True)

        aoi_list = []
        density, X_grid, Y_grid = super()._get_fixation_density(self, X)
        mx = maximum_filter(
            density, size=(self.window_size, self.window_size)
        )  # for all elements finds maximum in (W x W) window
        loc_max_matrix = np.where((mx == density) & (mx >= self.threshold), density, 0)
        loc_max_coord = super()._build_local_max_coordinates(loc_max_matrix)

        assert (
            loc_max_coord.shape[0] != 0
        ), "Error: Can't find the maximum with such parameters"

        aoi_counts = dict()
        aoi_points = dict()

        axis_x = X_grid.T[0]
        axis_y = Y_grid.T[0]
        centers = dict()
        entropy = 0
        for i in range(loc_max_coord.shape[0]):
            centers[f"aoi_{i}"] = [
                X_grid[loc_max_coord[i][0]][0],
                Y_grid[loc_max_coord[i][1]][0],
            ]  # initial centers of each AOI
            aoi_points[f"aoi_{i}"] = [centers[f"aoi_{i}"]]

        for index, row in X.iterrows():
            min_dist = np.inf
            min_dist_aoi = None
            x_coord = min(np.searchsorted(axis_x, row[self.x]), 99)
            y_coord = min(np.searchsorted(axis_y, row[self.y]), 99)
            for key in centers.keys():  # start of Kmeans algorithm
                if self.algorithm_type == "kmeans":
                    dist = math.sqrt(
                        np.sum(
                            (
                                np.array([row[self.x], row[self.y]])
                                - np.array(centers[key])
                            )
                            ** 2
                        )
                    )
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_aoi = key
                if self.algorithm_type == "basic":
                    l = np.inf
                    for point in aoi_points[key]:
                        dist = math.sqrt(
                            np.sum(
                                (np.array([row[self.x], row[self.y]]) - np.array(point))
                                ** 2
                            )
                        )
                        l = min(l, dist)
                        if dist >= self.threshold_dist:
                            l = np.inf
                            break
                    if min_dist > l:
                        min_dist = l
                        min_dist_aoi = key

            if (
                self.algorithm_type == "basic"
                and min_dist_aoi is not None
                and density[x_coord][y_coord] > self.threshold
            ):  # ???
                aoi_points[min_dist_aoi].append([row[self.x], row[self.y]])
                aoi_counts[min_dist_aoi] = len(aoi_points[min_dist_aoi]) - 1

            if self.algorithm_type == "kmeans":
                if (
                    min_dist_aoi in aoi_counts
                    and density[x_coord][y_coord] > self.threshold
                ):  # recalculate centers of AOI
                    aoi_counts[min_dist_aoi] += 1

                    centers[min_dist_aoi][0] = (
                        centers[min_dist_aoi][0]
                        * (aoi_counts[min_dist_aoi] - 1)
                        / aoi_counts[min_dist_aoi]
                        + row[self.x] / aoi_counts[min_dist_aoi]
                    )
                    centers[min_dist_aoi][1] = (
                        centers[min_dist_aoi][1]
                        * (aoi_counts[min_dist_aoi] - 1)
                        / aoi_counts[min_dist_aoi]
                        + row[self.y] / aoi_counts[min_dist_aoi]
                    )

                elif (
                    min_dist_aoi not in aoi_counts
                    and density[x_coord][y_coord] > self.threshold
                ):
                    aoi_counts[min_dist_aoi] = 1

                    centers[min_dist_aoi][0] += row[self.x]
                    centers[min_dist_aoi][1] += row[self.y]
                    centers[min_dist_aoi][0] /= 2
                    centers[min_dist_aoi][1] /= 2
                elif density[x_coord][y_coord] <= self.threshold:
                    min_dist_aoi = None

            aoi_list.append(min_dist_aoi)

        X[self.aoi] = aoi_list
        return X


class GradientBased(BaseAOIPreprocessor):

    def __init__(
        self,
        x: str = None,
        y: str = None,
        window_size: int = None,
        threshold: float = None,
        pk: List[str] = None,
        aoi_name: str = None,
    ):
        super().__init__(x=x, y=y, t=None, aoi=aoi_name, pk=pk)
        self.window_size = window_size
        self.threshold = threshold

    def _check_params(self):
        m = "GradientBased"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.window_size is not None, self._err_no_field(m, "window_size")
        assert self.threshold is not None, self._err_no_field(m, "threshold")
        assert self.window_size > 0, "Error: window size must be greater than zero"
        assert (
            self.threshold >= 0
        ), "Error: threshold must be greater than zero or equal to zero"

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        assert X.shape[0] != 0, "Error: there are no points"
        X.drop(columns=self.pk, inplace=True)

        aoi_list = []
        density, X_grid, Y_grid = super()._get_fixation_density(self, X)

        mx = maximum_filter(
            density, size=(self.window_size, self.window_size)
        )  # for all elements finds maximum in (W x W) window
        loc_max_matrix = np.where((mx == density) & (mx >= self.threshold), density, 0)
        loc_max_coord = super()._build_local_max_coordinates(loc_max_matrix)

        assert (
            loc_max_coord.shape[0] != 0
        ), "Error: Can't find the maximum with such parameters"

        axis_x = X_grid.T[0]
        axis_y = Y_grid.T[0]
        centers = dict()
        entropy = 0
        aoi_matrix = np.zeros((density.shape[0], density.shape[1]), dtype=int)
        horizontal_sobel = sobel(density, axis=0)
        vertical_sobel = sobel(density, axis=1)
        magnitude_sobel = np.sqrt(horizontal_sobel**2 + vertical_sobel**2)
        magnitude_sobel = np.pad(
            magnitude_sobel, 2, mode="constant", constant_values=-1
        )
        # centers_magn = []
        queue_of_centers = []
        for i in range(loc_max_coord.shape[0]):
            centers[f"aoi_{i}"] = (
                X_grid[loc_max_coord[i][0]][0],
                Y_grid[loc_max_coord[i][1]][0],
            )  # initial centers of each AOI
            queue_of_centers.append([])
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if (
                        0 <= loc_max_coord[i][0] + j < density.shape[0]
                        and 0 <= loc_max_coord[i][1] + k < density.shape[0]
                        and not (j == 0 and k == 0)
                        and aoi_matrix[loc_max_coord[i][0] + j][loc_max_coord[i][1] + k]
                        == 0
                    ):
                        queue_of_centers[-1].append(
                            (loc_max_coord[i][0] + j, loc_max_coord[i][1] + k)
                        )
            # centers_magn.append(magnitude[loc_max_coord[i][0]][loc_max_coord[i][1]])
            aoi_matrix[loc_max_coord[i][0]][loc_max_coord[i][1]] = i + 1
        ind = 0
        while any([len(x) > 0 for x in queue_of_centers]) > 0:
            if len(queue_of_centers[ind]) > 0:
                x_coord, y_coord = queue_of_centers[ind].pop(0)
                if aoi_matrix[x_coord][y_coord] != 0:  # are all fixation covered?
                    continue
                window_magnitude = magnitude_sobel[
                    x_coord + 1 : x_coord + 4, y_coord + 1 : y_coord + 4
                ]  # Add 2 for padding
                max_magnitude = -1
                aoi_to_add = None
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if (
                            0 <= x_coord + j < density.shape[0]
                            and 0 <= y_coord + k < density.shape[0]
                            and aoi_matrix[x_coord + j][y_coord + k] != 0
                            and not (j == 0 and k == 0)
                            and max_magnitude <= window_magnitude[1 + j][1 + k]
                        ):
                            aoi_to_add = aoi_matrix[x_coord + j][y_coord + k]
                            max_magnitude = window_magnitude[1 + j][1 + k]
                        elif (
                            0 <= x_coord + j < density.shape[0]
                            and 0 <= y_coord + k < density.shape[0]
                            and aoi_matrix[x_coord + j][y_coord + k] == 0
                            and not (j == 0 and k == 0)
                        ):  # and magnitude[1 + j][1 + k] >= gradient_eps:
                            queue_of_centers[ind].append((x_coord + j, y_coord + k))
                aoi_matrix[x_coord][y_coord] = aoi_to_add
            ind = (ind + 1) % len(queue_of_centers)

        for index, row in X.iterrows():
            x_coord = min(np.searchsorted(axis_x, row[self.x]), 99)
            y_coord = min(np.searchsorted(axis_y, row[self.y]), 99)
            if aoi_matrix[x_coord][y_coord] == 0:
                aoi_list.append(None)
            else:
                aoi_list.append(f"aoi_{aoi_matrix[x_coord][y_coord]}")

        X[self.aoi] = aoi_list
        return X


class OverlapCLustering(BaseAOIPreprocessor):

    def __init__(
        self,
        x: str = None,
        y: str = None,
        diameters: str = None,
        centers: str = None,
        pk: List[str] = None,
        aoi_name: str = None,
        eps: float = 0.0,
    ):
        super().__init__(x=x, y=y, t=None, aoi=aoi_name, pk=pk)
        self.diameters = diameters
        self.centers = centers
        self.eps = eps

    def _check_aoi(self):
        m = "OverlapCLustering"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.diameters is not None, self._err_no_field(m, "diameters")
        assert self.centers is not None, self._err_no_field(m, "centers")

    def _build_clusters(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.aoi] = 0
        cl = 1
        for index, row in X.iterrows():
            if X.loc[index, X.columns == self.aoi].values[0] == 0:
                X.loc[index, X.columns == self.aoi] = cl
                center = row[self.centers]
                diameter = row[self.diameters]
                X["diff_center"] = X[self.centers].apply(
                    lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
                )
                X["diff_diam"] = abs(diameter - X[self.diameters]) / 2
                fixation_in_cluster = X[X["diff_center"] <= X["diff_diam"]].index
                X.loc[fixation_in_cluster, self.aoi] = cl
                cl += 1

        return X

    def _merge_clusters(self, X: pd.DataFrame) -> pd.DataFrame:
        used = []
        while len(X[~X[self.aoi].isin(used)]) > 0:
            max_cluster = (
                X[~X[self.aoi].isin(used)].groupby(self.aoi).count().idxmax().iloc[0]
            )
            # print(max_cluster)
            points = X[X[self.aoi] == max_cluster].index.values.tolist()
            ind = 0
            end_ = len(points)
            used.append(max_cluster)
            while ind < end_:
                row = X.iloc[points[ind]]
                X["length"] = X[self.centers].apply(
                    lambda p: np.linalg.norm(p - row[self.centers])
                )
                to_merge = X[
                    (
                        X["length"]
                        <= abs((X[self.diameters] + row[self.diameters]) / 2 + self.eps)
                    )
                    & (~X[self.aoi].isin(used))
                ][self.aoi].unique()
                add_fixations = X[
                    (X[self.aoi].isin(to_merge)) & (~X[self.aoi].isin(used))
                ].index.values
                X.loc[X[self.aoi].isin(to_merge), (X.columns == self.aoi)] = max_cluster
                points.extend(add_fixations)
                ind += 1
                end_ += len(add_fixations)

        return X

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X.drop(columns=self.pk, inplace=True)
        X.reset_index(drop=True, inplace=True)
        copy_X = X.copy()
        copy_X = self._build_clusters(copy_X)
        copy_X = self._merge_clusters(copy_X)
        X[self.aoi] = copy_X[self.aoi]
        return X


# ======== EXTRACTOR FOR AOI CLASSES ========
class AOIExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        methods: List[BaseAOIPreprocessor],
        x: str,
        y: str,
        window_size: int = None,
        threshold: float = None,
        pk: List[str] = None,
        aoi_name: str = None,
        show_best: bool = False,
    ):
        self.x = x
        self.y = y
        self.methods = methods
        self.window_size = window_size
        self.threshold = threshold
        self.pk = pk
        self.aoi = aoi_name
        self.show_best = show_best

    # @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        for method in self.methods:
            method.x = self.x
            method.y = self.y
            if self.window_size is not None:
                method.window_size = self.window_size
            if self.threshold is not None:
                method.threshold = self.threshold
            method.pk = self.pk
            method.aoi = self.aoi
            method.fit(X)
        return self

    # @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.methods is None:
            return X

        data_df: pd.DataFrame = X[[self.x, self.y]]
        if self.pk is not None:
            data_df = pd.concat([data_df, X[self.pk]], axis=1)

        fixations = None
        groups: List[str, pd.DataFrame] = _split_dataframe(
            data_df, self.pk, encode=False
        )
        entropy_transformer = Entropy(aoi=self.aoi, pk=self.pk)
        prev_pattern = dict()
        for group_ids, group_X in groups:
            min_entropy = np.inf
            fixations_with_aoi = None
            add_pattern = None
            for method in self.methods:
                cur_fixations = method.transform(group_X)
                all_areas = np.unique(cur_fixations[self.aoi].values)
                areas_names = [f"aoi_{i}" for i in range(len(all_areas))]
                map_areas = dict(zip(all_areas, areas_names))
                cur_fixations[self.aoi] = cur_fixations[self.aoi].map(map_areas)
                used = []
                new_pattern = dict()
                for i in range(len(all_areas), 0, -1):
                    if prev_pattern.get(i, 0) != 0:
                        pattern = prev_pattern[i]
                        for cur_area in areas_names:
                            cur_x_max, cur_y_max, cur_x_min, cur_y_min = (
                                cur_fixations[cur_fixations[self.aoi] == cur_area][
                                    self.x
                                ].max(),
                                cur_fixations[cur_fixations[self.aoi] == cur_area][
                                    self.y
                                ].max(),
                                cur_fixations[cur_fixations[self.aoi] == cur_area][
                                    self.x
                                ].min(),
                                cur_fixations[cur_fixations[self.aoi] == cur_area][
                                    self.y
                                ].min(),
                            )
                            intersection = -1
                            new_name = None
                            for key, value in pattern.items():
                                x_max, y_max, x_min, y_min = (
                                    value[0],
                                    value[1],
                                    value[2],
                                    value[3],
                                )
                                width = min(x_max, cur_x_max) - max(x_min, cur_x_min)
                                height = min(y_max, cur_y_max) - max(y_min, cur_y_min)
                                if (
                                    (height > 0)
                                    and (width > 0)
                                    and (key not in used)
                                    and intersection <= width * height
                                ):
                                    new_name = key
                                    intersection = width * height
                            used.append(new_name)
                        for j in range(len(used)):
                            if used[j] is None:
                                for k in range(len(areas_names)):
                                    if areas_names[k] not in used:
                                        used[j] = areas_names[k]
                                        break
                        cur_fixations[self.aoi] = cur_fixations[self.aoi].map(
                            dict(zip(areas_names, used))
                        )
                        break
                for area in areas_names:
                    cur_x_max, cur_y_max, cur_x_min, cur_y_min = (
                        cur_fixations[cur_fixations[self.aoi] == area][self.x].max(),
                        cur_fixations[cur_fixations[self.aoi] == area][self.y].max(),
                        cur_fixations[cur_fixations[self.aoi] == area][self.x].min(),
                        cur_fixations[cur_fixations[self.aoi] == area][self.y].min(),
                    )
                    new_pattern[area] = [cur_x_max, cur_y_max, cur_x_min, cur_y_min]

                entropy = entropy_transformer.transform(cur_fixations)[
                    "entropy"
                ].values[0][0]
                if min_entropy > entropy:
                    min_entropy = entropy
                    fixations_with_aoi = cur_fixations
                    add_pattern = new_pattern
                    if self.show_best:
                        fixations_with_aoi["best_method"] = method.__class__.__name__
            prev_pattern[len(np.unique(fixations_with_aoi[self.aoi].values))] = (
                add_pattern
            )
            if fixations is None:
                fixations = fixations_with_aoi
            else:
                fixations = pd.concat(
                    [fixations, fixations_with_aoi], ignore_index=True, axis=0
                )

        return fixations
