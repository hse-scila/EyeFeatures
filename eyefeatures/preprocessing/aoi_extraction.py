from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, sobel
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from eyefeatures.features.measures import ShannonEntropy
from eyefeatures.preprocessing.base import BaseAOIPreprocessor
from eyefeatures.utils import _split_dataframe


# ======== AOI PREPROCESSORS ========
class ShapeBased(BaseAOIPreprocessor):
    """Defines AOI using the specified shapes.

    Args:
        x: x coordinate of fixation.
        y: y coordinate of fixation.
        aoi_name: name of AOI column.
        pk: list of column names used to split pd.DataFrame.
        shapes: list of shapes (list of tuple lists). Parameters for shape:\n
            \n
            0: 'r', 'c', 'e': rectangle, circle, ellipse\n
            For the rectangle:\n
            1: coordinates of the lower left corner of the rectangle.\n
            2: coordinates of the upper right corner of the rectangle.\n
            For the circle:\n
            1: coordinates of the center of the circle.\n
            2: radius of the circle.\n
            For the ellipse:\n
            :math:`\\frac{((x - x')\\cos(\\alpha) + (y - y')\\sin(\\alpha))^2}{a^2}`
            :math:`+ \\frac{(-(x - x')\\sin(\\alpha) + (y - y')\\cos(\\alpha))^2}{b^2} = c`

            1: coordinates of the center of the ellipse :math:`(x', y')`.\n
            2: "a" in the ellipse equation\n
            3: "b" in the ellipse equation\n
            4: "c" in the ellipse equation\n
            5: angle of inclination of th ellipse in radians (:math:`\\alpha`)\n
    """

    def __init__(
        self,
        x: str = None,
        y: str = None,
        shapes: List = None,
        aoi_name: str = "AOI",
        pk: List[str] = None,
    ):
        super().__init__(x=x, y=y, t=None, aoi=aoi_name, pk=None)
        self.shapes = shapes
        self.instance = pk

    def _check_params(self):
        m = "ShapeBased"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.shapes is not None, self._err_no_field(m, "shapes")

    def _is_inside_of_fig(self, X: pd.DataFrame, shape_id):
        ind = 0
        for shape in self.shapes[shape_id]:
            if shape[0] == "r":  # Rectangle
                X.loc[
                    (X[self.x] <= shape[2][0])
                    & (X[self.x] >= shape[1][0])
                    & (X[self.y] <= shape[2][1])
                    & (X[self.y] >= shape[1][1]),
                    self.aoi,
                ] = f"aoi_{ind}"
            elif shape[0] == "c":  # Circle
                X["length"] = X.apply(
                    lambda z: np.linalg.norm(
                        np.array((z[self.x], z[self.y])) - np.array(shape[1])
                    ),
                    axis=1,
                )
                X.loc[X["length"] <= shape[2], self.aoi] = f"aoi_{ind}"
            elif shape[0] == "e":  # Ellipse
                X.loc[
                    (
                        (X[self.x] - shape[1][0]) * np.cos(shape[5])
                        + (X[self.y] - shape[1][1]) * np.sin(shape[5])
                    )
                    ** 2
                    / (shape[2] ** 2)
                    + (
                        -(X[self.x] - shape[1][0]) * np.sin(shape[5])
                        + (X[self.y] - shape[1][1]) * np.cos(shape[5])
                    )
                    / (shape[3] ** 2)
                    <= shape[4],
                    self.aoi,
                ] = f"aoi_{ind}"
            ind += 1
        return X[self.aoi]

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        assert X.shape[0] != 0, "Error: there are no points"

        # if self.instance is not None:
        #     X.drop(columns=self.instance, inplace=True)
        # X[self.aoi] = None
        to_concat = []
        flag = False
        if len(self.shapes) != 1:
            flag = True
        shape_id = 0
        if self.instance is None:
            X[self.aoi] = self._is_inside_of_fig(X, shape_id)
            fixations = X
        else:
            instances: List[str, pd.DataFrame] = _split_dataframe(
                X, self.instance, encode=False
            )
            assert (not flag) or len(instances) == len(self.shapes), "Not enough shapes"
            for instance_ids, instance_X in instances:
                instance_X[self.aoi] = self._is_inside_of_fig(instance_X, shape_id)
                to_concat.append(instance_X)
                if flag:
                    shape_id += 1
            fixations = pd.concat(to_concat, axis=0)

        return fixations


class ThresholdBased(BaseAOIPreprocessor):
    """Defines the AOI for each fixation using density maximum and Kmeans.
    Finds local maximum, pre-threshold it, and uses it as a center of aoi.

    Args:
        x: x coordinate of fixation.
        y: y coordinate of fixation.
        window_size: size of search window.
        threshold: threshold density.
        pk: list of column names used to split pd.DataFrame.
        aoi_name: name of AOI column.
        algorithm_type: type of clustering algorithm to use.
        threshold_dist: maximum allowed distance between fixations in single AOI.
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

        if self.pk is not None:
            X.drop(columns=self.pk, inplace=True)

        aoi_list = []
        density, X_grid, Y_grid = self._get_fixation_density(X)

        # For each sliding window (window_size x window_size) finds maximum density
        mx = maximum_filter(density, size=(self.window_size, self.window_size))
        # Filter local maxima (0 if maximus less than threshold)
        loc_max_matrix = np.where((mx == density) & (mx >= self.threshold), density, 0)
        loc_max_coord = super()._find_local_max_coordinates(loc_max_matrix)

        assert (
            loc_max_coord.shape[0] != 0
        ), "Error: Can't find the maximum with such parameters"

        aoi_counts: Dict[str, int] = dict()  # Dict[aoi name] = count of points in aoi
        aoi_points: Dict[str, List[Tuple[float, float]]] = (
            dict()
        )  # Dict[aoi name] = list of points

        axis_x = X_grid.T[0]
        axis_y = Y_grid[0]
        centers: Dict[str, Tuple[float, float]] = dict()  # Dict with the centers of aoi
        for i in range(loc_max_coord.shape[0]):  # Initial centers for each AOI
            centers[f"aoi_{i}"] = (
                X_grid[loc_max_coord[i][0]][0],
                Y_grid[loc_max_coord[i][1]][0],
            )
            aoi_points[f"aoi_{i}"] = [centers[f"aoi_{i}"]]

        for index, row in X.iterrows():
            min_dist = np.inf
            min_dist_aoi = None
            x_coord = min(np.searchsorted(axis_x, row[self.x]), 99)
            y_coord = min(np.searchsorted(axis_y, row[self.y]), 99)
            # Find the nearst aoi
            for key in centers.keys():
                if self.algorithm_type == "kmeans":
                    dist = np.linalg.norm(
                        np.array([row[self.x], row[self.y]]) - np.array(centers[key])
                    )
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_aoi = key
                if self.algorithm_type == "basic":
                    length = np.inf
                    for point in aoi_points[key]:
                        # Find minimal distance between fixation and points in aoi
                        dist = np.linalg.norm(
                            np.array([row[self.x], row[self.y]]) - np.array(point)
                        )
                        length = min(length, dist)
                        if dist >= self.threshold_dist:
                            length = np.inf
                            break
                    if min_dist > length:
                        min_dist = length
                        min_dist_aoi = key

            # Add fixation to the aoi (basic algorithm)
            if (
                self.algorithm_type == "basic"
                and min_dist_aoi is not None
                and density[x_coord][y_coord] > self.threshold
            ):
                aoi_points[min_dist_aoi].append((row[self.x], row[self.y]))
                aoi_counts[min_dist_aoi] = len(aoi_points[min_dist_aoi]) - 1

            # Add fixation to the aoi (kmeans algorithm) and compute the new center
            if self.algorithm_type == "kmeans" and min_dist_aoi is not None:
                if (
                    min_dist_aoi in aoi_counts.keys()
                    and density[x_coord][y_coord] > self.threshold
                ):
                    aoi_counts[min_dist_aoi] += 1

                    new_center_x = (
                        centers[min_dist_aoi][0]
                        * (aoi_counts[min_dist_aoi] - 1)
                        / aoi_counts[min_dist_aoi]
                        + row[self.x] / aoi_counts[min_dist_aoi]
                    )
                    new_center_y = (
                        centers[min_dist_aoi][1]
                        * (aoi_counts[min_dist_aoi] - 1)
                        / aoi_counts[min_dist_aoi]
                        + row[self.y] / aoi_counts[min_dist_aoi]
                    )
                    centers[min_dist_aoi] = (new_center_x, new_center_y)
                elif (
                    min_dist_aoi not in aoi_counts.keys()
                    and density[x_coord][y_coord] > self.threshold
                ):
                    aoi_counts[min_dist_aoi] = 1
                    new_center_x = (centers[min_dist_aoi][0] + row[self.x]) / 2
                    new_center_y = (centers[min_dist_aoi][1] + row[self.y]) / 2
                    centers[min_dist_aoi] = (new_center_x, new_center_y)
                elif density[x_coord][y_coord] <= self.threshold:
                    min_dist_aoi = None

            aoi_list.append(min_dist_aoi)

        X[self.aoi] = aoi_list
        return X


class GradientBased(BaseAOIPreprocessor):
    """Defines the AOI for each fixation using a gradient-based algorithm.
    Locates local maxima, applies thresholding, and uses them as AOI centers.
    After that, uses the Sobel operator to compute the gradient magnitude for
    each point. Next, defines the queue of areas of interest. Algorithm of aoi
    defining:\n
    * Gets the point from the queue. It is a center\n
    * Looks at the points near the center\n
    * Tries to find the point with defined aoi and maximum gradient magnitude\n
    * Adds center to this aoi\n
    * Repeats for all points in the matrix\n

    Args:
        x: X coordinate of fixation.
        y: Y coordinate of fixation.
        window_size: size of search window.
        threshold: threshold density.
        pk: list of column names used to split pd.DataFrame.
        aoi_name: name of AOI column.
    """

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
        if self.pk is not None:
            X.drop(columns=self.pk, inplace=True)

        aoi_list = []
        density, X_grid, Y_grid = super()._get_fixation_density(X)

        # For each sliding window (window_size x window_size) finds maximum density
        mx = maximum_filter(density, size=(self.window_size, self.window_size))
        # Filter local maxima (0 if maximus less than threshold)
        loc_max_matrix = np.where((mx == density) & (mx >= self.threshold), density, 0)
        loc_max_coord = super()._find_local_max_coordinates(loc_max_matrix)

        assert (
            loc_max_coord.shape[0] != 0
        ), "Error: Can't find the maximum with such parameters"

        axis_x = X_grid.T[0]
        axis_y = Y_grid[0]
        centers: Dict[str, Tuple[float, float]] = dict()  # Dict with the centers of aoi
        aoi_matrix = np.zeros((density.shape[0], density.shape[1]), dtype=int)
        # Compute the gradient magnitude
        horizontal_sobel = sobel(density, axis=0)
        vertical_sobel = sobel(density, axis=1)
        magnitude_sobel = np.sqrt(horizontal_sobel**2 + vertical_sobel**2)
        magnitude_sobel = np.pad(
            magnitude_sobel, 2, mode="constant", constant_values=-1
        )

        queue_of_centers: List[List[Tuple[int, int]]] = (
            []
        )  # List of points to add for each aoi
        for i in range(loc_max_coord.shape[0]):  # Initial centers for each AOI
            centers[f"aoi_{i}"] = (
                X_grid[loc_max_coord[i][0]][0],
                Y_grid[loc_max_coord[i][1]][0],
            )
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
            aoi_matrix[loc_max_coord[i][0]][loc_max_coord[i][1]] = i + 1
        ind = 0
        while any(len(x) > 0 for x in queue_of_centers):
            # If the list of points to add for particular aoi is not empty,
            # then try to add them to aoi, else this aoi is built
            if len(queue_of_centers[ind]) > 0:
                x_coord, y_coord = queue_of_centers[ind].pop(
                    0
                )  # Get the point without aoi. It is a center of window
                if aoi_matrix[x_coord][y_coord] != 0:  # are all the fixation covered?
                    continue
                # Add 2 due to padding
                window_magnitude = magnitude_sobel[
                    x_coord + 1 : x_coord + 4, y_coord + 1 : y_coord + 4
                ]
                max_magnitude = -1
                aoi_to_add = None
                # Find point in window with max gradient magnitude and its aoi
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if (
                            0 <= x_coord + j < density.shape[0]
                            and 0 <= y_coord + k < density.shape[0]
                            and not (j == 0 and k == 0)
                        ):
                            # If non-center point has greater magnitude and aoi,
                            # then we take its aoi
                            if (
                                aoi_matrix[x_coord + j][y_coord + k] != 0
                                and max_magnitude <= window_magnitude[1 + j][1 + k]
                            ):
                                aoi_to_add = aoi_matrix[x_coord + j][y_coord + k]
                                max_magnitude = window_magnitude[1 + j][1 + k]
                            # If non-center point has no aoi, add it to queue
                            elif (
                                aoi_matrix[x_coord + j][y_coord + k] == 0
                                and (x_coord + j, y_coord + k)
                                not in queue_of_centers[ind]
                            ):  # and magnitude[1+j][1+k] >= gradient_eps:
                                queue_of_centers[ind].append((x_coord + j, y_coord + k))
                aoi_matrix[x_coord][y_coord] = aoi_to_add
            # Match points from queue with best AOI in window on density
            ind = (ind + 1) % len(queue_of_centers)

        # Match fixations and aoi in aoi matrix
        for index, row in X.iterrows():
            x_coord = min(np.searchsorted(axis_x, row[self.x]), 99)
            y_coord = min(np.searchsorted(axis_y, row[self.y]), 99)
            if aoi_matrix[x_coord][y_coord] == 0:
                aoi_list.append(None)
            else:
                aoi_list.append(f"aoi_{aoi_matrix[x_coord][y_coord]}")

        X[self.aoi] = aoi_list
        return X


class OverlapClustering(BaseAOIPreprocessor):
    """Defines the AOI for each fixation using the overlapping clustering algorithm.

    Args:
        x: X coordinate of fixation.
        y: Y coordinate of fixation.
        diameters: diameters of fixation.
        centers: centers of fixation.
        pk: list of column names used to split pd.DataFrame.
        aoi_name: name of AOI column.
        eps: additional length to sum of radius
    """

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

    def _check_params(self):
        m = "OverlapClustering"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.diameters is not None, self._err_no_field(m, "diameters")
        assert self.centers is not None, self._err_no_field(m, "centers")

    def _build_clusters(self, X: pd.DataFrame) -> pd.DataFrame:
        """First step of the overlapping clustering algorithm.
        Builds the clusters. If the fixation locates inside another
        one, then these fixations are in one aoi.
        """
        X[self.aoi] = 0
        cluster_id = 1
        for index, row in X.iterrows():
            if X.loc[index, X.columns == self.aoi].values[0] == 0:
                X.loc[index, X.columns == self.aoi] = cluster_id
                center = row[self.centers]
                diameter = row[self.diameters]
                # Calculate the distance between centers
                X["diff_center"] = X[self.centers].apply(
                    lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2
                )
                X["diff_diam"] = abs(diameter - X[self.diameters]) / 2
                # Add all fixation, which are
                fixation_in_cluster = X[X["diff_center"] <= X["diff_diam"]].index
                X.loc[fixation_in_cluster, self.aoi] = cluster_id
                cluster_id += 1

        return X

    def _merge_clusters(self, X: pd.DataFrame) -> pd.DataFrame:
        """Second step of the overlapping clustering algorithm.
        Merges the clusters. Selects aoi with the most amount of
        fixation in itself. Creates the queue of fixation and
        starts the cycle of merging.
        """
        used = []
        while len(X[~X[self.aoi].isin(used)]) > 0:
            # Find the largest cluster(aoi)
            max_cluster = (
                X[~X[self.aoi].isin(used)].groupby(self.aoi).count().idxmax().iloc[0]
            )
            points = X[X[self.aoi] == max_cluster].index.values.tolist()
            ind = 0
            end_ = len(points)
            used.append(max_cluster)
            while ind < end_:
                row = X.iloc[points[ind]]
                # Calculate the distance between centers
                X["length"] = X[self.centers].apply(
                    lambda p: np.linalg.norm(p - row[self.centers])
                )
                # Merge areas of interest that intersect the max cluster
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
                # Add fixation from those areas of interest to queue
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
    """Extractor of areas of interest. Selects the partition into
    zones of interest with the lowest entropy.

    Args:
        methods: list of aoi algorithms.
        x: X coordinate of fixation.
        y: Y coordinate of fixation.
        window_size: size of search window.
        threshold: threshold density.
        pk: list of column names used to split pd.DataFrame for scaling.
        instance_columns: names of columns used to split DataFrame for aoi.
        aoi_name: name of AOI column.
        show_best: if true, then return the best method for each instance
    """

    def __init__(
        self,
        methods: List[BaseAOIPreprocessor],
        x: str,
        y: str,
        window_size: int = None,
        threshold: float = None,
        pk: List[str] = None,
        instance_columns: List[str] = None,
        aoi_name: str = None,
        show_best: bool = False,
    ):
        self.x = x
        self.y = y
        self.methods = methods
        self.window_size = window_size
        self.threshold = threshold
        self.pk = pk
        self.instance_columns = instance_columns
        self.aoi = aoi_name
        self.show_best = show_best

    def fit(self, X: pd.DataFrame, y=None):
        for method in self.methods:
            method.x = self.x
            method.y = self.y
            if self.window_size is not None:
                method.window_size = self.window_size
            if self.threshold is not None:
                method.threshold = self.threshold
            # method.pk = self.pk
            method.aoi = self.aoi
            if not isinstance(method, ClusterMixin):
                method.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.methods is None:
            return X

        data_df: pd.DataFrame = X[[self.x, self.y]]
        if self.pk is not None:
            data_df = pd.concat([data_df, X[self.pk]], axis=1)

        if self.instance_columns is not None:
            to_add = [x for x in self.instance_columns if x not in self.pk]
            data_df = pd.concat([data_df, X[to_add]], axis=1)
        else:
            self.instance_columns = self.pk

        fixations = None
        instances: List[str, pd.DataFrame] = _split_dataframe(
            X, self.instance_columns, encode=False
        )
        shapes_id = 0  # For ShapeBased
        # Entropy for selecting the best method
        entropy_transformer = ShannonEntropy(aoi=self.aoi, pk=self.instance_columns)
        # Extract areas of interest for each instance
        for instance_ids, instance_X in instances:
            min_entropy = np.inf
            fixations_with_aoi = None
            # Select best aoi extraction
            for method in self.methods:
                copy_x = None
                copy_y = None
                to_transform = None
                groups: List[str, pd.DataFrame] = _split_dataframe(
                    instance_X, self.pk, encode=False
                )
                # Map points into (100, 100) matrix and build kde for groups
                for group_ids, group_X in groups:
                    if copy_x is None:
                        copy_x = group_X[self.x]
                        copy_y = group_X[self.y]
                    else:
                        copy_x = pd.concat(
                            [copy_x, group_X[self.x]], ignore_index=True, axis=0
                        )
                        copy_y = pd.concat(
                            [copy_y, group_X[self.y]], ignore_index=True, axis=0
                        )
                    group_X[self.x] -= group_X[self.x].mean()
                    group_X[self.y] -= group_X[self.y].mean()
                    if to_transform is None:
                        to_transform = group_X
                    else:
                        to_transform = pd.concat(
                            [to_transform, group_X], ignore_index=True, axis=0
                        )

                if isinstance(method, ClusterMixin):
                    cur_aoi = method.fit_predict(to_transform[[self.x, self.y]])
                    cur_fixations = pd.concat(
                        [to_transform, pd.Series(cur_aoi, name=self.aoi)], axis=1
                    )
                elif isinstance(method, ShapeBased):
                    save_shapes = method.shapes
                    assert (len(save_shapes) == 1) or (
                        len(save_shapes) != len(instances)
                    ), "Not enough shapes"
                    method.shapes = [
                        save_shapes[shapes_id],
                    ]
                    method.instance = None
                    cur_fixations = method.transform(to_transform)
                    method.shapes = save_shapes
                    if len(save_shapes) != 1:
                        shapes_id += 1
                else:
                    cur_fixations = method.transform(to_transform)

                all_areas = np.unique(
                    [el for el in cur_fixations[self.aoi].values if el is not None]
                )
                areas_names = [f"aoi_{i}" for i in range(len(all_areas))]
                map_areas = dict(zip(all_areas, areas_names))
                cur_fixations[self.aoi] = cur_fixations[self.aoi].map(map_areas)
                entropy = entropy_transformer.transform(cur_fixations)[
                    "entropy"
                ].values[0]
                if min_entropy > entropy:
                    min_entropy = entropy
                    fixations_with_aoi = cur_fixations
                    fixations_with_aoi[self.x] = copy_x
                    fixations_with_aoi[self.y] = copy_y
                    if self.show_best:
                        fixations_with_aoi["best_method"] = method.__class__.__name__
            if fixations is None:
                fixations = fixations_with_aoi
            else:
                fixations = pd.concat(
                    [fixations, fixations_with_aoi], ignore_index=True, axis=0
                )

        return fixations


class AOIMatcher(BaseEstimator, TransformerMixin):
    """Matches AOI in the dataset.

    Args:
        x: X coordinate column name.
        y: Y coordinate column name.
        pk: list of column names used to split pd.DataFrame for scaling.
        instance_columns: list of column names used to split pd.DataFrame
                          into the similar instances for aoi extraction.
        aoi: name of AOI column.
        n_aoi: count of aoi in the group.\n
               0: any number the areas of interest.\n
               integer > 0: count of the areas of interest.
    """

    def __init__(
        self,
        x: str,
        y: str,
        pk: List[str] = None,
        instance_columns: List[str] = None,
        aoi: str = None,
        n_aoi: int = 0,
    ):
        self.x = x
        self.y = y
        self.pk = pk
        self.instance_columns = instance_columns
        self.aoi = aoi
        self.n_aoi = n_aoi

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data_df: pd.DataFrame = X.copy()

        fixations = None
        instances: List[str, pd.DataFrame] = _split_dataframe(
            data_df, self.instance_columns, encode=False
        )
        prev_pattern = dict()
        for instance_ids, instance_X in instances:
            copy_x = None
            copy_y = None
            groups: List[str, pd.DataFrame] = _split_dataframe(
                instance_X, self.pk, encode=False
            )
            cur_fixations = None
            for group_ids, group_X in groups:
                if copy_x is None:
                    copy_x = group_X[self.x]
                    copy_y = group_X[self.y]
                else:
                    copy_x = pd.concat(
                        [copy_x, group_X[self.x]], ignore_index=True, axis=0
                    )
                    copy_y = pd.concat(
                        [copy_y, group_X[self.y]], ignore_index=True, axis=0
                    )
                group_X[self.x] -= group_X[self.x].mean()
                group_X[self.y] -= group_X[self.y].mean()
                if cur_fixations is None:
                    cur_fixations = group_X
                else:
                    cur_fixations = pd.concat(
                        [cur_fixations, group_X], ignore_index=True, axis=0
                    )

            # === Correction of the AOI labels ===
            # Make the new aoi labels
            all_areas = np.unique(cur_fixations[self.aoi].values)
            if (self.n_aoi > 0) and (len(all_areas) > self.n_aoi):
                centers = []
                for i in range(len(all_areas)):
                    x = cur_fixations[(cur_fixations[self.aoi] == all_areas[i])][
                        self.x
                    ].mean()
                    y = cur_fixations[(cur_fixations[self.aoi] == all_areas[i])][
                        self.y
                    ].mean()
                    count = (
                        cur_fixations[(cur_fixations[self.aoi] == all_areas[i])]
                        .count()
                        .values[0]
                    )
                    centers.append([all_areas[i], count, x, y, True])
                count_of_aoi = len(all_areas)
                while count_of_aoi > self.n_aoi:
                    min_dist = np.inf
                    points_to_merge = []
                    for i in range(len(centers)):
                        for j in range(i + 1, len(centers)):
                            dist = np.linalg.norm(
                                np.array(centers[i][2:]) - np.array(centers[j][2:])
                            )
                            if (dist < min_dist) and centers[j][-1]:
                                min_dist = dist
                                points_to_merge = [centers[i], centers[j]]
                    cur_fixations.loc[
                        cur_fixations[self.aoi] == points_to_merge[1][0],
                        cur_fixations.columns == self.aoi,
                    ] = points_to_merge[0][0]
                    for i in range(len(centers)):
                        if centers[i][0] == points_to_merge[0][0]:
                            centers[i][2] = (
                                (points_to_merge[0][2] * points_to_merge[0][1])
                                + (points_to_merge[1][2] * points_to_merge[1][1])
                            ) / (points_to_merge[0][1] + points_to_merge[1][1])
                            centers[i][3] = (
                                (points_to_merge[0][3] * points_to_merge[0][1])
                                + (points_to_merge[1][3] * points_to_merge[1][1])
                            ) / (points_to_merge[0][1] + points_to_merge[1][1])
                            centers[i][1] = (
                                points_to_merge[0][1] + points_to_merge[1][1]
                            )
                        if centers[i][0] == points_to_merge[1][0]:
                            centers[i][-1] = False
                    count_of_aoi -= 1

            all_areas = np.unique(cur_fixations[self.aoi].values)
            areas_names = [f"aoi_{i}" for i in range(len(all_areas))]
            map_areas = dict(zip(all_areas, areas_names))
            cur_fixations[self.aoi] = cur_fixations[self.aoi].map(map_areas)
            used = []
            to_zip = []
            new_pattern = dict()
            # Match labels with previous patterns
            for i in range(len(all_areas), 0, -1):
                if prev_pattern.get(i, 0) != 0:
                    pattern = prev_pattern[i]
                    # Compare areas
                    # From Python 3.6, dict.items() order corresponds to
                    # insertion order
                    for key, value in pattern.items():
                        # if key not in used:
                        x_max, y_max, x_min, y_min = (
                            value[0],
                            value[1],
                            value[2],
                            value[3],
                        )
                        intersection = -1
                        new_name = None
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
                            width = min(x_max, cur_x_max) - max(x_min, cur_x_min)
                            height = min(y_max, cur_y_max) - max(y_min, cur_y_min)
                            # Find aoi with the largest intersection
                            if (
                                height > 0
                                and width > 0
                                and (cur_area not in used)
                                and intersection <= width * height
                            ):
                                intersection = width * height
                                new_name = cur_area

                        used.append(new_name)
                        to_zip.append(key)
                    len_of_used = len(used)
                    for j in range(len(areas_names) - len(used)):
                        used.append(None)
                        to_zip.append(areas_names[len_of_used + j])
                    for j in range(len(used)):
                        if used[j] is None:
                            for area in areas_names:
                                if area not in used:
                                    used[j] = area
                    # Match the remaining areas of interest
                    for j in range(len(used)):
                        if used[j] is None:
                            for k in range(len(areas_names)):
                                if areas_names[k] not in used:
                                    used[j] = areas_names[k]
                                    break
                    cur_fixations[self.aoi] = cur_fixations[self.aoi].map(
                        dict(zip(used, to_zip))
                    )
                    break
            # Add new sample
            for area in areas_names:
                cur_x_max, cur_y_max, cur_x_min, cur_y_min = (
                    cur_fixations[cur_fixations[self.aoi] == area][self.x].max(),
                    cur_fixations[cur_fixations[self.aoi] == area][self.y].max(),
                    cur_fixations[cur_fixations[self.aoi] == area][self.x].min(),
                    cur_fixations[cur_fixations[self.aoi] == area][self.y].min(),
                )
                new_pattern[area] = [cur_x_max, cur_y_max, cur_x_min, cur_y_min]
            # === End of the correction ===
            prev_pattern[len(np.unique(cur_fixations[self.aoi].values))] = new_pattern
            cur_fixations[self.x] = copy_x
            cur_fixations[self.y] = copy_y
            if fixations is None:
                fixations = cur_fixations
            else:
                fixations = pd.concat(
                    [fixations, cur_fixations], ignore_index=True, axis=0
                )

        return fixations
