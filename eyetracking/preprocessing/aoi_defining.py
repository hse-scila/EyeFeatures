from typing import List, Union, Any, Dict
from typing import Any, List, Union

import math
import numpy as np
import pandas as pd
from numba import jit
from bisect import bisect_left

from scipy.stats import gaussian_kde
from scipy.ndimage import maximum_filter, sobel

from eyetracking.utils import _split_dataframe

import matplotlib.pyplot as plt


def _get_fixation_density(
    data: pd.DataFrame, x: str, y: str
) -> tuple[np.ndarray[Any, np.dtype], Any, Any]:
    """
    Finds the fixation density of a given dataframe.
    :param data: DataFrame with fixations.
    :param x: x coordinate of fixation.
    :param y: y coordinate of fixation.
    :return: density for each point in [x_min, x_max] x [y_min, y_max] area
    """
    df = data[[x, y]]
    assert df.shape[0] != 0, "Error: there are no points"
    kde = gaussian_kde(df.values.T)
    X, Y = np.mgrid[
        df[x].min() : df[x].max() : 100j, df[y].min() : df[y].max() : 100j
    ]  # is 100 enough?
    positions = np.vstack([X.ravel(), Y.ravel()])
    return np.reshape(kde(positions), X.shape), X, Y


# TODO: add numba
def threshold_based(
    data: pd.DataFrame,
    x: str,
    y: str,
    W: int,
    threshold: float,
    threshold_dist: float = None,
    type: str = "kmeans",
    pk: List[str] = None,
    aoi_name: str = "AOI",
    inplace: bool = False,
) -> Union[pd.DataFrame, None]:
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
    assert data.shape[0] != 0, "Error: there are no points"
    assert type in ["kmeans", "basic"], "Error: only 'kmeans' or 'basic' are supported"
    if type == "basic":
        assert threshold_dist is not None, "Error: threshold dist must be provided"

    data_splited = _split_dataframe(data, pk, encode=False)
    min_entropy_centers = dict()
    min_entropy = np.inf
    result = None
    aoi_list = []

    for group, current_data in data_splited:
        density, X, Y = _get_fixation_density(current_data, x, y)
        mx = maximum_filter(
            density, size=(W, W)
        )  # for all elements finds maximum in (W x W) window
        loc_max_matrix = np.where((mx == density) & (mx >= threshold), density, 0)
        for i in range(loc_max_matrix.shape[0]):
            for j in range(loc_max_matrix.shape[1]):
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

        loc_max_coord = np.transpose(np.nonzero(loc_max_matrix))

        assert (
            loc_max_coord.shape[0] != 0
        ), "Error: Can't find the maximum with such parameters"

        aoi_counts = dict()
        aoi_points = dict()
        if not inplace:
            aoi_list.clear()

        axis_x = X.T[0]
        axis_y = Y.T[0]
        centers = dict()
        entropy = 0
        for i in range(loc_max_coord.shape[0]):
            centers[f"aoi_{i}"] = [
                X[loc_max_coord[i][0]][0],
                Y[loc_max_coord[i][1]][0],
            ]  # initial centers of each AOI
            aoi_points[f"aoi_{i}"] = [centers[f"aoi_{i}"]]

        for index, row in current_data.iterrows():
            min_dist = np.inf
            min_dist_aoi = None
            x_coord = min(np.searchsorted(axis_x, row[x]), 99)
            y_coord = min(np.searchsorted(axis_y, row[y]), 99)
            for key in centers.keys():  # start of Kmeans algorithm
                if type == "kmeans":
                    dist = math.sqrt(
                        np.sum(
                            (np.array([row[x], row[y]]) - np.array(centers[key])) ** 2
                        )
                    )
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_aoi = key
                if type == "basic":
                    l = np.inf
                    for point in aoi_points[key]:
                        dist = math.sqrt(
                            np.sum((np.array([row[x], row[y]]) - np.array(point)) ** 2)
                        )
                        l = min(l, dist)
                        if dist >= threshold_dist:
                            l = np.inf
                            break
                    if min_dist > l:
                        min_dist = l
                        min_dist_aoi = key

            if (
                type == "basic"
                and min_dist_aoi is not None
                and density[x_coord][y_coord] > threshold
            ):  # ???
                aoi_points[min_dist_aoi].append([row[x], row[y]])
                aoi_counts[min_dist_aoi] = len(aoi_points[min_dist_aoi]) - 1

            if type == "kmeans":
                if (
                    min_dist_aoi in aoi_counts and density[x_coord][y_coord] > threshold
                ):  # recalculate centers of AOI
                    aoi_counts[min_dist_aoi] += 1

                    centers[min_dist_aoi][0] = (
                        centers[min_dist_aoi][0]
                        * (aoi_counts[min_dist_aoi] - 1)
                        / aoi_counts[min_dist_aoi]
                        + row[x] / aoi_counts[min_dist_aoi]
                    )
                    centers[min_dist_aoi][1] = (
                        centers[min_dist_aoi][1]
                        * (aoi_counts[min_dist_aoi] - 1)
                        / aoi_counts[min_dist_aoi]
                        + row[y] / aoi_counts[min_dist_aoi]
                    )

                elif (
                    min_dist_aoi not in aoi_counts
                    and density[x_coord][y_coord] > threshold
                ):
                    aoi_counts[min_dist_aoi] = 1

                    centers[min_dist_aoi][0] += row[x]
                    centers[min_dist_aoi][1] += row[y]
                    centers[min_dist_aoi][0] /= 2
                    centers[min_dist_aoi][1] /= 2
                elif density[x_coord][y_coord] <= threshold:
                    min_dist_aoi = None

            aoi_list.append(min_dist_aoi)

            for count in aoi_counts.values():  # calculate the entropy
                entropy -= (
                    count
                    / current_data.shape[0]
                    * np.log2(count / current_data.shape[0])
                )
            if entropy < min_entropy:  # TODO: use it to compare aoi defining methods
                min_entropy = entropy
                min_entropy_centers = centers

        if not inplace:
            to_concat = current_data.copy()
            to_concat.reset_index(drop=True, inplace=True)
            if result is None:
                result = pd.concat(
                    [to_concat, pd.Series(aoi_list, name=aoi_name)], axis=1
                )
                pk_values = group
                for i in range(len(pk)):
                    result[pk[i]] = pk_values[i]
            else:
                to_concat = pd.concat(
                    [to_concat, pd.Series(aoi_list, name=aoi_name)], axis=1
                )
                pk_values = group
                for i in range(len(pk)):
                    to_concat[pk[i]] = pk_values[i]
                result.reset_index(drop=True, inplace=True)
                result = pd.concat([result, to_concat], axis=0)
    if inplace:
        data[aoi_name] = aoi_list
        return None
    return result


# @jit(forceobj=True, looplift=True)
def gradient_based(
    data: pd.DataFrame,
    x: str,
    y: str,
    W: int,
    threshold: float,
    gradient_eps: float,
    pk: List[str] = None,
    aoi_name: str = "AOI",
    inplace: bool = False,
) -> Union[pd.DataFrame, None]:

    assert data.shape[0] != 0, "Error: there are no points"

    data_splited = _split_dataframe(data, pk, encode=False)
    min_entropy_centers = dict()
    min_entropy = np.inf
    result = None
    aoi_list = []

    for group, current_data in data_splited:
        density, X, Y = _get_fixation_density(current_data, x, y)

        mx = maximum_filter(
            density, size=(W, W)
        )  # for all elements finds maximum in (W x W) window
        loc_max_matrix = np.where((mx == density) & (mx >= threshold), density, 0)
        for i in range(loc_max_matrix.shape[0]):
            for j in range(loc_max_matrix.shape[1]):
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

        loc_max_coord = np.transpose(np.nonzero(loc_max_matrix))

        assert (
            loc_max_coord.shape[0] != 0
        ), "Error: Can't find the maximum with such parameters"

        aoi_counts = dict()
        aoi_points = dict()
        if not inplace:
            aoi_list.clear()

        axis_x = X.T[0]
        axis_y = Y.T[0]
        dist = axis_x[1] - axis_x[0]
        centers = dict()
        entropy = 0
        aoi_matrix = np.zeros((density.shape[0], density.shape[1]), dtype=int)
        # window_x = sobel(density, axis=0)
        # window_y = sobel(density, axis=1)
        # grad_orientation = np.arctan2(window_y, window_x)
        # plt.imshow(grad_orientation)
        # plt.colorbar(label='test')
        # break
        grad_x = sobel(density, axis=1)
        grad_y = sobel(density, axis=0)
        for i in range(loc_max_coord.shape[0]):
            centers[f"aoi_{i}"] = [
                X[loc_max_coord[i][0]][0],
                Y[loc_max_coord[i][1]][0],
            ]  # initial centers of each AOI
            x_coord = loc_max_coord[i][0]
            y_coord = loc_max_coord[i][1]
            aoi_matrix[x_coord][y_coord] = i + 1
            while y_coord >= 0:
                x_coord_left = x_coord
                x_coord_right = x_coord
                while x_coord_left >= 0 or x_coord_right < density.shape[0]:
                    x_coord_left -= 1
                    x_coord_right += 1
                    if x_coord_left >= 0:
                        if aoi_matrix[x_coord_left][y_coord] != 0:
                            x_coord_left = -1
                        else:
                            aoi_matrix[x_coord_left][y_coord] = i + 1
                            if (
                                x_coord_left < centers[f"aoi_{i}"][0] - 1
                                and (
                                    grad_x[x_coord_left][y_coord]
                                    * grad_x[x_coord_left + 1][y_coord]
                                    < 0
                                    or grad_y[x_coord_left][y_coord]
                                    * grad_y[x_coord_left + 1][y_coord]
                                    < 0
                                )
                            ) or density[x_coord_left][y_coord] < threshold:
                                x_coord_left = -1
                    if x_coord_right < density.shape[0]:
                        if aoi_matrix[x_coord_right][y_coord] != 0:
                            x_coord_right = density.shape[0] + 1
                        else:
                            aoi_matrix[x_coord_right][y_coord] = i + 1
                            if (
                                x_coord_right > centers[f"aoi_{i}"][0] + 1
                                and (
                                    grad_x[x_coord_right][y_coord]
                                    * grad_x[x_coord_right - 1][y_coord]
                                    < 0
                                    or grad_y[x_coord_right][y_coord]
                                    * grad_y[x_coord_right - 1][y_coord]
                                    < 0
                                )
                            ) or density[x_coord_right][y_coord] < threshold:
                                x_coord_right = density.shape[0] + 1
                y_coord -= 1

            y_coord = loc_max_coord[i][1]
            while y_coord < density.shape[1]:
                x_coord_left = x_coord
                x_coord_right = x_coord
                while x_coord_left >= 0 or x_coord_right < density.shape[0]:
                    x_coord_left -= 1
                    x_coord_right += 1
                    if x_coord_left >= 0:
                        if aoi_matrix[x_coord_left][y_coord] != 0:
                            x_coord_left = -1
                        else:
                            aoi_matrix[x_coord_left][y_coord] = i + 1
                            if (
                                x_coord_left < centers[f"aoi_{i}"][0] - 1
                                and (
                                    grad_x[x_coord_left][y_coord]
                                    * grad_x[x_coord_left + 1][y_coord]
                                    < 0
                                    or grad_y[x_coord_left][y_coord]
                                    * grad_y[x_coord_left + 1][y_coord]
                                    < 0
                                )
                            ) or density[x_coord_left][y_coord] < threshold:
                                x_coord_left = -1
                    if x_coord_right < density.shape[0]:
                        if aoi_matrix[x_coord_right][y_coord] != 0:
                            x_coord_right = density.shape[0] + 1
                        else:
                            aoi_matrix[x_coord_right][y_coord] = i + 1
                            if (
                                x_coord_right > centers[f"aoi_{i}"][0] + 1
                                and (
                                    grad_x[x_coord_right][y_coord]
                                    * grad_x[x_coord_right - 1][y_coord]
                                    < 0
                                    or grad_y[x_coord_right][y_coord]
                                    * grad_y[x_coord_right - 1][y_coord]
                                    < 0
                                )
                            ) or density[x_coord_right][y_coord] < threshold:
                                x_coord_right = density.shape[0] + 1
                y_coord += 1

        aoi_list = []
        # print(aoi_matrix)
        for index, row in current_data.iterrows():
            x_coord = min(np.searchsorted(axis_x, row[x]), 99)
            y_coord = min(np.searchsorted(axis_y, row[y]), 99)
            if aoi_matrix[x_coord][y_coord] == 0:
                aoi_list.append(None)
            else:
                aoi_list.append(f"aoi_{aoi_matrix[x_coord][y_coord]}")

        if not inplace:
            to_concat = current_data.copy()
            to_concat.reset_index(drop=True, inplace=True)
            if result is None:
                result = pd.concat(
                    [to_concat, pd.Series(aoi_list, name=aoi_name)], axis=1
                )
                pk_values = group
                for i in range(len(pk)):
                    result[pk[i]] = pk_values[i]
            else:
                to_concat = pd.concat(
                    [to_concat, pd.Series(aoi_list, name=aoi_name)], axis=1
                )
                pk_values = group
                for i in range(len(pk)):
                    to_concat[pk[i]] = pk_values[i]
                result.reset_index(drop=True, inplace=True)
                result = pd.concat([result, to_concat], axis=0)
    if inplace:
        data[aoi_name] = aoi_list
        return None
    return result
