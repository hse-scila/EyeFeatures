from typing import List, Union, Any
import numpy as np
from math import sqrt
import pandas as pd
from numba import jit

from scipy.stats import gaussian_kde
from scipy.ndimage import maximum_filter

from eyetracking.utils import _split_dataframe


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

    data_splited = _split_dataframe(data, pk, encode=True)
    min_entropy_centers = dict()
    min_entropy = np.inf
    result = None
    aoi_list = []

    for group, current_data in data_splited:
        density, X, Y = _get_fixation_density(current_data, x, y)
        mx = maximum_filter(density, size=(W, W))
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
        if not inplace:
            aoi_list.clear()

        centers = dict()
        entropy = 0
        for i in range(loc_max_coord.shape[0]):
            centers[f"aoi_{i}"] = [
                X[loc_max_coord[i][0]][0],
                Y[loc_max_coord[i][1]][0],
            ]  # initial centers of each AOI

        for index, row in current_data.iterrows():
            min_dist = np.inf
            min_dist_aoi = None
            for key in centers.keys():  # start of Kmeans algorithm
                if (
                    sqrt(
                        (row[x] - centers[key][0]) ** 2
                        + (row[y] - centers[key][1]) ** 2
                    )
                    < min_dist
                ):
                    min_dist = sqrt(
                        (row[x] - centers[key][0]) ** 2
                        + (row[y] - centers[key][1]) ** 2
                    )
                    min_dist_aoi = key
            if min_dist_aoi in aoi_counts:  # recalculate centers of AOI
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
            else:
                centers[min_dist_aoi][0] += row[x]
                centers[min_dist_aoi][1] += row[y]
                centers[min_dist_aoi][0] /= 2
                centers[min_dist_aoi][1] /= 2
                aoi_counts[min_dist_aoi] = 1

            for count in aoi_counts.values():  # calculate the entropy
                entropy -= (
                    count
                    / current_data.shape[0]
                    * np.log2(count / current_data.shape[0])
                )
            if entropy < min_entropy:  # TODO: use it to compare aoi defining methods
                min_entropy = entropy
                min_entropy_centers = centers
            aoi_list.append(min_dist_aoi)

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


def gradient_based(
    data: pd.DataFrame,
    x: str,
    y: str,
    W: int,
    threshold: float,
    pk: List[str] = None,
    aoi_name: str = "AOI",
) -> pd.DataFrame: ...
