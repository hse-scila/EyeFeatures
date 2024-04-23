from typing import List
import numpy as np
from math import sqrt
import pandas as pd
from numba import jit

from scipy.stats import gaussian_kde
from scipy.ndimage import maximum_filter

from eyetracking.utils import _split_dataframe


def _get_fixation_density(data: pd.DataFrame, x: str, y: str):
    df = data[[x, y]]
    assert df.shape[0] != 0, "Error: there are no points"
    kde = gaussian_kde(df.values.T)
    X, Y = np.mgrid[
        df[x].min() : df[x].max() : 100j, df[y].min() : df[y].max() : 100j
    ]  # is 100 enough?
    positions = np.vstack([X.ravel(), Y.ravel()])
    return np.reshape(kde(positions), X.shape), X, Y


def threshold_based(
    data: pd.DataFrame,
    x: str,
    y: str,
    W: int,
    threshold: float,
    pk: List[str] = None,
    aoi_name: str = "AOI",
) -> pd.DataFrame:
    assert data.shape[0] != 0, "Error: there are no points"
    data_splited = _split_dataframe(data, pk)
    min_entropy_centers = dict()
    min_entropy = np.inf
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
        aoi = dict()
        aoi_counts = dict()

        centers = dict()
        entropy = 0
        for i in range(loc_max_coord.shape[0]):
            centers[f"aoi_{i}"] = [X[loc_max_coord[i][0]][0], Y[loc_max_coord[i][1]][0]]
        for index, row in current_data.iterrows():
            min_dist = np.inf
            min_dist_aoi = None
            for i in range(loc_max_coord.shape[0]):
                if (
                    sqrt(
                        (row[x] - centers[f"aoi_{i}"][0]) ** 2
                        + (row[y] - centers[f"aoi_{i}"][1]) ** 2
                    )
                    < min_dist
                ):
                    min_dist = sqrt(
                        (row[x] - centers[f"aoi_{i}"][0]) ** 2
                        + (row[y] - centers[f"aoi_{i}"][1]) ** 2
                    )
                    min_dist_aoi = i
            if f"aoi_{min_dist_aoi}" in aoi:
                aoi[f"aoi_{min_dist_aoi}"].append([row[x], row[y]])

                centers[f"aoi_{min_dist_aoi}"][0] = centers[f"aoi_{min_dist_aoi}"][
                    0
                ] * (len(aoi[f"aoi_{min_dist_aoi}"]) - 1) / len(
                    aoi[f"aoi_{min_dist_aoi}"]
                ) + aoi[
                    f"aoi_{min_dist_aoi}"
                ][
                    -1
                ][
                    0
                ] / len(
                    aoi[f"aoi_{min_dist_aoi}"]
                )
                centers[f"aoi_{min_dist_aoi}"][1] = centers[f"aoi_{min_dist_aoi}"][
                    1
                ] * (len(aoi[f"aoi_{min_dist_aoi}"]) - 1) / len(
                    aoi[f"aoi_{min_dist_aoi}"]
                ) + aoi[
                    f"aoi_{min_dist_aoi}"
                ][
                    -1
                ][
                    1
                ] / len(
                    aoi[f"aoi_{min_dist_aoi}"]
                )
                aoi_counts[f"aoi_{min_dist_aoi}"] += 1
            else:
                aoi[f"aoi_{min_dist_aoi}"] = [[row[x], row[y]]]
                centers[f"aoi_{min_dist_aoi}"][0] += row[x]
                centers[f"aoi_{min_dist_aoi}"][1] += row[y]
                centers[f"aoi_{min_dist_aoi}"][0] /= 2
                centers[f"aoi_{min_dist_aoi}"][1] /= 2
                aoi_counts[f"aoi_{min_dist_aoi}"] = 1
            for count in aoi_counts.values():
                entropy -= (
                    count
                    / current_data.shape[0]
                    * np.log2(count / current_data.shape[0])
                )
            if entropy < min_entropy:
                min_entropy = entropy
                min_entropy_centers = centers

    result = None
    for i in range(len(data_splited)):
        aoi_counts = dict()
        group, current_data = data_splited[i]
        centers = min_entropy_centers.copy()
        aoi_list = []
        for index, row in current_data.iterrows():
            min_dist = np.inf
            min_dist_aoi = None
            for k in centers.keys():
                if (
                    sqrt((row[x] - centers[k][0]) ** 2 + (row[y] - centers[k][1]) ** 2)
                    < min_dist
                ):
                    min_dist = sqrt(
                        (row[x] - centers[k][0]) ** 2 + (row[y] - centers[k][1]) ** 2
                    )
                    min_dist_aoi = k
            aoi_list.append(min_dist_aoi)
            if min_dist_aoi in aoi_counts:
                aoi_counts[min_dist_aoi] += 1

                centers[min_dist_aoi][0] = (
                    centers[min_dist_aoi][0]
                    * (aoi_counts[min_dist_aoi] - 1)
                    / aoi_counts[min_dist_aoi]
                    + row[0] / aoi_counts[min_dist_aoi]
                )
                centers[min_dist_aoi][1] = (
                    centers[min_dist_aoi][1]
                    * (aoi_counts[min_dist_aoi] - 1)
                    / aoi_counts[min_dist_aoi]
                    + row[1] / aoi_counts[min_dist_aoi]
                )
            else:
                aoi_counts[min_dist_aoi] = 1
                centers[min_dist_aoi][0] += row[x]
                centers[min_dist_aoi][1] += row[y]
                centers[min_dist_aoi][0] /= 2
                centers[min_dist_aoi][1] /= 2
        if result is None:
            result = current_data.copy()
            result = pd.concat([result, pd.Series(aoi_list, name=aoi_name)], axis=1)
            pk_values = group.split("_")
            for i in range(len(pk)):
                result[pk[i]] = pk_values[i]
        else:
            to_res = current_data.copy()
            to_res.reset_index(drop=True, inplace=True)
            to_res = pd.concat([to_res, pd.Series(aoi_list, name=aoi_name)], axis=1)
            pk_values = group.split("_")
            for i in range(len(pk)):
                to_res[pk[i]] = pk_values[i]
            result = pd.concat([result, to_res], axis=0)
    return result


def gradient_based(
    data: pd.DataFrame,
    x: str,
    y: str,
    W: int,
    threshold: float,
    pk: List[str] = None,
    aoi_name: str = "AOI",
) -> pd.DataFrame:
    ...
