from typing import List
import numpy as np
from math import sqrt
import pandas as pd
from numba import jit

from scipy.stats import gaussian_kde
from scipy.ndimage import maximum_filter

from eyetracking.features.measures import Entropy
from eyetracking.utils import _split_dataframe

def _get_fixation_density(data: pd.DataFrame, x: str, y: str):
    df = data[[x, y]]
    assert ( df.shape[0] != 0 ), "Error: there are no points"
    kde = gaussian_kde(df.values.T)
    X, Y = np.mgrid[df[x].min():df[x].max():100j, df[y].min():df[y].max():100j] # is 100 enough?
    positions = np.vstack([X.ravel(), Y.ravel()])
    return np.reshape(kde(positions), X.shape), X, Y

def threshold_based(data: pd.DataFrame, x: str, y: str, W: int, threshold: float, pk: List[str] = None):
    density, X, Y = _get_fixation_density(data, x, y)
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
    result = data.copy()
    result['AOI'] = None
    aoi = dict()
    centers = dict()
    for i in range(loc_max_coord.shape[0]):
        centers[f'aoi_{i}'] = [ X[loc_max_coord[i][0]][0], Y[loc_max_coord[i][1]][0] ]
    for index, row in result.iterrows():
        min_dist = np.inf
        min_dist_aoi = None
        for i in range(loc_max_coord.shape[0]):
            if sqrt((row[x] - centers[f'aoi_{i}'][0]) ** 2 + (row[y] - centers[f'aoi_{i}'][1]) ** 2) < min_dist:
                min_dist = sqrt((row[x] - centers[f'aoi_{i}'][0]) ** 2 + (row[y] - centers[f'aoi_{i}'][1]) ** 2)
                min_dist_aoi = i
        if f'aoi_{min_dist_aoi}' in aoi:
            aoi[f'aoi_{min_dist_aoi}'].append([row[x], row[y]])
            # centers[f'aoi_{min_dist_aoi}'][0] = X[loc_max_coord[min_dist_aoi][0]][0]
            # centers[f'aoi_{min_dist_aoi}'][1] = Y[loc_max_coord[min_dist_aoi][1]][0]
            # for elem in aoi[f'aoi_{min_dist_aoi}']:
            #     centers[f'aoi_{min_dist_aoi}'][0] += elem[0]
            #     centers[f'aoi_{min_dist_aoi}'][1] += elem[1]
            # centers[f'aoi_{min_dist_aoi}'][0] /= (len(aoi[f'aoi_{min_dist_aoi}']) + 1)
            # centers[f'aoi_{min_dist_aoi}'][1] /= (len(aoi[f'aoi_{min_dist_aoi}']) + 1)
            centers[f'aoi_{min_dist_aoi}'][0] = (centers[f'aoi_{min_dist_aoi}'][0] * (len(aoi[f'aoi_{min_dist_aoi}']) - 1) /
                                                 len(aoi[f'aoi_{min_dist_aoi}']) + aoi[f'aoi_{min_dist_aoi}'][-1][0] / len(aoi[f'aoi_{min_dist_aoi}']))
            centers[f'aoi_{min_dist_aoi}'][1] = (centers[f'aoi_{min_dist_aoi}'][1] * (len(aoi[f'aoi_{min_dist_aoi}']) - 1) /
                                                 len(aoi[f'aoi_{min_dist_aoi}']) + aoi[f'aoi_{min_dist_aoi}'][-1][1] / len(aoi[f'aoi_{min_dist_aoi}']))
            result.loc[index, 'AOI'] = f'aoi_{min_dist_aoi}'
        else:
            aoi[f'aoi_{min_dist_aoi}'] = [[row[x], row[y]]]
            centers[f'aoi_{min_dist_aoi}'][0] += row[x]
            centers[f'aoi_{min_dist_aoi}'][1] += row[y]
            centers[f'aoi_{min_dist_aoi}'][0] /= 2
            centers[f'aoi_{min_dist_aoi}'][1] /= 2
            result.loc[index, 'AOI'] = f'aoi_{min_dist_aoi}'

    return result

def gradient_based(data: pd.DataFrame, x: str, y: str, W: int, threshold: float):
    ...