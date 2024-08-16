from typing import Callable, List

import numpy as np
import pandas as pd
from numba import jit, prange
from numpy.typing import NDArray
from typing import Union, Literal, Tuple
from scipy.stats import gaussian_kde
from scipy.signal import convolve2d


def get_heatmap(x: NDArray, y: NDArray, k: int):
    """
    Get heatmap from scanpath (given coordinates are scaled and sorted in time) using
    Gaussian KDE.
    :param x: x coordinates of scanpath
    :param y: y coordinates of scanpath
    :param k: size of heatmap
    :return: heatmap matrix
    """
    assert k > 0, "'k' must be positive"

    if len(x) <= 2:
        x, y = np.array([0.2, 0.55, 0.4, 0.25]), np.array([0.25, 0.5, 0.6, 0.7])

    scanpath = np.vstack([x, y])
    kernel = gaussian_kde(scanpath)
    interval = np.linspace(0, 1, k)
    x, y = np.meshgrid(interval, interval)

    positions = np.vstack([y.ravel(), x.ravel()])
    return np.reshape(kernel(positions), x.shape)


def get_heatmaps(data: pd.DataFrame, x: str, y: str, k: int, pk: List[str] = None):
    if pk is None:
        x_path, y_path = data[x].values, data[y].values
        heatmap = get_heatmap(x_path, y_path, k)
        heatmaps = heatmap[np.newaxis, :, :]
    else:
        groups = data[pk].drop_duplicates().values
        heatmaps = np.zeros((len(groups), k, k))

        for i, group in enumerate(groups):
            cur_X = data[pd.DataFrame(data[pk] == group).all(axis=1)]
            x_path, y_path = cur_X[x], cur_X[y]
            heatmaps[i, :, :] = get_heatmap(x_path, y_path, k)

    return heatmaps


def pca(matrix: NDArray, p: int, cum_sum: float = None):
    """
    PCA compression.
    :param matrix: matrix to get principal components from (n x m)
    :param p: number of first principal components to leave
    :param cum_sum: instead of p, leave such number of principal components, that 0.0 <= a_cum_sum <= 1.0
                    fraction of information is conserved.
    :return: matrix of eigenvectors (m x m), projection (n x p), rows means (n x 1)
    """
    assert len(matrix.shape) == 2, "'matrix' should be a matrix"
    assert 0 <= p <= matrix.shape[0], "given 'matrix' is n x m, 0 <= p <= n must hold"
    assert (p is not None) or (
            cum_sum is not None
    ), "either 'p' or 'cum_sum' must be provided"
    assert (cum_sum is None) or (
            0.0 <= cum_sum <= 1.0
    ), "'cum_sum' must be between 0.0 and 1.0"

    matrix = matrix.astype(np.float64)

    row_means = np.mean(matrix, axis=1)
    matrix -= row_means[:, None]
    c = np.cov(matrix)
    evals, evecs = np.linalg.eigh(c)
    sorted_evals_indexes = np.argsort(evals)[::-1]
    evecs = evecs[:, sorted_evals_indexes]

    if p is not None:
        evecs = evecs[:, :p]
    else:
        evals /= evals.sum()
        cumsum = 0.0
        p = 0
        while cumsum < cum_sum and p < evals.size():
            cumsum += evals[p]
            p += 1

        evecs = evecs[:, :p]

    projection = evecs.T @ matrix
    return evecs, projection, row_means


@jit(forceobj=True, looplift=True)
def get_rqa(
        data: pd.DataFrame, x: str, y: str, metric: Callable, rho: float
) -> np.ndarray:
    """
    Calculates recurrence quantification analysis matrix based on given fixations.
    :param data: dataframe containing fixations
    :param x: column name of x-coordinate
    :param y: column name of y-coordinate
    :param metric: callable metric on R^2 points
    :param rho: threshold radius

    :return: rqa matrix
    """
    n = data.size
    fixations = data[[x, y]].values
    rqa_matrix = np.zeros((n, n), dtype=np.int32)

    for i in prange(n):
        for j in prange(i + 1, n):
            dist = metric(fixations[i], fixations[j])
            rqa_matrix[i][j] = int(dist <= rho)
            rqa_matrix[j][i] = int(dist <= rho)

    return rqa_matrix


def get_mtf(
        data: pd.DataFrame, x: str, y: str,
        n_bins: int = 20,
        output_size: Union[int, float] = 1.0,
        shrink_strategy: Literal["max", "mean", "normal"] = "normal",
        flatten: bool = False
) -> np.array:
    """
    Calculates Markov Transition Field for (x,y) coordinates.
    :param data: dataframe containing fixation coordinates.
    :param x: x-coordinate column name.
    :param y: y-coordinate column name.
    :param n_bins: number of bins to discretize time series into.
    :param output_size: fraction between 0 and 1. Specifies fraction of input series length to shrink output to.
    :param shrink_strategy: strategy to use for convolution while shrinking. Ignored if 'output_size' is equal to
                            size of 'data'.
    :param flatten: bool, whether to flatten the array.

    :returns: tensor of shape (2, n_coords, n_coords), where n_coords is the length of input dataframe.
    """
    if isinstance(output_size, float):
        assert 0.0 < output_size <= 1.0, "Must be 0 < output_size <= 1."
        output_size = np.ceil(output_size * len(data)).astype(np.int64)

    elif isinstance(output_size, int):
        assert 0 < output_size <= len(data), "'output_size' must be positive integer not exceeding the input size."

    else:
        raise ValueError(f"'output_size' is of wrong type '{type(output_size)}'.")

    assert n_bins > 1, "'n_bins' must be greater than 1."
    assert len(data) > n_bins, "Input series must contain more than 'n_bins' samples."

    x_coords, y_coords = data[[x]].values.ravel(), data[[y]].values.ravel()
    fixations_mtf = _get_mtf(np.array([x_coords, y_coords]), n_bins=n_bins)

    n_samples, n_timestamps = 2, len(x_coords)
    if output_size < n_timestamps:
        shrunk_mtfs = []
        for i in prange(n_samples):
            shrunk_mtfs.append(_shrink_matrix(fixations_mtf[i, :, :],
                                              height=output_size,
                                              width=output_size,
                                              strategy=shrink_strategy))

        del fixations_mtf
        fixations_mtf = np.array(shrunk_mtfs)

    return fixations_mtf.flatten() if flatten else fixations_mtf


@jit(forceobj=True, looplift=True)
def _get_mtf(a: np.array, n_bins: int) -> np.array:
    assert len(a.shape) == 2, "Array of shape (n_samples, n_timestamps) must be passed."
    n_samples, n_timestamps = a.shape

    a_binned = np.zeros(a.shape, dtype=np.int64)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]                         # evenly spaced quantiles
    bins = np.quantile(a, q=quantiles, axis=1).T                            # bins for each sample
    for i in prange(n_samples):
        a_binned[i, :] = np.searchsorted(bins[i, :], a[i, :], side='left')  # squeeze coordinates

    mtm = np.zeros((n_samples, n_bins, n_bins))                             # build Markov Transition Matrix
    for i in prange(n_samples):
        for j in prange(n_timestamps - 1):
            mtm[i, a_binned[i, j], a_binned[i, j + 1]] += 1

    mtm_sum = mtm.sum(axis=2)                                               # normalize rows sums in each sample to 1
    np.place(mtm_sum, mtm_sum == 0, 1)
    mtm /= mtm_sum[:, :, None]

    mtf = np.zeros((n_samples, n_timestamps, n_timestamps))                 # build Markov Transition Field
    for i in prange(n_samples):
        for j in prange(n_timestamps):
            for k in prange(n_timestamps):
                mtf[i, j, k] = mtm[i, a_binned[i, j], a_binned[i, k]]

    return mtf


def _shrink_matrix(mat: np.array, height: int, width: int,
                   strategy: Literal["max", "mean", "normal"] = "normal"
                   ) -> np.array:
    """
    Shrinks matrix to be output_size x output_size.
    :param img: 2d matrix (image channel).
    :param height: height of shrunk matrix.
    :param width: width of shrunk matrix.
    :param strategy: strategy to use while shrinking.
            * 'mean' - 2d convolution with uniform kernel.
            * 'normal' - 2d convolution with Gauss kernel.
            * 'max' - max pooling, resulting image is the closest possible to provided 'size'.
    """
    ih, iw = mat.shape
    oh, ow = height, width

    assert len(mat.shape) == 2
    assert oh < ih and ow < iw

    if strategy in ('mean', 'normal'):  # convolution (uniform/normal)
        dh, dw = ih - oh + 1, iw - ow + 1
        if strategy == 'mean':
            kernel = np.ones((dh, dw)) / (dh * dw)
        else:  # 'normal'
            m = max(dh, dw)
            kernel = _gaussian_kernel(m, sigma=5)
            if dh > dw:
                d = dw % 2
                kernel = kernel[dh // 2 - dw // 2:dh // 2 + dw // 2 + d, :dw]
            else:
                d = dh % 2
                kernel = kernel[:dh, dw // 2 - dh //2:dw // 2 + dh // 2 + d]

        shrunk_mat = convolve2d(mat, kernel, mode='valid')

    elif strategy == 'max':  # max pooling
        dh, dw = int(np.ceil(ih / oh)), int(np.ceil(iw / ow))
        rh, rw = ih // dh, iw // dw

        if ih % dh == 0 and iw % dw == 0:  # filter divides matrix on equal blocks
            shrunk_mat = mat.reshape(rh, dh, rw, dw).max(axis=(1, 3))
        else:
            Q1 = mat[:rh * dh, :rw * dw].reshape(rh, dh, rw, dw).max(axis=(1, 3))
            Q2 = mat[rh * dh:, :rw * dw].reshape(-1, rw, dw).max(axis=2)
            Q3 = mat[:rh * dh, rw * dw:].reshape(rh, dh, -1).max(axis=1)
            Q4 = mat[rh * dh:, rw * dw:].max()
            shrunk_mat = np.vstack(np.c_[Q1, Q3], np.c_[Q2, Q4])

    else:
        raise ValueError(f"'shrink_strategy'={strategy} is not supported.")

    return shrunk_mat


def _gaussian_kernel(size, sigma) -> np.array:
    if size % 2 == 0:
        x = (np.arange(-size / 2, size / 2, 1) + 0.5).reshape(1, -1)
    else:
        x = np.arange(-size // 2 + 1, size // 2 + 1, 1).reshape(1, -1)

    xx = np.tile(x, (size, 1))
    yy = np.rot90(xx, 1)

    kernel = (1 / 2 * np.pi * np.square(sigma)) * np.exp(- (np.square(xx) + np.square(yy)) / (2 * np.square(sigma)))
    kernel = kernel / np.sum(kernel)
    return kernel


def get_gaf(
        data: pd.DataFrame, x: str, y: str, t: str = None,
        field_type: Literal["difference", "sum"] = "difference",
        to_polar: Literal["regular", "cosine"] = "cosine",
        flatten: bool = False
) -> np.array:
    """
    Calculates Gramian Angular Field for (x,y) coordinates.
    :param data: dataframe containing fixation coordinates.
    :param x: x-coordinate column name.
    :param y: y-coordinate column name.
    :param t: timestamps column name.
    :param field_type: which type of field to calculate. If "difference", then GADF is returned,
                       otherwise ("sum") GASF is returned.
    :param to_polar: conversion from cartesian to polar coordinates.
                    * 'regular': standard conversion calculating arctan(y/x).
                    * 'cosine': angle is calculated as cosine of series data, radius is taken as timestamps.
    :param flatten: bool, whether to flatten the array.

    :returns: tensor of shape (2, n_coords, n_coords), where n_coords is the length of input dataframe.
    """
    assert field_type in ('difference', 'sum'), f"'field_type'={field_type} is not supported."
    assert to_polar in ('regular', 'cosine'), f"'to_polar'={to_polar} is not supported."
    x_coords, y_coords = data[[x]].values.ravel(), data[[y]].values.ravel()
    timestamps = np.arange(len(x_coords)) if t is None else data[[t]].values.ravel()
    gaf = _get_gaf(np.array([x_coords, y_coords]), timestamps,
                   field_type=field_type, to_polar=to_polar)
    return gaf.flatten() if flatten else gaf


def _get_gaf(a: np.array, t: np.array,
             field_type: Literal["difference", "sum"],
             to_polar: Literal["regular", "cosine"]
             ) -> np.array:
    """
    :param a: array of shape (n_samples, n_timestamps) being angular values.
    :param t: array of shape (n_timestamps,) being timestamps for all samples in 'a' or
                             (n_samples, n_timestamps) being timestamps of corresponding angular values in 'a',
                                                       i.e. a[i, :] are expected to be the angles at time t[i, :].
    """
    def _get_t(t_: np.array, i_: int):
        return t_[i_, :] if len(t_.shape) > 1 else t_

    assert len(a.shape) == 2, "Array of shape (n_samples, n_timestamps) must be passed."
    if len(t.shape) == 2:
        assert t.shape == a.shape, "'a' and 't' must be of same shape."
    else:
        assert len(t.shape) == 1 and len(t) == a.shape[1]

    n_samples, n_timestamps = a.shape
    if field_type == 'sum':
        def f(phi1, phi2):
            return np.cos(phi1 + phi2)
    else:          # 'difference'
        def f(phi1, phi2):
            return np.sin(phi1 - phi2)

    gaf = np.zeros((n_samples, n_timestamps, n_timestamps))
    for i in prange(n_samples):
        if to_polar == 'regular':
            rho, phi = _car2pol(a[i, :], _get_t(t, i))         # _get_t used to avoid copies of 't' in 1d case
        else:
            rho, phi = _encode_car(a[i, :], _get_t(t, i))

        for j in prange(n_timestamps):
            for k in prange(n_timestamps):
                gaf[i, j, k] = f(phi[j], phi[k])

    return gaf


def _rescale(a: np.array) -> np.array:
    """
    Linearly map array to [-1, 1].
    """
    amm = _minmax(a)
    return amm * 2 - 1


def _minmax(a: np.array) -> np.array:
    min_, max_ = a.min(), a.max()
    if min_ == max_:
        return a if min_ == 0.0 else np.ones(a.shape)
    return (a - min_) / (max_ - min_)


def _car2pol(x, f_x):
    rho = np.sqrt(x ** 2 + f_x ** 2)
    phi = np.arctan2(f_x, x)
    return rho, phi


def _encode_car(x: np.array, t: np.array) -> Tuple[np.array, np.array]:
    rho = t
    phi = np.cos(_rescale(x))
    return rho, phi
