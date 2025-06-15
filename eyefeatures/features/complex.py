from typing import Callable, List, Literal, Tuple, Union

import gudhi as gd
import numpy as np
import pandas as pd
from numba import jit, prange
from numpy.typing import NDArray
from PyEMD.EMD2d import EMD2D
from scipy.signal import convolve2d
from scipy.stats import gaussian_kde

from eyefeatures.utils import _rec2square, _split_dataframe


# =========================== HEATMAPS ===========================
def _check_shape(shape: Tuple[int, int]):
    assert isinstance(shape, tuple), f"'shape' must be tuple, hot {type(shape)}."
    assert len(shape) == 2, f"'shape' must be of length 2, got {len(shape)}."
    for k in shape:
        assert isinstance(
            k, int
        ), f"Values in 'shape' must be integers, got '{type(k)}'."
        assert k > 0, f"Integers in 'shape' must be positive, got '{k}'."


def get_heatmap(
    x: NDArray, y: NDArray, shape: Tuple[int, int], check: bool = True
) -> np.ndarray:
    """Get heatmap from scanpath (given coordinates are scaled and sorted in time) using
    Gaussian KDE.

    Args:
        x: X coordinate column name.
        y: Y coordinate column name.
        shape: if tuple with single integer, then square matrix is returned, otherwise k must be (height, width)
               tuple and rectangular matrix is returned.
        check: whether to check 'shape' for correct typing.

    Returns:
        heatmap matrix.
    """
    if check:
        _check_shape(shape)

    if len(x) <= 2:  # TODO warning
        # in case of small number of samples, KDE cannot be applied and
        # default kernel estimate is returned instead
        x, y = np.array([0.25, 0.50, 0.75]), np.array([0.50, 0.50, 0.50])

    scanpath = np.vstack([x, y])
    kernel = gaussian_kde(scanpath)
    interval_x, interval_y = np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0])
    x, y = np.meshgrid(interval_x, interval_y)

    positions = np.vstack([y.ravel(), x.ravel()])
    return np.reshape(kernel(positions), x.shape)


def get_heatmaps(
    data: pd.DataFrame, x: str, y: str, shape: Tuple[int, int], pk: List[str] = None
) -> np.ndarray:
    """Get heatmaps from scanpaths (given coordinates are scaled and sorted in time) using
    Gaussian KDE.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        shape: if tuple with single integer, then square matrix is returned, otherwise k must be (height, width)
               tuple and rectangular matrix is returned.
        pk: List of columns being primary key.

    Returns:
        heatmap matrices.
    """
    _check_shape(shape)

    if pk is None:
        x_path, y_path = data[x].values, data[y].values
        heatmap = get_heatmap(x_path, y_path, shape)
        heatmaps = heatmap[np.newaxis, :, :]
    else:
        groups: List[str, pd.DataFrame] = _split_dataframe(data, pk)
        hshape = (len(groups), shape[0], shape[1])

        heatmaps = np.zeros(hshape)
        for i, (_, group_X) in enumerate(groups):
            x_path, y_path = group_X[x], group_X[y]
            heatmaps[i, :, :] = get_heatmap(x_path, y_path, shape, check=False)

    return heatmaps


# =========================== PCA ===========================
def pca(matrix: NDArray, p: int, cum_sum: float = None) -> np.ndarray:
    """PCA compression.

    Args:
        matrix: matrix to get principal components from (n x m)
        p: number of first principal components to leave
        cum_sum: instead of p, leave such number of principal components, that 0.0 <= a_cum_sum <= 1.0
                    fraction of information is conserved.

    Returns:
        matrix of eigenvectors (m x m), projection (n x p), rows means (n x 1)
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


# =========================== RQA ===========================
def get_rqa(
    data: pd.DataFrame, x: str, y: str, metric: Callable, rho: float
) -> np.ndarray:
    """Calculates recurrence quantification analysis matrix based on given fixations.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        metric: callable metric on :math:`\mathbb{R}^2` points.
        rho: threshold radius.

    Returns:
        rqa matrix.
    """
    fixations = data[[x, y]].values
    n = len(fixations)
    rqa_matrix = np.zeros((n, n), dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            dist = metric(fixations[i], fixations[j])
            rqa_matrix[i][j] = int(dist <= rho)
            rqa_matrix[j][i] = int(dist <= rho)

    return rqa_matrix


# =========================== MTF ===========================
def get_mtf(
    data: pd.DataFrame,
    x: str,
    y: str,
    n_bins: int = 10,
    output_size: Union[int, float] = 1.0,
    shrink_strategy: Literal["max", "mean", "normal"] = "normal",
    flatten: bool = False,
) -> np.ndarray:
    """Calculates Markov Transition Field for (x,y) coordinates.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        n_bins: number of bins to discretize time series into.
        output_size: fraction between 0 and 1. Specifies fraction of input series length to shrink output to.
        shrink_strategy: strategy to use for convolution while shrinking. Ignored if 'output_size' is equal to
                         size of 'data'.
        flatten: bool, whether to flatten the array.

    Returns:
        tensor of shape (2, n_coords, n_coords), where n_coords is the length of input dataframe.
    """
    if isinstance(output_size, float):
        assert 0.0 < output_size <= 1.0, "Must be 0 < output_size <= 1."
        output_size = np.ceil(output_size * len(data)).astype(np.int64)

    elif isinstance(output_size, int):
        assert (
            0 < output_size <= len(data)
        ), "'output_size' must be positive integer not exceeding the input size."

    else:
        raise ValueError(f"'output_size' is of wrong type '{type(output_size)}'.")

    assert n_bins > 1, "'n_bins' must be greater than 1."
    assert len(data) > n_bins, "Input series must contain more than 'n_bins' samples."

    x_coords, y_coords = data[[x]].values.ravel(), data[[y]].values.ravel()
    fixations_mtf = _get_mtf(np.array([x_coords, y_coords]), n_bins=n_bins)

    n_samples, n_timestamps = 2, len(x_coords)
    if output_size < n_timestamps:
        shrunk_mtfs = []
        for i in range(n_samples):
            shrunk_mtfs.append(
                _shrink_matrix(
                    fixations_mtf[i, :, :],
                    height=output_size,
                    width=output_size,
                    strategy=shrink_strategy,
                )
            )

        del fixations_mtf
        fixations_mtf = np.array(shrunk_mtfs)

    return fixations_mtf.flatten() if flatten else fixations_mtf


def _get_mtf(a: np.ndarray, n_bins: int) -> np.ndarray:
    assert len(a.shape) == 2, "Array of shape (n_samples, n_timestamps) must be passed."
    n_samples, n_timestamps = a.shape

    a_binned = np.zeros(a.shape, dtype=np.int64)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # evenly spaced quantiles
    bins = np.quantile(a, q=quantiles, axis=1).T  # bins for each sample
    for i in range(n_samples):
        a_binned[i, :] = np.searchsorted(
            bins[i, :], a[i, :], side="left"
        )  # squeeze coordinates

    mtm = np.zeros((n_samples, n_bins, n_bins))  # build Markov Transition Matrix
    for i in range(n_samples):
        for j in range(n_timestamps - 1):
            mtm[i, a_binned[i, j], a_binned[i, j + 1]] += 1

    mtm_sum = mtm.sum(axis=2)  # normalize rows sums in each sample to 1
    np.place(mtm_sum, mtm_sum == 0, 1)
    mtm /= mtm_sum[:, :, None]

    mtf = np.zeros(
        (n_samples, n_timestamps, n_timestamps)
    )  # build Markov Transition Field
    for i in range(n_samples):
        for j in range(n_timestamps):
            for k in range(n_timestamps):
                mtf[i, j, k] = mtm[i, a_binned[i, j], a_binned[i, k]]

    return mtf


def _shrink_matrix(
    mat: np.array,
    height: int,
    width: int,
    strategy: Literal["max", "mean", "normal"] = "normal",
) -> np.ndarray:
    """Shrinks matrix to be close to output_size x output_size.

    Args:
        mat: 2d matrix (image channel).
        height: height of shrunk matrix.
        width: width of shrunk matrix.
        strategy: strategy to use while shrinking.
         * 'mean' - 2d convolution with uniform kernel.
         * 'normal' - 2d convolution with Gauss kernel.
         * 'max' - max pooling, resulting image is the closest possible to provided 'size'.

    Returns:
        shrunk matrix.
    """
    ih, iw = mat.shape
    oh, ow = height, width

    assert len(mat.shape) == 2
    assert oh < ih and ow < iw

    if strategy in ("mean", "normal"):  # convolution (uniform/normal)
        dh, dw = ih - oh + 1, iw - ow + 1
        if strategy == "mean":
            kernel = np.ones((dh, dw)) / (dh * dw)
        else:  # 'normal'
            m = max(dh, dw)
            kernel = _gaussian_kernel(m, sigma=5)
            kernel = _rec2square(kernel)

        shrunk_mat = convolve2d(mat, kernel, mode="valid")

    elif strategy == "max":  # max pooling
        dh, dw = int(np.ceil(ih / oh)), int(np.ceil(iw / ow))
        rh, rw = ih // dh, iw // dw

        if ih % dh == 0 and iw % dw == 0:  # filter divides matrix on equal blocks
            shrunk_mat = mat.reshape(rh, dh, rw, dw).max(axis=(1, 3))
        else:
            Q1 = mat[: rh * dh, : rw * dw].reshape(rh, dh, rw, dw).max(axis=(1, 3))
            Q2 = mat[rh * dh :, : rw * dw].reshape(-1, rw, dw).max(axis=2)
            Q3 = mat[: rh * dh, rw * dw :].reshape(rh, dh, -1).max(axis=1)
            Q4 = mat[rh * dh :, rw * dw :].max()
            shrunk_mat = np.vstack(np.c_[Q1, Q3], np.c_[Q2, Q4])

    else:
        raise ValueError(f"'shrink_strategy'={strategy} is not supported.")

    return shrunk_mat


def _gaussian_kernel(size, sigma) -> np.ndarray:
    if size % 2 == 0:
        x = (np.arange(-size / 2, size / 2, 1) + 0.5).reshape(1, -1)
    else:
        x = np.arange(-size // 2 + 1, size // 2 + 1, 1).reshape(1, -1)

    xx = np.tile(x, (size, 1))
    yy = np.rot90(xx, 1)

    kernel = (1 / 2 * np.pi * np.square(sigma)) * np.exp(
        -(np.square(xx) + np.square(yy)) / (2 * np.square(sigma))
    )
    kernel = kernel / np.sum(kernel)
    return kernel


# =========================== GASF/GADF ===========================
def get_gaf(
    data: pd.DataFrame,
    x: str,
    y: str,
    t: str = None,
    field_type: Literal["difference", "sum"] = "difference",
    to_polar: Literal["regular", "cosine"] = "cosine",
    flatten: bool = False,
) -> np.ndarray:
    """Calculates Gramian Angular Field for (x,y) coordinates.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        t: timestamps column name.
        field_type: which type of field to calculate. If "difference", then GADF is returned,
                    otherwise ("sum") GASF is returned.
        to_polar: conversion from cartesian to polar coordinates.
                 * 'regular': standard conversion calculating arctan(y/x).
                 * 'cosine': angle is calculated as cosine of series data, radius is taken as timestamps.
        flatten: bool, whether to flatten the array.

    Returns:
        tensor of shape (2, n_coords, n_coords), where n_coords is the length of input dataframe.
    """
    assert field_type in (
        "difference",
        "sum",
    ), f"'field_type'={field_type} is not supported."
    assert to_polar in ("regular", "cosine"), f"'to_polar'={to_polar} is not supported."
    x_coords, y_coords = data[[x]].values.ravel(), data[[y]].values.ravel()
    timestamps = np.arange(len(x_coords)) if t is None else data[[t]].values.ravel()
    gaf = _get_gaf(
        np.array([x_coords, y_coords]),
        timestamps,
        field_type=field_type,
        to_polar=to_polar,
    )
    return gaf.flatten() if flatten else gaf


def _get_gaf(
    a: np.array,
    t: np.array,
    field_type: Literal["difference", "sum"],
    to_polar: Literal["regular", "cosine"],
) -> np.ndarray:
    """
    Args:
        a: array of shape (n_samples, n_timestamps) being angular values.
        t: array of shape (n_timestamps,) being timestamps for all samples in 'a' or
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
    if field_type == "sum":

        def f(phi1, phi2):
            return np.cos(phi1 + phi2)

    else:  # 'difference'

        def f(phi1, phi2):
            return np.sin(phi1 - phi2)

    gaf = np.zeros((n_samples, n_timestamps, n_timestamps))
    for i in range(n_samples):
        if to_polar == "regular":
            rho, phi = _car2pol(
                a[i, :], _get_t(t, i)
            )  # _get_t used to avoid copies of 't' in 1d case
        else:
            rho, phi = _encode_car(a[i, :], _get_t(t, i))

        for j in range(n_timestamps):
            for k in range(n_timestamps):
                gaf[i, j, k] = f(phi[j], phi[k])

    return gaf


def _rescale(a: np.array) -> np.ndarray:
    """Linearly map array to [-1, 1]."""
    amm = _minmax(a)
    return amm * 2 - 1


def _minmax(a: np.array) -> np.ndarray:
    min_, max_ = a.min(), a.max()
    if min_ == max_:
        return a if min_ == 0.0 else np.ones(a.shape, dtype=a.dtype)
    return (a - min_) / (max_ - min_)


def _car2pol(x: np.array, f_x: np.array) -> Tuple[np.ndarray, np.ndarray]:
    rho = np.sqrt(x**2 + f_x**2)
    phi = np.arctan2(f_x, x)
    return rho, phi


def _encode_car(x: np.array, t: np.array) -> Tuple[np.ndarray, np.ndarray]:
    rho = t
    phi = np.arccos(_rescale(x))
    return rho, phi


# =========================== HILBERT CURVE ===========================
def get_hilbert_curve_enc(
    data: pd.DataFrame, x: str, y: str, scale: bool = True, p: int = 4
) -> np.ndarray:
    """Map scanpath to values on Hilbert curve and encode to single feature vector.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        scale: whether to scale the scanpath to [0, 1] before mapping to Hilbert curve. If false, then
        p: order of Hilbert curve, unit square is divided into (2^p)x(2^p) smaller squares.
              Higher value of p indicates better locality preservation.

    Returns:
        scanpath encoding in 2^p-dimensional feature space using 1D Hilbert curve.
    """
    n = 2**p
    mapping = get_hilbert_curve(
        data=data, x=x, y=y, scale=scale, p=p
    )  # get Hilbert curve mapping
    mapping = np.unique(mapping)  # leave only unique values
    vec = np.zeros((n * n,))
    for i in range(n * n):
        vec[i] = i in mapping  # activate mapped values
    return vec


def get_hilbert_curve(
    data: pd.DataFrame, x: str, y: str, scale: bool = True, p: int = 4
) -> np.ndarray:
    """Map scanpath to points on 1D Hilbert curve.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        scale: whether to scale the scanpath to [0, 1] before mapping to Hilbert curve. If false, then
        p: order of Hilbert curve, unit square is divided into (2^p)x(2^p) smaller squares.
              Higher value of p indicates better locality preservation.

    Returns:
        scanpath mapping to 1D Hilbert curve.
    """
    x, y = data[x].values, data[y].values
    n_fixations = len(x)

    if scale:
        x, y = _minmax(x), _minmax(
            y
        )  # map x, y to [0, 1]  TODO: better approach than minmax?
    else:
        assert (
            0 <= x <= 1
        ), "Either scale 'x' to be between 0 and 1 or add 'scale'=True."
        assert (
            0 <= y <= 1
        ), "Either scale 'y' to be between 0 and 1 or add 'scale'=True."
    x, y = x * (2**p), y * (2**p)  # map x, y to [0, 2^p]
    x, y = np.array(np.round(x), dtype=int), np.array(
        np.round(y), dtype=int
    )  # map [0, 2^p] to {0, 1, .., 2^p}

    h = np.zeros((n_fixations,))
    for i in range(n_fixations):
        h[i] = xy2h(x[i], y[i], p=p)
    return h


def xy2h(x: int, y: int, p: int) -> int:
    """Mapping of 2D space to 1D using Hilbert curve.

    Args:
        x: x-coordinate of a point, 0 <= x < 2^p.
        y: y-coordinate of a point, 0 <= y < 2^p.
        p: order of Hilbert curve, unit square is divided into (2^p)x(2^p) smaller squares.
              Higher value of p indicates better locality preservation.

    Returns:
        corresponding point on 1D Hilbert curve.

    Notes:
        Algorithm: https://people.math.sc.edu/Burkardt/py_src/hilbert_curve/hilbert_curve.py.
    """
    n = 2**p

    s = n // 2
    d = 0

    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)  # adding length on Hilbert curve
        if ry == 0:  # rotation of quadrant
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s = s // 2
    return d


def hilbert_huang_transform(data: np.ndarray, max_imf: int = 1) -> np.ndarray:
    """Perform Hilbert-Huang transform on a given data sequence.

    Args:
        data: input sequence of form (x, y) coordinates
        max_imf: maximum number of intrinsic mode functions to extract

    Returns:
        intrinsic mode functions from an input data vector
    """

    emd = EMD2D()
    decomposed = emd(data, max_imf=max_imf)
    return decomposed


# =========================== PERSISTENCE CURVE ===========================
def vietoris_rips_filtration(
    scanpath: np.ndarray, max_dim: int = 2, max_radius: float = 1.0
):
    """Compute the Vietoris-Rips filtration for a point cloud.

    Args:
        scanpath: scanpath data as a numpy array of shape (n, 2).
        max_dim: maximum dimension for persistent homology.
        max_radius: maximum radius for the filtration.

    Returns:
        persistence diagram and simplex tree.
    """

    rips_complex = gd.RipsComplex(points=scanpath, max_edge_length=max_radius)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    persistence = simplex_tree.persistence()

    return persistence, simplex_tree


def lower_star_filtration(time_series: np.ndarray, persistence_dim_max: bool = False):
    """Compute the Lower Star filtration for a time series.

    Args:
        time_series: time series data.
        persistence_dim_max: If true, the persistent homology for the maximal dimension in the
            complex is computed. If False, it is ignored. Default is False.
    """

    simplex_tree = gd.SimplexTree(persistence_dim_max=persistence_dim_max)

    for i, val in enumerate(time_series):
        simplex_tree.insert([i], filtration=val)

    simplex_tree.make_filtration_non_decreasing()
    persistence = simplex_tree.persistence()

    return persistence, simplex_tree


def persistence_curve(persistence_diagram: List[Tuple | np.ndarray], t: float):
    """Compute the persistence curve for a persistence diagram at time t.

    Args:
        persistence_diagram: persistence diagram [(birth, death), ...].
        t: threshold time for persistence curve.

    Returns:
        sum of persistence intervals active at time t.
    """
    total_persistence = sum(
        (death - birth) for birth, death in persistence_diagram if birth <= t <= death
    )
    return total_persistence


def persistence_entropy_curve(persistence_diagram: List[Tuple | np.ndarray], t: float):
    """Compute the persistence entropy curve for a persistence diagram at time t.

    Args:
        persistence_diagram: persistence diagram [(birth, death), ...].
        t: threshold time for persistence entropy curve.

    Returns:
        entropy value at time t.
    """

    intervals = [
        (death - birth) for birth, death in persistence_diagram if birth <= t <= death
    ]
    total_persistence = sum(intervals)

    if total_persistence == 0:
        return 0

    probabilities = [interval / total_persistence for interval in intervals]
    entropy = -sum(p * np.log(p) for p in probabilities)

    return entropy


def calculate_topological_features(
    scanpath: np.ndarray,
    time_series: np.ndarray,
    max_radius: float = 1.0,
    max_time: float = 1.0,
    max_dim: int = 2,
    time_steps: int = 100,
):
    """Calculate topological features (persistence curve and persistence entropy) for a scanpath.

    Args:
        scanpath: scanpath data array of shape (n, 2).
        time_series: 1d array of time series data (e.g. x or y coordinates over time).
        max_radius: maximum radius for Vietoris-Rips filtration.
        max_time: maximum threshold for the persistence curve.
        max_dim: maximum dimension for persistent homology.
        time_steps: number of time steps for evaluating the persistence curve and entropy.

    Returns:
        values of persistence curve at each time step, Values of persistence entropy at each time step.
    """

    persistence_vr, _ = vietoris_rips_filtration(
        scanpath, max_dim=max_dim, max_radius=max_radius
    )
    persistence_ls, _ = lower_star_filtration(time_series, max_dim=max_dim)

    time_grid = np.linspace(0, max_time, time_steps)
    persistence_curve_vals = []
    persistence_entropy_vals = []

    for t in time_grid:
        pc_vr = persistence_curve([p[1] for p in persistence_vr], t)
        pe_vr = persistence_entropy_curve([p[1] for p in persistence_vr], t)

        pc_ls = persistence_curve([p[1] for p in persistence_ls], t)
        pe_ls = persistence_entropy_curve([p[1] for p in persistence_ls], t)

        persistence_curve_vals.append(pc_vr + pc_ls)
        persistence_entropy_vals.append(pe_vr + pe_ls)

    return np.array(persistence_curve_vals), np.array(persistence_entropy_vals)
