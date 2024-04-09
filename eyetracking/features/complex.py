import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde


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
