from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.manifold import MDS
from tqdm import tqdm

from eyetracking.utils import Types, _split_dataframe


def _target_norm(fwp: np.ndarray, fixations: np.ndarray) -> float:
    return np.linalg.norm(fixations - fwp, axis=1).sum()


def get_expected_path(
    data: pd.DataFrame,
    x: str,
    y: str,
    path_pk: List[str],
    pk: List[str],
    method: str = "mean",
    duration: str = None,
    return_df: bool = True,
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """
    Estimates expected path by a given method
    :param data: pd.Dataframe containing coordinates of fixations and its timestamps
    :param x: Column name of x-coordinate
    :param y: Column name of y-coordinate
    :param path_pk: List of column names of groups to calculate expected path (must be a subset of pk)
    :param pk: List of column names used to split pd.Dataframe
    :param method: method to calculate expected path ("mean" or "fwp")
    :param duration: Column name of fixations duration if needed
    :param return_df: Return pd.Dataframe object else np.ndarray
    :return: Dict of groups and Union[pd.Dataframe, np.ndarray] of form (x_est, y_est) or (x_est, y_est, duration_est)
    """

    if not set(path_pk).issubset(set(pk)):
        raise ValueError("path_pk must be a subset of pk")

    columns = [x, y]
    columns_ret = ["x_est", "y_est"]
    if duration is not None:
        columns.append(duration)
        columns_ret.append("duration_est")

    resulted_paths = {}
    path_groups: Types.Partition = _split_dataframe(data, path_pk)

    for group_nm, group_path in path_groups:
        expected_path = []
        data_part: Types.Partition = _split_dataframe(group_path, pk)
        path_length = max(len(part_path) for part_nm, part_path in data_part)

        if method == "mean":
            reshaped_paths = [
                np.pad(
                    part_path[columns],
                    pad_width=((0, path_length - len(part_path)), (0, 0)),
                    mode="constant",
                )
                for part_nm, part_path in data_part
            ]
            expected_path = np.mean(np.array(reshaped_paths), axis=0)
        elif method == "fwp":
            for step in range(path_length):
                duration_sum = 0
                observed_points = []
                for part_nm, part_path in data_part:
                    part_path = part_path[columns].values
                    if part_path.shape[0] > step:
                        observed_points.append(part_path[step, :2])
                        if len(columns) == 3:
                            duration_sum += part_path[step, 2]

                vector_points = np.array(observed_points)
                fwp_init = np.mean(vector_points, axis=0)
                fwp_opt = minimize(
                    _target_norm, fwp_init, args=(vector_points,), method="L-BFGS-B"
                )
                fix_opt = [fwp_opt.x[0], fwp_opt.x[1]]
                if len(columns) == 3:
                    fix_opt.append(duration_sum / max(1, len(observed_points)))

                expected_path.append(fix_opt)
        else:
            raise ValueError('Only "mean" and "fwp" methods are supported')

        expected_path_df = pd.DataFrame(expected_path, columns=columns_ret)
        resulted_paths[group_nm] = (
            expected_path_df if return_df else expected_path_df.values
        )

    return resulted_paths


def _get_fill_path(
    data: List[pd.DataFrame],
    x: str,
    y: str,
    method: str = "mean",
    duration: str = None,
) -> pd.DataFrame:
    """
    Calculates fill path as expected path of given expected paths
    :param data: paths data
    :param x: Column name of x-coordinate
    :param y: Column name of y-coordinate
    :param method: method to calculate expected path ("mean" or "fwp")
    :param duration: Column name of fixations duration if needed
    """

    paths = pd.concat(
        [path.assign(dummy=k) for k, path in enumerate(data)], ignore_index=True
    )

    fill_path = get_expected_path(
        data=paths,
        x=x,
        y=y,
        path_pk=["dummy"],
        pk=["dummy"],
        method=method,
        duration=duration,
    )
    return list(fill_path.values())[0]


# ======================== SIMILARITY MATRIX ========================
def restore_matrix(matrix: NDArray, tol=1e-9):
    """
    Estimates A assuming 'matrix' equals
    .. math: $A^T A$.
    """
    # Получаем собственные вектора и диагональную матрицу с.ч.
    evals, evecs = np.linalg.eigh(matrix)

    # Сортируем вектора и числа по убыванию модуля
    sorted_evals_indexes = np.argsort(evals)[::-1]
    # sorted_evals_indexes = np.argsort(np.abs(evals))[::-1]
    evecs = evecs[:, sorted_evals_indexes]
    evals = evals[sorted_evals_indexes]

    # Удалим вырожденные части из матриц
    A_rank = (evals > tol).sum()
    # A_rank = (np.abs(evals) > tol).sum()
    evals = np.diag(evals[:A_rank])
    evecs = evecs[:, :A_rank]

    # Посчитам матрицу A
    S = sqrtm(evals)
    U = np.identity(A_rank)

    US = U.dot(S)
    return US.dot(evecs.T).T, A_rank


def get_sim_matrix(scanpaths: List[NDArray], sim_metric: Callable) -> np.ndarray:
    """
    Computes similarity matrix given non-trivial metric.
    :param scanpaths: list of scanpaths, each being 2D-array of shape (2, n).
    :param sim_metric: similarity metric.
    :return: scaled similarity matrix.
    """
    n = len(scanpaths)
    sim_matrix = np.ones(shape=(n, n))
    for i in range(len(scanpaths)):
        for j in range(i + 1, len(scanpaths)):
            p, q = scanpaths[i], scanpaths[j]
            sim_matrix[i, j] = sim_metric(p, q)

    sim_matrix += sim_matrix.T
    m = np.max(sim_matrix)
    m = m if m != 0 else 1
    return sim_matrix / m


@jit(forceobj=True, looplift=True)
def get_dist_matrix(
    scanpaths: List[pd.DataFrame], dist_metric: Callable
) -> pd.DataFrame:
    """
    Computes pairwise distance matrix given distance metric.
    :param scanpaths: List of scanpaths DataFrames of form (x, y)
    :param dist_metric: Metric used to calculate distance from features.scanpath_dist
    """

    if len(scanpaths) == 0:
        raise ValueError("scanpaths list is empty")

    distances = []
    for i in tqdm(range(len(scanpaths))):
        for j in range(i + 1):
            distance = dist_metric(scanpaths[i], scanpaths[j])
            distances.append([i, j, distance])
            distances.append([j, i, distance])

    distances_df = pd.DataFrame(distances, columns=["p", "q", "dist"])
    return distances_df.reset_index().pivot_table(index="p", columns="q", values="dist")


def hierarchical_clustering_order(
    sim_matrix: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Reorder matrix using hierarchical clustering.
    :param sim_matrix: similarity matrix to reorder
    :param metric: metric used in building matrix
    :return: reordered matrix
    """

    Z = linkage(sim_matrix, method="ward", metric=metric)
    ordered_leaves = leaves_list(Z)
    reordered_matrix = sim_matrix[ordered_leaves, :][:, ordered_leaves]
    return reordered_matrix


def optimal_leaf_ordering_clustering(
    sim_matrix: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Reorder matrix using optimal leaf ordering.
    :param sim_matrix: similarity matrix to reorder
    :return: reordered matrix
    """

    Z = linkage(sim_matrix, method="ward", metric=metric)
    Z = optimal_leaf_ordering(Z, sim_matrix)
    ordered_leaves = leaves_list(Z)
    reordered_matrix = sim_matrix[ordered_leaves, :][:, ordered_leaves]
    return reordered_matrix


def dimensionality_reduction_order(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Reorder matrix using Multi-Dimensional Scaling (MDS).
    :param sim_matrix: similarity matrix to reorder
    :return: reordered matrix
    """

    mds = MDS(n_components=2, dissimilarity="precomputed")
    coords = mds.fit_transform(1 - sim_matrix)
    # reorder the matrix based on the first MDS component
    order = np.argsort(coords[:, 0])
    reordered_matrix = sim_matrix[order, :][:, order]
    return reordered_matrix


def spectral_order(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Reorder matrix using spectral reordering.
    :param sim_matrix: similarity matrix to reorder
    :return: reordered matrix
    """

    L = laplacian(sim_matrix, normed=True)

    _, eigenvectors = eigsh(L, k=2, which="SM")

    # reorder based on the second smallest eigenvector (fiedler vector)
    fiedler_vector = eigenvectors[:, 1]
    order = np.argsort(fiedler_vector)
    reordered_matrix = sim_matrix[order, :][:, order]
    return reordered_matrix
