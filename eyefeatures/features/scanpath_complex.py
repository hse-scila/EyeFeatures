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

from eyefeatures.utils import Types, _split_dataframe


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
    """Estimates expected path by a given method.

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        path_pk: List of column names of groups to calculate expected path (must be a subset of pk).
        pk: List of column names used to split pd.Dataframe.
        method: method to calculate expected path ("mean" or "fwp").
        duration: Column name of fixations duration if needed.
        return_df: Return pd.Dataframe object else np.ndarray.

    Returns:
        Dict of groups and Union[pd.Dataframe, np.ndarray] of form (x_est, y_est) or (x_est, y_est, duration_est).
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
    """Calculates fill path as expected path of given expected paths

    Args:
        data: input Dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        method: method to calculate expected path ("mean" or "fwp").
        duration: Column name of fixations duration if needed.
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
    """Estimates A assuming 'matrix' equals :math:`A^T A`."""
    # Get eigenvectors and eigenvalues
    evals, evecs = np.linalg.eigh(matrix)

    # Sort by decrease of eigenvalue modulus
    sorted_evals_indexes = np.argsort(evals)[::-1]
    # sorted_evals_indexes = np.argsort(np.abs(evals))[::-1]
    evecs = evecs[:, sorted_evals_indexes]
    evals = evals[sorted_evals_indexes]

    # Rank is determined by positive eigenvalues
    A_rank = (evals > tol).sum()
    # A_rank = (np.abs(evals) > tol).sum()
    evals = np.diag(evals[:A_rank])
    evecs = evecs[:, :A_rank]

    # Restore matrix
    S = sqrtm(evals)
    U = np.identity(A_rank)

    US = U.dot(S)
    return US.dot(evecs.T).T, A_rank


def get_sim_matrix(scanpaths: List[NDArray], sim_metric: Callable) -> np.ndarray:
    """Computes similarity matrix given non-trivial metric.

    Args:
        scanpaths: list of scanpaths, each being 2D-array of shape (2, n).
        sim_metric: similarity metric.

    Returns:
        scaled similarity matrix.
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


def get_dist_matrix(
    scanpaths: List[pd.DataFrame], dist_metric: Callable
) -> pd.DataFrame:
    """Computes pairwise distance matrix given distance metric.

    Args:
        scanpaths: List of scanpaths DataFrames of form (x, y)
        dist_metric: Metric used to calculate distance from features.scanpath_dist
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
    """Reorder matrix using hierarchical clustering.

    Args:
        sim_matrix: similarity matrix to reorder.
        metric: metric used in building matrix.

    Returns:
        reordered matrix.
    """

    Z = linkage(sim_matrix, method="ward", metric=metric)
    ordered_leaves = leaves_list(Z)
    reordered_matrix = sim_matrix[ordered_leaves, :][:, ordered_leaves]
    return reordered_matrix


def optimal_leaf_ordering_clustering(
    sim_matrix: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Reorder matrix using optimal leaf ordering.

    Args:
        sim_matrix: similarity matrix to reorder.

    Returns:
        reordered matrix.
    """

    Z = linkage(sim_matrix, method="ward", metric=metric)
    Z = optimal_leaf_ordering(Z, sim_matrix)
    ordered_leaves = leaves_list(Z)
    reordered_matrix = sim_matrix[ordered_leaves, :][:, ordered_leaves]
    return reordered_matrix


def dimensionality_reduction_order(sim_matrix: np.ndarray) -> np.ndarray:
    """Reorder matrix using Multi-Dimensional Scaling (MDS).

    Args:
        sim_matrix: similarity matrix to reorder.

    Returns:
        reordered matrix.
    """

    mds = MDS(n_components=2, dissimilarity="precomputed")
    coords = mds.fit_transform(1 - sim_matrix)
    # reorder the matrix based on the first MDS component
    order = np.argsort(coords[:, 0])
    reordered_matrix = sim_matrix[order, :][:, order]
    return reordered_matrix


def spectral_order(sim_matrix: np.ndarray) -> np.ndarray:
    """Reorder matrix using spectral reordering.

    Args:
        sim_matrix: similarity matrix to reorder.

    Returns:
        reordered matrix.
    """

    L = laplacian(sim_matrix, normed=True)

    _, eigenvectors = eigsh(L, k=2, which="SM")

    # reorder based on the second smallest eigenvector (fiedler vector)
    fiedler_vector = eigenvectors[:, 1]
    order = np.argsort(fiedler_vector)
    reordered_matrix = sim_matrix[order, :][:, order]
    return reordered_matrix


# ======================== COMPROMISE MATRIX ========================
def get_center_matrix(weight_vector: np.ndarray) -> np.ndarray:
    """Calculates centering matrix Theta.

    Args:
        weight_vector: vector of weights.

    Returns:
        centering matrix.
    """
    assert np.sum(weight_vector) > 0, "Sum of weights must be greater than 0"
    weight_vector = weight_vector.astype(np.float32)
    weight_vector /= np.sum(weight_vector)
    E = np.eye(weight_vector.size)
    Theta = E - np.matmul(
        np.ones(weight_vector.size)[:, np.newaxis], weight_vector[:, np.newaxis].T
    )
    return Theta


def get_cross_product_matrix(
    D: np.ndarray, weight_vector: np.ndarray = None
) -> np.ndarray:
    """Calculates cross-product matrix.

    Args:
        D: distance matrix.
        weight_vector: vector of weights.

    Returns:
        cross-product matrix.
    """

    if weight_vector is None:
        weight_vector = np.ones(D.shape[0])

    Theta = get_center_matrix(weight_vector)
    return -0.5 * Theta @ D @ Theta.T


def compute_rv_coefficient(S1: np.ndarray, S2: np.ndarray) -> float:
    """Calculate the RV coefficient between two cross-product matrices.

    Args:
        S1: first cross-product matrix.
        S2: second cross-product matrix.

    Returns:
        RV coefficient.
    """
    numerator = np.trace(S1 @ S2.T)
    denominator = np.sqrt(np.trace(S1 @ S1.T) * np.trace(S2 @ S2.T))
    return numerator / denominator


def get_compromise_matrix(distance_matrices: List[np.ndarray]) -> np.ndarray:
    """Compute the compromise matrix from a list of distance matrices.

    Args:
        distance_matrices: List of distance matrices (each a ndarray).

    Returns:
        compromise cross-product matrix.
    """
    assert len(distance_matrices) > 0, "At least one distance matrix is required"
    num_matrices = len(distance_matrices)

    # compute the cross-product matrices
    cross_product_matrices = [get_cross_product_matrix(D) for D in distance_matrices]

    # compute the similarity matrix using RV coefficients
    similarity_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(i, num_matrices):
            rv = compute_rv_coefficient(
                cross_product_matrices[i], cross_product_matrices[j]
            )
            similarity_matrix[i, j] = similarity_matrix[j, i] = rv

    # eigen-decomposition of the similarity matrix
    _, eigvecs = np.linalg.eigh(similarity_matrix)
    weights = eigvecs[:, -1]  # eigenvector corresponding to the largest eigenvalue
    # Get the compromise matrix as a weighted sum
    comp_mat = sum(w * G for w, G in zip(weights, cross_product_matrices))

    return comp_mat
