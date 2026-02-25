from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.linalg import sqrtm
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.manifold import MDS
from tqdm import tqdm


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


def get_sim_matrix(scanpaths: list[NDArray], sim_metric: Callable) -> np.ndarray:
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
    scanpaths: list[pd.DataFrame], dist_metric: Callable
) -> pd.DataFrame:
    """Computes pairwise distance matrix given distance metric.

    Args:
        scanpaths: List of scanpaths DataFrames of form (x, y)
        dist_metric: Metric used to calculate distance from features.dist
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


def get_compromise_matrix(distance_matrices: list[np.ndarray]) -> np.ndarray:
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
    comp_mat = sum(w * G for w, G in zip(weights, cross_product_matrices, strict=False))

    return comp_mat
