from typing import Union

import numpy as np
from numpy.typing import NDArray


def _get_distance(
    v: Union[float, NDArray], u: Union[float, NDArray], distance: str
) -> Union[float, NDArray]:
    """
    Method computes vectors norm given distance.
    :param v: number/vector/matrix.
    :param u: number/vector/matrix.
    :param distance: str, distance to compute.
    :return: computed distance, number/vector.

    .. note::
        If u and v are numbers, then all distances are equivalent to |v - u|.
        If u and v are vectors of size d, then corresponding metric in R^d is returned.
        If u and v are matrices, then must be of size n x d, rows are treated as vectors.

        All distances are norms in Euclidean space.
    """
    is_matrix = isinstance(v, np.ndarray) and len(v.shape) > 1
    if distance == "euc":
        return _euc_distance(v, u, is_matrix)
    if distance == "manhattan":
        return _manhattan_distance(v, u, is_matrix)
    if distance == "chebyshev":
        return _chebyshev_distance(v, u, is_matrix)

    raise NotImplementedError(f"Provided distance '{distance}' is not supported.")


# Warning: All distances are norms in Euclidean space. If this changes,
# modify the docstring of _get_distance accordingly.


def _euc_distance(
    v: Union[float, NDArray], u: Union[float, NDArray], is_matrix
) -> Union[float, NDArray]:
    if not is_matrix:
        return np.sqrt(np.sum(np.square(v - u)))
    else:
        return np.sqrt(np.sum(np.square(v - u), axis=1))


def _manhattan_distance(
    v: Union[float, NDArray], u: Union[float, NDArray], is_matrix
) -> Union[float, NDArray]:
    if not is_matrix:
        return np.sum(np.abs(v - u))
    else:
        return np.sum(np.abs(v - u), axis=1)


def _chebyshev_distance(
    v: Union[float, NDArray], u: Union[float, NDArray], is_matrix
) -> Union[float, NDArray]:
    if not is_matrix:
        return np.max(np.abs(v - u))
    else:
        return np.max(np.abs(v - u), axis=1)
