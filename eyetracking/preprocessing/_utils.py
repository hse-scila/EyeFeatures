from typing import Union, List

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


# ====== Bounding Boxes =========
def _is_valid_circle(points: NDArray, center: NDArray, radius: float):
    for p in points:
        if np.linalg.norm(p - center) > radius:
            return False
    return True


def _build_min_circle(points: NDArray):
    assert points.shape[0] <= 3
    if points.shape[0] == 0:
        return np.array([0, 0, 0])
    if points.shape[0] == 1:
        return np.array([points[0][0], points[0][1], 0])
    if points.shape[0] == 2:
        x = (points[0][0] + points[1][0]) / 2
        y = (points[0][1] + points[1][1]) / 2
        r = np.linalg.norm(points[0] - points[1]) / 2
        return np.array([x, y, r])

    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            x = (points[i][0] + points[j][0]) / 2
            y = (points[i][1] + points[j][1]) / 2
            r = np.linalg.norm(points[i] - points[j]) / 2
            if _is_valid_circle(points, np.array([x, y]), r):
                return np.array([x, y, r])
    # define circle by 3 points
    bx = points[1][0] - points[0][0]
    by = points[1][1] - points[0][1]
    cx = points[2][0] - points[0][0]
    cy = points[2][1] - points[0][1]
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    assert D != 0.0, f"Division by zero"
    x = (cy * B - by * C) / (2 * D) + points[0][0]
    y = (bx * C - cx * B) / (2 * D) + points[0][1]
    r = np.linalg.norm(np.array([x, y]) - points[0])
    return np.array([x, y, r])


def _welzl_algorithm(points: NDArray, border: List[int], N: int):
    if N == 0 or len(border) == 3:
        return _build_min_circle(np.array(border))

    d = _welzl_algorithm(points, border.copy(), N - 1)

    if np.linalg.norm(d[:2] - points[N - 1]) <= d[2]:
        return d
    border.append(points[N - 1])

    return _welzl_algorithm(points, border.copy(), N - 1)


def _get_MEC(points: NDArray):
    points = points.copy()
    return _welzl_algorithm(points, [], len(points))
