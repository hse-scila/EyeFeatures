from typing import Dict, List, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize


def _target_norm(fwp: np.ndarray, fixations: np.ndarray) -> float:
    return np.linalg.norm(fixations - fwp, axis=1).sum()


@jit(forceobj=True, looplift=True)
def get_expected_path(
    data: pd.DataFrame,
    x: str,
    y: str,
    path_pk: List[str],
    pk: List[str],
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
    :param duration: Column name of fixations duration if needed
    :param return_df: Return pd.Dataframe object else np.ndarray
    :return: Dict of groups and pd.Dataframe or np.ndarray of form (x_est, y_est, duration_est [if duration is passed])
    """

    assert set(path_pk).issubset(set(pk)), "path_pk must be a subset of pk"

    columns = [x, y]
    if duration is not None:
        columns.append(duration)

    expected_paths = dict()
    path_groups = data[path_pk].drop_duplicates().values
    pk_dif = [col for col in pk if col not in path_pk]  # pk \ path_pk

    for path_group in path_groups:
        length = 0
        cur_data = data[pd.DataFrame(data[path_pk] == path_group).all(axis=1)]
        expected_path, cur_paths = [], []
        groups = cur_data[pk_dif].drop_duplicates().values
        for group in groups:
            mask = pd.DataFrame(cur_data[pk_dif] == group).all(axis=1)
            path_data = cur_data[mask]
            length = max(length, len(path_data))
            cur_paths.append(path_data[columns].values)

        for i in range(length):
            vector_coord = []
            cnt, total_duration = 0, 0
            for path in cur_paths:
                if path.shape[0] > i:
                    cnt += 1
                    vector_coord.append(path[i, :2])
                    if len(columns) == 3:
                        total_duration += path[i, 2]

            vector_coord = np.array(vector_coord)
            fwp_init = np.mean(vector_coord, axis=0)
            fwp_init = minimize(
                _target_norm, fwp_init, args=(vector_coord,), method="L-BFGS-B"
            )
            next_fixation = [fwp_init.x[0], fwp_init.x[1]]
            if len(columns) == 3:
                next_fixation.append(total_duration / cnt)
            expected_path.append(next_fixation)
        ret_columns = ["x_est", "y_est"]
        if len(columns) == 3:
            ret_columns.append("duration_est")
        path_df = pd.DataFrame(expected_path, columns=ret_columns)
        expected_paths["_".join([str(g) for g in path_group])] = (
            path_df if return_df else path_df.values
        )

    return expected_paths


def get_fill_path(
    paths: List[pd.DataFrame], x: str, y: str, duration: str = None
) -> pd.DataFrame:
    all_paths = pd.concat(
        [path.assign(pid=k) for k, path in enumerate(paths)], ignore_index=True
    )
    all_paths["dummy"] = 1
    return list(
        get_expected_path(
            data=all_paths,
            x=x,
            y=y,
            path_pk=["dummy"],
            pk=["dummy", "pid"],
            duration=duration,
        ).values()
    )[0]
