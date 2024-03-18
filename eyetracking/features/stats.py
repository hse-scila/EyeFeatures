import numpy as np
import pandas as pd

from numba import jit

from typing import List, Union
from extractor import BaseTransformer


class SaccadeLength(BaseTransformer):
    def __init__(
        self,
        stats: List[str],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, duration, dispersion, aoi, pk, return_df)
        self.stats = stats

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            sac_len: pd.DataFrame = np.sqrt(dx ** 2 + dy ** 2)
            column_names = [f"sac_len_{stat}" for stat in self.stats]
            gathered_features = [[sac_len.apply(stat)] for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                sac_len: pd.DataFrame = np.sqrt(dx ** 2 + dy ** 2)
                for stat in self.stats:
                    column_names.append(
                        f'sac_len_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append([sac_len.apply(stat)])

        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )

        return features_df if self.return_df else features_df.values


class FixationDuration(BaseTransformer):
    def __init__(
        self,
        stats: List[str],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, duration, dispersion, aoi, pk, return_df)
        self.stats = stats

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert self.t is not None, "Error: provide t column before calling transform"

        if self.pk is None:
            if self.duration is None:
                fix_dur: pd.DataFrame = X[self.t].diff()
            else:
                fix_dur: pd.DataFrame = X[self.t]
            column_names = [f"fix_dur_{stat}" for stat in self.stats]
            gathered_features = [[fix_dur.apply(stat)] for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.duration is None:
                    fix_dur: pd.DataFrame = current_X[self.t].diff()
                else:
                    fix_dur: pd.DataFrame = current_X[self.duration]
                for stat in self.stats:
                    column_names.append(
                        f'fix_dur_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append([fix_dur.apply(stat)])

        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )

        return features_df if self.return_df else features_df.values


class SaccadeVelocity(BaseTransformer):
    def __init__(
        self,
        stats: List[str],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(x, y, t, duration, dispersion, aoi, pk, return_df)
        self.stats = stats
        self.eps = eps

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            dr = np.sqrt(dx ** 2 + dy ** 2)
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
            else:
                dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
            sac_vel: pd.DataFrame = dr / (dt + self.eps)
            column_names = [f"sac_vel_{stat}" for stat in self.stats]
            gathered_features = [[sac_vel.apply(stat)] for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                dr = np.sqrt(dx ** 2 + dy ** 2)
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                else:
                    dt = current_X[self.t] - (
                        current_X[self.t] + current_X[self.duration] / 1000
                    ).shift(1)
                sac_vel: pd.DataFrame = dr / (dt + self.eps)
                for stat in self.stats:
                    column_names.append(
                        f'sac_vel_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append([sac_vel.apply(stat)])

        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )

        return features_df if self.return_df else features_df.values


class RegressionCount(BaseTransformer):
    def __init__(
        self,
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, duration, dispersion, aoi, pk, return_df)

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            gaze_vec = pd.concat([dx, dy], axis=1)
            reg_count: pd.DataFrame = gaze_vec[
                (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
            ].shape[0]
            column_names = [f"reg_count"]
            gathered_features = [[reg_count]]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                gaze_vec = pd.concat([dx, dy], axis=1)
                reg_count: pd.DataFrame = gaze_vec[
                    (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                ].shape[0]
                column_names.append(f'reg_count_{"_".join([str(g) for g in group])}')
                gathered_features.append([reg_count])
        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )
        return features_df if self.return_df else features_df.values


class FixationVAD(BaseTransformer):
    def __init__(
        self,
        stats: List[str],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,
        dispersion: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x, y, t, duration, dispersion, aoi, pk, return_df)
        self.stats = stats

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert (
            self.dispersion is not None
        ), "Error: provide dispersion column before calling transform"

        if self.pk is None:
            fix_vad: pd.DataFrame = X[self.dispersion]
            column_names = [f"fix_disp_{stat}" for stat in self.stats]
            gathered_features = [[fix_vad.apply(stat)] for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                fix_vad: pd.DataFrame = current_X[self.dispersion]
                for stat in self.stats:
                    column_names.append(
                        f'fix_disp_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append([fix_vad.apply(stat)])

        features_df = pd.DataFrame(
            data=np.array(gathered_features).T, columns=column_names
        )

        return features_df if self.return_df else features_df.values
