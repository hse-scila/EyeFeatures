from typing import List, Union

import numpy as np
import pandas as pd
from extractor import BaseTransformer
from numba import jit


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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
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
            sac_len: pd.DataFrame = np.sqrt(dx**2 + dy**2)
            column_names = [f"sac_len_{stat}" for stat in self.stats]
            gathered_features = [sac_len.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                sac_len: pd.DataFrame = np.sqrt(dx**2 + dy**2)
                for stat in self.stats:
                    column_names.append(
                        f'sac_len_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(sac_len.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class SaccadeAcceleration(BaseTransformer):
    """
    Acceleration: dx = v0 * t + 1/2 * a * t^2.
    Above formula is law of uniformly accelerated
    motion (TODO consider another way to calculate acceleration).
    """

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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
        self.stats = stats
        self.eps = eps

    def _check_init(self) -> None:
        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        self._check_init()

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            dr = np.sqrt(dx**2 + dy**2)
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
            else:
                dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
            sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
            feature_names = [f"sac_acc_{stat}" for stat in self.stats]
            gathered_features = [sac_acc.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            feature_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                dr = np.sqrt(dx**2 + dy**2)
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                else:
                    dt = current_X[self.t] - (
                        current_X[self.t] + current_X[self.duration] / 1000
                    ).shift(1)
                sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                for stat in self.stats:
                    feature_names.append(
                        f'sac_acc_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(sac_acc.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=feature_names)

        return features_df if self.return_df else features_df.values


class SaccadeVelocity(
    BaseTransformer
):  # TODO 1. Negative velocity? 2. We have speed, not velocity
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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
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
            dr = np.sqrt(dx**2 + dy**2)
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(
                    1
                )  # TODO consider units for t/duration
            else:
                dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
            sac_vel: pd.DataFrame = dr / (dt + self.eps)
            column_names = [f"sac_vel_{stat}" for stat in self.stats]
            gathered_features = [sac_vel.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                dr = np.sqrt(dx**2 + dy**2)
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
                    gathered_features.append(sac_vel.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
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
            gathered_features = [fix_dur.apply(stat) for stat in self.stats]
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
                    gathered_features.append(fix_dur.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
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
            gathered_features = [fix_vad.apply(stat) for stat in self.stats]
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
                    gathered_features.append(fix_vad.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values

class RegressionLength(BaseTransformer):
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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
        self.stats = stats

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            gaze_vec = pd.concat([dx, dy], axis=1)
            reg_only = gaze_vec[
                (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                ]
            reg_len: pd.DataFrame = np.sqrt(reg_only.norm_pos_x ** 2 + reg_only.norm_pos_y ** 2)
            column_names = [f"reg_len_{stat}" for stat in self.stats]
            gathered_features = [reg_len.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                gaze_vec = pd.concat([dx, dy], axis=1)
                reg_only = gaze_vec[
                    (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                    ]
                reg_len: pd.DataFrame = np.sqrt(reg_only.norm_pos_x ** 2 + reg_only.norm_pos_y ** 2)
                for stat in self.stats:
                    column_names.append(f'reg_len_{stat}_{"_".join([str(g) for g in group])}')
                    gathered_features.append(reg_len.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)
        return features_df if self.return_df else features_df.values

class RegressionVelocity(BaseTransformer):
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
        eps: float = 1e-8
    ):
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
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
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
            else:
                dur = X[self.duration]
            gaze_vec = pd.concat([dx, dy, dur, X[self.t]],  axis=1)
            reg_only = gaze_vec[
                (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                ]
            dr = np.sqrt(reg_only.norm_pos_x ** 2 + reg_only.norm_pos_y ** 2)
            dt = reg_only.start_timestamp - (reg_only.start_timestamp + reg_only.duration / 1000).shift(1)
            reg_vel: pd.DataFrame = dr / (dt + self.eps)
            column_names = [f"reg_vel_{stat}" for stat in self.stats]
            gathered_features = [reg_vel.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                else:
                    dur = current_X[self.duration]
                gaze_vec = pd.concat([dx, dy, dur, current_X[self.t]], axis=1)
                reg_only = gaze_vec[
                    (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                    ]
                dr = np.sqrt(reg_only.norm_pos_x ** 2 + reg_only.norm_pos_y ** 2)
                dt = reg_only.start_timestamp - (reg_only.start_timestamp + reg_only.duration / 1000).shift(1)
                reg_vel: pd.DataFrame = dr / (dt + self.eps)
                for stat in self.stats:
                    column_names.append(
                        f'reg_vel_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(reg_vel.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values

class RegressionAcceleration(BaseTransformer):
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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
        self.stats = stats
        self.eps = eps

    def check_init(self) -> None:
        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        self.check_init()

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
            else:
                dur = X[self.duration]
            gaze_vec = pd.concat([dx, dy, dur, X[self.t]], axis=1)
            reg_only = gaze_vec[
                (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                ]
            dr = np.sqrt(reg_only.norm_pos_x ** 2 + reg_only.norm_pos_y ** 2)
            dt = reg_only.start_timestamp - (reg_only.start_timestamp + reg_only.duration / 1000).shift(1)
            reg_acc: pd.DataFrame = dr / (dt ** 2 + self.eps) * 1 / 2
            feature_names = [f"reg_acc_{stat}" for stat in self.stats]
            gathered_features = [reg_acc.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            feature_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                else:
                    dur = current_X[self.duration]
                gaze_vec = pd.concat([dx, dy, dur, current_X[self.t]], axis=1)
                reg_only = gaze_vec[
                    (gaze_vec.norm_pos_x < 0) | (gaze_vec.norm_pos_y < 0)
                    ]
                dr = np.sqrt(reg_only.norm_pos_x ** 2 + reg_only.norm_pos_y ** 2)
                dt = reg_only.start_timestamp - (reg_only.start_timestamp + reg_only.duration / 1000).shift(1)
                reg_acc: pd.DataFrame = dr / (dt ** 2 + self.eps) * 1 / 2
                for stat in self.stats:
                    feature_names.append(
                        f'reg_acc_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(reg_acc.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=feature_names)

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
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )

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
            gathered_features = reg_count
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
                gathered_features.append(reg_count)
        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)
        return features_df if self.return_df else features_df.values


class MicroSaccadeLength(BaseTransformer):
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
        min_dispersion: float = None,
        max_velocity: float = None,
        return_df: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
        self.stats = stats
        self.eps = eps
        self.min_dispersion = min_dispersion
        self.max_velocity = max_velocity

    def _check_init(self):
        assert self.x is not None, "Error: provide 'x' column before calling transform"
        assert self.y is not None, "Error: provide 'y' column before calling transform"
        assert self.t is not None, "Error: provide 't' column before calling transform"
        assert (
            self.min_dispersion is not None
        ), "Error: provide 'min_dispersion' for microsaccades detection"
        assert (
            self.max_velocity is not None
        ), "Error: provide 'max_velocity' for microsaccades detection"
        assert (
            self.dispersion is not None
        ), "Error: provide 'dispersion' column before calling transform"

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        self._check_init()

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            dis = X[self.dispersion]
            dr = np.sqrt(dx**2 + dy**2)
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
            else:
                dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
            v = dr / (dt + self.eps)

            sac_len: pd.DataFrame = dr[
                (dis > self.min_dispersion) & (v < self.max_velocity)
            ]
            column_names = [f"microsac_len_{stat}" for stat in self.stats]
            gathered_features = [sac_len.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                dis = current_X[self.dispersion]
                dr = np.sqrt(dx**2 + dy**2)
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                else:
                    dt = current_X[self.t] - (
                        current_X[self.t] + current_X[self.duration] / 1000
                    ).shift(1)
                v = dr / (dt + self.eps)

                # TODO is empty after filtering
                sac_len: pd.DataFrame = dr[
                    (dis > self.min_dispersion) & (v < self.max_velocity)
                ]
                for stat in self.stats:
                    column_names.append(
                        f'microsac_len_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(sac_len.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class MicroSaccadeVelocity(BaseTransformer):
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
        min_dispersion: float = None,
        max_velocity: float = None,
        return_df: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
        self.stats = stats
        self.eps = eps
        self.min_dispersion = min_dispersion
        self.max_velocity = max_velocity

    def _check_init(self):
        assert self.x is not None, "Error: provide 'x' column before calling transform"
        assert self.y is not None, "Error: provide 'y' column before calling transform"
        assert self.t is not None, "Error: provide 't' column before calling transform"
        assert (
            self.min_dispersion is not None
        ), "Error: provide 'min_dispersion' for microsaccades detection"
        assert (
            self.max_velocity is not None
        ), "Error: provide 'max_velocity' for microsaccades detection"
        assert (
            self.dispersion is not None
        ), "Error: provide 'dispersion' column before calling transform"

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        self._check_init()

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            dis = X[self.dispersion]
            dr = np.sqrt(dx**2 + dy**2)
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
            else:
                dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
            v = dr / (dt + self.eps)

            sac_len: pd.DataFrame = v[
                (dis > self.min_dispersion) & (v < self.max_velocity)
            ]
            column_names = [f"microsac_vel_{stat}" for stat in self.stats]
            gathered_features = [sac_len.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                dis = current_X[self.dispersion]
                dr = np.sqrt(dx**2 + dy**2)
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                else:
                    dt = current_X[self.t] - (
                        current_X[self.t] + current_X[self.duration] / 1000
                    ).shift(1)
                v = dr / (dt + self.eps)

                # TODO is empty after filtering
                sac_vel: pd.DataFrame = v[
                    (dis > self.min_dispersion) & (v < self.max_velocity)
                ]
                for stat in self.stats:
                    column_names.append(
                        f'microsac_vel_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(sac_vel.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values


class MicroSaccadeAcceleration(BaseTransformer):
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
        min_dispersion: float = None,
        max_velocity: float = None,
        return_df: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__(
            x=x,
            y=y,
            t=t,
            duration=duration,
            dispersion=dispersion,
            aoi=aoi,
            pk=pk,
            return_df=return_df,
        )
        self.stats = stats
        self.eps = eps
        self.min_dispersion = min_dispersion
        self.max_velocity = max_velocity

    def _check_init(self):
        assert self.x is not None, "Error: provide 'x' column before calling transform"
        assert self.y is not None, "Error: provide 'y' column before calling transform"
        assert self.t is not None, "Error: provide 't' column before calling transform"
        assert (
            self.min_dispersion is not None
        ), "Error: provide 'min_dispersion' for microsaccades detection"
        assert (
            self.max_velocity is not None
        ), "Error: provide 'max_velocity' for microsaccades detection"
        assert (
            self.dispersion is not None
        ), "Error: provide 'dispersion' column before calling transform"

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        self._check_init()

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            dis = X[self.dispersion]
            dr = np.sqrt(dx**2 + dy**2)
            if self.duration is None:
                dur = X[self.t].diff().shift(-1).fillna(0)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
            else:
                dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
            v = dr / (dt + self.eps)
            acc = dr / (dt**2 + self.eps) * 1 / 2

            sac_acc: pd.DataFrame = acc[
                (dis > self.min_dispersion) & (v < self.max_velocity)
            ]
            column_names = [f"microsac_acc_{stat}" for stat in self.stats]
            gathered_features = [sac_acc.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            column_names = []
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                dis = current_X[self.dispersion]
                dr = np.sqrt(dx**2 + dy**2)
                if self.duration is None:
                    dur = current_X[self.t].diff().shift(-1).fillna(0)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                else:
                    dt = current_X[self.t] - (
                        current_X[self.t] + current_X[self.duration] / 1000
                    ).shift(1)
                v = dr / (dt + self.eps)
                acc = dr / (dt**2 + self.eps) * 1 / 2

                # TODO is empty after filtering
                sac_acc: pd.DataFrame = acc[
                    (dis > self.min_dispersion) & (v < self.max_velocity)
                ]
                for stat in self.stats:
                    column_names.append(
                        f'microsac_acc_{stat}_{"_".join([str(g) for g in group])}'
                    )
                    gathered_features.append(sac_acc.apply(stat))

        features_df = pd.DataFrame(data=[gathered_features], columns=column_names)

        return features_df if self.return_df else features_df.values
