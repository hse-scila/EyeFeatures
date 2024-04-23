from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray
from pandas._typing import AggFuncType  # TODO discuss

from eyetracking.features.extractor import BaseTransformer
from eyetracking.utils import _get_id, _split_dataframe


class StatsTransformer(BaseTransformer):
    def __init__(
        self,
        features_stats: Dict[str, List[AggFuncType]],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,  # TODO consider units, i.e. ps, ns, ms.
        dispersion: str = None,
        aoi: str = None,
        pk: List[str] = None,
        shift_pk: List[str] = None,
        shift_features: Dict[str, List[AggFuncType]] = None,
        return_df: bool = True,
    ):
        """
        Base class for statistical features.
        """
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
        self.feature_stats = features_stats
        self.feature_names_in_ = list(features_stats.keys())
        self.shift_mem = None
        self.shift_fill = None
        self.shift_pk = shift_pk
        self.shift_features = shift_features
        self.available_feats = ...
        self.eps = 1e-20

    def _check_feature_names(
        self, X, *, reset
    ):  # TODO discuss this BaseEstimator's method
        # Since feature names must always be provided with statistics to
        # calculate and are restricted to certain set of available features,
        # this method is irrelevant
        raise AttributeError("Use '_check_feature_stats' instead.")

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        cast_to_ndarray=True,
        **check_params,
    ):
        # Same reason as for _check_feature_names
        raise AttributeError("This method must not be used.")

    def _check_feature_stats(self):
        """
        Method checks `self.feature_stats` for correct feature names (i.e. keys).
        """
        err_msg = (
            lambda f: f"Feature '{f}' is not supported. Must be one of: "
            f"{', '.join(self.available_feats)}."
        )
        for feat in self.feature_names_in_:
            assert feat in self.available_feats, err_msg(feat)

    def _check_shift_features(self):
        """
        Method check that provided shift features are correct.
        """
        err_msg_feat = (
            lambda f: f"Passed shift feature '{f}' not found in `feature_stats`."
        )
        err_msg_stat = (
            lambda s: f"Passed shift feature stat '{s}' not found in `feature_stats`."
        )
        for feat_nm in self.shift_features.keys():
            assert feat_nm in self.feature_names_in_, err_msg_feat(feat_nm)
            for stat in self.shift_features[feat_nm]:
                assert stat in self.feature_stats[feat_nm], err_msg_stat(stat)
        assert self.shift_pk is not None, "Provide `shift_pk` for shift features."

    @abstractmethod
    def _calc_feats(
        self, X: pd.DataFrame, features: List[str]
    ) -> List[Tuple[str, pd.Series]]:
        """
        Method calculates features passed to constructor, i.e. keys of `self.feature_stats`.
        In case of `SaccadeFeatures`, it returns dictionary `{'length': np.array, 'velocity': np.array, ...}`.
        """
        ...

    def _is_shift_stat(self, feat_nm, stat):
        if self.shift_features is None:
            return False
        if feat_nm in self.shift_features and stat in self.shift_features[feat_nm]:
            return True
        return False

    def _is_shift_feat(self, feat_nm):
        if self.shift_features is None:
            return False
        return feat_nm in self.shift_features

    def _get_shift_val(self, shift_group_id, feat_nm, stat):
        """
        Retrieves corresponding shift value for shift features calculation.
        """
        return (
            self.shift_mem[shift_group_id][feat_nm][str(stat)]
            if shift_group_id in self.shift_mem.keys()
            else self.shift_fill[feat_nm][str(stat)]
        )

    # @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        self._check_feature_stats()

        if self.shift_features is None:
            return self
        self._check_shift_features()

        feat_nms = list(self.shift_features.keys())  # names of features
        groups: List[str, pd.DataFrame] = _split_dataframe(
            X, self.shift_pk
        )  # split by shift_pk

        # calc stats for each group
        self.shift_mem = dict()
        for group_id, group_X in groups:
            self.shift_mem[group_id] = dict()
            group_feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                group_X, feat_nms
            )

            for feat_nm, feat_arr in group_feats:  # memoize feats for each group_id
                self.shift_mem[group_id][feat_nm] = dict()
                feat_stats: List[AggFuncType] = self.feature_stats[feat_nm]

                for stat in feat_stats:  # memoize stats for each feat
                    self.shift_mem[group_id][feat_nm][str(stat)] = feat_arr.apply(stat)

        # calc mean for each stat (by groups) to use for unknown groups on transform
        self.shift_fill = dict()
        group_ids = list(self.shift_mem.keys())

        for feat_nm in feat_nms:
            self.shift_fill[feat_nm] = dict()
            feat_stats: List[AggFuncType] = self.feature_stats[feat_nm]

            for stat in feat_stats:
                stat_sum = 0
                for group_id in group_ids:
                    stat_sum += self.shift_mem[group_id][feat_nm][str(stat)]
                self.shift_fill[feat_nm][str(stat)] = stat_sum / max(1, len(group_ids))

        return self

    # @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, NDArray]:
        if self.feature_stats is None:
            return X if self.return_df else X.values

        feat_nms = list(self.feature_stats.keys())
        gathered_stats = []
        column_nms = []

        if self.pk is None:
            feats: List[Tuple[str, pd.Series]] = self._calc_feats(X, feat_nms)

            for feat_nm, feat_arr in feats:
                feat_stats: List[AggFuncType] = self.feature_stats[feat_nm]
                gathered_stats.extend([feat_arr.apply(stat) for stat in feat_stats])
                column_nms.extend([f"sac_{feat_nm}_{str(stat)}" for stat in feat_stats])

            stats_df = pd.DataFrame(data=[gathered_stats], columns=column_nms)
            # No shift features if pk is None

        else:
            groups: List[str, pd.DataFrame] = _split_dataframe(
                X, self.pk
            )  # split by unique groups

            group_ids = []
            for group_id, group_X in groups:
                group_ids.append(group_id)
                gath_stats_group = []

                group_feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                    group_X, feat_nms
                )

                add_cols_nms = len(group_ids) == 1
                for feat_nm, feat_arr in group_feats:
                    feat_stats: List[AggFuncType] = self.feature_stats[feat_nm]
                    stats_group = [feat_arr.apply(stat) for stat in feat_stats]

                    if self._is_shift_feat(feat_nm):  # calc shifts
                        shift_group_id = _get_id(group_X[self.shift_pk].values[0])
                        stats_group.extend(
                            [
                                stats_group[i]
                                - self._get_shift_val(
                                    shift_group_id, feat_nm, feat_stats[i]
                                )
                                for i in range(len(stats_group))
                                if self._is_shift_stat(feat_nm, feat_stats[i])
                            ]
                        )

                    gath_stats_group.extend(stats_group)

                    if add_cols_nms:
                        column_nms.extend(
                            [f"sac_{feat_nm}_{str(stat)}" for stat in feat_stats]
                        )
                        if self._is_shift_feat(feat_nm):
                            column_nms.extend(
                                [
                                    f"sac_{feat_nm}_{str(stat)}_shift"
                                    for stat in feat_stats
                                    if self._is_shift_stat(feat_nm, stat)
                                ]
                            )

                gathered_stats.append(gath_stats_group)

            stats_df = pd.DataFrame(
                data=gathered_stats, columns=column_nms, index=group_ids
            )

        return stats_df if self.return_df else stats_df.values


class SaccadeFeatures(StatsTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed")

    def _calc_dt(self, X: pd.DataFrame) -> pd.Series:
        if self.duration is None:
            dur = X[self.t].diff().shift(-1).fillna(0)
            dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
        else:
            dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
        return dt

    # @jit(forceobj=True, looplift=True)
    def _calc_feats(
        self, X: pd.DataFrame, features: List[str]
    ) -> List[Tuple[str, pd.Series]]:
        feats = []

        dx: pd.Series = X[self.x].diff()
        dy: pd.Series = X[self.y].diff()
        dr = np.sqrt(dx**2 + dy**2)
        dt = None
        if "length" in features:
            sac_len = dr
            feats.append(("length", sac_len))
        if "acceleration" in features:
            # Acceleration: dx = v0 * t + 1/2 * a * t^2.
            # Above formula is law of uniformly accelerated motion TODO consider direction
            dt = self._calc_dt(X)
            sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
            feats.append(("acceleration", sac_acc))
        if "speed" in features:
            dt = dt if dt is not None else self._calc_dt(X)
            sac_spd = dr / (dt + self.eps)
            feats.append(("speed", sac_spd))

        return feats


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
        shift_pk: List[str] = None,
        shift_features: bool = False,
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
        self.shift_mem = None
        self.shift_fill = None
        self.shift_pk = shift_pk
        self.shift_features = shift_features

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        assert (
            self.x is not None
        ), "Error: provide x column before calling fit in SaccadeLength"
        assert (
            self.y is not None
        ), "Error: provide y column before calling fit in SaccadeLength"
        assert (
            self.t is not None
        ), "Error: provide t column before calling fit in SaccadeLength"

        if self.pk is not None and self.shift_features:
            assert (
                self.shift_features is not None
            ), "Shift features primary key required"
            self.shift_mem = dict()
            groups = X[self.pk].drop_duplicates().values
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                sac_len: pd.DataFrame = np.sqrt(dx**2 + dy**2)
                group_id = "_".join(
                    [str(g) for g in current_X[self.shift_pk].values[0]]
                )
                if group_id not in self.shift_mem.keys():
                    self.shift_mem[group_id] = [[]] * len(self.stats)
                for i in range(len(self.stats)):
                    stat_v = sac_len.apply(self.stats[i])
                    self.shift_mem[group_id][i].append(stat_v)

            group_ids = list(self.shift_mem.keys())
            for i in range(len(group_ids)):
                upd_vals = []
                for s in range(len(self.stats)):
                    stat_sum = 0
                    stat_vs = self.shift_mem[group_ids[i]][s]
                    for j in range(len(stat_vs)):
                        stat_sum += stat_vs[j]
                    upd_vals.append(stat_sum / max(1, len(stat_vs)))
                self.shift_mem[group_ids[i]] = upd_vals

            self.shift_fill = [0] * len(self.stats)
            for i in range(len(self.stats)):
                stat_sum = 0
                for j in range(len(group_ids)):
                    stat_sum += self.shift_mem[group_ids[j]][i]
                self.shift_fill[i] = stat_sum / max(1, len(group_ids))

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

        column_names = [f"sac_len_{stat}" for stat in self.stats]

        if self.pk is None:
            dx = X[self.x].diff()
            dy = X[self.y].diff()
            sac_len: pd.DataFrame = np.sqrt(dx**2 + dy**2)
            gathered_features = [[sac_len.apply(stat) for stat in self.stats]]
        else:
            if self.shift_features:
                for stat in self.stats:
                    column_names.append(f"sac_len_{stat}_shift")
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                dx = current_X[self.x].diff()
                dy = current_X[self.y].diff()
                sac_len: pd.DataFrame = np.sqrt(dx**2 + dy**2)
                cur_features = [sac_len.apply(stat) for stat in self.stats]
                if self.shift_features:
                    group_id = "_".join(
                        [str(g) for g in current_X[self.shift_pk].values[0]]
                    )
                    for i in range(len(self.stats)):
                        shift_v = (
                            self.shift_mem[group_id][i]
                            if group_id in self.shift_mem.keys()
                            else self.shift_fill[i]
                        )
                        cur_features.append(cur_features[i] - shift_v)
                gathered_features.append(cur_features)

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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
        self.shift_mem = None
        self.shift_fill = None

    def _check_init(self) -> None:
        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"
        assert self.t is not None, "Error: provide t column before calling transform"

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        if self.stats is None:
            return X if self.return_df else X.values

        self._check_init()

        column_names = [f"sac_acc_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"sac_acc_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                sac_acc_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = len(X) - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dt.append(row[self.t])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dr = np.sqrt(dx**2 + dy**2)
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = pd.Series(dt).diff().shift(-1).fillna(0)
                                dt = dt - (dt + dur / 1000).shift(1)
                            else:
                                dt = dt - (dt + pd.Series(dur) / 1000).shift(1)
                            sac_acc_aoi[prev_area] = pd.concat(
                                [
                                    sac_acc_aoi[prev_area],
                                    dr / (dt**2 + self.eps) * 1 / 2,
                                ],
                                axis=0,
                            )
                        dx = list()
                        dy = list()
                        dt = list()
                        dur = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dt.append(row[self.t])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(sac_acc_aoi[area]).apply(stat)
                            if sac_acc_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]

            else:
                dx = X[self.x].diff()
                dy = X[self.y].diff()
                dr = np.sqrt(dx**2 + dy**2)
                if self.duration is None:
                    dur = X[self.t].diff().shift(-1).fillna(0)
                    dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
                else:
                    dt = X[self.t] - (X[self.t] + X[self.duration] / 1000).shift(1)
                sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                gathered_features = [[sac_acc.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    sac_acc_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dt.append(row[self.t])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dr = np.sqrt(dx**2 + dy**2)
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = pd.Series(dt).diff().shift(-1).fillna(0)
                                    dt = dt - (dt + dur / 1000).shift(1)
                                else:
                                    dt = dt - (dt + pd.Series(dur) / 1000).shift(1)
                                sac_acc_aoi[prev_area] = pd.concat(
                                    [
                                        sac_acc_aoi[prev_area],
                                        dr / (dt**2 + self.eps) * 1 / 2,
                                    ],
                                    axis=0,
                                )
                            dx = list()
                            dy = list()
                            dt = list()
                            dur = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dt.append(row[self.t])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(sac_acc_aoi[area]).apply(stat)
                                if sac_acc_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    dr = np.sqrt(dx**2 + dy**2)
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                        dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(
                            1
                        )
                    else:
                        dt = current_X[self.t] - (
                            current_X[self.t] + current_X[self.duration] / 1000
                        ).shift(1)
                    sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                    gathered_features.append(
                        [sac_acc.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

        column_names = [f"sac_vel_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"sac_vel_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                sac_vel_aoi = dict([(area, None) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dt.append(row[self.t])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dr = np.sqrt(dx**2 + dy**2)
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = pd.Series(dt).diff().shift(-1).fillna(0)
                                dt = dt - (dt + dur / 1000).shift(1)
                            else:
                                dt = dt - (dt + pd.Series(dur) / 1000).shift(1)
                            if sac_vel_aoi[prev_area] == None:
                                sac_vel_aoi[prev_area] = dr / (dt + self.eps)
                            else:
                                sac_vel_aoi[prev_area] = pd.concat(
                                    [sac_vel_aoi[prev_area], dr / (dt + self.eps)],
                                    axis=0,
                                )
                        dx = list()
                        dy = list()
                        dt = list()
                        dur = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dt.append(row[self.t])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(sac_vel_aoi[area]).apply(stat)
                            if sac_vel_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]
            else:
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
                gathered_features = [[sac_vel.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    sac_vel_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dt.append(row[self.t])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dr = np.sqrt(dx**2 + dy**2)
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = pd.Series(dt).diff().shift(-1).fillna(0)
                                    dt = dt - (dt + dur / 1000).shift(1)
                                else:
                                    dt = dt - (dt + pd.Series(dur) / 1000).shift(1)
                                sac_vel_aoi[prev_area] = pd.concat(
                                    [sac_vel_aoi[prev_area], dr / (dt + self.eps)],
                                    axis=0,
                                )
                            dx = list()
                            dy = list()
                            dt = list()
                            dur = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dt.append(row[self.t])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        prev_row = row.copy()
                        count -= 1
                        # result.append(
                        #     pd.Series(sac_vel_aoi[area]).apply(stat)
                        # )
                    gathered_features.append(
                        [
                            (
                                pd.Series(sac_vel_aoi[area]).apply(stat)
                                if sac_vel_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    dr = np.sqrt(dx**2 + dy**2)
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                        dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(
                            1
                        )
                    else:
                        dt = current_X[self.t] - (
                            current_X[self.t] + current_X[self.duration] / 1000
                        ).shift(1)
                    sac_vel: pd.DataFrame = dr / (dt + self.eps)
                    gathered_features.append(
                        [sac_vel.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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
        assert (
            self.duration is not None
        ), "Error: provide duration column before calling transform"

        column_names = [f"fix_dur_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"fix_dur_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                fix_dur: pd.DataFrame = X.loc[:, [self.duration, self.aoi]]
                gathered_features = [
                    [
                        (
                            fix_dur[fix_dur[self.aoi] == area][self.duration].apply(
                                stat
                            )
                            if fix_dur[fix_dur[self.aoi] == area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]
            else:
                fix_dur: pd.DataFrame = X[self.duration, self.aoi]
                gathered_features = [fix_dur.apply(stat) for stat in self.stats]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    fix_dur: pd.DataFrame = current_X.loc[:, [self.duration, self.aoi]]
                    gathered_features.append(
                        [
                            (
                                fix_dur[fix_dur[self.aoi] == area][self.duration].apply(
                                    stat
                                )
                                if fix_dur[fix_dur[self.aoi] == area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    fix_dur: pd.DataFrame = current_X[self.duration]
                    gathered_features.append(
                        [fix_dur.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

        column_names = [f"fix_disp_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"fix_disp_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                fix_vad: pd.DataFrame = X.loc[:, [self.dispersion, self.aoi]]
                gathered_features = [
                    [
                        (
                            fix_vad[fix_vad[self.aoi] == area][self.dispersion].apply(
                                stat
                            )
                            if fix_vad[fix_vad[self.aoi] == area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]
            else:
                fix_vad: pd.DataFrame = X[self.dispersion]
                gathered_features = [[fix_vad.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    fix_vad: pd.DataFrame = current_X.loc[
                        :, [self.dispersion, self.aoi]
                    ]
                    gathered_features.append(
                        [
                            (
                                fix_vad[fix_vad[self.aoi] == area][
                                    self.dispersion
                                ].apply(stat)
                                if fix_vad[fix_vad[self.aoi] == area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    fix_vad: pd.DataFrame = current_X[self.dispersion]
                    gathered_features.append(
                        [fix_vad.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

        column_names = [f"reg_len_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"reg_len_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                reg_len_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            gaze_vec = pd.concat([dx, dy], axis=1)
                            reg_only = gaze_vec[
                                (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                            ]
                            reg_len_aoi[prev_area] = pd.concat(
                                [
                                    reg_len_aoi[prev_area],
                                    np.sqrt(
                                        reg_only.iloc[:, 0] ** 2
                                        + reg_only.iloc[:, 1] ** 2
                                    ),
                                ],
                                axis=0,
                            )
                        dx = list()
                        dy = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(reg_len_aoi[area]).apply(stat)
                            if reg_len_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]

            else:
                dx = X[self.x].diff()
                dy = X[self.y].diff()
                gaze_vec = pd.concat([dx, dy], axis=1)
                reg_only = gaze_vec[
                    (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                ]
                reg_len: pd.DataFrame = np.sqrt(
                    reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2
                )
                gathered_features = [[reg_len.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    reg_len_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                gaze_vec = pd.concat([dx, dy], axis=1)
                                reg_only = gaze_vec[
                                    (gaze_vec.iloc[:, 0] < 0)
                                    | (gaze_vec.iloc[:, 1] < 0)
                                ]
                                reg_len_aoi[prev_area] = pd.concat(
                                    [
                                        reg_len_aoi[prev_area],
                                        np.sqrt(
                                            reg_only.iloc[:, 0] ** 2
                                            + reg_only.iloc[:, 1] ** 2
                                        ),
                                    ],
                                    axis=0,
                                )
                            dx = list()
                            dy = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(reg_len_aoi[area]).apply(stat)
                                if reg_len_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )

                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    gaze_vec = pd.concat([dx, dy], axis=1)
                    # TODO make regression recognition less sensitive and add square selection
                    reg_only = gaze_vec[
                        (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                    ]
                    reg_len: pd.DataFrame = np.sqrt(
                        reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2
                    )
                    gathered_features.append(
                        [reg_len.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)
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

        column_names = [f"reg_vel_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"reg_vel_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                reg_vel_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dt.append(row[self.t])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = pd.Series(dt).diff().shift(-1).fillna(0)
                            gaze_vec = pd.concat([dx, dy], axis=1)
                            reg_only = gaze_vec[
                                (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                            ]
                            dr = np.sqrt(
                                reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2
                            )
                            dt = dt - (dt + dur / 1000).shift(1)
                            dt = dt.loc[~dt.index.isin(reg_only)]
                            reg_vel_aoi[prev_area] = pd.concat(
                                [reg_vel_aoi[prev_area], dr / (dt + self.eps)], axis=0
                            )
                        dx = list()
                        dy = list()
                        dt = list()
                        dur = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dt.append(row[self.t])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(reg_vel_aoi[area]).apply(stat)
                            if reg_vel_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]

            else:
                dx = X[self.x].diff()
                dy = X[self.y].diff()
                if self.duration is None:
                    dur = X[self.t].diff().shift(-1).fillna(0)
                else:
                    dur = X[self.duration]
                gaze_vec = pd.concat([dx, dy], axis=1)
                reg_only = gaze_vec[
                    (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                ]
                dr = np.sqrt(reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
                dt = dt.loc[~dt.index.isin(reg_only)]
                reg_vel: pd.DataFrame = dr / (dt + self.eps)
                gathered_features = [[reg_vel.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    reg_vel_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dt.append(row[self.t])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = pd.Series(dt).diff().shift(-1).fillna(0)
                                else:
                                    dur = pd.Series(dur)
                                gaze_vec = pd.concat([dx, dy], axis=1)
                                reg_only = gaze_vec[
                                    (gaze_vec.iloc[:, 0] < 0)
                                    | (gaze_vec.iloc[:, 1] < 0)
                                ]
                                dr = np.sqrt(
                                    reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2
                                )
                                dt = dt - (dt + dur / 1000).shift(1)
                                dt = dt.loc[~dt.index.isin(reg_only)]
                                reg_vel_aoi[prev_area] = pd.concat(
                                    [reg_vel_aoi[prev_area], dr / (dt + self.eps)],
                                    axis=0,
                                )
                            dx = list()
                            dy = list()
                            dt = list()
                            dur = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dt.append(row[self.t])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(reg_vel_aoi[area]).apply(stat)
                                if reg_vel_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                    else:
                        dur = current_X[self.duration]
                    gaze_vec = pd.concat([dx, dy], axis=1)
                    reg_only = gaze_vec[
                        (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                    ]
                    dr = np.sqrt(reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                    dt = dt.loc[~dt.index.isin(reg_only)]
                    reg_vel: pd.DataFrame = dr / (dt + self.eps)
                    gathered_features.append(
                        [reg_vel.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

        column_names = [f"reg_acc_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"reg_acc_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                reg_acc_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dt.append(row[self.t])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = pd.Series(dt).diff().shift(-1).fillna(0)
                            gaze_vec = pd.concat([dx, dy], axis=1)
                            reg_only = gaze_vec[
                                (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                            ]
                            dr = np.sqrt(
                                reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2
                            )
                            dt = dt - (dt + dur / 1000).shift(1)
                            dt = dt.loc[~dt.index.isin(reg_only)]
                            reg_acc_aoi[prev_area] = pd.concat(
                                [
                                    reg_acc_aoi[prev_area],
                                    dr / (dt**2 + self.eps) * 1 / 2,
                                ],
                                axis=0,
                            )
                        dx = list()
                        dy = list()
                        dt = list()
                        dur = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dt.append(row[self.t])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(reg_acc_aoi[area]).apply(stat)
                            if reg_acc_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]
            else:
                dx = X[self.x].diff()
                dy = X[self.y].diff()
                if self.duration is None:
                    dur = X[self.t].diff().shift(-1).fillna(0)
                else:
                    dur = X[self.duration]
                gaze_vec = pd.concat([dx, dy], axis=1)
                reg_only = gaze_vec[
                    (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                ]
                dr = np.sqrt(reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2)
                dt = X[self.t] - (X[self.t] + dur / 1000).shift(1)
                dt = dt.loc[~dt.index.isin(reg_only)]
                reg_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                gathered_features = [[reg_acc.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    reg_acc_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dt.append(row[self.t])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = pd.Series(dt).diff().shift(-1).fillna(0)
                                else:
                                    dur = pd.Series(dur)
                                gaze_vec = pd.concat([dx, dy], axis=1)
                                reg_only = gaze_vec[
                                    (gaze_vec.iloc[:, 0] < 0)
                                    | (gaze_vec.iloc[:, 1] < 0)
                                ]
                                dr = np.sqrt(
                                    reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2
                                )
                                dt = dt - (dt + dur / 1000).shift(1)
                                dt = dt.loc[~dt.index.isin(reg_only)]
                                reg_acc_aoi[prev_area] = pd.concat(
                                    [
                                        reg_acc_aoi[prev_area],
                                        dr / (dt**2 + self.eps) * 1 / 2,
                                    ],
                                    axis=0,
                                )
                            dx = list()
                            dy = list()
                            dt = list()
                            dur = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dt.append(row[self.t])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(reg_acc_aoi[area]).apply(stat)
                                if reg_acc_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                    else:
                        dur = current_X[self.duration]
                    gaze_vec = pd.concat([dx, dy], axis=1)
                    reg_only = gaze_vec[
                        (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                    ]
                    dr = np.sqrt(reg_only.iloc[:, 0] ** 2 + reg_only.iloc[:, 1] ** 2)
                    dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(1)
                    dt = dt.loc[~dt.index.isin(reg_only)]
                    reg_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                    gathered_features.append(
                        [reg_acc.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

    # @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        assert self.x is not None, "Error: provide x column before calling transform"
        assert self.y is not None, "Error: provide y column before calling transform"

        column_names = [f"reg_count"]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                column_names.append(f"reg_count_{area}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                reg_count_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            gaze_vec = pd.concat([dx, dy], axis=1)
                            reg_only = gaze_vec[
                                (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                            ]
                            reg_count_aoi[prev_area] = pd.concat(
                                [reg_count_aoi[prev_area], reg_only.shape[0]], axis=0
                            )
                        dx = list()
                        dy = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(reg_count_aoi[area])
                            if reg_count_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                    ]
                ]
            else:
                dx = X[self.x].diff()
                dy = X[self.y].diff()
                gaze_vec = pd.concat([dx, dy], axis=1)
                reg_count: pd.DataFrame = gaze_vec[
                    (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                ].shape[0]
                gathered_features = [reg_count]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    reg_count_aoi = dict([(area, 0) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                gaze_vec = pd.concat([dx, dy], axis=1)
                                reg_only = gaze_vec[
                                    (gaze_vec.iloc[:, 0] < 0)
                                    | (gaze_vec.iloc[:, 1] < 0)
                                ]
                                reg_count_aoi[prev_area] += reg_only.shape[0]
                            dx = list()
                            dy = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append([reg_count_aoi[area] for area in areas])
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    gaze_vec = pd.concat([dx, dy], axis=1)
                    reg_count: pd.DataFrame = gaze_vec[
                        (gaze_vec.iloc[:, 0] < 0) | (gaze_vec.iloc[:, 1] < 0)
                    ].shape[0]
                    gathered_features.append([reg_count])

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)
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

        column_names = [f"microsac_len_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"microsac_len_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                dis = list()
                microsac_len_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dis.append(row[self.dispersion])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                            dt.append(row[self.t])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dr = np.sqrt(dx**2 + dy**2)
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = dt.diff().shift(-1).fillna(0)
                                dt = dt - (dt + dur / 1000).shift(1)
                            v = dr / (dt + self.eps)
                            microsac_only: pd.DataFrame = dr[
                                (pd.Series(dis) > self.min_dispersion)
                                & (v < self.max_velocity)
                            ]
                            microsac_len_aoi[prev_area] = pd.concat(
                                [microsac_len_aoi[prev_area], microsac_only], axis=0
                            )
                        dx = list()
                        dy = list()
                        dis = list()
                        dur = list()
                        dt = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dis.append(row[self.dispersion])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    dt.append(row[self.t])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(microsac_len_aoi[area]).apply(stat)
                            if microsac_len_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]
            else:
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
                gathered_features = [[sac_len.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    dis = list()
                    microsac_len_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dis.append(row[self.dispersion])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                                dt.append(row[self.t])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dr = np.sqrt(dx**2 + dy**2)
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = dt.diff().shift(-1).fillna(0)
                                    dt = dt - (dt + dur / 1000).shift(1)
                                v = dr / (dt + self.eps)
                                microsac_only: pd.DataFrame = dr[
                                    (pd.Series(dis) > self.min_dispersion)
                                    & (v < self.max_velocity)
                                ]
                                microsac_len_aoi[prev_area] = pd.concat(
                                    [microsac_len_aoi[prev_area], microsac_only], axis=0
                                )
                            dx = list()
                            dy = list()
                            dis = list()
                            dur = list()
                            dt = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dis.append(row[self.dispersion])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        dt.append(row[self.t])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(microsac_len_aoi[area]).apply(stat)
                                if microsac_len_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    dis = current_X[self.dispersion]
                    dr = np.sqrt(dx**2 + dy**2)
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                        dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(
                            1
                        )
                    else:
                        dt = current_X[self.t] - (
                            current_X[self.t] + current_X[self.duration] / 1000
                        ).shift(1)
                    v = dr / (dt + self.eps)

                    # TODO is empty after filtering
                    sac_len: pd.DataFrame = dr[
                        (dis > self.min_dispersion) & (v < self.max_velocity)
                    ]
                    gathered_features.append(
                        [sac_len.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

        column_names = [f"microsac_vel_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"microsac_vel_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                dis = list()
                microsac_vel_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dis.append(row[self.dispersion])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                            dt.append(row[self.t])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dr = np.sqrt(dx**2 + dy**2)
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = dt.diff().shift(-1).fillna(0)
                                dt = dt - (dt + dur / 1000).shift(1)
                            v = dr / (dt + self.eps)
                            microsac_only: pd.DataFrame = v[
                                (pd.Series(dis) > self.min_dispersion)
                                & (v < self.max_velocity)
                            ]
                            microsac_vel_aoi[prev_area] = pd.concat(
                                [microsac_vel_aoi[prev_area], microsac_only], axis=0
                            )
                        dx = list()
                        dy = list()
                        dis = list()
                        dur = list()
                        dt = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dis.append(row[self.dispersion])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    dt.append(row[self.t])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(microsac_vel_aoi[area]).apply(stat)
                            if microsac_vel_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]

            else:
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

                sac_vel: pd.DataFrame = v[
                    (dis > self.min_dispersion) & (v < self.max_velocity)
                ]
                gathered_features = [[sac_vel.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    dis = list()
                    microsac_vel_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dis.append(row[self.dispersion])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                                dt.append(row[self.t])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dr = np.sqrt(dx**2 + dy**2)
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = dt.diff().shift(-1).fillna(0)
                                    dt = dt - (dt + dur / 1000).shift(1)
                                v = dr / (dt + self.eps)
                                microsac_only: pd.DataFrame = v[
                                    (pd.Series(dis) > self.min_dispersion)
                                    & (v < self.max_velocity)
                                ]
                                microsac_vel_aoi[prev_area] = pd.concat(
                                    [microsac_vel_aoi[prev_area], microsac_only], axis=0
                                )
                            dx = list()
                            dy = list()
                            dis = list()
                            dur = list()
                            dt = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dis.append(row[self.dispersion])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        dt.append(row[self.t])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(microsac_vel_aoi[area]).apply(stat)
                                if microsac_vel_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    dis = current_X[self.dispersion]
                    dr = np.sqrt(dx**2 + dy**2)
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                        dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(
                            1
                        )
                    else:
                        dt = current_X[self.t] - (
                            current_X[self.t] + current_X[self.duration] / 1000
                        ).shift(1)
                    v = dr / (dt + self.eps)

                    # TODO is empty after filtering
                    sac_vel: pd.DataFrame = v[
                        (dis > self.min_dispersion) & (v < self.max_velocity)
                    ]
                    gathered_features.append(
                        [sac_vel.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

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

        column_names = [f"microsac_acc_{stat}" for stat in self.stats]
        if self.aoi is not None:
            column_names = []
            areas = X[self.aoi].drop_duplicates().values
            for area in areas:
                for stat in self.stats:
                    column_names.append(f"microsac_acc_{area}_{stat}")

        if self.pk is None:
            if self.aoi is not None:
                areas = X[self.aoi].drop_duplicates().values
                dx = list()
                dy = list()
                dt = list()
                dur = list()
                dis = list()
                microsac_acc_aoi = dict([(area, pd.Series([])) for area in areas])
                prev_row = None
                count = X.shape[0] - 1
                for index, row in X.iterrows():
                    if prev_row is not None and (
                        row[self.aoi] != prev_row[self.aoi] or count == 0
                    ):
                        if row[self.aoi] == prev_row[self.aoi]:
                            dx.append(row[self.x])
                            dy.append(row[self.y])
                            dis.append(row[self.dispersion])
                            if self.duration is not None:
                                dur.append(row[self.duration])
                            dt.append(row[self.t])
                        if len(dx) > 1:
                            prev_area = prev_row[self.aoi]
                            dx = pd.Series(dx).diff()
                            dy = pd.Series(dy).diff()
                            dr = np.sqrt(dx**2 + dy**2)
                            dt = pd.Series(dt)
                            if self.duration is None:
                                dur = dt.diff().shift(-1).fillna(0)
                                dt = dt - (dt + dur / 1000).shift(1)
                            v = dr / (dt + self.eps)
                            acc = dr / (dt**2 + self.eps) * 1 / 2
                            microsac_only: pd.DataFrame = acc[
                                (pd.Series(dis) > self.min_dispersion)
                                & (v < self.max_velocity)
                            ]
                            microsac_acc_aoi[prev_area] = pd.concat(
                                [microsac_acc_aoi[prev_area], microsac_only], axis=0
                            )
                        dx = list()
                        dy = list()
                        dis = list()
                        dur = list()
                        dt = list()
                    dx.append(row[self.x])
                    dy.append(row[self.y])
                    dis.append(row[self.dispersion])
                    if self.duration is not None:
                        dur.append(row[self.duration])
                    dt.append(row[self.t])
                    prev_row = row.copy()
                    count -= 1
                gathered_features = [
                    [
                        (
                            pd.Series(microsac_acc_aoi[area]).apply(stat)
                            if microsac_acc_aoi[area].shape[0] > 0
                            else None
                        )
                        for area in areas
                        for stat in self.stats
                    ]
                ]
            else:
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
                gathered_features = [[sac_acc.apply(stat) for stat in self.stats]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                if self.aoi is not None:
                    areas = X[self.aoi].drop_duplicates().values
                    dx = list()
                    dy = list()
                    dt = list()
                    dur = list()
                    dis = list()
                    microsac_acc_aoi = dict([(area, pd.Series([])) for area in areas])
                    prev_row = None
                    count = current_X.shape[0] - 1
                    for index, row in current_X.iterrows():
                        if prev_row is not None and (
                            row[self.aoi] != prev_row[self.aoi] or count == 0
                        ):
                            if row[self.aoi] == prev_row[self.aoi]:
                                dx.append(row[self.x])
                                dy.append(row[self.y])
                                dis.append(row[self.dispersion])
                                if self.duration is not None:
                                    dur.append(row[self.duration])
                                dt.append(row[self.t])
                            if len(dx) > 1:
                                prev_area = prev_row[self.aoi]
                                dx = pd.Series(dx).diff()
                                dy = pd.Series(dy).diff()
                                dr = np.sqrt(dx**2 + dy**2)
                                dt = pd.Series(dt)
                                if self.duration is None:
                                    dur = dt.diff().shift(-1).fillna(0)
                                    dt = dt - (dt + dur / 1000).shift(1)
                                v = dr / (dt + self.eps)
                                acc = dr / (dt**2 + self.eps) * 1 / 2
                                microsac_only: pd.DataFrame = acc[
                                    (pd.Series(dis) > self.min_dispersion)
                                    & (v < self.max_velocity)
                                ]
                                microsac_acc_aoi[prev_area] = pd.concat(
                                    [microsac_acc_aoi[prev_area], microsac_only], axis=0
                                )
                            dx = list()
                            dy = list()
                            dis = list()
                            dur = list()
                            dt = list()
                        dx.append(row[self.x])
                        dy.append(row[self.y])
                        dis.append(row[self.dispersion])
                        if self.duration is not None:
                            dur.append(row[self.duration])
                        dt.append(row[self.t])
                        prev_row = row.copy()
                        count -= 1
                    gathered_features.append(
                        [
                            (
                                pd.Series(microsac_acc_aoi[area]).apply(stat)
                                if microsac_acc_aoi[area].shape[0] > 0
                                else None
                            )
                            for area in areas
                            for stat in self.stats
                        ]
                    )
                else:
                    dx = current_X[self.x].diff()
                    dy = current_X[self.y].diff()
                    dis = current_X[self.dispersion]
                    dr = np.sqrt(dx**2 + dy**2)
                    if self.duration is None:
                        dur = current_X[self.t].diff().shift(-1).fillna(0)
                        dt = current_X[self.t] - (current_X[self.t] + dur / 1000).shift(
                            1
                        )
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
                    gathered_features.append(
                        [sac_acc.apply(stat) for stat in self.stats]
                    )

        features_df = pd.DataFrame(data=gathered_features, columns=column_names)

        return features_df if self.return_df else features_df.values


if __name__ == "__main__":
    data = pd.concat(
        [pd.read_excel("itog_fix_1.xlsx"), pd.read_excel("itog_fix_2.xlsx")], axis=0
    )
    x = "norm_pos_x"
    y = "norm_pos_y"
    t = "start_timestamp"
    dur = "duration"
    dis = "dispersion"

    sac_feats_stats = {"length": ["min", "max"], "speed": ["mean", "kurtosis"]}
    sf = SaccadeFeatures(x=x, y=y, t=t, duration=dur, features_stats=sac_feats_stats)
    sf.fit(data)
    sf.transform(data)
