from abc import abstractmethod
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray

from eyetracking.features.extractor import BaseTransformer
from eyetracking.utils import _calc_dt, _get_id, _split_dataframe


class StatsTransformer(BaseTransformer):
    def __init__(
        self,
        features_stats: Dict[str, List[str]],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,  # TODO consider units, i.e. ps, ns, ms.
        dispersion: str = None,
        aoi: str = None,  # TODO add option to calc regular features even with aoi?
        pk: List[str] = None,
        shift_pk: List[str] = None,
        shift_features: Dict[str, List[str]] = None,
        return_df: bool = True,
    ):
        """
        Base class for statistical features. Aggregate function strings must be
        compatible with `pandas`.
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
        self.features_stats = features_stats
        # feature -- i.e. saccade length/speed
        self.feature_names_in = list(features_stats.keys())
        self.shift_mem = None
        self.shift_fill = None
        self.shift_pk = shift_pk
        self.shift_features = shift_features
        self.available_feats = ...
        self.eps = 1e-20

        self.feature_names_in_ = None
        if self.aoi is not None:
            assert self.aoi != "", "Provide non-empty column name for aoi."
            self.aoi_nms = ...
        else:
            self.aoi_nms = [""]  # convenience placeholder

    @staticmethod
    def _err_no_col(f, c):
        return f"Requested feature {f} requires {c} for calculation."

    def _check_feature_names(self, X, *, reset):
        # Since feature names must always be provided with statistics to
        # calculate and are restricted to certain set of available features,
        # this method is irrelevant
        raise AttributeError("Use '_check_features_stats' instead.")

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

    @abstractmethod
    def _check_params(self):
        """
        Method checks that all requested features could be calculated with provided data.
        """
        ...

    def _check_features_stats(self):
        """
        Method checks `self.features_stats` for correct feature names (i.e. keys).
        """
        err_msg = (
            lambda f: f"Feature '{f}' is not supported. Must be one of: "
            f"{', '.join(self.available_feats)}."
        )
        for feat in self.feature_names_in:
            assert feat in self.available_feats, err_msg(feat)

        self._check_params()

    def _check_shift_features(self):
        """
        Method check that provided shift features are correct.
        """
        err_msg_feat = (
            lambda f: f"Passed shift feature '{f}' not found in `features_stats`."
        )
        err_msg_stat = (
            lambda s: f"Passed shift feature stat '{s}' not found in `features_stats`."
        )
        for feat_nm in self.shift_features.keys():
            assert feat_nm in self.feature_names_in, err_msg_feat(feat_nm)
            for stat in self.shift_features[feat_nm]:
                assert stat in self.features_stats[feat_nm], err_msg_stat(stat)
        assert self.shift_pk is not None, "Provide `shift_pk` for shift features."
        assert self.pk is not None, "`shift_pk` must be subset of `pk`."

    def _check_aoi(self, aoi: Iterable[str]):
        for area in aoi:
            assert (
                area in self.aoi_nms
            ), f"Unknown AOI {area} was not seen during `fit`."

    @property
    @abstractmethod
    def _fp(self) -> str:
        """
        Feature prefix to use in feature names.
        """
        ...

    @abstractmethod
    def _calc_feats(
        self, X: pd.DataFrame, features: List[str], transition_mask: NDArray
    ) -> List[Tuple[str, pd.Series]]:
        """
        Method calculates features passed to constructor, i.e. keys of `self.features_stats`.
        In case of `SaccadeFeatures`, it returns dictionary `{'length': np.array, 'velocity': np.array, ...}`.
        `transition_mask` is boolean mask of the same shape as X, i-th value is False if X's i-th value is
        first fixation in block. Block is defined as sequential fixations in same AOI, maximum by inclusion
        (which means that block cannot contain another block). Thus, each AOI is split in blocks and
        first fixation in each block is then removed.
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
            self.shift_mem[shift_group_id][feat_nm][stat]
            if shift_group_id in self.shift_mem.keys()
            else self.shift_fill[feat_nm][stat]
        )

    def _calc_with_aoi(
        self, feat_nms: List[str], X: pd.DataFrame, aoi_nm: str
    ) -> List[Tuple[str, pd.Series]]:
        """
        Helper function to calculate features based on `aoi_nm`.
        """
        if aoi_nm == "":  # internal lib placeholder
            transition_mask = np.ones(len(X)).astype(bool)
            feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                X, feat_nms, transition_mask
            )

        else:
            X_aoi = X[X[self.aoi] == aoi_nm]
            all_aoi = X[self.aoi]
            all_transition_mask = all_aoi == all_aoi.shift(1)
            transition_mask = all_transition_mask[all_aoi == aoi_nm].values

            feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                X_aoi, feat_nms, transition_mask
            )

        return feats

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        self._check_features_stats()

        if self.aoi is not None:
            self.aoi_nms = X[self.aoi].drop_duplicates().values

        self.feature_names_in_ = [
            f"{feat_nm}{aoi_nm}_{stat}"
            for feat_nm in self.feature_names_in
            for stat in self.features_stats[feat_nm]
            for aoi_nm in self.aoi_nms
        ]

        if self.shift_features is None:
            return self

        # Otherwise, self.pk must not be None
        self._check_shift_features()

        if self.aoi is not None:
            self.aoi_nms = X[self.aoi].drop_duplicates().values

        feat_nms = list(self.shift_features.keys())  # names of features
        groups: List[str, pd.DataFrame] = _split_dataframe(
            X, self.shift_pk
        )  # split by shift_pk

        # calc stats for each group
        self.shift_mem = dict()
        for group_id, group_X in groups:
            self.shift_mem[group_id] = dict()

            for aoi_nm in self.aoi_nms:
                group_feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                    feat_nms, group_X, aoi_nm
                )
                aoi_str = "" if aoi_nm == "" else f"_{aoi_nm}"

                for feat_nm, feat_arr in group_feats:  # memoize feats for each group_id
                    self.shift_mem[group_id][feat_nm + aoi_str] = dict()
                    feat_stats: List[str] = self.features_stats[feat_nm]

                    for stat in feat_stats:  # memoize stats for each feat
                        self.shift_mem[group_id][feat_nm + aoi_str][stat] = (
                            feat_arr.apply(stat)
                        )

        # calc mean for each stat (by groups) to use for unknown groups on transform
        self.shift_fill = dict()
        group_ids = list(self.shift_mem.keys())

        for aoi_nm in self.aoi_nms:
            aoi_str = "" if aoi_nm == "" else f"_{aoi_nm}"

            for feat_nm in feat_nms:
                self.shift_fill[feat_nm + aoi_str] = dict()
                feat_stats: List[str] = self.features_stats[feat_nm]

                for stat in feat_stats:
                    stat_sum = 0
                    for group_id in group_ids:
                        stat_sum += self.shift_mem[group_id][feat_nm + aoi_str][stat]
                    self.shift_fill[feat_nm + aoi_str][stat] = stat_sum / max(
                        1, len(group_ids)
                    )

        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, NDArray]:
        if self.features_stats is None:
            return X if self.return_df else X.values

        if self.aoi is not None:
            self._check_aoi(X[self.aoi].drop_duplicates().values)

        feat_nms = list(self.features_stats.keys())
        gathered_stats = []
        column_nms = []

        if self.pk is None:

            for aoi_nm in self.aoi_nms:
                feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                    feat_nms, X, aoi_nm
                )
                aoi_str = "" if aoi_nm == "" else f"_{aoi_nm}"

                for feat_nm, feat_arr in feats:
                    feat_stats: List[str] = self.features_stats[feat_nm]
                    gathered_stats.extend([feat_arr.apply(stat) for stat in feat_stats])
                    column_nms.extend(
                        [f"{self._fp}_{feat_nm}{aoi_str}_{stat}" for stat in feat_stats]
                    )

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

                for aoi_nm in self.aoi_nms:
                    group_feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                        feat_nms, group_X, aoi_nm
                    )
                    aoi_str = "" if aoi_nm == "" else f"_{aoi_nm}"

                    add_cols_nms = len(group_ids) == 1
                    for feat_nm, feat_arr in group_feats:
                        feat_stats: List[str] = self.features_stats[feat_nm]

                        if not feat_arr.empty:  # group_X was not empty
                            stats_group = [feat_arr.apply(stat) for stat in feat_stats]

                            if self._is_shift_feat(feat_nm):  # calc shifts
                                if not feat_arr.empty:
                                    shift_group_id = _get_id(
                                        group_X[self.shift_pk].values[0]
                                    )
                                    stats_group.extend(
                                        [
                                            stats_group[i]
                                            - self._get_shift_val(
                                                shift_group_id,
                                                feat_nm + aoi_str,
                                                feat_stats[i],
                                            )
                                            for i in range(len(stats_group))
                                            if self._is_shift_stat(
                                                feat_nm + aoi_str, feat_stats[i]
                                            )
                                        ]
                                    )
                        else:  # no AOI for given group
                            stats_group = [None for _ in feat_stats]
                            if self._is_shift_feat(feat_nm):  # calc shifts
                                stats_group.extend(
                                    [
                                        None
                                        for i in range(len(stats_group))
                                        if self._is_shift_stat(
                                            feat_nm + aoi_str, feat_stats[i]
                                        )
                                    ]
                                )

                        gath_stats_group.extend(stats_group)

                        if add_cols_nms:
                            column_nms.extend(
                                [
                                    f"{self._fp}_{feat_nm}{aoi_str}_{stat}"
                                    for stat in feat_stats
                                ]
                            )
                            if self._is_shift_feat(feat_nm):
                                column_nms.extend(
                                    [
                                        f"{self._fp}_{feat_nm}{aoi_str}_{stat}_shift"
                                        for stat in feat_stats
                                        if self._is_shift_stat(feat_nm + aoi_str, stat)
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

    @property
    def _fp(self) -> str:
        return "sac"

    def _check_params(self):
        for feat in self.feature_names_in:
            assert self.x is not None, self._err_no_col(feat, "x")
            assert self.y is not None, self._err_no_col(feat, "y")
            if feat in ("speed", "acceleration"):
                assert self.t is not None
                self._err_no_col(feat, "t")

    def _calc_feats(
        self, X: pd.DataFrame, features: List[str], transition_mask: NDArray
    ) -> List[Tuple[str, pd.Series]]:
        feats = []

        dx: pd.Series = X[self.x].diff()
        dy: pd.Series = X[self.y].diff()
        dr = np.sqrt(dx**2 + dy**2)
        dt = None

        if "length" in features:
            sac_len = dr
            feats.append(("length", sac_len[transition_mask]))
        if "acceleration" in features:
            # Acceleration: dx = v0 * t + 1/2 * a * t^2.
            # Above formula is law of uniformly accelerated motion TODO consider direction
            dt = _calc_dt(X, self.duration, self.t)
            sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
            feats.append(("acceleration", sac_acc[transition_mask]))
        if "speed" in features:
            dt = dt if dt is not None else _calc_dt(X, self.duration, self.t)
            sac_spd = dr / (dt + self.eps)
            feats.append(("speed", sac_spd[transition_mask]))

        return feats


class RegressionFeatures(StatsTransformer):
    def __init__(self, rule: List[int], **kwargs):
        """
        :param rule: specify list of quadrants direction to which classifies
        regressions, 1st quadrant being upper-right square of plane and counting
        anti-clockwise.
        """
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed")
        self.rule = rule

    @property
    def _fp(self) -> str:
        return "reg"

    def _check_params(self):
        for r in self.rule:
            assert r in (1, 2, 3, 4), f"Wrong quadrant {r} in `rule`."

        for feat in self.feature_names_in:
            assert self.x is not None, self._err_no_col(feat, "x")
            assert self.y is not None, self._err_no_col(feat, "y")
            if feat in ("speed", "acceleration"):
                assert self.t is not None
                self._err_no_col(feat, "t")

    def _select_regressions(self, dx, dy) -> NDArray:
        mask = np.zeros(len(dx))
        if 1 in self.rule:
            mask = mask | ((dx > 0) & (dy > 0))
        if 2 in self.rule:
            mask = mask | ((dx < 0) & (dy > 0))
        if 3 in self.rule:
            mask = mask | ((dx < 0) & (dy < 0))
        if 4 in self.rule:
            mask = mask | ((dx > 0) & (dy < 0))
        return mask

    def _calc_feats(
        self, X: pd.DataFrame, features: List[str], transition_mask: NDArray
    ) -> List[Tuple[str, pd.Series]]:
        feats = []

        dx: pd.Series = X[self.x].diff()
        dy: pd.Series = X[self.y].diff()
        sm = self._select_regressions(dx, dy)  # selection_mask
        dr = np.sqrt(dx**2 + dy**2)
        dt = None

        tm = transition_mask[sm]
        if "length" in features:
            sac_len = dr
            feats.append(("length", sac_len[sm][tm]))
        if "acceleration" in features:
            # Acceleration: dx = v0 * t + 1/2 * a * t^2.
            # Above formula is law of uniformly accelerated motion TODO consider direction
            dt = _calc_dt(X, self.duration, self.t)
            sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
            feats.append(("acceleration", sac_acc[sm][tm]))
        if "speed" in features:
            dt = dt if dt is not None else _calc_dt(X, self.duration, self.t)
            sac_spd = dr / (dt + self.eps)
            feats.append(("speed", sac_spd[sm][tm]))

        return feats


class MicroSaccades(StatsTransformer):
    def __init__(self, min_dispersion: float, max_speed: float, **kwargs):
        """
        :param rule: specify list of quadrants direction to which classifies
        regressions, 1st quadrant being upper-right square of plane and counting
        anti-clockwise.
        """
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed")
        self.min_dispersion = min_dispersion
        self.max_speed = max_speed

    @property
    def _fp(self) -> str:
        return "microsac"

    def _check_params(self):
        for feat in self.feature_names_in:
            assert self.x is not None, self._err_no_col(feat, "x")
            assert self.y is not None, self._err_no_col(feat, "y")
            assert self.dispersion is not None, self._err_no_col(feat, "dispersion")
            if feat in ("speed", "acceleration"):
                assert self.t is not None
                self._err_no_col(feat, "t")

    def _calc_feats(
        self, X: pd.DataFrame, features: List[str], transition_mask: NDArray
    ) -> List[Tuple[str, pd.Series]]:
        feats = []

        dx: pd.Series = X[self.x].diff()
        dy: pd.Series = X[self.y].diff()
        dr = np.sqrt(dx**2 + dy**2)

        # selection_mask
        sm = (dr < self.max_speed) & (X[self.dispersion] > self.min_dispersion)

        dt = None

        tm = transition_mask[sm]
        if "length" in features:
            sac_len = dr
            feats.append(("length", sac_len[sm][tm]))
        if "acceleration" in features:
            # Acceleration: dx = v0 * t + 1/2 * a * t^2.
            # Above formula is law of uniformly accelerated motion TODO consider direction
            dt = _calc_dt(X, self.duration, self.t)
            sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
            feats.append(("acceleration", sac_acc[sm][tm]))
        if "speed" in features:
            dt = dt if dt is not None else _calc_dt(X, self.duration, self.t)
            sac_spd = dr / (dt + self.eps)
            feats.append(("speed", sac_spd[sm][tm]))

        return feats


class FixationFeatures(StatsTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_feats = ("duration", "vad")

    @property
    def _fp(self) -> str:
        return "fix"

    def _check_params(self):
        for feat in self.feature_names_in:
            if feat == "duration":
                assert self.duration is not None, self._err_no_col(feat, "duration")
            elif feat == "vad":
                assert self.dispersion is not None, self._err_no_col(feat, "dispersion")

    def _calc_feats(
        self, X: pd.DataFrame, features: List[str], transition_mask: NDArray
    ) -> List[Tuple[str, pd.Series]]:
        feats = []

        if "duration" in features:
            feats.append(("duration", X[self.duration][transition_mask]))
        if "vad" in features:
            feats.append(("vad", X[self.dispersion][transition_mask]))

        return feats
