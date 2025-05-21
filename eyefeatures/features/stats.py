from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eyefeatures.features.extractor import BaseTransformer
from eyefeatures.utils import (
    _calc_dt,
    _get_id,
    _get_objs,
    _select_regressions,
    _split_dataframe,
    Types
)


class StatsTransformer(BaseTransformer):
    def __init__(
        self,
        features_stats: Dict[str, List[str]],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: None | str = None,  # TODO consider units, i.e. ps, ns, ms.
        dispersion: None | str = None,
        aoi: None | str | List[str] = None,
        calc_without_aoi: bool = False,  # if True, then calculate regular features even with aoi passed
        pk: None | List[str] = None,
        shift_pk: None | List[str] | Tuple[List[str]] = None,
        shift_features: None | Dict[str, List[str]] | Tuple[Dict[str, List[str]]] = None,
        return_df: bool = True,
        warn: bool = True
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
        self.aoi = aoi
        self.calc_without_aoi = calc_without_aoi
        self.aoi_mapper = ...

        self.warn = warn

        self.feature_names_in_ = None

    @staticmethod
    def _err_no_col(f, c):
        return f"Requested feature {f} requires {c} for calculation."

    def _check_feature_names(self, X, *, reset):
        """"""
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
        """"""
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
        Method checks that provided shift features are correct.
        """
        assert self.shift_pk is not None, "Provide `shift_pk` for shift features."
        assert isinstance(self.shift_pk, list) or isinstance(self.shift_pk, tuple),\
            f"`shift_pk` must be list or tuple, got {type(self.shift_pk)}."
        assert len(self.shift_pk) > 0, "`shift_pk` must be non-empty."

        assert self.shift_features is not None, "Provide `shift_features` for shift features."
        assert isinstance(self.shift_features, dict) or isinstance(self.shift_features, tuple),\
            f"`shift_features` must be dict or tuple, got {type(self.shift_features)}."

        # assert self.pk is not None, "`shift_pk` must be subset of `pk`."  # could be not a subset

        self._preprocess_shift_features()

        err_msg_feat = (
            lambda f: f"Passed shift feature '{f}' not found in `features_stats`."
        )
        err_msg_stat = (
            lambda s: f"Passed shift feature stat '{s}' not found in `features_stats`."
        )
        for shift_features in self.shift_features:
            for feat_nm in shift_features.keys():
                assert feat_nm in self.feature_names_in, err_msg_feat(feat_nm)
                for stat in shift_features[feat_nm]:
                    assert stat in self.features_stats[feat_nm], err_msg_stat(stat)

    # method called on fit
    def _check_aoi_fit(self, X):
        if self.aoi is not None:  # check if aoi columns contain any NaNs
            assert isinstance(self.aoi, str) or isinstance(self.aoi, list),\
                f"`aoi` must be str or List[str], got {type(self.aoi)}."

            if isinstance(self.aoi, str):
                self.aoi = [self.aoi]

            assert "" not in self.aoi, 'Empty string "" as value in `aoi` columns is not allowed.'

            for aoi_col in self.aoi:
                if aoi_col != "":
                    aoi_view = X[aoi_col]
                    if aoi_view.isnull().values.any():
                        raise RuntimeError(f"Passed column '{aoi_col}' for AOI contains NaNs.")

        self._preprocess_aoi(X)

    def _preprocess_aoi(self, X: pd.DataFrame):
        if self.aoi is not None:
            self.aoi_mapper = dict()

            if self.calc_without_aoi:
                self.aoi_mapper[""] = [""]

            for aoi_col in self.aoi:
                aoi_view = X[aoi_col]
                self.aoi_mapper[aoi_col] = aoi_view.drop_duplicates().values.tolist()

        else:
            self.aoi_mapper = {"": [""]}  # convenience placeholder

    # method called on transform
    def _check_aoi_transform(self, X: pd.DataFrame):
        if self.aoi is not None:  # check if aoi column contains any NaNs
            assert "" not in self.aoi, 'Empty string "" as value in `aoi` columns is not allowed.'

            for aoi_col in self.aoi:
                if aoi_col == "":  # lib placeholder
                    continue

                aoi_view = X[aoi_col]
                if aoi_view.isnull().values.any():
                    raise RuntimeError(f"Passed column '{aoi_col}' for AOI contains NaNs.")

                for v in aoi_view:
                    assert (
                            v in self.aoi_mapper[aoi_col]
                    ), f"Unknown AOI value {v} was not seen during `fit` in '{aoi_col}'."

    def _preprocess_shift_features(self):
        if isinstance(self.shift_pk, list) and self.shift_features is None:
            self.shift_pk = (self.shift_pk,)
            self.shift_features = (None,)

        elif isinstance(self.shift_pk, list) and isinstance(self.shift_features, dict):
            self.shift_pk = (self.shift_pk,)
            self.shift_features = (self.shift_features,)

        elif isinstance(self.shift_pk, tuple) and isinstance(self.shift_features, dict):
            # same shift_features for different shift_pk
            self.shift_features = [
                self.shift_features for _ in range(len(self.shift_pk))
            ]

        # several shift_features for single shift_pk are not allowed - just merge shift_features
        # into single dict

        # elif isinstance(self.shift_pk, list) and isinstance(self.shift_features, tuple):

        elif isinstance(self.shift_pk, tuple) and isinstance(self.shift_features, tuple):
            # different shift_features for different shift_pk
            assert len(self.shift_features) == len(self.shift_pk),\
                f"""If tuple of lists, length of `shift_features`
                    ({len(self.shift_features)}) must correspond
                    to length of `shift_pk` ({len(self.shift_pk)})."""

        else:
            raise ValueError(f"Wrong combination of types for `shift_features` and `shift_pk`."
                             f"Read docs for an example.")

        # sf = Tuple[Dict], sp = Tuple[List]
        for i in range(len(self.shift_pk)):
            assert isinstance(self.shift_features[i], dict), "Wrong value for `shift_features`."
            assert isinstance(self.shift_pk[i], list), "Wrong value for `shift_pk`."


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

    @staticmethod
    def _is_shift_stat(shift_features, feat_nm, stat):
        if shift_features is None:
            return False
        if feat_nm in shift_features and stat in shift_features[feat_nm]:
            return True
        return False

    @staticmethod
    def _is_shift_feat(shift_features, feat_nm):
        if shift_features is None:
            return False
        return feat_nm in shift_features

    def _get_shift_val(self, shift_pk_id, shift_group_id, aoi_col, aoi_val, feat_nm, stat):
        """
        Retrieves corresponding shift value for shift features calculation.
        """
        shift_mem = self.shift_mem[shift_pk_id]
        shift_fill = self.shift_fill[shift_pk_id]
        if shift_group_id not in shift_mem.keys() and self.warn:
            warnings.warn(
                message=f"Group {shift_group_id} for shift_pk {shift_pk_id} was not seen during `fit`."
                        f"Average across all values of {shift_pk_id} is used instead.",
                stacklevel=5)
        return (
            shift_mem[shift_group_id][aoi_col][aoi_val][feat_nm][stat]
            if shift_group_id in shift_mem.keys()
            # and aoi_val in shift_mem[shift_group_id][aoi_val]  # unknown aoi values on transform are not allowed
            else shift_fill[aoi_col][aoi_val][feat_nm][stat]
        )

    def _calc_with_aoi(
        self, feat_nms: List[str], X: pd.DataFrame, aoi_col: str, aoi_val: Any
    ) -> List[Tuple[str, pd.Series]]:
        """
        Helper function to calculate features based on `aoi_col` (name for aoi column)
        and `aoi_val` (aoi value in this column).
        """
        if aoi_col == "":  # internal lib placeholder
            transition_mask = np.ones(len(X)).astype(bool)
            feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                X, feat_nms, transition_mask
            )

        else:
            X_aoi = X[X[aoi_col] == aoi_val]
            all_aoi = X[aoi_col]
            all_transition_mask: pd.Series = (all_aoi == all_aoi.shift(1))
            transition_mask = all_transition_mask[all_aoi == aoi_val].values

            feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                X_aoi, feat_nms, transition_mask
            )

        return feats

    def fit(self, X: pd.DataFrame, y=None):
        self._check_features_stats()
        self._check_aoi_fit(X)

        if self.shift_features is not None:
            self._check_shift_features()

        # all output features that will appear on transform
        self.feature_names_in_ = []
        for aoi_col in self.aoi_mapper:
            for aoi_val in self.aoi_mapper[aoi_col]:
                for feat_nm in self.features_stats:
                    for stat in self.features_stats[feat_nm]:
                        self.feature_names_in_.append(
                            f"{self._fp}_{feat_nm}_{aoi_col}[{aoi_val}]_{stat}"
                            if aoi_col != "" else
                            f"{self._fp}_{feat_nm}_{stat}"
                        )
                        if self.shift_features is not None:
                            for shift_features, shift_pk in zip(self.shift_features, self.shift_pk):
                                shift_pk_id = _get_id(shift_pk)

                                if self._is_shift_feat(shift_features, feat_nm):
                                    if self._is_shift_stat(shift_features, feat_nm, stat):
                                        self.feature_names_in_.append(
                                            f"{self._fp}_{feat_nm}_{aoi_col}[{aoi_val}]_{stat}_shift_{shift_pk_id}"
                                            if aoi_col != "" else
                                            f"{self._fp}_{feat_nm}_{stat}_shift_{shift_pk_id}"
                                        )
        if self.shift_features is None:
            return self

        # OK if self.pk is None
        # self._check_shift_features()

        self.shift_mem = dict()
        for shift_features, shift_pk in zip(self.shift_features, self.shift_pk):
            shift_pk_id = _get_id(shift_pk)

            feat_nms = list(shift_features.keys())  # names of features
            groups: Types.EncodedPartition = _split_dataframe(
                X, shift_pk
            )  # split by shift_pk

            # calc stats for each group
            shift_mem = dict()
            for group_id, group_X in groups:
                shift_mem[group_id] = dict()

                # split group_X by pk into subgroups, calc features
                # for subgroups and take a mean
                if self.pk is None:
                    subgroups_X = [("", group_X)]
                else:
                    subgroups_X: List[(str, pd.DataFrame)] = _split_dataframe(
                        group_X, self.pk
                    )

                for aoi_col in self.aoi_mapper:
                    shift_mem[group_id][aoi_col] = dict()

                    for aoi_val in self.aoi_mapper[aoi_col]:
                        shift_mem[group_id][aoi_col][aoi_val] = dict()

                        subgroups_feats = []
                        for _, subgroup_X in subgroups_X:
                            subgroup_feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                                feat_nms, subgroup_X, aoi_col, aoi_val
                            )
                            subgroups_feats.append({k: v for (k, v) in subgroup_feats})

                        # group_feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                        #     feat_nms, group_X, aoi_col, aoi_val
                        # )
                        for feat_nm in feat_nms:
                        # for feat_nm, feat_arr in group_feats:  # memoize feats for each group_id
                            shift_mem[group_id][aoi_col][aoi_val][feat_nm] = dict()
                            feat_stats: List[str] = self.features_stats[feat_nm]

                            for stat in feat_stats:  # memoize stats for each feat
                                mean = 0
                                for subgroup_dict in subgroups_feats:
                                    feat_arr = subgroup_dict[feat_nm]
                                    feat_val = feat_arr.apply(stat) if len(feat_arr) > 0 else 0
                                    mean += feat_val if not np.isnan(feat_val) else 0
                                shift_mem[group_id][aoi_col][aoi_val][feat_nm][stat] = np.mean(mean)
                                # feat_arr = feat_arr[~np.isnan(feat_arr)]
                                # shift_mem[group_id][aoi_col][aoi_val][feat_nm][stat] = (
                                #     feat_arr.apply(stat) if len(feat_arr) > 0 else 0
                                # )

            self.shift_mem[shift_pk_id] = shift_mem

        # All shift features are calculated up to that point.
        # Given fixed key shift_pk, there could be unknown groups on transform.
        # Calc mean for each stat (by groups) to use for unknown groups on transform.
        self.shift_fill = dict()

        for shift_features, shift_pk in zip(self.shift_features, self.shift_pk):
            shift_pk_id = _get_id(shift_pk)
            shift_mem = self.shift_mem[shift_pk_id]
            shift_fill = dict()

            group_ids = list(shift_mem.keys())
            feat_nms = list(shift_features.keys())  # names of features

            for aoi_col in self.aoi_mapper:
                shift_fill[aoi_col] = dict()

                for aoi_val in self.aoi_mapper[aoi_col]:
                    shift_fill[aoi_col][aoi_val] = dict()

                    for feat_nm in feat_nms:
                        shift_fill[aoi_col][aoi_val][feat_nm] = dict()
                        # feat_stats: List[str] = self.features_stats[feat_nm]
                        feat_stats: List[str] = shift_features[feat_nm]

                        for stat in feat_stats:
                            stat_sum = 0
                            for group_id in group_ids:
                                stat_sum += shift_mem[group_id][aoi_col][aoi_val][feat_nm][stat]
                            shift_fill[aoi_col][aoi_val][feat_nm][stat] = stat_sum / max(
                                1, len(group_ids)
                            )

            self.shift_fill[shift_pk_id] = shift_fill

        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, NDArray]:
        if self.features_stats is None:
            return X if self.return_df else X.values

        self._check_aoi_transform(X)

        feat_nms = list(self.features_stats.keys())
        gathered_stats = []
        column_nms = []

        if self.pk is None:
            groups: Types.EncodedPartition = [("0", X)]
        else:
            groups: Types.EncodedPartition = _split_dataframe(
                X, self.pk
            )  # split by unique groups

        group_ids = []
        for group_id, group_X in groups:
            group_ids.append(group_id)
            gath_stats_group = []

            for aoi_col in self.aoi_mapper:
                for aoi_val in self.aoi_mapper[aoi_col]:
                    group_feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                        feat_nms, group_X, aoi_col, aoi_val
                    )

                    add_cols_nms = len(group_ids) == 1
                    for feat_nm, feat_arr in group_feats:
                        feat_stats: List[str] = self.features_stats[feat_nm]

                        if not feat_arr.empty:  # group_X with AOI was not empty
                            stats_group = [feat_arr.apply(stat) for stat in feat_stats]

                            # if initially shift_features=None
                            if self.shift_features is not None:
                                shift_stats_group = []
                                for shift_features, shift_pk in zip(self.shift_features, self.shift_pk):
                                    shift_pk_id = _get_id(shift_pk)

                                    if self._is_shift_feat(shift_features, feat_nm):  # calc shifts
                                        # --- Because shift_pk was required to be subset of pk ---
                                        # shift_group_id = _get_id(
                                        #     group_X[shift_pk].values[0]
                                        # )
                                        # --------------------------------------------------------
                                        values = group_X[shift_pk].values
                                        shift_group_ids = [_get_id(v) for v in values]
                                        shift_stats_group.extend(
                                            [
                                                np.mean([
                                                    stats_group[i]
                                                    - self._get_shift_val(
                                                        shift_pk_id,
                                                        shift_group_id,
                                                        aoi_col,
                                                        aoi_val,
                                                        feat_nm,
                                                        feat_stats[i],
                                                    )
                                                    for shift_group_id
                                                    in shift_group_ids
                                                ])
                                                for i in range(len(stats_group))
                                                if self._is_shift_stat(  # no aoi_str
                                                    shift_features, feat_nm, feat_stats[i]
                                                )
                                            ]
                                        )
                                stats_group.extend(shift_stats_group)

                        else:  # no AOI for given group
                            stats_group = [None for _ in feat_stats]

                            if self.shift_features is not None:
                                shift_stats_group = []
                                for shift_features, shift_pk in zip(self.shift_features, self.shift_pk):
                                    if self._is_shift_feat(shift_features, feat_nm):  # calc shifts
                                        shift_stats_group.extend(
                                            [
                                                None
                                                for i in range(len(stats_group))
                                                if self._is_shift_stat(
                                                    shift_features, feat_nm, feat_stats[i]
                                                )
                                            ]
                                        )
                                stats_group.extend(shift_stats_group)

                        gath_stats_group.extend(stats_group)
                        # TODO remove, have self.features_names_in_ on fit. But order matters,
                        #  so not removed for now
                        if add_cols_nms:
                            column_nms.extend(
                                [
                                    f"{self._fp}_{feat_nm}_{aoi_col}[{aoi_val}]_{stat}"
                                    if aoi_col != "" else
                                    f"{self._fp}_{feat_nm}_{stat}"
                                    for stat in feat_stats
                                ]
                            )
                            if self.shift_features is not None:
                                for shift_features, shift_pk in zip(self.shift_features, self.shift_pk):
                                    shift_pk_id = _get_id(shift_pk)

                                    if self._is_shift_feat(shift_features, feat_nm):
                                        column_nms.extend(
                                            [
                                                f"{self._fp}_{feat_nm}_{aoi_col}[{aoi_val}]_{stat}_shift_{shift_pk_id}"
                                                if aoi_col != "" else
                                                f"{self._fp}_{feat_nm}_{stat}_shift_{shift_pk_id}"
                                                for stat in feat_stats
                                                if self._is_shift_stat(shift_features, feat_nm, stat)
                                            ]
                                        )

            gathered_stats.append(gath_stats_group)

        assert set(self.feature_names_in_) == set(column_nms)
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
                assert self.t is not None, self._err_no_col(feat, "t")

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
    def __init__(
        self,
        rule: Tuple[int, ...],
        deviation: Union[int, Tuple[int, ...]] = None,
        **kwargs,
    ):
        """
        :param rule: must be either 1) tuple of quadrants direction to classify
            regressions, 1st quadrant being upper-right square of plane and counting
            anti-clockwise or 2) tuple of angles in degrees (0 <= angle <= 360).
        :param deviation: if None, then `rule` is interpreted as quadrants. Otherwise,
            `rule` is interpreted as angles. If integer, then is a +-deviation for all angles.
            If tuple of integers, then must be of the same length as `rule`, each value being
            a corresponding deviation for each angle. Angle = 0 is positive x-axis direction,
            rotating anti-clockwise.
        """
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed", "mask")
        self.rule = rule
        self.deviation = deviation

    @property
    def _fp(self) -> str:
        return "reg"

    def _check_params(self):
        if self.deviation is None:
            for q in self.rule:
                assert q in (1, 2, 3, 4), f"Wrong quadrant {q} in 'rule'."
        else:
            for a in self.rule:
                assert 0 <= a <= 360, f"Angles must be 0 <= angle <= 360, got {a}."
            if isinstance(self.deviation, int):
                assert 0 <= self.deviation <= 180, (
                    f"Deviation must be 0 <= deviation <= 180," f"got {self.deviation}."
                )
            elif isinstance(self.deviation, tuple):
                for d in self.deviation:
                    assert (
                        0 <= d <= 180
                    ), f"Deviation must be 0 <= deviation <= 180, got {d}."
            else:
                raise ValueError(
                    f"Wrong type for 'deviation': '{type(self.deviation)}'."
                )

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
        sm = _select_regressions(dx, dy, self.rule, self.deviation)  # selection_mask
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
        if "mask" in features:
            feats.append(("mask", sm))

        return feats


class MicroSaccades(StatsTransformer):
    def __init__(self, min_dispersion: float, max_speed: float, **kwargs):
        """
        :param rule: specify list of quadrants direction to which classifies
        regressions, 1st quadrant being upper-right square of plane and counting
        anti-clockwise.
        """
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed", "mask")
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
            
        if "mask" in features:
            feats.append(("mask", sm))

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