from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eyefeatures.features.extractor import BaseTransformer
from eyefeatures.utils import (
    Types,
    _calc_dt,
    _get_angle,
    _select_regressions,
    _split_dataframe,
)


class StatsTransformer(BaseTransformer):
    """Base class for statistical features. Aggregate function strings must be
    compatible with `pandas`. Expected dataframe with fixations.

    Args:
        features_stats: Dictionary of format
            {'feature_1': ['statistic_1', 'statistic_2'], ...}.
        x: X coordinate column name.
        y: Y coordinate column name.
        t: timestamp column name.
        duration: duration column name (milliseconds expected).
        dispersion: fixation dispersion column name.
        aoi: Area Of Interest column name(-s). If provided, features
            can be calculated inside the specified AOI.
        calc_without_aoi: if True, then, in addition to AOI-wise features,
            calculate regular features ignoring AOI.
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        warn: whether to enable warnings.
    """

    def __init__(
        self,
        features_stats: Dict[str, List[str]],
        x: str = None,
        y: str = None,
        t: str = None,
        duration: str = None,  # TODO consider units, i.e. ps, ns, ms.
        dispersion: str = None,
        aoi: str | List[str] = None,
        calc_without_aoi: bool = False,
        pk: List[str] = None,
        return_df: bool = True,
        warn: bool = True,
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
        self.features_stats = features_stats
        # feature -- i.e. saccade length/speed
        self.feature_names_in = list(features_stats.keys())
        self.available_feats = None
        self.eps = 1e-20
        self.aoi = aoi
        self.calc_without_aoi = calc_without_aoi
        self.aoi_mapper = None

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
        Method checks that all requested features could be calculated
        with provided data.
        """
        ...

    def _check_features_stats(self):
        """
        Method checks `self.features_stats` for correct feature names (i.e. keys).
        """

        def err_msg(f):
            return (
                f"Feature '{f}' is not supported. Must be one of: "
                f"{', '.join(self.available_feats)}."
            )

        for feat in self.feature_names_in:
            assert feat in self.available_feats, err_msg(feat)

        self._check_params()

    # method called on fit
    def _check_aoi_fit(self, X):
        if self.aoi is not None:  # check if aoi columns contain any NaNs
            assert isinstance(self.aoi, str) or isinstance(
                self.aoi, list
            ), f"`aoi` must be str or List[str], got {type(self.aoi)}."

            if isinstance(self.aoi, str):
                self.aoi = [self.aoi]

            assert (
                "" not in self.aoi
            ), 'Empty string "" as value in `aoi` columns is not allowed.'

            for aoi_col in self.aoi:
                if aoi_col != "":
                    aoi_view = X[aoi_col]
                    if aoi_view.isnull().values.any():
                        raise RuntimeError(
                            f"Passed column '{aoi_col}' for AOI contains NaNs."
                        )

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
            assert (
                "" not in self.aoi
            ), 'Empty string "" as value in `aoi` columns is not allowed.'

            for aoi_col in self.aoi:
                if aoi_col == "":  # lib placeholder
                    continue

                aoi_view = X[aoi_col]
                if aoi_view.isnull().values.any():
                    raise RuntimeError(
                        f"Passed column '{aoi_col}' for AOI contains NaNs."
                    )

                for v in aoi_view:
                    assert (
                        v in self.aoi_mapper[aoi_col]
                    ), f"Unknown AOI value {v} was not seen during `fit` in \
                    '{aoi_col}'."

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
        Method calculates features passed to constructor, i.e. keys of
        `self.features_stats`. In case of `SaccadeFeatures`, it returns
        dictionary `{'length': np.array, 'velocity': np.array, ...}`.
        `transition_mask` is boolean mask of the same shape as X, i-th value
        is False if X's i-th value is first fixation in block. Block is
        defined as sequential fixations in same AOI, maximum by inclusion
        (which means that block cannot contain another block). Thus, each
        AOI is split in blocks and first fixation in each block is then removed.
        """
        ...

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
            all_transition_mask: pd.Series = all_aoi == all_aoi.shift(1)
            transition_mask = all_transition_mask[all_aoi == aoi_val].values

            feats: List[Tuple[str, pd.Series]] = self._calc_feats(
                X_aoi, feat_nms, transition_mask
            )

        return feats

    def fit(self, X: pd.DataFrame, y=None):
        self._check_features_stats()
        self._check_aoi_fit(X)

        # all output features that will appear on transform
        self.feature_names_in_ = []
        for aoi_col in self.aoi_mapper:
            for aoi_val in self.aoi_mapper[aoi_col]:
                for feat_nm in self.features_stats:
                    for stat in self.features_stats[feat_nm]:
                        self.feature_names_in_.append(
                            f"{self._fp}_{feat_nm}_{aoi_col}[{aoi_val}]_{stat}"
                            if aoi_col != ""
                            else f"{self._fp}_{feat_nm}_{stat}"
                        )
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
            gathered_stats_group = []

            for aoi_col in self.aoi_mapper:
                for aoi_val in self.aoi_mapper[aoi_col]:
                    group_feats: List[Tuple[str, pd.Series]] = self._calc_with_aoi(
                        feat_nms, group_X, aoi_col, aoi_val
                    )

                    add_cols_nms = len(group_ids) == 1
                    for feat_nm, feat_arr in group_feats:
                        feat_stats: List[str] = self.features_stats[feat_nm]

                        if not feat_arr.empty:  # group_X with AOI was not empty
                            stats_group = [
                                feat_arr.agg(func=stat) for stat in feat_stats
                            ]

                        else:  # no AOI for given group
                            stats_group = [None for _ in feat_stats]

                        gathered_stats_group.extend(stats_group)
                        # TODO remove, have self.features_names_in_ on fit.
                        # Serves as sanity check for ordering of features names.
                        if add_cols_nms:
                            column_nms.extend(
                                [
                                    (
                                        f"{self._fp}_{feat_nm}_{aoi_col}[{aoi_val}]_{stat}"
                                        if aoi_col != ""
                                        else f"{self._fp}_{feat_nm}_{stat}"
                                    )
                                    for stat in feat_stats
                                ]
                            )

            gathered_stats.append(gathered_stats_group)

        assert len(self.feature_names_in_) == len(column_nms)
        for i in range(len(column_nms)):
            assert (
                self.feature_names_in_[i] == column_nms[i]
            ), f"Fit: {self.feature_names_in_}\nTransform: {column_nms}."
        stats_df = pd.DataFrame(
            data=gathered_stats, columns=column_nms, index=group_ids
        )

        return stats_df if self.return_df else stats_df.values


class SaccadeFeatures(StatsTransformer):
    """Saccade Features Transformer.
    The transformer identifies saccades from fixations and extract
    their features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed", "angle")

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
        dt = (
            _calc_dt(X, self.duration, self.t)
            if any(map(lambda f: f != "length", features))
            else None
        )

        for feat_nm in features:
            if feat_nm == "length":
                sac_len = dr
                feat_arr = sac_len[transition_mask]
            elif feat_nm == "acceleration":
                # Acceleration: dx = v0 * t + 1/2 * a * t^2.
                # Above formula is law of uniformly accelerated motion
                # TODO consider direction
                sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                feat_arr = sac_acc[transition_mask]
            elif feat_nm == "speed":
                sac_spd = dr / (dt + self.eps)
                feat_arr = sac_spd[transition_mask]
            elif feat_nm == "angle":
                angles = pd.Series(
                    [_get_angle(dx_val, dy_val) for dx_val, dy_val in zip(dx, dy)],
                    index=dx.index,
                )
                feat_arr = angles[transition_mask]
            else:
                raise NotImplementedError(feat_nm)
            feats.append((feat_nm, feat_arr))

        return feats


class RegressionFeatures(StatsTransformer):
    """Regression Features Transformer.
    The transformer identifies saccades, and then selects regressions
    from them using user-defined set of rules.

    Args:
        rule: must be either 1) tuple of quadrants direction to classify
            regressions, 1st quadrant being upper-right square of plane and counting
            anti-clockwise or 2) tuple of angles in degrees (0 <= angle <= 360).
            Default: (2, 3) selects left-ward regressions (common in reading).
        deviation: if None, then `rule` is interpreted as quadrants. Otherwise,
            `rule` is interpreted as angles. If integer, then is a +-deviation
            for all angles. If tuple of integers, then must be of the same
            length as `rule`, each value being a corresponding deviation for
            each angle. Angle = 0 is positive x-axis direction,
            rotating anti-clockwise.

    Example:
        Quick start with default parameters::

            from eyefeatures.features.stats import RegressionFeatures

            # Detect left-ward regressions (quadrants 2 and 3)
            transformer = RegressionFeatures(
                features_stats={"length": ["mean", "std"]},
                x="x", y="y", t="time"
            )
            features = transformer.fit_transform(fixations_df)
    """

    def __init__(
        self,
        rule: Tuple[int, ...] = (2, 3),
        deviation: int | Tuple[int, ...] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.available_feats = ("length", "acceleration", "speed", "angle", "mask")
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
        dt = (
            _calc_dt(X, self.duration, self.t)
            if any(map(lambda f: f != "length", features))
            else None
        )

        tm = transition_mask[sm]
        for feat_nm in features:
            if feat_nm == "length":
                sac_len = dr
                feat_arr = sac_len[sm][tm]
            elif feat_nm == "acceleration":
                sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                feat_arr = sac_acc[sm][tm]
            elif feat_nm == "speed":
                sac_spd = dr / (dt + self.eps)
                feat_arr = sac_spd[sm][tm]
            elif feat_nm == "angle":
                angles = pd.Series(
                    [_get_angle(dx_val, dy_val) for dx_val, dy_val in zip(dx, dy)],
                    index=dx.index,
                )
                feat_arr = angles[sm][tm]
            elif feat_nm == "mask":
                feat_arr = pd.Series(sm)
            else:
                raise NotImplementedError(feat_nm)
            feats.append((feat_nm, feat_arr))

        return feats


class MicroSaccadeFeatures(StatsTransformer):
    """Micro Saccade Features.
    The transformer identities saccades, and then selects micro saccades
    from them using user-defined set of rules.

    Args:
        min_dispersion: minimum dispersion of fixation.
        max_speed: maximum speed between fixations.
    """

    def __init__(self, min_dispersion: float, max_speed: float, **kwargs):
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

        dt = (
            _calc_dt(X, self.duration, self.t)
            if any(map(lambda f: f != "length", features))
            else None
        )

        tm = transition_mask[sm]
        for feat_nm in features:
            if feat_nm == "length":
                sac_len = dr
                feat_arr = sac_len[sm][tm]
            elif feat_nm == "acceleration":
                sac_acc: pd.DataFrame = dr / (dt**2 + self.eps) * 1 / 2
                feat_arr = sac_acc[sm][tm]
            elif feat_nm == "speed":
                sac_spd = dr / (dt + self.eps)
                feat_arr = sac_spd[sm][tm]
            elif feat_nm == "mask":
                feat_arr = pd.Series(sm)
            else:
                raise NotImplementedError(feat_nm)
            feats.append((feat_nm, feat_arr))

        return feats


class FixationFeatures(StatsTransformer):
    """Fixation Features Transformer.
    The transformer uses input fixations to extract features.
    """

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

        for feat_nm in features:
            if feat_nm == "duration":
                feat_arr = X[self.duration][transition_mask]
            elif feat_nm == "vad":
                feat_arr = X[self.dispersion][transition_mask]
            else:
                raise NotImplementedError(feat_nm)
            feats.append((feat_nm, feat_arr))

        return feats
