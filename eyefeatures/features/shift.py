from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from eyefeatures.utils import Types, _get_id, _split_dataframe


class IndividualNormalization(TransformerMixin):
    """Normalization of features based on slices, produced by grouping with primary key.

    If `dependent_features` and `independent_features` are both None (default), the
    transformer will automatically discover all numeric columns in the input DataFrame
    (excluding columns in `pk`) and normalize them during `fit`.

    Args:
        pk: primary key to use for grouping.
        independent_features: features to normalize without fitting statistics (not used
            on fit). Can be a Dict[str, List[str]], a simple List[str], or None.
        dependent_features: features to normalize with fitting statistics (calculated
            on fit). Can be a Dict[str, List[str]], a simple List[str], or None.
        use_mean: means to use for normalization. Its keys must be same as
            `eyefeatures.utils._get_id` output.
        use_std: standard deviations to use for normalization. Its keys must be same as
            `eyefeatures.utils._get_id` output.
        inplace: if true, then provided `features` are normalized inplace, otherwise
            new columns are created.
        dense_index: if true, then grouping keys are converted to strings.
        return_df: whether to return output as DataFrame or numpy array.
    """

    def __init__(
        self,
        pk: list[str] | tuple[list[str]],
        independent_features: (
            dict[str, list[str]] | tuple[dict[str, list[str]]] | list[str]
        ) = None,
        dependent_features: (
            dict[str, list[str]] | tuple[dict[str, list[str]]] | list[str]
        ) = None,
        inplace: bool = True,
        use_mean: dict[str, Any] | dict[list[str], Any] = None,
        use_std: dict[str, Any] | dict[list[str], Any] = None,
        dense_index: bool = True,
        return_df: bool = True,
    ):
        self.return_df = return_df

        self.pk = (pk,) if isinstance(pk, list) else pk
        self.ind_features = independent_features
        self.d_features = dependent_features
        self.use_mean = use_mean if use_mean else {}
        self.use_std = use_std if use_std else {}
        self.inplace = inplace
        self.dense_index = dense_index

        # Will be populated in fit if None
        self._auto_discover = self.d_features is None and self.ind_features is None

        self.features = None
        self.features_stats = None

        self._preprocess_init()

    def _preprocess_init(self):
        # Handle simple list case for features
        if isinstance(self.ind_features, list):
            self.ind_features = ({feat: [feat] for feat in self.ind_features},)
        elif self.ind_features is None:
            self.ind_features = ({},)
        elif isinstance(self.ind_features, dict):
            self.ind_features = (self.ind_features,)

        if isinstance(self.d_features, list):
            self.d_features = ({feat: [feat] for feat in self.d_features},)
        elif self.d_features is None:
            self.d_features = ({},)
        elif isinstance(self.d_features, dict):
            self.d_features = (self.d_features,)

        self.features = []
        for i in range(len(self.pk)):
            features = {}
            if i < len(self.d_features):
                features.update(self.d_features[i])
            if i < len(self.ind_features):
                features.update(self.ind_features[i])
            self.features.append(features)
        self.features = tuple(self.features)

    def _is_dependent_feat(self, feat_nm, feat_stat):
        return feat_nm in self.d_features and feat_stat in self.d_features[feat_nm]

    def fit(self, X, y=None):
        if self._auto_discover:
            # Discover numeric columns except pk
            all_pk_cols = [col for p in self.pk for col in p]
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            discovered = [col for col in numeric_cols if col not in all_pk_cols]
            self.d_features = ({feat: [feat] for feat in discovered},)
            self._preprocess_init()

        self.features_stats = []

        for features, pk in zip(self.d_features, self.pk, strict=False):
            features_stats = {}
            feat_nms = list(features.keys())  # names of features
            groups: Types.EncodedPartition = _split_dataframe(X, pk, encode=True)
            for group in groups:
                group_id = group[0]
                features_stats[group_id] = {}

            input_feat_nms = X.columns
            for input_feat_nm in input_feat_nms:
                for feat_nm in feat_nms:
                    feat_stats = features[feat_nm]
                    for feat_stat in feat_stats:
                        # Both prefix 'feat_nm' and suffix 'feat_stat' must be present
                        # in feature name
                        if (
                            feat_nm in input_feat_nm
                            and feat_stat in input_feat_nm
                            and not input_feat_nm.endswith("_norm")
                        ):
                            for group_id, group_X in groups:
                                mean = self.use_mean.get(
                                    group_id, group_X[input_feat_nm].mean()
                                )
                                std = self.use_std.get(
                                    group_id, group_X[input_feat_nm].std()
                                )
                                features_stats[group_id][input_feat_nm] = {
                                    "mean": mean,
                                    "std": std,
                                }
            self.features_stats.append(features_stats)

        return self

    def transform(self, X, y=None):
        for features, pk, features_stats in zip(
            self.features, self.pk, self.features_stats, strict=False
        ):
            feat_nms = list(features.keys())  # names of features
            groups: Types.EncodedPartition = _split_dataframe(X, pk, encode=True)

            X["_group_id"] = [_get_id(index) for index in X[pk].values]
            input_feat_nms = X.columns
            for input_feat_nm in input_feat_nms:
                for feat_nm in feat_nms:
                    feat_stats = features[feat_nm]
                    for feat_stat in feat_stats:
                        # Both prefix 'feat_nm' and suffix 'feat_stat' must be present
                        # in feature name
                        if (
                            feat_nm in input_feat_nm
                            and feat_stat in input_feat_nm
                            and not input_feat_nm.endswith("_norm")
                        ):
                            if self.inplace:
                                sf = input_feat_nm
                            else:
                                sf = f"{input_feat_nm}_norm"
                                X[sf] = X[input_feat_nm].copy()
                            for group_id, group_X in groups:
                                group_feat_nms = features_stats.get(group_id, [])
                                if input_feat_nm in group_feat_nms:
                                    mean = group_feat_nms[input_feat_nm]["mean"]
                                    std = group_feat_nms[input_feat_nm]["std"]
                                else:
                                    mean = group_X[input_feat_nm].mean()
                                    std = group_X[input_feat_nm].std()

                                mask = X["_group_id"] == group_id
                                X.loc[mask, sf] -= mean
                                X.loc[mask, sf] /= std

            X.drop(["_group_id"], axis=1, inplace=True)
        return X if self.return_df else X.values
