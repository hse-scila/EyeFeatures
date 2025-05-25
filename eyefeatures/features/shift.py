from typing import Dict, List, Tuple, Any

from sklearn.base import TransformerMixin

from eyefeatures.utils import (
    Types,
    _get_id,
    _split_dataframe
)


class IndividualNormalization(TransformerMixin):
    def __init__(
        self,
        pk: List[str] | Tuple[List[str]],
        independent_features: Dict[str, List[str]] | Tuple[Dict[str, List[str]]],
        dependent_features: Dict[str, List[str]] | Tuple[Dict[str, List[str]]],
        inplace: bool = True,
        use_mean: Dict[str, Any] | Dict[List[str], Any] = None,
        use_std:  Dict[str, Any] | Dict[List[str], Any] = None,
        dense_index: bool = True,
        return_df: bool = True,
    ):
        self.pk = pk
        self.ind_features = independent_features
        self.d_features = dependent_features
        self.use_mean = use_mean if use_mean else dict()
        self.use_std  = use_std  if use_std  else dict()
        self.inplace = inplace
        self.dense_index = dense_index
        self.return_df = return_df

        self.features = None
        self.features_stats = None

        self._preprocess_init()
    """
    Normalization of features based on slices, produced by grouping with primary key.
    :param pk: primary key to use for grouping.
    :param independent_features: Dict, features to normalize without fitting statistics (not used
        on fit).
    :param dependent_features: Dict, features to normalize with fitting statistics (calculated
        on fit).
    :param use_mean: Dict | None, means to use for normalization. Its keys must be same as
        `eyefeatures.utils._get_id` output.
    :param use_std: Dict | None, stds to use for normalization. Its keys must be same as
        `eyefeatures.utils._get_id` output.
    :param inplace: bool, if true, then provided `features` are normalized inplace, otherwise
        new columns are created.
    :param return_df: bool, it true then pandas DataFrame is returned, else numpy ndarray.
    """

    def _preprocess_init(self):
        args = [self.ind_features, self.d_features]
        is_all_multiple = all([isinstance(x, tuple) for x in args] + [isinstance(self.pk, tuple)])
        is_all_single = all([isinstance(x, dict) for x in args] + [isinstance(self.pk, list)])
        assert is_all_single or is_all_multiple, ("Must be one of two cases:\n"
                                                  "1. pk is list of columns and features are dicts.\n"
                                                  "2. pk is tuple of lists of columns and features"
                                                  "are tuples of dicts.")

        if is_all_single:
            self.pk = (self.pk,)
            self.d_features = (self.d_features,)
            self.ind_features = (self.ind_features,)

        for i in range(len(self.pk)):
            assert isinstance(self.ind_features[i], dict), "Wrong value for `independent_features`."
            assert isinstance(self.d_features[i], dict), "Wrong value for `dependent_features`."
            assert isinstance(self.pk[i], list), "Wrong value for `pk`."

        self.features = []
        for i in range(len(self.pk)):
            features = {}
            features.update(self.d_features[i])
            features.update(self.ind_features[i])
            self.features.append(features)
        self.features = tuple(self.features)

    def _is_dependent_feat(self, feat_nm, feat_stat):
        return feat_nm in self.d_features and feat_stat in self.d_features[feat_nm]

    def fit(self, X, y=None):
        self.features_stats = []

        for features, pk in zip(self.d_features, self.pk):
            features_stats = {}
            feat_nms = list(features.keys())  # names of features
            groups: Types.EncodedPartition = _split_dataframe(
                X, pk, encode=True
            )
            for group in groups:
                group_id = group[0]
                features_stats[group_id] = {}

            input_feat_nms = X.columns
            for input_feat_nm in input_feat_nms:
                for feat_nm in feat_nms:
                    feat_stats = features[feat_nm]
                    for feat_stat in feat_stats:
                        if feat_nm in input_feat_nm and feat_stat in input_feat_nm:
                            for group_id, group_X in groups:
                                mean = self.use_mean.get(group_id, group_X[input_feat_nm].mean())
                                std  = self.use_std.get(group_id, group_X[input_feat_nm].std())
                                features_stats[group_id][input_feat_nm] = {'mean': mean, 'std': std}
            self.features_stats.append(features_stats)

        return self

    def transform(self, X, y=None):
        for features, pk, features_stats in zip(self.features, self.pk, self.features_stats):
            feat_nms = list(features.keys())  # names of features
            groups: Types.EncodedPartition = _split_dataframe(
                X, pk, encode=True
            )

            X['_group_id'] = [_get_id(index) for index in X[pk].values]
            input_feat_nms = X.columns
            for input_feat_nm in input_feat_nms:
                for feat_nm in feat_nms:
                    feat_stats = features[feat_nm]
                    for feat_stat in feat_stats:
                        if feat_nm in input_feat_nm and feat_stat in input_feat_nm:
                            if self.inplace:
                                sf = input_feat_nm
                            else:
                                sf = f"{input_feat_nm}_norm"
                                X[sf] = X[input_feat_nm].copy()
                            for group_id, group_X in groups:
                                group_feat_nms = features_stats.get(group_id, [])
                                if input_feat_nm in group_feat_nms:
                                    mean = group_feat_nms[input_feat_nm]['mean']
                                    std  = group_feat_nms[input_feat_nm]['std']
                                else:
                                    mean = group_X[input_feat_nm].mean()
                                    std  = group_X[input_feat_nm].std()

                                mask = X['_group_id'] == group_id
                                X.loc[mask, sf] -= mean
                                X.loc[mask, sf] /= std

            X.drop(['_group_id'], axis=1, inplace=True)
        return X if self.return_df else X.values
