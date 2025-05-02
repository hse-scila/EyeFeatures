from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Any, Literal
import warnings

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray

from eyefeatures.utils import (
    Types
)


class IndividualNormalization:
    def __init__(
        self,
        pk: List[str] | Tuple[List[str]],
        features: Dict[str, List[str]] | Tuple[Dict[str, List[str]]],
        inplace: bool = True,
        use_mean: Dict[str, Any] | Dict[List[str], Any] = None,
        use_std:  Dict[str, Any] | Dict[List[str], Any] = None,
        dense_index: bool = True,
        return_df: bool = True,
    ):
        self.pk = pk
        self.features = features
        self.use_mean = use_mean if use_mean else dict()
        self.use_std  = use_std  if use_std  else dict()
        self.inplace = inplace
        self.dense_index = dense_index
        self.return_df = return_df

        self._preprocess_init()

    def _preprocess_init(self):
        if isinstance(self.pk, list) and isinstance(self.features, dict):
            self.pk = (self.pk,)
            self.features = (self.features,)

        elif isinstance(self.pk, tuple) and isinstance(self.features, dict):
            # same features for different pk
            self.features = (
                self.features for _ in range(len(self.pk))
            )

        elif isinstance(self.pk, list) and isinstance(self.features, tuple):
            raise ValueError(
                "Several `features` for single `pk` are"
                "not allowed - just merge `features` into"
                "single dict."
            )

        elif isinstance(self.pk, tuple) and isinstance(self.features, tuple):
            # different shift_features for different shift_pk
            assert len(self.features) == len(self.pk),\
                f"""If tuple of lists, length of `features`
                    ({len(self.features)}) must correspond
                    to length of `pk` ({len(self.pk)})."""

        else:
            raise ValueError(f"Wrong combination of types for `features` and `pk`."
                             f"Read docs for an example.")

        # sf = Tuple[Dict], sp = Tuple[List]
        for i in range(len(self.pk)):
            assert isinstance(self.features[i], dict), "Wrong value for `features`."
            assert isinstance(self.pk[i], list), "Wrong value for `pk`."


    def fit(self, X, y=None):
        for features, pk in zip(self.features, self.pk):
            feat_nms = list(features.keys())  # names of features
            group_ids: Types.Partition = list(X[pk].groupby(by=pk).index)

            input_feat_nms = X.columns
            for input_feat_nm in input_feat_nms:
                for feat_nm in feat_nms:
                    feat_stat = features[feat_nm]
                    if feat_nm in input_feat_nm and feat_stat in input_feat_nm:
                        if self.inplace:
                            sf = input_feat_nm
                        else:
                            sf = f"{input_feat_nm}_norm"
                            X[sf] = X[input_feat_nm].copy()
                        for group_id in group_ids:
                            mean = self.use_mean.get(group_id, X[group_id, input_feat_nm].mean())
                            std  = self.use_std.get(group_id, X[group_id, input_feat_nm].std())
                            X[group_id, sf] -= mean
                            X[group_id, sf] /= std

            return X if self.return_df else X.values
