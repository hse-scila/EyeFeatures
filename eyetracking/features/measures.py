from typing import List, Literal, Union

import numpy as np
import pandas as pd
from numba import jit

from eyetracking.features.extractor import BaseTransformer
from eyetracking.utils import _split_dataframe


class HurstExponent(BaseTransformer):
    def __init__(
        self,
        n_iters=10,
        fill_strategy: Literal["mean", "reduce", "last"] = "last",
        var: str = None,
        pk: List[str] = None,
        eps: float = 1e-22,
        return_df: bool = True,
    ):
        """
        Approximates Hurst Exponent using R/S analysis.
        https://en.wikipedia.org/wiki/Hurst_exponent
        :param n_iters: number of iterations to complete. Note: data must be of length more than 2 ^ `n_iters`.
        :param fill_strategy: how to make vector be length of power of 2. If "reduce", then all values
                after 2 ^ k-th are removed, where n < 2 ^ (k + 1). Other strategies specify the value
                to fill the vector with up to the closest power of 2, "mean" being the mean of vector, "last"
                being the last value of vector (makes time-series constant at the end).
        :param var: column name of sequence points.
        :param pk: list of column names used to split pd.DataFrame.
        :param eps: division epsilon.
        :param return_df: Return pd.Dataframe object else np.ndarray.
        """
        super().__init__(pk=pk, return_df=return_df)
        self.var = var
        self.n_iters = n_iters
        self.fill_strategy = fill_strategy
        self.eps = eps

    def _check_init(self, X_len: int):
        assert (
            isinstance(self.n_iters, int) and self.n_iters > 0
        ), "Error: 'n_iters' must be positive integer."
        assert (
            X_len > 2**self.n_iters
        ), "Error: data must be of length more than 2 ^ `n_iters`."

        fill_strategies = ("reduce", "mean", "last")
        assert self.fill_strategy in fill_strategies, (
            f"Error: 'fill_strategy' must be one of " f"{','.join(fill_strategies)}."
        )

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
    def _make_pow2(self, x: np.array) -> np.array:
        n = len(x)
        k = np.log2(len(x)).astype(np.int32)  # 2 ^ k <= n < 2 ^ (k + 1)

        # makes n a power of 2
        if self.fill_strategy == "reduce":
            return x[: 2**k]

        elif self.fill_strategy == "mean":
            tail = np.zeros(2 ** (k + 1) - n)
            tail[:] = x.mean()
            return np.hstack([x, tail])

        elif self.fill_strategy == "last":
            tail = np.zeros(2 ** (k + 1) - n)
            tail[:] = x[-1]
            return np.hstack([x, tail])

        else:
            raise NotImplementedError

    @jit(forceobj=True, looplift=True)
    def _compute_hurst(self, x):
        x = self._make_pow2(x)
        n = len(x)

        cnt = 0
        rs = np.zeros(self.n_iters)  # range to std ratios
        bs = np.zeros(self.n_iters)  # block sizes
        while cnt < self.n_iters and n > 2:
            bs[cnt] = n

            blocks = x.copy().reshape(-1, n)  # partition in blocks of size n
            blocks -= blocks.mean(axis=1)[:, None]  # center blocks individually

            stds = np.sqrt(np.square(blocks).sum(axis=1) / n)  # stds of blocks
            blocks = blocks.cumsum(axis=1)  # cumulative sums of blocks
            ranges = blocks.max(axis=1) - blocks.min(axis=1)  # ranges
            ratio = (ranges / (stds + self.eps)).mean()
            rs[cnt] = ratio

            n //= 2
            cnt += 1

        # OLS
        rs = rs[:cnt]
        bs = bs[:cnt]
        bs = np.vstack([np.ones(cnt), bs]).T
        grad = (np.linalg.inv(bs.T @ bs) @ bs.T) @ np.log(rs)

        return grad[1]

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=len(X))

        x = X[self.var].values / 1000

        features_names = [f"he_{self.var}"]

        if self.pk is None:
            grad = self._compute_hurst(x)
            gathered_features = [[grad]]
        else:
            groups = X[self.pk].drop_duplicates().values
            gathered_features = []
            for group in groups:
                current_X = X[pd.DataFrame(X[self.pk] == group).all(axis=1)]
                x = current_X[self.var].values / 1000
                grad = self._compute_hurst(x.copy())
                gathered_features.append([grad])

        features_df = pd.DataFrame(data=gathered_features, columns=features_names)
        return features_df if self.return_df else features_df.values

class Entropy(BaseTransformer):
    def __init__(
        self,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert (self.aoi is not None), "Error: Provide aoi column"
        assert (X_len != 0), "Error: there are no fixations"

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        features_names = ["entropy"]

        if self.pk is None:
            X_splited = _split_dataframe(X, [self.aoi])
            all_fix = X.shape[0]
            aoi_probability = []
            for group, current_X in X_splited:
                aoi_probability.append(current_X.shape[0] / all_fix)

            entropy = 0
            for p in aoi_probability:
                entropy -= p * np.log2(p)

            gathered_features = [[entropy]]
        else:
            X_splited = _split_dataframe(X, self.pk)
            gathered_features = []
            for group, current_X in X_splited:
                all_fix = current_X.shape[0]
                aoi_probability = []
                X_aoi = _split_dataframe(current_X, [self.aoi])
                for aoi_group, current_aoi_X in X_aoi:
                    aoi_probability.append(current_aoi_X.shape[0] / all_fix)

                entropy = 0
                for p in aoi_probability:
                    entropy -= p * np.log2(p)

                gathered_features.append([[entropy]])

        features_df = pd.DataFrame(data=gathered_features, columns=features_names)
        return features_df if self.return_df else features_df.values
