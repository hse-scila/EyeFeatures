from typing import List, Literal, Union

import numpy as np
import pandas as pd
from numba import jit

from scipy import ifft
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

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


class ShannonEntropy(BaseTransformer):
    def __init__(
        self,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

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


class SpectralEntropy(BaseTransformer):
    def __init__(
        self,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
    def spectral_entropy(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        transformed_seq = ifft(X[coords].values)
        power_spectrum_seq = np.linalg.norm(transformed_seq, axis=1) ** 2
        proba_distribution = power_spectrum_seq / np.sum(power_spectrum_seq)
        return entropy(proba_distribution)

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("spec_ent")
            gathered_features.append([self.spectral_entropy(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"spec_ent_{str(group)}")
                gathered_features.append([self.spectral_entropy(current_X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class FuzzyEntropy(BaseTransformer):
    """
    :param m: embedding dimension
    :param r: tolerance threshold for matches acceptance (usually std)
    """

    def __init__(
        self,
        m: int = 2,
        r: float = 0.2,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.m = m
        self.r = r
        self.aoi = aoi
        self.eps = 1e-7

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def fuzzy_entropy(self, X: pd.DataFrame) -> float:
        n = 2 * len(X)
        phi_m = np.zeros(2)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        for i in range(2):
            X_emb = np.array(
                [X_coord[j : j + self.m + i] for j in range(n - self.m - i)]
            )
            dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
            phi_m[i] = np.sum(np.exp(-(dist_matrix**2) / (2 * self.r**2))) / (
                n - self.m - i
            )

        return np.log(phi_m[0] / (phi_m[1] + self.eps))

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("fuzzy_ent")
            gathered_features.append([self.fuzzy_entropy(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"fuzzy_ent_{str(group)}")
                gathered_features.append([self.fuzzy_entropy(X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class SampleEntropy(BaseTransformer):
    """
    :param m: embedding dimension
    :param r: tolerance threshold for matches acceptance (usually std)
    """

    def __init__(
        self,
        m: int = 2,
        r: float = 0.2,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.m = m
        self.r = r
        self.aoi = aoi
        self.eps = 1e-7

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def sample_entropy(self, X: pd.DataFrame) -> float:
        n = 2 * len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = np.array([X_coord[j : j + self.m] for j in range(n - self.m + 1)])
        dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
        B = np.sum(np.sum(dist_matrix < self.r, axis=0) - 1)
        X_emb = np.array([X_coord[j : j + self.m + 1] for j in range(n - self.m)])
        dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
        A = np.sum(np.sum(dist_matrix < self.r, axis=0) - 1)
        return -np.log(A / (B + self.eps))

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("sample_ent")
            gathered_features.append([self.sample_entropy(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"sample_ent_{str(group)}")
                gathered_features.append([self.sample_entropy(X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class IncrementalEntropy(BaseTransformer):
    def __init__(self, aoi: str = None, pk: List[str] = None, return_df: bool = True):
        super().__init__(pk=pk, return_df=return_df)
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def incremental_entropy(self, X: pd.DataFrame) -> float:
        n = len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        incremental_entropies = np.zeros(n)
        for i in range(1, n):
            hist, _ = np.histogram(X_coord[: i + 1], bins="auto", density=True)
            hist = hist[hist > 0]
            incremental_entropies[i] = -np.sum(hist * np.log(hist))

        return incremental_entropies.mean()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("inc_ent")
            gathered_features.append([self.incremental_entropy(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"inc_ent_{str(group)}")
                gathered_features.append([self.incremental_entropy(X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class GriddedDistributionEntropy(BaseTransformer):
    """
    :param grid_size: the number of bins (grid cells) for creating the histogram
    """

    def __init__(
        self,
        grid_size: int = 10,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.grid_size = grid_size
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def gridded_distribution_entropy(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values
        H, edges = np.histogramdd(X_coord, bins=self.grid_size)
        P = H / np.sum(H)
        P = P[P > 0]
        return -np.sum(P * np.log(P))

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("grid_ent")
            gathered_features.append([self.gridded_distribution_entropy(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"grid_ent_{str(group)}")
                gathered_features.append([self.gridded_distribution_entropy(X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class PhaseEntropy(BaseTransformer):
    """
    :param m: embedding dimension
    :param tau: time delay for phase space reconstruction, the lag between each point in the phase space vectors.
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.m = m
        self.tau = tau
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def phase_entropy(self, X: pd.DataFrame) -> float:
        n = 2 * len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = np.array(
            [
                X_coord[j : j + self.m * self.tau : self.tau]
                for j in range(n - (self.m - 1) * self.tau)
            ]
        )
        dist_matrix = squareform(pdist(X_emb, metric="euclidean"))
        prob_dist = np.histogram(dist_matrix, bins="auto", density=True)[0]
        prob_dist = prob_dist[prob_dist > 0]
        return -np.sum(prob_dist * np.log(prob_dist))

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("phase_ent")
            gathered_features.append([self.phase_entropy(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"phase_ent_{str(group)}")
                gathered_features.append([self.phase_entropy(X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class LyapunovExponent(BaseTransformer):
    """
    :param m: embedding dimension
    :param tau: time delay for phase space reconstruction
    :param T: time steps to average the divergence over
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        T: int = 1,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.m = m
        self.tau = tau
        self.T = T
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def embed(self, X):
        N = len(X)
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(N - (self.m - 1) * self.tau)
            ]
        )

    def largest_lyapunov_exponent(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = self.embed(X_coord)
        n = len(X_emb)
        divergence = []

        for i in range(n):
            dist_matrix = squareform(pdist(X_emb, metric="euclidean"))
            np.fill_diagonal(dist_matrix, np.inf)
            nearest_index = np.argmin(dist_matrix[i])
            distances = []

            for t in range(1, self.T + 1):
                if i + t < n and nearest_index + t < n:
                    distance = np.linalg.norm(X_emb[i + t] - X_emb[nearest_index + t])
                    if distance != 0:
                        distances.append(np.log(distance))

            if distances:
                divergence.append(np.mean(distances))

        return np.mean(divergence) / self.T if divergence else np.inf

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("lyap_exp")
            gathered_features.append([self.largest_lyapunov_exponent(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"lyap_exp_{str(group)}")
                gathered_features.append([self.largest_lyapunov_exponent(X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values
