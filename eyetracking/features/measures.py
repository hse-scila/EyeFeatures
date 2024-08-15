from typing import Callable, List, Literal, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy.fftpack import ifft
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import entropy

from eyetracking.features.complex import get_rqa
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

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
    def gridded_distribution_entropy(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values
        H, edges = np.histogramdd(X_coord, bins=self.grid_size)
        P = H / np.sum(H)
        P = P[P > 0]
        return -np.sum(P * np.log(P))

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
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

    @jit(forceobj=True, looplift=True)
    def build_embedding(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(len(X) - (self.m - 1) * self.tau)
            ]
        )

    @jit(forceobj=True, looplift=True)
    def largest_lyapunov_exponent(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = self.build_embedding(X_coord)
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

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
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


class FractalDimension(BaseTransformer):
    """
    :param m: embedding dimension
    :param tau: time delay for phase space reconstruction
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

    @jit(forceobj=True, looplift=True)
    def build_embedding(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(len(X) - (self.m - 1) * self.tau)
            ]
        )

    @jit(forceobj=True, looplift=True)
    def box_counting_dimension(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = self.build_embedding(X_coord)

        min_val, max_val = np.min(X_emb), np.max(X_emb)
        box_sizes = np.logspace(np.log10(1), np.log10(max_val - min_val), num=20)
        counts = []

        for box_size in box_sizes:
            grid = (X_emb - min_val) // box_size
            unique_boxes = np.unique(grid, axis=0)
            counts.append(len(unique_boxes))

        coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
        return -coeffs[0]

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("fractal_dim")
            gathered_features.append([self.box_counting_dimension(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"fractal_dim_{str(group)}")
                gathered_features.append([self.box_counting_dimension(current_X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class CorrelationDimension(BaseTransformer):
    """
    :param m: embedding dimension
    :param tau: time delay for phase space reconstruction
    :param r: radius threshold for correlation sum
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        r: float = 0.5,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.m = m
        self.tau = tau
        self.r = r
        self.aoi = aoi
        self.eps = 1e-7

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    @jit(forceobj=True, looplift=True)
    def build_embedding(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(len(X) - (self.m - 1) * self.tau)
            ]
        )

    @jit(forceobj=True, looplift=True)
    def correlation_dimension(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = self.build_embedding(X_coord)

        dist_matrix = squareform(pdist(X_emb, metric="euclidean"))
        count = np.sum(dist_matrix < self.r) - len(dist_matrix)

        corr_dim = np.log(self.eps + count / len(dist_matrix)) / np.log(self.r)
        return corr_dim

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            columns_names.append("corr_dim")
            gathered_features.append([self.correlation_dimension(X)])
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                columns_names.append(f"corr_dim_{str(group)}")
                gathered_features.append([self.correlation_dimension(current_X)])

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values


class RQAMeasures(BaseTransformer):
    """
    Calculates Reccurence (REC), Determinism (DET), Laminarity (LAM) and Center of Recurrence Mass (CORM) measures.
    :param metric: callable metric on R^2 points
    :param rho: threshold radius for RQA matrix
    :param min_length: min length of lines
    :param measures: list of measure to calculate (corresponding str)
    """

    def __init__(
        self,
        metric: Callable = euclidean,
        rho: float = 1e-1,
        min_length: float = 1e-1,
        measures: List[str] = ["rec", "det", "lam", "corm"],
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.metric = metric
        self.rho = rho
        self.min_length = min_length
        self.measures = measures
        self.aoi = aoi
        self.eps = 1e-7

    def _check_init(self, X_len: int):
        assert len(self.measures) > 0, "Error: at least one measure must be passed"
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    @jit(forceobj=True, looplift=True)
    def fit(self, X: pd.DataFrame, y=None):
        return self

    @jit(forceobj=True, looplift=True)
    def calculate_measures(self, X: pd.DataFrame) -> List[float]:
        columns, features = [], []
        rqa_matrix = get_rqa(X, self.x, self.y, self.metric, self.rho)
        n = rqa_matrix.shape[0]
        r = np.sum(np.triu(rqa_matrix, k=1)) + self.eps

        if "rec" in self.measures:
            columns.append("rec")
            features.append(100 * 2 * r / (n * (n - 1)))

        if "det" in self.measures:
            DL = []
            for offset in range(1, n):
                diagonal = np.diag(rqa_matrix, k=offset)
                if len(diagonal) >= self.min_length:
                    for k in range(len(diagonal) - self.min_length + 1):
                        if np.all(diagonal[k : k + self.min_length]):
                            DL.append(np.sum(diagonal[k:]))
            columns.append("det")
            features.append(100 * np.sum(DL) / r)

        if "lam" in self.measures:
            HL, VL = [], []
            for i in range(n):
                horizontal = rqa_matrix[i, :]
                vertical = rqa_matrix[:, i]
                for k in range(n - self.min_length + 1):
                    if np.all(horizontal[k : k + self.min_length]):
                        HL.append(np.sum(horizontal[k:]))
                    if np.all(vertical[k : k + self.min_length]):
                        VL.append(np.sum(vertical[k:]))
            columns.append("lam")
            features.append(100 * (np.sum(HL) + np.sum(VL)) / (2 * r))

        if "corm" in self.measures:
            corm_num = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    corm_num += (j - i) * rqa_matrix[i, j]
            columns.append("corm")
            features.append(100 * corm_num / ((n - 1) * r))

        return columns, features

    @jit(forceobj=True, looplift=True)
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []

        if self.pk is None:
            cur_names, cur_features = self.calculate_measures(X)
            columns_names.extend(cur_names)
            gathered_features.extend(cur_features)
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                gnm = str(group)
                cur_names, cur_features = self.calculate_measures(current_X)
                for i in range(len(cur_names)):
                    cur_names[i] += f"_{gnm}"
                columns_names.extend(cur_names)
                gathered_features.extend(cur_features)

        features_df = pd.DataFrame(data=gathered_features, columns=columns_names)
        return features_df if self.return_df else features_df.values
