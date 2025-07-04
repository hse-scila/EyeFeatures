from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Literal, Union

import numpy as np
import pandas as pd
from numba import jit
from scipy.fftpack import fft2, ifft
from scipy.spatial.distance import euclidean, pdist, squareform
from scipy.stats import entropy, kurtosis, skew

from eyefeatures.features.complex import get_rqa, hilbert_huang_transform
from eyefeatures.features.extractor import BaseTransformer
from eyefeatures.utils import _split_dataframe


class MeasureTransformer(ABC, BaseTransformer):
    """Base Transformer class for measures.

    Args:
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area of Interest column name.
        aoi: Area Of Interest column name(-s). If provided, features can be calculated inside
            the specified AOI.
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        feature_name: Column name for resulting feature.
    """
    def __init__(
        self,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
        feature_name: str = "feature",
    ):
        super().__init__(x=x, y=y, pk=pk, return_df=return_df)
        self.aoi = aoi
        self.feature_name = feature_name

    def _check_init(self, X_len: int):
        assert X_len != 0, "Error: there are no fixations"

    @abstractmethod
    def calculate_feature(self, X: pd.DataFrame) -> float:
        raise NotImplementedError("Abstract method called")

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        group_names = []
        gathered_features = []
        columns_names = [self.feature_name]

        if self.pk is None:
            group_names.append("all")
            gathered_features.append([self.calculate_feature(X)])
        else:
            X_split = _split_dataframe(X, self.pk)
            for group, current_X in X_split:
                group_names.append(group)
                gathered_features.append([self.calculate_feature(current_X)])

        features_df = pd.DataFrame(
            data=gathered_features, columns=columns_names, index=group_names
        )
        return features_df if self.return_df else features_df.values


class HurstExponent(MeasureTransformer):
    """Approximates Hurst Exponent using R/S analysis.

    Args:
        n_iters: number of iterations to complete. Note: data must be of length more than :math:`2^{n\_iters}`.
        fill_strategy: how to make vector be length of power of :math:`2`. If "reduce", then all values
            after :math:`2^k`-th are removed, where :math:`n < 2^{(k + 1)}`. Other strategies specify the value
            to fill the vector with up to the closest power of :math:`2`, "mean" being the mean of vector, "last"
            being the last value of vector (makes time-series constant at the end).
        pk: list of column names used to split pd.DataFrame.
        eps: division epsilon.
        return_df: Return pd.Dataframe object else np.ndarray.
    """
    def __init__(
        self,
        n_iters=10,
        fill_strategy: Literal["mean", "reduce", "last"] = "last",
        pk: List[str] = None,
        eps: float = 1e-22,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df, feature_name="hurst_exponent")
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

    def calculate_feature(self, X: pd.DataFrame) -> float:
        x = X[self.x].values / 1000  # TODO make in 2D
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


class ShannonEntropy(BaseTransformer):
    """Shannon Entropy."""
    def __init__(
        self,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(pk=pk, return_df=return_df)
        self.aoi = aoi
        self.feature_name = f"shannon_entropy_aoi_{aoi}"

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        features_names = ["entropy"]
        group_names = []

        if self.pk is None:
            group_names.append("all")
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
                group_names.append(group)
                all_fix = current_X.shape[0]
                aoi_probability = []
                X_aoi = _split_dataframe(current_X, [self.aoi])
                for aoi_group, current_aoi_X in X_aoi:
                    aoi_probability.append(current_aoi_X.shape[0] / all_fix)

                entropy = 0
                for p in aoi_probability:
                    entropy -= p * np.log2(p)

                gathered_features.append([entropy])

        features_df = pd.DataFrame(
            data=gathered_features, columns=features_names, index=group_names
        )
        return features_df if self.return_df else features_df.values


class SpectralEntropy(MeasureTransformer):
    """Spectral Entropy."""
    def __init__(
        self,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x, y=y, pk=pk, return_df=return_df, feature_name="spectral_entropy"
        )
        self.aoi = aoi

    def calculate_feature(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        transformed_seq = ifft(X[coords].values)
        power_spectrum_seq = np.linalg.norm(transformed_seq, axis=1) ** 2
        proba_distribution = power_spectrum_seq / np.sum(power_spectrum_seq)
        return entropy(proba_distribution)


class FuzzyEntropy(MeasureTransformer):
    """Fuzzy Entropy.

    Args:
        m: embedding dimension
        r: tolerance threshold for matches acceptance (usually std)
    """
    def __init__(
        self,
        x: str = None,
        y: str = None,
        m: int = 2,
        r: float = 0.2,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x, y=y, pk=pk, return_df=return_df, feature_name=f"fuzzy_m_{m}_r_{r}"
        )
        self.m = m
        self.r = r
        self.aoi = aoi
        self.eps = 1e-7

    def calculate_feature(self, X: pd.DataFrame) -> float:
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


class SampleEntropy(MeasureTransformer):
    """Sample Entropy.

    Args:
        m: embedding dimension
        r: tolerance threshold for matches acceptance (usually std)
    """
    def __init__(
        self,
        m: int = 2,
        r: float = 0.2,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"sample_entropy_m={m}_r={r}",
        )
        self.m = m
        self.r = r
        self.aoi = aoi
        self.eps = 1e-6

    def calculate_feature(self, X: pd.DataFrame) -> float:
        n = 2 * len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = np.array([X_coord[j : j + self.m] for j in range(n - self.m + 1)])
        dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
        B = np.sum(np.sum(dist_matrix < self.r, axis=0) - 1)
        X_emb = np.array([X_coord[j : j + self.m + 1] for j in range(n - self.m)])
        dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
        A = np.sum(np.sum(dist_matrix < self.r, axis=0) - 1)
        return -np.log(A / (B + self.eps) + 1e-100)


class IncrementalEntropy(MeasureTransformer):
    """Incremental Entropy."""
    def __init__(
        self,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x, y=y, pk=pk, return_df=return_df, feature_name=f"incremental_entropy"
        )
        self.aoi = aoi

    def calculate_feature(self, X: pd.DataFrame) -> float:
        n = len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        incremental_entropies = np.zeros(n)
        for i in range(1, n):
            hist, _ = np.histogram(X_coord[: i + 1], bins="auto", density=True)
            hist = hist[hist > 0]
            incremental_entropies[i] = -np.sum(hist * np.log(hist))

        return incremental_entropies.mean()


class GriddedDistributionEntropy(MeasureTransformer):
    """Gridded Distrbituion Entropy.

    Args:
        grid_size: the number of bins (grid cells) for creating the histogram
    """
    def __init__(
        self,
        grid_size: int = 10,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"gridded_entropy_grid_size_{grid_size}",
        )
        self.grid_size = grid_size
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert X_len != 0, "Error: there are no fixations"

    def calculate_feature(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values
        H, edges = np.histogramdd(X_coord, bins=self.grid_size)
        P = H / np.sum(H)
        P = P[P > 0]
        return -np.sum(P * np.log(P))


class PhaseEntropy(MeasureTransformer):
    """Phase Entropy.

    Args:
        m: embedding dimension
        tau: time delay for phase space reconstruction, the lag between each point in the phase space vectors.
    """
    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"phase_entropy_m_{m}_tau_{tau}",
        )
        self.m = m
        self.tau = tau
        self.aoi = aoi

    def calculate_feature(self, X: pd.DataFrame) -> float:
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


class LyapunovExponent(MeasureTransformer):
    """Lyapunov Exponent.

    Args:
        m: embedding dimension
        tau: time delay for phase space reconstruction
        T: time steps to average the divergence over
    """
    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        T: int = 1,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"lyapunov_exponent_m_{m}_tau_{tau}_T_{T}",
        )
        self.m = m
        self.tau = tau
        self.T = T
        self.aoi = aoi

    def build_embedding(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(len(X) - (self.m - 1) * self.tau)
            ]
        )

    def calculate_feature(self, X: pd.DataFrame) -> float:
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


class FractalDimension(MeasureTransformer):
    """Fractal Dimension.

    Args:
        m: embedding dimension
        tau: time delay for phase space reconstruction
    """
    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"fractal_dim_m_{m}_tau_{tau}",
        )
        self.m = m
        self.tau = tau
        self.aoi = aoi

    def build_embedding(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(len(X) - (self.m - 1) * self.tau)
            ]
        )

    def calculate_feature(self, X: pd.DataFrame) -> float:
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


class CorrelationDimension(MeasureTransformer):
    """Correlation Dimension.

    Args:
        m: embedding dimension
        tau: time delay for phase space reconstruction
        r: radius threshold for correlation sum
    """
    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        r: float = 0.5,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"corr_dim_m_{m}_tau_{tau}_r_{r}",
        )
        self.m = m
        self.tau = tau
        self.r = r
        self.aoi = aoi
        self.eps = 1e-7

    def build_embedding(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                X[i : i + self.m * self.tau : self.tau]
                for i in range(len(X) - (self.m - 1) * self.tau)
            ]
        )

    def calculate_feature(self, X: pd.DataFrame) -> float:
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = self.build_embedding(X_coord)

        dist_matrix = squareform(pdist(X_emb, metric="euclidean"))
        count = np.sum(dist_matrix < self.r) - len(dist_matrix)

        corr_dim = np.log(self.eps + count / len(dist_matrix)) / np.log(self.r)
        return corr_dim


class RQAMeasures(BaseTransformer):
    """Calculates Recurrence (REC), Determinism (DET), Laminarity (LAM) and Center of Recurrence Mass (CORM) measures.
    These are parts of the Recurrence Quantification Analysis.

    Args:
        metric: callable metric on R^2 points
        rho: threshold radius for RQA matrix
        min_length: min length of lines
        measures: list of measure to calculate (corresponding str)
    """
    def __init__(
        self,
        metric: Callable = euclidean,
        rho: float = 1e-1,
        min_length: int = 1,
        measures: List[str] = ["rec", "det", "lam", "corm"],
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x=x, y=y, pk=pk, return_df=return_df)
        self.rho = rho
        self.metric = metric
        self.min_length = min_length
        self.measures = measures
        self.aoi = aoi
        self.eps = 1e-7

    def _check_init(self, X_len: int):
        assert len(self.measures) > 0, "Error: at least one measure must be passed"
        assert X_len != 0, "Error: there are no fixations"
        assert self.rho is not None, "Error: rho must be a float"
        assert self.min_length is not None, "Error: min_length must be an integer"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def calculate_measures(self, X: pd.DataFrame) -> List[float]:
        columns, features = [], []
        rqa_matrix = get_rqa(X, self.x, self.y, self.metric, self.rho)
        n = rqa_matrix.shape[0]
        r = np.sum(np.triu(rqa_matrix, k=1)) + self.eps

        if "rec" in self.measures:
            columns.append(
                f"rec_metric_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
            )
            features.append(100 * 2 * r / (n * (n - 1)))

        if "det" in self.measures:
            DL = []
            for offset in range(1, n):
                diagonal = np.diag(rqa_matrix, k=offset)
                if len(diagonal) >= self.min_length:
                    for k in range(len(diagonal) - self.min_length + 1):
                        if np.all(diagonal[k : k + self.min_length]):
                            DL.append(np.sum(diagonal[k:]))
            columns.append(
                f"det_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
            )
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
            columns.append(
                f"lam_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
            )
            features.append(100 * (np.sum(HL) + np.sum(VL)) / (2 * r))

        if "corm" in self.measures:
            corm_num = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    corm_num += (j - i) * rqa_matrix[i, j]
            columns.append(
                f"corm_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
            )
            features.append(100 * corm_num / ((n - 1) * r))

        return columns, features

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []
        group_names = []

        if self.pk is None:
            group_names.append("all")
            cur_names, cur_features = self.calculate_measures(X)
            columns_names.extend(cur_names)
            gathered_features.extend(cur_features)
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                group_names.append(group)
                cur_names, cur_features = self.calculate_measures(current_X)
                if len(columns_names) == 0:
                    columns_names.extend(cur_names)
                gathered_features.extend([cur_features])

        features_df = pd.DataFrame(
            data=gathered_features, columns=columns_names, index=group_names
        )
        return features_df if self.return_df else features_df.values


class SaccadeUnlikelihood(MeasureTransformer):
    """Saccade Unlikelihood.

    Calculates cumulative negative log-likelihood of all the saccades in a
    scanpath with respect to the saccade transition model. Default distribution
    parameters are derived from Potsdam Sentence Corpus.

    Args:
        mu_p: mean of the progression distribution
        sigma_p1: left standard deviation of the progression distribution
        sigma_p2: right standard deviation of the progression distribution
        mu_r: mean of the regression distribution
        sigma_r1: left standard deviation of the regression distribution
        sigma_r2: right standard deviation of the regression distribution
        psi: probability of performing a progressive saccade

    Returns:
        the cumulative Negative Log-Likelihood (NLL) of the saccades
    """

    def __init__(
        self,
        mu_p: float = 1.0,
        sigma_p1: float = 0.5,
        sigma_p2: float = 1.0,
        mu_r: float = 1.0,
        sigma_r1: float = 0.3,
        sigma_r2: float = 0.7,
        psi: float = 0.9,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(
            x=x, y=y, pk=pk, return_df=return_df, feature_name="saccade_nll"
        )
        self.mu_p = mu_p
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.mu_r = mu_r
        self.sigma_r1 = sigma_r1
        self.sigma_r2 = sigma_r2
        self.psi = psi
        self.aoi = aoi

    @staticmethod
    def nassym(s: float, mu: float, sigma1: float, sigma2: float) -> float:
        """Calculates assymetric Gaussian PDF at point s.

        Args:
            s: saccade length
            mu: mean of the distribution
            sigma1: standard deviation for the left part of the distribution (s < mu)
            sigma2: standard deviation for the right part of the distribution (s >= mu)

        Returns:
            probability density value for the given saccade length s
        """
        Z = np.sqrt(np.pi / 2) * (sigma1 + sigma2)  # pdf normalization constant
        sigma = sigma1 if s < mu else sigma2
        return np.exp(-((s - mu) ** 2 / (2 * (sigma**2)))) / Z

    def calculate_saccade_proba(self, s: float) -> float:
        """Calculates saccade probability.

        Args:
            s: saccade length

        Returns:
            The probability of the saccade length s.
        """
        progression_proba = self.psi * self.nassym(
            s, self.mu_p, self.sigma_p1, self.sigma_p2
        )
        regression_proba = (1 - self.psi) * self.nassym(
            s, self.mu_r, self.sigma_r1, self.sigma_r2
        )
        return progression_proba + regression_proba

    def calculate_features(self, X: pd.DataFrame) -> List[float]:
        nll = 0
        coords = [self.x, self.y]
        X_sac_len = np.linalg.norm(X[coords].diff().values[1:], axis=1)
        for s_len in X_sac_len:
            p_s = self.calculate_saccade_proba(s_len)
            nll -= np.log(p_s)
        return nll


class HHTFeatures(BaseTransformer):
    """Hilbert-Huang Transform.

    Args:
        max_imfs: maximum number of intrinsic mode functions (IMFs) to extract
        features: list of features to extract from the HHT (aggregation functions, special features)
            List of special functions: ['entropy', 'energy', 'dom_freq', 'sample_entropy', 'complexity_index']

    Returns:
        features extracted from the HHT
    """
    def __init__(
        self,
        max_imfs: int = -1,
        features: List[str] = ["mean", "std"],
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: List[str] = None,
        return_df: bool = True,
    ):
        super().__init__(x=x, y=y, pk=pk, return_df=return_df)
        self.max_imfs = max_imfs
        self.features = features
        self.aoi = aoi
        self._feature_mapping = {
            "mean": partial(np.mean, axis=(0, 1)),
            "std": partial(np.std, axis=(0, 1)),
            "var": partial(np.var, axis=(0, 1)),
            "median": partial(np.median, axis=(0, 1)),
            "max": partial(np.max, axis=(0, 1)),
            "min": partial(np.min, axis=(0, 1)),
            "skew": partial(skew, axis=(0, 1)),
            "kurtosis": partial(kurtosis, axis=(0, 1)),
            "entropy": partial(entropy, axis=(0, 1)),
            "energy": lambda data: np.sum(data**2, axis=(0, 1)),
            "dom_freq": self.dominant_freq,
            # "sample_entropy": self.sample_entropy,
            # "complexity_index": self.complexity_index,
        }

    def _check_init(self, X_len: int):
        assert X_len != 0, "Error: there are no fixations"
        assert self.features, "Error: at least one feature must be passed"
        assert (
            self.max_imfs > 0 or self.max_imfs == -1
        ), "Error: max_imfs must be a positive integer or -1"
        for feature in self.features:
            assert feature in self._feature_mapping, f"Error: unknown feature {feature}"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def dominant_freq(self, imf_data: np.ndarray) -> List[float]:
        """Calculates dominant frequency of the intrinsic mode functions (IMFs) using FFT.

        Args:
            imf_data: intrinsic mode functions (IMFs) data

        Returns:
            dominant frequency of each IMF
        """
        imf_fft = fft2(imf_data, axes=(0, 1))
        dom_freq = []
        for imf in imf_fft:
            signal = np.abs(imf.flatten())
            dom_freq.append(np.argmax(signal) / np.max(len(signal), 0))
        return dom_freq

    def coarse_grain(self, imf_data: np.ndarray, scale: int = 5) -> np.ndarray:
        """Calculates coarse-grained standard deviation of the intrinsic mode functions (IMFs).

        Args:
            imf_data: intrinsic mode functions (IMFs) data

        Returns:
            coarse-grained standard deviation of each IMF
        """
        cg = []
        for imf in imf_data:
            signal = imf.flatten()
            assert (
                len(signal) >= scale
            ), "Error: signal length must be greater than scale"
            current = []
            for i in range(0, len(signal), scale):
                current.append(np.mean(signal[i : i + scale]))
            cg.append(np.array(current))
        return np.array(cg)

    def sample_entropy(
        self, imf_data: np.ndarray, m: int = 1, r: float = 0.2
    ) -> List[float]:
        """Calculates sample entropy of the intrinsic mode functions (IMFs).

        Args:
            imf_data: intrinsic mode functions (IMFs) data
            m: length of sequences to compare
            r: tolerance for accepting mathces

        Returns:
            sample entropy of each IMF
        """
        se = []
        for imf in imf_data:
            cur_imf = imf.flatten()
            cur_r = r * np.std(cur_imf)

            def _phi(m: int) -> float:
                X = np.array([cur_imf[i : i + m] for i in range(len(cur_imf) - m)])
                B = np.sum(
                    np.all(np.abs(X[:, np.newaxis] - X[np.newaxis, :]) <= cur_r, axis=0)
                ) - len(X)
                return B / (len(cur_imf) - m)

            se.append(_phi(m + 1) / _phi(m))
        return se

    def complexity_index(
        self, imf_data: np.ndarray, m: int = 5, r: float = 0.20, max_scale: int = 2
    ) -> List[float]:
        """Calculates complexity index of the intrinsic mode functions (IMFs).

        Args:
            imf_data: intrinsic mode functions (IMFs) data
            m: length of sequences for sample entropy
            r: tolerance for sample entropy
            max_scale: maximum scale for coarse-graining

        Returns:
            complexity index of each IMF
        """
        cis = []
        for imf in imf_data:
            ci = 0
            for scale in range(1, max_scale + 1):
                cg_data = self.coarse_grain(imf[np.newaxis, :], scale)[0]
                ci += self.sample_entropy(cg_data, m=m, r=r)[0]
            cis.append(ci)
        return cis

    def calculate_features(self, data: pd.DataFrame) -> List[float]:
        """Feature extraction from the HHT.

        Args:
            data: 1D array of the HHT signal

        Returns:
            list of features extracted from the HHT
        """

        cols = [self.x, self.y]
        X_data = data[cols].values

        decomposed = hilbert_huang_transform(X_data, max_imf=self.max_imfs)
        n_imfs = decomposed.shape[0]

        columns_names = []
        gathered_features = []
        for feature in self.features:
            for i in range(n_imfs):
                columns_names.append(f"imf{i}_{feature}")
            gathered_features.extend(
                list(map(self._feature_mapping[feature], decomposed))
            )

        return columns_names, gathered_features

    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        self._check_init(X_len=X.shape[0])

        columns_names = []
        gathered_features = []
        group_names = []

        if self.pk is None:
            group_names.append("all")
            columns_names, gathered_features = self.calculate_features(X)
        else:
            X_splited = _split_dataframe(X, self.pk)
            for group, current_X in X_splited:
                group_names.append(group)
                cur_names, cur_features = self.calculate_features(current_X)
                if len(columns_names) == 0:
                    columns_names.extend(cur_names)
                gathered_features.extend([cur_features])

        features_df = pd.DataFrame(
            data=gathered_features, columns=columns_names, index=group_names
        )
        return features_df if self.return_df else features_df.values
