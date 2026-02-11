from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
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
        aoi: Area Of Interest column name(-s). If provided, features can be
            calculated inside the specified AOI.
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        feature_name: Column name for resulting feature.
        ignore_errors: If True, return NaN values when feature computation fails
            instead of raising an error. Default is False.
    """

    def __init__(
        self,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        feature_name: str = "feature",
        ignore_errors: bool = False,
    ):
        super().__init__(x=x, y=y, pk=pk, return_df=return_df)
        self.aoi = aoi
        self.feature_name = feature_name
        self.ignore_errors = ignore_errors

    def _check_init(self, X_len: int):
        assert X_len != 0, "Error: there are no fixations"

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [self.feature_name]

    @abstractmethod
    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        # Handle empty DataFrame case
        if X.shape[0] == 0:
            if self.ignore_errors:
                # Return DataFrame with NaN values
                names = self.get_feature_names_out()
                if self.pk is None:
                    features_df = pd.DataFrame(
                        data=[[np.nan] * len(names)], columns=names, index=["all"]
                    )
                else:
                    # Return empty DataFrame with correct columns
                    features_df = pd.DataFrame(columns=names)
                return features_df if self.return_df else features_df.values
            else:
                self._check_init(X_len=X.shape[0])

        # Check initialization (catch errors if ignore_errors=True)
        try:
            self._check_init(X_len=X.shape[0])
        except (AssertionError, RuntimeError) as e:
            if self.ignore_errors:
                # Return NaN for all groups
                names = self.get_feature_names_out()
                if self.pk is None:
                    features_df = pd.DataFrame(
                        data=[[np.nan] * len(names)], columns=names, index=["all"]
                    )
                else:
                    # For grouped data, return NaN for each group
                    X_split = _split_dataframe(X, self.pk)
                    group_names = [group for group, _ in X_split]
                    gathered_features = [[np.nan] * len(names) for _ in group_names]
                    features_df = pd.DataFrame(
                        data=gathered_features, columns=names, index=group_names
                    )
                return features_df if self.return_df else features_df.values
            else:
                raise type(e)(
                    f"{e!s} Set ignore_errors=True to return NaN instead."
                ) from e

        group_names = []
        gathered_features = []
        columns_names = []

        if self.pk is None:
            group_names.append("all")
            try:
                names, values = self.calculate_features(X)
                columns_names = names
                gathered_features.append(values)
            except Exception as e:
                if self.ignore_errors:
                    # Get feature names to create appropriate number of NaNs
                    names = self.get_feature_names_out()
                    if not columns_names:
                        columns_names = names
                    # Create NaN values for all features
                    gathered_features.append([np.nan] * len(names))
                else:
                    raise type(e)(
                        f"{e!s} Set ignore_errors=True to return NaN instead."
                    ) from e
        else:
            X_split = _split_dataframe(X, self.pk)
            for group, current_X in X_split:
                group_names.append(group)
                try:
                    names, values = self.calculate_features(current_X)
                    if not columns_names:
                        columns_names = names
                    gathered_features.append(values)
                except Exception as e:
                    if self.ignore_errors:
                        # Get feature names to create appropriate number of NaNs
                        names = self.get_feature_names_out()
                        if not columns_names:
                            columns_names = names
                        # Create NaN values for all features
                        gathered_features.append([np.nan] * len(names))
                    else:
                        raise type(e)(
                            f"{e!s} Set ignore_errors=True to return NaN instead."
                        ) from e

        features_df = pd.DataFrame(
            data=gathered_features, columns=columns_names, index=group_names
        )
        return features_df if self.return_df else features_df.values


class HurstExponent(MeasureTransformer):
    r"""Approximates Hurst Exponent using R/S analysis.

    The Hurst exponent is a measure of the long-term memory of time series.
    It relates to the autocorrelations of the time series, and the rate at
    which these decrease as the lag between pairs of values increases.
    $H \in (0.5, 1)$ indicates a persistent behavior (trend).
    $H \in (0, 0.5)$ indicates an anti-persistent behavior (mean-reverting).
    $H = 0.5$ indicates a completely random series (Geometric Brownian Motion).

    Args:
        coordinate: coordinate column name (1D Hurst exponent currently available).
        n_iters: number of iterations to complete. Note: data must be
            of length more than :math:`2^{n\_iters}`.
        fill_strategy: how to make vector be length of power of :math:`2`.
            If "reduce", then all values after :math:`2^k`-th are removed,
            where :math:`n < 2^{(k + 1)}`. Other strategies specify the value
            to fill the vector up to the closest power of :math:`2`, "mean"
            being the mean of vector, "last" being the last value
            (makes time-series constant at the end).
        pk: list of column names used to split pd.DataFrame.
        eps: division epsilon.
        return_df: Return pd.Dataframe object else np.ndarray.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.

    Example:
        Quick start with default parameters::

            from eyefeatures.features.measures import HurstExponent

            transformer = HurstExponent(coordinate="x")
            features = transformer.fit_transform(fixations_df)
    """

    def __init__(
        self,
        coordinate: str,
        n_iters: int = 10,
        fill_strategy: Literal["mean", "reduce", "last"] = "last",
        pk: list[str] = None,
        eps: float = 1e-22,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=coordinate,
            pk=pk,
            return_df=return_df,
            feature_name="hurst_exponent",
            ignore_errors=ignore_errors,
        )
        self.coordinate = coordinate
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

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Generate feature name that includes coordinate column and hyperparameters."""
        # Determine coordinate name (x or y) from the column name
        # Build feature name with coordinate and hyperparameters
        feature_name = (
            f"hurst_{self.coordinate}_n{self.n_iters}_fill_{self.fill_strategy}"
        )
        return [feature_name]

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

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        x = X[self.coordinate].values
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

        # OLS via polynomial fitting
        rs = rs[:cnt]
        bs = bs[:cnt]
        # Hurst is slope of log(RS) vs log(n)
        X_ols = np.vstack([np.ones(cnt), np.log(bs)]).T
        grad = np.linalg.lstsq(X_ols, np.log(rs), rcond=None)[0]

        # Use dynamic feature name that includes coordinate and hyperparameters
        feature_name = self.get_feature_names_out()[0]
        return [feature_name], [grad[1]]


class ShannonEntropy(MeasureTransformer):
    """Shannon Entropy.

    Measures the uncertainty or randomness of the gaze distribution over Areas of Interest (AOIs).
    Higher entropy indicates a more uniform distribution of fixations across AOIs (scanning),
    while lower entropy indicates concentration on specific AOIs.

    Args:
        aoi: Area Of Interest column name.
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            aoi=aoi,
            pk=pk,
            return_df=return_df,
            feature_name="entropy",
            ignore_errors=ignore_errors,
        )

    def _check_init(self, X_len: int):
        assert self.aoi is not None, "Error: Provide aoi column"
        assert X_len != 0, "Error: there are no fixations"

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        all_fix = X.shape[0]
        aoi_probability = []
        X_aoi = _split_dataframe(X, [self.aoi])
        for aoi_group, current_aoi_X in X_aoi:
            aoi_probability.append(current_aoi_X.shape[0] / all_fix)

        entropy_val = 0
        for p in aoi_probability:
            entropy_val -= p * np.log2(p)

        return [self.feature_name], [entropy_val]


class SpectralEntropy(MeasureTransformer):
    """Spectral Entropy.

    Measures the complexity of the fixation trajectory in the frequency domain using
    Power Spectral Density (PSD). It treats the scanpath coordinates as a signal.
    High spectral entropy implies a more random/complex signal (white noise),
    while low entropy implies a more periodic/predictable signal.

    Args:
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name="spectral_entropy",
            ignore_errors=ignore_errors,
        )
        self.aoi = aoi

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        coords = [self.x, self.y]
        transformed_seq = ifft(X[coords].values)
        power_spectrum_seq = np.linalg.norm(transformed_seq, axis=1) ** 2
        proba_distribution = power_spectrum_seq / np.sum(power_spectrum_seq)
        return [self.feature_name], [entropy(proba_distribution)]


class FuzzyEntropy(MeasureTransformer):
    """Fuzzy Entropy.

    A variation of Sample Entropy that uses a fuzzy membership function (typically exponential)
    to assess the similarity between vectors, rather than a hard Heaviside step function.
    This makes it more robust to noise and less sensitive to the choice of parameters.
    Measures the complexity of the scanpath.

    Args:
        m: embedding dimension (length of sequences to compare).
        r: tolerance threshold/width of the fuzzy membership function (usually 0.2 * std).
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        x: str = None,
        y: str = None,
        m: int = 2,
        r: float = 0.2,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"fuzzy_m_{m}_r_{r}",
            ignore_errors=ignore_errors,
        )
        self.m = m
        self.r = r
        self.aoi = aoi
        self.eps = 1e-7

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
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

        return [self.feature_name], [np.log(phi_m[0] / (phi_m[1] + self.eps))]


class SampleEntropy(MeasureTransformer):
    """Sample Entropy.

    Measures the complexity or irregularity of the scanpath. It is defined as the negative
    natural logarithm of the conditional probability that two sequences similar for `m` points
    remain similar at the next point, excluding self-matches.
    Lower values indicate more self-similarity (regularity), higher values indicate more
    complexity/randomness.

    Args:
        m: embedding dimension (length of sequences to compare).
        r: tolerance threshold for matches acceptance (usually 0.2 * std).
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        m: int = 2,
        r: float = 0.2,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"sample_entropy_m={m}_r={r}",
            ignore_errors=ignore_errors,
        )
        self.m = m
        self.r = r
        self.aoi = aoi
        self.eps = 1e-6

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        n = 2 * len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = np.array([X_coord[j : j + self.m] for j in range(n - self.m + 1)])
        dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
        B = np.sum(np.sum(dist_matrix < self.r, axis=0) - 1)
        X_emb = np.array([X_coord[j : j + self.m + 1] for j in range(n - self.m)])
        dist_matrix = squareform(pdist(X_emb, metric="chebyshev"))
        A = np.sum(np.sum(dist_matrix < self.r, axis=0) - 1)
        return [self.feature_name], [-np.log(A / (B + self.eps) + 1e-100)]


class IncrementalEntropy(MeasureTransformer):
    """Incremental Entropy.

    Measures the average entropy of the fixation distribution as it evolves over time.
    It calculates the Shannon entropy of the coordinate distribution at each step $i$
    (using fixations $1$ to $i$) and then averages these values. This captures how
    the spatial distribution complexity changes as more of the visual stimulus is explored.

    Args:
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name="incremental_entropy",
            ignore_errors=ignore_errors,
        )
        self.aoi = aoi

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        n = len(X)
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        incremental_entropies = np.zeros(n)
        for i in range(1, n):
            hist, _ = np.histogram(X_coord[: i + 1], bins="auto", density=True)
            hist = hist[hist > 0]
            incremental_entropies[i] = -np.sum(hist * np.log(hist))

        return [self.feature_name], [incremental_entropies.mean()]


class GriddedDistributionEntropy(MeasureTransformer):
    """Gridded Distribution Entropy.

    Measures the randomness of the spatial distribution of fixations by discretizing the
    2D plane into a grid. It calculates the Shannon entropy of the 2D histogram of
    fixations over this grid. High entropy indicates fixations are spread out across
    the grid; low entropy indicates clustering.

    Args:
        grid_size: the number of bins (grid cells) per dimension for creating the histogram.
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        grid_size: int = 10,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"gridded_entropy_grid_size_{grid_size}",
            ignore_errors=ignore_errors,
        )
        self.grid_size = grid_size
        self.aoi = aoi

    def _check_init(self, X_len: int):
        assert X_len != 0, "Error: there are no fixations"

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        coords = [self.x, self.y]
        X_coord = X[coords].values
        H, edges = np.histogramdd(X_coord, bins=self.grid_size)
        P = H / np.sum(H)
        P = P[P > 0]
        return [self.feature_name], [-np.sum(P * np.log(P))]


class PhaseEntropy(MeasureTransformer):
    """Phase Entropy.

    Measures the complexity of the phase space trajectory. The scanpath (time series) is
    embedded into a multi-dimensional phase space using time-delay embedding.
    The entropy of the distribution of pairwise distances (or density) in this phase space
    is calculated. Higher values distinguish chaotic signals from periodic/predictable ones.

    Args:
        m: embedding dimension (default: 2).
        tau: time delay for phase space reconstruction (default: 1).
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.

    Example:
        from eyefeatures.features.measures import PhaseEntropy

        transformer = PhaseEntropy(x="x", y="y")
        features = transformer.fit_transform(fixations_df)
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"phase_entropy_m_{m}_tau_{tau}",
            ignore_errors=ignore_errors,
        )
        self.m = m
        self.tau = tau
        self.aoi = aoi

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
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
        entropy = -np.sum(prob_dist * np.log(prob_dist))
        return [self.feature_name], [entropy]


class LyapunovExponent(MeasureTransformer):
    """Lyapunov Exponent.

    Estimates the largest Lyapunov exponent, which characterizes the rate of separation of
    infinitesimally close trajectories in phase space.
    A positive Lyapunov exponent indicates chaos (sensitive dependence on initial conditions).
    Calculated using the Rosenstein algorithm (tracking divergence of nearest neighbors).

    Args:
        m: embedding dimension (default: 2).
        tau: time delay for phase space reconstruction (default: 1).
        T: time steps to average the divergence over (default: 1).
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.

    Example:
        from eyefeatures.features.measures import LyapunovExponent

        transformer = LyapunovExponent(x="x", y="y")
        features = transformer.fit_transform(fixations_df)
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        T: int = 1,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"lyapunov_exponent_m_{m}_tau_{tau}_T_{T}",
            ignore_errors=ignore_errors,
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

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
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

        val = np.mean(divergence) / self.T if divergence else np.inf
        return [self.feature_name], [val]


class FractalDimension(MeasureTransformer):
    """Fractal Dimension.

    Estimates the fractal dimension of the scanpath (or its embedding) using the
    Box-Counting method. It measures how the scanpath fills the space. A higher
    fractal dimension indicates a more complex, space-filling pattern.

    Args:
        m: embedding dimension.
        tau: time delay for phase space reconstruction.
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"fractal_dim_m_{m}_tau_{tau}",
            ignore_errors=ignore_errors,
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

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
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
        return [self.feature_name], [-coeffs[0]]


class CorrelationDimension(MeasureTransformer):
    """Correlation Dimension.

    A measure of the dimensionality of the space occupied by a set of random points
    (the attractor of the dynamical system). It is related to the fractal dimension but
    calculated based on the correlation sum (fraction of pairs of points closer than distance r).
    It provides a lower bound for the fractal dimension.

    Args:
        m: embedding dimension.
        tau: time delay for phase space reconstruction.
        r: radius threshold for correlation sum.
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        m: int = 2,
        tau: int = 1,
        r: float = 0.5,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name=f"corr_dim_m_{m}_tau_{tau}_r_{r}",
            ignore_errors=ignore_errors,
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

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        coords = [self.x, self.y]
        X_coord = X[coords].values.flatten()
        X_emb = self.build_embedding(X_coord)

        dist_matrix = squareform(pdist(X_emb, metric="euclidean"))
        count = np.sum(dist_matrix < self.r) - len(dist_matrix)

        corr_dim = np.log(self.eps + count / len(dist_matrix)) / np.log(self.r)
        return [self.feature_name], [corr_dim]


class RQAMeasures(MeasureTransformer):
    """Calculates REC, DET, LAM and CORM measures.

    Recurrence Quantification Analysis (RQA) is a nonlinear technique to quantify the
    structure of dynamical systems based on recurrence plots. It identifies recurrent states
    and their patterns.
    - REC (Recurrence Rate): Percentage of recurrent points.
    - DET (Determinism): Percentage of recurrent points forming diagonal lines.
    - LAM (Laminarity): Percentage of recurrent points forming vertical/horizontal lines.
    - CORM (Center of Recurrence Mass): Average distance of recurrence points from the
        main diagonal.

    Args:
        metric: callable metric on R^2 points (e.g., `scipy.spatial.distance.euclidean`).
        rho: threshold radius for RQA matrix. Two points are considered recurrent if their
            distance is less than `rho`.
        min_length: minimum length of diagonal/vertical/horizontal lines to be counted.
        measures: list of measures to calculate (subset of ['rec', 'det', 'lam', 'corm']).
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.
    """

    def __init__(
        self,
        metric: Callable = euclidean,
        rho: float = 1e-1,
        min_length: int = 1,
        measures: list[str] = None,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        if measures is None:
            measures = ["rec", "det", "lam", "corm"]
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name="rqa",
            ignore_errors=ignore_errors,
        )
        self.rho = rho
        self.metric = metric
        self.min_length = min_length
        self.measures = measures
        self.aoi = aoi

        self.eps = 1e-7

    def get_feature_names_out(self, input_features=None) -> list[str]:
        columns = []
        for measure in self.measures:
            if measure == "rec":
                columns.append(
                    f"rec_metric_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
                )
            elif measure == "det":
                columns.append(
                    f"det_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
                )
            elif measure == "lam":
                columns.append(
                    f"lam_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
                )
            elif measure == "corm":
                columns.append(
                    f"corm_{self.metric.__name__}_length_{self.min_length}_rho_{self.rho}"
                )
        return columns

    def _check_init(self, X_len: int):
        assert len(self.measures) > 0, "Error: at least one measure must be passed"
        assert X_len != 0, "Error: there are no fixations"
        assert self.rho is not None, "Error: rho must be a float"
        assert self.min_length is not None, "Error: min_length must be an integer"

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        columns, features = [], []
        rqa_matrix = get_rqa(X, self.x, self.y, self.metric, self.rho)
        n = rqa_matrix.shape[0]
        r = np.sum(np.triu(rqa_matrix, k=1)) + self.eps

        if "rec" in self.measures:
            f_name = (
                f"rec_metric_{self.metric.__name__}_length_"
                f"{self.min_length}_rho_{self.rho}"
            )
            columns.append(f_name)
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


class SaccadeUnlikelihood(MeasureTransformer):
    """Saccade Unlikelihood.

    Calculates cumulative negative log-likelihood of all the saccades in a
    scanpath with respect to a probabilistic saccade transition model. This model assumes
    saccades are either progressive or regressive, each following an asymmetric Gaussian
    distribution. Default distribution parameters are derived from Potsdam Sentence Corpus.
    Higher NLL indicates a less typical (or more unusual) scanpath according to the model.

    Args:
        mu_p: mean of the progression (forward saccade) distribution.
        sigma_p1: left standard deviation of the progression distribution (for lengths < mu_p).
        sigma_p2: right standard deviation of the progression distribution (for lengths >= mu_p).
        mu_r: mean of the regression (backward saccade) distribution.
        sigma_r1: left standard deviation of the regression distribution.
        sigma_r2: right standard deviation of the regression distribution.
        psi: probability of performing a progressive saccade (vs. a regression).
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.

    Returns:
        the cumulative Negative Log-Likelihood (NLL) of the saccades.
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
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name="saccade_nll",
            ignore_errors=ignore_errors,
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

    def calculate_features(self, X: pd.DataFrame) -> tuple[list[str], list[float]]:
        nll = 0
        coords = [self.x, self.y]
        X_sac_len = np.linalg.norm(X[coords].diff().values[1:], axis=1)
        for s_len in X_sac_len:
            p_s = self.calculate_saccade_proba(s_len)
            nll -= np.log(p_s) if p_s > 1e-15 else -30.0
        return [self.feature_name], [nll]


class HHTFeatures(MeasureTransformer):
    """Hilbert-Huang Transform (HHT) Features.

    Decomposes the signal (scanpath coordinates) into Intrinsic Mode Functions (IMFs) using
    Empirical Mode Decomposition (EMD), then extracts statistical features from these IMFs.
    HHT is well-suited for analyzing non-linear, non-stationary signals.

    Args:
        max_imfs: maximum number of intrinsic mode functions (IMFs) to extract.
            Set to -1 for automatic determination.
        features: list of features to extract from each IMF. Available options are:
            'mean', 'std', 'var', 'median', 'max', 'min', 'skew', 'kurtosis',
            'entropy', 'energy', 'dom_freq'.
        x: X coordinate column name.
        y: Y coordinate column name.
        aoi: Area Of Interest column name(-s).
        pk: primary key.
        return_df: whether to return output as DataFrame or numpy array.
        ignore_errors: If True, return NaN when feature computation fails; otherwise raise.

    Returns:
        features extracted from each IMF of the HHT decomposition.
    """

    def __init__(
        self,
        max_imfs: int = -1,
        features: list[str] = None,
        x: str = None,
        y: str = None,
        aoi: str = None,
        pk: list[str] = None,
        return_df: bool = True,
        ignore_errors: bool = False,
    ):
        if features is None:
            features = ["mean", "std"]
        super().__init__(
            x=x,
            y=y,
            pk=pk,
            return_df=return_df,
            feature_name="hht",
            ignore_errors=ignore_errors,
        )
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
            # Cannot be used because require additional parameters.
            # "sample_entropy": self.sample_entropy,
            # "complexity_index": self.complexity_index,
        }

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return [f"{self.feature_name}_{feat_nm}" for feat_nm in self.features]

    def _check_init(self, X_len: int):
        assert X_len != 0, "Error: there are no fixations"
        assert self.features, "Error: at least one feature must be passed"
        assert (
            self.max_imfs > 0 or self.max_imfs == -1
        ), "Error: max_imfs must be a positive integer or -1"
        for feature in self.features:
            assert feature in self._feature_mapping, f"Error: unknown feature {feature}"

    def dominant_freq(self, imf_data: np.ndarray) -> list[float]:
        """Calculates dominant frequency of the IMFs using FFT.

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
        """Calculates coarse-grained std of the IMFs.

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
    ) -> list[float]:
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
    ) -> list[float]:
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

    def calculate_features(self, data: pd.DataFrame) -> tuple[list[str], list[float]]:
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
