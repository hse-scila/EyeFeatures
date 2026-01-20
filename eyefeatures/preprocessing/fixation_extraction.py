from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy

from eyefeatures.preprocessing._utils import _get_distance, _get_MEC
from eyefeatures.preprocessing.base import BaseFixationPreprocessor


# ======== FIXATION PREPROCESSORS ========
class IVT(BaseFixationPreprocessor):
    """Velocity Threshold Identification.

    Complexity: O(n).
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str,
        threshold: float,
        min_duration: float,
        distance: str = "euc",
        pk: list[str] = None,
        eps: float = 1e-10,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.threshold = threshold
        self.min_duration = min_duration
        self.distance = distance
        self.eps = eps

        self.available_distances = ("euc", "manhattan", "chebyshev")  # TODO add more

    def _check_params(self):
        m = "IVT"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.t is not None, self._err_no_field(m, "t")
        assert self.distance in self.available_distances, (
            f"'distance' must be one of ({', '.join(self.available_distances)}),"
            f"got {self.distance}."
        )

    # @jit(forceobj=True, looplift=True)
    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        points = X[[self.x, self.y]].values
        dist = self._get_distances(points, self.distance)

        x = X[self.x].values
        y = X[self.y].values
        t = X[self.t].values

        dt = np.diff(t)
        vel = dist / (dt + self.eps)

        is_fixation = (vel <= self.threshold).astype(np.int32)
        fixations = self._squash_fixations(is_fixation)

        fixations_df = pd.DataFrame(
            data={
                "fixation_id": fixations,
                self.x: x[:-1],
                self.y: y[:-1],
                "start_time": t[:-1],
                "end_time": t[:-1],
                "distance_min": dist,
                "distance_max": dist,
            }
        )

        fixations_df = fixations_df[fixations_df["fixation_id"] != 0]
        diameters = []
        centers = []

        for i in set(fixations):
            if i != 0:
                points = fixations_df.loc[
                    fixations_df["fixation_id"] == i,
                    fixations_df.columns.isin([self.x, self.y]),
                ].values
                x, y, radius = _get_MEC(np.unique(points, axis=0))
                diameters.append(radius * 2)
                centers.append(np.array([x, y]))

        fixations_df = fixations_df.groupby(by=["fixation_id"]).agg(
            {
                self.x: "mean",
                self.y: "mean",
                "start_time": "min",
                "end_time": "max",
                "distance_min": "min",  # between consecutive gazes
                "distance_max": "max",
            }
        )

        fixations_df["diameters"] = diameters
        fixations_df["centers"] = centers

        # default features
        feats = (
            "duration",
            "saccade_duration",
            "saccade_length",
            "saccade_angle",
            "saccade2_angle",
        )
        fixations_df = self._compute_feats(fixations_df, feats)
        fixations_df = fixations_df[fixations_df["duration"] >= self.min_duration]

        return fixations_df


class IDT(BaseFixationPreprocessor):
    """Dispersion Threshold Identification.

    Complexity: O(n^2 * log n) worst case, O(n * log n) average.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str,
        min_duration: float,
        max_duration: float,
        max_dispersion: float,
        distance: str = "euc",  # norm in R^2 for distance calculation
        pk: list[str] = None,
        eps: float = 1e-20,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_dispersion = max_dispersion
        self.distance = distance
        self.eps = eps

        self.available_distances = ("euc", "manhattan", "chebyshev")  # TODO add more

    def _check_params(self):
        m = "IDT"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.t is not None, self._err_no_field(m, "t")
        assert self.min_duration is not None, self._err_no_field(m, "min_duration")
        assert self.min_duration > 0, "'min_duration' must be non-negative."
        assert self.max_duration is not None, self._err_no_field(m, "max_duration")
        assert (
            self.max_duration > self.min_duration
        ), "'max_duration' must be greater than min_duration."
        assert self.max_dispersion is not None, self._err_no_field(m, "min_duration")
        assert self.max_dispersion > 0, "'max_dispersion' must be non-negative."

        assert self.distance in self.available_distances, (
            f"'distance' must be one of ({', '.join(self.available_distances)}),"
            f"got {self.distance}."
        )

    @staticmethod
    def _build_sparse_tables(a):
        n = a.shape[0]
        logn = int(np.ceil(np.log2(n)))

        mint = np.zeros((logn, n))
        maxt = np.zeros((logn, n))

        mint[0] = a
        maxt[0] = a

        for t in range(logn - 1):
            for i in range(n - (2 << t) + 1):
                mint[t + 1, i] = min(mint[t, i], mint[t, i + (1 << t)])
                maxt[t + 1, i] = max(maxt[t, i], maxt[t, i + (1 << t)])
        return mint, maxt

    @staticmethod
    def _rmq(table: np.ndarray, left_idx: int, right_idx: int, f: Callable):
        """RMQ on range [left_idx, right_idx)."""

        t = int(np.log2(right_idx - left_idx))
        return f(table[t, left_idx], table[t, right_idx - (1 << t)])

    def _get_disp_window(
        self, left_idx: int, r_idx: int, mintx, maxtx, minty, maxty
    ) -> [int, float]:
        """Binary search to find the widest dispersion window in [left_idx, r_idx]."""

        right_border = left_idx
        left_border = left_idx
        window_disp = -np.inf
        while r_idx >= left_idx:
            m = left_idx + int((r_idx - left_idx) / 2)

            minx = self._rmq(mintx, left_border, m + 1, min)
            miny = self._rmq(minty, left_border, m + 1, min)
            maxx = self._rmq(maxtx, left_border, m + 1, max)
            maxy = self._rmq(maxty, left_border, m + 1, max)
            disp = _get_distance(
                np.array([minx, miny]), np.array([maxx, maxy]), self.distance
            )
            if disp <= self.max_dispersion:
                right_border = m
                window_disp = disp
                left_idx = m + 1
            else:
                r_idx = m - 1
        return right_border, window_disp

    def _get_dur_window(self, left_idx: int, r: int, t: np.ndarray) -> [int, int]:
        """Sliding window to find the widest duration window in [left_idx, r]."""

        right_border = left_idx
        end_time = t[left_idx] + self.max_duration
        while right_border + 1 <= r and t[right_border + 1] < end_time:
            right_border += 1

        start_time = t[left_idx] + self.min_duration
        return right_border if t[right_border + 1] >= start_time else -1

    # @jit(forceobj=True, looplift=True)
    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        points = X[[self.x, self.y]].values
        dist = self._get_distances(points, self.distance)

        x = X[self.x].values
        y = X[self.y].values
        t = X[self.t].values

        dt = np.diff(t)
        vel = dist / (dt + self.eps)

        n = len(vel)
        fixations = np.zeros(n)
        fixation_id = 1
        disp = np.full(n, -np.inf)

        # === IDT Algorithm ===
        left_idx = 0
        r = n - 1  # [left_idx, r]
        mintx, maxtx = self._build_sparse_tables(x)
        minty, maxty = self._build_sparse_tables(y)

        # restrictions on duration and dispersion are considered consequently
        while left_idx < n:
            # 1. Maintain the widest duration window using sliding window.
            # 2. Inside, find the widest dispersion window using binary search.
            # 3. Classify gazes from window in step 2 as fixation and continue.

            dur_right_border = self._get_dur_window(left_idx, r, t)
            if dur_right_border == -1:
                break

            disp_right_border, window_disp = self._get_disp_window(
                left_idx, dur_right_border, mintx, maxtx, minty, maxty
            )

            # max_duration and max_dispersion satisfied
            if t[disp_right_border] - t[left_idx] < self.min_duration:
                left_idx += 1
                continue

            fixations[left_idx:disp_right_border] = (
                fixation_id  # [left_idx, dur_right_border] is single fixation
            )
            disp[left_idx:disp_right_border] = window_disp

            fixation_id += 1
            left_idx = disp_right_border + 1

        # === IDT Algorithm ===

        fixations_df = pd.DataFrame(
            data={
                "fixation_id": fixations,
                self.x: x[:-1],
                self.y: y[:-1],
                "start_time": t[:-1],
                "end_time": t[:-1],
                "distance_min": dist,
                "distance_max": dist,
                "dispersion": disp,
            }
        )

        fixations_df = fixations_df[fixations_df["fixation_id"] != 0]
        diameters = []
        centers = []

        for i in set(fixations):
            if i != 0:
                points = fixations_df.loc[
                    fixations_df["fixation_id"] == i,
                    fixations_df.columns.isin([self.x, self.y]),
                ].values
                x, y, radius = _get_MEC(np.unique(points, axis=0))
                diameters.append(radius * 2)
                centers.append(np.array([x, y]))

        fixations_df = fixations_df.groupby(by=["fixation_id"]).agg(
            {
                self.x: "mean",
                self.y: "mean",
                "start_time": "min",
                "end_time": "max",
                "distance_min": "min",
                "distance_max": "max",
                "dispersion": "max",  # just for API, window has same values
            }
        )

        if len(fixations_df) <= 1:
            raise RuntimeError(
                f"Found only {len(fixations_df)} fixations, you either provided "
                f"infeasible constraints or have a mismatch of units in data "
                f"and constraints."
            )

        fixations_df["diameters"] = diameters
        fixations_df["centers"] = centers

        # default features
        feats = (
            "duration",
            "saccade_duration",
            "saccade_length",
            "saccade_angle",
            "saccade2_angle",
        )
        fixations_df = self._compute_feats(fixations_df, feats)

        return fixations_df


class IHMM(BaseFixationPreprocessor):
    """Hidden Markov Model Identification. Based on Viterbi algorithm.
    Complexity: O(n^2) for single group.

    Args:
        fix2sac: probability of transition from fixation to saccade.
        sac2fix: probability of transition from saccade to fixation.
        fix_distrib: distribution of fixations.
        sac_distrib: distribution of saccades.
        distrib_params: 'auto' for default params or dict with
           {"fixation": params1, "saccade": params2}, where
           "params" are arguments for  appropriate `scipy.stats` function.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str,
        fix2sac: float = 0.05,
        sac2fix: float = 0.05,
        fix_distrib: str = "norm",  # fixation distribution
        sac_distrib: str = "norm",
        distrib_params: str | dict[str, float] = "auto",
        distance: str = "euc",
        pk: list[str] = None,
        eps: float = 1e-20,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.fix2sac = fix2sac
        self.sac2fix = sac2fix

        self.fix_distrib = fix_distrib
        self.sac_distrib = sac_distrib
        self.distrib_params = distrib_params
        self.distance = distance
        self.eps = eps

        self.available_distances = ("euc", "manhattan", "chebyshev")
        self.available_distributions = ("norm",)  # TODO: add more

    def _check_params(self):
        m = "IHMM"
        assert self.x is not None, self._err_no_field(m, "x")
        assert self.y is not None, self._err_no_field(m, "y")
        assert self.t is not None, self._err_no_field(m, "t")

        # TODO wrap this error message
        assert self.distance in self.available_distances, (
            f"'distance' must be one of ({', '.join(self.available_distances)}),"
            f"got '{self.distance}'."
        )
        assert self.fix_distrib in self.available_distributions, (
            f"'fix_distrib' must be one of ({', '.join(self.available_distributions)}),"
            f"got '{self.sac_distrib}'."
        )
        assert self.sac_distrib in self.available_distributions, (
            f"'sac_distrib' must be one of ({', '.join(self.available_distributions)}),"
            f"got '{self.sac_distrib}'."
        )
        assert isinstance(self.fix2sac, float) and (
            0.0 < self.fix2sac < 1.0
        ), "'fix2sac' must be float between 0.0 and 1.0."
        assert isinstance(self.sac2fix, float) and (
            0.0 < self.sac2fix < 1.0
        ), "'sac2fix' must be float between 0.0 and 1.0."

    def _get_distribution(self, ed, ep):
        if ed == "norm":
            return scipy.stats.norm(**ep)

        raise NotImplementedError(f"Distribution '{ed}' is not supported.")

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        points = X[[self.x, self.y]].values
        dist = self._get_distances(points, self.distance)

        x = X[self.x].values
        y = X[self.y].values
        t = X[self.t].values

        dt = np.diff(t)
        vel = dist / (dt + self.eps)

        states = ("fixation", "saccade")
        start_probs = {"fixation": 0.5, "saccade": 0.5}

        transition_probs = {
            "fixation": {"fixation": 1 - self.fix2sac, "saccade": self.fix2sac},
            "saccade": {"fixation": self.sac2fix, "saccade": 1 - self.sac2fix},
        }

        # fixations have lower velocities than saccades
        if self.distrib_params == "auto":
            # TODO commented approach does not work
            # quantiles = np.quantile(vel, q=[0.15, 0.35, 0.55, 0.85])
            # fixation_mean = quantiles[0]
            # fixation_std = vel[vel <= quantiles[1]].std()
            #
            # dp = {
            #     "fixation": {
            #         "loc": fixation_mean,
            #         "scale": fixation_std
            #     },
            #     "saccade": {
            #         "loc": vel.mean(),
            #         "scale": vel.std()
            #     }
            # }
            dp = {
                "fixation": {"loc": 0.002, "scale": 0.001},
                "saccade": {"loc": 0.03, "scale": 0.005},
            }
        else:
            dp = self.distrib_params

        fix_vel_distr = self._get_distribution(self.fix_distrib, dp["fixation"])
        sac_vel_distr = self._get_distribution(self.sac_distrib, dp["saccade"])

        emission_probs = {
            "fixation": lambda velocity: 1 - fix_vel_distr.cdf(velocity),
            "saccade": lambda velocity: sac_vel_distr.cdf(velocity),
        }

        _, best_path = self._viterbi(
            observations=vel,
            states=states,
            sp=start_probs,
            tp=transition_probs,
            ep=emission_probs,
        )

        is_fixation = (best_path == "fixation").astype(np.int32)

        fixations = self._squash_fixations(is_fixation)

        fixations_df = pd.DataFrame(
            data={
                "fixation_id": fixations,
                self.x: x[:-1],
                self.y: y[:-1],
                "start_time": t[:-1],
                "end_time": t[:-1],
                "distance_min": dist,
                "distance_max": dist,
            }
        )

        fixations_df = fixations_df[fixations_df["fixation_id"] != 0]
        diameters = []
        centers = []

        for i in set(fixations):
            if i != 0:
                points = fixations_df.loc[
                    fixations_df["fixation_id"] == i,
                    fixations_df.columns.isin([self.x, self.y]),
                ].values
                x, y, radius = _get_MEC(np.unique(points, axis=0))
                diameters.append(radius * 2)
                centers.append(np.array([x, y]))

        fixations_df = fixations_df.groupby(by=["fixation_id"]).agg(
            {
                self.x: "mean",
                self.y: "mean",
                "start_time": "min",
                "end_time": "max",
                "distance_min": "min",
                "distance_max": "max",
            }
        )

        fixations_df["diameters"] = diameters
        fixations_df["centers"] = centers

        # default features
        feats = (
            "duration",
            "saccade_duration",
            "saccade_length",
            "saccade_angle",
            "saccade2_angle",
        )
        fixations_df = self._compute_feats(fixations_df, feats)

        return fixations_df

    def _viterbi(self, observations, states, sp, tp, ep):
        """Computes hidden states vector Q s.t. probability
        of observing vector 'observations' given sp, tp, and ep is maximum.\n
        st ~ states\n
        sp ~ start_probs\n
        tp ~ transition_probs\n
        ep ~ emission_probs\n
        observations ~ velocities\n
        """
        prev_probs_nlog = None
        best_path = []
        if len(observations) == 0:
            return None

        for i, o_i in enumerate(observations):
            cur_probs_nlog = {}

            for j, q in enumerate(states):
                if i == 0:
                    q_prob_nlog_max = -np.log(max(sp[q], self.eps))
                else:
                    q_prob_nlog_max = max(
                        [prev_probs_nlog[q_] - np.log(tp[q_][q]) for q_ in states]
                    )

                cur_probs_nlog[q] = -np.log(max(ep[q](o_i), self.eps)) + q_prob_nlog_max

            best_path.append(max(cur_probs_nlog.keys(), key=cur_probs_nlog.get))
            prev_probs_nlog = cur_probs_nlog

        best_path_prob = max(np.exp(-prev_probs_nlog[q]) for q in states)
        return best_path_prob, np.array(best_path)
