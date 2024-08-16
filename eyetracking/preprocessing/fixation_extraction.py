from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy
from numba import jit

from eyetracking.preprocessing._utils import _get_distance
from eyetracking.preprocessing.base import BaseFixationPreprocessor

from eyetracking.preprocessing._utils import _get_MEC


# ======== FIXATION PREPROCESSORS ========
class IVT(BaseFixationPreprocessor):
    """
    Velocity Threshold Identification.
    Complexity: O(n) for single group.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str,
        threshold: float,
        distance: str = "euc",
        pk: List[str] = None,
        eps: float = 1e-10,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.threshold = threshold
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
        fixations_df["duration"] = fixations_df.end_time - fixations_df.start_time

        return fixations_df


class IDT(BaseFixationPreprocessor):
    """
    Dispersion Threshold Identification.
    Complexity: O(n * W)  for single group, where W is size of maximum window.
    Worst case is O(n^2) for W = n.
    """

    def __init__(
        self,
        x: str,
        y: str,
        t: str,
        min_duration: float,
        max_dispersion: float,
        distance: str = "euc",  # norm in R^2 for distance calculation
        pk: List[str] = None,
        eps: float = 1e-20,
    ):
        super().__init__(x=x, y=y, t=t, pk=pk)
        self.min_duration = min_duration
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
        assert self.min_duration > 0, f"'min_duration' must be non-negative."
        assert self.max_dispersion is not None, self._err_no_field(m, "min_duration")
        assert self.max_dispersion > 0, f"'max_dispersion' must be non-negative."

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

        n = len(vel)
        is_fixation = np.zeros(n)

        cur = 0
        last = 0  # [cur, last]
        disp = np.full(n, -np.inf)

        while cur < n:
            # initiate time for window
            end_time = t[cur] + self.min_duration

            # invariant:
            # 1. current window's dispersion <= max_dispersion
            # 2. current window's duration < min_duration
            # so, algorithm classifies first correct window as fixation
            # TODO: implement approaches: widest possible window,
            #  some other characteristic

            # seek for first point to exceed end_time
            skip_gazes = False
            window_disp = -np.inf
            while last < n - 1 and t[last] <= end_time:
                next_idx = last + 1
                next_point = np.array(x[next_idx], y[next_idx])

                # update distances in window
                next_point_disp = -np.inf
                j = (
                    -1
                )  # index of gaze with which 'last'-th gaze achieved max dispersion
                for i in range(cur, next_idx):  # [cur, next_idx) == [cur, last]
                    d = _get_distance(
                        np.array(x[i], y[i]), next_point, distance=self.distance
                    )

                    disp[i] = max(disp[i], d)
                    if next_point_disp < d:
                        next_point_disp = d
                        j = i

                # check for dispersion
                if next_point_disp > self.max_dispersion:
                    # gazes [cur, j] cannot be part of fixation -> skip them
                    disp[j + 1 : next_idx] = -np.inf
                    cur = j + 1
                    last = j + 1
                    skip_gazes = True
                    break

                window_disp = max(window_disp, next_point_disp)
                disp[next_idx] = next_point_disp
                last = next_idx  # last = last + 1

            if skip_gazes:
                continue

            # if (last + 1 == len(df)) and there is not enough points
            if t[last] <= end_time:
                break

            # here we have correct window with duration >= min_duration
            # and dispersion <= max_dispersion.
            # Window could be extended further.
            is_fixation[cur : last + 1] = 1  # [cur, last] is single fixation
            disp[cur : last + 1] = window_disp
            cur = last + 1
            last = last + 1

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
                "dispersion": disp,
            }
        )

        fixations_df = fixations_df[fixations_df["fixation_id"] != 0]

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

        fixations_df["duration"] = fixations_df.end_time - fixations_df.start_time

        return fixations_df


class IHMM(BaseFixationPreprocessor):
    """
    Hidden Markov Model Identification.
    Complexity: O(n^2) for single group.
    :param fix2sac: probability of transition from fixation to saccade.
    :param sac2fix: probability of transition from saccade to fixation.
    :param fix_distrib: distribution of fixations.
    :param sac_distrib: distribution of saccades.
    :param distrib_params: 'auto' for default params and dict {"fixation": params1, "saccade": params2}, where
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
        distrib_params: Union[str, Dict[str, float]] = "auto",
        distance: str = "euc",
        pk: List[str] = None,
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
        ), f"'fix2sac' must be float between 0.0 and 1.0."
        assert isinstance(self.sac2fix, float) and (
            0.0 < self.sac2fix < 1.0
        ), f"'sac2fix' must be float between 0.0 and 1.0."

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

        if self.distrib_params == "auto":
            m, s = vel.mean(), vel.std()
            dp = {"fixation": {"loc": m, "scale": s}, "saccade": {"loc": m, "scale": s}}
        else:
            dp = self.distrib_params

        fix_vel_distr = self._get_distribution(self.fix_distrib, dp["fixation"])
        sac_vel_distr = self._get_distribution(self.sac_distrib, dp["saccade"])

        emission_probs = {
            "fixation": lambda velocity: fix_vel_distr.cdf(velocity),
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

        fixations_df["duration"] = fixations_df.end_time - fixations_df.start_time

        return fixations_df

    def _viterbi(self, observations, states, sp, tp, ep):
        """
        Computes hidden states vector Q s.t. probability
        of observing vector 'observations' given sp, tp, and ep is maximum.
        st ~ states
        sp ~ start_probs
        tp ~ transition_probs
        ep ~ emission_probs
        observations ~ velocities
        """
        prev_probs = None
        best_path = []
        if len(observations) == 0:
            return None

        for i, o_i in enumerate(observations):
            cur_probs = {}

            for j, q in enumerate(states):
                if i == 0:
                    q_prob_max = -np.log(sp[q])
                else:
                    q_prob_max = min(
                        prev_probs[q_] - np.log(tp[q_][q]) for q_ in states
                    )

                cur_probs[q] = -np.log(max(ep[q](o_i), self.eps)) + q_prob_max

            best_path.append(min(cur_probs, key=cur_probs.get))
            prev_probs = cur_probs

        best_path_prob = max(np.exp(-cur_probs[q]) for q in states)
        return best_path_prob, np.array(best_path)
