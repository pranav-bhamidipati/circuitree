from functools import cached_property, partial
from itertools import permutations
from typing import Callable, Optional, Iterable
from numba import stencil, njit
import numpy as np
from scipy.signal import correlate

from circuitree import SimpleNetworkTree
from circuitree.parallel import DefaultFactoryDict, ParallelTree


from models.oscillation.gillespie import (
    GillespieSSA,
    make_matrices_for_ssa,
    PARAM_RANGES,
    DEFAULT_PARAMS,
)


class TFNetworkModel:
    """
    Model of a transcription factor network.
    =============

    10 parameters
        TF-promoter binding rates:
            k_on
            k_off_1
            k_off_2

            \\ NOTE: k_off_2 less than k_off_1 indicates cooperative binding)

        Transcription rates:
            km_unbound
            km_act
            km_rep
            km_act_rep

        Translation rate:
            kp

        Degradation rates:
            gamma_m
            gamma_p

    """

    def __init__(
        self,
        genotype: str,
        initialize: bool = False,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        **kwargs,
    ):
        self.genotype = genotype

        (
            self.components,
            self.activations,
            self.inhibitions,
        ) = SimpleNetworkTree.parse_genotype(genotype)
        self.m = len(self.components)
        self.a = len(self.activations)
        self.r = len(self.inhibitions)

        # self._params_dict = _default_parameters
        # self.update_params(params or param_updates or {})

        # self.population = self.get_initial_population(
        #     init_method=init_method,
        #     init_params=init_params,
        #     **kwargs,
        # )

        self.dt: Optional[float] = None
        self.nt: Optional[int] = None
        self.t: Optional[Iterable[float]] = None

        self.ssa: GillespieSSA | None = None
        self.dt = dt
        self.nt = nt
        if initialize:
            self.initialize_ssa(dt, nt)

    def initialize_ssa(
        self,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        init_mean: float = 10.0,
        **kwargs
    ):
        dt = dt or self.dt
        nt = nt or self.nt
        t = dt * np.arange(nt)
        self.dt = dt
        self.nt = nt
        self.t = t

        Am, Rm, U = make_matrices_for_ssa(self.m, self.activations, self.inhibitions)
        self.ssa = GillespieSSA(
            self.m,
            Am,
            Rm,
            U,
            dt,
            nt,
            init_mean,
            PARAM_RANGES,
            DEFAULT_PARAMS,
        )

    def run_ssa_with_params(self, pop0, params, tau_leap=False):
        if tau_leap:
            return self.ssa.run_with_params_tau_leap(pop0, params)
        else:
            return self.ssa.run_with_params(pop0, params)

    def run_batch_with_params(self, pop0, params, n, tau_leap=False):
        pop0 = np.asarray(pop0)
        params = np.asarray(params)
        is_vectorized = pop0.ndim == 2 and params.ndim == 2
        if tau_leap:
            if is_vectorized:
                return self.ssa.run_batch_with_params_tau_leap_vector(pop0, params, n)
            else:
                return self.ssa.run_batch_with_params_tau_leap(pop0, params, n)
        else:
            if is_vectorized:
                return self.ssa.run_batch_with_params_vector(pop0, params, n)
            else:
                return self.ssa.run_batch_with_params(pop0, params, n)

    def run_ssa_random_params(self, tau_leap=False):
        if tau_leap:
            pop0, params, y_t = self.ssa.run_random_sample_tau_leap()
        else:
            pop0, params, y_t = self.ssa.run_random_sample()
        pop0 = pop0[self.m : self.m * 2]
        return pop0, params, y_t

    def run_batch_random(self, n_samples, tau_leap=False):
        if tau_leap:
            return self.ssa.run_batch_tau_leap(n_samples)
        else:
            return self.ssa.run_batch(n_samples)

    def run_job(self, **kwargs):
        """
        Run the simulation with random parameters and default time-stepping.
        For convenience, this returns the genotype ("state") and visit number in addition
        to simulation results.
        """
        y_t, pop0, params, peak_height = self.run_ssa_and_get_acf_extrema(
            self.dt, self.nt, size=1, freqs=False, indices=False, **kwargs
        )
        return peak_height, pop0, params

    def run_batch_job(self, batch_size: int, **kwargs):
        """
        Run the simulation with random parameters and default time-stepping.
        For convenience, this returns the genotype ("state") and visit number in addition
        to simulation results.
        """
        y_t, pop0s, param_sets, peak_heights = self.run_ssa_and_get_acf_extrema(
            self.dt, self.nt, size=batch_size, freqs=False, indices=False, **kwargs
        )
        return y_t, pop0s, param_sets, peak_heights

    # def run_batch(self, size, **kwargs):
    #     """Run the simulation with random parameters and default time-stepping."""
    #     return self.run_ssa_and_get_secondary_autocorrelation_peaks(
    #         self.dt, self.nt, size=size, freqs=False, indices=False, **kwargs
    #     )

    # def get_secondary_autocorrelation_peaks(
    #     self,
    #     t: np.ndarray,
    #     pop: np.ndarray,
    #     freqs: bool = True,
    #     indices: bool = False,
    #     **kwargs,
    # ):
    #     """
    #     Get the location and height of the secondary autocorrelation peaks.
    #     """

    #     # Compute autocorrelation
    #     acorr = self.get_autocorrelation(pop)
    #     acorr = acorr.reshape(-1, *acorr.shape[-2:])
    #     n_traj, nt, m = acorr.shape

    #     # Compute the location and height of the second peak
    #     second_peaks = np.apply_along_axis(find_secondary_peak, -2, acorr)
    #     has_peak = np.any(second_peaks[:, 0, :] > 0, axis=-1)
    #     where_peak = np.where(has_peak, np.argmax(second_peaks[:, 1, :], axis=-1), -1)

    #     second_peak_loc = np.where(
    #         has_peak, second_peaks[np.arange(n_traj), 0, where_peak], -1
    #     ).astype(int)
    #     second_peak_val = np.where(
    #         has_peak, second_peaks[np.arange(n_traj), 1, where_peak], 0.0
    #     )
    #     second_peak_freq = np.where(has_peak, 1 / t[second_peak_loc], 0.0)

    #     if freqs and indices:
    #         return second_peak_loc, second_peak_freq, second_peak_val
    #     elif freqs:
    #         return second_peak_freq, second_peak_val
    #     elif indices:
    #         return second_peak_loc, second_peak_val
    #     else:
    #         return second_peak_val

    @staticmethod
    def largest_acf_extremum(y_t: np.ndarray[np.float64 | np.int64]):
        acorrs = autocorrelate_vectorized(y_t)
        return compute_largest_extremum(acorrs)

    @staticmethod
    def get_acf_extrema(
        self,
        t: np.ndarray,
        pop_t: np.ndarray,
        freqs: bool = True,
        indices: bool = False,
    ):
        """
        Get the location and height of the largest extremum of the autocorrelation
        function, excluding the bounds.
        """

        # Compute autocorrelation
        acorrs = autocorrelate_vectorized(pop_t)

        # Compute the location and size of the largest interior extremum over
        # all species
        where_extrema, extrema = compute_largest_extremum_and_loc(acorrs)
        if freqs:
            extrema_freqs = np.where(where_extrema > 0, 1 / t[where_extrema], 0.0)

        squeeze = where_extrema.size == 1
        if squeeze:
            where_extrema = where_extrema.flat[0]
            extrema = extrema.flat[0]
            if freqs:
                extrema_freqs = extrema_freqs.flat[0]

        if freqs:
            if indices:
                return where_extrema, extrema_freqs, extrema
            else:
                return extrema_freqs, extrema
        elif indices:
            return where_extrema, extrema
        else:
            return extrema

    def run_ssa_and_get_acf_extrema(
        self,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        size: int = 1,
        freqs: bool = False,
        indices: bool = False,
        init_mean: float = 10.0,
        tau_leap: bool = False,
        **kwargs,
    ):
        """
        Run the stochastic simulation algorithm for the system and get the
        secondary autocorrelation peaks.
        """

        if (dt is not None) and (nt is not None):
            self.initialize_ssa(dt, nt, init_mean)
            t = self.t

        if size > 1:
            pop0, params, y_t = self.run_batch_random(size, tau_leap=tau_leap)
        else:
            pop0, params, y_t = self.run_ssa_random_params(tau_leap=tau_leap)
        pop0 = pop0[..., self.m : self.m * 2]
        y_t = y_t[..., self.m : self.m * 2]

        if not (freqs or indices):
            results = self.largest_acf_extremum(y_t)
        else:
            results = self.get_acf_extrema(t, y_t, freqs=freqs, indices=indices)

        return y_t, pop0, params, results


def autocorrelate(data: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    norm = data - data.mean()
    acorr = correlate(norm, norm, mode="same")[len(norm) // 2 :]
    acorr = acorr / acorr.max()
    return acorr


def autocorrelate_vectorized(data: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return np.apply_along_axis(autocorrelate, -2, data)


@stencil(cval=False)
def extremum_kernel(a):
    """
    Returns a 1D mask that is True at local extrema, excluding the bounds.
    Computes when the finite difference changes sign (or is zero).
    """
    return (a[0] - a[-1]) * (a[1] - a[0]) <= 0


@njit
def find_extremum(seq: np.ndarray[np.float_]) -> int:
    """
    Find the extremum in a sequence of values with the greatest absolute
    value, excluding the bounds.
    """
    extrema_mask = extremum_kernel(seq)
    if not extrema_mask.any():
        return -1
    else:
        extrema = np.where(extrema_mask)[0]
        return extrema[np.argmax(np.abs(seq[extrema]))]


@njit
def compute_extremum(arr1d: np.ndarray[np.float_]) -> float:
    where_extrema = find_extremum(arr1d)
    if where_extrema < 0:
        return where_extrema, 0.0  # No interior extremum
    else:
        return where_extrema, arr1d[where_extrema]


@njit
def compute_largest_extremum(ndarr: np.ndarray) -> float:
    """
    Get the largest interior extremum of a batch of n 1d arrays, each of length m.
    Vectorizes over arbitrary leading axes. For an input of shape (k, l, m, n),
    k x l x n extrema are calculated, and the max-of-abs is taken over the last axis
    to return an array of shape (k, l).
    """
    nd_shape = ndarr.shape[:-2]
    largest_extrema = np.zeros(nd_shape, dtype=np.float64)
    for leading_index in np.ndindex(nd_shape):
        extrema = np.array([compute_extremum(a)[1] for a in ndarr[leading_index].T])
        largest_extrema[leading_index] = np.max(np.abs(extrema))
    return largest_extrema


@njit
def compute_largest_extremum_and_loc(ndarr: np.ndarray) -> float:
    """
    Get the largest interior extremum of a batch of n 1d arrays, each of length m.
    Vectorizes over arbitrary leading axes. For an input of shape (k, l, m, n),
    k x l x n extrema are calculated, and the max-of-abs is taken over the last axis
    to return an array of shape (k, l).
    Also returns the index of the extremum if it exists, otherwise -1.
    """
    nd_shape = ndarr.shape[:-2]
    where_largest_extrema = np.zeros(nd_shape, dtype=np.int64)
    largest_extrema = np.zeros(nd_shape, dtype=np.float64)
    for leading_index in np.ndindex(nd_shape):
        argmaxval = -1
        maxval = 0.0
        abs_maxval = 0.0
        for a in ndarr[leading_index].T:
            where_extrema, extremum = compute_extremum(a)
            if np.abs(extremum) > abs_maxval:
                argmaxval = where_extrema
                maxval = extremum
                abs_maxval = np.abs(extremum)
        where_largest_extrema[leading_index] = argmaxval
        largest_extrema[leading_index] = maxval
    return where_largest_extrema, largest_extrema


class OscillationTreeBase(SimpleNetworkTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        success_threshold: float = 0.005,
        autocorr_threshold: float = 0.5,
        init_mean: float = 10.0,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        tau_leap: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_points = time_points
        self.autocorr_threshold = autocorr_threshold
        self.success_threshold = success_threshold

        self.dt = dt
        self.nt = nt
        self.init_mean = init_mean
        self.tau_leap = tau_leap

    @cached_property
    def _recolor(self):
        return [dict(zip(self.components, p)) for p in permutations(self.components)]

    @staticmethod
    def _recolor_string(mapping, string):
        return "".join([mapping.get(c, c) for c in string])

    def get_interaction_recolorings(self, genotype: str) -> Iterable[str]:
        if "::" in genotype:
            components, interactions = genotype.split("::")
        else:
            interactions = genotype

        interaction_recolorings = []
        for mapping in self._recolor:
            recolored_interactions = sorted(
                [self._recolor_string(mapping, ixn) for ixn in interactions.split("_")]
            )
            interaction_recolorings.append("_".join(recolored_interactions).strip("_"))

        return interaction_recolorings

    def get_component_recolorings(self, genotype: str) -> Iterable[str]:
        if "::" in genotype:
            components, interactions = genotype.split("::")
        else:
            components = genotype

        component_recolorings = []
        for mapping in self._recolor:
            recolored_components = "".join(
                sorted(self._recolor_string(mapping, components))
            )
            component_recolorings.append(recolored_components)

        return component_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        ris = self.get_interaction_recolorings(genotype)
        rcs = self.get_component_recolorings(genotype)
        recolorings = ["::".join([rc, ri]) for rc, ri in zip(rcs, ris)]

        return recolorings

    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    def is_success(self, state: str) -> bool:
        payout = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and payout / visits > self.success_threshold


class OscillationTree(OscillationTreeBase):
    """Searches the space of TF networks for oscillatory topologies.
    Searches independently, not in parallel.
    """

    def __init__(
        self,
        *args,
        model_table: DefaultFactoryDict = None,
        model_factory: Callable = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_table = model_table or DefaultFactoryDict(
            default_factory=model_factory
        )

    @property
    def model_table(self):
        return self._model_table

    def get_reward(self, state: str, visit_num: int) -> float | int:
        """Run the model and get a random reward"""
        model = self.model_table[state]
        y_t, pop0, params, reward = model.run_ssa_and_get_acf_extrema(
            tau_leap=self.tau_leap, freqs=False, indices=False
        )
        return reward
