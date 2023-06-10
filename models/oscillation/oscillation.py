from functools import cached_property, partial
from itertools import permutations
from typing import Any, Callable, Optional, Iterable
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
        seed: Optional[int] = None,
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
        self.seed = seed
        self.dt = dt
        self.nt = nt
        if initialize:
            if any(arg is None for arg in (seed, dt, nt)):
                raise ValueError("seed, dt and nt must be specified if initialize=True")
            self.initialize_ssa(seed, dt, nt)

    def initialize_ssa(
        self,
        seed: int,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        init_mean: float = 10.0,
        **kwargs,
    ):
        seed = seed or self.seed
        dt = dt or self.dt
        nt = nt or self.nt
        t = dt * np.arange(nt)
        self.dt = dt
        self.nt = nt
        self.t = t

        Am, Rm, U = make_matrices_for_ssa(self.m, self.activations, self.inhibitions)
        self.ssa = GillespieSSA(
            seed,
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
        return pop0, params, y_t

    def run_batch_random(self, n_samples, tau_leap=False):
        if tau_leap:
            return self.ssa.run_batch_tau_leap(n_samples)
        else:
            return self.ssa.run_batch(n_samples)

    def run_job(self, abs: bool = False, **kwargs):
        """
        Run the simulation with random parameters and default time-stepping.
        For convenience, this returns the genotype ("state") and visit number in addition
        to simulation results.
        """
        y_t, pop0, params, reward = self.run_ssa_and_get_acf_minima(
            self.dt, self.nt, size=1, freqs=False, indices=False, abs=abs, **kwargs
        )
        return reward, pop0, params

    def run_batch_job(self, batch_size: int, abs: bool = False, **kwargs):
        """
        Run the simulation with random parameters and default time-stepping.
        For convenience, this returns the genotype ("state") and visit number in addition
        to simulation results.
        """
        y_t, pop0s, param_sets, rewards = self.run_ssa_and_get_acf_minima(
            self.dt,
            self.nt,
            size=batch_size,
            freqs=False,
            indices=False,
            abs=abs,
            **kwargs,
        )
        return y_t, pop0s, param_sets, rewards

    @staticmethod
    def get_acf_minima(y_t: np.ndarray[np.float64 | np.int64], abs: bool = False):
        filtered = filter_ndarray_binomial9(y_t.astype(np.float64))[..., 4:-4, :]
        acorrs = autocorrelate_vectorized(filtered)
        where_minima, minima = compute_lowest_minima(acorrs)
        if abs:
            return np.abs(minima)
        else:
            return minima

    @staticmethod
    def get_acf_minima_and_results(
        t: np.ndarray,
        pop_t: np.ndarray,
        freqs: bool = True,
        indices: bool = False,
        abs: bool = False,
    ):
        """
        Get the location and height of the largest extremum of the autocorrelation
        function, excluding the bounds.
        """

        # Filter out high-frequency (salt-and-pepper) noise
        filtered = filter_ndarray_binomial9(pop_t.astype(np.float64))[..., 4:-4, :]

        # Compute autocorrelation
        acorrs = autocorrelate_vectorized(filtered)

        # Compute the location and size of the largest interior extremum over
        # all species
        where_minima, minima = compute_lowest_minima(acorrs)
        if abs:
            minima = np.abs(minima)

        tdiff = t - t[0]
        if freqs:
            minima_freqs = np.where(where_minima > 0, 1 / tdiff[where_minima], 0.0)

        squeeze = where_minima.size == 1
        if squeeze:
            where_minima = where_minima.flat[0]
            minima = minima.flat[0]
            if freqs:
                minima_freqs = minima_freqs.flat[0]

        if freqs:
            if indices:
                return where_minima, minima_freqs, minima
            else:
                return minima_freqs, minima
        elif indices:
            return where_minima, minima
        else:
            return minima

    def run_ssa_and_get_acf_minima(
        self,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        seed: Optional[int] = None,
        size: int = 1,
        freqs: bool = False,
        indices: bool = False,
        init_mean: float = 10.0,
        tau_leap: bool = False,
        abs: bool = False,
        **kwargs,
    ):
        """
        Run the stochastic simulation algorithm for the system and get the
        autocorrelation-based reward.
        """

        if all(arg is not None for arg in (seed, dt, nt)):
            self.initialize_ssa(seed, dt, nt, init_mean)
            t = self.t

        if size > 1:
            pop0, params, y_t = self.run_batch_random(size, tau_leap=tau_leap)
        else:
            pop0, params, y_t = self.run_ssa_random_params(tau_leap=tau_leap)

        pop0 = pop0[..., self.m : self.m * 2]
        y_t = y_t[..., self.m : self.m * 2]

        if not (freqs or indices):
            results = self.get_acf_minima(y_t, abs=abs)
        else:
            results = self.get_acf_minima_and_results(
                t, y_t, freqs=freqs, indices=indices, abs=abs
            )

        return y_t, pop0, params, results


def autocorrelate_mean0(arr1d_norm: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    "Autocorrelation of an array with mean 0"
    return correlate(arr1d_norm, arr1d_norm, mode="same")[len(arr1d_norm) // 2 :]


def autocorrelate(data1d: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    arr = data1d - data1d.mean()
    arr = autocorrelate_mean0(arr)
    arr /= arr.max()
    return arr


def autocorrelate_vectorized(
    data: np.ndarray[np.float_], axis=-2
) -> np.ndarray[np.float_]:
    """Compute autocorrelation of 1d signals arranged in an nd array, where `axis` is the
    time axis."""
    ndarr = data - data.mean(axis=axis, keepdims=True)
    ndarr = np.apply_along_axis(autocorrelate_mean0, axis, ndarr)
    ndarr /= ndarr.max(axis=axis, keepdims=True)
    return ndarr


@stencil
def binomial3_kernel(a):
    """Basic 3-point binomial filter."""
    return (a[-1] + a[0] + a[0] + a[1]) / 4


@stencil
def binomial5_kernel(a):
    """Basic 5-point binomial filter."""
    return (a[-2] + 4 * a[-1] + 6 * a[0] + 4 * a[1] + a[2]) / 16


@stencil
def binomial7_kernel(a):
    """Basic 7-point binomial filter."""
    return (
        a[-3] + 6 * a[-2] + 15 * a[-1] + 20 * a[0] + 15 * a[1] + 6 * a[2] + a[3]
    ) / 64


@stencil
def binomial9_kernel(a):
    """9-point binomial filter."""
    return (
        a[-4]
        + 8 * a[-3]
        + 28 * a[-2]
        + 56 * a[-1]
        + 70 * a[0]
        + 56 * a[1]
        + 28 * a[2]
        + 8 * a[3]
        + a[4]
    ) / 256


@njit
def filter_ndarray_binomial5(ndarr: np.ndarray) -> float:
    """Apply a binomial filter to 1d signals arranged in an nd array, where the time axis
    is the second to last axis (``axis = -2``).
    """
    ndarr_shape = ndarr.shape
    leading_shape = ndarr_shape[:-2]
    n = ndarr_shape[-1]
    filtered = np.zeros_like(ndarr)
    for leading_index in np.ndindex(leading_shape):
        for i in range(n):
            arr1d = ndarr[leading_index][:, i]
            filt1d = binomial5_kernel(arr1d)
            filtered[leading_index][:, i] = filt1d
    return filtered


@njit
def filter_ndarray_binomial7(ndarr: np.ndarray) -> float:
    """Apply a binomial filter to 1d signals arranged in an nd array, where the time axis
    is the second to last axis (``axis = -2``).
    """
    ndarr_shape = ndarr.shape
    leading_shape = ndarr_shape[:-2]
    n = ndarr_shape[-1]
    filtered = np.zeros_like(ndarr)
    for leading_index in np.ndindex(leading_shape):
        for i in range(n):
            arr1d = ndarr[leading_index][:, i]
            filt1d = binomial7_kernel(arr1d)
            filtered[leading_index][:, i] = filt1d
    return filtered


@njit
def filter_ndarray_binomial9(ndarr: np.ndarray) -> float:
    """Apply a binomial filter to 1d signals arranged in an nd array, where the time axis
    is the second to last axis (``axis = -2``).
    """
    ndarr_shape = ndarr.shape
    leading_shape = ndarr_shape[:-2]
    n = ndarr_shape[-1]
    filtered = np.zeros_like(ndarr)
    for leading_index in np.ndindex(leading_shape):
        for i in range(n):
            arr1d = ndarr[leading_index][:, i]
            filt1d = binomial9_kernel(arr1d)
            filtered[leading_index][:, i] = filt1d
    return filtered


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


@stencil(cval=False)
def minimum_kernel(a):
    """
    Returns a 1D mask that is True at local minima, excluding the bounds.
    Computes when the finite difference changes sign from - to + (or is zero).
    """
    return (a[0] - a[-1] <= 0) and (a[1] - a[0] >= 0)


@njit
def compute_lowest_minimum(seq: np.ndarray[np.float_]) -> tuple[int, float]:
    """
    Find the minimum in a sequence of values with the greatest absolute
    value, excluding the bounds.
    """
    minimum_mask = minimum_kernel(seq)
    if not minimum_mask.any():
        return -1, 0.0
    else:
        minima = np.where(minimum_mask)[0]
        where_lowest_minimum = minima[np.argmin(seq[minima])]
        return where_lowest_minimum, seq[where_lowest_minimum]


@njit
def compute_lowest_minima(ndarr: np.ndarray) -> float:
    """
    Get the lowest interior minimum of a batch of n 1d arrays, each of length m.
    Vectorizes over arbitrary leading axes. For an input of shape (k, l, m, n),
    k x l x n minima are calculated, and the min-of-minima is taken over the last axis
    to return an array of shape (k, l).
    Also returns the index of the minimum if it exists, otherwise -1.
    """
    nd_shape = ndarr.shape[:-2]
    where_largest_minima = np.zeros(nd_shape, dtype=np.int64)
    largest_minima = np.zeros(nd_shape, dtype=np.float64)
    for leading_index in np.ndindex(nd_shape):
        argmin = -1
        minval = 0.0
        for a in ndarr[leading_index].T:
            where_minimum, minimum = compute_lowest_minimum(a)
            if minimum < minval:
                minval = minimum
                argmin = where_minimum
        where_largest_minima[leading_index] = argmin
        largest_minima[leading_index] = minval
    return where_largest_minima, largest_minima


class OscillationTree(SimpleNetworkTree):
    def __init__(
        self,
        time_points: Optional[np.ndarray[np.float64]] = None,
        success_threshold: float = 0.005,
        autocorr_threshold: float = 0.5,
        init_mean: float = 10.0,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        tau_leap: bool = False,
        batch_size: int = 1,
        results_table: Optional[Any] = None,
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
        self.batch_size = batch_size

        if results_table is not None:
            self._results_table = results_table
        else:
            self._results_table = DefaultFactoryDict(default_factory=list)

    @property
    def results_table(self):
        return self._results_table

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

    def get_reward(
        self,
        state: str,
        batch_size: Optional[int] = None,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        tau_leap: Optional[bool] = None,
        record: bool = False,
    ) -> float:
        dt = dt if dt is not None else self.dt
        nt = nt if nt is not None else self.nt
        tau_leap = tau_leap if tau_leap is not None else self.tau_leap
        batch_size = batch_size if batch_size is not None else self.batch_size

        model = TFNetworkModel(state, initialize=True, dt=dt, nt=nt)
        y_t, pop0s, param_sets, rewards = model.run_batch_job(
            batch_size, abs=True, tau_leap=tau_leap
        )

        if record:
            self.results_table[state].append((pop0s, param_sets, rewards))

        return np.mean(rewards)
