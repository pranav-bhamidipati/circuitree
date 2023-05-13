from functools import cached_property, partial
import h5py
from itertools import product, permutations
from math import ceil
from pathlib import Path
from typing import Any, Literal, Optional, Iterable, Mapping
from biocircuits import gillespie_ssa
from numba import njit
import numpy as np
from numpy.typing import NDArray
from sacred import Experiment
from scipy.signal import correlate

from circuitree import SimpleNetworkTree
from circuitree.rewards import (
    sequential_reward,
    sequential_reward_and_modularity_estimate,
    mcts_reward,
    mcts_reward_and_modularity_estimate,
)

_default_parameters = dict(
    k_on=1.0,
    k_off_1=224.0,
    k_off_2=9.0,
    km_unbound=0.5,
    km_act=1.0,
    km_rep=5e-4,
    km_act_rep=0.5,
    kp=0.167,
    gamma_m=0.005776,
    gamma_p=0.001155,
)

_param_ranges = dict(
    log10_k_on=(-1, 2),
    log10_k_off_1=(0, 3),
    k_off_2_1_ratio=(0, 1),
    km_unbound=(0, 1),
    km_act=(1, 10),
    nlog10_km_rep_unbound_ratio=(0, 5),
    kp=(0.015, 0.25),
    nlog10_gamma_m=(1, 3),
    nlog10_gamma_p=(1, 3),
)


class OscillationTree(SimpleNetworkTree):
    def __init__(
        self,
        time_points: NDArray[np.float64],
        success_threshold: float = 0.005,
        autocorr_threshold: float = 0.3,
        init_params: tuple[float, ...] = (10.0,),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.time_points = time_points
        self.init_params = init_params
        self.autocorr_threshold = autocorr_threshold
        self.success_threshold = success_threshold

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
        components, interactions = genotype.split("::")
        recolorings = self.get_recolorings(genotype)
        genotype_unique = min(recolorings)

        return genotype_unique

    def get_reward(self, state) -> float | int:
        params = self.draw_param_set()
        mod = TFNetworkModel(
            genotype=state,
            params=tuple(params.values()),
            init_method="poisson",
            init_params=self.init_params,
        )
        frequencies, second_peaks = mod.run_ssa_and_get_secondary_autocorrelation_peaks(
            t=self.time_points,
            n_threads=self.batch_size,
        )

        win = max(second_peaks) > self.autocorr_threshold

        return int(win)

    def is_success(self, state: str) -> bool:
        payout = self.graph.nodes[state]["reward"]
        visits = self.graph.nodes[state]["visits"]
        return visits > 0 and payout / visits > self.success_threshold

    def draw_param_set(
        self, param_ranges: Optional[dict[str, tuple[float | int]]] = None
    ) -> tuple[float]:
        if param_ranges is None:
            param_ranges = _param_ranges

        # Make random draws for each quantity needed to define a parameter set
        draws = {q: self.rg.uniform(l, h) for q, (l, h) in param_ranges.items()}

        # Calculate derived parameters
        params = {
            "k_on": 10 ** draws["log10_k_on"],
            "k_off_1": 10 ** draws["log10_k_off_1"],
            "k_off_2": draws["log10_k_off_1"] * draws["k_off_2_1_ratio"],
            "km_unbound": draws["km_unbound"],
            "km_act": draws["km_act"],
            "km_rep": draws["km_unbound"] * 10 ** -draws["nlog10_km_rep_unbound_ratio"],
            "km_act_rep": draws[
                "km_unbound"
            ],  # Act and rep together lead to no net effect
            "kp": draws["kp"],
            "gamma_m": 10 ** -draws["nlog10_gamma_m"],
            "gamma_p": 10 ** -draws["nlog10_gamma_p"],
        }

        return params


def autocorrelate(data: NDArray[np.float_]) -> NDArray[np.float_]:
    norm = data - data.mean()
    acorr = correlate(norm, norm, mode="same")[len(norm) // 2 :]
    acorr = acorr / acorr.max()
    return acorr


def find_secondary_peak(acorr: NDArray[np.float_]) -> float:
    """Find the secondary peak in the autocorrelation function."""
    nt = len(acorr)
    cross_generator = (
        i + 1 for i in range(nt - 1) if np.sign(acorr[i]) != np.sign(acorr[i + 1])
    )
    where_neg = next(cross_generator, -1)
    if where_neg == -1:
        return -1, 0.0
    where_pos = next(cross_generator, -1)
    if where_pos == -1:
        return -1, 0.0
    else:
        where_max = where_pos
        where_max += np.argmax(acorr[where_pos:])
        return where_max, acorr[where_max]


def parse_genotype(genotype: str):
    if not genotype.startswith("*"):
        raise ValueError(
            f"Assembly incomplete. Genotype {genotype} is not a terminal genotype."
        )
    components, interaction_codes = genotype.strip("*").split("::")
    component_indices = {c: i for i, c in enumerate(components)}

    interactions = interaction_codes.split("_")

    activations = []
    inhbitions = []
    for left, ixn, right in interactions:
        if ixn.lower() == "a":
            activations.append((component_indices[left], component_indices[right]))
        elif ixn.lower() == "i":
            inhbitions.append((component_indices[left], component_indices[right]))

    activations = np.array(activations, dtype=int)
    inhbitions = np.array(inhbitions, dtype=int)

    return components, activations, inhbitions


class TFNetworkModel:
    """

    10 parameters
    =============
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
        params: Optional[tuple[float]] = None,
        rg: np.random.BitGenerator = None,
        seed: int = 2023,
        init_method: str = "poisson",
        init_params: tuple[float] = (10.0,),
        **kwargs,
    ):
        self.genotype = genotype

        components, activations, inhibitions = parse_genotype(genotype)
        self.components = components
        self.activations = activations
        self.inhibitions = inhibitions
        self.n_components = len(components)
        self.n_activations = len(activations)
        self.n_inhibitions = len(inhibitions)

        if params is None:
            self.param_dict = _default_parameters
            self.params = tuple(self.param_dict.values())
        else:
            self.params = params
            self.param_dict = dict(zip(_default_parameters.keys(), self.params))

        if rg is None:
            self.rg = np.random.default_rng(seed)
        else:
            self.rg = rg

        self.population = self.get_initial_population(
            init_method=init_method,
            init_params=init_params,
            **kwargs,
        )

        if activations.size > 0:
            activations_left = activations[:, 0]
            activations_right = activations[:, 1]
        else:
            activations_left = np.array([], dtype=int)
            activations_right = np.array([], dtype=int)

        if inhibitions.size > 0:
            inhibitions_left = inhibitions[:, 0]
            inhibitions_right = inhibitions[:, 1]
        else:
            inhibitions_left = np.array([], dtype=int)
            inhibitions_right = np.array([], dtype=int)

        self._result: Optional[tuple[Any]] = None

        @njit
        def _propensity_func(
            propensities,
            population,
            t,
            *params,
        ):
            """
            The propensity function returns an array of propensities for each reaction.
            For M TFs, and A activation interactions, and R repression interactions,
            there are 6M + 2A + 2R elementary reactions.
            The ordering and indexing of the reactions is as follows:
                - Transcription of mRNA (0, M)
                - Translation of mRNA (M, 2M)
                - Degradation of bound activator (2M, 3M)
                - Degradation of bound inhibitor (3M, 4M)
                - Degradation of mRNA (4M, 5M)
                - Degradation of unbound TF (5M, 6M)
                - Binding of activator to promoter (6M, 6M + A)
                - Unbinding of activator from promoter (6M + A, 6M + 2A)
                - Binding of inhibitor to promoter (6M + 2A, 6M + 2A + R)
                - Unbinding of inhibitor from promoter (6M + 2A + R, 6M + 2A + 2R)
            """
            (
                k_on,
                k_off_1,
                k_off_2,
                km_unbound,
                km_act,
                km_rep,
                km_act_rep,
                kp,
                gamma_m,
                gamma_p,
            ) = params

            m = len(population) // 4
            a = len(activations)
            r = len(inhibitions)
            m6 = 6 * m
            a2 = 2 * a
            r2 = 2 * r

            # Get bound activators, bound repressors, mRNA, protein for each TF
            pop = np.reshape(population, (m, 4))
            a_s, r_s, m_s, p_s = pop.T
            n_bound = a_s + r_s

            # Transcription rates are stored in a nested list
            # First layer is number of activators and second layer is number of repressors
            k_tx = [[km_unbound, km_rep, km_rep], [km_act, km_act_rep], [km_act]]

            # Transcription of mRNA
            propensities[:m] = [k_tx[a][r] for a, r, m, p in pop]

            # Translation of mRNA
            propensities[m : 2 * m] = kp * m_s

            # Degradation of bound activator
            propensities[2 * m : 3 * m] = gamma_m * a_s

            # Degradation of bound inhibitor
            propensities[3 * m : 4 * m] = gamma_p * r_s

            # Degradation of mRNA
            propensities[4 * m : 5 * m] = gamma_m * m_s

            # Degradation of unbound TF
            propensities[5 * m : 6 * m] = gamma_p * p_s

            # Binding of activator to promoter
            if activations.size > 0:
                propensities[m6 : m6 + a] = (
                    k_on * p_s[activations_left] * (n_bound[activations_right] < 2)
                )

                # Unbinding of activator from promoter
                # (different rates for 1 or 2 activators bound)
                propensities[m6 + a : m6 + a2] = (
                    k_off_1 * (a_s[activations_right] == 1)
                ) + (k_off_2 * (a_s[activations_right] == 2))

            if inhibitions.size > 0:
                # Binding of inhibitor to promoter
                propensities[m6 + a2 : m6 + a2 + r] = (
                    k_on * p_s[inhibitions_left] * (n_bound[inhibitions_right] < 2)
                )
                # Unbinding of inhibitor from promoter
                propensities[m6 + a2 + r : m6 + a2 + r2] = (
                    k_off_1 * (r_s[inhibitions_right] == 1)
                ) + (k_off_2 * (r_s[inhibitions_right] == 2))

            return propensities

        self.propensity_func = _propensity_func
        self.update_matrix = self.get_update_matrix()

    @property
    def result(self):
        return self._result

    def get_initial_population(
        self,
        init_method: Literal["poisson", "equal"] = "poisson",
        init_params: tuple[Any] = (10.0,),
        init_num: int = 10,
        **kwargs,
    ):
        """
        The population is a tuple of with 4 entries for each component:
            - activators bound to its promoter
            - repressors bound to its promoter
            - mRNAs
            - proteins
        """
        population = np.zeros(4 * self.n_components, dtype=int)
        if init_method == "poisson":
            population[2::4] = self.rg.poisson(*init_params, size=self.n_components)
            population[3::4] = self.rg.poisson(*init_params, size=self.n_components)
        elif init_method == "equal":
            population[2::4] = init_num
            population[3::4] = init_num
        else:
            raise NotImplementedError(f"init_method {init_method} not implemented")

        return population

    def get_update_matrix(self):
        """
        Generate the update matrix for the system. Describes the change in each
        species for each reaction.
        """
        m = self.n_components
        a = len(self.activations)
        r = len(self.inhibitions)
        m6 = 6 * m
        a2 = 2 * a
        r2 = 2 * r
        U = np.zeros((m6 + a2 + r2, 4 * m), dtype=int)

        for i in range(m):
            U[0 * m + i, 2 + 4 * i] = 1  # transcription
            U[1 * m + i, 3 + 4 * i] = 1  # translation
            U[2 * m + i, 0 + 4 * i] = -1  # bound activator degradation
            U[3 * m + i, 1 + 4 * i] = -1  # bound inhibitor degradation
            U[4 * m + i, 2 + 4 * i] = -1  # mRNA degradation
            U[5 * m + i, 3 + 4 * i] = -1  # unbound protein degradation

        for j, (left, right) in enumerate(self.activations):
            U[m6 + j, 3 + 4 * left] = -1  # activator binding
            U[m6 + j, 0 + 4 * right] = 1  # activator binding

            U[m6 + a + j, 0 + 4 * right] = -1  # activator unbinding
            U[m6 + a + j, 3 + 4 * left] = 1  # activator unbinding

        for k, (left, right) in enumerate(self.inhibitions):
            U[m6 + a2 + k, 3 + 4 * left] = -1  # inhibitor binding
            U[m6 + a2 + k, 1 + 4 * right] = 1  # inhibitor binding

            U[m6 + a2 + r + k, 1 + 4 * right] = -1  # inhibitor unbinding
            U[m6 + a2 + r + k, 3 + 4 * left] = 1  # inhibitor unbinding

        return U

    def run_ssa(
        self,
        t: Iterable[float],
        params: Optional[tuple[float]] = None,
        param_updates: Optional[dict] = None,
        size: int = 1,
        n_threads: int = 1,
        progress_bar: bool = False,
        **kwargs,
    ):
        """
        Run the stochastic simulation algorithm for the system.

        Parameters
        ----------
        t : array-like
            Time points to evaluate the system at.
        params: tuple, optional
            Parameters to use for the simulation.
        param_updates : dict, optional
            Dictionary of parameters to update for the simulation. Used to update
            the `params` variable.
        kwargs : dict, optional
            Keyword arguments to pass to the simulation.

        Returns
        -------
        t : array-like
            Time points evaluated at.
        x : array-like
            Species values at each time point.
        """

        params = self.params if params is None else params

        if param_updates is not None:
            params = tuple(
                param_updates.get(p, v) for p, v in zip(self.param_dict.keys(), params)
            )

        pop = gillespie_ssa(
            self.propensity_func,
            self.update_matrix,
            self.population,
            t,
            args=params,
            size=size,
            n_threads=n_threads,
            progress_bar=progress_bar,
            **kwargs,
        )

        self._result = t, pop

        return t, pop

    def get_autocorrelation(self, pop: Optional[np.ndarray] = None, **kwargs):
        """
        Get the autocorrelation of the system.
        """
        if pop is None:
            t, pop = self.result
        autocorr_func = partial(autocorrelate)
        return np.apply_along_axis(autocorr_func, 1, pop[:, :, 3::4])

    def get_secondary_autocorrelation_peaks(
        self,
        t: Optional[np.ndarray] = None,
        pop: Optional[np.ndarray] = None,
        indices: bool = False,
        **kwargs,
    ):
        """
        Get the location and height of the secondary autocorrelation peaks.
        """

        if t is None:
            t, _ = self.result

        # Compute autocorrelation (pulls from self.result if pop is None)
        acorr = self.get_autocorrelation(pop)
        nthreads = acorr.shape[0]

        # Compute the location and height of the second peak
        second_peaks = np.apply_along_axis(find_secondary_peak, 1, acorr)
        has_peak = np.any(second_peaks[:, 0, :] > 0, axis=-1)
        where_peak = np.where(has_peak, np.argmax(second_peaks[:, 1, :], axis=-1), -1)

        second_peak_loc = np.where(
            has_peak, second_peaks[np.arange(nthreads), 0, where_peak], -1
        ).astype(int)
        second_peak_val = np.where(
            has_peak, second_peaks[np.arange(nthreads), 1, where_peak], 0.0
        )

        second_peak_freq = np.where(has_peak, 1 / t[second_peak_loc], 0.0)

        if indices:
            return second_peak_loc, second_peak_freq, second_peak_val
        else:
            return second_peak_freq, second_peak_val

    def run_ssa_and_get_secondary_autocorrelation_peaks(
        self,
        t: Iterable[float],
        params: Optional[tuple[float]] = None,
        param_updates: Optional[dict] = None,
        size: int = 1,
        n_threads: int = 1,
        progress_bar: bool = False,
        indices: bool = False,
        **kwargs,
    ):
        """
        Run the stochastic simulation algorithm for the system and get the
        secondary autocorrelation peaks.
        """

        self.run_ssa(
            t,
            params=params,
            param_updates=param_updates,
            size=size,
            n_threads=n_threads,
            progress_bar=progress_bar,
            **kwargs,
        )

        return self.get_secondary_autocorrelation_peaks(indices=indices)


def search_sequential(
    components: Iterable[Iterable[str]],
    interactions: Iterable[str],
    time_points: NDArray[np.float64],
    n_samples_per_topology: int,
    estimate_modularity: bool = False,
    success_threshold: float = 0.005,
    max_samples: int = 10000000,
    seed: Optional[int] = None,
    rg: Optional[np.random.Generator] = None,
    **kwargs,
):
    if rg is None:
        if seed is None:
            raise ValueError("Must specify random seed if rg is not specified")
        else:
            rg = np.random.default_rng(seed)
    else:
        if seed is None:
            seed = rg.bit_generator._seed_seq.entropy

    ot = OscillationTree(
        components=components,
        interactions=interactions,
        time_points=time_points,
        batch_size=1,
        success_threshold=success_threshold,
        **kwargs,
    )

    if estimate_modularity:
        metric_func = partial(sequential_reward_and_modularity_estimate, ot.root)

        results = ot.search_bfs(
            1,
            n_samples_per_topology,
            max_steps=max_samples,
            metric_func=metric_func,
            shuffle=True,
        )

        rewards, modularity_estimates = zip(*results[1:])
        rewards = np.array(rewards, dtype=int)
        modularity_estimates = np.array(modularity_estimates, dtype=float)
        data = {"modularity_estimates": modularity_estimates}

    else:
        metric_func = sequential_reward

        rewards = ot.search_bfs(
            1,
            n_samples_per_topology,
            max_steps=max_samples,
            metric_func=metric_func,
            shuffle=True,
        )[1:]

        data = {}

    data["n_per_topology"] = n_samples_per_topology
    data["N"] = max_samples
    data["seed"] = seed
    data["rewards"] = rewards
    data["final_modularity"] = ot.modularity

    return data


def search_mcts(
    components: Iterable[Iterable[str]],
    interactions: Iterable[str],
    time_points: NDArray[np.float64],
    N: int,
    batch_size: int = 5,
    estimate_modularity: bool = False,
    success_threshold: float = 0.005,
    seed: Optional[int] = None,
    rg: Optional[np.random.Generator] = None,
    root: str = "ABC::",
    **kwargs,
):
    if rg is None:
        if seed is None:
            raise ValueError("Must specify random seed if rg is not specified")
        else:
            rg = np.random.default_rng(seed)
    else:
        if seed is None:
            seed = rg.bit_generator._seed_seq.entropy

    ot = OscillationTree(
        components=components,
        interactions=interactions,
        time_points=time_points,
        batch_size=batch_size,
        success_threshold=success_threshold,
        root=root,
        **kwargs,
    )

    n_iterations = ceil(N / batch_size)

    if estimate_modularity:
        metric_func = partial(mcts_reward_and_modularity_estimate, ot.root)
        results = ot.search_mcts(n_iterations, metric_func=metric_func)[1:]

        rewards, modularity_estimates = zip(*results)
        rewards = np.array(rewards, dtype=int)
        modularity_estimates = np.array(modularity_estimates, dtype=float)

        data = {"modularity_estimates": modularity_estimates}

    else:
        metric_func = mcts_reward
        rewards = ot.search_mcts(n_iterations, metric_func=metric_func)[1:]
        data = {}

    data["N"] = N
    data["batch_size"] = batch_size
    data["seed"] = seed
    data["rewards"] = rewards
    data["final_modularity"] = ot.modularity

    return data


def oscillator_search(
    components: Iterable[Iterable[str]],
    interactions: Iterable[str],
    N: int,
    nt: int,
    dt: float = 30.0,  # seconds
    n_samples_per_topology: int = 100,
    method: Literal["mcts", "sequential"] = "mcts",
    seed: int = 2023,
    save: bool = False,
    estimate_modularity: bool = False,
    ex: Optional[Experiment] = None,
    **kwargs,
):
    if method == "mcts":
        search = search_mcts
    elif method == "sequential":
        search = search_sequential
    else:
        raise ValueError(f"Unknown search method: {method}")

    rg = np.random.default_rng(seed)

    t = np.arange(0, nt * dt, nt)
    data = search(
        components,
        interactions,
        t,
        N=N,
        n_samples_per_topology=n_samples_per_topology,
        rg=rg,
        estimate_modularity=estimate_modularity,
        **kwargs,
    )

    if ex is not None:
        artifacts = []

        save_dir = Path(ex.observers[0].dir)

        if save:
            p = save_dir.joinpath("results.hdf5")
            # print(f"Writing to: {p.resolve().absolute()}")

            with h5py.File(p, "w") as f:
                for k, v in data.items():
                    f.create_dataset(k, data=v)

            artifacts.append(p)

        for a in artifacts:
            ex.add_artifact(a)
