from numba import njit, float64, int64, float32, int32
from numba.typed import List
from numba.experimental import jitclass
import numpy as np

"""
    This module contains numerical code (largely JIT-compiled) for running a 
    stochastic simulation algorithm using Gillespie sampling.
    
    This was adapted from the code in the `biocircuits` package written by Justin Bois.
    
"""

__all__ = [
    "make_matrices_for_ssa",
    "package_params_for_ssa",
    "DEFAULT_PARAMS",
    "PARAM_RANGES",
    "GillespieSSA",
]


DEFAULT_PARAMS = np.array(
    [
        1.0,  # k_on
        224.0,  # k_off_1
        9.0,  # k_off_2
        0.5,  # km_unbound
        1.0,  # km_act
        5e-4,  # km_rep
        0.5,  # km_act_rep
        0.167,  # kp
        0.005776,  # gamma_m
        0.001155,  # gamma_p
    ],
    dtype=np.float64,
)

PARAM_RANGES = np.array(
    [
        (-1.0, 2.0),  # log10_k_on
        (0.0, 3.0),  # log10_k_off_1
        (0.0, 1.0),  # k_off_2_1_ratio
        (0.0, 1.0),  # km_unbound
        (1.0, 10.0),  # km_act
        (0.0, 5.0),  # nlog10_km_rep_unbound_ratio
        (0.015, 0.25),  # kp
        (1.0, 3.0),  # nlog10_gamma_m
        (1.0, 3.0),  # nlog10_gamma_p
    ],
    dtype=np.float64,
)

# Estimated computational cost of a tau leap vs a Gillespie step
TAU_LEAP_COST = 1.76


def make_matrices_for_ssa(n_components, activations, inhibitions):
    """
    Generates:
    (1) The activation matrix ``Am``, with zeros everywhere except for 1s at the
    indices i, j where the ith TF activates the jth promoter.
    (2) The inhibition matrix ``Rm``, defined similarly
    (3) The update matrix ``U`` for the system. ``U`` is a
    ``(4M + 3A + 3R + 1, 2M + A + R)`` matrix that describes the change in each
    species for each reaction.
        - ``M`` is the number of TFs in the network
        - ``A`` is the number of activating interactions
        - ``R`` is the number of repressive interactions

    Reactions (rows of the update matrix):
        Transcription of mRNA (0, M)
        Translation of mRNA (M, 2M)
        Degradation of mRNA (2M, 3M)
        Degradation of unbound TF (3M, 4M)
        Binding of activator to promoter (4M, 4M + A)
        Unbinding of activator from promoter (4M + A, 4M + 2A)
        Degradation of bound activator (4M + 2A, 4M + 3A)
        Binding of inhibitor to promoter (4M + 3A, 4M + 3A + R)
        Unbinding of inhibitor from promoter (4M + 3A + R, 4M + 3A + 2R)
        Degradation of bound inhibitor (4M + 3A + 2R, 4M + 3A + 3R)
        Null reaction - all zeros (4M + 3A + 3R, 4M + 3A + 3R + 1)

    Species (columns of the update matrix):
        mRNA (0, M)
        Free TFs (M, 2M)
        Activator-promoter complexes (2M, 2M + A)
        Inhibitor-promoter complexes (2M + A, 2M + A + R)

    """
    m = n_components
    a = len(activations)
    r = len(inhibitions)
    m2 = 2 * m
    m3 = 3 * m
    m4 = 4 * m
    m6 = 6 * m
    a2 = 2 * a
    a3 = 3 * a
    r2 = 2 * r
    r3 = 3 * r
    m4a = m4 + a
    m4a2 = m4 + a2
    m4a3 = m4 + a3
    m4a3r = m4a3 + r
    m4a3r2 = m4a3 + r2
    m4a3r3 = m4a3 + r3
    m2a = m2 + a
    m2ar = m2a + r

    # Activation matrix
    Am = np.zeros((m, m)).astype(np.int64)
    for left, right in activations:
        Am[left, right] = 1

    # INhibition matrix
    Rm = np.zeros((m, m)).astype(np.int64)
    for left, right in inhibitions:
        Rm[left, right] = 1

    # Update matrix
    U = np.zeros((m4a3r3 + 1, m2ar)).astype(np.int64)

    U[:m2, :m2] = np.eye(m2).astype(np.int64)  # transcription/translation
    U[m2:m4, :m2] = -np.eye(m2).astype(np.int64)  # mRNA/free TF degradation

    ## Reactions relating to activation
    for j, (left, right) in enumerate(activations):
        # Binding
        U[m4 + j, m + left] = -1
        U[m4 + j, m2 + j] = 1

        # Unbinding
        U[m4a + j, m + left] = 1
        U[m4a + j, m2 + j] = -1

    # Degradation
    U[m4a2:m4a3, m2:m2a] = -np.eye(a).astype(np.int64)

    ## Reactions relating to inhibition (repression)
    for k, (left, right) in enumerate(inhibitions):
        # Binding
        U[m4a3 + k, m + left] = -1
        U[m4a3 + k, m2a + k] = 1

        # Unbinding
        U[m4a3r + k, m + left] = 1
        U[m4a3r + k, m2a + k] = -1

    # Degradation
    U[m4a3r2:m4a3r3, m2a:] = -np.eye(r).astype(np.int64)

    return Am, Rm, U


@njit
def _sample_discrete(rg, probs, probs_sum):
    q = rg.uniform() * probs_sum
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return rg, i - 1


@njit
def _sum(ar):
    return ar.sum()


@njit
def _draw_time(rg, props_sum):
    return rg, rg.exponential(1 / props_sum)


@njit
def _sample_propensities(rg, propensities):
    """
    Draws a reaction and the time it took to do that reaction.
    """

    props_sum = _sum(propensities)

    # Bail if the sum of propensities is zero
    if props_sum == 0.0:
        rxn = -1
        time = -1.0

    # Compute time and draw reaction from propensities
    else:
        rg, time = _draw_time(rg, props_sum)
        rg, rxn = _sample_discrete(rg, propensities, props_sum)

    return rg, rxn, time


@njit
def package_params_for_ssa(params) -> tuple:
    """Set up reaction propensities in convenient form"""

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

    # Transcription rates are stored in a nested list
    # First layer is number of activators and second layer is number of repressors
    k_tx = np.array(
        [
            [km_unbound, km_rep, km_rep],
            [km_act, km_act_rep, 0.0],
            [km_act, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Promoter binding/unbinding are stored in index-able arrays, where the index
    # is the number of bound species
    k_ons = np.array([k_on, k_on, 0.0]).astype(np.float64)
    k_offs = np.array([0.0, k_off_1, k_off_2]).astype(np.float64)

    # This is the parameter tuple actually used for the propensity function
    # due to added efficiency
    return k_tx, kp, gamma_m, gamma_p, k_ons, k_offs


@njit
def draw_random_initial(rg, m, a, r, poisson_mean):
    m2 = 2 * m
    n_species = m2 + a + r
    pop0 = np.zeros(n_species).astype(np.int64)
    pop0[m:m2] = rg.poisson(poisson_mean, m)

    return rg, pop0


@njit
def draw_random_params(rg, PARAM_RANGES):
    # Make random draws for each quantity needed to define a parameter set
    (
        log10_k_on,
        log10_k_off_1,
        k_off_2_1_ratio,
        km_unbound,
        km_act,
        nlog10_km_rep_unbound_ratio,
        kp,
        nlog10_gamma_m,
        nlog10_gamma_p,
    ) = [rg.uniform(lo, hi) for lo, hi in PARAM_RANGES]

    # Calculate derived parameters
    k_on = 10**log10_k_on
    k_off_1 = 10**log10_k_off_1
    k_off_2 = k_off_1 * k_off_2_1_ratio
    km_rep = km_unbound * 10**-nlog10_km_rep_unbound_ratio
    gamma_m = 10**-nlog10_gamma_m
    gamma_p = 10**-nlog10_gamma_p

    # activation and repression together have no effect on transcription
    km_act_rep = km_unbound

    params = (
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
    )
    return rg, params


@njit
def draw_random_initial_and_params(rg, PARAM_RANGES, m, a, r, poisson_mean):
    rg, pop0 = draw_random_initial(rg, m, a, r, poisson_mean)
    rg, params = draw_random_params(rg, PARAM_RANGES)
    return rg, pop0, params


@njit(fastmath=True)
def add_to_zeros(m, at, vals):
    """Add values to an array of zeros at given indices. Equivalent to `np.add.at`."""
    out = np.zeros((m,)).astype(np.int64)
    for idx, v in zip(at, vals):
        out[idx] += v
    return out


@njit
def get_propensities(
    propensities,
    population,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params,
):
    """
    Returns an array of propensities for each reaction.
    For M TFs, and A activation interactions, and R repression interactions,
    there are 6M + 2A + 2R elementary reactions.
    The ordering and indexing of the reactions is as follows:
        - Transcription of mRNA (0, M)
        - Translation of mRNA (M, 2M)
        - Degradation of mRNA (2M, 3M)
        - Degradation of unbound TF (3M, 4M)
        - Binding of activator to promoter (4M, 4M + A)
        - Unbinding of activator from promoter (4M + A, 4M + 2A)
        - Degradation of bound activator (4M + 2A, 4M + 3A)
        - Binding of inhibitor to promoter (4M + 3A, 4M + 3A + R)
        - Unbinding of inhibitor from promoter (4M + 3A + R, 4M + 3A + 2R)
        - Degradation of bound inhibitor (4M + 3A + 2R, 4M + 3A + 3R)
    """
    (
        k_tx,
        kp,
        gamma_m,
        gamma_p,
        k_ons,
        k_offs,
    ) = ssa_params

    m2 = 2 * m
    m3 = m2 + m
    m4 = m3 + m
    m4a = m4 + a
    m4a2 = m4a + a
    m4a3 = m4a2 + a
    m4a3r = m4a3 + r
    m4a3r2 = m4a3r + r
    m4a3r3 = m4a3r2 + r

    # mRNA and protein for each TF + activators/repressors bound to each promoter
    m_s = population[:m]
    p_s = population[m:m2]
    ap_complexes = population[m2 : m2 + a]
    rp_complexes = population[m2 + a :]

    # Get the number of activators and repressors bound to each promoter
    a_bound = add_to_zeros(m, activations_right, ap_complexes)
    r_bound = add_to_zeros(m, inhibitions_right, rp_complexes)
    n_bound = a_bound + r_bound

    # Transcription
    propensities[:m] = [k_tx[a, b] for a, b in zip(a_bound, r_bound)]

    # Translation
    propensities[m:m2] = kp * m_s

    # mRNA degradation
    propensities[m2:m3] = gamma_m * m_s

    # protein degradation
    propensities[m3:m4] = gamma_p * p_s

    # Activator binding
    propensities[m4:m4a] = p_s[activations_left] * k_ons[n_bound[activations_right]]

    # Activator unbinding
    propensities[m4a:m4a2] = k_offs[ap_complexes]

    # Bound activator degradation
    propensities[m4a2:m4a3] = gamma_p * ap_complexes

    # inhibitor binding
    propensities[m4a3:m4a3r] = p_s[inhibitions_left] * k_ons[n_bound[inhibitions_right]]

    # inhibitor unbinding
    propensities[m4a3r:m4a3r2] = k_offs[rp_complexes]

    # Bound inhibitor degradation
    propensities[m4a3r2:m4a3r3] = gamma_p * rp_complexes

    return propensities


@njit
def take_gillespie_step(
    rg,
    t,
    population,
    event,
    propensities,
    U,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params,
):
    # Update the population
    # On the first loop, population is unchanged
    population += U[event]

    # draw the event and time step
    propensities = get_propensities(
        propensities,
        population,
        activations_left,
        activations_right,
        inhibitions_left,
        inhibitions_right,
        m,
        a,
        r,
        *ssa_params,
    )
    rg, event, dt = _sample_propensities(rg, propensities)

    # Skip to the end of the simulation
    if event == -1:
        t = np.inf
    else:
        # Increment time
        # If t exceeds the next time point, population isn't updated
        t += dt

    return rg, t, population, event


@njit
def gillespie_trajectory(
    rg,
    time_points,
    population_0,
    U,
    m,
    a,
    r,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    *ssa_params,
):
    """ """

    # Initialize output
    nt = len(time_points)
    m4a3r3p1, m2ar = U.shape
    m4a3r3 = m4a3r3p1 - 1
    pop_out = -np.ones((nt, m2ar)).astype(np.int64)

    # Initialize and perform simulation
    propensities = np.zeros(m4a3r3).astype(np.float64)
    population = population_0.copy().astype(np.int64)
    pop_out[0] = population
    j = 0
    t = time_points[0]

    # First loop makes no changes (indexes the all-zero row of update matrix)
    event = m4a3r3
    while j < nt:
        tj = time_points[j]
        while t <= tj:
            rg, t, population, event = take_gillespie_step(
                rg,
                t,
                population,
                event,
                propensities,
                U,
                activations_left,
                activations_right,
                inhibitions_left,
                inhibitions_right,
                m,
                a,
                r,
                *ssa_params,
            )

        # Update the index (Be careful about types for Numba)
        new_j = j + np.searchsorted(time_points[j:], t)

        # Update the population
        pop_out[j:new_j] = population

        # Increment index
        j = new_j

    return rg, pop_out


@njit
def _sample_poisson(rg, propensities, dt):
    """Sample the number of events occurring in the time interval dt
    from a Poisson distribution."""
    n = len(propensities)
    out = np.zeros((n, 1)).astype(np.int64)
    for i, mean_num_events in enumerate(propensities * dt):
        out[i, 0] = rg.poisson(mean_num_events)
    return rg, out


@njit
def get_tau(epsilon, population, mu, sigma2, highest_rxn_order):
    """Compute the time increment for tau-leaping."""

    # No zero-order reactions, so division by zero is not an issue
    times = np.maximum(epsilon * population / highest_rxn_order, 1.0)
    min_mu_time = (times / np.abs(mu)).min()
    min_s2_time = (times**2 / sigma2).min()
    return min(min_mu_time, min_s2_time)


@njit
def take_tau_leap(
    rg,
    population,
    events,
    propensities,
    U,
    Usq,
    epsilon,
    highest_rxn_order,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    m,
    a,
    r,
    *ssa_params,
):
    # Update the population
    population += (U * events).sum(axis=0)
    population = np.maximum(population, 0)  # clip negative values to zero

    # Compute reaction propensities
    propensities = get_propensities(
        propensities,
        population,
        activations_left,
        activations_right,
        inhibitions_left,
        inhibitions_right,
        m,
        a,
        r,
        *ssa_params,
    )
    props_sum = _sum(propensities)

    if props_sum > 0:
        # Make a leap in time
        mu = np.dot(propensities, U.astype(np.float64))
        sigma2 = np.dot(propensities, Usq.astype(np.float64))
        tau = get_tau(
            epsilon,
            population,
            mu,
            sigma2,
            highest_rxn_order,
        )

        # Accept the tau leap if the expected number of events in the leap is
        # more than the approximate computational cost of leaping
        accept = tau * props_sum > TAU_LEAP_COST

        if accept:
            # Calculate the number of each event occuring in the time interval
            rg, events = _sample_poisson(rg, propensities, tau)
        else:
            # Wipe the events array until tau-leaping resumes. Otherwise, when it
            # resumes, the events from the previous tau-leap will be executed
            events.fill(0)

    # If no reactions occur, skip to the end of the simulation
    else:
        accept = True
        tau = np.inf

    return rg, accept, tau, population, events


@njit
def gillespie_tau_leaping(
    rg,
    epsilon,
    highest_rxn_order,
    time_points,
    population_0,
    Uz1,
    m,
    a,
    r,
    activations_left,
    activations_right,
    inhibitions_left,
    inhibitions_right,
    *ssa_params,
):
    """ """

    # Initialize output
    nt = len(time_points)

    U = Uz1[:-1]  # Remove the all-zero row at the end, used for vanilla Gillespie
    Usq = U**2
    m4a3r3, m2ar = U.shape

    pop_out = -np.ones((nt, m2ar)).astype(np.int64)

    # Set up auxiliary variables for tau-leaping
    events = np.zeros((m4a3r3, 1)).astype(np.int64)

    # Initialize and perform simulation
    propensities = np.zeros(m4a3r3).astype(np.float64)
    population = population_0.copy().astype(np.int64)
    pop_out[0] = population
    j = 0
    t = time_points[0]

    # Start with one Gillespie step to compute propensities
    n_gillespie_steps = 1

    # First event indexes the all-zero row of Uz1
    event = m4a3r3
    while j < nt:
        tj = time_points[j]
        while t <= tj:
            # When tau is too small, take Gillespie steps
            if n_gillespie_steps > 0:
                rg, t, population, event = take_gillespie_step(
                    rg,
                    t,
                    population,
                    event,
                    propensities,
                    Uz1,
                    activations_left,
                    activations_right,
                    inhibitions_left,
                    inhibitions_right,
                    m,
                    a,
                    r,
                    *ssa_params,
                )
                n_gillespie_steps -= 1
            else:
                rg, accept_leap, tau, population, events = take_tau_leap(
                    rg,
                    population,
                    events,
                    propensities,
                    U,
                    Usq,
                    epsilon,
                    highest_rxn_order,
                    activations_left,
                    activations_right,
                    inhibitions_left,
                    inhibitions_right,
                    m,
                    a,
                    r,
                    *ssa_params,
                )
                if accept_leap:
                    t += tau
                else:
                    # Prepare for some vanilla Gillespie steps
                    event = m4a3r3
                    n_gillespie_steps = 100

        # Update the index (Be careful about types for Numba)
        new_j = j + np.searchsorted(time_points[j:], t)

        # Update the population
        pop_out[j:new_j] = population

        # Increment index
        j = new_j

    return rg, pop_out


# @jitclass
class GillespieSSA:
    seed: int64
    PARAM_RANGES: float64[:, :]
    time_points: float64[:]
    dt: float64
    nt: int64
    init_mean: float32
    activations_left: int64[:]
    activations_right: int64[:]
    inhibitions_left: int64[:]
    inhibitions_right: int64[:]
    U: int64[:, :]
    DEFAULT_PARAMS: float64[:]
    n_propensities: int64
    n_species: int64
    n_params: int64
    epsilon: float64
    mu: float64[:]
    sigma2: float64[:]
    highest_rxn_order: int64[:]
    m: int64
    a: int64
    r: int64
    m2: int64
    m3: int64
    m4: int64
    m6: int64
    a2: int64
    a3: int64
    r2: int64
    r3: int64
    m4a: int64
    m4a2: int64
    m4a3: int64
    m4a3r: int64
    m4a3r2: int64
    m4a3r3: int64
    m2a: int64
    m2ar: int64

    def __init__(
        self,
        seed,
        n_species,
        activation_mtx,
        inhibition_mtx,
        update_mtx,
        dt,
        nt,
        mean_mRNA_init,
        PARAM_RANGES,
        DEFAULT_PARAMS,
        epsilon=0.03,
    ):
        self.rg = np.random.default_rng(seed)

        self.init_mean = mean_mRNA_init

        self.PARAM_RANGES = PARAM_RANGES
        self.DEFAULT_PARAMS = DEFAULT_PARAMS

        # Compute some numbers for convenient indexing
        self.m = n_species
        self.a = activation_mtx.sum()
        self.r = inhibition_mtx.sum()
        self.m2 = 2 * self.m
        self.m3 = 3 * self.m
        self.m4 = 4 * self.m
        self.m6 = 6 * self.m
        self.a2 = 2 * self.a
        self.a3 = 3 * self.a
        self.r2 = 2 * self.r
        self.r3 = 3 * self.r
        self.m4a = self.m4 + self.a
        self.m4a2 = self.m4 + self.a2
        self.m4a3 = self.m4 + self.a3
        self.m4a3r = self.m4a3 + self.r
        self.m4a3r2 = self.m4a3 + self.r2
        self.m4a3r3 = self.m4a3 + self.r3
        self.m2a = self.m2 + self.a
        self.m2ar = self.m2a + self.r

        self.n_propensities = self.m4a3r3
        self.n_species = self.m2ar
        self.n_params = len(self.DEFAULT_PARAMS)

        # Get indices of TFs involved in the left- and right-hand side of each reaction
        self.activations_left, self.activations_right = activation_mtx.nonzero()
        self.inhibitions_left, self.inhibitions_right = inhibition_mtx.nonzero()

        self.U = update_mtx

        self.dt = dt
        self.nt = nt
        self.time_points = dt * np.arange(nt)

        # Store statistics for tau-leaping
        self.mu = np.zeros(self.n_species).astype(np.float64)
        self.sigma2 = np.zeros(self.n_species).astype(np.float64)

        # Error tolerance for tau-leaping
        self.epsilon = epsilon

        # To compute the tau-leap, we will find the largest tau that satisfies the error
        # tolerance. To do this, we need to know the highest order reaction for each species.
        # For mRNA, the highest order reaction is first-order (degradation/translation)
        # For proteins, hte highest is first-order (degradation), unless it participates in
        #   promoter binding, in which case that reaction is second-order
        # For TF-promoter complexes, the highest is first-order (unbinding)
        highest_rxn_order = np.ones(self.n_species).astype(np.int64)
        for prot in self.activations_left:
            highest_rxn_order[self.m + prot] = 2
        for prot in self.inhibitions_left:
            highest_rxn_order[self.m + prot] = 2
        self.highest_rxn_order = highest_rxn_order

        # Run once to compile
        # _ = self.run_with_params(
        #     np.zeros(self.n_species).astype(np.int64), self.DEFAULT_PARAMS
        # )

    def gillespie_trajectory(self, population_0, *ssa_params):
        """ """
        self.rg, y_t = gillespie_trajectory(
            self.rg,
            self.time_points,
            population_0,
            self.U,
            self.m,
            self.a,
            self.r,
            self.activations_left,
            self.activations_right,
            self.inhibitions_left,
            self.inhibitions_right,
            *ssa_params,
        )
        return y_t

    def gillespie_tau_leaping(self, population_0, *ssa_params):
        """ """
        self.rg, y_t = gillespie_tau_leaping(
            self.rg,
            self.epsilon,
            self.highest_rxn_order,
            self.time_points,
            population_0,
            self.U,
            self.m,
            self.a,
            self.r,
            self.activations_left,
            self.activations_right,
            self.inhibitions_left,
            self.inhibitions_right,
            *ssa_params,
        )
        return y_t

    def package_params_for_ssa(self, params):
        """ """
        return package_params_for_ssa(params)

    def draw_random_initial_and_params(self):
        self.rg, pop0, params = draw_random_initial_and_params(
            self.rg, self.PARAM_RANGES, self.m, self.a, self.r, self.init_mean
        )
        return pop0, params

    def run_with_params(self, pop0, params):
        """ """
        ssa_params = self.package_params_for_ssa(params)
        return self.gillespie_trajectory(pop0, *ssa_params)

    def run_with_params_tau_leap(self, pop0, params):
        """ """
        ssa_params = self.package_params_for_ssa(params)
        return self.gillespie_tau_leaping(pop0, *ssa_params)

    def run_batch_with_params(self, pop0, params, n):
        """ """
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i in range(n):
            y_ts[i] = self.run_with_params(pop0, params)
        return y_ts

    def run_batch_with_params_vector(self, pop0s, param_sets):
        """ """
        n = len(pop0s)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i in range(n):
            y_ts[i] = self.run_with_params(pop0s[i], param_sets[i])
        return y_ts

    def run_batch_with_params_tau_leap(self, pop0, params, n):
        """ """
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i in range(n):
            y_ts[i] = self.run_with_params_tau_leap(pop0, params)
        return y_ts

    def run_batch_with_params_tau_leap_vector(self, pop0s, param_sets, n):
        """ """
        n = len(pop0s)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i in range(n):
            y_ts[i] = self.run_with_params_tau_leap(pop0s[i], param_sets[i])
        return y_ts

    def run_random_sample(self):
        """ """
        pop0, params = self.draw_random_initial_and_params()
        ssa_params = self.package_params_for_ssa(params)
        return pop0, params, self.gillespie_trajectory(pop0, *ssa_params)

    def run_random_sample_tau_leap(self):
        """ """
        pop0, params = self.draw_random_initial_and_params()
        ssa_params = self.package_params_for_ssa(params)
        return pop0, params, self.gillespie_tau_leaping(pop0, *ssa_params)

    def run_batch(self, n):
        """ """
        pop0s = np.zeros((n, self.n_species)).astype(np.int64)
        param_sets = np.zeros((n, self.n_params)).astype(np.float64)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i in range(n):
            pop0, params, y_t = self.run_random_sample()
            pop0s[i] = pop0
            param_sets[i] = params
            y_ts[i] = y_t

        return pop0s, param_sets, y_ts

    def run_batch_tau_leap(self, n):
        """ """
        pop0s = np.zeros((n, self.n_species)).astype(np.int64)
        param_sets = np.zeros((n, self.n_params)).astype(np.float64)
        y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
        for i in range(n):
            pop0, params, y_t = self.run_random_sample_tau_leap()
            pop0s[i] = pop0
            param_sets[i] = params
            y_ts[i] = y_t

        return pop0s, param_sets, y_ts
