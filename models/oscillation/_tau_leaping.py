########## Tau leaping functions ##########
###########################################
## NOTE: These are not used in oscillation.py because tau leaping was *not* found to be
##       faster than the vanilla Gillespie sampler. This is due to the stiffness of
##       the problem. The fast rates of promoter binding/unbinding result in a small tau
##       when performing explicit tau leaping. Furthermore, because multiple TFs may bind
##       to the same promoter, it is non-trivial to compute the equilibrium occupancy of
##       promoters in generality. This virtually precludes slow-scale approximations such
##       as slow-scale SSA and slow-scale tau leaping.
##       Nonetheless, the code for explicit tau leaping is included here in case it is of
##       interest for future work.

############### JIT-compiled global functions


# @njit
# def _apply_event_nonneg(population, U, event):
#     """Returns the new population after applying an event. Rejects the change if any
#     species drops below zero."""
#     new_pop = population + U[event]
#     for p in new_pop:
#         if p < 0:
#             return population
#     return new_pop


# @njit
# def _get_promoter_occupancy(m, a, promoter, population, act_right, inh_right):
#     m2 = m * 2
#     n_bound = 0
#     for i, p in enumerate(act_right):
#         if p == promoter:
#             n_bound += population[m2 + i]
#     for i, p in enumerate(inh_right):
#         if p == promoter:
#             n_bound += population[m2 + a + i]
#     return n_bound


# @njit
# def _apply_events_safe(
#     population, events_in_order, U, m, a, r, activations_right, inhibitions_right
# ):
#     """Apply an event while ensuring (1) No species drops below zero and (2) No promoters
#     have more than 2 TFs bound to it. Used for applying multiple events sequentially
#     during tau leaping.

#     Actions are taken based on the event number (an integer index of the update matrix).
#     Events are ordered as follows. An asterisk ``*`` indicates cases where promoter
#     saturation must be specially ensured.
#         Transcription of mRNA (0, M)
#         Translation of mRNA (M, 2M)
#         Degradation of mRNA (2M, 3M)
#         Degradation of unbound TF (3M, 4M)
#       * Binding of activator to promoter (4M, 4M + A)
#         Unbinding of activator from promoter (4M + A, 4M + 2A)
#         Degradation of bound activator (4M + 2A, 4M + 3A)
#       * Binding of inhibitor to promoter (4M + 3A, 4M + 3A + R)
#         Unbinding of inhibitor from promoter (4M + 3A + R, 4M + 3A + 2R)
#         Degradation of bound inhibitor (4M + 3A + 2R, 4M + 3A + 3R)
#         Null reaction - all zeros (4M + 3A + 3R, 4M + 3A + 3R + 1)

#     The necessary action is taken by dividing the index range into 5 categories based on
#     the above ordering and taking the appropriate action for each category.
#     """
#     m4 = m * 4
#     m4a3 = m4 + a * 3
#     boundaries = np.array([m4, m4 + a, m4a3, m4a3 + r])
#     categories = np.searchsorted(boundaries, events_in_order, side="right")
#     for event, category in zip(events_in_order, categories):
#         # Catch binding events and apply them only if the promoter is not saturated
#         if category == 1:
#             which_promoter = activations_right[event - m4]
#             n_bound = _get_promoter_occupancy(
#                 m, a, which_promoter, population, activations_right, inhibitions_right
#             )
#             if n_bound != 2:
#                 population = _apply_event_nonneg(population, U, event)

#         elif category == 3:
#             which_promoter = activations_right[event - m4a3]
#             n_bound = _get_promoter_occupancy(
#                 m, a, which_promoter, population, activations_right, inhibitions_right
#             )
#             if n_bound != 2:
#                 population = _apply_event_nonneg(population, U, event)

#         else:
#             population = _apply_event_nonneg(population, U, event)

#     return population


# @njit
# def apply_events(
#     rg, events, population, U, m, a, r, m4a3r3, activations_right, inhibitions_right
# ):
#     """Apply multiple events while ensuring (1) No species drops below zero and (2) No
#     promoters are occupied by more than 2 TFs. Used for applying multiple events
#     sequentially during tau leaping."""
#     events_in_order = np.repeat(np.arange(m4a3r3), events.flatten())
#     rg.shuffle(events_in_order)
#     population = _apply_events_safe(
#         population, events_in_order, U, m, a, r, activations_right, inhibitions_right
#     )
#     return rg, population


# @njit
# def _sample_poisson(rg, out, propensities, dt):
#     """Sample the number of events occurring in the time interval dt
#     from a Poisson distribution."""
#     for i, mean_num_events in enumerate(propensities * dt):
#         out[i, 0] = rg.poisson(mean_num_events)
#     return rg, out


# @njit
# def get_tau(epsilon, population, mu, sigma2, highest_rxn_order):
#     """Compute the time increment for tau-leaping."""

#     # No zero-order reactions, so division by zero is not an issue
#     times = np.maximum(epsilon * population / highest_rxn_order, 1.0)
#     min_mu_time = (times / np.abs(mu)).min()
#     min_s2_time = (times**2 / sigma2).min()
#     return min(min_mu_time, min_s2_time)


# @njit
# def take_tau_leap(
#     population,
#     propensities,
#     U,
#     Usq,
#     epsilon,
#     highest_rxn_order,
#     activations_left,
#     activations_right,
#     inhibitions_left,
#     inhibitions_right,
#     m,
#     a,
#     r,
#     tau_leap_cost,
#     *ssa_params,
# ):
#     # Compute reaction propensities
#     propensities = get_propensities(
#         propensities,
#         population,
#         activations_left,
#         activations_right,
#         inhibitions_left,
#         inhibitions_right,
#         m,
#         a,
#         r,
#         *ssa_params,
#     )
#     props_sum = _sum(propensities)

#     if props_sum > 0:
#         # Make a leap in time
#         mu = np.dot(propensities, U.astype(np.float64))
#         sigma2 = np.dot(propensities, Usq.astype(np.float64))
#         tau = get_tau(
#             epsilon,
#             population,
#             mu,
#             sigma2,
#             highest_rxn_order,
#         )

#         # Accept the tau leap if the expected number of events in the leap is
#         # more than the approximate computational cost of leaping
#         accept = tau * props_sum > tau_leap_cost

#     # If no reactions occur, skip to the end of the simulation
#     else:
#         accept = True
#         tau = np.inf

#     return accept, tau


# @njit
# def gillespie_tau_leaping(
#     rg,
#     epsilon,
#     highest_rxn_order,
#     time_points,
#     population_0,
#     Uz1,
#     m,
#     a,
#     r,
#     activations_left,
#     activations_right,
#     inhibitions_left,
#     inhibitions_right,
#     tau_leap_cost,
#     *ssa_params,
# ):
#     """ """

#     # Initialize output
#     nt = len(time_points)

#     U = Uz1[:-1]  # Remove the all-zero row at the end, used for vanilla Gillespie
#     Usq = U**2
#     m4a3r3, m2ar = U.shape

#     pop_out = -np.ones((nt, m2ar)).astype(np.int64)

#     # Set up auxiliary variables for tau-leaping
#     events = np.zeros((m4a3r3, 1)).astype(np.int64)

#     # Initialize and perform simulation
#     propensities = np.zeros(m4a3r3).astype(np.float64)
#     population = population_0.copy().astype(np.int64)
#     pop_out[0] = population
#     j = 0
#     t = time_points[0]

#     n_gillespie_steps = 0

#     # First event indexes the all-zero row of Uz1
#     event = m4a3r3
#     while j < nt:
#         tj = time_points[j]
#         while t <= tj:
#             # When tau is too small, take Gillespie steps
#             if n_gillespie_steps > 0:
#                 _apply_event(population, Uz1, event)
#                 rg, t, event = take_gillespie_step(
#                     rg,
#                     t,
#                     population,
#                     event,
#                     propensities,
#                     activations_left,
#                     activations_right,
#                     inhibitions_left,
#                     inhibitions_right,
#                     m,
#                     a,
#                     r,
#                     *ssa_params,
#                 )
#                 n_gillespie_steps -= 1

#                 if n_gillespie_steps == 0:
#                     events.fill(0)
#                     events[event] = 1

#             else:
#                 rg, population = apply_events(
#                     rg,
#                     events,
#                     population,
#                     U,
#                     m,
#                     a,
#                     r,
#                     m4a3r3,
#                     activations_right,
#                     inhibitions_right,
#                 )

#                 accept_leap, tau = take_tau_leap(
#                     population,
#                     propensities,
#                     U,
#                     Usq,
#                     epsilon,
#                     highest_rxn_order,
#                     activations_left,
#                     activations_right,
#                     inhibitions_left,
#                     inhibitions_right,
#                     m,
#                     a,
#                     r,
#                     tau_leap_cost,
#                     *ssa_params,
#                 )

#                 if accept_leap:
#                     # Draw events occuring in the time interval
#                     rg, events = _sample_poisson(rg, events, propensities, tau)
#                     t += tau
#                 else:
#                     # Prepare for some vanilla Gillespie steps instead
#                     event = m4a3r3
#                     n_gillespie_steps = 100

#         # Update the index and population
#         new_j = j + np.searchsorted(time_points[j:], t)
#         pop_out[j:new_j] = population
#         j = new_j

#     return rg, pop_out


############# Additional lines in the __init__ method of the GillespieSSA class

# # Store statistics for tau-leaping
# self.mu = np.zeros(self.n_species).astype(np.float64)
# self.sigma2 = np.zeros(self.n_species).astype(np.float64)

# # Error tolerance for tau-leaping
# self.epsilon = epsilon

# # Estimated computational cost of a tau leap vs a Gillespie step
# self.tau_leap_cost = tau_leap_cost

# To compute the tau-leap, we will find the largest tau that satisfies the error
# tolerance. To do this, we need to know the highest order reaction for each species.
# For mRNA, the highest order reaction is first-order (degradation/translation)
# For proteins, hte highest is first-order (degradation), unless it participates in
#   promoter binding, in which case that reaction is second-order
# For TF-promoter complexes, the highest is first-order (unbinding)
# highest_rxn_order = np.ones(self.n_species).astype(np.int64)
# for prot in self.activations_left:
#     highest_rxn_order[self.m + prot] = 2
# for prot in self.inhibitions_left:
#     highest_rxn_order[self.m + prot] = 2
# self.highest_rxn_order = highest_rxn_order


############# Class methods of the GillespieSSA class #############

# def gillespie_tau_leaping(self, population_0, *ssa_params):
#     """ """
#     self.rg, y_t = gillespie_tau_leaping(
#         self.rg,
#         self.epsilon,
#         self.highest_rxn_order,
#         self.time_points,
#         population_0,
#         self.U,
#         self.m,
#         self.a,
#         self.r,
#         self.activations_left,
#         self.activations_right,
#         self.inhibitions_left,
#         self.inhibitions_right,
#         self.tau_leap_cost,
#         *ssa_params,
#     )

#     return y_t


# def run_with_params_tau_leap(self, pop0, params):
#     """ """
#     ssa_params = self.package_params_for_ssa(params)
#     return self.gillespie_tau_leaping(pop0, *ssa_params)

# def run_batch_with_params_tau_leap(self, pop0, params, n):
#     """ """
#     y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
#     for i in range(n):
#         y_ts[i] = self.run_with_params_tau_leap(pop0, params)
#     return y_ts

# def run_batch_with_params_tau_leap_vector(self, pop0s, param_sets, n):
#     """ """
#     n = len(pop0s)
#     y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
#     for i in range(n):
#         y_ts[i] = self.run_with_params_tau_leap(pop0s[i], param_sets[i])
#     return y_ts

# def run_random_sample_tau_leap(self):
#     """ """
#     pop0, params = self.draw_random_initial_and_params()
#     ssa_params = self.package_params_for_ssa(params)
#     return pop0, params, self.gillespie_tau_leaping(pop0, *ssa_params)

# def run_batch_tau_leap(self, n):
#     """ """
#     pop0s = np.zeros((n, self.n_species)).astype(np.int64)
#     param_sets = np.zeros((n, self.n_params)).astype(np.float64)
#     y_ts = np.zeros((n, self.nt, self.n_species)).astype(np.float64)
#     for i in range(n):
#         pop0, params, y_t = self.run_random_sample_tau_leap()
#         pop0s[i] = pop0
#         param_sets[i] = params
#         y_ts[i] = y_t

#     return pop0s, param_sets, y_ts
