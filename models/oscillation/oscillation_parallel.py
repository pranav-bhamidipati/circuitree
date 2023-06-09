from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional
import numpy as np
import pandas as pd
import ray
from circuitree.parallel import DefaultFactoryDict
from oscillation import OscillationTree, TFNetworkModel


@ray.remote
def get_results(state, dt, nt, tau_leap=False):
    model = TFNetworkModel(state)
    y_t, pop0, params, reward = model.run_ssa_and_get_acf_minima(
        size=1, tau_leap=tau_leap, freqs=False, indices=False, dt=dt, nt=nt, abs=True
    )
    return pop0, params, np.ravel(reward)[0]


def get_batch_results(size, state, dt, nt, tau_leap=False) -> list[float]:
    rewards = [get_results.remote(state, dt, nt, tau_leap) for _ in range(size)]
    rewards = ray.get(rewards)
    return list(zip(*rewards))


@dataclass
class StateResults:
    state: str
    rewards: list[np.float64] = field(default_factory=list)
    pop0s: list[np.ndarray[np.int64]] = field(default_factory=list)
    param_sets: list[np.ndarray[np.float64]] = field(default_factory=list)


class TranspositionTable:
    def __init__(self):
        self.table: Mapping[str, StateResults] = DefaultFactoryDict(
            default_factory=StateResults
        )

    def __getitem__(self, state):
        return self.table[state]

    def add_results(self, state, results):
        pop0s, param_sets, rewards = results
        self.table[state].pop0s.extend(pop0s)
        self.table[state].param_sets.extend(param_sets)
        self.table[state].rewards.extend(rewards)

    def to_df(self, init_columns, param_names) -> pd.DataFrame:
        state_dfs = []
        for state, state_results in self.table.items():
            pop0_data = np.array(state_results.pop0s).T
            param_data = np.array(state_results.param_sets).T
            data = (
                dict(state=state, reward=state_results.rewards)
                | dict(zip(init_columns, pop0_data))
                | dict(zip(param_names, param_data))
            )
            state_df = pd.DataFrame(data)
            state_dfs.append(state_df)
        return pd.concat(state_dfs, ignore_index=True)


class OscillationTreeParallel(OscillationTree):
    """Searches the space of TF networks for oscillatory topologies.
    Each step of the search takes the average of multiple draws.
    Uses a transposition table to store and access results. If desired
    results are not present in the table, they will be computed in parallel
    and added to the table."""

    def __init__(
        self,
        results_table: Optional[TranspositionTable] = None,
        counter: Optional[Counter] = None,
        **kwargs,
    ):
        if not isinstance(results_table, TranspositionTable):
            results_table = TranspositionTable()

        super().__init__(results_table=results_table, **kwargs)

        self.visit_counter = counter or Counter()

    def get_reward(
        self,
        state: str,
        batch_size: Optional[int] = None,
        dt: Optional[float] = None,
        nt: Optional[int] = None,
        tau_leap: Optional[bool] = None,
    ) -> float:
        """Run the model and get a random reward"""
        dt = dt if dt is not None else self.dt
        nt = nt if nt is not None else self.nt
        tau_leap = tau_leap if tau_leap is not None else self.tau_leap
        batch_size = batch_size if batch_size is not None else self.batch_size

        # Get results from table
        visit_num = self.visit_counter[state]

        print(f"Visit {visit_num} to {state}")

        start = visit_num * batch_size
        end = start + batch_size
        rewards = self.results_table[state].rewards[start:end]

        n_found_in_table = len(rewards)
        n_to_run = batch_size - n_found_in_table
        if n_to_run > 0:
            print(f"\tRunning {n_to_run} new simulations")

            new_results = get_batch_results(batch_size, state, dt, nt, tau_leap)
            self.results_table.add_results(state, new_results)
            pop0s, param_sets, new_rewards = new_results

            print(f"\t\tNew rewards: {new_rewards}")

            rewards.extend(new_rewards)

        self.visit_counter[state] += 1

        reward = np.mean(rewards)
        return reward
