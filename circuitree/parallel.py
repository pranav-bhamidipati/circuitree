from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
import numpy as np
import pandas as pd

from .utils import DefaultFactoryDict, DefaultMapping

__all__ = ["MCTSResult", "ResultsRegistry", "TranspositionTable"]


@dataclass
class MCTSResult:
    reward: float
    initial_conditions: Iterable[float | int]
    params: Iterable[float | int]
    extras: Optional[Any] = field(default_factory=list)

    def unpack(self):
        return self.reward, *self.initial_conditions, *self.params, *self.extras


@dataclass
class ResultsRegistry:
    state: str
    visit_results: list[MCTSResult] = field(default_factory=list)

    def __getitem__(self, visit: int):
        return self.visit_results[visit]

    def __len__(self):
        return len(self.visit_results)

    def __iter__(self):
        return iter(self.visit_results)

    def append(self, results):
        if isinstance(results, MCTSResult):
            self.visit_results.append(results)
        else:
            self.visit_results.append(MCTSResult(*results))

    def extend(self, batch_results):
        batch = [
            r if isinstance(r, MCTSResult) else MCTSResult(*r) for r in batch_results
        ]
        self.visit_results.extend(batch)

    def unpack_results(self):
        return list(zip(*(results.unpack() for results in self.visit_results)))

    @property
    def rewards(self):
        return (r.reward for r in self.visit_results)


class TranspositionTable:
    """
    A transposition table for storing results of MCTS with stochastic playouts.

    In a stochastic game, each visit to a terminal state yields a different result.
    For a given random series of playouts, the result `vals` of visit number `i` to
    state `s` is stored as ...


    [[under construction]]

    `TranspositionTable[s, i] = vals`.

    The first element of `vals` is the reward value, which will be accessed by the
    MCTS algorithm. The next elements are the initial conditions of the playout and
    the parameters used. The remaining elements of `vals` are other metadata of the
    simulated playout (returned by the `metric_func` function).
    """

    def __new__(cls, *args, **kwargs):
        # Idempotent
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(
        self,
        results_colnames: Iterable[str],
        table: Optional[DefaultMapping[str, ResultsRegistry]] = None,
    ):
        self.columns = tuple(results_colnames)
        self.ncols = len(self.columns)
        if table is None:
            _table = DefaultFactoryDict(default_factory=ResultsRegistry)
        elif isinstance(table, DefaultMapping):
            _table = table
        else:
            raise TypeError(
                f"table must be a DefaultMapping, not {type(table).__name__}"
            )

        self.table: DefaultMapping[str, ResultsRegistry] = _table

    def __getitem__(self, state_wwo_visit):
        """Access the results of visit(s) to a state. If no visit number is specified,
        return the list of all visit results."""
        match state_wwo_visit:
            case (state, visit):
                return self.table[state][visit]
            case state:
                return self.table[state]

    def __missing__(self, state_wwo_visit):
        """If the state is not in the table, create a new entry. If the visit number
        is not in the registry for that state, raise an IndexError."""
        match state_wwo_visit:
            case (state, visit):
                n_visits = len(self.table[state])
                raise IndexError(
                    f"Visit index {visit} does not exist for state {state} with "
                    f"{n_visits} total visits."
                )
            case state:
                self.table[state] = ResultsRegistry(state)
                return self.table[state]

    def __contains__(self, state_wwo_visit):
        match state_wwo_visit:
            case (state, visit):
                return state in self.table and visit < len(self.table[state])
            case state:
                return state in self.table

    def __len__(self):
        return len(self.table)

    @property
    def shape(self):
        return len(self.table), self.ncols

    def n_visits(self, state):
        """Return number of visits with triggering a default factory call."""
        if state in self.table:
            return len(self.table[state])
        else:
            return 0

    def get_reward(self, state: str, visit: int):
        return self.table[state][visit].reward

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()

    def items(self):
        return self.table.items()

    def draw_random_result(self, state: str, rg: Optional[np.random.Generator] = None):
        """Draw a random result from state `state`."""
        index = rg.integers(0, len(self.table[state]))
        return self.table[state][index]

    def draw_random_reward(self, state: str, rg: Optional[np.random.Generator] = None):
        """Draw a random result from state `state`."""
        index = rg.integers(0, len(self.table[state]))
        return self.table[state][index].reward

    def draw_bootstrap(
        self, state: str, size: int, rg: Optional[np.random.Generator] = None
    ):
        """Draw a bootstrap sample of size `size` from the results for `state`."""
        if rg is None:
            rg = np.random.default_rng()
        indices = rg.integers(0, len(self.table[state]), size=size)
        return indices, [self.table[state][i] for i in indices]

    def draw_bootstrap_reward(
        self, state: str, size: int, rg: Optional[np.random.Generator] = None
    ):
        """Draw a bootstrap sample of size `size` from the reward values for `state`."""
        if rg is None:
            rg = np.random.default_rng()
        indices = rg.integers(0, len(self.table[state]), size=size)
        return indices, np.array([self.table[state][i].reward for i in indices])

    @classmethod
    def _load_from_source(
        cls,
        read_func: Callable,
        src: Iterable[Path | str] | Path | str,
        progress: bool = False,
        visit_col: str = "visit",
        load_kw: Mapping[str, Any] = {},
        **kwargs,
    ):
        if isinstance(src, Iterable):
            df = pd.concat([read_func(f, **load_kw) for f in src])
        else:
            df = read_func(src, **load_kw)

        if "state" not in df.columns:
            if df.index.name == "state":
                df = df.reset_index()
            else:
                raise ValueError(
                    "Dataframe must have a 'state' column or have 'state' as the index."
                )
        df["state"] = pd.Categorical(df["state"])
        if visit_col not in df.columns:
            df[visit_col] = df.groupby("state").cumcount()

        return cls.from_dataframe(
            df, visit_column=visit_col, progress=progress, **kwargs
        )

    @classmethod
    def from_csv(cls, source, **kwargs):
        return cls._load_from_source(pd.read_csv, source, **kwargs)

    @classmethod
    def from_parquet(cls, source, **kwargs):
        return cls._load_from_source(pd.read_parquet, source, **kwargs)

    @classmethod
    def from_feather(cls, source, **kwargs):
        return cls._load_from_source(pd.read_feather, source, **kwargs)

    @classmethod
    def from_hdf(cls, source, **kwargs):
        return cls._load_from_source(pd.read_hdf, source, **kwargs)

    @classmethod
    def from_json(cls, source, **kwargs):
        return cls._load_from_source(pd.read_json, source, **kwargs)

    @classmethod
    def from_pickle(cls, source, **kwargs):
        return cls._load_from_source(pd.read_pickle, source, **kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        init_columns: Optional[Iterable[str]] = None,
        param_columns: Optional[Iterable[str]] = None,
        visit_column: str = "visit",
        reward_column: str = "reward",
        state_column: str = "state",
        progress: bool = False,
        **kwargs,
    ):
        table = DefaultFactoryDict(default_factory=ResultsRegistry)
        init_columns = init_columns or []
        param_columns = param_columns or []
        df = df.reindex(
            columns=[state_column, visit_column, reward_column]
            + list(init_columns)
            + list(param_columns)
        ).sort_values([state_column, visit_column])
        grouped = df.groupby(state_column)

        if progress:
            from tqdm import tqdm

            iterator = tqdm(grouped, desc="Loading transposition table")
        else:
            iterator = grouped

        for state, state_table in iterator:
            rewards = state_table[reward_column].abs()
            init_conds = state_table[init_columns].values
            params = state_table[param_columns].values
            table[state].extend(zip(rewards, init_conds, params))

        colnames = [reward_column] + list(init_columns) + list(param_columns)
        return cls(results_colnames=colnames, table=table)

    def to_dataframe(self):
        data = {
            state: pd.DataFrame([v.unpack() for v in res], columns=self.columns)
            for state, res in self.table.items()
        }
        df = pd.concat(data, names=["state", "visit"])
        df = df.reset_index(level="visit")
        df.index = pd.CategoricalIndex(df.index, ordered=True, name="state")
        return df

    def to_csv(self, fname, **kwargs):
        self.to_dataframe().to_csv(fname, index=True, **kwargs)

    def to_parquet(self, fname, **kwargs):
        self.to_dataframe().to_parquet(fname, index=True, **kwargs)

    def to_hdf(self, fname, **kwargs):
        self.to_dataframe().to_hdf(fname, index=True, **kwargs)
