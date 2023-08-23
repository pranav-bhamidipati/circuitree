from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
import numpy as np
import pandas as pd

from .circuitree import CircuiTree
from .utils import DefaultMapping, defaultlist

__all__ = [
    "ParameterTable",
    "TranspositionTable",
    "ParallelTree",
]


@dataclass
class ParameterTable:
    seeds: Iterable[int]
    initial_conditions: Optional[Sequence[Any]] = field(default_factory=list)
    parameter_sets: Optional[Sequence[Any]] = field(default_factory=list)
    wrap_index: bool = True

    """Stores a table of parameter sets that are used as inputs for a stochastic game.
    For each visit there is a random seed, a set of initial conditions, and a set of 
    parameter values."""

    def __post_init__(self):
        n = len(self.seeds)
        n_init = len(self.initial_conditions)
        n_params = len(self.parameter_sets)
        if n_init != n or n_params != n:
            raise ValueError(
                "Number of initial conditions, parameter sets, and random seeds "
                "must be equal."
            )

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, index):
        if self.wrap_index:
            index = index % len(self)
        return (
            self.seeds[index],
            self.initial_conditions[index],
            self.parameter_sets[index],
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        seed_col: str = "seed",
        param_cols: Optional[Iterable[str]] = None,
        init_cols: Optional[Iterable[str]] = None,
    ):
        if df.index.name == seed_col:
            df.reset_index(inplace=True)
        elif seed_col not in df.columns:
            raise ValueError(
                f"Dataframe does not have a column or index named '{seed_col}'."
            )

        kw = {}
        if init_cols:
            kw["initial_conditions"] = defaultlist(df[init_cols].values)
        if param_cols:
            kw["parameter_sets"] = defaultlist(df[param_cols].values)

        return cls(df[seed_col].tolist(), **kw)


class TranspositionTable:
    """
    Stores the history of reward payouts at each visit to each state in a decision tree.
    Used to store results and simulate playouts of a stochastic game.

    In a stochastic game, each visit to a terminal state yields a different result.
    For a given random series of playouts, the reward values to a terminal state `s` are
    stored in a list `vals`. The reward for visit `n` to state `s` can be accessed as:

    ```TranspositionTable[s, n]```

    The full list of rewards for state `s` can be accessed as `TranspositionTable[s]`.
    """

    def __new__(cls, *args, **kwargs):
        # Idempotent
        if not kwargs and len(args) == 1 and isinstance(args[0], cls):
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(
        self,
        table: Optional[DefaultMapping[str, list]] = None,
    ):
        if table is None:
            self._table = defaultdict(list)
        elif isinstance(table, DefaultMapping):
            self._table = table
        else:
            raise TypeError(
                f"table must be a DefaultMapping, not {type(table).__name__}"
            )

    @property
    def table(self):
        return self._table

    def __getitem__(self, state_wwo_visits):
        """Access the results of visit(s) to a state. If no visit number is specified,
        return the list of all visit results."""
        match state_wwo_visits:
            case state, slice(start=start, stop=stop, step=step):
                return self.table[state][start:stop:step]
            case state, [*visits]:
                state_table = self.table[state]
                return [state_table[visit] for visit in visits]
            case state, visit:
                return self.table[state][visit]
            case state:
                return self.table[state]

    def __missing__(self, state_wwo_visits):
        """If the state is not in the table, raise an error. If accessed without
        visit number(s), create an empty entry for this state."""
        n_visits = len(self.table[state])
        match state_wwo_visits:
            case state, slice(start=start, stop=stop, step=step):
                raise IndexError(
                    f"Cannot index slice({start}, {stop}, {step}) for state {state} "
                    f"with {n_visits} total visits."
                )
            case state, [*visits]:
                try:
                    for v in visits:
                        _ = self[state, v]
                except IndexError:
                    raise IndexError(
                        f"Visit index {v} does not exist for state {state} with "
                        f"{n_visits} total visits."
                    )
            case (state, visit):
                raise IndexError(
                    f"Visit index {visit} does not exist for state {state} with "
                    f"{n_visits} total visits."
                )
            case state:
                return self.table[state]  # defaultdict will create a new entry

    def __contains__(self, state_wwo_visit):
        n_visits = len(self.table[state])
        match state_wwo_visit:
            case state, slice(start=start, stop=stop, step=step):
                return state in self.table and stop <= n_visits
            case state, [*visits]:
                return state in self.table and all(v < n_visits for v in visits)
            case (state, visit):
                return state in self.table and visit < n_visits
            case state:
                return state in self.table

    def __len__(self):
        return len(self.table)

    def n_visits(self, state):
        """Return number of visits with triggering a default factory call."""
        if state in self.table:
            return len(self.table[state])
        else:
            return 0

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()

    def items(self):
        return self.table.items()

    def draw_random_reward(
        self, state: str, rg: Optional[np.random.Generator] = None
    ) -> tuple[int, float]:
        """Draw a random visit number and resulting reward from state `state`."""
        if rg is None:
            rg = np.random.default_rng()
        index = rg.integers(0, len(self.table[state]))
        return index, self.table[state, index]

    def draw_bootstrap_reward(
        self, state: str, size: int, rg: Optional[np.random.Generator] = None
    ):
        """Draw a bootstrap sample of size `size` from the reward values for `state`."""
        if rg is None:
            rg = np.random.default_rng()
        indices = rg.integers(0, len(self.table[state]), size=size)
        state_table = self.table[state]
        return indices, np.array([state_table[i] for i in indices])

    @classmethod
    def _load_from_source(
        cls,
        read_func: Callable,
        src: Iterable[Path | str] | Path | str,
        progress: bool = False,
        state_col: str = "state",
        visit_col: str = "visit",
        reward_col: str = "reward",
        **load_kwargs,
    ):
        if isinstance(src, Iterable):
            if progress:
                from tqdm import tqdm

                iter_src = tqdm(src)
            else:
                iter_src = src
            df = pd.concat([read_func(f, **load_kwargs) for f in iter_src])
        else:
            df = read_func(src, **load_kwargs)

        if state_col in df.columns:
            df.set_index(state_col, inplace=True)

        if df.index.name == state_col:
            df.index = pd.CategoricalIndex(df.index, ordered=True)
        else:
            raise ValueError(
                f"Dataframe does not have a column or index named '{state_col}'."
            )

        if reward_col not in df.columns:
            raise ValueError(f"Dataframe does not have a column named '{reward_col}'.")

        if visit_col not in df.columns:
            df[visit_col] = df.groupby(state_col).cumcount()

        return cls.from_dataframe(
            df,
            visit_column=visit_col,
            reward_column=reward_col,
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
        visit_column: str = "visit",
        reward_column: str = "reward",
    ):
        table = defaultdict(list)
        grouped = (
            df[[visit_column, reward_column]]
            .sort_values(visit_column)
            .groupby(level=0)[reward_column]
        )
        for state, rewards in grouped:
            table[state] = rewards.to_list()
        return cls(table=table)

    def to_dataframe(self, state_col: str = "state", visit_col: str = "visit"):
        df = pd.concat({k: pd.Series(v) for k, v in self.table.items()})
        df.reset_index(names=[state_col, visit_col], inplace=True)
        df[state_col] = df[state_col].astype("category")
        return df

    def to_csv(self, fname, **kwargs):
        self.to_dataframe().to_csv(fname, index=True, **kwargs)

    def to_parquet(self, fname, **kwargs):
        self.to_dataframe().to_parquet(fname, index=True, **kwargs)

    def to_hdf(self, fname, **kwargs):
        self.to_dataframe().to_hdf(fname, index=True, **kwargs)


class ParallelTree(CircuiTree):
    """A parallelizable implementation of CircuiTree. Parallelism is achieved
    via drawing multiple rewards at once. If boostrap is True, rewards for a
    set of visits to a state are drawn as bootstrap samples from the transposition
    table entries for that state. Otherwise, rewards are generated by indexing the
    transposition table by visit number. If desired results are not present in the table,
    they will be computed in parallel.

    Random seeds, parameter sets, and initial conditions are selected from the
    parameter table `self.param_table`.

    The methods `simulate_visits` and `save_results` must be implemented in a subclass.

    An invalid simulation result (e.g. a timeout) should be represented by a
    NaN reward value. If all rewards in a batch are NaNs, a new batch will be
    drawn from the transposition table. Otherwise, any NaN rewards will be
    ignored and the mean of the remaining rewards will be returned as the
    reward for the batch.
    """

    def __init__(
        self,
        *,
        parameter_table: Optional[ParameterTable] = None,
        transposition_table: Optional[TranspositionTable] = None,
        counter: Counter = None,
        seed_col: Optional[str] = None,
        param_cols: Optional[Iterable[str]] = None,
        init_cols: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(parameter_table, pd.DataFrame):
            self._parameter_table = ParameterTable.from_dataframe(
                parameter_table,
                seed_col=seed_col,
                param_cols=param_cols,
                init_cols=init_cols,
            )
        else:
            self._parameter_table = parameter_table
        self._transposition_table = TranspositionTable(transposition_table)

        self.visit_counter = Counter(counter)

        # Specify any attributes that should not be serialized when dumping to file
        self._non_serializable_attrs.extend(
            [
                "_parameter_table",
                "_transposition_table",
                "visit_counter",
            ]
        )

    @property
    def param_table(self) -> ParameterTable:
        return self._parameter_table

    @property
    def ttable(self) -> TranspositionTable:
        return self._transposition_table

    def reset_counter(self):
        self.visit_counter.clear()
