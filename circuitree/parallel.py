from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
import json
from pathlib import Path
import networkx as nx
from numpy.random import default_rng, SeedSequence
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from typing import Any, Callable, Iterable, Optional, Sequence

from .modularity import tree_modularity, tree_modularity_estimate
from .circuitree import CircuiTree, accumulate_visits_and_rewards, ucb_score
from .utils import DefaultMapping

__all__ = [
    "MultithreadedCircuiTree",
    "search_mcts_in_thread",
    "ParameterTable",
    "TranspositionTable",
    "ParallelTree",
]


class MultithreadedCircuiTree(ABC):
    def __init__(
        self,
        root: str,
        threads: Optional[int] = None,
        seed: int = 2023,
        exploration_constant: Optional[float] = None,
        graph: Optional[nx.DiGraph] = None,
        **kwargs,
    ):
        if exploration_constant is None:
            self.exploration_constant = np.sqrt(2)
        else:
            self.exploration_constant = exploration_constant

        if threads is None:
            threads = cpu_count()
        self.threads = threads

        self.seed = seed
        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(threads)]

        self.root = root
        if graph is None:
            self.graph = nx.DiGraph()
            self.graph.add_node(self.root, visits=0, reward=0)
        else:
            self.graph = graph
            if self.root not in self.graph:
                raise ValueError(
                    f"Supplied graph does not contain the root node: {root}"
                )
        self.graph.root = self.root

        # Attributes that should not be saved to file
        self._non_serializable_attrs = [
            "_non_serializable_attrs",
            "_random_generators",
            "graph",
        ]

    @property
    def default_attrs(self):
        return dict(visits=0, reward=0)

    def _do_action(self, state: Any, action: Any):
        new_state = self.do_action(state, action)
        return self.get_unique_state(new_state)

    @abstractmethod
    def get_actions(self, state: Any) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def do_action(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, state, visit_number, **kwargs) -> float | int:
        raise NotImplementedError

    @abstractmethod
    def get_unique_state(self, state: Any) -> Any:
        raise NotImplementedError

    def is_success(self, state) -> bool:
        """Must be defined to calculate modularity."""
        raise NotImplementedError

    def get_ucb_score(self, parent, child):
        return ucb_score(self.graph, parent, child, self.exploration_constant)

    def select(self, thread_idx: int):
        node = self.root
        selection_path = [node]
        while not self.is_leaf(node):
            node = self.best_child(thread_idx, node)
            selection_path.append(node)
        return selection_path

    def expand(
        self, thread_idx: int, selection_path: list[Any]
    ) -> tuple[list[Any], int]:
        """Expands a candidate selected node and returns the nodes in the resulting
        selection path and the visit number at the selected node.

        If the candidate is non-terminal, selects a random child and appends it to the
        selection path. Otherwise, does nothing."""
        node = selection_path[-1]
        actions = self.get_actions(node)
        if actions:
            children = [self._do_action(node, action) for action in actions]
            for action, child in zip(actions, children):
                if child not in self.graph.nodes:
                    self.graph.add_node(child, **self.default_attrs)
                    self.graph.add_edge(
                        node, child, action=action, **self.default_attrs
                    )
            rg = self._random_generators[thread_idx]
            selection_path.append(rg.choice(children))

        if len(selection_path) == 1:
            raise ValueError(f"Terminated at root state: {node}")

        visit_number = self.graph.nodes[selection_path[-1]]["visits"]
        return selection_path, visit_number

    def simulate(
        self, thread_idx, node, visit_number, **kwargs
    ) -> tuple[Any, float | int]:
        sim_node = node
        rg: np.random.Generator = self._random_generators[thread_idx]
        while not self.is_terminal(sim_node):
            action = rg.choice(self.get_actions(sim_node))
            sim_node = self._do_action(sim_node, action)
        reward = self.get_reward(sim_node, visit_number, **kwargs)
        return sim_node, reward

    def backpropagate_reward(self, selection_path: list, reward: float | int):
        self._backpropagate(selection_path, "reward", reward)

    def is_leaf(self, node):
        return self.graph.out_degree(node) == 0

    def best_child(self, thread_idx, node):
        rg: np.random.Generator = self._random_generators[thread_idx]
        children = list(self.graph.neighbors(node))
        rg.shuffle(children)
        scores = [self.get_ucb_score(node, child) for child in children]
        best = children[np.argmax(scores)]
        return best

    def _backpropagate(self, path: list, attr: str, value: float | int):
        """Update the value of an attribute for each node and edge in the path."""
        _path = path.copy()
        child = _path.pop()
        self.graph.nodes[child][attr] += value
        while _path:
            parent = _path.pop()
            self.graph.edges[parent, child][attr] += value
            child = parent
            self.graph.nodes[child][attr] += value

    def backpropagate_visit(self, selection_path: list) -> None:
        """Update the visit count for each node in the selection path. Happens
        before simulation and reward calculation, so until the reward is known,
        there is 'virtual loss' on each node in the selection path."""
        self._backpropagate(selection_path, "visits", 1)

    def traverse(self, thread_idx: int, **kwargs):
        selection_path = self.select(thread_idx)
        selection_path, visit_number = self.expand(thread_idx, selection_path)

        # Between backprop of visit and reward, we incur virtual loss
        self.backpropagate_visit(selection_path)
        sim_node, reward = self.simulate(
            thread_idx, selection_path[-1], visit_number, **kwargs
        )
        self.backpropagate_reward(selection_path, reward)
        return selection_path, reward, sim_node

    def accumulate_visits_and_rewards(self, graph: Optional[nx.DiGraph] = None):
        _accumulated = self.graph if graph is None else graph
        accumulate_visits_and_rewards(_accumulated)
        if graph is None:
            return _accumulated

    @property
    def terminal_states(self):
        return (node for node in self.graph.nodes if self.is_terminal(node))

    def bfs_iterator(self, root=None, shuffle=False):
        root = self.root if root is None else root
        return chain(*(l for l in nx.bfs_layers(self.graph, root)))

    def modularity(self) -> float:
        return tree_modularity(self.graph, self.root, self.is_terminal, self.is_success)

    def modularity_estimate(self) -> float:
        return tree_modularity_estimate(self.graph, self.root)

    def to_file(
        self,
        gml_file: str | Path,
        json_file: Optional[str | Path] = None,
        save_attrs: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        gml_target = Path(gml_file).with_suffix(".gml")
        nx.write_gml(self.graph, gml_target, **kwargs)

        if json_file is not None:
            if save_attrs is None:
                keys = set(self.__dict__.keys()) - set(self._non_serializable_attrs)
            else:
                keys = set(save_attrs)
                if non_serializable := (keys & set(self._non_serializable_attrs)):
                    repr_non_ser = ", ".join(non_serializable)
                    raise ValueError(
                        f"Attempting to save non-serializable attributes: {repr_non_ser}."
                    )

            attrs = {k: v for k, v in self.__dict__.items() if k in keys}

            json_target = Path(json_file).with_suffix(".json")
            with json_target.open("w") as f:
                json.dump(attrs, f, indent=4)

            return gml_target, json_target

        else:
            return gml_target

    @classmethod
    def from_file(
        cls, graph_gml: str | Path, attrs_json: Optional[str | Path] = None, **kwargs
    ):
        if attrs_json is not None:
            with open(attrs_json, "r") as f:
                kwargs.update(json.load(f))

        graph = nx.read_gml(graph_gml)

        return cls(graph=graph, **kwargs)


def search_mcts_in_thread(
    mtree: MultithreadedCircuiTree,
    thread_idx: int,
    n_steps: int,
    callback: Optional[Callable] = None,
    callback_every: int = 1,
    return_metrics: Optional[bool] = None,
    **kwargs,
):
    if callback is None:
        callback = lambda *a, **kw: None

    m0 = callback(mtree, 0, [None], None, None)
    if return_metrics is None:
        return_metrics = m0 is not None

    metrics = [m0]
    for iteration in range(1, n_steps + 1):
        selection_path, reward, sim_node = mtree.traverse(thread_idx, **kwargs)
        if iteration % callback_every == 0:
            m = callback(mtree, iteration, selection_path, reward, sim_node)
            if return_metrics:
                metrics.append(m)

    if return_metrics:
        return mtree, metrics
    else:
        return mtree


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
            kw["initial_conditions"] = list(df[init_cols].values)
        if param_cols:
            kw["parameter_sets"] = list(df[param_cols].values)

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
        df = pd.concat(
            {k: pd.Series(v) for k, v in self.table.items()},
            names=[state_col, visit_col],
        ).reset_index()
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
