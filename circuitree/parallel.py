from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
import pandas as pd
import threading

from .circuitree import CircuiTree

__all__ = [
    "MCTSResults",
    "TranspositionTable",
    "TableBuffer",
    "DefaultFactoryDict",
    "ParallelTree",
]


@dataclass
class MCTSResults:
    reward: float
    initial_conditions: Iterable[float | int]
    params: Iterable[float | int]
    _metadata: Optional[Any] = None

    @property
    def metadata(self):
        if self._metadata is None:
            return ()
        elif isinstance(self._metadata, Iterable):
            return tuple(self._metadata)
        else:
            return (self._metadata,)

    def unpack(self):
        return self.reward, *self.initial_conditions, *self.params, *self.metadata


@dataclass
class StateTable:
    state: str
    visit_results: list[MCTSResults] = field(default_factory=list)

    def __getitem__(self, visit):
        return self.visit_results[visit]

    def __len__(self):
        return len(self.visit_results)

    def __iter__(self):
        return iter(self.visit_results)

    def append(self, results):
        self.visit_results.append(MCTSResults(results))

    def extend(self, batch_results):
        self.visit_results.extend([MCTSResults(r) for r in batch_results])

    def unpack(self):
        return list(zip(*(results.unpack() for results in self.visit_results)))


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

    def __init__(self, results_colnames: Iterable[str], **kwargs):
        self.columns = tuple(results_colnames)
        self.ncols = len(self.columns)

        self.table: dict[str, StateTable] = dict()

    def __getitem__(self, state):
        return self.table[state]

    def __missing__(self, state):
        self.table[state] = StateTable(state)
        return self.table[state]

    def __contains__(self, state_visit):
        return state_visit in self.table

    def __len__(self):
        return len(self.table)

    def n_visits(self, state):
        return len(self.table[state])

    def get_reward(self, state, visit):
        return self.table[state][visit].reward

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


class TableBuffer(TranspositionTable):
    """A buffer for storing results of MCTS with stochastic playouts.
    Results are stored the same as TranspositionTable, but when the buffer reaches
    a certain size, it is flushed to disk to periodically save results."""

    def __init__(
        self,
        columns: Iterable[str],
        save_dir: Path,
        maxsize: int = 1000,
        extension: str = "parquet",
        save_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(columns, **kwargs)
        self.maxsize = maxsize
        self.extension = extension
        self.flush_counter = 0
        self.save_dir = Path(save_dir)
        self.save_kwargs = save_kwargs or {}

    def __setitem__(self, state_visit, value):
        super().__setitem__(state_visit, value)
        if len(self) >= self.maxsize:
            self.flush()

    def clear(self):
        self.table = dict()

    def flush(self, extension: str = None, **kwargs):
        """Save buffer to disk, clear buffer, and increment flush counter."""
        extension = extension or self.extension
        filename = self.save_dir.joinpath(f"{self.flush_counter}.{extension}")
        kw = kwargs | self.save_kwargs
        if extension == "csv":
            self.to_csv(filename, **kw)
        elif extension == "parquet":
            self.to_parquet(filename, **kw)
        elif extension == "hdf":
            self.to_hdf(filename, **kw)
        else:
            raise ValueError(f"Unknown file extension {extension}")

        self.flush_counter += 1
        self.clear()

    def reset(self, flush: bool = False, **kwargs):
        if flush:
            self.flush()
        else:
            self.clear()
        self.flush_counter = 0


class DefaultFactoryDict(dict):
    """Similar to a defaultdict, but the default value is a function of the key."""

    def __init__(self, *args, default_factory=None, **kwargs):
        if default_factory is None:
            raise ValueError("Must provide a default_factory")
        super().__init__(*args, **kwargs)
        self._default_factory = default_factory

    def __missing__(self, key):
        self[key] = self._default_factory(key)
        return self[key]


class ParallelTree(CircuiTree):
    def __init__(
        self,
        *args,
        model_factory: Optional[Callable] = None,
        model_table: Optional[DefaultFactoryDict] = None,
        transposition_table: Optional[TranspositionTable | bool] = None,
        columns: Optional[Iterable[str]] = None,
        buffer: Optional[TableBuffer | bool] = None,
        save_dir: Optional[Path] = None,
        maxsize: int = 1000,
        extension: str = "parquet",
        save_kwargs: dict = None,
        counter: Counter = None,
        table_lock: Any = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if model_table is None:
            self._model_table = DefaultFactoryDict(default_factory=model_factory)
        else:
            self._model_table = model_table

        if transposition_table is False:
            self._transposition_table = None
        elif transposition_table is None:
            self._transposition_table = TranspositionTable(results_colnames=columns)
        else:
            self._transposition_table = transposition_table

        if buffer is False:
            self._buffer = None
        elif buffer is None:
            if columns is None or save_dir is None:
                raise ValueError("Must provide a save_dir for buffer")
            self._buffer = TableBuffer(
                columns=columns,
                save_dir=save_dir,
                maxsize=maxsize,
                extension=extension,
                save_kwargs=save_kwargs,
            )
        else:
            self._buffer = buffer

        self.visit_counter = Counter(counter)

        if table_lock is None:
            self._table_lock = threading.Lock()
        else:
            self._table_lock = table_lock

    @property
    def table_lock(self):
        return self._table_lock

    @property
    def model_table(self):
        return self._model_table

    @property
    def transposition_table(self):
        return self._transposition_table

    @property
    def buffer(self):
        return self._buffer

    def get_reward(self, state) -> float | int:
        visit_num = self.visit_counter.get(state, 0)
        state_visit = (state, visit_num)
        if state_visit in self.transposition_table:
            reward = self.transposition_table.get_reward(state_visit)
        else:
            results = self._compute_results(state)

            # Acquire a lock to prevent race conditions - i.e. multiple threads writing
            # to the transposition table and buffer table at the same time
            with self.table_lock:
                self.transposition_table[state_visit] = results
                self.buffer[state_visit] = results

            reward = results[0]

        self.visit_counter[state] += 1
        return reward

    def _compute_results(self, state) -> float | int:
        raise NotImplementedError
