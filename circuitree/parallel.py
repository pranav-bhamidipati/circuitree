from abc import abstractmethod
from collections import Counter
from numpy.random import default_rng, SeedSequence
import numpy as np
from multiprocessing import cpu_count
from typing import Any, Callable, Optional

from .circuitree import CircuiTree

__all__ = [
    "MultithreadedCircuiTree",
    "search_mcts_in_thread",
]


class MultithreadedCircuiTree(CircuiTree):
    def __init__(
        self,
        threads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if threads is None:
            threads = cpu_count()
        self.threads = threads

        seq = SeedSequence(self.seed)
        self._random_generators: list[np.random.Generator] = [
            default_rng(s) for s in seq.spawn(threads)
        ]

        self.sample_counter = Counter()

        # Attributes that should not be saved to file
        self._non_serializable_attrs.extend(
            [
                "_random_generators",
                "sample_counter",
            ]
        )

    @abstractmethod
    def get_reward(self, node: Any, sample_number: int, **kwargs) -> float | int:
        """Given a terminal node and the number of samples to that node, compute the
        reward for that node. Note the difference in call signature compared to
        CircuiTree.get_reward"""
        raise NotImplementedError

    def traverse(self, thread_idx: int, **kwargs):
        # Select the next state to sample and the terminal state to be simulated.
        # Expands a child if possible.
        rg = self._random_generators[thread_idx]
        selection_path = self.select_and_expand(rg=rg)
        sim_node = self.get_random_terminal_descendant(selection_path[-1], rg=rg)

        # Between backprop of visit and reward, we incur virtual loss
        self.backpropagate_visit(selection_path)

        # Keep track of samples to terminal nodes
        sample_number = self.sample_counter[sim_node]
        self.sample_counter[sim_node] += 1
        reward = self.get_reward(sim_node, sample_number, **kwargs)
        self.backpropagate_reward(selection_path, reward)

        return selection_path, reward, sim_node


def search_mcts_in_thread(
    thread_idx: int,
    mtree: MultithreadedCircuiTree,
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
