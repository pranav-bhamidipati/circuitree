from abc import abstractmethod
from collections import Counter
from numpy.random import default_rng, SeedSequence
import numpy as np
from multiprocessing import cpu_count
from typing import Any, Callable, Iterable, Optional

from .models import DimersGrammar, SimpleNetworkGrammar

from .circuitree import CircuiTree

__all__ = [
    "MultithreadedCircuiTree",
    "ParallelNetworkTree",
    "ParallelDimerTree",
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


class ParallelNetworkTree(MultithreadedCircuiTree):
    def __init__(
        self,
        components: Iterable[Iterable[str]],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        root: Optional[str] = None,
        cache_maxsize: int = 128,
        **kwargs,
    ):
        grammar = SimpleNetworkGrammar(
            components=components,
            interactions=interactions,
            max_interactions=max_interactions,
            root=root,
            cache_maxsize=cache_maxsize,
        )
        super().__init__(grammar=grammar, root=root, **kwargs)


class DimerNetworkTree(CircuiTree):
    """
    DimerNetworkTree
    =================
    A CircuiTree for the design space of dimerizing TF networks. Intended to recapitulate
    the dimerization of zinc-finger proteins.

    Models a system of dimerizing transcription factors (e.g. zinc-fingers) that regulate
    each other's transcription. The circuit consists a set of ``components``, which
    represent transcription factors that are being regulated. Components can form homo-
    or heterodimers that bind to a component's promoter region and regulate
    transcription. There is also a set of ``regulators``, which can dimerize and regulate
    transcription but are not themselves regulated. Regulator-regulator homodimers and
    regulator-component heterodimers can act as TFs, but regulator-regulator homodimers
    are assumed to be inactive.

    The circuit topology (also referred to as the "state" during search) is encoded using
    a string representation (aka "genotype") with the following rules:
        - Components and regulators are represented by single uppercase characters
        - Interactions are represented by a 4-character string
            - Characters 1-2 (uppercase): the dimerizing species (components/regulators)
            - Character 3 (lowercase): the type of regulation upon binding
            - Character 4 (uppercase): the target of regulation (a component)
        - Components are separated from regulators by a ``+``
        - Components/regulators are separated from interactions by a ``::``
        - Interactions are separated from one another by underscores ``_``
        - A terminal assembly is denoted with a leading asterisk ``*``

        For example, the following string represents a 2-component MultiFate system that
        has not been fully assembled (lacks the terminal asterisk):

            ``AB+::AAa_BBa``

        While the following string represents a terminally assembled 2-component
        MultiFate system with a regulator L that flips the system into the A state:

            ``*AB+L::AAa_ALa_BBa_BLi``

    """

    def __init__(
        self,
        components: Iterable[str],
        regulators: Iterable[str],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        max_interactions_per_promoter: int = 2,
        root: Optional[str] = None,
        cache_maxsize: int = 128,
        **kwargs,
    ):
        grammar = DimersGrammar(
            components=components,
            regulators=regulators,
            interactions=interactions,
            max_interactions=max_interactions,
            max_interactions_per_promoter=max_interactions_per_promoter,
            root=root,
            cache_maxsize=cache_maxsize,
        )
        super().__init__(grammar=grammar, root=root, **kwargs)


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
