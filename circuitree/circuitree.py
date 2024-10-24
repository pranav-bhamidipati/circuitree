from abc import ABC, abstractmethod
from functools import partial
from itertools import cycle, chain, islice, repeat
import json
from pathlib import Path
from typing import Callable, Hashable, Iterator, Literal, Optional, Iterable, Any
import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
import warnings

from .grammar import CircuitGrammar

__all__ = [
    "ucb_score",
    "CircuiTree",
]


def ucb_score(
    graph: nx.DiGraph,
    parent: Hashable,
    node: Hashable,
    exploration_constant: Optional[float] = np.sqrt(2.0),
    **kw,
):
    """
    Compute the UCB score for a node in the search graph.

    The UCB score balances exploitation and exploration during tree search.
    It combines the mean reward of a child node with an exploration term
    based on the visit counts of the parent and child nodes.

    Parameters:
        graph (nx.DiGraph): The directed search graph.
        parent (str): The ID of the parent node of the target node.
        node (str): The ID of the target node for which to compute the UCB score.
        exploration_constant (float, optional): The exploration constant that
            controls the balance between exploitation and exploration. Defaults to
            ``np.sqrt(2.0)``.

    Returns:
        float: The UCB score of the target node.
    """
    attrs = graph.edges[parent, node]

    visits = attrs["visits"]
    if visits == 0:
        return np.inf
    reward = attrs["reward"]
    parent_visits = graph.nodes[parent]["visits"]

    mean_reward = reward / visits
    exploration_term = exploration_constant * np.sqrt(np.log(parent_visits) / visits)
    ucb = mean_reward + exploration_term
    return ucb


class ExhaustionError(RuntimeError):
    """Raised when the entire search tree is exhaustively sampled."""


class CircuiTree(ABC):
    """A base class for implementing the Monte Carlo Tree Search (MCTS) algorithm for
    circuit topologies. The class defines the basic structure of the search tree and
    provides methods for tree traversal, expansion, and backpropagation.

    The class is designed to be subclassed with a specific implementation of the
    ``get_reward()`` method, which calculates the reward for a given state. The class
    also requires a ``CircuitGrammar`` object to define the search space. There are also
    methods for extracting meaningful patterns (motifs) from the search graph, by
    comparing successful and unsuccessful paths in the graph. These methods are
    experimental and may be removed in future versions.
    """

    def __init__(
        self,
        grammar: CircuitGrammar,
        root: str,
        exploration_constant: Optional[float] = None,
        seed: Optional[int] = None,
        n_exhausted: Optional[int | float] = None,
        graph: Optional[nx.DiGraph] = None,
        tree_shape: Optional[Literal["tree", "dag"]] = None,
        compute_unique: bool = True,
        **kwargs,
    ):
        # Initialize RNG
        self.rg = np.random.default_rng(seed)
        self.seed: int = self.rg.bit_generator._seed_seq.entropy

        # Initialize search graph
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

        # Decide whether to compute the uniqueness of each visited state
        self.compute_unique = compute_unique
        if tree_shape is not None:
            if tree_shape not in ("tree", "dag"):
                raise ValueError("Argument `tree_shape` must be `tree` or `dag`.")
            warnings.warn(
                "The `tree_shape` argument is deprecated and will be removed in a "
                "future version. Please use `compute_symmetries` instead."
            )

        # Grammar defining the search space
        self.grammar = grammar

        # Exploration constant for UCB
        if exploration_constant is None:
            self.exploration_constant = np.sqrt(2)
        else:
            self.exploration_constant = exploration_constant

        # Optionally mark terminal nodes as exhausted if they have been visited enough
        self.n_exhausted = n_exhausted

        # Attributes that should not be serialized to JSON
        self._non_serializable_attrs = [
            "_non_serializable_attrs",
            "rg",
            "graph",
        ]

    @abstractmethod
    def get_reward(self, state: Hashable, **kwargs) -> float | int:
        """
        Abstract method that calculates the reward for a given state.

        This method must be implemented by a subclass of ``CircuiTree``. It defines
        the interface for reward calculation within the framework.

        Args:
            state (Hashable): The state for which to calculate the reward.

        Returns:
            float | int: The reward for the given state. Reward values can be
                deterministic or stochastic. The range of possible outputs should be
                finite and ideally normalized to the range [0, 1].
        """

        pass

    @property
    def default_attrs(self):
        """Default attributes for nodes and edges in the search graph."""
        return dict(visits=0, reward=0)

    @property
    def terminal_states(self) -> Iterator[Hashable]:
        """
        Generator that yields terminal states in the search graph.

        This property provides an iterator that traverses the terminal states
        present in the search graph associated with the ``CircuiTree`` instance.

        Yields:
            Hashable: Each element returned by the iterator represents a terminal
            state within the search graph (see ``CircuitGrammar.is_terminal``).

        Example:

        .. code-block:: python

            # Find terminal states with mean reward > 0.5
            for state in tree.terminal_states:
                if tree.graph.nodes[state]["reward"] / tree.graph.nodes[state]["visits"] > 0.5:
                    print(state)

        """

        return (node for node in self.graph.nodes if self.grammar.is_terminal(node))

    def _do_action(self, state: Hashable, action: Hashable) -> Hashable:
        """
        Apply an action to a state and compute a unique representation (internal method).

        This is an internal function that applies an action to the given state using the
        supplied grammar. It then optionally computes a unique representation of the
        resulting state, depending on the ``compute_unique`` attribute.

        Args:
            state (Hashable): The current state.
            action (Hashable): The action to apply to the state.

        Returns:
            Hashable: The resulting state after applying the action. If
                ``self.compute_unique`` is True, ``self.grammar.get_unique_state()`` is
                called on the resulting state before returning.

        """

        new_state = self.grammar.do_action(state, action)
        if self.compute_unique:
            new_state = self.grammar.get_unique_state(new_state)
        return new_state

    def _undo_action(self, state: Hashable, action: Hashable) -> Hashable:
        """(Experimental) Undo one action from the given state."""

        if state == self.root:
            return None
        new_state = self.grammar.undo_action(state, action)
        if self.compute_unique:
            new_state = self.grammar.get_unique_state(new_state)
        return new_state

    @staticmethod
    def _get_random_terminal_descendant(
        grammar: CircuitGrammar, start: Hashable, rg: np.random.Generator
    ) -> Hashable:
        """
        Sample a random terminal state by following random actions from a given start
        state.

        Performs a random walk on the search space starting from the specified ``start``
        state by recursively selecting random actions until a terminal state is reached,
        as determined by the ``grammar`` (see :func:`CircuitGrammar.is_terminal`).

        Args:
            grammar (CircuitGrammar): The grammar associated with the search space.
            start (Hashable): The starting state for the random walk.
            rg (np.random.Generator): A NumPy random number generator.

        Returns:
            Hashable: A randomly sampled terminal state.

        """

        state = start
        while not grammar.is_terminal(state):
            actions = grammar.get_actions(state)
            action = rg.choice(actions)
            state = grammar.do_action(state, action)
            state = grammar.get_unique_state(state)
        return state

    @staticmethod
    def _sample_from_terminals_with_rejection(
        terminals: set[Hashable],
        grammar: CircuitGrammar,
        start: Hashable,
        seed: int,
        max_iter: Optional[int] = None,
    ) -> Hashable:
        """
        Sample from a set of terminal states using rejection sampling.

        A state is sampled by performing a random walk on the search space from the
        ``start`` state until a terminal state is reached that belongs to the ``terminals``
        set.

        Args:
            terminals (set[Hashable]): The set of valid terminal states for acceptance.
            grammar (CircuitGrammar): The grammar associated with the search space.
            start (Hashable): The starting state for the random walk.
            seed (int): Seed for a NumPy random number generator.
            max_iter (Optional[int], optional): The maximum number of iterations allowed
                for sampling. Defaults to 1e8.

        Raises:
            RuntimeError: If the maximum number of iterations is reached without finding
                a suitable terminal state.

        Returns:
            Hashable: A randomly sampled terminal state in ``terminals``.

        """

        max_iter = max_iter or 100_000_000
        rg = np.random.default_rng(seed)
        terminals = set(terminals)

        for _ in range(max_iter):
            state = start
            while not grammar.is_terminal(state):
                actions = grammar.get_actions(state)
                action = rg.choice(actions)
                state = grammar.do_action(state, action)
                state = grammar.get_unique_state(state)
            if state in terminals:
                return state
        raise RuntimeError(f"Maximum number of iterations reached: {max_iter}")

    def get_random_terminal_descendant(
        self, start: Hashable, rg: Optional[np.random.Generator] = None
    ) -> Hashable:
        """
        Sample a random terminal state by following random actions from a given start
        state.

        Args:
            start (Hashable): The starting state from which to begin the random walk.
            rg (Optional[np.random.Generator], optional): Defaults to None, in which
            case the ``rg`` attribute of the ``CircuiTree`` instance is used.

        Returns:
            Hashable: The randomly sampled terminal state.

        """

        rg = self.rg if rg is None else rg
        return self._get_random_terminal_descendant(self.grammar, start, rg)

    def select_and_expand(
        self, rg: Optional[np.random.Generator] = None
    ) -> list[Hashable]:
        """Selects a path through the search graph using the UCB score. Adds the last
        node to the graph if it is not already present.

        This method implements the core selection and expansion step of the UCB algorithm.
        It iteratively selects the child node with the highest UCB score until a terminal
        state or an unexpanded edge is encountered.

        Args:
            rg (Optional[np.random.Generator], optional): Random number generator.
                Defaults to None, in which case the ``rg`` attribute of the ``CircuiTree``
                instance is used.

        Returns:
            list[Hashable]: A list of states representing the selected path through the
            search graph.
        """

        rg = self.rg if rg is None else rg

        # Start at root
        node = self.root
        selection_path = [node]
        actions = self.grammar.get_actions(node)
        while actions:
            max_ucb = -np.inf
            best_child = None
            rg.shuffle(actions)
            for action in actions:
                child = self._do_action(node, action)

                # Skip this child if it has been exhausted (i.e. all of its terminal
                # descendants have been expanded and simulated >= n_exhausted times)
                if (
                    self.n_exhausted is not None
                    and child in self.graph.nodes
                    and self.graph.nodes[child].get("is_exhausted", False)
                ):
                    continue

                ucb = self.get_ucb_score(node, child)

                # An unexpanded edge has UCB score of infinity.
                # In this case, expand and select the child.
                if ucb == np.inf:
                    self.expand_edge(node, child)
                    selection_path.append(child)
                    return selection_path

                # Otherwise, recursively pick the child with the highest UCB score.
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_child = child

            node = best_child
            selection_path.append(node)
            actions = self.grammar.get_actions(node)

        # If no actions can be taken, we have reached a terminal state.

        # Mark a terminal state "exhausted" if it has been visited n_exhausted times.
        # An exhausted state is considered to have been simulated to sufficient depth.
        # Its results are not updated further, its parent nodes are modified to
        # forget results from this node, and it is skipped during subsequent selection.
        if self.graph.nodes[node].get("visits", 0) >= self.n_exhausted - 1:
            self.mark_as_exhausted(node)

        return selection_path

    def mark_as_exhausted(self, node: str) -> None:
        """Marks a terminal node as exhausted by setting its attribute "is_exhausted" to
        True in the search graph. Its parent nodes are modified to forget results from
        this node (i.e. the visits are decremented and the reward is subtracted from the
        total reward). If all nodes of the parent are exhausted, the parent is also
        marked as exhausted - this is done recursively up the tree.

        Args:
            node (str): The terminal node to mark as exhausted.

        Raises:
            ExhaustionError: If all nodes in the tree have been visited to exhaustion.
        """
        self.graph.nodes[node]["is_exhausted"] = True

        # If the whole tree is exhausted, we are done
        if node == self.root:
            raise ExhaustionError(
                "Every node in the tree has been visited to exhaustion."
            )

        # Forget results from this node
        for parent in self.graph.predecessors(node):
            edge_visits = self.graph.edges[parent, node]["visits"]
            edge_reward = self.graph.edges[parent, node]["reward"]
            self.graph.nodes[parent]["visits"] -= edge_visits
            self.graph.nodes[parent]["reward"] -= edge_reward

        # Recursively mark parent nodes as exhausted
        for parent in self.graph.predecessors(node):
            if all(
                self.graph.nodes[c].get("is_exhausted", False)
                for c in self.graph.successors(parent)
            ):
                self.mark_as_exhausted(parent)

    def expand_edge(self, parent: Hashable, child: Hashable):
        """Expands the search graph by adding the child node and/or the parent-child
        edge to the search graph if they do not already exist.

        Args:
            parent (Hashable): The parent node of the edge.
            child (Hashable): The child node.
        """

        if not self.graph.has_node(child):
            self.graph.add_node(child, **self.default_attrs)
        self.graph.add_edge(parent, child, **self.default_attrs)

    def get_ucb_score(self, parent: Hashable, child: Hashable):
        """Calculates the UCB score for a child node given its parent.

        Args:
            parent (Hashable): The parent node.
            child (Hashable): The child node.

        Returns:
            float: The UCB score of the child node.
        """

        if self.graph.has_edge(parent, child):
            return ucb_score(self.graph, parent, child, self.exploration_constant)
        else:
            return np.inf

    def _backpropagate(self, path: list, attr: str, value: float | int):
        """Update the value of an attribute for each node and edge in the path.

        Args:
            path (list): The list of nodes in the path.
            attr (str): The attribute to update.
            value (float | int): The value to add to the attribute.
        """
        _path = path.copy()
        child = _path.pop()
        self.graph.nodes[child][attr] += value
        while _path:
            parent = _path.pop()
            self.graph.edges[parent, child][attr] += value
            child = parent
            self.graph.nodes[child][attr] += value

    def backpropagate_visit(self, selection_path: list) -> None:
        """Increment the visit count for each node and edge in ``selection_path``.

        Notes:

        * The reward function (``get_reward``) is called after ``backpropagate_visit`` and
          before ``backpropagate_reward``.
        * In parallel mode, each node and edge in the selection path incurs "virtual loss"
          until the reward is computed and backpropagated. This is because the visit
          count is incremented before the actual reward is known.

        Args:
            selection_path (list): The list of nodes in the selection path.
        """
        self._backpropagate(selection_path, "visits", 1)

    def backpropagate_reward(self, selection_path: list, reward: float | int):
        """Update the reward for each node and edge in ``selection_path``.

        Notes:

        * The reward function (``get_reward``) is called after ``backpropagate_visit`` and
          before ``backpropagate_reward``.
        * In parallel mode, each node and edge in the selection path incurs "virtual loss"
          until the reward is computed and backpropagated. This is because the visit
          count is incremented before the actual reward is known.

        Args:
            selection_path (list): The list of nodes in the selection path.
            reward (float | int): The reward value to add to each node and edge.
        """
        self._backpropagate(selection_path, "reward", reward)

    def traverse(self, **kwargs) -> tuple[list[Hashable], float | int, Hashable]:
        """Performs a single iteration of the MCTS algorithm.

        This method implements the core traversal step of the UCB algorithm. It selects
        a path through the search tree using UCB scores, expands the tree if necessary,
        obtains a reward estimate by simulating a random downstream terminal state, and
        updates the search graph based on the reward value.

        Notes:

        * The reward function (``get_reward``) is called after ``backpropagate_visit`` and
          before ``backpropagate_reward``.
        * In parallel mode, each node and edge in the selection path incurs "virtual loss"
          until the reward is computed and backpropagated. This is because the visit
          count is incremented before the actual reward is known.

        Args:
            **kwargs: Additional keyword arguments passed to the :func:``get_reward`` method.

        Returns:
            tuple[list[Hashable], float | int, Hashable]:
                - selection_path (list): The selected path through the search graph.
                - reward (float | int): The reward obtained from the simulation.
                - sim_node (Hashable): The simulated terminal state.
        """

        # Select the next state to sample and the terminal state to be simulated.
        # Expands a child if possible.
        selection_path = self.select_and_expand()
        sim_node = self.get_random_terminal_descendant(selection_path[-1])

        # Between backprop of visit and reward, we incur virtual loss
        self.backpropagate_visit(selection_path)
        reward = self.get_reward(sim_node, **kwargs)
        self.backpropagate_reward(selection_path, reward)

        return selection_path, reward, sim_node

    def grow_tree(
        self,
        root: Hashable = None,
        n_visits: int = 0,
        print_updates: bool = False,
        print_every: int = 1000,
    ):
        """Exhaustively expand the search tree from a root node (not recommended for large spaces).

        This method performs a depth-first search expansion of the search graph,
        adding all possible nodes and edges based on the grammar until no new states
        can be reached.

        **Warning:** This method can be computationally expensive and memory-intensive
        for large search spaces.

        Args:
            root (Hashable, optional): The starting node for the expansion. Defaults to None.
            n_visits (int, optional): The initial visit count for all added nodes. Defaults to 0.
            print_updates (bool, optional): Whether to print information about the number of
                added nodes during growth. Defaults to False.
            print_every (int, optional): The frequency at which to print updates
                (if print_updates is True). Defaults to 1000.
        """

        if root is None:
            root = self.root
            if print_updates:
                print(f"Adding root: {root}")
            self.graph.add_node(root, visits=n_visits, reward=0)

        stack = [(root, action) for action in self.grammar.get_actions(root)]
        n_added = 1
        while stack:
            node, action = stack.pop()
            if not self.grammar.is_terminal(node):
                next_node = self._do_action(node, action)
                if next_node not in self.graph.nodes:
                    n_added += 1
                    self.graph.add_node(next_node, visits=n_visits, reward=0)
                    stack.extend(
                        [(next_node, a) for a in self.grammar.get_actions(next_node)]
                    )
                    if print_updates:
                        if n_added % print_every == 0:
                            print(f"Graph size: {n_added} nodes.")
                if not self.graph.has_edge(node, next_node):
                    self.graph.add_edge(node, next_node, visits=n_visits, reward=0)

    def bfs_iterator(self, root=None, shuffle=False) -> Iterator[Hashable]:
        """Iterate over all terminal nodes in breadth-first (BFS) order starting from
        a root node."""
        root = self.root if root is None else root
        layers = (l for l in nx.bfs_layers(self.graph, root))

        if shuffle:
            layers = list(layers)
            for l in layers:
                self.rg.shuffle(l)

        return filter(self.grammar.is_terminal, chain.from_iterable(layers))

    def search_bfs(
        self,
        n_steps: Optional[int] = None,
        n_repeats: Optional[int] = None,
        n_cycles: Optional[int] = None,
        callback: Optional[Callable] = None,
        callback_every: int = 1,
        shuffle: bool = False,
        progress: bool = False,
        run_kwargs: Optional[dict] = None,
    ) -> None:
        """Performs a breadth-first search (BFS) traversal of the search graph.

        This method iterates over the search graph in a breadth-first manner, visiting
        nodes layer by layer. The number of iterations can be controlled using various
        parameters:

        * ``n_steps``: Stop the search after a fixed number of total iterations over all
          explored nodes (considering repeats and cycles).
        * ``n_repeats``: For each node encountered during BFS traversal, repeat the visit
          ``n_repeats`` times before moving on to the next node in the layer.
        * ``n_cycles``: Repeat the entire BFS traversal `n_cycles` times.

        Args:
            n_steps (Optional[int], optional): Number of total iterations (repeats and
                cycles considered). Defaults to None.
            n_repeats (Optional[int], optional): Number of repeats per node. Defaults to
                None.
            n_cycles (Optional[int], optional): Number of BFS traversal cycles. Defaults
                to None.
            callback (Optional[Callable], optional): A callback function to be executed
                at specific points during the search. Defaults to None. The callback is
                called with three arguments:

                - ``tree``: The ``CircuiTree`` instance calling the callback.
                - ``node``: The current node being visited (value is None during
                  initialization).
                - ``reward``: The reward obtained from the node (value is None during
                  initialization).

            callback_every (int, optional): How often to call the callback (in terms of
                iterations). Defaults to 1.
            shuffle (bool, optional): If True, shuffles the order of nodes within each
                BFS layer. Defaults to False.
            progress (bool, optional): If True, displays a progress bar during the
                search. Defaults to False.
            run_kwargs (Optional[dict], optional): A dictionary of additional keyword
                arguments passed to the ``get_reward`` method during node evaluation.
                Defaults to None.

        Raises:
            ValueError: If both ``n_steps`` and ``n_cycles`` are specified (exactly one should be provided).
        """

        if self.graph.number_of_nodes() < 2:
            self.graph.add_node(self.root, **self.default_attrs)
            self.grow_tree(root=self.root, n_visits=0)

        if run_kwargs is None:
            run_kwargs = {}

        iterator = self.bfs_iterator(root=self.root, shuffle=shuffle)
        if not ((n_steps is None) ^ (n_cycles is None)):
            raise ValueError("Must specify exactly one of n_steps or n_cycles.")

        ### Iterate in BFS order
        # If n_repeats is specified, repeat each node n_repeats times before moving on
        # If n_cycles is specified, repeat the entire BFS traversal n_cycles times
        # Otherwise if n_steps is specified, stop after n_steps total iterations
        if n_repeats is not None:
            iterator = chain.from_iterable(repeat(elem, n_repeats) for elem in iterator)
        if n_cycles is not None:
            iterator = chain.from_iterable(repeat(iterator, n_cycles))
        elif n_steps is not None:
            iterator = islice(cycle(iterator), n_steps)

        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="BFS search")

        if callback is not None:
            callback(self, None, None)

        for i, node in enumerate(iterator):
            self.graph.nodes[node]["visits"] += 1
            reward = self.get_reward(node, **run_kwargs)
            self.graph.nodes[node]["reward"] += reward

            if callback is not None and i % callback_every == 0:
                _ = callback(self.graph, node, reward)

    def search_mcts(
        self,
        n_steps: int,
        callback: Optional[Callable] = None,
        callback_every: int = 1,
        progress_bar: bool = False,
        run_kwargs: Optional[dict] = None,
        callback_before_start: bool = True,
    ) -> None:
        """Performs a Monte Carlo Tree Search (MCTS) traversal of circuit topology space.

        This method implements the core MCTS algorithm for exploring and exploiting the
        search tree. It performs a sequence of ``n_steps`` iterations, each consisting of
        selection, expansion, simulation, and backpropagation steps. If provided, a
        callback function is called at specific points during the search. This can be
        used for various purposes:

        - Logging search progress
        - Backing up intermediate results
        - Recording search statistics
        - Checking for convergence or early stopping conditions

        Args:
            n_steps (int): The total number of MCTS iterations to perform.
            callback_every (int, optional): How often to call the callback function
                (in terms of iterations). Defaults to 1 (every iteration).
            callback (Optional[Callable], optional): A callback function to be executed
                at specific points during the search. Defaults to None. The callback is
                called with five arguments:

                - ``tree``: The ``CircuiTree`` instance calling the callback.
                - ``step``: The current MCTS iteration (0-based index).
                - ``path``: A list of nodes representing the selected path in the tree.
                - ``sim_node``: The state used for the simulation step. Chosen by
                  following random actions from the last node in the path until
                  a terminal state is reached.
                - ``reward``: The reward obtained from the simulation step.

            progress_bar (bool, optional): If True, displays a progress bar during the
                search. Defaults to False. Requires the ``tqdm`` package.
            run_kwargs (Optional[dict], optional): A dictionary of additional keyword
                arguments passed to the ``get_reward`` method during node evaluation.
                Defaults to None.
            callback_before_start (bool, optional): Whether to call the callback before
                starting the search. If so, the callback is called with ``step = -1``.
                Defaults to True.

        Returns:
            None
        """

        if progress_bar:
            from tqdm import trange

            iterator = trange(n_steps, desc="MCTS search")
        else:
            iterator = range(n_steps)

        # Optionally run the callback before starting the search
        if callback is not None and callback_before_start:
            callback(self, -1, [None], None, None)

        run_kwargs = {} if run_kwargs is None else run_kwargs
        print(f"Starting MCTS search with {n_steps} iterations.")
        if callback is None:
            self._run_mcts(self, iterator, **run_kwargs)
        else:
            self._run_mcts_with_callback(
                self, iterator, callback, callback_every, **run_kwargs
            )
        return

    @staticmethod
    def _run_mcts(tree: "CircuiTree", iterator: Iterable[int], **kwargs) -> None:
        """(Internal) Runs an MCTS search."""
        # Performs MCTS iterations on the tree using the provided iterator
        for _ in iterator:
            tree.traverse(**kwargs)

    @staticmethod
    def _run_mcts_with_callback(
        tree: "CircuiTree",
        iterator: Iterable[int],
        callback: Optional[Callable],
        callback_every: int,
        **kwargs,
    ) -> None:
        """(Internal) Runs an MCTS search with a callback."""
        for i in iterator:
            selection_path, reward, sim_node = tree.traverse(**kwargs)
            if callback is not None and i % callback_every == 0:
                callback(tree, i, selection_path, sim_node, reward)

    def search_mcts_parallel(
        self,
        n_steps: int,
        n_threads: int,
        callback: Optional[Callable] = None,
        callback_every: int = 1,
        callback_before_start: bool = True,
        run_kwargs: Optional[dict] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """Performs a Monte Carlo Tree Search (MCTS) in parallel using multiple threads.
        This method leverages the ``gevent`` library (included with
        ``circuitree[distributed]``) to execute the MCTS search algorithm across multiple
        execution threads on the same search graph.

        Key differences from ``search_mcts``:

        * This function utilizes multiple threads for parallel execution, whereas
          ``search_mcts`` runs sequentially on a single thread
        * For intended performance, reward computations should be performed by a separate
          pool of worker processes (see User Guide > Parallelization)
        * Requires the ``gevent`` library

        Args:
            n_steps (int): The total number of MCTS iterations to perform (divided among threads).
            n_threads (int): The number of threads to use for parallel MCTS. Must be at least 1.
            callback (Optional[Callable], optional): A callback function to be executed
                at specific points during the search. Defaults to None. (See ``search_mcts`` docstring for details).
            callback_every (int, optional): How often to call the callback function
                (in terms of iterations). Defaults to 1 (every iteration).
            callback_before_start (bool, optional): Whether to call the callback before
                starting the search (step=-1). Defaults to True.
            run_kwargs (Optional[dict], optional): A dictionary of additional keyword arguments
                passed to the ``get_reward`` method during node evaluation. Defaults to None.
            logger (Optional[Any], optional): A logger object to be used for logging messages
                during the search. Can be useful for monitoring progress. Defaults to None.

        Raises:
            ImportError: If ``gevent`` is not installed.
            ValueError: If the number of threads is less than 1.
        """

        # Check if the `gevent` package is installed
        try:
            import gevent
        except ImportError:
            raise ImportError(
                "The gevent package is required to run parallel MCTS. You can install "
                "it with `pip install gevent` or as a dependency of circuitree with "
                "`pip install circuitree[distributed]`."
            )

        if n_threads < 1:
            raise ValueError("Number of threads must be at least 1.")
        if n_threads == 1:
            print("Detected n_threads==1. Running MCTS with a single thread.")
            self.search_mcts(
                n_steps=n_steps,
                callback_every=callback_every,
                callback=callback,
                run_kwargs=run_kwargs,
            )
            return

        # Optionally run the callback before starting the search
        if callback is not None and callback_before_start:
            callback(self, -1, [None], None, None)

        # Distribute the steps evenly among the threads
        quotient, remainder = divmod(n_steps, n_threads)
        n_per_thread = [quotient + 1] * remainder + [quotient] * (n_threads - remainder)
        if remainder:
            n_repr = f"{quotient}-{quotient + 1}"
        else:
            n_repr = f"{quotient}"
        start_msg = (
            f"Starting MCTS search with {n_steps} iters on {n_threads} threads "
            f"({n_repr} iters per thread)."
        )

        # Pass logger to the get_reward() function if supplied and log the start message
        run_kwargs = {} if run_kwargs is None else run_kwargs
        if logger is not None:
            run_kwargs["logger"] = logger
            logger.info()
        print(start_msg)

        if callback is None:
            gthreads = [
                gevent.spawn(self._run_mcts, self, range(n), **run_kwargs)
                for n in n_per_thread
            ]
            gevent.joinall(gthreads)
        else:
            gthreads = [
                gevent.spawn(
                    self._run_mcts_with_callback,
                    self,
                    range(n),
                    callback,
                    callback_every,
                    **run_kwargs,
                )
                for n in n_per_thread
            ]
            gevent.joinall(gthreads)

        return

    def is_success(self, terminal_state: Hashable) -> bool:
        """
        Determines whether a state represents a successful outcome in the search.

        Designed to be implemented in a subclass, since the definition of success
        depends on the specific search problem. It takes a terminal state as input, which
        represents a potential solution for the design problem.

        Args:
            terminal_state (Hashable): The state to evaluate for success.

        Raises:
            NotImplementedError: This base class implementation is intended to be
            overridden in subclasses.

        Returns:
            bool: Whether the provided state represents a successful outcome (True) or
            not (False).
        """

        raise NotImplementedError

    def copy_graph(self) -> nx.DiGraph:
        """Return a shallow copy of the search graph (the ``graph`` attribute). Use
        ``copy.deepcopy()`` for a deep copy."""
        return self.graph.copy()

    def get_attributes(self, attrs_copy: Optional[Iterable[str]]) -> dict:
        """Returns a dictionary of the object's attributes.

        This method allows controlled access to the object's attributes, potentially
        excluding ones that should not be serialized when writing to a JSON file. For
        any attribute with a ``to_dict()`` method, the output of that method is used.
        Otherwise, the attribute is copied directly.

        Args:
            attrs_copy (Optional[Iterable[str]]): An optional list of attribute names
                to include. If None, all attributes except those in
                ``_non_serializable_attrs`` are returned.

        Raises:
            ValueError: If ``attrs_copy`` contains attributes from ``_non_serializable_attrs``.

        Returns:
            dict: A dictionary containing the requested attributes. Attributes with a
            ``to_dict()`` method are converted using that method, otherwise they are
            copied directly.
        """

        # Get the attributes to copy
        if attrs_copy is None:
            keys = set(self.__dict__.keys()) - set(self._non_serializable_attrs)
        else:
            keys = set(attrs_copy)
            non_serializable_keys = keys & set(self._non_serializable_attrs)
            if non_serializable_keys:
                repr_attrs = ", ".join(non_serializable_keys)
                raise ValueError(
                    f"Attempting to save non-serializable attributes: {repr_attrs}."
                )

        # Copy the attributes
        attrs_copy = {}
        for k, v in self.__dict__.items():
            if k not in keys:
                continue
            if hasattr(v, "to_dict"):
                attrs_copy[k] = v.to_dict()
            else:
                attrs_copy[k] = v

        return attrs_copy

    def generate_gml(self) -> str:
        """Convert the ``graph`` attribute to a GML formatted string.

        Returns:
            str: The GML representation of ``self.graph``.
        """
        return nx.generate_gml(self.graph)

    def to_string(self) -> tuple[str, str]:
        """Return a a GML-formatted string of the ``graph`` attribute and a JSON-formatted
        string of the other serializable attributes."""
        graph_src = self.generate_gml()
        attrs_src = json.dumps(self.get_attributes(None), indent=4)
        return graph_src, attrs_src

    def to_file(
        self,
        gml_file: str | Path,
        json_file: Optional[str | Path] = None,
        save_attrs: Optional[Iterable[str]] = None,
        compress: bool = False,
        **kwargs,
    ):
        """Save the ``CircuiTree`` object to a gml file and optionally a JSON file
        containing the object's attributes.

        This method allows saving the CircuiTree object to disk in a GML format for the
        graph and optionally a JSON format for the serializable attributes. The saved
        object can be loaded later using the ``from_file`` class method (see
        ``CircuiTree.from_file``).

        Args:
            gml_file (str | Path): The path to the GML file as a ``str`` or ``Path`` object.
            json_file (Optional[str | Path], optional): The path to the optional JSON file
                for saving other attributes. Defaults to None.
            save_attrs (Optional[Iterable[str]], optional): An optional list of attribute
                names to include in the JSON file. If None, all attributes except those in
                ``_non_serializable_attrs`` are saved. Defaults to None.
            compress (bool, optional): If True, the GML file will be compressed with gzip.
                Defaults to False.
            **kwargs: Additional keyword arguments passed to ``networkx.write_gml``.

        Returns:
            Path | Tuple[Path, Path]: Returns the path to the saved GML file and,
            optionally, the JSON file.
        """

        # Save the graph
        if compress:
            gml_target = Path(gml_file).with_suffix(".gml.gz")
        else:
            gml_target = Path(gml_file).with_suffix(".gml")
        nx.write_gml(self.graph, gml_target, **kwargs)

        # Save the other attributes
        if json_file is not None:
            attrs = self.get_attributes(save_attrs)
            json_target = Path(json_file).with_suffix(".json")
            with json_target.open("w") as f:
                json.dump(attrs, f, indent=4)
            return gml_target, json_target
        else:
            return gml_target

    def _generate_grammar(
        grammar_cls: CircuitGrammar | None, grammar_kwargs: dict
    ) -> CircuitGrammar:
        """
        (Internal) Generate a grammar object from class and kwargs.

        This function can also handle grammar class specification by looking for a
        ``__grammar_cls__`` key in ``grammar_kwargs`` and looking for the class in
        ``globals()``.
        """

        # Find the grammar class if not provided
        _grammar_cls_name = grammar_kwargs.pop("__grammar_cls__", None)
        if grammar_cls is None:
            if _grammar_cls_name is None:
                raise ValueError(
                    "Must specify grammar class as a keyword argument "
                    "(to_file(grammar_cls=...)) or in the attributes json file "
                    "(grammar = {'__grammar_cls__': ..., **grammar_kws})."
                )
            if _grammar_cls_name not in globals():
                raise ValueError(
                    f"Grammar class {_grammar_cls_name} not found in global scope. "
                    "Did you import it?"
                )
            _grammar_cls = globals()[_grammar_cls_name]
        else:
            _grammar_cls = grammar_cls

        # For backwards compatibility. Older versions had a bug where
        # _non_serializable_attrs was saved to the json file
        grammar_kwargs.pop("_non_serializable_attrs", None)

        return _grammar_cls(**grammar_kwargs)

    @classmethod
    def from_file(
        cls,
        graph_gml: str | Path | None,
        attrs_json: str | Path,
        grammar_cls: Optional[CircuitGrammar] = None,
        grammar_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Load a CircuiTree object from a JSON file/string containing the object's
        attributes and an optional GML file/string of the search graph.

        These files are typically saved with the ``to_file()`` method.

        The grammar attribute is loaded by looking for a key "grammar" in the JSON file,
        whose value should be a dict ``grammar_kwargs`` used to create a grammar object.
        The ``grammar_cls`` keyword argument can be used to directly specify the grammar class
        constructor. Alternatively, the JSON file can include a key "``__grammar_cls__``" that
        specifies the class name string (which will be looked up in ``globals()``).

        Args:
            graph_gml (str | Path | None, optional): The path to the GML file or the
                serialized GML string (optional).
            attrs_json (str | Path): The path to the JSON file containing the object's
                attributes or the serialized JSON string.
            grammar_cls (Optional[CircuitGrammar], optional): The grammar class constructor
                to use. If None, the class is loaded from the JSON file. Defaults to None.
            grammar_kwargs (Optional[dict], optional): Keyword arguments used to create the
                grammar object.
            **kwargs: Keyword arguments to pass to the CircuiTree constructor.

        Returns:
            CircuiTree: A CircuiTree object loaded from the provided files or strings.
        """

        # Load the attributes from the json file
        if Path(attrs_json).is_file():
            with open(attrs_json, "r") as f:
                kwargs.update(json.load(f))
        else:
            kwargs.update(json.loads(attrs_json))

        # Read the gml to a `nx.DiGraph` object if provided. Can be supplied as a file
        # path or a string.
        if graph_gml is None:
            graph = None
        elif Path(graph_gml).is_file():
            graph = nx.read_gml(graph_gml)
        else:
            graph = nx.parse_gml(graph_gml)

        # Make the grammar attribute
        grammar_kwargs = kwargs.pop("grammar", {}) | (grammar_kwargs or {})
        grammar = cls._generate_grammar(grammar_cls, grammar_kwargs)

        return cls(grammar=grammar, graph=graph, **kwargs)

    def to_complexity_graph(
        self, successes: bool | Iterable[Hashable] = True
    ) -> nx.DiGraph:
        """Generates a directed acyclic graph (DAG) representing the search graph as a
        "complexity atlas".

        A complexity atlas [1]_ is a subgraph of the search graph that includes only certain
        terminal states (determined by the ``successes`` argument) and their parent nodes.
        The returned graph can be used to visualize the search space and identify
        clusters of topologically similar solutions that occur, for example, due to
        motifs.

        Args:
            successes (bool | Iterable[Hashable], optional): A flag or an iterable of
                states representing successful terminal states.

                - If ``True`` (default), the ``is_success`` method is used to identify the
                  successful states, and only these are included.
                - If ``False``, all terminal states are included.
                - If an iterable, the states in the iterable are included.

        Raises:
            ValueError: If an invalid value is provided for ``successes``.
            NotImplementedError: If ``successes`` is ``True`` and ``is_success`` is not
                implemented. The intended usage is to subclass ``CircuiTree`` and implement
                the ``is_success`` method.

        Returns:
            nx.DiGraph: A directed acyclic graph representing the complexity atlas.

        References:
            .. [1] Cotterell J, Sharpe J. *An atlas of gene regulatory networks reveals
                multiple three-gene mechanisms for interpreting morphogen gradients*. Mol
                Syst Biol. 2010 Nov 2;6:425. doi: 10.1038/msb.2010.74. PMID: 21045819;
                PMCID: PMC3010108.
        """

        if isinstance(successes, Iterable):
            successful_children = set(successes)
        elif successes is True:

            # Check if is_success() is implemented
            try:
                _ = self.is_success(self.root)
            except NotImplementedError:
                raise NotImplementedError(
                    "If successes=True, the CircuiTree subclass must implement "
                    "the is_success() method."
                )

            # Keep only the successful nodes
            successful_children = set(
                c for c in self.terminal_states if self.is_success(c)
            )
        elif successes is False:
            # keep all terminal nodes
            successful_children = set(self.terminal_states)
        else:
            raise ValueError(f"Invalid value for `successes`: {successes}")

        # Store the attributes of the terminal states
        child_attrs: dict[Hashable, dict[str, Any]] = {}
        for child in successful_children:
            parents = [p for p, _ in self.graph.in_edges(child)]
            for p in parents:
                child_attrs[(p, child)] = self.graph.edges[(p, child)]

        complexity_graph: nx.DiGraph = self.graph.subgraph(
            (p for p, c in child_attrs.keys())
        ).copy()

        for (parent, child), d in child_attrs.items():
            complexity_graph.nodes[parent]["terminal_state"] = d | {"name": child}

        return complexity_graph

    def sample_terminal_states(
        self,
        n_samples: int,
        progress: bool = False,
        nprocs: int = 1,
        chunksize: int = 100,
    ) -> list[Hashable]:
        """Sample n_samples random terminal states from the grammar."""
        if nprocs == 1:
            if progress:
                from tqdm import trange

                _range = trange(n_samples, desc="Sampling all terminal circuits")
            else:
                _range = range(n_samples)
            return [self.get_random_terminal_descendant(self.root) for _ in _range]
        else:
            from multiprocessing import Pool

            if progress:
                from tqdm import tqdm

                pbar = tqdm(desc="Sampling all terminal circuits", total=n_samples)

            seed_seq = np.random.SeedSequence(self.seed)
            rgs = (
                np.random.default_rng(seed_seq.spawn(1)[0]) for _ in range(n_samples)
            )
            draw_one_sample = partial(
                self._get_random_terminal_descendant, self.grammar, self.root
            )
            samples = []
            with Pool(nprocs) as pool:
                for sample in pool.imap_unordered(
                    draw_one_sample, rgs, chunksize=chunksize
                ):
                    if progress:
                        pbar.update(1)
                    samples.append(sample)
            return samples

    @staticmethod
    def _sample_leaf(
        graph: nx.DiGraph,
        root: Hashable,
        rg: np.random.Generator,
    ):
        node = root
        children = list(graph.successors(node))
        while children:
            node = rg.choice(children)
            children = list(graph.successors(node))
        return node

    def _sample_leaves(
        self,
        n_leaves: int,
        graph: Optional[nx.DiGraph] = None,
        root: Optional[Hashable] = True,
        rg: Optional[np.random.Generator] = None,
        terminal: bool = True,
        progress: bool = False,
        max_iter: int = 10_000_000,
    ) -> list[Hashable]:
        """Sample a leaf by traversing randomly from the root. Only samples along edges
        in the supplied graph, unlike get_random_terminal_descendant().
        Note that leaves of the search graph (nodes without out-edges) are not
        necessarily terminal states. If terminal=True, rejection sampling is used to
        enforce that the samples are all terminal states."""
        graph = self.graph if graph is None else graph
        root = self.root if root is True else root
        rg = self.rg if rg is None else rg
        sample_leaf = partial(self._sample_leaf, graph, root, rg)

        if terminal:
            term = next((n for n in graph.nodes if self.grammar.is_terminal(n)), None)
            if term is None:
                raise ValueError("No terminal states in the graph.")

        if progress:
            from tqdm import tqdm

            pbar = tqdm(desc="Sampling leaves", total=n_leaves)

        samples = []
        for _ in range(max_iter):
            sample = sample_leaf()

            # If a terminal state is required and this is not terminal, reject
            if terminal and not self.grammar.is_terminal(sample):
                continue

            # else, accept the sample
            if progress:
                pbar.update(1)
            samples.append(sample)

            if len(samples) == n_leaves:
                break

        if len(samples) < n_leaves:
            raise RuntimeError(f"Maximum number of iterations reached: {max_iter}")

        return samples

    def sample_successful_circuits_by_enumeration(
        self,
        n_samples: int,
        progress: bool = False,
        nprocs: int = 1,
        chunksize: int = 100,
    ) -> list[Hashable]:
        """Sample a random successful state by first creating a new graph that contains
        all possible paths from the root to a successful terminal state. Then, sample
        paths by random traversal from the root."""

        # Check if is_success() is implemented
        try:
            _ = self.is_success(self.root)
        except NotImplementedError:
            raise NotImplementedError(
                "The CircuiTree subclass must implement the is_success() method to "
                "use this function."
            )

        ## Create a graph with all possible paths to success
        successful_terminals = set(
            s for s in self.terminal_states if self.is_success(s)
        )

        # Generate the graph with all possible paths to success
        all_paths_to_success = self.grow_tree_from_leaves(successful_terminals)

        if nprocs == 1:
            samples = self._sample_leaves(
                n_samples, graph=all_paths_to_success, terminal=True, progress=progress
            )
        else:
            if progress:
                from tqdm import tqdm

                pbar = tqdm(
                    desc="Sampling successful terminal circuits", total=n_samples
                )

            from multiprocessing import Pool

            seed_seq = np.random.SeedSequence(self.seed)
            rgs = (
                np.random.default_rng(seed_seq.spawn(1)[0]) for _ in range(n_samples)
            )
            draw_one_sample = partial(
                self._sample_leaf, all_paths_to_success, self.root
            )
            samples = []
            with Pool(nprocs) as pool:
                for state in pool.imap_unordered(
                    draw_one_sample, rgs, chunksize=chunksize
                ):
                    if progress:
                        pbar.update(1)
                    samples.append(state)

        return samples

    def sample_successful_circuits_by_rejection(
        self,
        n_samples: int,
        max_iter: int = 10_000_000,
        progress: bool = False,
        nprocs: int = 1,
        chunksize: int = 100,
    ) -> list[Hashable]:
        """Sample a random successful state with rejection sampling. Starts from the
        root state, selects random actions until termination, and accepts the sample if
        it is successful."""

        # Check if is_success() is implemented
        try:
            _ = self.is_success(self.root)
        except NotImplementedError:
            raise NotImplementedError(
                "The CircuiTree subclass must implement the is_success() method to "
                "use this function."
            )

        # Use rejection sampling to sample paths with the given pattern
        successful_terminals = set(
            s for s in self.terminal_states if self.is_success(s)
        )
        if progress:
            from tqdm import tqdm

            pbar = tqdm(desc="Sampling successful terminal circuits", total=n_samples)
        if nprocs == 1:
            samples = []
            for _ in range(max_iter):
                state = self.get_random_terminal_descendant(self.root)
                if state in successful_terminals:
                    if progress:
                        pbar.update(1)
                    samples.append(state)
                if len(samples) == n_samples:
                    break
        else:
            from multiprocessing import Pool

            prng_seeds = np.random.SeedSequence(self.seed).generate_state(n_samples)
            draw_one_sample = partial(
                self._sample_from_terminals_with_rejection,
                successful_terminals,
                self.grammar,
                self.root,
                max_iter=max_iter,
            )
            samples = []
            with Pool(nprocs) as pool:
                for state in pool.imap_unordered(
                    draw_one_sample, prng_seeds, chunksize=chunksize
                ):
                    if state in successful_terminals:
                        if progress:
                            pbar.update(1)
                        samples.append(state)
                        if len(samples) == n_samples:
                            break

        if len(samples) < n_samples:
            raise RuntimeError(f"Maximum number of iterations reached: {max_iter}")

        return samples

    def enumerate_terminal_states(
        self,
        root: Optional[Hashable] = None,
        progress: bool = False,
        max_iter: int = None,
    ) -> Iterable[Hashable]:
        """Enumerate all terminal states reachable from the given root state."""

        root = self.root if root is None else root
        max_iter = np.inf if max_iter is None else max_iter
        if progress:
            from tqdm import tqdm

            pbar = tqdm(max_iter, desc="Enumerating terminal states...")
            _callback = lambda: pbar.update(1)
        else:
            _callback = lambda: None

        # Use a post-order traversal to enumerate all terminal states
        visited = set()
        terminal_set = set()
        stack = [root]
        k = 0
        while stack and k < max_iter:
            state = stack.pop()
            visited.add(state)
            if self.grammar.is_terminal(state):
                terminal_set.add(state)
            else:
                descendants = set(
                    self._do_action(state, a) for a in self.grammar.get_actions(state)
                )
                stack.extend(descendants - visited)
            k += 1
            _callback()

        print(f"Found {len(terminal_set)} terminal states.")
        return terminal_set

    @staticmethod
    def _contingency_test(
        pattern: Hashable,
        grammar: CircuitGrammar,
        null_samples: list[Hashable],
        succ_samples: list[Hashable],
        correction: bool = True,
        barnard_ok: bool = True,
        exclude_self: bool = True,
    ):
        """Returns a contingency table with test results for the given pattern."""
        if exclude_self:
            null_samples = [s for s in null_samples if s != pattern]
            succ_samples = [s for s in succ_samples if s != pattern]
        pattern_in_null = sum(grammar.has_pattern(s, pattern) for s in null_samples)
        pattern_in_succ = sum(grammar.has_pattern(s, pattern) for s in succ_samples)
        n_null_samples = len(null_samples)
        n_succ_samples = len(succ_samples)

        # Create the contingency table. Rows represent whether or not the pattern is
        # present, columns represent whether or not the path is successful. Columns
        # sum to n_samples.
        table = np.array(
            [
                [pattern_in_succ, pattern_in_null],
                [n_succ_samples - pattern_in_succ, n_null_samples - pattern_in_null],
            ]
        )

        # Test using chi2 (or Barnard's exact test if chi2 is not appropriate)
        table_df = contingency_test(table, correction=correction, barnard_ok=barnard_ok)

        # Make a multi-index with the pattern and presence/absence of the pattern
        table_df.index = pd.MultiIndex.from_tuples(
            [(pattern, True), (pattern, False)],
            names=["pattern", "has_pattern"],
        )
        return table_df

    def test_pattern_significance(
        self,
        patterns: Iterable[Any],
        n_samples: int,
        confidence: float | None = 0.95,
        correction: bool = True,
        progress: bool = False,
        null_samples: Optional[list[Hashable]] = None,
        succ_samples: Optional[list[Hashable]] = None,
        sampling_method: Literal["rejection", "enumeration"] = "rejection",
        nprocs_sampling: int = 1,
        nprocs_testing: int = 1,
        max_iter: int = 10_000_000,
        null_kwargs: Optional[dict] = None,
        succ_kwargs: Optional[dict] = None,
        barnard_ok: bool = True,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """Test whether a pattern is successful by sampling random paths from the
        design space. Returns the contingency table (a Pandas DataFrame) containing
        test statistics and p-values.

        Samples ``n_samples`` paths from the overall design space and uses rejection
        sampling to sample ``n_samples`` paths that terminate in a successful circuit as
        determined by the ``is_success`` method.

        if ``exclude_self`` is True, the pattern being tested is excluded from the null
        and successful samples. This is to properly evaluate the significance of rare
        patterns.
        """
        if null_samples is None:
            null_kwargs = {} if null_kwargs is None else null_kwargs
            null_samples = self.sample_terminal_states(
                n_samples, progress=progress, nprocs=nprocs_sampling, **null_kwargs
            )

        if succ_samples is None:
            succ_kwargs = {} if succ_kwargs is None else succ_kwargs
            if sampling_method == "enumeration":
                succ_samples = self.sample_successful_circuits_by_enumeration(
                    n_samples, progress=progress, nprocs=nprocs_sampling, **succ_kwargs
                )
            elif sampling_method == "rejection":
                succ_samples = self.sample_successful_circuits_by_rejection(
                    n_samples,
                    max_iter=max_iter,
                    progress=progress,
                    nprocs=nprocs_sampling,
                    **succ_kwargs,
                )
            else:
                raise ValueError(
                    f"Invalid sampling method: {sampling_method}. "
                    "Must be one of ['rejection', 'enumeration']."
                )

        do_one_contingency_test = partial(
            self._contingency_test,
            grammar=self.grammar,
            null_samples=null_samples,
            succ_samples=succ_samples,
            correction=correction,
            barnard_ok=barnard_ok,
            exclude_self=exclude_self,
        )

        dfs = []
        if nprocs_testing == 1:
            iterator = patterns
            if progress:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Testing patterns")
            for pat in iterator:
                dfs.append(do_one_contingency_test(pat))

        else:
            from multiprocessing import Pool

            if progress:
                from tqdm import tqdm

                pbar = tqdm(desc="Testing patterns", total=len(patterns))

            with Pool(nprocs_testing) as pool:
                for results_df in pool.imap_unordered(
                    do_one_contingency_test, patterns
                ):
                    dfs.append(results_df)
                    if progress:
                        pbar.update(1)

        ## Concatenate the results and wrangle the data
        # Pivot the 2x2 contingency table columns into 4 separate columns
        results_df = pd.concat(dfs).reset_index()
        pivoted = results_df.pivot_table(
            index="pattern",
            columns="has_pattern",
            values=["successful_paths", "overall_paths"],
        )
        pivoted = pivoted.astype(int)
        pivoted.columns = pivoted.columns.map(
            {
                ("overall_paths", False): "others_in_null",
                ("overall_paths", True): "pattern_in_null",
                ("successful_paths", False): "others_in_succ",
                ("successful_paths", True): "pattern_in_succ",
            }
        )

        # Drop the columns that are not needed anymore
        results_df = (
            results_df.loc[results_df["has_pattern"]]
            .set_index("pattern")
            .drop(columns=["has_pattern", "successful_paths", "overall_paths"])
        )
        results_df = pd.concat([results_df, pivoted], axis=1).reset_index()

        # Perform multiple test correction (Bonferroni)
        results_df["p_corrected"] = results_df["pvalue"] * len(results_df)

        # Compute confidence intervals for the odds ratio
        abcd = results_df[
            ["pattern_in_succ", "pattern_in_null", "others_in_succ", "others_in_null"]
        ].values

        if confidence is None:
            results_df["odds_ratio"] = compute_odds_ratios(abcd, progress=progress)
        else:
            odds_ratios, cis_low, cis_high = compute_odds_ratios_with_ci(
                abcd, confidence_level=confidence, progress=progress
            )
            ci_level = f"{int(confidence * 100)}%"
            results_df["odds_ratio"] = odds_ratios
            results_df[f"ci_{ci_level}_low"] = cis_low
            results_df[f"ci_{ci_level}_high"] = cis_high

        results_df = results_df.sort_values("odds_ratio", ascending=False)
        return results_df

    def grow_tree_from_leaves(self, leaves: Iterable[Hashable]) -> nx.DiGraph:
        """Returns the tree (or DAG) of all paths that start at the root and ending at
        a node in ``leaves``."""
        # Maintain a stack of leaves or (state, undo_action) pairs to add to the tree
        # Grow the tree in reverse, from the leaves to the root, by undoing actions
        graph = nx.DiGraph()
        graph.add_nodes_from(leaves)
        stack = list(leaves)
        while stack:
            item = stack.pop()
            if not isinstance(item, tuple):
                leaf = item
                graph.add_node(leaf, **self.default_attrs)
                stack.extend([(leaf, a) for a in self.grammar.get_undo_actions(leaf)])
            else:
                state, undo_action = item
                parent = self._undo_action(state, undo_action)
                if (parent, state) in graph.edges:
                    continue
                if parent not in graph:
                    graph.add_node(parent, **self.default_attrs)
                    stack.extend(
                        [(parent, a) for a in self.grammar.get_undo_actions(parent)]
                    )
                graph.add_edge(parent, state, **self.default_attrs)

        return graph


def compute_odds_ratio_and_ci(
    table: np.ndarray, confidence_level: float
) -> tuple[float, tuple[float, float]]:
    """Compute the odds ratio and confidence interval for a 2x2 contingency table."""
    # Compute the odds ratio
    (a, b), (c, d) = table
    bc = b * c
    if bc == 0:
        odds_ratio = np.inf
    else:
        odds_ratio = (a * d) / bc

    # Compute the confidence interval
    if any(table.flatten() == 0):
        return odds_ratio, (np.nan, np.nan)
    else:
        upper_quantile = (1 + confidence_level) / 2
        log_odds_ratio = np.log(odds_ratio)
        std_err_log_OR = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        log_ci_width = stats.norm.ppf(upper_quantile) * std_err_log_OR
        ci_low = np.exp(log_odds_ratio - log_ci_width)
        ci_high = np.exp(log_odds_ratio + log_ci_width)
        return odds_ratio, (ci_low, ci_high)


def compute_odds_ratios(abcd: np.ndarray, progress: bool = False) -> np.ndarray:
    """Compute the odds ratio for multiple 2x2 contingency tables. Takes a 2D array of
    shape (n, 4) where n is the number of contingency tables.
    Each row [a, b, c, d] in the array corresponds to the 2x2 contingnecy table:

            [[a, b],
             [c, d]]

    """
    tables = abcd.reshape(-1, 2, 2)
    odds_ratios = np.zeros(len(tables))
    iterator = tables
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Computing odds ratios")
    for i, table in enumerate(iterator):
        (a, b), (c, d) = table
        bc = b * c
        if bc == 0:
            odds_ratios[i] = np.inf
        else:
            odds_ratios[i] = (a * d) / (b * c)
    return odds_ratios


def compute_odds_ratios_with_ci(
    abcd: np.ndarray, confidence_level: float, progress: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the odds ratio and confidence intervals for multiple 2x2 contingency
    tables. Takes a 2D array of shape (n, 4) where n is the number of contingency
    tables.
    Each row [a, b, c, d] in the array corresponds to the 2x2 table:

            [[a, b],
             [c, d]]

    """
    tables = abcd.reshape(-1, 2, 2)
    odds_ratios = np.zeros(len(tables))
    cis = np.zeros((len(tables), 2))
    iterator = tables
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="Computing odds ratios +/- CI")
    for i, table in enumerate(iterator):
        odds_ratios[i], cis[i] = compute_odds_ratio_and_ci(table, confidence_level)
    return odds_ratios, *cis.T


def contingency_test(
    table: np.ndarray, correction: bool = True, barnard_ok: bool = True
) -> pd.DataFrame:
    """Perform a two-tailed test for P(has_pattern | successful) != P(has_pattern)"""

    table_df = pd.DataFrame(
        data=table,
        index=["has_pattern", "lacks_pattern"],
        columns=["successful_paths", "overall_paths"],
    )
    table_df.index.name = "pattern_present"

    minval = table.min()
    if minval == 0 or (minval < 5 and not barnard_ok):
        table_df["test"] = pd.NA
        table_df["statistic"] = np.nan
        table_df["pvalue"] = np.nan
    elif minval < 5:
        try:
            res = stats.barnard_exact(table, alternative="two-sided")
            table_df["test"] = "barnard"
            table_df["statistic"] = res.statistic
            table_df["pvalue"] = res.pvalue
        except MemoryError:
            table_df["test"] = pd.NA
            table_df["statistic"] = np.nan
            table_df["pvalue"] = np.nan
    else:
        res = stats.chi2_contingency(table, correction=correction)
        table_df["test"] = "chi2"
        table_df["statistic"] = res.statistic
        table_df["pvalue"] = res.pvalue

    return table_df
