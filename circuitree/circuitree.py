from abc import ABC, abstractmethod
from functools import partial
from itertools import cycle, chain, islice, repeat
import json
from pathlib import Path
from typing import Callable, Hashable, Literal, Optional, Iterable, Any
import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
import warnings

from .modularity import tree_modularity, tree_modularity_estimate
from .grammar import CircuitGrammar

__all__ = ["CircuiTree"]


def ucb_score(
    graph: nx.DiGraph,
    parent,
    node,
    exploration_constant: Optional[float] = np.sqrt(2),
    **kw,
):
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


class CircuiTree(ABC):
    def __init__(
        self,
        grammar: CircuitGrammar,
        root: str,
        exploration_constant: Optional[float] = None,
        seed: int = 2023,
        graph: Optional[nx.DiGraph] = None,
        tree_shape: Optional[Literal["tree", "dag"]] = None,
        compute_unique: bool = True,
        **kwargs,
    ):
        # Initialize RNG
        self.rg = np.random.default_rng(seed)
        self.seed = self.rg.bit_generator._seed_seq.entropy

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

        self._non_serializable_attrs = [
            "_non_serializable_attrs",
            "rg",
            "graph",
        ]

    @abstractmethod
    def get_reward(self, state) -> float | int:
        raise NotImplementedError

    @property
    def default_attrs(self):
        return dict(visits=0, reward=0)

    @property
    def terminal_states(self):
        return (node for node in self.graph.nodes if self.grammar.is_terminal(node))

    def _do_action(self, state: Hashable, action: Hashable):
        new_state = self.grammar.do_action(state, action)
        if self.compute_unique:
            new_state = self.grammar.get_unique_state(new_state)
        return new_state

    def _undo_action(self, state: Hashable, action: Hashable) -> Hashable:
        """Undo one action from the given state."""
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
        """Starting from the state `start`, select random actions until termination."""
        state = start
        while not grammar.is_terminal(state):
            actions = grammar.get_actions(state)
            action = rg.choice(actions)
            state = grammar.do_action(state, action)
            state = grammar.get_unique_state(state)
        return state

    @staticmethod
    def _sample_from_terminals_with_rejection(
        terminals: set[Any],
        grammar: CircuitGrammar,
        start: Hashable,
        seed: int,
        max_iter: Optional[int] = None,
    ):
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
        """Starting from the state `start`, select random actions until termination."""
        rg = self.rg if rg is None else rg
        return self._get_random_terminal_descendant(self.grammar, start, rg)

    def select_and_expand(
        self, rg: Optional[np.random.Generator] = None
    ) -> list[Hashable]:
        rg = self.rg if rg is None else rg

        # Start at root
        node = self.root
        selection_path = [node]
        actions = self.grammar.get_actions(node)

        # Select the child with the highest UCB score until you reach a terminal
        # state or an unexpanded edge
        while actions:
            max_ucb = -np.inf
            best_child = None
            rg.shuffle(actions)
            for action in actions:
                child = self._do_action(node, action)
                ucb = self.get_ucb_score(node, child)

                # An unexpanded edge has UCB score of infinity.
                # In this case, expand and select the child.
                if ucb == np.inf:
                    self.expand_edge(node, child)
                    selection_path.append(child)
                    return selection_path

                # Otherwise, track the child with the highest UCB score
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_child = child

            node = best_child
            selection_path.append(node)
            actions = self.grammar.get_actions(node)

        # If the loop breaks, we have reached a terminal state.
        return selection_path

    def expand_edge(self, parent: Hashable, child: Hashable):
        if not self.graph.has_node(child):
            self.graph.add_node(child, **self.default_attrs)
        self.graph.add_edge(parent, child, **self.default_attrs)

    def get_ucb_score(self, parent, child):
        if self.graph.has_edge(parent, child):
            return ucb_score(self.graph, parent, child, self.exploration_constant)
        else:
            return np.inf

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
        """Update the visit count for each node and edge in the selection path.
        Visit update happens before simulation and reward calculation, so until the
        reward is computed and backpropagated, there is 'virtual loss' on each node in
        the selection path."""
        self._backpropagate(selection_path, "visits", 1)

    def backpropagate_reward(self, selection_path: list, reward: float | int):
        """Update the reward for each node and edge in the selection path.
        Visit update happens before simulation and reward calculation, so until the
        reward is computed and backpropagated, there is 'virtual loss' on each node in
        the selection path."""
        self._backpropagate(selection_path, "reward", reward)

    def traverse(self, **kwargs):
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
        self, root=None, n_visits: int = 0, print_updates=False, print_every=1000
    ):
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

    def bfs_iterator(self, root=None, shuffle=False):
        root = self.root if root is None else root
        layers = (l for l in nx.bfs_layers(self.graph, root))

        if shuffle:
            layers = list(layers)
            for l in layers:
                self.rg.shuffle(l)

        # Iterate over all terminal nodes in BFS order
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
        callback_every: int = 1,
        callback: Optional[Callable] = None,
        progress_bar: bool = False,
        run_kwargs: Optional[dict] = None,
        callback_before_start: bool = True,
    ) -> None:

        # Optionally set up a progress bar
        if progress_bar:
            from tqdm import trange

            iterator = trange(n_steps, desc="MCTS search")
        else:
            iterator = range(n_steps)

        # Optionally run the callback before starting the search
        if callback is not None and callback_before_start:
            callback(self, -1, [None], None, None)

        # Run the search
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
        """Run the MCTS search algorithm on the given CircuiTree object. Can be used to
        run the search in parallel threads or processes."""
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
        """Run the MCTS search algorithm on the given CircuiTree object, calling a
        callback function every `callback_every` iterations. Can be used to run the
        search in parallel threads or processes."""
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

    def is_success(self, state: Hashable) -> bool:
        """Returns whether or not a state is successful. Used to infer which patterns
        lead to more successes (i.e. motif candidates)."""
        raise NotImplementedError

    def copy_graph(self) -> nx.DiGraph:
        """Return a shallow copy of the graph. Use copy.deepcopy() for a deep copy."""
        return self.graph.copy()

    def get_attributes(self, attrs_copy: Optional[Iterable[str]]) -> dict:
        """Return a dictionary of the object's attributes. If `attrs_copy` is not
        provided, all attributes are returned except those in the
        `_non_serializable_attrs` list. If `attrs_copy` is provided, only the specified
        attributes are returned. If any of the specified attributes are in
        `_non_serializable_attrs`, a ValueError is raised.

        If an attribute has a `to_dict()` method, it is called to get the attribute's
        value. Otherwise, the attribute is copied as-is.
        """

        # Get the attributes to copy
        if attrs_copy is None:
            keys = set(self.__dict__.keys()) - set(self._non_serializable_attrs)
        else:
            keys = set(attrs_copy)
            if non_serializable := (keys & set(self._non_serializable_attrs)):
                repr_non_ser = ", ".join(non_serializable)
                raise ValueError(
                    f"Attempting to save non-serializable attributes: {repr_non_ser}."
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
        """Return a string representation of the graph in GML format."""
        return nx.generate_gml(self.graph)

    def to_string(self) -> tuple[str, str]:
        """Return a a GML-formatted string of the `graph` attribute and a JSON-formatted
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
        """Save the CircuiTree object to a gml file and optionally a json file
        containing the object's attributes. The `save_attrs` argument can be used to
        specify which attributes to save. If `compress` is True, the gml file is
        compressed with gzip.

        A saved CircuiTree object can be loaded with the `from_file` class method.

        The grammar is saved by calling its `to_dict()` method, which returns a
        dictionary of the grammar's attributes and its class name string
        `__grammar_cls__`. These attributes are saved in the JSON file and used to
        create the grammar object upon loading."""

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

    @staticmethod
    def generate_grammar(grammar_cls: CircuitGrammar, grammar_kwargs: dict):
        """Generate a grammar object from the class and keyword arguments.

        The `CircuitGrammar` class used to construct the object can be passed using the
        grammar_cls keyword grammar object. Alternatively, if `grammar_kwargs` contains
        a key "__grammar_cls__" that specifies a class name string, that class will be
        searched for in `globals()`."""

        # Make the grammar object
        # Get kwargs from the grammar_kwargs in this function and/or from the json
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

        # Patch an external bug where _non_serializable_attrs is saved to the json file
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
        """Load a CircuiTree object from a JSON file containing the object's attributes
        and an optional GML file of the search graph, typically saved with the `.to_file()`
        method. Both can be supplied as serialized strings as well.

        The grammar attribute is loaded by looking for a key "grammar" in the JSON file,
        whose value should be a dict `grammar_kwargs` used to create a grammar object.
        The grammar_cls keyword can be passed to specify the class constructor for the
        grammar object. Alternatively, if `grammar_kwargs` contains a key
        "__grammar_cls__" that specifies a class name string, that class will be found
        in globals() and used to construct the grammar object.
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
        grammar = cls.generate_grammar(grammar_cls, grammar_kwargs)

        return cls(grammar=grammar, graph=graph, **kwargs)

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

        Samples `n_samples` paths from the overall design space and uses rejection
        sampling to sample `n_samples` paths that terminate in a successful circuit as
        determined by the is_successful() method.

        if `exclude_self` is True, the pattern being tested is excluded from the null
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

    def to_complexity_graph(
        self, successes: bool | Iterable[Hashable] = True
    ) -> nx.DiGraph:
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
