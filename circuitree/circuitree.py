from abc import ABC, abstractmethod
from itertools import cycle, chain, islice, repeat
import json
from pathlib import Path
from typing import Callable, Hashable, Literal, Optional, Iterable, Any
import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats

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
    parent_in_edges = graph.in_edges(parent, data="visits")
    if parent_in_edges:
        parent_visits = sum(v for _, _, v in parent_in_edges)
    else:
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
        tree_shape: Literal["tree", "dag"] = "dag",
        **kwargs,
    ):
        self.rg = np.random.default_rng(seed)
        self.seed = self.rg.bit_generator._seed_seq.entropy

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

        if tree_shape not in ("tree", "dag"):
            raise ValueError("Argument `tree_shape` must be `tree` or `dag`.")
        self.tree_shape = tree_shape

        self.grammar = grammar

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
        if self.tree_shape == "dag":
            new_state = self.grammar.get_unique_state(new_state)
        return new_state

    def _undo_action(self, state: Hashable, action: Hashable) -> Hashable:
        """Undo one action from the given state."""
        if state == self.root:
            return None
        new_state = self.grammar.undo_action(state, action)
        if self.tree_shape == "dag":
            new_state = self.grammar.get_unique_state(new_state)
        return new_state

    def get_random_terminal_descendant(
        self, start: Hashable, rg: Optional[np.random.Generator] = None
    ) -> Hashable:
        """Starting from the state `start`, select cs random actions until termination."""
        rg = self.rg if rg is None else rg
        state = start
        while not self.grammar.is_terminal(state):
            actions = self.grammar.get_actions(state)
            action = rg.choice(actions)
            state = self._do_action(state, action)
        return state

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

    def accumulate_visits_and_rewards(self, graph: Optional[nx.DiGraph] = None):
        _accumulated = self.graph if graph is None else graph
        accumulate_visits_and_rewards(_accumulated)
        if graph is None:
            return _accumulated

    def search_mcts(
        self,
        n_steps: int,
        callback_every: int = 1,
        callback: Optional[Callable] = None,
        exploration_constant: Optional[float] = None,
        progress_bar: bool = False,
        run_kwargs: Optional[dict] = None,
    ) -> None:
        if exploration_constant is None:
            exploration_constant = self.exploration_constant

        run_kwargs = {} if run_kwargs is None else run_kwargs

        if progress_bar:
            from tqdm import trange

            iterator = trange(n_steps, desc="MCTS search")
        else:
            iterator = range(n_steps)

        if callback is None:
            callback_every = np.inf
        for i in iterator:
            selection_path, reward, sim_node = self.traverse(**run_kwargs)
            if i % callback_every == 0:
                _ = callback(self.graph, selection_path, sim_node, reward)

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
        **kwargs,
    ):
        if self.graph.number_of_nodes() < 2:
            self.graph.add_node(self.root, visits=0, reward=0)
            self.grow_tree(root=self.root, n_visits=0)

        if callback is None:
            callback = lambda *a, **kw: None

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

        cb_results = []
        if callback is None:
            return_results = False
            callback = lambda *a, **kw: None
            callback_every = np.inf
        else:
            return_results = True
            cb_results.append(callback(self.graph, None, None))

        for i, node in enumerate(iterator):
            self.graph.nodes[node]["visits"] += 1
            reward = self.get_reward(node)
            self.graph.nodes[node]["reward"] += reward

            if i % callback_every == 0:
                cb_result = callback(self.graph, node, reward)
                cb_results.append(cb_result)

        if return_results:
            return cb_results
        else:
            return None

    def is_success(self, state: Hashable) -> bool:
        """Returns whether or not a state is successful. Used to infer which patterns
        lead to more successes (i.e. motif candidates)."""
        raise NotImplementedError

    def to_file(
        self,
        gml_file: str | Path,
        json_file: Optional[str | Path] = None,
        save_attrs: Optional[Iterable[str]] = None,
        compress: bool = False,
        **kwargs,
    ):
        if compress:
            gml_target = Path(gml_file).with_suffix(".gml.gz")
        else:
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

            attrs = {}
            for k, v in self.__dict__.items():
                if k not in keys:
                    continue
                if hasattr(v, "to_dict"):
                    attrs[k] = v.to_dict()
                else:
                    attrs[k] = v

            json_target = Path(json_file).with_suffix(".json")
            with json_target.open("w") as f:
                json.dump(attrs, f, indent=4)

            return gml_target, json_target

        else:
            return gml_target

    @classmethod
    def from_file(
        cls,
        graph_gml: str | Path,
        attrs_json: str | Path,
        grammar_cls: Optional[CircuitGrammar] = None,
        **kwargs,
    ):
        """Load a CircuiTree from a gml file and a JSON file containing the object's
        attributes.

        The grammar attribute is loaded by looking for a key "grammar" in the JSON file,
        whose value should be a dict `grammar_kwargs` used to create a grammar object.
        The grammar_cls keyword can be passed to specify the class constructor for the
        grammar object. Alternatively, if `grammar_kwargs` contains a key
        "__grammar_cls__" that specifies a class name string, that class will be found
        in globals() and used to construct the grammar object.

        When dumping a CircuiTree to a JSON file, the grammar object is dumped using
        the method `to_dict()`, which automatically creates the "__grammar_cls__"
        key-value entry.
        """
        # Load the attributes from the json file
        with open(attrs_json, "r") as f:
            kwargs.update(json.load(f))

        # Load the keywords for the grammar __init__() method from the json file
        grammar_kwargs: dict[str, Any] = kwargs.pop("grammar", {})
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

        grammar = _grammar_cls(**kwargs.pop("grammar", {}))
        graph = nx.read_gml(graph_gml)

        return cls(grammar=grammar, graph=graph, **kwargs)

    def sample_terminal_states(self, n_samples: int) -> list[Hashable]:
        """Sample n_samples random terminal states."""
        return [
            self.get_random_terminal_descendant(self.root) for _ in range(n_samples)
        ]

    def sample_successful_circuits(
        self, n_samples: int, max_iter: int = 10_000_000
    ) -> list[Hashable]:
        """Sample a random successful state with rejection sampling. Starts from the
        root state, selects random actions until termination, and accepts the sample if
        it is successful."""

        # Use rejection sampling to sample paths with the given pattern
        samples = []
        for _ in range(max_iter):
            state = self.get_random_terminal_descendant(self.root)
            if self.is_success(state):
                samples.append(state)
            if len(samples) == n_samples:
                break

        if len(samples) < n_samples:
            raise RuntimeError(f"Maximum number of iterations reached: {max_iter}")

        return samples

    def test_contingency(
        table: np.ndarray,
        test: Literal["chi2", "barnard", "auto"] = "auto",
        correction: bool = True,
    ) -> stats._hypotests.BarnardExactResult | stats.contingency.Chi2ContingencyResult:
        """Perform a two-tailed test for P(has_pattern | successful) != P(has_pattern)"""
        if test == "auto":
            if table.min() < 5:
                print(
                    f"Contingency table has one or more entries < 5. "
                    "Using Barnard's exact test."
                )
                test = "barnard"
            else:
                print(
                    f"Contingency table has all entries >= 5. Using chi-squared test."
                )
                test = "chi2"
        elif test not in ("chi2", "barnard"):
            raise ValueError("Argument `test` must be 'chi2', 'barnard', or 'auto'.")

        if test == "barnard":
            res = stats.barnard_exact(table, alternative="two-sided")
        else:
            res = stats.chi2_contingency(table, correction=correction)
        return res

    def test_pattern_success_by_sampling(
        self,
        pattern: Any,
        n_samples: int,
        max_iter: int = 10_000_000,
        test: Literal["chi2", "barnard", "auto"] = "auto",
        correction: bool = True,
    ) -> tuple[pd.DataFrame]:
        """Test whether a pattern is successful by sampling random paths from the
        design space. Returns the contingency table (Pandas DataFrame) and the p-value
        for significance.

        Samples `n_samples` paths from the overall design space and uses rejection
        sampling to sample `n_samples` paths that terminate in a successful circuit as
        determined by the is_successful() method."""
        null_samples = self.sample_terminal_states(n_samples)
        success_samples = self.sample_successful_circuits(
            n_samples, max_iter=max_iter, path=False
        )

        pattern_cache: dict[Hashable, bool] = {}

        def _has_pattern(state):
            """Check if the given state contains the pattern. Caches results."""
            has_pattern = pattern_cache.get(state)
            if has_pattern is None:
                has_pattern = self.grammar.has_pattern(state, pattern)
                pattern_cache[state] = has_pattern
            return has_pattern

        pattern_in_null = sum(_has_pattern(s) for s in null_samples)
        pattern_in_successes = sum(_has_pattern(s) for s in success_samples)

        # Create the contingency table. Rows represent whether or not the pattern is
        # present, columns represent whether or not the path is successful. Columns
        # sum to n_samples.
        table = np.array(
            [
                [pattern_in_successes, pattern_in_null],
                [n_samples - pattern_in_successes, n_samples - pattern_in_null],
            ]
        )
        test_result = self.test_contingency(table, test=test, correction=correction)
        table_df = pd.DataFrame(
            data=table,
            index=["has_pattern", "lacks_pattern"],
            columns=["successful_paths", "overall_paths"],
        )

        return test_result, table_df

    def grow_tree_from_leaves(self, leaves: Iterable[Hashable]) -> nx.DiGraph:
        """Returns the tree (or DAG) of all paths that start at the root and ending at
        a node in ``leaves``."""

        # Maintain a stack of (state, undo_action) pairs to add to the tree
        stack = []
        for leaf in leaves:
            stack.extend([(leaf, a) for a in self.grammar.get_undo_actions(leaf)])

        # Build the tree by undoing actions from each leaf
        tree = nx.DiGraph()
        while stack:
            state, undo_action = stack.pop()
            if state == self.root:
                continue
            parent = self._undo_action(state, undo_action)
            if parent not in tree:
                tree.add_node(parent, **self.default_attrs)
            tree.add_edge(parent, state, **self.default_attrs)
            stack.extend([(parent, a) for a in self.grammar.get_undo_actions(parent)])

        return tree

    def to_complexity_graph(self, successes: bool = False) -> nx.DiGraph:
        # Keep only the successful nodes
        if successes:
            parent_to_child: dict[Hashable, tuple[Hashable, float, int]] = {}
            for child in self.graph.nodes:
                if self.grammar.is_terminal(child) and self.is_success(child):
                    parent = list(self.graph.in_edges(child))[0][0]
                    parent_to_child[parent] = (
                        child,
                        self.graph.nodes[child]["reward"],
                        self.graph.nodes[child]["visits"],
                    )
            complexity_graph: nx.DiGraph = self.graph.subgraph(
                parent_to_child.keys()
            ).copy()
            for parent, (child, reward, visits) in parent_to_child.items():
                complexity_graph.nodes[parent]["terminal_state"] = dict(
                    name=child, reward=reward, visits=visits
                )

        # Keep everything
        else:
            complexity_graph = self.graph.copy()
            nodes_to_remove = []
            for child in self.graph.nodes:
                if self.grammar.is_terminal(child):
                    parent = list(self.graph.in_edges(child))[0][0]
                    complexity_graph.nodes[parent]["terminal_state"] = dict(
                        name=child,
                        reward=self.graph.nodes[child]["reward"],
                        visits=self.graph.nodes[child]["visits"],
                    )
                    nodes_to_remove.append(child)
            complexity_graph.remove_nodes_from(nodes_to_remove)

        return complexity_graph

    # def sample_random_path(self) -> list[Hashable]:
    #     """Sample a random path from the root to a terminal state."""
    #     path = []
    #     state = self.root
    #     while not self.grammar.is_terminal(state):
    #         actions = self.grammar.get_actions(state)
    #         action = self.rg.choice(actions)
    #         path.append(action)
    #         state = self._do_action(state, action)
    #     return path

    # def sample_circuits_with_pattern(
    #     self,
    #     pattern: Hashable,
    #     n_samples: int,
    #     max_iter: int = 10_000_000,
    #     return_terminal: bool = True,
    # ) -> list[Hashable]:
    #     """Sample a random path from the root to a terminal state that contains the
    #     given pattern. Uses rejection sampling."""

    #     pattern_cache: dict[Hashable, bool] = {}

    #     def _path_has_pattern(path):
    #         """Check if the given path contains the given pattern. Use cached results to
    #         speed up search."""
    #         for i, node in enumerate(path):
    #             has_pattern = pattern_cache.get(node)
    #             if has_pattern is None:
    #                 has_pattern = self.grammar.has_pattern(node, pattern)
    #                 pattern_cache[node] = has_pattern
    #             if has_pattern:
    #                 pattern_cache.update({n: True for n in path[i + 1 :]})
    #                 return True
    #         return False

    #     samples = []
    #     if return_terminal:
    #         _append_sample = lambda path: samples.append(path[-1])
    #     else:
    #         _append_sample = lambda path: samples.append(path)

    #     # Use rejection sampling to sample paths with the given pattern
    #     for _ in range(max_iter):
    #         path = self.sample_random_path()
    #         if _path_has_pattern(path):
    #             _append_sample(path)
    #         if len(samples) == n_samples:
    #             break

    #     if len(samples) < n_samples:
    #         raise RuntimeError(f"Maximum number of iterations reached: {max_iter}")

    #     return samples

    @property
    def modularity(self) -> float:
        return tree_modularity(
            self.graph, self.root, self.grammar.is_terminal, self.is_success
        )

    @property
    def modularity_estimate(self) -> float:
        return tree_modularity_estimate(self.graph, self.root)


def accumulate_visits_and_rewards(
    graph: nx.DiGraph, visits_attr: str = "visits", reward_attr: str = "reward"
):
    """Accumulate results on nodes post-hoc"""
    for n in graph.nodes:
        total_visits = sum([v for _, _, v in graph.in_edges(n, data=visits_attr)])
        total_reward = sum([r for _, _, r in graph.in_edges(n, data=reward_attr)])
        graph.nodes[n]["visits"] = total_visits
        graph.nodes[n]["reward"] = total_reward

    for n in graph.nodes:
        graph.nodes[n][visits_attr] = 0
        graph.nodes[n][reward_attr] = 0

    for parent, child, data in graph.edges(data=True):
        graph.nodes[parent][visits_attr] += data[visits_attr]
        graph.nodes[parent][reward_attr] += data[reward_attr]
