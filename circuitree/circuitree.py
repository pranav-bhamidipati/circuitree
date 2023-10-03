from abc import ABC, abstractmethod
from itertools import cycle, chain, islice, repeat
import json
from pathlib import Path
from typing import Callable, Literal, Optional, Iterable, Any
import numpy as np
import networkx as nx

from .modularity import tree_modularity, tree_modularity_estimate


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
        root: str,
        exploration_constant: Optional[float] = None,
        variance_constant: float = 0.0,
        seed: int = 2023,
        rg: Optional[np.random.Generator] = None,
        graph: Optional[nx.DiGraph] = None,
        tree_shape: Literal["tree", "dag"] = "dag",
        score_func: Optional[Callable] = None,
        **kwargs,
    ):
        if exploration_constant is None:
            self.exploration_constant = np.sqrt(2)
        else:
            self.exploration_constant = exploration_constant

        self.variance_constant = variance_constant

        if rg is None:
            if seed is None:
                raise ValueError("Must specify seed if rg is not specified")
            else:
                rg = np.random.default_rng(seed)

        self.rg = rg
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

        if score_func is None:
            self._score_func = ucb_score
        else:
            self._score_func = score_func

        self._non_serializable_attrs = [
            "_non_serializable_attrs",
            "rg",
            "graph",
            "_score_func",
        ]

    def _do_action(self, state: Any, action: Any):
        new_state = self.do_action(state, action)
        if self.tree_shape == "dag":
            new_state = self.get_unique_state(new_state)
        return new_state

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
    def get_reward(self, state) -> float | int:
        raise NotImplementedError

    @abstractmethod
    def get_unique_state(self, state: Any) -> Any:
        raise NotImplementedError

    @property
    def default_attrs(self):
        return dict(visits=0, reward=0)

    def get_state_to_simulate(self, start: Any) -> Any:
        """Uses the random generator for the given thread to select a state to simulate,
        starting from the given starting state. If the given starting state is terminal,
        returns it. Otherwise, selects a random child recursively until a terminal state
        is reached."""
        state = start
        while not self.is_terminal(state):
            actions = self.get_actions(state)
            action = self.rg.choice(actions)
            state = self._do_action(state, action)
        return state

    def select_and_expand(self) -> list[Any]:
        # Start at root
        node = self.root
        selection_path = [node]

        # Shuffle actions to prevent ordering bias
        actions = self.get_actions(node)
        self.rg.shuffle(actions)

        # Recursively select the child with the highest UCB score until either a terminal
        # state is reached or an unexpanded edge is found.
        while actions:
            max_ucb = -np.inf
            best_child = None
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

            # If no child can be expanded (all children have been visited at least once)
            # then move on to the child with the highest UCB score and repeat.
            node = best_child
            selection_path.append(node)

            # If the node is terminal, actions will be empty and the loop will break
            actions = self.get_actions(node)
            self.rg.shuffle(actions)

        # If the loop breaks, we have reached a terminal state.
        return selection_path

    def expand_edge(self, parent: Any, child: Any):
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
        sim_node = self.get_state_to_simulate(selection_path[-1])

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

    @property
    def terminal_states(self):
        return (node for node in self.graph.nodes if self.is_terminal(node))

    def search_mcts(
        self,
        n_steps: int,
        callback_every: int = 1,
        callback: Optional[Callable] = None,
        exploration_constant: Optional[float] = None,
        progress_bar: bool = False,
        run_kwargs: Optional[dict] = None,
    ) -> None | list[Any]:
        if exploration_constant is None:
            exploration_constant = self.exploration_constant

        run_kwargs = {} if run_kwargs is None else run_kwargs

        if progress_bar:
            from tqdm import trange

            iterator = trange(n_steps, desc="MCTS search")
        else:
            iterator = range(n_steps)

        cb_results = []
        if callback is None:
            return_results = False
            callback = lambda *a, **kw: None
            callback_every = np.inf
        else:
            return_results = True
            cb_results.append(callback(self.graph, [], None, None))
        for i in iterator:
            selection_path, reward, sim_node = self.traverse(**run_kwargs)
            if i % callback_every == 0:
                cb_result = callback(self.graph, selection_path, sim_node, reward)
                cb_results.append(cb_result)

        if return_results:
            return cb_results
        else:
            return None

    def grow_tree(
        self, root=None, n_visits: int = 0, print_updates=False, print_every=1000
    ):
        if root is None:
            root = self.root
            if print_updates:
                print(f"Adding root: {root}")
            self.graph.add_node(root, visits=n_visits, reward=0)

        stack = [(root, action) for action in self.get_actions(root)]
        n_added = 1
        while stack:
            node, action = stack.pop()
            if not self.is_terminal(node):
                next_node = self._do_action(node, action)
                if next_node not in self.graph.nodes:
                    n_added += 1
                    self.graph.add_node(next_node, visits=n_visits, reward=0)
                    stack.extend([(next_node, a) for a in self.get_actions(next_node)])
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
        return filter(self.is_terminal, chain.from_iterable(layers))

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
        if n_steps is None and n_repeats is None and n_cycles is None:
            raise ValueError(
                "Must specify at least one of n_steps, n_repeats, or n_cycles."
            )

        ### Iterate in BFS order
        # If n_repeats is specified, repeat each node n_repeats times before moving on
        # If n_cycles is specified, repeat the entire BFS traversal n_cycles times
        # If n_steps is specified, stop after n_steps total iterations
        if n_repeats is not None:
            iterator = chain.from_iterable(repeat(elem, n_repeats) for elem in iterator)
        if n_cycles is not None:
            iterator = chain.from_iterable(repeat(iterator, n_cycles))
        iterator = islice(iterator, n_steps)

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

    def is_success(self, state) -> bool:
        """Must be defined to calculate modularity."""
        raise NotImplementedError

    @property
    def modularity(self) -> float:
        return tree_modularity(self.graph, self.root, self.is_terminal, self.is_success)

    @property
    def modularity_estimate(self) -> float:
        return tree_modularity_estimate(self.graph, self.root)

    @staticmethod
    def complexity_graph_from_mcts(
        search_graph: nx.DiGraph,
        selection_path: Optional[Iterable[str]] = None,
        sim_node: Optional[str] = None,
        reward: Optional[int | float] = None,
    ) -> nx.DiGraph:
        graph = search_graph.copy()
        nodes_to_remove = []
        for n in graph.nodes:
            if n[0] == "*":
                p = n[1:]
                nodes_to_remove.append(n)
                data = graph.edges[p, n]
                graph.nodes[p]["visits"] = data.get("visits", 0)
                graph.nodes[p]["reward"] = data.get("reward", 0)
                # graph.nodes[p]["history"] = data.get("history", ())
        graph.remove_nodes_from(nodes_to_remove)
        empty_nodes = [n for n, v in graph.nodes("visits") if v is None]
        graph.remove_nodes_from(empty_nodes)

        return graph, selection_path, sim_node, reward

    @staticmethod
    def complexity_graph_from_bfs(
        search_graph: nx.DiGraph,
        node: Optional[Iterable[str]] = None,
        reward: Optional[float | int] = None,
    ) -> nx.DiGraph:
        graph: nx.DiGraph = search_graph.copy()
        nodes_to_remove = []
        for n, d in graph.nodes(data=True):
            if n[0] == "*":
                nodes_to_remove.append(n)
                if d.get("visits", 0) == 0:
                    nodes_to_remove.append(n[1:])
                else:
                    graph.nodes[n[1:]].update(d)
            else:
                if d.get("visits") is None:
                    nodes_to_remove.append(n)

        graph.remove_nodes_from(nodes_to_remove)

        for n1, n2 in graph.edges:
            graph.edges[n1, n2].update(graph.nodes[n2])

        return graph, node, reward

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


def accumulate_visits_and_rewards(
    graph: nx.DiGraph, visits_attr: Any = "visits", reward_attr: Any = "reward"
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
