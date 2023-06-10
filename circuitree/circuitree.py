from abc import ABC, abstractmethod
from itertools import cycle, chain, repeat
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

        if graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = graph

        self.root = root
        self.graph.add_node(self.root, visits=0, reward=0)

        if tree_shape not in ("tree", "dag"):
            raise ValueError("Argument `tree_shape` must be `tree` or `dag`.")
        self.tree_shape = tree_shape

        if score_func is None:
            self._score_func = ucb_score
        else:
            self._score_func = score_func

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

    def select(self, node, **kwargs):
        selection_path = [node]
        while not self.is_leaf(node):
            best_child = self.best_child(node, **kwargs)
            node = best_child
            selection_path.append(node)
        return node, selection_path

    def expand(self, node):
        actions = self.get_actions(node)
        for action in actions:
            child = self._do_action(node, action)
            self.graph.add_node(child, visits=0, reward=0)
            self.graph.add_edge(node, child, action=action, visits=0, reward=0)

    def simulate(self, node):
        while not self.is_terminal(node):
            action = self.rg.choice(self.get_actions(node))
            node = self._do_action(node, action)
        return node, self.get_reward(node)

    def backpropagate(
        self, selection_path: list, reward: float | int, accumulate: bool = True
    ):
        node = selection_path.pop()
        while selection_path:
            parent = selection_path.pop()

            self.graph.edges[parent, node]["visits"] += 1
            self.graph.edges[parent, node]["reward"] += reward

            if accumulate:
                self.graph.nodes[node]["visits"] += 1
                self.graph.nodes[node]["reward"] += reward

            node = parent

        # Root
        self.graph.nodes[node]["visits"] += 1
        self.graph.nodes[node]["reward"] += reward

    def is_leaf(self, node):
        return self.graph.out_degree(node) == 0

    def score_func(self, node, child, *args, **kwargs):
        return self._score_func(self.graph, node, child, *args, **kwargs)

    def best_child(self, node, **kw):
        children = list(self.graph.neighbors(node))
        self.rg.shuffle(children)
        scores = [self.score_func(node, child, **kw) for child in children]
        best = children[np.argmax(scores)]
        return best

    def traverse(self, root: Any = None, accumulate: bool = True, **selection_kw):
        root = self.root if root is None else root
        node, selection_path = self.select(root, **selection_kw)

        if self.is_terminal(node):
            sim_node = node
            reward = self.get_reward(node)

        else:
            self.expand(node)
            child_node = self.rg.choice(list(self.graph.neighbors(node)))
            selection_path.append(child_node)
            sim_node, reward = self.simulate(child_node)

        spath_backup = selection_path.copy()

        self.backpropagate(selection_path, reward, accumulate=accumulate)

        return spath_backup, reward, sim_node

    def accumulate_visits_and_rewards(self, graph: Optional[nx.DiGraph] = None):
        _accumulated = self.graph if graph is None else graph
        accumulate_visits_and_rewards(_accumulated)
        if graph is None:
            return _accumulated

    def search_mcts(
        self,
        n_steps: int,
        root: Optional[Any] = None,
        metric_func: Optional[Callable] = None,
        save_every: int = 1,
        exploration_constant: Optional[float] = None,
        accumulate: bool = True,
        accumulate_post: bool = False,
        progress_bar: bool = False,
        **kwargs,
    ):
        if accumulate and accumulate_post:
            raise ValueError(
                "Cannot set accumulate=True and accumulate_post=True."
                "Accumulation should occur either during or after search."
            )

        if exploration_constant is None:
            exploration_constant = self.exploration_constant
        if root is None:
            root = self.root
        if root not in self.graph.nodes:
            self.graph.add_node(root, visits=0, reward=0)

        if metric_func is None:
            metric_func = lambda *a, **kw: None

        metrics = [metric_func(self.graph, [], root, None, **kwargs)]
        if progress_bar:
            from tqdm import trange

            _range = trange
        else:
            _range = range
        for i in _range(n_steps):
            selection_path, reward, sim_node = self.traverse(
                root, accumulate=accumulate, **kwargs
            )
            if not i % save_every:
                m = metric_func(self.graph, selection_path, sim_node, reward, **kwargs)
                metrics.append(m)

        # Accumulate results on nodes post-hoc rather than at each step
        if accumulate_post:
            self.accumulate_visits_and_rewards()

        if metric_func is not None:
            return metrics

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

    def grow_tree_recursive(self, root=None):
        if root is None:
            root = self.root
            self.graph.add_node(root, visits=0, reward=0)

        def _grow_tree_recursive(node, action):
            if self.is_terminal(node):
                return
            next_node = self._do_action(node, action)
            if next_node not in self.graph:
                self.graph.add_node(next_node, visits=0, reward=0)
                for action in self.get_actions(next_node):
                    _grow_tree_recursive(next_node, action)
            self.graph.add_edge(
                node,
                next_node,
                visits=0,
                reward=0,
            )

        def _grow_tree(root):
            for action in self.get_actions(root):
                _grow_tree_recursive(root, action)

        _grow_tree(root)

    def bfs_iterator(self, root=None, shuffle=False):
        root = self.root if root is None else root
        layers = (l for l in nx.bfs_layers(self.graph, root))

        if shuffle:
            layers = list(layers)
            for l in layers:
                self.rg.shuffle(l)

        return chain(*layers)

    def make_bfs_search_generator(
        self,
        root: Optional[Any] = None,
        n_steps_per_node: int = 1,
        metric_func: Optional[Callable] = None,
        shuffle: bool = False,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        """
        Returns a generator that performs breadth-first search step-wise. Repeats the
        search indefinitely, or until the maximum number of steps is reached.
        Should be performed on a tree that has already been grown (all leaves are known)
        """

        if root is None:
            root = self.root

        if metric_func is None:
            metric_func = lambda *a, **kw: None

        if max_steps is None:
            max_steps = np.inf

        leaf_is_nonterminal = lambda n: (
            len(self.graph.neighbors(n)) == 0
        ) and not self.is_terminal(n)
        if any(leaf_is_nonterminal(n) for n in self.graph.nodes):
            self.grow_tree(root=root, n_visits=0)

        # Make an iterator that repeats each leaf n_steps_per_node times, cycling
        # endlessly in BFS order
        leaves = [
            repeat(n, n_steps_per_node)
            for n in self.bfs_iterator(root, shuffle=shuffle)
            if self.is_terminal(n)
        ]
        bfs = cycle(chain(*leaves))

        def bfs_do_one_iteration():
            k = 0
            while k < max_steps:
                n = next(bfs)
                reward = self.get_reward(n)
                self.graph.nodes[n]["reward"] += reward
                self.graph.nodes[n]["visits"] += 1
                k += 1
                yield reward

        return bfs_do_one_iteration

    def search_bfs(
        self,
        n_steps_per_node: int,
        n_repeats: int = 1,
        max_steps: int = np.inf,
        root: Optional[Any] = None,
        metric_func: Optional[Callable] = None,
        shuffle: bool = False,
        progress: bool = False,
        **kwargs,
    ):
        if root is None:
            root = self.root

        if metric_func is None:
            metric_func = lambda *a, **kw: None

        if self.graph.number_of_nodes() < 2:
            self.graph.add_node(root, visits=0, reward=0)
            self.grow_tree(root=root, n_visits=0)

        metrics = [metric_func(self.graph, root, None, **kwargs)]

        leaves = [
            n for n in self.bfs_iterator(root, shuffle=shuffle) if self.is_terminal(n)
        ]
        bfs = cycle(leaves)
        n_steps = min(n_repeats * len(leaves), max_steps)
        iterator = range(n_steps)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        for i in iterator:
            n = next(bfs)
            reward = sum(self.get_reward(n) for _ in range(n_steps_per_node))
            self.graph.nodes[n]["reward"] += reward
            self.graph.nodes[n]["visits"] += n_steps_per_node
            # self.graph.nodes[n]["history"].append(reward)
            metrics.append(metric_func(self.graph, n, reward, **kwargs))

        return metrics

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
