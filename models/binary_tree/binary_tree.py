from datetime import date
from functools import partial
import h5py
from math import floor, ceil
from more_itertools import sliced
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Iterable, Mapping

from sacred import Experiment

from circuitree import CircuiTree
from circuitree.rewards import (
    sequential_reward,
    sequential_reward_and_modularity_estimate,
    mcts_reward,
    mcts_reward_and_modularity_estimate,
)

# __all__ = ["BinaryTree"]


def int_to_bincode(n: int, w: int = 0, prefix: str = ""):
    """Convert an integer to a binary string"""
    return prefix + bin(n)[2:].zfill(w)


def bincode_to_int(b: str, prefix: str = ""):
    """Convert an integer to a binary string"""
    return int(b.lstrip(prefix), 2)


def get_branch(b: str):
    """Convert an integer to a binary string"""
    return (b, b + "0"), (b, b + "1")


def unique_binary_outcomes(b: str, D: Optional[int] = None):
    """Hierarchically sorts the binary-labeled leaves of a complete (binary) tree"""
    if D is None:
        D = round(np.log2(len(b)))

    for d in range(D):
        s = list(sliced(b, 2**d, strict=True))
        grouped_arr = np.array(s, dtype=str).reshape(-1, 2)
        b = "".join(np.sort(grouped_arr, axis=1).flat)

    return b


def idx_to_outcome(successes, depth, unique=True):
    o = np.array(["0"] * 2**depth, dtype=str)
    o[successes] = "1"
    s = "".join(o)
    if unique:
        s = unique_binary_outcomes(s, depth)
    return s


def build_binary_tree(
    max_depth: int,
    root: str = "",
    n_visits: int = 0,
    extend_from_graph: nx.DiGraph = None,
):
    if extend_from_graph is None:
        # create an empty directed graph
        G = nx.DiGraph()

        # add the root node
        G.add_node(root, visits=n_visits, reward=0, ssq_reward=0)
    else:
        G = extend_from_graph.copy()

    # create a list-of-lists to store the nodes at each depth
    nodes = [[] for i in range(max_depth + 1)]
    for i, layer in enumerate(nx.bfs_layers(G, root)):
        nodes[i].extend(layer)

    # loop through each depth level and add child nodes
    for depth in range(1, max_depth + 1):
        # loop through the nodes at the previous depth level
        for parent in nodes[depth - 1]:
            left_child = parent + "0"
            right_child = parent + "1"
            if left_child not in G:
                G.add_node(left_child, visits=n_visits, reward=0, ssq_reward=0)
                G.add_edge(parent, left_child, visits=n_visits, reward=0, ssq_reward=0)
                nodes[depth].append(left_child)
            if right_child not in G:
                G.add_node(right_child, visits=n_visits, reward=0, ssq_reward=0)
                G.add_edge(parent, right_child, visits=n_visits, reward=0, ssq_reward=0)
                nodes[depth].append(right_child)

    return G


class BinaryTree(CircuiTree):
    def __init__(
        self,
        success_codes: Optional[Iterable[str]] = None,
        n_successes: Optional[int] = None,
        depth: Optional[int] = None,
        density: Optional[float] = None,
        build: bool = True,
        random: bool = False,
        rg: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        root: str = "",
        **kwargs,
    ):
        super().__init__(root=root, seed=seed, rg=rg, **kwargs)

        if depth is None:
            if success_codes is None:
                raise ValueError("Must specify success_codes if depth is not specified")
            else:
                self.depth = max(map(len, success_codes))
        else:
            self.depth = depth

        n_outcomes = 2**self.depth

        if success_codes is None:
            if self.depth is None:
                raise ValueError("Must specify depth if success_codes is not specified")

            if n_successes is None:
                if density is not None:
                    if (density > 1) or (density < 0):
                        raise ValueError("p must be in the domain [0, 1]")
                else:
                    density = np.sqrt(n_outcomes) / n_outcomes

                n_successes = floor(n_outcomes * density)

            if random:
                self.success_codes = [
                    f"{i:0{depth}b}"
                    for i in self.rg.choice(n_outcomes, n_successes, replace=False)
                ]
            else:
                self.success_codes = [f"{i:0{depth}b}" for i in range(n_successes)]

            self.n_outcomes = n_outcomes
            self.n_successes = n_successes
        else:
            self.success_codes = success_codes
            self.n_successes = len(success_codes)

        self.p = self.n_successes / n_outcomes

        delattr(self, "graph")
        if build:
            self.extend_tree()
            for n in self.graph.nodes:
                self.graph.nodes[n]["is_terminal"] = self.is_terminal(n)
                self.graph.nodes[n]["is_success"] = self.is_success(n)
        else:
            self.graph = nx.DiGraph()
            self.graph.add_node(self.root, visits=0, reward=0, ssq_reward=0)

    def extend_tree(self, n_visits: int = 0):
        self_graph = getattr(self, "graph", None)
        self.graph = build_binary_tree(
            max_depth=self.depth,
            root=self.root,
            n_visits=n_visits,
            extend_from_graph=self_graph,
        )

    def do_action(self, state: str, action: str):
        if action == "left":
            new_state = state + "0"
        elif action == "right":
            new_state = state + "1"
        return new_state

    def get_actions(self, state: str) -> Iterable[str]:
        if self.is_terminal(state):
            return []
        else:
            return ["left", "right"]

    def is_terminal(self, state: str) -> bool:
        if len(state) < self.depth:
            return False
        elif len(state) == self.depth:
            return True
        elif len(state) > self.depth:
            raise ValueError("State is too long")

    def is_success(self, state: str) -> bool:
        return state in self.success_codes

    def get_reward(self, state: str) -> int:
        reward = int(self.is_success(state))
        return reward

    @staticmethod
    def get_unique_state(self, state: str) -> str:
        return state


def search_sequential(
    outcome_code: str,
    N: int,
    seed: Optional[int] = None,
    rg: Optional[np.random.Generator] = None,
    estimate_modularity: bool = False,
    **kwargs,
):
    if rg is None:
        if seed is None:
            raise ValueError("Must specify random seed if rg is not specified")
        else:
            rg = np.random.default_rng(seed)
    else:
        if seed is None:
            seed = rg.bit_generator._seed_seq.entropy

    l = len(outcome_code)
    depth = round(np.log2(l))
    if 2**depth != l:
        raise ValueError("Number of outcomes in outcome_code is not a power of 2")

    success_codes = [
        np.binary_repr(i, width=depth) for i, o in enumerate(outcome_code) if o == "1"
    ]

    bt = BinaryTree(
        seed=seed,
        success_codes=success_codes,
        rg=rg,
        build=True,
    )

    n_repeats = ceil(N / l)

    if estimate_modularity:
        metric_func = partial(sequential_reward_and_modularity_estimate, bt.root)

        results = bt.search_bfs(
            1, n_repeats, max_steps=N, metric_func=metric_func, shuffle=True
        )

        rewards, modularity_estimates = zip(*results[1:])
        rewards = np.array(rewards, dtype=int)
        modularity_estimates = np.array(modularity_estimates, dtype=float)
        data = {"modularity_estimates": modularity_estimates}

    else:
        metric_func = sequential_reward

        rewards = bt.search_bfs(
            1, n_repeats, max_steps=N, metric_func=metric_func, shuffle=True
        )[1:]

        data = {}

    data["outcome_codes"] = outcome_code
    data["N"] = N
    data["seed"] = seed
    data["rewards"] = rewards
    data["modularity"] = bt.modularity
    data["depth"] = depth

    return data


def success_codes_from_outcome_code(outcome_code: str, depth: int) -> list[str]:
    return [
        np.binary_repr(i, width=depth) for i, o in enumerate(outcome_code) if o == "1"
    ]


def search_mcts(
    outcome_code: str,
    N: int,
    exploration_constant: Optional[float] = None,
    seed: Optional[int] = None,
    rg: Optional[np.random.Generator] = None,
    estimate_modularity: bool = False,
    **kwargs,
):
    if rg is None:
        if seed is None:
            raise ValueError("Must specify random seed if rg is not specified")
        else:
            rg = np.random.default_rng(seed)
    else:
        if seed is None:
            seed = rg.bit_generator._seed_seq.entropy

    l = len(outcome_code)
    depth = round(np.log2(l))
    if 2**depth != l:
        raise ValueError("Number of outcomes in outcome_code is not a power of 2")

    success_codes = success_codes_from_outcome_code(outcome_code, depth)

    bt = BinaryTree(
        seed=seed,
        success_codes=success_codes,
        rg=rg,
        build=False,
    )

    if estimate_modularity:
        metric_func = partial(mcts_reward_and_modularity_estimate, bt.root)

        results = bt.search_mcts(
            N, metric_func=metric_func, exploration_constant=exploration_constant
        )

        rewards, modularity_estimates = zip(*results[1:])
        rewards = np.array(rewards, dtype=int)
        modularity_estimates = np.array(modularity_estimates, dtype=float)

        data = {"modularity_estimates": modularity_estimates}

    else:
        metric_func = mcts_reward
        rewards = bt.search_mcts(
            N, metric_func=metric_func, exploration_constant=exploration_constant
        )[1:]
        data = {}

    bt.extend_tree()  # extend tree to max depth
    modularity = bt.modularity

    data["outcome_codes"] = outcome_code
    data["N"] = N
    data["seed"] = seed
    data["rewards"] = rewards
    data["modularity"] = modularity
    data["depth"] = depth

    return data


def binary_tree_search(
    outcome_code: str,
    N: int,
    method: Literal["mcts", "sequential"] = "mcts",
    seed: int = 2023,
    exploration_constant: Optional[float] = None,
    save: bool = False,
    estimate_modularity: bool = False,
    ex: Optional[Experiment] = None,
    **kwargs,
):
    if method == "mcts":
        search = search_mcts
    elif method == "sequential":
        search = search_sequential
    else:
        raise ValueError(f"Unknown search method: {method}")

    rg = np.random.default_rng(seed)

    data = search(
        outcome_code,
        N,
        rg=rg,
        exploration_constant=exploration_constant,
        estimate_modularity=estimate_modularity,
        **kwargs,
    )

    if ex is not None:
        artifacts = []

        save_dir = Path(ex.observers[0].dir)

        if save:
            p = save_dir.joinpath("results.hdf5")
            # print(f"Writing to: {p.resolve().absolute()}")

            with h5py.File(p, "w") as f:
                for k, v in data.items():
                    f.create_dataset(k, data=v)

            artifacts.append(p)

        for a in artifacts:
            ex.add_artifact(a)
