from functools import lru_cache, partial
from typing import Any, Callable, Optional
import networkx as nx
import numpy as np

__all__ = [
    "information_gain",
    "entropy",
    "get_mean_outcome",
    "get_outcome",
    "tree_modularity",
    "tree_modularity_estimate",
]


def information_gain(p, P):
    """Calculate tree modularity (avg information gain per decision)"""
    if P == 0 or P == 1:
        return 0
    else:
        HP = entropy(P)
        return (HP - entropy(p)) / HP


def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        not_p = 1 - p
        return -p * np.log2(p) - not_p * np.log2(not_p)


def get_mean_outcome(
    tree: nx.DiGraph,
    is_success: Callable[[Any], bool],
    is_terminal: Callable[[Any], bool],
    subroot: Any,
):
    """Get the mean outcome of a subtree (sub-DAG) of T rooted at subroot"""
    bfs_oriented: nx.DiGraph = nx.bfs_tree(tree, subroot)
    subnodes = bfs_oriented.nodes
    return np.mean([get_outcome(n, is_success) for n in subnodes if is_terminal(n)])


@lru_cache
def get_outcome(node: Any, is_success: Callable[[Any], bool]):
    return int(is_success(node))


def get_successes_and_outcomes(
    tree: nx.DiGraph,
    is_terminal: Callable[[str], bool],
    is_success: Callable[[Any], bool | float | int],
):
    n_successes = 0
    n_outcomes = 0
    for n in tree.nodes:
        if is_terminal(n):
            outcome = get_outcome(n, is_success)
            n_outcomes += 1
            n_successes += outcome

    return n_successes, n_outcomes


def tree_modularity(
    T: nx.DiGraph,
    root: Any,
    is_terminal: Callable[[str], bool],
    is_success: Callable[[str], bool],
) -> float:
    n_successes, n_outcomes = get_successes_and_outcomes(T, is_terminal, is_success)
    root_probability = n_successes / n_outcomes

    mean_outcome = partial(get_mean_outcome, T, is_success, is_terminal)
    IG = partial(information_gain, P=root_probability)

    modularity = 0
    node_layers = list(nx.bfs_layers(T, root))
    for layer in node_layers:
        n_layer = len(layer)
        mean_IG = sum(map(IG, map(mean_outcome, layer))) / n_layer
        modularity += mean_IG

    return modularity / len(node_layers)


def tree_modularity_estimate(
    T: nx.DiGraph,
    root: Any,
    reward_attr: str = "reward",
    visits_attr: str = "visits",
    p_success_attr: Optional[str] = None,
) -> float:
    """Estimate the modularity of a tree based on the leaves of the search tree."""

    # Get the mean outcome of each leaf in the search tree
    # (NOTE: The leaves are not necessarily terminal states of the MDP)
    if p_success_attr:
        p_success = {n: p for n, p in T.nodes(p_success_attr) if T.out_degree(n) == 0}
    else:
        if reward_attr and visits_attr:
            p_success = {
                n: attrs[reward_attr] / max(attrs[visits_attr], 1)
                for n, attrs in T.nodes(data=True)
                if T.out_degree(n) == 0
            }
        else:
            raise ValueError(
                "Must provide either p_success_attr or both reward_attr and visits_attr"
            )

    root_probability = np.mean(list(p_success.values()))

    bfs_layers = list(nx.bfs_layers(T, root))
    modularity = 0
    for layer in bfs_layers:
        IG_layer = 0
        for subroot in layer:
            # For each state-action pair, compute the mean probability of success
            # over all leaves that are accessible from that state-action pair
            mean_p_success = np.mean(
                [p_success[n] for n in nx.dfs_postorder_nodes(T, subroot) if n in p_success]
            )
            IG_layer += information_gain(mean_p_success, root_probability)

        # Add the mean information gain of all state-action pairs in the layer
        modularity += IG_layer / len(layer)

    # Compute the mean IG over all layers
    modularity = modularity / len(bfs_layers)
    
    if modularity < 0:
        ...
        
    return modularity
