from .modularity import tree_modularity_estimate

__all__ = [
    "sequential_reward",
    "sequential_reward_and_modularity_estimate",
    "mcts_reward",
    "mcts_reward_and_modularity_estimate",
]

"""Utility functions for recording metrics during a search"""


def sequential_reward(graph, n, reward=None, **kwargs):
    return reward


def sequential_reward_and_modularity_estimate(root, graph, n, reward=None, **kwargs):
    return reward, tree_modularity_estimate(graph, root, **kwargs)


def mcts_reward(graph, selection_path, n, reward=None, **kwargs):
    return reward


def mcts_reward_and_modularity_estimate(
    graph, selection_path, n, reward=None, **kwargs
):
    return reward, tree_modularity_estimate(graph, graph.root, **kwargs)


def mcts_reward_and_nodes_visited(graph, selection_path, n, reward=None, **kwargs):
    simulated_node = selection_path[-1] if selection_path else graph.root
    return reward, n, simulated_node


def mcts_reward_nodes_and_modularity_estimate(graph, selection_path, n, reward=None, **kwargs):
    simulated_node = selection_path[-1] if selection_path else graph.root
    MThat = tree_modularity_estimate(graph, graph.root, **kwargs)
    return reward, n, simulated_node, MThat
