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
    root, graph, selection_path, n, reward=None, **kwargs
):
    return reward, tree_modularity_estimate(graph, root, **kwargs)
