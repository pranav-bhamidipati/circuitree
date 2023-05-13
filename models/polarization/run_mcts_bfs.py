import numpy as np
import pandas as pd
import networkx as nx

from circuitree import Circuit, CircuiTree

components = ["A", "B"]
interactions = ["activates", "inhibits"]
polar_df = pd.read_csv("polarization2_data.csv")

winners = polar_df.genotype.values
cores = winners[polar_df.core]
win_probabilities = dict(zip(polar_df.genotype.values, polar_df.Q.values))

# Two components, no interactions
root = "AB::"

# Exploration constant
c = np.sqrt(2)

# n_visits_per_node = 100
# batch_size = 10
# n_steps = 81 * n_visits_per_node

bfs = CircuiTree(
    components,
    interactions,
    # winners=winners,
    win_probabilities=win_probabilities,
    root=root,
    exploration_constant=c,
    tree_shape="dag",
)
results_bfs = bfs.search_bfs(
    10000,
    metric_func=bfs.complexity_graph_from_bfs,
)

graphs, simulated_nodes, reward = zip(*results_bfs)
data_bfs = {
    "graph": graphs,
    "simulated_node": simulated_nodes,
    "reward": reward,
}

...

mcts = CircuiTree(
    components,
    interactions,
    # winners=winners,
    win_probabilities=win_probabilities,
    root=root,
    exploration_constant=c,
    batch_size=1,
    tree_shape="dag",
)
results_mcts = mcts.search_mcts(
    810000, save_every=10000, metric_func=mcts.complexity_graph_from_mcts
)

graphs, selection_paths, simulated_nodes, reward = zip(*results_mcts)
data_mcts = {
    "graph": graphs,
    "selection_path": selection_paths,
    "simulated_node": simulated_nodes,
    "reward": reward,
}

...


# Pickle data (temporary solution - long-term, use a standard format)
import pickle
from pathlib import Path

p_mcts = Path("../figures/2023-03-03_MCTS_polarization.pickle")
p_bfs = Path("../figures/2023-03-03_BFS_polarization.pickle")

pickle.dump(data_mcts, p_mcts.open("wb"))
pickle.dump(data_bfs, p_bfs.open("wb"))
