import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from circuitree import Circuit, CircuiTree

components = ["A", "B"]
interactions = ["activates", "inhibits"]
polar_df = pd.read_csv("polarization2_data.csv")

winners = polar_df.genotype.values
cores = winners[polar_df.core]
win_probabilities = dict(zip(polar_df.genotype.values, polar_df.Q.values))


# Load data
import pickle
from pathlib import Path

p_mcts = Path("../figures/2023-03-03_MCTS_polarization.pickle")
data_mcts = pickle.load(p_mcts.open("rb"))

p_bfs = Path("../figures/2023-03-03_BFS_polarization.pickle")
data_bfs = pickle.load(p_bfs.open("rb"))


# color_array = np.array(
#     [plt.get_cmap("gray_r")(0.4), plt.get_cmap("Dark2")(0), plt.get_cmap("Dark2")(1)]
# )
color_array = np.array(
    [plt.get_cmap("gray_r")(0.2), plt.get_cmap("gray_r")(1.0), plt.get_cmap("tab10")(0)]
)


def plot_complexity_graph(
    G: nx.DiGraph,
    color_array=color_array,
    pos=None,
    fig=None,
    figsize=(10, 5),
    save=False,
    save_path="",
    axis_limits=None,
    max_node_visits=None,
    min_node_visits=0,
    max_edge_visits=None,
    **kwargs,
):
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if pos is None:
        pos = graphviz_layout(G, prog="dot")

    unique_genotypes = np.array([CircuiTree.get_unique_genotype(n) for n in G.nodes])
    # solution_type = np.isin(unique_genotypes, winners).astype(int) + np.isin(
    #     unique_genotypes, cores
    # ).astype(int)
    # node_edge_colors = color_array[solution_type]

    colors = np.array(
        [win_probabilities.get(CircuiTree.get_edge_code(n), 0.0) for n in G.nodes]
    )

    # max_P = max(win_probabilities.values())
    # sizes = 75 + 475 * np.array(
    #     [win_probabilities.get(mcts.get_edge_code(n), 0.0) / max_P for n in G.nodes]
    # )

    edge_visits = [v for _, _, v in G.edges(data="visits")]
    if max_edge_visits is None:
        max_edge_visits = max(edge_visits, default=2)

    edge_colors = []
    for v in edge_visits:
        if v > 0:
            logv_norm = np.log10(v) / np.log10(max_edge_visits)
            c = plt.get_cmap("gray_r")(0.15 + 0.85 * logv_norm)
        else:
            c = (0, 0, 0, 0)
        edge_colors.append(c)
    edge_colors = np.array(edge_colors)

    # edge_colors = plt.get_cmap("gray_r")(edge_visits / edge_visits.max())

    node_visits = []
    for _, v in G.nodes("visits"):
        if v is None:
            node_visits.append(0)
        else:
            node_visits.append(v)
    node_visits = np.array(node_visits)

    if max_node_visits is None:
        max_node_visits = max(node_visits, default=2)

    logv = np.log10(node_visits + 1)
    logv_min = np.log10(min_node_visits + 1)
    logv_max = np.log10(max_node_visits + 1)
    if len(logv) > 0:
        # logv_norm = (logv - logv_min) / (logv_max - logv_min)
        sizes = 150 * logv / 4
    else:
        # logv_norm = np.array([])
        sizes = np.array([])

    # sizes = 30 + 600 * np.nan_to_num(logv_norm)
    # sizes = 30 + 1e-3 * node_visits

    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color=colors,
        cmap=plt.get_cmap("gist_heat_r"),
        vmin=0.0,
        vmax=max(win_probabilities.values()),
        # linewidths=2,
        edgecolors="k",
        node_size=sizes,
        edge_color=edge_colors,
        width=2,
        # edge_cmap=,
        **kwargs,
    )

    if axis_limits is not None:
        plt.axis(axis_limits)

    if save:
        print(f"Writing to: {save_path}")
        fig.savefig(save_path)
        plt.close()
    else:
        return fig


n_plot = 3

gb = data_bfs["graph"][-1]
pos = graphviz_layout(gb, prog="dot")
fig = plt.figure(figsize=(10, 5))
plot_complexity_graph(gb, fig=fig)
axis_lims = plt.axis()

gm = data_mcts["graph"][-1]
max_edge_v = max(v for _, _, v in gm.edges(data="visits"))
max_node_v = max(v for _, v in gm.nodes(data="visits"))

# save_steps = np.linspace(0, 81, n_plot, dtype=int)
save_steps = 15, 30, 81
for step in save_steps:
    plot_complexity_graph(
        G=data_mcts["graph"][step],
        max_node_visits=max_node_v,
        max_edge_visits=max_edge_v,
        pos=pos,
        axis_limits=axis_lims,
        save=True,
        save_path=f"../figures/tmp/polarization_MCTS_step{step}.png",
    )
    plot_complexity_graph(
        G=data_bfs["graph"][step],
        max_node_visits=max_node_v,
        max_edge_visits=max_edge_v,
        pos=pos,
        axis_limits=axis_lims,
        save=True,
        save_path=f"../figures/tmp/polarization_BFS_step{step}.png",
    )


fig = plt.figure(figsize=(6, 4))

# Plot number of successes vs number of samples
samples = np.arange(1, 82) / 10
bfs_successes = 10000 * np.cumsum(data_bfs["reward"][1:])
mcts_successes = np.array(
    [sum(r for _, r in g.nodes("reward")) for g in data_mcts["graph"][1:]]
)
plt.plot(samples, bfs_successes, label="BFS")
plt.plot(samples, mcts_successes, label="MCTS")
plt.xlabel(r"$10^5$ Samples")
plt.ylabel("Successes")
plt.legend()

save_path = "../figures/success_v_samples_bfs_mcts.png"
print(f"Writing to: {save_path}")
plt.savefig(save_path)

...
