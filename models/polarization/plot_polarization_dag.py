from datetime import date
from pathlib import Path
import pickle

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import pandas as pd

from polarization import PolarizationTree

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib import cm

save = True

components = ["A", "B"]
interactions = ["activates", "inhibits"]
polar_df = pd.read_csv("data/polarization2_data.csv")

winners = polar_df.genotype.values
cores = winners[polar_df.core]
win_probabilities = dict(zip(polar_df.genotype.values, polar_df.Q.values))

# Two components, no interactions
root = "AB::"

dag = PolarizationTree(
    components=components,
    interactions=interactions,
    root=root,
    winners=winners,
    tree_shape="dag",
)
dag.grow_tree(n_visits=1)
D = dag.complexity_graph_from_bfs(dag.graph)[0]

today = date.today()

...

pos = graphviz_layout(D, prog="dot")

fig = plt.figure(figsize=(10, 5))

nx.draw(
    D,
    pos=pos,
    node_color=plt.get_cmap("gray")(0.5),
    node_size=50,
    with_labels=False,
    edge_color=plt.get_cmap("gray")(0.65),
)

path = f"figures/{today}_polarization_dag_simple.png"
if save:
    print(f"Writing to: {path}")
    plt.savefig(path)
plt.close()

...

p = next(Path("../data/2023-03-07_mcts_polarization").glob("*_mcts_*_batch5.pickle"))
mcts_data = pickle.load(p.open("rb"))
batchsize = mcts_data["batch_size"]

# color_array = np.array(
#     [plt.get_cmap("gray")(0.5), plt.get_cmap("Set1")(0), plt.get_cmap("Set1")(2)]
# )


# color_array = np.array(
#     [plt.get_cmap("gray_r")(0.4), plt.get_cmap("tab20b")(0), plt.get_cmap("tab20b")(2)]
# )
color_array = [plt.get_cmap("gray_r")(0.4), "#77191c", "#4b2023"]


solution_code = {}
for n in D.nodes:
    g = dag.get_unique_genotype(n)
    solution_code[g] = int(g in winners) + int(g in cores)

pos = graphviz_layout(D, prog="dot")


# steps = np.arange(1, 21)
# step = 50
# step = 40
step = 80

G = mcts_data["graph"][step]
# sizes = []
colors = []
for n in G.nodes:
    g = dag.get_unique_genotype(n)
    p = win_probabilities.get(g, 0.0)
    # sizes.append(50 + p / 0.06 * 400)
    # colors.append(color_array[solution_code[g]])
    colors.append(p)

# edge_visits = [v for _, _, v in G.edges(data="visits")]
# max_edge_visits = max(max(edge_visits, default=10), 10)
# min_edge_visits = max(min(edge_visits, default=1), 1)
# log_max_edge_visits = np.log10(max_edge_visits)
# log_min_edge_visits = np.log10(min_edge_visits)
# denom = log_max_edge_visits - log_min_edge_visits

# if edge_visits:
#     edge_colors = []
#     edge_widths = []
#     for v in edge_visits:
#         if v > 0:
#             logv = np.log10(v)
#             logv_norm = (logv - log_min_edge_visits) / denom
#             c = plt.get_cmap("gray_r")(logv_norm)
#             w = 5 * logv_norm
#         else:
#             c = (0, 0, 0, 0)
#             w = 0
#         edge_colors.append(c)
#         edge_widths.append(w)

#     edge_colors = np.array(edge_colors)
#     edge_widths = np.array(edge_widths)
# else:
#     edge_colors = None
#     edge_widths = None

# edge_flow_dict = {}
# for n in G.nodes:
#     out_edges = G.out_edges(n, data="visits")
#     if not out_edges:
#         continue
#     edges, visits = zip(*((e, v) for *e, v in out_edges))
#     flows = np.array(visits) / np.sum(visits)
#     for e, f in zip(edges, flows):
#         print(e)
#         print("\t", f)
#         edge_flow_dict[tuple(e)] = f
# edge_flows = np.array([edge_flow_dict[e] for e in G.edges])
# edge_widths = 3 * edge_flows

depth = {}
depth_total_visits = {}
for n, n2, v in G.edges(data="visits"):
    d = 0
    if edges := n.split("::")[-1]:
        d = 1 + edges.count("_")
    depth[(n, n2)] = d
    if d in depth_total_visits:
        depth_total_visits[d].append(v)
    else:
        depth_total_visits[d] = list()

edge_weights = []
for e in G.edges:
    tv = depth_total_visits[depth[e]]
    w = G.edges[e]["visits"] / sum(tv) * len(tv)
    edge_weights.append(w)
edge_widths = [1 * w for w in edge_weights]

...

fig = plt.figure(figsize=(10, 5))

nx.draw(
    G,
    pos=pos,
    node_color=colors,
    # node_size=sizes,
    node_size=200,
    with_labels=False,
    edge_color=plt.get_cmap("gray")(0.4),
    width=edge_widths,
)

for n, (x, y) in pos.items():
    g = dag.get_unique_genotype(n)
    if g in winners:
        plt.scatter(
            x, y, color=plt.get_cmap("gray")(0.9), marker="*", s=75, zorder=1000
        )

# plt.suptitle(f"MCTS search traffic after {step}0,000 samples", fontsize=16)

plt.colorbar(
    cm.ScalarMappable(mplcolors.Normalize(0, max(win_probabilities.values()))),
    ax=plt.gca(),
    cmap=plt.get_cmap("cividis"),
    location="right",
    shrink=0.6,
    # label="P(success)",
)

path = f"../figures/{today}_polarization_dag.png"
if save:
    print(f"Writing to: {path}")
    plt.savefig(path)
plt.close()

...

fig = plt.figure(figsize=(10, 5))

nx.draw(
    G,
    pos=pos,
    node_color=colors,
    # node_size=sizes,
    node_size=200,
    with_labels=False,
    edge_color=plt.get_cmap("gray")(0.4),
    width=0.5,
)

for n, (x, y) in pos.items():
    g = dag.get_unique_genotype(n)
    if g in winners:
        plt.scatter(
            x, y, color=plt.get_cmap("gray")(0.9), marker="*", s=75, zorder=1000
        )

plt.colorbar(
    cm.ScalarMappable(mplcolors.Normalize(0, max(win_probabilities.values()))),
    ax=plt.gca(),
    cmap=plt.get_cmap("cividis"),
    location="right",
    shrink=0.6,
    # label="P(success)",
)

path = (
    f"../figures/polarization_dag/{today}_polarization_dag_step0_batch{batchsize}.png"
)
if save:
    print(f"Writing to: {path}")
    plt.savefig(path)
plt.close()

...


### Calculate tree modularity (avg information gain per decision)
def information_gain(p, P):
    HP = entropy(P)
    return (HP - entropy(p)) / HP


def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        not_p = 1 - p
        return -p * np.log2(p) - not_p * np.log2(not_p)


# An earlier draft of this function - this has been deprecated in 
# favor of the version in the `modularity` module
def tree_modularity(G, root, winners):
    modularity = 0
    root_probability = len(winners) / len(G.nodes)
    node_layers = list(nx.bfs_layers(G, root))
    depth = len(node_layers)
    for layer in node_layers:
        n_subroots = len(layer)
        layer_probabilities = np.zeros(n_subroots)
        for i, subroot in enumerate(layer):
            layer_probabilities[i] = np.mean(
                [
                    float(dag.get_unique_genotype(n) in winners)
                    for n in nx.bfs_tree(G, subroot).nodes
                ]
            )
        mean_information_gain = np.mean(
            [information_gain(p, root_probability) for p in layer_probabilities]
        )
        modularity += mean_information_gain
        print(f"Layer IG: {mean_information_gain:.3f}")
        print(f"Modularity: {modularity:.3f}")
        ...
    return modularity / depth


Mt = tree_modularity(G, root, winners)
print(f"Tree modularity: {Mt:.3f}")

...
