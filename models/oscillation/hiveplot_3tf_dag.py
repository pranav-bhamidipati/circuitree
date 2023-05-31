from datetime import date
from pathlib import Path

import networkx as nx
import circuitree as ct
from models.oscillation.oscillation import OscillationTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from hiveplotlib import hive_plot_n_axes
from hiveplotlib.converters import networkx_to_nodes_edges
from hiveplotlib.node import split_nodes_on_variable
from hiveplotlib.viz import hive_plot_viz

save = True
save_dir = Path("./figures")
fmt = "png"
dpi = 300

pos_df = pd.read_csv("data/3tf_dag_nodes_xy.csv", index_col=0)
edges_df = pd.read_csv("data/3tf_dag_edges_xy.csv", index_col=0)

n_top = pos_df.shape[0]

# Get number of motifs in circuit
ot = OscillationTree(
    components=["A", "B", "C"],
    interactions=["activates", "inhibits"],
    root="ABC::",
    tree_shape="dag",
)
ring_motifs = ["AAi", "ABa_BAi", "ABa_BCa_CAi", "ABi_BCi_CAi"]
n_motifs = np.zeros(n_top, dtype=int)
for motif in ring_motifs:
    has_motif = np.zeros(n_top, dtype=bool)
    recolorings = ot.get_interaction_recolorings(motif)
    for recoloring in recolorings:
        interactions = recoloring.split("_")
        has_motif_coloring = np.all(
            [pos_df.index.str.contains(i) for i in interactions], axis=0
        )
        has_motif += has_motif_coloring
    pos_df[motif] = pd.Categorical(has_motif.astype(int), ordered=True)
    n_motifs += has_motif.astype(int)
pos_df["n_motifs"] = pd.Categorical(n_motifs, ordered=True)
max_n_motifs = n_motifs.max()

# Get depth in tree (number of interactions)
n_interactions = []
for genotype in pos_df.index:
    _, ints = genotype.split("::")
    if ints:
        ints = ints.count("_") + 1
    else:
        ints = 0
    n_interactions.append(ints)

pos_df["depth"] = n_interactions
max_depth = pos_df["depth"].max()

# Get order of nodes in the layer
pos_df["order"] = pos_df.groupby(pos_df["depth"])["n_motifs"].transform(np.argsort)

# Group the edges by number of motifs
edges_df["n_motifs"] = pos_df.loc[edges_df["source"], "n_motifs"].values
for motif in ring_motifs:
    edges_df[motif] = pos_df.loc[edges_df["source"], motif].values

# G: nx.DiGraph = nx.from_pandas_edgelist(
#     edges_df, source="source", target="target", create_using=nx.DiGraph
# )

...

today = date.today()

# convert `networkx` graph into `hiveplotlib`-ready nodes and edges
nodes, edges = networkx_to_nodes_edges(
    nx.from_pandas_edgelist(
        edges_df, source="source", target="target", create_using=nx.DiGraph
    )
)
edges_by_n_motifs = list(
    e.values for _, e in edges_df.groupby("n_motifs")[["source", "target"]]
)
edges_by_motif_presence = [
    [e.values for _, e in edges_df.groupby(motif)[["source", "target"]]]
    for motif in ring_motifs
]

edge_list_kwargs_n_motifs = [dict(alpha=0.0)] + [
    dict(color=f"C{i}", alpha=0.1) for i in range(max_n_motifs)
]
edge_list_kwargs_motif_presence = [dict(alpha=0.0)] + [dict(color="gray", alpha=0.1)]

# # add degree information to Node instances
# degrees = dict(G.degree)
# for node in nodes:
#     node.add_data(data=dict(degree=degrees[node.unique_id]))

# also store node id as data for later use
for node in nodes:
    node.add_data(data=dict(name=node.unique_id))

# add depth and graphviz position information to Node instances
for node in nodes:
    d, x, o = pos_df.loc[node.unique_id, ["depth", "x", "order"]]
    node.add_data(data=dict(depth=d))
    # node.add_data(data=dict(pos=o))
    node.add_data(data=dict(pos=x))

# partition nodes by group (we generated groups sequentially)
group_ids = split_nodes_on_variable(nodes, "depth")
group_splits = list(group_ids.values())


for i, motif in enumerate(ring_motifs):
    
    axes_names = [f"{d} interactions" for d in range(max_depth + 1)]
    hp = hive_plot_n_axes(
        node_list=nodes,
        edges=edges_by_motif_presence[i],
        # edges=edges_by_n_motifs,
        # edges=edges,
        axes_assignments=group_splits,
        sorting_variables=["pos"] * (max_depth + 1),
        axes_names=axes_names,
        edge_list_kwargs=edge_list_kwargs_motif_presence,
        # edge_list_kwargs=edge_list_kwargs_n_motif,
        # all_edge_kwargs=dict(lw=0.2),
        repeat_axes=[False] * (max_depth + 1),
        orient_angle=90,
    )

    fig, ax = hive_plot_viz(hp)

    ax.set_title(
        rf"3-Transcription Factor Circuits with motif `{motif}`",
        fontsize=24,
        y=1.1,
    )

    if save:
        fname = f"{today}_3tf_dag_hiveplot_motif{i}"
        fpath = Path(save_dir).joinpath(f"{fname}.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    
    ...

...

# hp = hive_plot_n_axes(
#     node_list=nodes,
#     edges=edges_by_n_motifs,
#     # edges=edges,
#     axes_assignments=group_splits,
#     sorting_variables=["pos"] * (max_depth + 1),
#     axes_names=[f"{d} interactions" for d in range(max_depth + 1)],
#     edge_list_kwargs=edge_list_kwargs_motif_presence[0],
#     # edge_list_kwargs=edge_list_kwargs_n_motif,
#     # all_edge_kwargs=dict(lw=0.2),
#     repeat_axes=[False] * (max_depth + 1),
#     orient_angle=90,
# )

# fig, ax = hive_plot_viz(hp)

# ax.set_title(
#     "Tree of all 3-Transcription Factor Circuits",
#     fontsize=20,
#     y=1.05,
# )

# if save:
#     fname = f"{today}_3tf_dag_hiveplot"
#     fpath = Path(save_dir).joinpath(f"{fname}.{fmt}")
#     print(f"Writing to: {fpath.resolve().absolute()}")
#     fig.savefig(fpath, dpi=dpi, bbox_inches="tight")

...
