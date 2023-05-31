from datetime import date
from pathlib import Path

import networkx as nx
import circuitree as ct
from models.oscillation.oscillation import OscillationTree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datashader import Canvas, count, count_cat
import datashader.transfer_functions as tf

# from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
from datashader.utils import export_image


def nodesplot(nodes, name=None, canvas=None, cat=None, cmap=["#FF3333"], opts=None):
    if opts is None:
        opts = dict()
    canvas = Canvas(**opts) if canvas is None else canvas
    aggregator = None if cat is None else count_cat(cat)
    agg = canvas.points(nodes, "x", "y", aggregator)
    return tf.spread(tf.shade(agg, cmap=cmap), px=3, name=name)


def edgesplot(edges, name=None, canvas=None, opts=None):
    if opts is None:
        opts = dict()
    canvas = Canvas(**opts) if canvas is None else canvas
    return tf.shade(
        canvas.line(edges, "x", "y", agg=count()),
        name=name,
    )


def graphplot(
    nodes, edges, name="", canvas=None, cat=None, cmap=["#FF3333"], opts=None
):
    if opts is None:
        opts = dict()
    if canvas is None:
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        canvas = Canvas(x_range=xr, y_range=yr, **opts)

    np = nodesplot(nodes, name + " nodes", canvas, cat=cat, cmap=cmap)
    ep = edgesplot(edges, name + " edges", canvas)
    return tf.stack(ep, np, how="over", name=name)


save = True
save_dir = Path("./figures")
fmt = "png"

pos_df = pd.read_csv("data/3tf_dag_nodes_xy.csv", index_col=0)
edges_df = pd.read_csv("data/3tf_dag_edges_xy.csv", index_col=0)

decay = 0.95
bw = 0.5
cvs_opts = dict(plot_height=400, plot_width=1000)

...

today = date.today()

n_top = pos_df.shape[0]
ring_motifs = [("AAi",), ("ABa", "BAi"), ("ABa", "BCa", "CAi"), ("ABi", "BCi", "CAi")]
n_motifs = np.zeros(n_top, dtype=int)
for interactions in ring_motifs:
    has_interactions = [pos_df.index.str.contains(i) for i in interactions]
    has_motif = np.all(has_interactions, axis=0)
    n_motifs += has_motif.astype(int)

pos_df["n_motifs"] = pd.Categorical(n_motifs, ordered=True)

# colors = ["#FFFFFF"] + [ct.viz.rgb2hex(plt.get_cmap("Dark2")(i)) for i in range(3)]

# pos_df = pos_df.loc[pos_df.n_motifs > 0]

# img = tf.Image(
#     graphplot(
#         pos_df,
#         connect_edges(pos_df, edges_df),
#         "DAG",
#         cat="n_motifs",
#         cmap=plt.get_cmap("Dark2")(range(4)),
#         # cmap=["#0aa844"],
#         opts=cvs_opts,
#     ),
# )

# if save:
#     fname = f"{today}_3tf_dag"
#     fpath = Path(save_dir).joinpath(f"{fname}.{fmt}")
#     print(f"Writing to: {fpath.resolve().absolute()}")
#     export_image(img, fpath.stem, export_path=fpath.parent)

...

# img2 = tf.Image(
    # nodesplot(
    #     pos_df,
    #     "DAG nodes",
    #     cat="n_motifs",
    #     cmap=colors,
    #     opts=cvs_opts,
    # )
    # graphplot(
    #     pos_df,
    #     hammer_bundle(pos_df, edges_df, decay=decay, initial_bandwidth=bw),
    #     "DAG bundled",
    #     cat="n_motifs",
    #     cmap=plt.get_cmap("Dark2")(range(4)),
    #     # cmap=["#0aa844"],
    #     opts=cvs_opts,
    # ),
# )

# if save:
#     fname = f"{today}_3tf_dag_bundled"
#     fpath = Path(save_dir).joinpath(f"{fname}.{fmt}")
#     print(f"Writing to: {fpath.resolve().absolute()}")
#     export_image(img2, fpath.stem, export_path=fpath.parent)

...

# pos = graphviz_layout(D, prog="dot")

# fig = plt.figure(figsize=(10, 5))

# nx.draw(
#     D,
#     pos=pos,
#     node_color=plt.get_cmap("gray")(0.5),
#     node_size=50,
#     with_labels=False,
#     edge_color=plt.get_cmap("gray")(0.65),
# )

# path = f"figures/{today}_3TF_dag_simple.png"
# if save:
#     print(f"Writing to: {path}")
#     plt.savefig(path)
# plt.close()

...

# Plot the same, but separate graph for each motif
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

edges_by_n_motifs = list(
    e.values for _, e in edges_df.groupby("n_motifs")[["source", "target"]]
)
edges_by_motif_presence = [
    [e.values for _, e in edges_df.groupby(motif)[["source", "target"]]]
    for motif in ring_motifs
]

...

today = date.today()

for i, (motif, edges) in enumerate(zip(ring_motifs, edges_by_motif_presence)):

    node_data = pos_df.loc[pos_df[motif] == 1][["x", "y"]]
    edge_data = edges_df.loc[edges_df[motif] == 1][["source", "target"]]
    img = tf.Image(
        graphplot(
            node_data,
            connect_edges(node_data, edge_data),
            motif,
            # cat="n_motifs",
            cmap=[ct.viz.rgb2hex(plt.get_cmap("gray")(0.5))],
            opts=cvs_opts,
        ),
    )
    
    if save:
        fname = f"{today}_3tf_dag_graphviz_motif_{i}"
        fpath = Path(save_dir).joinpath(f"{fname}.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        export_image(img, fpath.stem, export_path=fpath.parent, background="white")

