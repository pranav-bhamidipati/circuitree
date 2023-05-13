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
Qs = polar_df.Q.values
cores = winners[polar_df.core]
win_probabilities = dict(zip(polar_df.genotype.values, polar_df.Q.values))
top10_winners = winners[np.argsort(Qs)][::-1][:10]
top10_mask = np.isin(winners, top10_winners)

solution_codes = {g: 1 + int(g in cores) for g in winners}

color_array = np.array(
    [plt.get_cmap("gray_r")(0.4), plt.get_cmap("Dark2")(0), plt.get_cmap("Dark2")(1)]
)

# Load data
import pickle
from pathlib import Path

data_dir = Path("../data/2023-03-05_single_player")
data_mcts = {}
for f in data_dir.glob(f"*.pickle"):
    data = pickle.load(f.open("rb"))
    batchsize = data["batch_size"]
    # var_const = data["variance_constant"]
    method = data["method"]
    if method.lower() == "sp-mcts":
        continue
    data_mcts[(batchsize, method)] = data

p_bfs = Path("../figures/2023-03-03_BFS_polarization.pickle")
data_bfs = pickle.load(p_bfs.open("rb"))

...

fig = plt.figure(figsize=(6, 4))

# Plot number of successes vs number of samples
samples = np.arange(1, 82) / 10

sorted_data = sorted(data_mcts.items())
for (batchsize, method), data in sorted_data:
    successes = batchsize * np.array(
        [sum(r for _, r in g.nodes("reward")) for g in data["graph"][1:]]
    )
    plt.plot(samples, successes, label=f"{method.upper()}, b={batchsize}")

    # Plot a point when each circuit is "discovered"
    # for gen, code in solution_codes.items():
    #     node_label = f"AB::{gen}"
    #     for i, g in enumerate(data["graph"][1:], start=1):
    #         if node_label in g.nodes:
    #             payout = g.nodes[node_label]["reward"] * batchsize
    #             if payout > 5:
    #                 plt.scatter(samples[i], successes[i], color=color_array[code], s=15)
    #                 break

bfs_batchsize = 10000
bfs_successes = bfs_batchsize * np.cumsum(data_bfs["reward"][1:])
plt.plot(samples, bfs_successes, label=f"BFS", color="k")

# g = data_bfs["graph"][-1]
# payouts = bfs_batchsize * np.array(data_bfs["reward"][1:])
# for node, payout in zip(data_bfs["simulated_node"][1:], payouts):
#     if payout > 5:
#         code = solution_codes.get(CircuitTreeSearch.get_unique_genotype(node))
#         plt.scatter(samples[i], bfs_successes[i], color=color_array[code], s=15)


plt.xlabel(r"$10^5$ Samples")
plt.ylabel("Successes")
plt.legend(
    title="Search method",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    fancybox=True,
    shadow=True,
)

# plt.tight_layout()
# save_path = "../figures/tmp/batchsize/bfs_mcts_success_v_trials_w_batchsize.png"
# print(f"Writing to: {save_path}")
# plt.savefig(save_path)

...

fig = plt.figure(figsize=(6, 4))

# Plot number of circuits found vs number of samples
samples = np.arange(82) / 10

formats = {
    (1, "MCTS"): ("k", "solid"),
    (5, "MCTS"): ("k", "dashed"),
    (10, "MCTS"): ("gray", "solid"),
    (25, "MCTS"): ("gray", "dashed"),
    # (1, "SP-MCTS"): ("k", "dotted"),
    # (5, "SP-MCTS"): ("k", "dashdot"),
    # (10, "SP-MCTS"): ("gray", "dotted"),
    # (25, "SP-MCTS"): ("gray", "dashdot"),
}
sorted_data = sorted(data_mcts.items())
for (batchsize, method), data in sorted_data:
    winners_found = np.zeros((82, len(winners)), dtype=int)
    for i, g in enumerate(data["graph"][1:]):
        for j, w in enumerate(winners):
            label = f"AB::{w}"
            if label in g.nodes:
                payout = g.nodes[label]["reward"] * batchsize
                winners_found[i + 1, j] = int(payout > 5)
    n_winners_found = winners_found.sum(axis=1)

    color, linestyle = formats[(batchsize, method.upper())]
    plt.plot(
        samples,
        n_winners_found,
        label=f"{method.upper()}, nbatch={batchsize}",
        color=color,
        linestyle=linestyle,
        linewidth=1,
    )

bfs_batchsize = 10000
bfs_winners_found = []
for n, r in zip(data_bfs["simulated_node"], data_bfs["reward"]):
    if n.split("::")[-1] in winners:
        bfs_winners_found.append(r * bfs_batchsize > 5)
    else:
        bfs_winners_found.append(False)
bfs_n_winners_found = np.cumsum(bfs_winners_found)

# bfs_n_winners_found = np.cumsum(
#     [n.split("::")[-1] in winners for n in data_bfs["simulated_node"]]
# )

plt.plot(samples, bfs_n_winners_found, label=f"BFS", color="k", linestyle="dotted")

plt.legend(title="Search method")
plt.xlabel(r"$10^5$ Samples")
plt.ylabel("# Solutions found")
plt.tight_layout()

save_path = "../figures/tmp/batchsize/bfs_mcts_circuits_v_trials.png"
print(f"Writing to: {save_path}")
plt.savefig(save_path)

...

fig = plt.figure(figsize=(6, 4))

# Plot number of circuits found vs number of samples
samples = np.arange(82) / 10

formats = {
    (1, "MCTS"): ("k", "solid"),
    (5, "MCTS"): ("k", "dashed"),
    (10, "MCTS"): ("gray", "solid"),
    (25, "MCTS"): ("gray", "dashed"),
    # (1, "SP-MCTS"): ("k", "dotted"),
    # (5, "SP-MCTS"): ("k", "dashdot"),
    # (10, "SP-MCTS"): ("gray", "dotted"),
    # (25, "SP-MCTS"): ("gray", "dashdot"),
}
sorted_data = sorted(data_mcts.items())
for (batchsize, method), data in sorted_data:
    top10_found = np.zeros((82, 10), dtype=int)
    for i, g in enumerate(data["graph"][1:]):
        for j, w in enumerate(top10_winners):
            label = f"AB::{w}"
            if label in g.nodes:
                payout = g.nodes[label]["reward"] * batchsize
                top10_found[i + 1, j] = int(payout >= 5)
    n_top10_found = top10_found.sum(axis=1)

    color, linestyle = formats[(batchsize, method.upper())]
    plt.plot(
        samples,
        n_top10_found,
        label=f"{method.upper()}, nbatch={batchsize}",
        color=color,
        linestyle=linestyle,
        linewidth=1,
    )

bfs_batchsize = 10000
bfs_top10_found = []
for n, r in zip(data_bfs["simulated_node"], data_bfs["reward"]):
    if n.split("::")[-1] in top10_winners:
        bfs_top10_found.append(r * bfs_batchsize > 5)
    else:
        bfs_top10_found.append(False)
bfs_n_top10_found = np.cumsum(bfs_top10_found)

# bfs_n_top10_found = np.cumsum(
#     [n.split("::")[-1] in top10_winners for n in data_bfs["simulated_node"]]
# )
plt.plot(samples, bfs_n_top10_found, label=f"BFS", color="k", linestyle="dotted")


plt.legend(title="Search method")
plt.xlabel(r"$10^5$ Samples")
plt.ylabel("# Top-10 Solutions found")
plt.tight_layout()

save_path = "../figures/tmp/batchsize/bfs_mcts_top10_v_trials.png"
print(f"Writing to: {save_path}")
plt.savefig(save_path)

...
