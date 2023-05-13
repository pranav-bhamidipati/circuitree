import matplotlib.cm as cm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

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

exploited_cutoff = 0.9 * 33
explored_cutoff = 0.9 * 81

save = True

# Load data
import pickle
from pathlib import Path

data_dir = Path("../data/2023-03-07_mcts_polarization")
data_mcts = {}
for f in data_dir.glob(f"*.pickle"):
    data = pickle.load(f.open("rb"))
    batchsize = data["batch_size"]
    if batchsize != 5:
        continue
    # expl_const = data["exploration_constant"]
    # data_mcts[(batchsize, expl_const)] = data
    data_mcts[batchsize] = data

p_bfs = Path("../figures/2023-03-03_BFS_polarization.pickle")
data_bfs = pickle.load(p_bfs.open("rb"))

# ...

# fig = plt.figure(figsize=(6, 4))

# # Plot number of successes vs number of samples
# samples = np.arange(1, 82) / 10

# sorted_data = sorted(data_mcts.items())
# for (batchsize, expl_const), data in sorted_data:
#     successes = batchsize * np.array(
#         [sum(r for _, r in g.nodes("reward")) for g in data["graph"][1:]]
#     )
#     plt.plot(samples, successes, label=f"MCTS, b={batchsize}, c={expl_const:.2f}")

#     # Plot a point when each circuit is "discovered"
#     # for gen, code in solution_codes.items():
#     #     node_label = f"AB::{gen}"
#     #     for i, g in enumerate(data["graph"][1:], start=1):
#     #         if node_label in g.nodes:
#     #             payout = g.nodes[node_label]["reward"] * batchsize
#     #             if payout > 5:
#     #                 plt.scatter(samples[i], successes[i], color=color_array[code], s=15)
#     #                 break

# bfs_batchsize = 10000
# bfs_successes = bfs_batchsize * np.cumsum(data_bfs["reward"][1:])
# plt.plot(samples, bfs_successes, label=f"BFS", color="k")

# # g = data_bfs["graph"][-1]
# # payouts = bfs_batchsize * np.array(data_bfs["reward"][1:])
# # for node, payout in zip(data_bfs["simulated_node"][1:], payouts):
# #     if payout > 5:
# #         code = solution_codes.get(CircuitTreeSearch.get_unique_genotype(node))
# #         plt.scatter(samples[i], bfs_successes[i], color=color_array[code], s=15)


# plt.xlabel(r"$10^5$ Samples")
# plt.ylabel("Successes")
# plt.legend(
#     title="Search method",
# )

# if save:
#     plt.tight_layout()
#     # save_path = "../figures/tmp/expl_const/bfs_mcts_success_v_trials_w_batchsize.png"
#     # print(f"Writing to: {save_path}")
#     # plt.savefig(save_path)
# plt.close()

...

fig = plt.figure(figsize=(5, 3))

# Plot number of circuits found vs number of samples
samples = np.arange(82) / 10

formats = [
    ("k", "solid"),
    ("k", "dashed"),
    # ("k", "dotted"),
    ("k", "dashdot"),
    ("gray", "solid"),
    ("gray", "dashed"),
    ("gray", "dotted"),
    ("gray", "dashdot"),
    ("purple", "solid"),
    ("purple", "dashed"),
    ("purple", "dotted"),
    ("purple", "dashdot"),
]
sorted_data = sorted(data_mcts.items())
# for k, ((batchsize, expl_const), data) in enumerate(sorted_data):
for k, (batchsize, data) in enumerate(sorted_data):
    winners_found = np.zeros((82, len(winners)), dtype=int)
    for i, g in enumerate(data["graph"][1:]):
        for j, w in enumerate(winners):
            label = f"AB::{w}"
            if label in g.nodes:
                payout = g.nodes[label]["reward"] * batchsize
                winners_found[i + 1, j] = int(payout > 5)
    n_winners_found = winners_found.sum(axis=1)

    color, linestyle = formats[k % len(formats)]
    plt.plot(
        samples,
        n_winners_found,
        # label=f"MCTS, nbatch={batchsize}, c={expl_const:.2f}",
        # label=f"MCTS, nbatch={batchsize}",
        label=f"MCTS",
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

# xmin, xmax, ymin, ymax = plt.axis()
# plt.fill(
#     (xmin, xmax, xmax, xmin),
#     (exploited_cutoff, exploited_cutoff, ymax + 1, ymax + 1),
#     lw=0,
#     fc="k",
#     alpha=0.2,
#     zorder=0,
# )
# plt.axis((xmin, xmax, ymin, ymax + 1))

plt.plot(samples, bfs_n_winners_found, label=f"BFS", color="k", linestyle="dotted")

plt.legend(
    title="Search method",
    loc="upper left",
    bbox_to_anchor=(1.05, 0.95),
    # ncol=2,
    # fancybox=True,
    # shadow=True,
)

plt.xlabel(r"$10^5$ Samples")
plt.ylabel("# Solutions found")
plt.tight_layout()

if save:
    save_path = "../figures/tmp/expl_const/bfs_mcts_circuits_v_trials.png"
    print(f"Writing to: {save_path}")
    plt.savefig(save_path)
plt.close()

...


fig = plt.figure(figsize=(5, 3))

# Plot number of circuits found vs number of samples
samples = np.arange(82) / 10

sorted_data = sorted(data_mcts.items())
# for k, ((batchsize, expl_const), data) in enumerate(sorted_data):
for k, (batchsize, data) in enumerate(sorted_data):
    top10_found = np.zeros((82, 10), dtype=int)
    for i, g in enumerate(data["graph"][1:]):
        for j, w in enumerate(top10_winners):
            label = f"AB::{w}"
            if label in g.nodes:
                payout = g.nodes[label]["reward"] * batchsize
                top10_found[i + 1, j] = int(payout >= 5)
    n_top10_found = top10_found.sum(axis=1)

    color, linestyle = formats[k % len(formats)]
    plt.plot(
        samples,
        n_top10_found,
        # label=f"MCTS, nbatch={batchsize}, c={expl_const:.2f}",
        # label=f"MCTS, nbatch={batchsize}",
        label=f"MCTS",
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

plt.legend(
    title="Search method",
    loc="upper left",
    bbox_to_anchor=(1.05, 0.95),
    # ncol=2,
    # fancybox=True,
    # shadow=True,
)
plt.xlabel(r"$10^5$ Samples")
plt.ylabel("# Top-10 Solutions found")
plt.tight_layout()

if save:
    save_path = "../figures/tmp/expl_const/bfs_mcts_top10_v_trials.png"
    print(f"Writing to: {save_path}")
    plt.savefig(save_path)
plt.close()

...


# Plot number of circuits found vs number of samples
samples = np.arange(82) / 10

circuits = np.array(sorted(n for n in data_bfs["graph"][-1].nodes))
n_visits_explored = 100

fig = plt.figure(figsize=(5, 3))
sorted_data = sorted(data_mcts.items())
for k, (batchsize, data) in enumerate(sorted_data):
    explored = np.zeros(82, dtype=int)
    for i, g in enumerate(data["graph"][1:]):
        explored[i + 1] = sum(v > n_visits_explored for _, v in g.nodes("visits"))

    color, linestyle = formats[k % len(formats)]
    plt.plot(
        samples,
        explored,
        # label=f"MCTS, nbatch={batchsize}, c={expl_const:.2f}",
        # label=f"MCTS, nbatch={batchsize}",
        label=f"MCTS",
        color=color,
        linestyle=linestyle,
        linewidth=1,
    )

bfs_batchsize = 10000
bfs_explored = np.arange(82)
plt.plot(samples, bfs_explored, label=f"BFS", color="k", linestyle="dotted")

plt.legend(
    title="Search method",
    loc="upper left",
    bbox_to_anchor=(1.05, 0.95),
    # ncol=2,
    # fancybox=True,
    # shadow=True,
)

# xmin, xmax, ymin, ymax = plt.axis()
# plt.fill(
#     (xmin, xmax, xmax, xmin),
#     (explored_cutoff, explored_cutoff, ymax, ymax),
#     lw=0,
#     fc="k",
#     alpha=0.2,
#     zorder=0,
# )
# plt.axis((xmin, xmax, ymin, ymax))

plt.xlabel(r"$10^5$ Samples")
plt.ylabel("# Circuits with >100 samples")
plt.tight_layout()

if save:
    save_path = "../figures/tmp/expl_const/bfs_mcts_exploration_v_trials.png"
    print(f"Writing to: {save_path}")
    plt.savefig(save_path)
plt.close()

...

markers = [
    # ("", ""),
    ("o", "dashed"),
    ("s", "dotted"),
    ("D", "dashdot"),
    ("^", "dashed"),
    ("", ""),
]
# skip = 5
cmap = plt.get_cmap("turbo")
colors = cmap(np.linspace(0, 1, len(samples)))

fig = plt.figure(figsize=(4, 4))

sorted_data = sorted(data_mcts.items())
for k, (batchsize, data) in enumerate(sorted_data):
    if batchsize not in (1, 5, 10, 25):
        continue
    explored = np.zeros(82, dtype=int)
    exploited = np.zeros(82, dtype=int)
    for i, g in enumerate(data["graph"][1:]):
        explored[i + 1] = sum(v > n_visits_explored for _, v in g.nodes("visits"))
        exploited[i + 1] = sum(
            (r * batchsize) > 5
            for n, r in g.nodes("reward")
            if n.split("::")[-1] in winners
            # if n.split("::")[-1] in top10_winners
        )
    # when_complete = (explored > explored_cutoff) & (exploited > exploited_cutoff)
    # end_idx = 1 + min(when_complete.nonzero()[0], default=len(explored))
    end_idx = len(samples)

    mstyle, lstyle = markers[k % len(formats)]
    plt.plot(
        explored[:end_idx],
        exploited[:end_idx],
        linestyle=lstyle,
        color="k",
        # label=f"MCTS, nbatch={batchsize}",
        label=f"MCTS",
    )
    plt.scatter(
        explored[:end_idx],
        exploited[:end_idx],
        marker=mstyle,
        color=colors[:end_idx],
        # c=samples[:end_idx],
    )

bfs_batchsize = 10000
bfs_explored = np.arange(82)
bfs_exploited = np.zeros(82, dtype=int)
for i, (n, r) in enumerate(zip(data_bfs["simulated_node"], data_bfs["reward"])):
    if r is None:
        continue
    is_winner = ((r * bfs_batchsize) > 5) and (n.split("::")[-1] in winners)
    bfs_exploited[i] = int(is_winner)
    # is_top10_winner = ((r * bfs_batchsize) > 5) and (n.split("::")[-1] in top10_winners)
    # bfs_exploited[i] = int(is_top10_winner)
bfs_exploited = np.cumsum(bfs_exploited)
bfs_when_complete = (bfs_explored > explored_cutoff) & (
    bfs_exploited > exploited_cutoff
)
bfs_end_idx = len(samples)
# bfs_end_idx = 1 + min(bfs_when_complete.nonzero()[0], default=len(bfs_explored))

plt.plot(
    bfs_explored[:bfs_end_idx],
    bfs_exploited[:bfs_end_idx],
    color="k",
    linestyle="solid",
    label=f"BFS",
)
plt.scatter(
    bfs_explored[:bfs_end_idx],
    bfs_exploited[:bfs_end_idx],
    color=colors[:bfs_end_idx],
    # c=samples[:bfs_end_idx],
    marker="x",
)
plt.colorbar(
    cm.ScalarMappable(mplcolors.Normalize(0, 8.1), cmap=cmap),
    location="top",
    ax=plt.gca(),
)
plt.legend(
    title="Search method",
    # loc="upper left",
    # bbox_to_anchor=(1.05, 0.95),
    # ncol=2,
    # fancybox=True,
    # shadow=True,
)

# plt.fill(
#     (explored_cutoff, 81, 81, explored_cutoff),
#     (exploited_cutoff, exploited_cutoff, 33, 33),
#     lw=0,
#     fc="k",
#     alpha=0.2,
#     zorder=0,
# )
# plt.axis((None, 81, None, 33))

plt.xlabel("Circuits sampled >100 times")
plt.ylabel("Solutions found")
plt.tight_layout()

if save:
    save_path = "../figures/tmp/expl_const/bfs_mcts_exploitation_v_exploration.png"
    print(f"Writing to: {save_path}")
    plt.savefig(save_path)
plt.close()

...
