from datetime import date
from itertools import product
from pathlib import Path
import pickle
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from circuitree import Circuit, CircuiTree

components = ["A", "B"]
interactions = ["activates", "inhibits"]
polar_df = pd.read_csv("polarization2_data.csv")

# winners = polar_df.genotype.values
# cores = winners[polar_df.core]
win_probabilities = dict(zip(polar_df.genotype.values, polar_df.Q.values))

# Two components, no interactions
root = "AB::"


def run_bfs(
    n_samples_per_circuit: int = 10000,
    shuffle: bool = True,
    seed: int = 2023,
):
    method = "bfs"
    bfs = CircuiTree(
        components,
        interactions,
        win_probabilities=win_probabilities,
        root=root,
        search_method=method,
        seed=seed,
    )
    results = bfs.search_bfs(
        n_samples_per_circuit,
        metric_func=bfs.complexity_graph_from_bfs,
        shuffle=shuffle,
    )

    graphs, simulated_nodes, rewards = zip(*results)
    data = {
        "graph": graphs,
        "simulated_node": simulated_nodes,
        "reward": rewards,
        "method": method,
        "shuffle": shuffle,
        "seed": seed,
    }

    today = date.today()
    p = Path(f"../data/{today}_polarization_bfs/")
    p.mkdir(exist_ok=True)
    fname = f"{today}_polarization_{method}_seed{seed}.pickle"
    p = p.joinpath(fname)

    print(f"Writing to: {p.resolve().absolute()}")
    pickle.dump(data, p.open("wb"))


def main(
    samples_per_circuit: Optional[Iterable[int]] = None,
    shuffle_opts: Optional[Iterable[bool]] = None,
    seeds: Optional[Iterable[int]] = None,
    threads: int = None,
):
    import psutil
    import multiprocessing as mp

    if samples_per_circuit is None:
        samples_per_circuit = [10000]

    if shuffle_opts is None:
        shuffle_opts = [True]

    if seeds is None:
        seeds = [2023]

    settings = list(product(samples_per_circuit, shuffle_opts, seeds))

    if threads is None:
        threads = np.inf
    n_threads = min(threads, psutil.cpu_count(logical=True), len(settings))
    print(f"Assembling thread pool ({n_threads} workers)")

    pool = mp.Pool(n_threads)
    _ = pool.starmap(run_bfs, settings)
    pool.close()
    pool.join()
    print("Complete")


if __name__ == "__main__":
    # run_bfs(
    #     shuffle=True,
    #     seed=2023,
    # )
    main(
        seeds=np.arange(25),
        # threads=9,
    )
