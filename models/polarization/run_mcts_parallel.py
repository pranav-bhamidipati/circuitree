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


def run_mcts(
    batch_size: int,
    c: float = np.sqrt(2),
    d: float = 10.0,
    method: str = "mcts",
    n_steps_total: int = 810000,
    sample_depth: int = 10000,
):
    mcts = CircuiTree(
        components,
        interactions,
        win_probabilities=win_probabilities,
        root=root,
        exploration_constant=c,
        variance_constant=d,
        batch_size=batch_size,
        tree_shape="dag",
        search_method=method,
    )

    n_steps = n_steps_total // batch_size
    save_every = sample_depth // batch_size

    results_mcts = mcts.search_mcts(
        n_steps, save_every=save_every, metric_func=mcts.complexity_graph_from_mcts
    )

    graphs, selection_paths, simulated_nodes, reward = zip(*results_mcts)
    data = {
        "graph": graphs,
        "selection_path": selection_paths,
        "simulated_node": simulated_nodes,
        "reward": reward,
        "batch_size": batch_size,
        "exploration_constant": c,
        "variance_constant": d,
        "method": method,
    }

    if method.lower() == "sp-mcts":
        method_and_params = f"{method}_c{c:.2f}_d{d:.2f}"
    elif method.lower() == "mcts":
        method_and_params = f"{method}_c{c:.2f}"

    today = date.today()
    p = Path(f"../data/{today}_mcts_polarization/")
    p.mkdir(exist_ok=True)
    fname = f"{today}_polarization_{method_and_params}_batch{batch_size}.pickle"
    p = p.joinpath(fname)
    print(f"Writing to: {p.resolve().absolute()}")
    pickle.dump(data, p.open("wb"))


def main(
    batch_sizes: Iterable[int],
    exploration_constants: Optional[Iterable[float]] = None,
    variance_constants: Optional[Iterable[float]] = None,
    methods: Optional[Iterable[str]] = None,
    threads: int = None,
):
    import psutil
    import multiprocessing as mp

    if exploration_constants is None:
        exploration_constants = [np.sqrt(2)]

    if variance_constants is None:
        variance_constants = [0.0]

    if methods is None:
        methods = ["mcts"]

    settings = list(
        product(batch_sizes, exploration_constants, variance_constants, methods)
    )

    if threads is None:
        threads = np.inf
    n_threads = min(threads, psutil.cpu_count(logical=True), len(settings))
    print(f"Assembling thread pool ({n_threads} workers)")

    pool = mp.Pool(n_threads)
    _ = pool.starmap(run_mcts, settings)
    pool.close()
    pool.join()
    print("Complete")


if __name__ == "__main__":
    # run_mcts(
    #     batch_size=25,
    # )
    main(
        batch_sizes=(1, 5, 10, 25, 100),
        # exploration_constants=np.linspace(0, 2, 11),
        # variance_constants=(0, 0.5, 1, 2, 5, 10),
        # methods=["sp-mcts"],
        # threads=9,
    )
