from functools import partial
import h5py
from math import ceil
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
from sacred import Experiment

from circuitree.rewards import *

from .search_parallel import OscillationTree


def search_sequential(
    components: Iterable[Iterable[str]],
    interactions: Iterable[str],
    time_points: np.ndarray[np.float64],
    n_samples_per_topology: int,
    estimate_modularity: bool = False,
    success_threshold: float = 0.005,
    max_samples: int = 10000000,
    seed: Optional[int] = None,
    rg: Optional[np.random.Generator] = None,
    **kwargs,
):
    if rg is None:
        if seed is None:
            raise ValueError("Must specify random seed if rg is not specified")
        else:
            rg = np.random.default_rng(seed)
    else:
        if seed is None:
            seed = rg.bit_generator._seed_seq.entropy

    ot = OscillationTree(
        components=components,
        interactions=interactions,
        time_points=time_points,
        batch_size=1,
        success_threshold=success_threshold,
        **kwargs,
    )

    if estimate_modularity:
        metric_func = partial(sequential_reward_and_modularity_estimate, ot.root)

        results = ot.search_bfs(
            1,
            n_samples_per_topology,
            max_steps=max_samples,
            metric_func=metric_func,
            shuffle=True,
        )

        rewards, modularity_estimates = zip(*results[1:])
        rewards = np.array(rewards, dtype=int)
        modularity_estimates = np.array(modularity_estimates, dtype=float)
        data = {"modularity_estimates": modularity_estimates}

    else:
        metric_func = sequential_reward

        rewards = ot.search_bfs(
            1,
            n_samples_per_topology,
            max_steps=max_samples,
            metric_func=metric_func,
            shuffle=True,
        )[1:]

        data = {}

    data["n_per_topology"] = n_samples_per_topology
    data["N"] = max_samples
    data["seed"] = seed
    data["rewards"] = rewards
    data["final_modularity"] = ot.modularity

    return data


def search_mcts(
    components: Iterable[Iterable[str]],
    interactions: Iterable[str],
    time_points: np.ndarray[np.float64],
    N: int,
    batch_size: int = 5,
    estimate_modularity: bool = False,
    success_threshold: float = 0.005,
    seed: Optional[int] = None,
    rg: Optional[np.random.Generator] = None,
    root: str = "ABC::",
    **kwargs,
):
    if rg is None:
        if seed is None:
            raise ValueError("Must specify random seed if rg is not specified")
        else:
            rg = np.random.default_rng(seed)
    else:
        if seed is None:
            seed = rg.bit_generator._seed_seq.entropy

    ot = OscillationTree(
        components=components,
        interactions=interactions,
        time_points=time_points,
        batch_size=batch_size,
        success_threshold=success_threshold,
        root=root,
        **kwargs,
    )

    n_iterations = ceil(N / batch_size)

    if estimate_modularity:
        metric_func = partial(mcts_reward_and_modularity_estimate, ot.root)
        results = ot.search_mcts(n_iterations, metric_func=metric_func)[1:]

        rewards, modularity_estimates = zip(*results)
        rewards = np.array(rewards, dtype=int)
        modularity_estimates = np.array(modularity_estimates, dtype=float)

        data = {"modularity_estimates": modularity_estimates}

    else:
        metric_func = mcts_reward
        rewards = ot.search_mcts(n_iterations, metric_func=metric_func)[1:]
        data = {}

    data["N"] = N
    data["batch_size"] = batch_size
    data["seed"] = seed
    data["rewards"] = rewards
    data["final_modularity"] = ot.modularity

    return data


def oscillator_search(
    components: Iterable[Iterable[str]],
    interactions: Iterable[str],
    N: int,
    nt: int,
    dt: float = 30.0,  # seconds
    n_samples_per_topology: int = 100,
    method: Literal["mcts", "sequential"] = "mcts",
    seed: int = 2023,
    save: bool = False,
    estimate_modularity: bool = False,
    ex: Optional[Experiment] = None,
    **kwargs,
):
    if method == "mcts":
        search = search_mcts
    elif method == "sequential":
        search = search_sequential
    else:
        raise ValueError(f"Unknown search method: {method}")

    rg = np.random.default_rng(seed)

    t = np.arange(0, nt * dt, nt)
    data = search(
        components,
        interactions,
        t,
        N=N,
        n_samples_per_topology=n_samples_per_topology,
        rg=rg,
        estimate_modularity=estimate_modularity,
        **kwargs,
    )

    if ex is not None:
        artifacts = []

        save_dir = Path(ex.observers[0].dir)

        if save:
            p = save_dir.joinpath("results.hdf5")
            # print(f"Writing to: {p.resolve().absolute()}")

            with h5py.File(p, "w") as f:
                for k, v in data.items():
                    f.create_dataset(k, data=v)

            artifacts.append(p)

        for a in artifacts:
            ex.add_artifact(a)
