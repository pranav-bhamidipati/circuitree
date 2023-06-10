from concurrent.futures import as_completed
import h5py
from itertools import cycle, islice, chain, repeat
import json
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from uuid import uuid4

from oscillation import TFNetworkModel, OscillationTree

import ray


@ray.remote
def run_batch_and_save(
    seed,
    genotype,
    batch_size,
    dt,
    nt,
    save_dir,
    init_columns,
    param_names,
    oscillation_thresh,
    tau_leap=False,
):
    print(f"Running {batch_size} samples of genotype: {genotype}")
    model = TFNetworkModel(genotype)
    y_t, pop0s, param_sets, rewards = model.run_ssa_and_get_acf_minima(
        size=batch_size,
        tau_leap=tau_leap,
        freqs=False,
        indices=False,
        seed=seed,
        dt=dt,
        nt=nt,
        abs=True,
    )
    run_results = genotype, y_t, pop0s, param_sets, rewards

    print(f"Saving results for {genotype}")

    save_run(
        run_results=run_results,
        save_dir=save_dir,
        init_columns=init_columns,
        param_names=param_names,
        oscillation_thresh=oscillation_thresh,
    )
    return genotype


def run_bfs_in_batches(
    genotypes: Iterable[str],
    n_workers: int,
    save_dir: Path,
    batch_size: int,
    dt: float,
    nt: int,
    oscillation_thresh: float,
    init_columns: Iterable[str],
    param_names: Iterable[str],
    n_batches: Optional[int] = None,
    tau_leap: bool = False,
    queue_size: Optional[int] = None,
    seed_start: int = 0,
):
    if n_batches is None:
        iter_seeds_and_genotypes = enumerate(cycle(genotypes), start=seed_start)
    else:
        iter_seeds_and_genotypes = enumerate(
            chain.from_iterable(repeat(genotypes, n_batches)), start=seed_start
        )

    if queue_size is None:
        queue_size = 2 * cpu_count(logical=True)

    kw = dict(
        batch_size=batch_size,
        dt=dt,
        nt=nt,
        tau_leap=tau_leap,
        save_dir=save_dir,
        init_columns=init_columns,
        param_names=param_names,
        oscillation_thresh=oscillation_thresh,
    )

    ray.init(
        num_cpus=n_workers,
        _system_config={
            "max_io_workers": 4,
            "local_fs_capacity_threshold": 0.99,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}
            ),
        },
    )

    # Fill the queue with the first set of genotypes
    obj_references = (
        run_batch_and_save.remote(seed, g, **kw)
        for seed, g in islice(iter_seeds_and_genotypes, queue_size)
    )
    futures = {ref.future(): ref for ref in obj_references}
    futures_as_completed = as_completed(futures.keys())
    for completed in futures_as_completed:
        del futures[completed]
        next_seed, next_genotype = next(iter_seeds_and_genotypes, (None, None))
        if next_genotype is not None:
            obj_ref = run_batch_and_save.remote(next_seed, next_genotype, **kw)
            futures[obj_ref.future()] = obj_ref
            futures_as_completed = as_completed(futures.keys())

    ray.shutdown()


def save_run(
    run_results,
    save_dir,
    init_columns,
    param_names,
    oscillation_thresh,
):
    genotype, y_t, pop0s, param_sets, rewards = run_results
    state_dir = Path(save_dir).joinpath(f"state_{genotype.strip('*')}")
    state_dir.mkdir(exist_ok=True)

    save_results(
        state_dir, genotype, rewards, pop0s, param_sets, init_columns, param_names
    )

    if np.any(rewards > oscillation_thresh):
        pop_data_dir = Path(save_dir).joinpath(f"extras")
        pop_data_dir.mkdir(exist_ok=True)
        save_pop_data(
            pop_data_dir,
            genotype,
            y_t,
            pop0s,
            param_sets,
            rewards,
            init_columns,
            param_names,
            oscillation_thresh,
        )

    return genotype


def save_results(
    state_dir,
    genotype,
    rewards,
    pop0s,
    param_sets,
    init_columns,
    param_names,
    ext="parquet",
):
    data = (
        dict(state=genotype, reward=rewards)
        | dict(zip(init_columns, np.atleast_2d(pop0s).T))
        | dict(zip(param_names, np.atleast_2d(param_sets).T))
    )
    df = pd.DataFrame(data)
    df["state"] = df["state"].astype("category")

    fname = state_dir.joinpath(f"{uuid4()}.{ext}").resolve().absolute()
    print(f"Writing to: {fname}")
    if ext == "csv":
        df.to_csv(fname, index=False)
    elif ext == "parquet":
        df.to_parquet(fname, index=False)


def save_pop_data(
    pop_data_dir,
    genotype,
    y_t,
    pop0s,
    param_sets,
    rewards,
    init_columns,
    param_names,
    thresh,
):
    save_idx = np.where(rewards > thresh)[0]
    data = (
        dict(state=genotype, reward=rewards[save_idx])
        | dict(zip(init_columns, np.atleast_2d(pop0s[save_idx]).T))
        | dict(zip(param_names, np.atleast_2d(param_sets[save_idx]).T))
    )
    df = pd.DataFrame(data)

    state_no_asterisk = genotype.strip("*")
    fname = pop_data_dir.joinpath(f"state_{state_no_asterisk}_ID#{uuid4()}.hdf5")
    fname = fname.resolve().absolute()
    print(f"Writing population data to: {fname}")
    with h5py.File(fname, "w") as f:
        f.create_dataset("y_t", data=y_t[save_idx])
    df.to_hdf(fname, key="metadata", mode="a", format="table")


def main(
    batch_size: int = 100,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    n_samples: Optional[int] = None,
    n_workers: Optional[int] = None,
    save_dir: Path = Path("data/oscillation/bfs"),
    oscillation_thresh: float = 0.35,
    tau_leap: bool = False,
    queue_size: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
):
    init_columns = ["A_0", "B_0", "C_0"]
    param_names = [
        "k_on",
        "k_off_1",
        "k_off_2",
        "km_unbound",
        "km_act",
        "km_rep",
        "km_act_rep",
        "kp",
        "gamma_m",
        "gamma_p",
    ]
    components = ["A", "B", "C"]
    interactions = ["activates", "inhibits"]
    root = "ABC::"
    tree = OscillationTree(
        components=components,
        interactions=interactions,
        root=root,
        dt=dt_seconds,
        nt=nt,
    )
    tree.grow_tree()

    # Split up the BFS into rounds
    # In each round, we will run n_samples simulations for each genotype.
    # Samples will be run and results saved in batches of size ``save_every``.
    # This ensures that workers with shorter simulations can steal work periodically.
    if n_workers is None:
        n_workers = cpu_count(logical=True)

    # Cycle through nodes in BFS order, taking n_batches_per_cycle batches of samples
    # from each node. n_cycles is set by balancing two factors. More cycles (fewer
    # samples per cycle) allows us to gradually accumulate data on all genotypes, rather
    # than one-by-one. However, it also means that for every cycle, we will end up JIT-
    # compiling the models again.
    bfs_arr = np.array([n for n in tree.bfs_iterator() if tree.is_terminal(n)])
    if shuffle_seed is not None:
        rg = np.random.default_rng(shuffle_seed)
        rg.shuffle(bfs_arr)

    # Run indefinitely
    if n_samples is None:
        bfs = cycle(bfs_arr.tolist())
        n_batches = None
        print(
            f"Using {n_workers} workers to sample each genotype in batches of size "
            f"{batch_size}. Sampling will continue indefinitely."
        )

    # Or run until a fixed number of samples
    else:
        n_batches, mod = divmod(n_samples, batch_size)
        if mod != 0:
            raise ValueError(
                f"n_samples ({n_samples}) must be divisible by batch_size ({batch_size})"
            )
        bfs = chain.from_iterable(repeat(bfs_arr.tolist(), n_batches))
        print(
            f"Using {n_workers} workers to make {n_samples} samples for each genotype "
            f"({n_batches} batches of size {batch_size})."
        )

    run_bfs_in_batches(
        genotypes=bfs,
        n_workers=n_workers,
        save_dir=save_dir,
        batch_size=batch_size,
        dt=dt_seconds,
        nt=nt,
        oscillation_thresh=oscillation_thresh,
        init_columns=init_columns,
        param_names=param_names,
        n_batches=n_batches,
        tau_leap=tau_leap,
        queue_size=queue_size,
    )


if __name__ == "__main__":
    save_dir = Path("data/oscillation/bfs")
    save_dir.mkdir(exist_ok=True)
    main(
        # n_samples=10000,
        batch_size=2,
        nt=2000,
        dt_seconds=20.0,
        n_workers=4,
        save_dir=save_dir,
        oscillation_thresh=0.35,
        tau_leap=False,
        shuffle_seed=2023,
    )
