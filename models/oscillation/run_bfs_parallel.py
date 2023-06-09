from collections import Counter
from concurrent.futures import as_completed
import h5py
from itertools import islice, chain, repeat
import json
from math import ceil
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from uuid import uuid4


# from models.oscillation.oscillation import TFNetworkModel, OscillationTreeBase
from oscillation import TFNetworkModel, OscillationTree

import ray


@ray.remote
class Model(object):
    def __init__(
        self,
        model=None,
        init_columns=None,
        param_names=None,
        save_dir: Path = None,
        initialize: bool = False,
        oscillation_thresh: float = 0.4,
        **kwargs,
    ):
        self.model = model or TFNetworkModel(**kwargs)
        self.genotype = self.model.genotype
        self.init_columns = init_columns
        self.param_names = param_names

        # Directory to save results
        self.state_dir = Path(save_dir).joinpath(f"state_{self.genotype.strip('*')}")
        self.state_dir.mkdir(exist_ok=True)

        # Directory to save all the data for any exceptional runs
        self.extras_dir = Path(save_dir).joinpath(f"extras")
        self.extras_dir.mkdir(exist_ok=True)

        self.oscillation_thresh = oscillation_thresh

        if initialize:
            self.initialize_ssa(**kwargs)

    def initialize_ssa(self, *args, **kwargs):
        print(f"Initializing SSA for {self.genotype}")
        self.model.initialize_ssa(*args, **kwargs)

    def run_batch(
        self, model_idx: int, n: int, save: bool = True, oscillation_thresh=None
    ):
        if self.model.ssa is None:
            self.initialize_ssa()

        print(f"Running batch")
        y_t, pop0s, param_sets, rewards = self.model.run_batch_job(n, abs=True)
        print(f"Finished batch")

        if save:
            self.save_results(pop0s, param_sets, rewards)

            # Save all the data if any of the runs showed oscillation
            thresh = oscillation_thresh or self.oscillation_thresh
            if (rewards > thresh).any():
                print(f"Found a run from {self.genotype} with reward {max(rewards)}!")
                self.save_extra(y_t, pop0s, param_sets, rewards, thresh)

        return model_idx, self.genotype, pop0s, param_sets, rewards

    def save_results(self, pop0s, param_sets, rewards, ext="parquet"):
        data = (
            dict(state=self.genotype, reward=rewards)
            | dict(zip(self.init_columns, np.atleast_2d(pop0s).T))
            | dict(zip(self.param_names, np.atleast_2d(param_sets).T))
        )
        df = pd.DataFrame(data)
        df["state"] = df["state"].astype("category")

        fname = self.state_dir.joinpath(f"{uuid4()}.{ext}").resolve().absolute()
        print(f"Writing to: {fname}")
        if ext == "csv":
            df.to_csv(fname, index=False)
        elif ext == "parquet":
            df.to_parquet(fname, index=False)

    def save_extra(self, y_t, pop0s, param_sets, rewards, thresh):
        save_idx = np.where(rewards > thresh)[0]
        data = (
            dict(state=self.genotype, reward=rewards[save_idx])
            | dict(zip(self.init_columns, np.atleast_2d(pop0s[save_idx]).T))
            | dict(zip(self.param_names, np.atleast_2d(param_sets[save_idx]).T))
        )
        df = pd.DataFrame(data)

        state_no_asterisk = self.genotype.strip("*")
        fname = self.extras_dir.joinpath(f"state_{state_no_asterisk}_ID#{uuid4()}.hdf5")
        fname = fname.resolve().absolute()
        print(f"Writing extra data to: {fname}")
        with h5py.File(fname, "w") as f:
            f.create_dataset("y_t", data=y_t[save_idx])
        df.to_hdf(fname, key="metadata", mode="a", format="table")


def main(
    n_samples: int = 10000,
    n_cycles: int = 100,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    batchsize: int = 100,
    n_workers: Optional[int] = None,
    save_dir: Path = Path("data/oscillation"),
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
    kw = dict(init_columns=init_columns, param_names=param_names, save_dir=save_dir)
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

    n_batches_per_cycle = ceil(n_samples / batchsize / n_cycles)

    # Cycle through nodes in BFS order, taking n_batches_per_cycle batches of samples
    # from each node. n_cycles is set by balancing two factors. More cycles (fewer
    # samples per cycle) allows us to gradually accumulate data on all genotypes, rather
    # than one-by-one. However, it also means that for every cycle, we will end up JIT-
    # compiling the models again.
    bfs = (n for n in tree.bfs_iterator() if tree.is_terminal(n))
    bfs = chain.from_iterable(repeat(bfs, n_cycles))
    job_counter = Counter({g: n_batches_per_cycle for g in islice(bfs, n_workers)})

    print(
        f"Using {n_workers} workers to make {n_samples} samples for each genotype "
        f"in {n_batches_per_cycle} batches of size {batchsize} over {n_cycles} cycles."
    )

    ray.init(
        _system_config={
            "max_io_workers": 4,
            "local_fs_capacity_threshold": 0.99,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/tmp/ray_spill"}}
            ),
        }
    )

    models = {
        g: [Model.remote(genotype=g, initialize=True, dt=dt_seconds, nt=nt, **kw)]
        for g in job_counter
    }
    n_models_available = sum(len(m) for m in models.values())
    print(f"Num. workers: {n_workers}")
    print(f"Num. models available to run: {n_models_available}")
    print(f"Num. unique models being simulated: {len(job_counter)}")

    sim_refs = [models[g][0].run_batch.remote(0, batchsize) for g in job_counter]
    futures = [ref.future() for ref in sim_refs]
    futures_as_completed = as_completed(futures)
    while (sum(job_counter.values()) > 0) or futures:
        n_models_available = sum(len(m) for m in models.values())
        print(f"Num. unique models being simulated: {len(job_counter)}")
        print(
            f"Currently {n_workers} workers running {n_models_available} models "
            f"({len(job_counter)} unique)"
        )
        print(f"\t-> est. {min(n_workers / n_models_available, 1):.2%} CPU utilization")

        completed = next(futures_as_completed)
        futures.remove(completed)
        model_idx, completed_g, *_ = completed.result()

        # If this was the last simulation for this genotype, launch the next one
        job_counter[completed_g] -= 1
        if job_counter[completed_g] == 0:
            print(f"Removing model: {completed_g}")
            del job_counter[completed_g]
            del models[completed_g]

            new_g = next(bfs, None)
            if new_g is not None:
                print(f"Launching model: {new_g}")
                job_counter[new_g] = n_batches_per_cycle
                models[new_g] = [
                    Model.remote(
                        genotype=new_g, initialize=True, dt=dt_seconds, nt=nt, **kw
                    )
                ]
                next_g = new_g
                next_model_idx = 0

            # If there are no new topologies left to simulate, add models to help wrap up
            # the pending ones
            else:
                job_to_worker_ratio = {
                    g: job_counter[g] / len(models[g]) for g in models
                }
                highest_jwr_g = max(job_to_worker_ratio, key=job_to_worker_ratio.get)
                print(f"Launching duplicate model: {highest_jwr_g}")
                models[highest_jwr_g].append(
                    Model.remote(
                        genotype=highest_jwr_g,
                        initialize=True,
                        dt=dt_seconds,
                        nt=nt,
                        **kw,
                    )
                )
                next_g = highest_jwr_g
                next_model_idx = len(models[highest_jwr_g]) - 1

        else:
            next_g = completed_g
            next_model_idx = model_idx

        next_model = models[next_g][next_model_idx]
        sim_ref = next_model.run_batch.remote(next_model_idx, batchsize)
        futures.append(sim_ref.future())
        futures_as_completed = as_completed(futures)

    ray.shutdown()


if __name__ == "__main__":
    save_dir = Path("data/oscillation/bfs")
    save_dir.mkdir(exist_ok=True)
    main(
        n_samples=10_000,
        n_cycles=100,
        batchsize=50,
        save_dir=save_dir,
    )
