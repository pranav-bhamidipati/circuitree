from concurrent.futures import as_completed
from itertools import islice
import json
from math import ceil
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from uuid import uuid4


from models.oscillation.oscillation import TFNetworkModel, OscillationTreeBase

import ray


@ray.remote
class Model(object):
    def __init__(
        self,
        model=None,
        init_columns=None,
        param_names=None,
        save_dir: Path = None,
        **kwargs,
    ):
        self.model = model or TFNetworkModel(**kwargs)
        self.genotype = self.model.genotype
        self.init_columns = init_columns
        self.param_names = param_names
        self.state_dir = Path(save_dir).joinpath(f"state_{self.genotype.strip('*')}")

        self.state_dir.mkdir(exist_ok=True)

    def initialize_ssa(self, *args, **kwargs):
        self.model.initialize_ssa(*args, **kwargs)
        print(f"Initialized SSA for {self.genotype}")

    def run_batch(self, n: int, save: bool = True):
        if self.model.ssa is None:
            self.initialize_ssa()

        print(f"Running batch")
        pop0s, param_sets, extrema = self.model.run_batch_job(n)
        print(f"Finished batch")

        rewards = np.abs(extrema)

        if save:
            self.save_results(pop0s, param_sets, rewards)

        return self.genotype, pop0s, param_sets, rewards

    def save_results(self, pop0s, param_sets, rewards):
        data = (
            dict(state=self.genotype, reward=rewards)
            | dict(zip(self.init_columns, np.atleast_2d(pop0s).T))
            | dict(zip(self.param_names, np.atleast_2d(param_sets).T))
        )
        df = pd.DataFrame(data)
        df["state"] = df["state"].astype("category")

        csv = self.state_dir.joinpath(f"{uuid4()}.csv").resolve().absolute()
        if csv.exists():
            raise FileExistsError(f"{csv} already exists")

        print(f"Writing to: {csv}")
        df.to_csv(csv, index=False)


def main(
    n_samples: int = 10000,
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
    tree = OscillationTreeBase(
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

    n_batches = ceil(n_samples / batchsize)

    bfs = (n for n in tree.bfs_iterator() if tree.is_terminal(n))
    bfs = list(bfs)[: n_workers * 2]  # For testing...

    def get_jobs():
        def bfs_iter():
            for genotype in bfs:
                for _ in range(n_batches):
                    yield genotype

        return bfs_iter()

    print(
        f"Using {n_workers} workers to make {n_samples} samples for each genotype "
        f"in {n_batches} batches of size {batchsize}"
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

    jobs = get_jobs()
    jobs_to_do = list(islice(jobs, 4))
    models = {
        g: Model.remote(genotype=g, initialize=True, dt=dt_seconds, nt=nt, **kw)
        for g in set(jobs_to_do)
    }
    sim_refs = [models[g].run_batch.remote(batchsize) for g in jobs_to_do]
    futures = [ref.future() for ref in sim_refs]
    futures_as_completed = as_completed(futures)
    while jobs_to_do or futures_as_completed:
        print(f"To do: {jobs_to_do}")
        print(f"Models: {list(models.keys())}")
        print(f"n_futures: {list(models.keys())}")

        completed = next(futures_as_completed)
        completed_g, *_ = completed.result()

        print(f"Completed: {completed_g}")

        jobs_to_do.remove(completed_g)

        # If we don't need this genotype anymore, delete it from the models
        if completed_g not in jobs_to_do:
            del models[completed_g]

        next_g = next(jobs, None)
        if next_g is not None:
            jobs_to_do.append(next_g)
            if next_g not in models:
                models[next_g] = Model.remote(
                    genotype=next_g, initialize=True, dt=dt_seconds, nt=nt, **kw
                )

            sim_ref = models[next_g].run_batch.remote(batchsize)

            futures.remove(completed)
            futures.append(sim_ref.future())
            futures_as_completed = as_completed(futures)


if __name__ == "__main__":
    save_dir = Path("data/oscillation/tmp")
    save_dir.mkdir(exist_ok=True)

    main(
        n_samples=10,
        batchsize=5,
        # n_workers=1,
        # n_workers=4,
        save_dir=save_dir,
    )
