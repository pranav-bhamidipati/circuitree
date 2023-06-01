from collections import Counter
from concurrent.futures import as_completed
from itertools import islice
import json
from math import ceil
from more_itertools import chunked
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from uuid import uuid4


# from models.oscillation.oscillation import TFNetworkModel, OscillationTreeBase
from oscillation import TFNetworkModel, OscillationTreeBase

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
        **kwargs,
    ):
        self.model = model or TFNetworkModel(**kwargs)
        self.genotype = self.model.genotype
        self.init_columns = init_columns
        self.param_names = param_names
        self.state_dir = Path(save_dir).joinpath(f"state_{self.genotype.strip('*')}")

        self.state_dir.mkdir(exist_ok=True)

        if initialize:
            self.initialize_ssa(**kwargs)

    def initialize_ssa(self, *args, **kwargs):
        print(f"Initializing SSA for {self.genotype}")
        self.model.initialize_ssa(*args, **kwargs)

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

    def save_results(self, pop0s, param_sets, rewards, ext="parquet"):
        data = (
            dict(state=self.genotype, reward=rewards)
            | dict(zip(self.init_columns, np.atleast_2d(pop0s).T))
            | dict(zip(self.param_names, np.atleast_2d(param_sets).T))
        )
        df = pd.DataFrame(data)
        df["state"] = df["state"].astype("category")

        fname = self.state_dir.joinpath(f"{uuid4()}.{ext}").resolve().absolute()
        if fname.exists():
            raise FileExistsError(f"{fname} already exists")

        print(f"Writing to: {fname}")
        if ext == "csv":
            df.to_csv(fname, index=False)
        elif ext == "parquet":
            df.to_parquet(fname, index=False)


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
    job_counter = Counter({g: n_batches for g in islice(bfs, n_workers)})

#     def update_job_counter(completed_g):
#         job_counter[completed_g] -= 1
#         next_g = None
#         if job_counter[completed_g] == 0:
#             del job_counter[completed_g]
#             next_g = next(bfs, None)
#             if next_g is not None:
#                 job_counter[next_g] = n_batches
#         return next_g

#     def get_jobs():
#         def bfs_iter():
#             bfs = (n for n in tree.bfs_iterator() if tree.is_terminal(n))
#             for genotypes in chunked(bfs, n_workers):
#                 for _ in range(n_batches):
#                     for g in genotypes:
#                         yield g
#         return bfs_iter()

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

    models = {
        g: [Model.remote(genotype=g, initialize=True, dt=dt_seconds, nt=nt, **kw)]
        for g in job_counter
    }
    sim_refs = [models[g][0].run_batch.remote(batchsize) for g in job_counter]
    futures = [ref.future() for ref in sim_refs]
    futures_as_completed = as_completed(futures)
    print(f"To do: {job_counter}")
    while (sum(job_counter.values()) > 0) or futures:
        
        completed = next(futures_as_completed)
        futures.remove(completed)
        completed_g, *_ = completed.result() 
        
        # If this was the last simulation for this genotype, launch the next one 
        job_counter[completed_g] -= 1
        if job_counter[completed_g] == 0:
            print(f"Removing model: {completed_g}")
            del job_counter[completed_g]
            del models[completed_g]

            new_g = next(bfs, None)
            if new_g is not None:
                print(f"Launching model: {new_g}")
                job_counter[new_g] = n_batches
                models[new_g] = [Model.remote(
                    genotype=new_g, initialize=True, dt=dt_seconds, nt=nt, **kw
                )]
                next_g = new_g
            else:
                # If all topologies have been added to the counter, spawn a new
                # actor to help wrap up the existing topologies
                job_to_worker_ratio = {g: job_counter[g] / len(models[g]) for g in models}
                highest_g = max(job_to_worker_ratio, key=job_to_worker_ratio.get)
                print(f"Launching additional model: {highest_g}")
                models[highest_g].append(Model.remote(
                    genotype=highest_g, initialize=True, dt=dt_seconds, nt=nt, **kw
                ))
                next_g = highest_g

        else:
            next_g = completed_g

        next_ready_model = ray.wait(models[next_g])[0][0]
        
        sim_ref = next_ready_model.run_batch.remote(batchsize)
        futures.append(sim_ref.future())
        futures_as_completed = as_completed(futures)

    ray.shutdown()


if __name__ == "__main__":
    save_dir = Path("data/oscillation/bfs")
    save_dir.mkdir(exist_ok=True)

    main(
        n_samples=10_000,
        batchsize=40,
        save_dir=save_dir,
    )
