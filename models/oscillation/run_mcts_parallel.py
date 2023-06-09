import json
from typing import Optional
import numpy as np
from pathlib import Path
from psutil import cpu_count
import ray

from models.oscillation.oscillation_parallel import (
    TranspositionTable,
    OscillationTreeParallel,
)


def main(
    save_dir: Path,
    n_iterations=100_000,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    n_workers: Optional[int] = None,
    save_every: int = 25,
):
    save_dir = Path(save_dir)
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

    if n_workers is None:
        n_workers = cpu_count(logical=True)

    trans_table = TranspositionTable()
    tree = OscillationTreeParallel(
        components=components,
        interactions=interactions,
        root=root,
        dt=dt_seconds,
        nt=nt,
        results_table=trans_table,
        batch_size=n_workers,
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

    df_kwargs = dict(init_columns=init_columns, param_names=param_names)
    print(f"Running {n_iterations} iterations with {n_workers} workers.")
    for step in range(n_iterations):
        print(f"Iteration {step + 1} / {n_iterations}")
        _ = tree.traverse()
        if step % save_every == 0:
            df = tree.results_table.to_df(**df_kwargs)
            fpath = save_dir.joinpath(f"results_{step}.csv").resolve().absolute()
            print(f"Saving results at step {step} to {fpath.name}")
            df.to_csv(fpath, index=False)

    ray.shutdown()


if __name__ == "__main__":
    save_dir = Path("data/oscillation/mcts/230608")
    main(
        save_dir=save_dir,
        n_iterations=10,
        save_every=1,
        n_workers=16,
    )
