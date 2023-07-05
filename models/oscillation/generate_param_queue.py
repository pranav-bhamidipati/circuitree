from typing import Optional
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from gillespie import (
    SAMPLING_RANGES,
    PARAM_NAMES,
    MEAN_INITIAL_POPULATION,
    convert_uniform_to_params,
    draw_random_initial_protein,
)


def main(
    n_samples: int,
    save_dir: Path,
    n_components: int = 3,
    seed: int = 2023,
    save: bool = False,
):
    n_params = len(SAMPLING_RANGES)
    rg = np.random.default_rng(seed)
    lh_sampler = LatinHypercube(n_params, seed=rg)
    uniform_samples = lh_sampler.random(n_samples)
    param_sets = np.array(
        [convert_uniform_to_params(u, SAMPLING_RANGES) for u in uniform_samples]
    )
    initial_conditions = rg.poisson(MEAN_INITIAL_POPULATION, size=(n_samples, n_components))
    init_columns = [f"{chr(ord('A') + i)}_0" for i in range(n_components)]
    init_data = pd.DataFrame(initial_conditions, columns=init_columns)
    init_data.index.name = "sample_num"
    param_data = pd.DataFrame(param_sets, columns=PARAM_NAMES)
    param_data.index.name = "sample_num"
    data = pd.concat([init_data, param_data], axis=1)
    ...

    if save:
        fname = Path(save_dir) / f"param_sets_queue_{n_samples}.csv"
        fname = fname.resolve().absolute()
        print(f"Writing parameter queue to {fname}")
        data.to_csv(fname)


if __name__ == "__main__":
    save_dir = Path("data/oscillation")
    save_dir.mkdir(exist_ok=True)
    main(
        10000,
        save_dir,
        save=True,
    )
