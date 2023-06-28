from pathlib import Path
import pandas as pd
from functools import partial
from time import perf_counter
import numpy as np
import multiprocessing as mp
from psutil import cpu_count
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
from gillespie import (
    GillespieSSA,
    make_matrices_for_ssa,
    SAMPLING_RANGES,
    DEFAULT_PARAMS,
    SAMPLED_VAR_NAMES,
    _normalize,
    _rescale,
    convert_params_to_sampled_quantities,
    convert_uniform_to_params,
)


def run_gillespie_random_params(idx_params, pop0, ssa: GillespieSSA):
    idx, params = idx_params
    start = perf_counter()
    ssa.run_with_params(pop0, params)
    end = perf_counter()
    return idx, end - start


def main(
    save_dir: Path,
    n_samples,
    n_replicates=1,
    n_threads=None,
    seed=2023,
    save: bool = False,
):
#     # Activator-inhibitor circuit
#     # 2 components A and B. AAa_ABa_BAi
#     nc = 2
#     Am, Rm, U = make_matrices_for_ssa(
#         nc, activations=[[0, 0], [0, 1]], inhibitions=[[1, 0]]
#     )
# 
#     ssa = GillespieSSA(
#         seed=0,
#         n_species=2,
#         activation_mtx=Am,
#         inhibition_mtx=Rm,
#         update_mtx=U,
#         dt=20.0,
#         nt=100,
#         mean_mRNA_init=10.0,
#         SAMPLING_RANGES=SAMPLING_RANGES,
#         DEFAULT_PARAMS=DEFAULT_PARAMS,
#     )
# 

    # Activator-inhibitor-quencher circuit
    # 3 components A and B. AAa_ABa_BAi_CBi
    nc = 3
    Am, Rm, U = make_matrices_for_ssa(
        nc, activations=[[0, 0], [0, 1]], inhibitions=[[1, 0], [2, 1]]
    )
    ssa = GillespieSSA(
        seed=0,
        n_species=nc,
        activation_mtx=Am,
        inhibition_mtx=Rm,
        update_mtx=U,
        dt=20.0,
        nt=2000,
        mean_mRNA_init=10.0,
        DEFAULT_PARAMS=DEFAULT_PARAMS,
        SAMPLING_RANGES=SAMPLING_RANGES,
    )
    
    pop0 = np.zeros(ssa.n_species, dtype=np.int64)
    pop0[nc : nc * 2] = 10

    if n_threads is None:
        n_threads = cpu_count(logical=True)
    else:
        n_threads = min(n_threads, cpu_count(logical=True))
    print(f"Assembling pool of {n_threads} processes")

    n_params = len(SAMPLING_RANGES)
    rg = np.random.default_rng(seed)
    lh_sampler = LatinHypercube(n_params, seed=rg)
    uniform_samples = lh_sampler.random(n_samples)
    param_sets = np.array(
        [convert_uniform_to_params(u, SAMPLING_RANGES) for u in uniform_samples]
    )
    replicates = np.repeat(np.arange(n_replicates), n_samples)
    inputs = list(enumerate(np.tile(param_sets, (n_replicates, 1))))

    run_one_param_set = partial(run_gillespie_random_params, ssa=ssa, pop0=pop0)

    
    def _init_thread():
        print("Initializing...")
        run_one_param_set(inputs[0])
        print("Ready!")
    # Processes should be initialized by first running an SSA to JIT-compile the model
#     with mp.Pool(n_threads, initializer=_init_thread) as pool:
    with mp.Pool(n_threads) as pool:
        print("Initializing...")
        pbar = tqdm(total=n_threads)
        for _ in pool.imap_unordered(run_one_param_set, inputs[:n_threads]):
            pbar.update(1)
        print("Done with initialization")
        print(f"Running {n_replicates} replicates of {n_samples} samples")
        pbar = tqdm(total=n_samples * n_replicates)
        results = []
        for res in pool.imap_unordered(run_one_param_set, inputs):
            results.append(res)
            pbar.update(1)

    ...

    results_order, times = zip(*results)
    reorder = np.argsort(results_order)
    times = np.array(times)[reorder]
    sampled_quantities = np.array(
        [_rescale(u, lo, hi) for u, (lo, hi) in zip(uniform_samples.T, SAMPLING_RANGES)]
    )
    sampled_quantities = np.tile(sampled_quantities, (1, n_replicates))
    data = dict(runtime_seconds=times, replicate=replicates) | dict(
        zip(SAMPLED_VAR_NAMES, sampled_quantities)
    )
    df = pd.DataFrame(data).sort_values(by="runtime_seconds", ascending=False)

    if save:
        fpath = Path(save_dir).joinpath(
            f"2023-06-16_random_sample_runtimes_AIcircuit.csv"
        )
        fpath = fpath.resolve().absolute()
        print(f"Writing results to: {fpath}")
        df.to_csv(fpath, index=False)

    ...


if __name__ == "__main__":
    save_dir = Path("data/oscillation/gillespie_runtime_newranges")

    main(save_dir=save_dir, n_samples=1500, n_replicates=1, n_threads=None, save=True)
