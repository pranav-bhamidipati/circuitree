from functools import partial
import json
from math import ceil
import h5py
import logging
from multiprocessing import Pool
from more_itertools import chunked_even
import numpy as np
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from time import perf_counter
from typing import Optional
from uuid import uuid4

from oscillation import TFNetworkModel


def run_batch_and_save(
    input_args,
    dt,
    nt,
    save_dir,
    init_columns,
    param_names,
    oscillation_thresh,
    log_dir,
    task_id=None,
):
    start_time = perf_counter()

    start_logging(log_dir, task_id)

    seed, genotype, prots0, param_sets = input_args
    chunksize = len(param_sets)
    logging.info(f"Seed: {seed}")
    logging.info(f"Genotype: {genotype}")
    logging.info(f"# parameter sets: {chunksize}")

    start_sim_time = perf_counter()
    model = TFNetworkModel(genotype)
    model.initialize_ssa(seed=seed, dt=dt, nt=nt)
    pop0s = np.array([model.ssa.population_from_proteins(p0) for p0 in prots0])
    prots_t, prots0, param_sets, rewards = model.run_ssa_and_get_acf_minima(
        pop0=pop0s,
        params=param_sets,
        seed=seed,
        freqs=False,
        indices=False,
        abs=False,
    )
    end_sim_time = perf_counter()
    sim_time = end_sim_time - start_sim_time
    logging.info(f"Simulation took {sim_time:.2f}s ")

    run_results = seed, genotype, pop0s, param_sets, rewards, prots_t
    save_run(
        run_results=run_results,
        save_dir=save_dir,
        init_columns=init_columns,
        param_names=param_names,
        oscillation_thresh=oscillation_thresh,
    )

    end_time = perf_counter()
    total_time = end_time - start_time
    logging.info(f"Total time {total_time:.2f}s")

    return seed, sim_time, total_time


def save_run(
    run_results, save_dir, init_columns, param_names, oscillation_thresh, **kwargs
):
    seed, genotype, pop0s, param_sets, rewards, prots_t = run_results

    state_dir = Path(save_dir).joinpath(f"state_{genotype.strip('*')}")
    state_dir.mkdir(exist_ok=True)

    save_results(
        state_dir,
        seed,
        genotype,
        rewards,
        pop0s,
        param_sets,
        init_columns,
        param_names,
        **kwargs,
    )

    if np.any(rewards > oscillation_thresh):
        pop_data_dir = Path(save_dir).joinpath(f"extras")
        pop_data_dir.mkdir(exist_ok=True)
        save_pop_data(
            pop_data_dir,
            genotype,
            prots_t,
            pop0s,
            param_sets,
            rewards,
            init_columns,
            param_names,
            oscillation_thresh,
            **kwargs,
        )

    return genotype


def save_results(
    state_dir,
    seed,
    genotype,
    rewards,
    pop0s,
    param_sets,
    init_columns,
    param_names,
    ext="parquet",
):
    data = (
        dict(seed=seed, state=genotype, reward=rewards)
        | dict(zip(init_columns, np.atleast_2d(pop0s).T))
        | dict(zip(param_names, np.atleast_2d(param_sets).T))
    )
    df = pd.DataFrame(data)
    df["state"] = df["state"].astype("category")

    fname = state_dir.joinpath(f"{uuid4()}.{ext}").resolve().absolute()

    logging.info(f"Writing results to: {fname}")

    if ext == "csv":
        df.to_csv(fname, index=False)
    elif ext == "parquet":
        df.to_parquet(fname, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def save_pop_data(
    pop_data_dir,
    genotype,
    prots_t,
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

    logging.info(f"\tWriting all data for {len(save_idx)} runs to: {fname}")

    with h5py.File(fname, "w") as f:
        f.create_dataset("y_t", data=prots_t[save_idx])
    df.to_hdf(fname, key="metadata", mode="a", format="table")


def start_logging(log_dir, run_id=None):
    task_id = uuid4().hex
    if run_id is not None:
        task_id = f"{run_id}_{task_id}"
    logger, logfile = _init_logger(task_id, log_dir)
    logger.info(f"Initialized logger for task {task_id}")
    logger.info(f"Logging to {logfile}")


def _init_logging(level=logging.INFO, mode="a"):
    fmt = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s --- %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    global _log_meta
    _log_meta = {"mode": mode, "fmt": fmt}


def _init_logger(task_id, log_dir: Path):
    logger = logging.getLogger()
    logger.handlers = []  # remove all handlers
    logfile = Path(log_dir).joinpath(f"task_{task_id}.log")
    fh = logging.FileHandler(logfile, mode=_log_meta["mode"])
    fh.setFormatter(_log_meta["fmt"])
    logger.addHandler(fh)
    return logger, logfile


def prompt_before_wiping_logs(log_dir):
    while True:
        decision = input(f"Delete all files in log directory?\n\t{log_dir}\n[Y/n]: ")
        if decision.lower() in ("y", "yes"):
            import shutil

            shutil.rmtree(log_dir)
            log_dir.mkdir()
            break
        elif decision.lower() in ("n", "no"):
            import sys

            print("Exiting...")
            sys.exit(0)
        else:
            print(f"Invalid input: {decision}")


def main(
    log_dir: Path = Path("data/oscillation/bfs"),
    save_dir: Path = Path("logs/oscillation/bfs"),
    params_queue_file: Path = Path("data/oscillation/param_sets_queue_10000.csv"),
    genotype_queue_file: Path = Path("data/oscillation/genotypes_queue.json"),
    chunksize: int = 100,
    parallel: bool = False,
    first_chunk_idx: int = 0,
    nt: int = 2000,
    dt_seconds: float = 20.0,
    n_workers: Optional[int] = None,
    oscillation_thresh: float = 0.35,
    auto_wipe_log_dir: bool = False,
    prompt_wipe: bool = False,
    log_level: int = logging.INFO,
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

    import os

    run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID")) + int(os.environ.get("JOB_QUEUE_FIRST_INDEX"))
    
    save_dir = save_dir.resolve().absolute()
    save_dir.mkdir(exist_ok=True)
    log_dir = log_dir.resolve().absolute()
    log_dir.mkdir(exist_ok=True)

    with genotype_queue_file.open("r") as f:
        genotypes = json.load(f)

    n_genotypes = len(genotypes)
    params_chunk_index, genotype_index = divmod(run_id, n_genotypes)
    params_chunk_index += first_chunk_idx
    genotype = genotypes[genotype_index]
    start_index = chunksize * params_chunk_index
    end_index = chunksize * (1 + params_chunk_index)

    init_and_param_data = pd.read_csv(
        params_queue_file, skiprows=range(1, start_index + 1), nrows=chunksize, header=0
    )
    param_sets = init_and_param_data[param_names].values
    prot0s = init_and_param_data[init_columns].values
    kw = dict(
        dt=dt_seconds,
        nt=nt,
        save_dir=save_dir,
        init_columns=init_columns,
        param_names=param_names,
        oscillation_thresh=oscillation_thresh,
        log_dir=log_dir,
    )

    # Start logging
    logging.basicConfig(
        filename=Path(log_dir).joinpath(f"main_{run_id}.log"), level=logging.INFO
    )
    if any(log_dir.iterdir()):
        if prompt_wipe:
            prompt_before_wiping_logs(log_dir)
        elif auto_wipe_log_dir:
            print(f"Auto-wiping log directory: {log_dir}")
            for f in log_dir.iterdir():
                f.unlink()
            logging.info(f"Wiped contents of log directory {log_dir}")

    if parallel:
        n_workers = cpu_count(logical=False) if n_workers is None else n_workers
        _msg = (
            f"Starting job with SLURM_ARRAY_TASK_ID={run_id}. Using {n_workers} process(es) "
            f"to run param sets {start_index} to {end_index} with genotype {genotype}. "
            f"Logging to {log_dir}"
        )
        logging.info(_msg)
        print(_msg)

        batchsize = ceil(chunksize / n_workers)
        prot0s_batched = [np.array(c) for c in chunked_even(prot0s, batchsize)]
        param_sets_batched = [np.array(c) for c in chunked_even(param_sets, batchsize)]

        args = []
        for i, (prot0s_batch, param_sets_batch) in enumerate(
            zip(prot0s_batched, param_sets_batched)
        ):
            seed = start_index + i
            args.append((seed, genotype, prot0s_batch, param_sets_batch))

        run_batch_job = partial(run_batch_and_save, **kw)

        with Pool(n_workers, initializer=_init_logging, initargs=(log_level,)) as pool:
            k = 0
            for seed, simulation_time, total_time in pool.imap_unordered(
                run_batch_job, args
            ):
                k += 1
                total = total_time / 60
                sim = simulation_time / 60
                _completion_msg = (
                    f"[{k / n_workers:.1%}] -- Finished batch {seed} in {total:.2f} mins "
                    f"({sim:.2f} mins of SSA) for genotype {genotype}"
                )
                print(_completion_msg)
                logging.info(_completion_msg)
    else:
        _msg = (
            f"Starting job with SLURM_ARRAY_TASK_ID={run_id}. Running param sets "
            f"{start_index} to {end_index} with genotype {genotype}. Logging to {log_dir}"
        )
        logging.info(_msg)
        print(_msg)

        seed = start_index
        args = seed, genotype, prot0s, param_sets
        _init_logging(log_level)

        seed, sim_time, total_time = run_batch_and_save(input_args=args, **kw)
        total_time /= 60
        sim_time /= 60
        _completion_msg = (
            f"Finished batch {seed} in {total_time:.2f} mins "
            f"({sim_time:.2f} mins of SSA) for genotype {genotype}"
        )
        print(_completion_msg)
        logging.info(_completion_msg)


if __name__ == "__main__":
    log_dir = Path("logs/oscillation/bfs")
    save_dir = Path("/home/pbhamidi/scratch/circuitree/data/oscillation/bfs")
    params_queue_file = Path("data/oscillation/param_sets_queue_10000.csv")
    genotype_queue_file = Path("data/oscillation/genotypes_queue.json")

    main(
        log_dir=log_dir,
        save_dir=save_dir,
        params_queue_file=params_queue_file,
        genotype_queue_file=genotype_queue_file,
        chunksize=100,
        parallel=False,
        # For debugging
        # nt=100,
        # chunksize=5,
    )
