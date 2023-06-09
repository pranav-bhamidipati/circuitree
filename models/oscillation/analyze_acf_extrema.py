import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def main(
    data_dir, save_dir, n_top=None, t=np.arange(0, 40000, 20.0), save=True, key=None
):
    hdfs = list(data_dir.glob("*.hdf5"))
    if not hdfs:
        raise ValueError(f"No HDF5 files found in {data_dir}")
    with h5py.File(hdfs[0], "r") as f:
        _, nt, n_species = f["y_t"][...].shape

    if n_top is None:
        n_top = len(hdfs)
    n_top = min(n_top, len(hdfs))

    dfs = []
    for h in hdfs:
        _df = pd.read_hdf(h, key=key)
        _df["index_in_file"] = np.arange(len(_df))
        _df["file"] = h.name
        dfs.append(_df)
    df: pd.DataFrame = pd.concat(dfs)

    # Pick the states with the most oscillating runs and save the best run
    n_per_state = df.groupby("state").size().sort_values(ascending=False)
    top_states = list(n_per_state.index[:n_top])
    state_ordering = {s: i for i, s in enumerate(top_states)}
    max_reward = {s: df[df.state == s].reward.max() for s in top_states}

    top_state_data = np.zeros((n_top, n_species, nt), dtype=np.float64)
    for state, reward, fname, idx in df[
        ["state", "reward", "file", "index_in_file"]
    ].values:
        rank = state_ordering.get(state, -1)
        if (rank >= 0) and (max_reward.get(state, 0.0) == reward):
            h = data_dir.joinpath(fname).resolve().absolute()
            with h5py.File(h, "r") as f:
                top_state_data[rank] = f["y_t"][idx].T

    if save:
        fname = save_dir.joinpath("top_oscillating_states.hdf5").resolve().absolute()
        print(f"Writing to: {fname}")
        with h5py.File(fname, "w") as f:
            f.create_dataset("t", data=t)
            f.create_dataset("y_t", data=top_state_data)
            f.create_dataset("states", data=top_states)
        df.to_hdf(fname, key="metadata", mode="a")

    # Pick the best oscillating runs
    top_runs = df.sort_values("reward", ascending=False).head(n_top)
    top_run_data = np.zeros((n_top, n_species, nt), dtype=np.float64)
    for i, (state, fname, idx) in enumerate(
        top_runs[["state", "file", "index_in_file"]].values
    ):
        h = data_dir.joinpath(fname).resolve().absolute()
        with h5py.File(h, "r") as f:
            top_run_data[i] = f["y_t"][idx].T

    if save:
        fname = save_dir.joinpath("top_oscillating_runs.hdf5").resolve().absolute()
        print(f"Writing to: {fname}")
        with h5py.File(fname, "w") as f:
            f.create_dataset("t", data=t)
            f.create_dataset("y_t", data=top_run_data)
            f.create_dataset("states", data=top_runs.state.tolist())
        top_runs.to_hdf(fname, key="metadata", mode="a")


if __name__ == "__main__":
    data_dir = Path("~/git/circuitree/data/oscillation/bfs/extras").expanduser()
    save_dir = Path("~/git/circuitree/data/oscillation/bfs").expanduser()
    main(data_dir=data_dir, save_dir=save_dir)
