import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def main(data_dir, save_dir, n_top_states=None, t=np.arange(0, 40000, 20.0), save=True, key=None):
    hdfs = list(data_dir.glob("*.hdf5"))
    if not hdfs:
        raise ValueError(f"No HDF5 files found in {data_dir}")
    with h5py.File(hdfs[0], "r") as f:
        _, nt, n_species = f["y_t"][...].shape

    if n_top_states is None:
        n_top_states = len(hdfs)
    n_top_states = min(n_top_states, len(hdfs))

    df: pd.DataFrame = pd.concat(pd.read_hdf(h, key=key) for h in hdfs)

    n_per_state = df.groupby("state").size().sort_values(ascending=False)
    top_states = list(n_per_state.index[:n_top_states])
    state_ordering = {s: i for i, s in enumerate(top_states)}
    max_reward = {s: df[df.state == s].reward.max() for s in top_states}

    data = np.zeros((n_top_states, n_species, nt), dtype=np.float64)
    for h in hdfs:
        _df = pd.read_hdf(h, key=key)
        mask_max = _df.reward == _df.reward.max()
        _state, _reward = _df.loc[mask_max, ["state", "reward"]].squeeze()
        idx = state_ordering.get(_state, np.inf)
        if (idx < n_top_states) and (max_reward.get(_state, 0.0) == _reward):
            where_max = mask_max.values.nonzero()[0][0]
            with h5py.File(h, "r") as f:
                data[idx] = f["y_t"][where_max].T

    if save:
        fname = save_dir.joinpath("top_oscillating_states.hdf5").resolve().absolute()
        print(f"Writing to: {fname}")
        with h5py.File(fname, "w") as f:
            f.create_dataset("t", data=t)
            f.create_dataset("y_t", data=data)
            f.create_dataset("top_states", data=top_states)
        df.to_hdf(fname, key="metadata", mode="a")


if __name__ == "__main__":
    data_dir = Path("~/git/circuitree/data/oscillation/bfs/extras").expanduser()
    save_dir = Path("~/git/circuitree/data/oscillation/bfs").expanduser()
    main(data_dir=data_dir, save_dir=save_dir)
