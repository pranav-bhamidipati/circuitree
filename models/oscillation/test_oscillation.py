import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from oscillation import TFNetworkModel


def main(
    code: str = "*ABC::AiB_BiC_CiA",
    n_threads: int = 4,
    t: np.ndarray = np.linspace(0, 80000, 4001),
    prows: int = 2,
    pcols: int = 2,
    save_dir=Path("./figures/oscillation"),
    save=True,
    fmt="png",
    dpi=300,
):
    t_mins = t / 60
    t_hrs = t_mins / 60
    half_nt = len(t) // 2
    model = TFNetworkModel(code, seed=0)
    _, pop = model.run_ssa(t, n_threads=n_threads, progress_bar=True)

    acorr = model.get_autocorrelation()
    indices, freqs, peaks = model.get_secondary_autocorrelation_peaks(indices=True)

    fig1 = plt.figure(figsize=(4, 3))
    for i in range(n_threads):
        ax = fig1.add_subplot(prows, pcols, i + 1)
        ax.plot(t_hrs, pop[i, :, 3], label="TF A", lw=0.5)
        ax.plot(t_hrs, pop[i, :, 7], label="TF B", lw=0.5)
        ax.plot(t_hrs, pop[i, :, 11], label="TF C", lw=0.5)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Copy number")
    if save:
        fpath = Path(save_dir).joinpath(f"repressilator_traces").with_suffix(f".{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig1.savefig(str(fpath), dpi=dpi, bbox_inches="tight")

    fig2 = plt.figure(figsize=(4, 3))
    for i in range(n_threads):
        ax = fig2.add_subplot(prows, pcols, i + 1)
        ax.plot(t_hrs[:-half_nt], acorr[i, :, 0], label="TF A", lw=0.5)
        ax.plot(t_hrs[:-half_nt], acorr[i, :, 1], label="TF B", lw=0.5)
        ax.plot(t_hrs[:-half_nt], acorr[i, :, 2], label="TF C", lw=0.5)

        # Point out second peak of the autocorrelation function
        ax.scatter(t_hrs[indices[i]], peaks[i], marker="x", s=50, c="gray", zorder=100)

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Autocorrelation")

    if save:
        fpath = (
            Path(save_dir).joinpath(f"repressilator_autocorr").with_suffix(f".{fmt}")
        )
        print(f"Writing to: {fpath.resolve().absolute()}")
        fig2.savefig(str(fpath), dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":
    main(
        # code="*ABC::AaA_AaC_AiB_BiC_CiA",
    )
