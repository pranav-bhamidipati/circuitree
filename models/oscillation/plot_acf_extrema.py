import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from oscillation import (
    autocorrelate,
    compute_lowest_minima,
    binomial9_kernel,
    OscillationTree,
)


otree = OscillationTree(
    components=["A", "B", "C"], interactions=["activates", "inhibits"], root="ABC::"
)


def has_motif(state, motif):
    interaction_code = state.split("::")[1]
    if not interaction_code:
        return False
    state_interactions = set(interaction_code.split("_"))

    for recoloring in otree.get_interaction_recolorings(motif):
        motif_interactions = set(recoloring.split("_"))
        if motif_interactions.issubset(state_interactions):
            return True
    return False


def plot_copy_number(
    t,
    data,
    states,
    prows,
    pcols,
    plot_dir,
    save,
    dpi,
    fmt,
    fig=None,
    suffix="",
    plot_motifs=(),
    suptitle=None,
    **kwargs,
):
    nplot, n_species, nt = data.shape
    t_mins = t / 60.0

    if fig is None:
        fig = plt.figure(figsize=(pcols * 2, prows * 2))

    for i, (state, y_t) in enumerate(zip(states, data)):
        ax = fig.add_subplot(prows, pcols, i + 1)
        ax.set_title(f"{state[6:]}", size=10)
        for j in range(n_species):
            ax.plot(t_mins, y_t[j], lw=0.5)
        x = 0.98
        y = 0.95
        if plot_motifs:
            ax.text(
                x, y, "Motifs:", ha="left", va="top", transform=ax.transAxes, size=8
            )
            y -= 0.1
        for motif, s in plot_motifs:
            if has_motif(state, motif):
                ax.text(x, y, s, ha="left", va="top", transform=ax.transAxes, size=8)
                y -= 0.1
        ax.yaxis.set_ticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if suptitle is not None:
        plt.suptitle(suptitle, size=14)
    plt.tight_layout()

    if save:
        fname = (
            plot_dir.joinpath(f"highest_acfx_traces{suffix}.{fmt}").resolve().absolute()
        )
        print("Writing to:", fname)
        plt.savefig(fname, dpi=dpi)


def plot_acf(
    t,
    data,
    states,
    acorrs,
    locs,
    extrema,
    prows,
    pcols,
    plot_dir,
    save,
    dpi,
    fmt,
    fig=None,
    suffix="",
    plot_motifs=(),
    locs_compare=None,
    extrema_compare=None,
    suptitle=None,
    **kwargs,
):
    nplot, n_species, nt = data.shape
    half_nt = nt // 2
    t_mins = t / 60.0

    if fig is None:
        fig = plt.figure(figsize=(pcols * 2, prows * 2))

    for i, (acorr, corr_time, extremum) in enumerate(zip(acorrs, locs, extrema)):
        ax = fig.add_subplot(prows, pcols, i + 1)
        state = states[i]
        ax.set_title(f"{state[6:]}", size=10)
        for j in range(n_species):
            ax.plot(t_mins[:-half_nt], acorr[:, j], lw=0.5)
        corr_time_mins = corr_time / 60.0
        ax.scatter(corr_time_mins, extremum, marker="x", s=50, c="gray", zorder=100)
        ax.text(
            corr_time_mins + 0.05 * t_mins.max(),
            extremum,
            f"{extremum:.2f}",
            ha="left",
            va="center",
            size=10,
        )

        if (locs_compare is not None) and (extrema_compare is not None):
            corr_time_compare = locs_compare[i] / 60.0
            extremum_compare = extrema_compare[i]
            ax.scatter(
                corr_time_compare,
                extremum_compare,
                marker="x",
                s=50,
                c="red",
                zorder=100,
            )
            ax.text(
                corr_time_compare + 0.05 * t_mins.max(),
                extremum_compare,
                f"{extremum_compare:.2f}",
                ha="left",
                va="center",
                size=10,
                color="red",
            )

        x = 0.75
        y = 0.95
        if plot_motifs:
            ax.text(
                x, y, "Motifs:", ha="left", va="top", transform=ax.transAxes, size=8
            )
            y -= 0.1
        for motif, s in plot_motifs:
            if has_motif(state, motif):
                ax.text(x, y, s, ha="left", va="top", transform=ax.transAxes, size=8)
                y -= 0.1

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if suptitle is not None:
        plt.suptitle(suptitle, size=14)
    plt.tight_layout()

    if save:
        fname = (
            plot_dir.joinpath(f"highest_acfx_acorrs{suffix}.{fmt}").resolve().absolute()
        )
        print("Writing to:", fname)
        plt.savefig(fname, dpi=dpi)


def main(
    data_fpath,
    plot_dir,
    prows=5,
    pcols=5,
    save=False,
    dpi=300,
    fmt="png",
):
    nplot = prows * pcols
    with h5py.File(data_fpath, "r") as f:
        t = f["t"][...]
        data = f["y_t"][:nplot]
        states = f["states"][:nplot].astype(str).tolist()
    data_T = np.swapaxes(data, -2, -1)

    plot_motifs = [
        ("*ABC::ABa_BAi", "A-I"),
        ("*ABC::ABi_BCi_CAi", "I-I-I"),
        ("*ABC::ABi_BAi", "Toggle"),
    ]

    # acorrs = np.apply_along_axis(autocorrelate, -2, data_T)
    # where_extrema, extrema = compute_largest_extremum_and_loc(acorrs)
    # locs = t[where_extrema[i]]

    # Before autocorrelation, we filter the data with a 9-point binomial filter
    filtered9 = np.apply_along_axis(binomial9_kernel, -2, data_T)[..., 4:-4, :]
    acorrs_f9 = np.apply_along_axis(autocorrelate, -2, filtered9)
    data_f9 = np.swapaxes(filtered9, -2, -1)
    t_f9 = t[4:-4]

    # where_extrema_f9, extrema_f9 = compute_largest_extremum_and_loc(acorrs_f9)
    # locs_f9 = t_f9[where_extrema_f9]

    # Compare to the lowest minimum instead of largest extremum
    where_minima_f9, minima_f9 = compute_lowest_minima(acorrs_f9)
    locs_f9 = t_f9[where_minima_f9]
    plot_data = dict(
        t=t_f9,
        data=data_f9,
        states=states,
        acorrs=acorrs_f9,
        locs=locs_f9,
        extrema=minima_f9,
        prows=prows,
        pcols=pcols,
        plot_dir=plot_dir,
        save=save,
        dpi=dpi,
        fmt=fmt,
        plot_motifs=plot_motifs,
        suffix="_230608",
    )

    plot_copy_number(
        suptitle=r"Runs with the highest reward - Copy number vs $t$ (mins)",
        # suptitle=r"3-TF circuits with the most oscillating runs - Copy number vs $t$ (mins)",
        **plot_data,
    )
    plot_acf(
        suptitle=r"Runs with the highest reward - Autocorrelation vs $t-t'$ (mins)",
        # suptitle=r"3-TF circuits with the most oscillating runs - Autocorrelation vs $t-t'$ (mins)",
        **plot_data,
    )

    ...


if __name__ == "__main__":
    data_fpath = Path(
        "~/git/circuitree/data/oscillation/bfs/top_oscillating_runs.hdf5"
        # "~/git/circuitree/data/oscillation/bfs/top_oscillating_states.hdf5"
    ).expanduser()
    plot_dir = Path("~/git/circuitree/figures/oscillation").expanduser()
    main(
        data_fpath,
        plot_dir,
        save=True,
    )
