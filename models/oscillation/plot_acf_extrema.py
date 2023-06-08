import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from oscillation import (
    autocorrelate,
    compute_largest_extremum_and_loc,
    compute_lowest_minima,
    binomial9_kernel,
    OscillationTreeBase,
)


otree = OscillationTreeBase(
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

    plt.suptitle(
        r"3-TF circuits with the most extreme ACF - Copy number vs $t$ (mins)", size=14
    )
    plt.tight_layout()

    if save:
        fname = (
            plot_dir.joinpath(f"highest_acfx_traces{suffix}.{fmt}").resolve().absolute()
        )
        print("Writing to:", fname)
        plt.savefig(fname, dpi=dpi)


def plot_acfx(
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

    plt.suptitle(
        r"3-TF circuits with the most extreme ACF - Autocorrelation vs $t-t'$ (mins)",
        size=14,
    )
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
        states = f["top_states"][:nplot].astype(str).tolist()
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
    where_extrema_f9, extrema_f9 = compute_largest_extremum_and_loc(acorrs_f9)
    t_f9 = t[4:-4]
    data_f9 = np.swapaxes(filtered9, -2, -1)
    locs_f9 = t_f9[where_extrema_f9]
    plot_data = dict(
        t=t_f9,
        data=data_f9,
        states=states,
        acorrs=acorrs_f9,
        locs=locs_f9,
        extrema=extrema_f9,
        prows=prows,
        pcols=pcols,
        plot_dir=plot_dir,
        save=save,
        dpi=dpi,
        fmt=fmt,
        plot_motifs=plot_motifs,
    )
    plot_copy_number(**plot_data)
    # plot_acfx(**plot_data)

    ...

    # Compare to the lowest minimum instead of largest extremum
    where_minima_f9, minima_f9 = compute_lowest_minima(acorrs_f9)
    locs_minima_f9 = t_f9[where_minima_f9]
    plot_data_minima = plot_data | dict(
        locs_compare=locs_minima_f9,
        extrema_compare=minima_f9,
        suffix="_with_minima",
    )
    plot_acfx(**plot_data_minima)

    ...

    # # Filter signal with a larger filter (9-point)
    # filtered9 = np.apply_along_axis(binomial9_kernel, -2, data_T)[..., 4:-4, :]
    # acorrs_f9 = np.apply_along_axis(autocorrelate, -2, filtered9)
    # where_extrema_f9, extrema_f9 = compute_largest_extremum_and_loc(acorrs_f9)
    # t_f9 = t[4:-4]
    # data_f9 = np.swapaxes(filtered9, -2, -1)
    # locs_f9 = t_f9[where_extrema_f9]

    # # Plot filtered data
    # plot_data_f9 = plot_data | dict(
    #     t=t_f9,
    #     data=data_f9,
    #     acorrs=acorrs_f9,
    #     locs=locs_f9,
    #     extrema=extrema_f9,
    # )
    # plot_conc(**plot_data_f9, suffix="_filtered_binomial9")
    # plot_acfx(**plot_data_f9, suffix="_filtered_binomial9")

    # # Try a 5-point filter
    # filtered5 = np.apply_along_axis(binomial5_kernel, -2, data_T)[..., 2:-2, :]
    # acorrs_f5 = np.apply_along_axis(autocorrelate, -2, filtered5)
    # where_extrema_f5, extrema_f5 = compute_largest_extremum_and_loc(acorrs_f5)
    # t_f5 = t[2:-2]
    # data_f5 = np.swapaxes(filtered5, -2, -1)
    # locs_f5 = t_f5[where_extrema_f5]
    # plot_data_f5 = dict(
    #     t=t_f5,
    #     data=data_f5,
    #     acorrs=acorrs_f5,
    #     locs=locs_f5,
    #     extrema=extrema_f5,
    # )
    # plot_acfx(**plot_data_f5, suffix="_filtered_binomial5")
    # plot_conc(**plot_data_f5, suffix="_filtered_binomial5")

    # # Try a 7-point filter
    # filtered7 = np.apply_along_axis(binomial7_kernel, -2, data_T)[..., 3:-3, :]
    # acorrs_f7 = np.apply_along_axis(autocorrelate, -2, filtered7)
    # where_extrema_f7, extrema_f7 = compute_largest_extremum_and_loc(acorrs_f7)
    # t_f7 = t[3:-3]
    # data_f7 = np.swapaxes(filtered7, -2, -1)
    # locs_f7 = t_f7[where_extrema_f7]
    # plot_data_f7 = dict(
    #     t=t_f7,
    #     data=data_f7,
    #     acorrs=acorrs_f7,
    #     locs=locs_f7,
    #     extrema=extrema_f7,
    # )
    # plot_acfx(**plot_data_f7, suffix="_filtered_binomial7")
    # plot_conc(**plot_data_f7, suffix="_filtered_binomial7")


if __name__ == "__main__":
    data_fpath = Path(
        "~/git/circuitree/data/oscillation/bfs/top_oscillating_states.hdf5"
    ).expanduser()
    plot_dir = Path("~/git/circuitree/figures/oscillation").expanduser()
    main(
        data_fpath,
        plot_dir,
        save=True,
    )
