from functools import wraps
import inspect
from itertools import chain
from typing import Any, Callable, Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from matplotlib.figure import Figure
from pathlib import Path
from utils import normalize, vround, ValidPath

# import colorcet as cc

__all__ = [
    "_hex_vertices",
    "_hex_x",
    "_hex_y",
    "default_rcParams",
    "plot_hex_sheet",
    "get_frames",
    "animate_over_kwargs",
]

# Vertices of a regular hexagon centered at (0,0) with width 1.
_hex_vertices = np.array(
    [
        np.cos(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
        np.sin(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
    ]
).T / np.sqrt(3)

# X and Y values of a hexagon's vertices
_hex_x, _hex_y = _hex_vertices.T


def default_rcParams(
    SMALL_SIZE=12,
    MEDIUM_SIZE=14,
    BIGGER_SIZE=16,
):
    """Set default parameters for Matplotlib"""

    # Set font sizes
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_hex_sheet(
    ax: plt.Axes,
    X,
    var,
    vmin=None,
    vmax=None,
    cmap="viridis",
    cell_types=None,
    ctype_cmap="Set2",
    r=1.0,
    ec=None,
    title=None,
    xlim=(),
    ylim=(),
    axis_off=True,
    aspect=None,
    colorbar=False,
    cbar_aspect=20,
    extend=None,
    cbar_kwargs=dict(),
    poly_padding=0.0,
    **kwargs,
):

    # Clear axis (allows you to reuse axis when animating)
    # ax.clear()
    if axis_off:
        ax.axis("off")

    # Get min/max values in color space
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()

    if cell_types is None:
        cell_types = np.zeros_like(var, dtype=int)

    # Get colors for variable and cell type
    if isinstance(cmap, str):
        _cmap = plt.cm.get_cmap(cmap)
    else:
        _cmap = cmap
    colors = np.asarray(_cmap(normalize(var, vmin, vmax)))

    if isinstance(ctype_cmap, str):
        _ctype_cmap = plt.cm.get_cmap(ctype_cmap)
    else:
        _ctype_cmap = ctype_cmap
    ctype_colors = np.asarray(_ctype_cmap(cell_types))

    # Plot cells as polygons
    for i, (x, y) in enumerate(X):
        ax.fill(r * _hex_x + x, r * _hex_y + y, fc=colors[i], ec=ec, **kwargs)

    # Color code cell type
    ax.scatter(*X.T, s=3, c=ctype_colors)

    # Set figure args, accounting for defaults
    if title is not None:
        ax.set_title(title)
    if not xlim:
        xlim = [X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim = [X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect = 1
    ax.set(
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
    )

    if colorbar:

        # # Calculate colorbar extension if necessary
        # if extend is None:
        #     n = var.shape[0]
        #     ns_mask = ~np.isin(np.arange(n), sender_idx)
        #     is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
        #     is_over_max = var.max(initial=0.0, where=ns_mask) > vmax
        #     _extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
        # else:
        #     _extend = extend

        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=_cmap),
            ax=ax,
            aspect=cbar_aspect,
            **cbar_kwargs,
        )


def get_frames(nt: int, n_frames: int):
    return vround(np.linspace(0, nt - 1, n_frames))


class CallableAsGetitem:
    def __init__(self, callable: Callable[[int], Any]):

        if not isinstance(callable, Callable):
            raise ValueError("Argument is not a callable function.")

        self._callable = callable

    def __call__(self, i: int) -> Any:
        return self._callable.__call__(i)

    def __getitem__(self, i: int):
        return self.__call__(i)


def animate_over_kwargs(
    fig: Optional[Figure] = None,
    ax_or_axs: Optional[
        plt.Axes | Iterable[plt.Axes] | Iterable[Iterable[plt.Axes]]
    ] = None,
    iter_kwargs: tuple[str] = (),
    **kwargs,
) -> Callable[..., None]:
    def _decorator(plot_func):
        plot_func_args = inspect.signature(plot_func).parameters.keys()
        axes_kwargs = plot_func_args & set(["ax", "axs"])
        if len(axes_kwargs) != 1:
            raise ValueError(
                f"Plotting function `{plot_func}` must have either an `ax` or `axs` "
                f"argument."
            )
        else:
            ax_or_axs_key = axes_kwargs.pop()

        @wraps(plot_func)
        def _wrapper(
            save_dir: ValidPath,
            filename: str,
            n_frames: int,
            fps: int = 15,
            writer: str = "ffmpeg",
            dpi: int = 300,
            **kw,
        ):

            nt = len(kw[iter_kwargs[0]])
            frames = get_frames(nt, n_frames)

            get_iter_kw_at_step = lambda step: {k: kw[k][step] for k in iter_kwargs}
            other_kw = {k: kw[k] for k in set(kw) - set(iter_kwargs)}
            ax_kw = {ax_or_axs_key: ax_or_axs}

            def _render_frame(frame):
                iter_kw = get_iter_kw_at_step(frames[frame])
                for ax in fig.axes:
                    ax.clear()
                plot_func(**ax_kw, **iter_kw, **other_kw)

            try:
                _writer = animation.writers[writer](fps=fps, bitrate=1800)
            except RuntimeError:
                print(
                    """
                The `ffmpeg` writer must be installed inside the runtime environment.
                Writer availability can be checked in the current enviornment by executing 
                `matplotlib.animation.writers.list()` in Python. Install location can be
                checked by running `which ffmpeg` on a command line/terminal.
                """
                )

            _anim_FA = animation.FuncAnimation(
                fig, _render_frame, frames=n_frames, interval=200
            )

            # Get path and print to output
            save_dir = Path(save_dir)
            fpath = save_dir.joinpath(filename).with_suffix(".mp4")
            print("Writing to:", fpath.resolve().absolute())

            # Save animation
            _anim_FA.save(
                fpath,
                writer=_writer,
                dpi=dpi,
                progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
            )

        return _wrapper

    return _decorator
