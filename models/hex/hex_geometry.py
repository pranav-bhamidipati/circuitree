from functools import partial
from typing import Optional, Sequence
import numpy as np

from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "hex_grid",
    "get_center_cells",
    "get_adjacency_hexagonal_nnb",
    "get_adjacency_distance_weighted",
    "get_hex_grid_with_nnb_adjacency",
]


def hex_grid(rows, cols: Optional[int] = None, r=1.0, sigma=0):
    """
    Returns XY coordinates of a regular 2D hexagonal grid
    (rows x cols) with edge length r. Points are optionally
    passed through a Gaussian filter with std. dev. = sigma * r.
    """

    # Check if square grid
    if cols is None:
        cols = rows

    # Populate grid
    x_coords = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols)
    y_coords = np.linspace(
        -np.sqrt(3) * (rows - 1) / 4, np.sqrt(3) * (rows - 1) / 4, rows
    )
    X = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            X.append(np.array([x + (j % 2) / 2, y]))
    X = np.array(X)

    # Apply Gaussian filter if specified
    if sigma != 0:
        X = np.array([np.random.normal(loc=x, scale=sigma) for x in X])

    # Center on first point
    X = X - X[0]

    return r * X


def get_center_cells(X, n=1):
    """
    Returns indices of the `n` cells closest to the origin given
    their coordinates as an array X.
    """
    return np.argpartition([np.linalg.norm(x) for x in X], n)[:n]


def get_adjacency_hexagonal_nnb(
    rows: int,
    cols: Optional[int] = None,
    periodic: bool | tuple[bool] = True,
    normalize: bool = True,
):
    """ """
    invalid_periodic_error = ValueError("periodic must be a boolean or tuple of 1 or 2 booleans.")
    if isinstance(periodic, Sequence):
        if len(periodic) == 1:
            (periodic_xy,) = periodic
            periodic_x = periodic_xy
            periodic_y = periodic_xy
        elif len(periodic) == 2:
            periodic_x, periodic_y = periodic
            periodic_xy = periodic_x & periodic_y
        else:
            raise invalid_periodic_error
    elif isinstance(periodic, bool):
        periodic_xy = periodic
        periodic_x = periodic
        periodic_y = periodic
    else:
        raise invalid_periodic_error

    if not all(isinstance(p, bool) for p in (periodic_xy, periodic_x, periodic_y)):
        raise ValueError("`periodic` must be either a boolean or tuple of booleans.")

    # Check for square grid
    if cols is None:
        cols = rows

    if periodic_y & (rows % 2):
        raise ValueError(
            "Number of rows must be even to make a lattice with periodic y-boundaries"
        )

    # For periodic conditoins, indices outside grid wrap around
    if periodic_xy:
        ij_to_flat = partial(np.ravel_multi_index, dims=(cols, rows), mode="wrap")

    # For nonperiodic boundaries, indices outside grid coordinates yield NaN
    elif periodic_x:

        def ij_to_flat(multi_index: tuple[int]):
            xi, yi = multi_index
            try:
                if (yi < 0) or (yi > rows):
                    raise ValueError
                else: 
                    idx = np.ravel_multi_index(
                        multi_index, dims=(cols, rows), mode="wrap"
                    )
            except ValueError:
                idx = np.nan
            
            return idx

    elif periodic_y:

        def ij_to_flat(multi_index: tuple[int]):
            xi, yi = multi_index
            try:
                if (xi < 0) or (xi > cols):
                    raise ValueError
                else:
                    idx = np.ravel_multi_index(
                        multi_index, dims=(cols, rows), mode="wrap"
                    )
            except ValueError:
                idx = np.nan
            
            return idx

    else:

        def ij_to_flat(multi_index: tuple[int]):
            try:
                return np.ravel_multi_index(
                    multi_index, dims=(cols, rows), mode="raise"
                )
            except ValueError:
                return np.nan

    flat_to_ij = partial(np.unravel_index, shape=(cols, rows))

    # Assemble adjacency matrix by finding the neighbors of each point
    n = rows * cols
    Adj = np.zeros((n, n), dtype=np.float_)
    for idx in range(n):
        i, j = flat_to_ij(idx)

        stagger = (-1, 1)[j % 2]  # Account for staggered rows in grid
        nbs = [[0, 1], [0, -1], [1, 0], [-1, 0], [stagger, 1], [stagger, -1]]

        nb_flat = np.array([ij_to_flat((nbi + i, nbj + j)) for nbi, nbj in nbs])

        # For non-periodic conditions, remove invalid indices
        valid_nb = nb_flat[~np.isnan(nb_flat)].astype(int)
        Adj[idx, valid_nb] = 1

    # Each interface takes up 1/6 of the surface area of the hexagon
    if normalize:
        Adj /= 6

    return Adj


def get_adjacency_distance_weighted(
    X, r_int, dtype=np.float32, sparse=False, row_stoch=False, atol=1e-8, **kwargs
):
    """
    Construct adjacency matrix for a non-periodic set of
    points (cells). Adjacency is determined by calculating pairwise
    distance and applying a threshold `r_int` (radius of interaction).
    Within this radius, weights are calculated from pairwise distance
    as the value of the PDF of a Normal distribution with standard
    deviation `r_int / 2`.
    """

    n = X.shape[0]
    d = pdist(X)
    a = stats.norm.pdf(d, loc=0, scale=r_int / 2)
    a[d > (r_int + atol)] = 0
    A = squareform(a)

    if row_stoch:
        rowsum = np.sum(A, axis=1)[:, np.newaxis]
        A = np.divide(A, rowsum)
    else:
        A = (A > 0).astype(dtype)

    if sparse:
        A = csr_matrix(A)

    return A


def get_hex_grid_with_nnb_adjacency(
    rows: int,
    cols: int = 0,
    r: float = 1.0,
    periodic: bool = True,
    normalize: bool = True,
    **kwargs
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    X = hex_grid(rows, cols, r)
    Adj = get_adjacency_hexagonal_nnb(rows, cols, periodic, normalize)

    return X, Adj
