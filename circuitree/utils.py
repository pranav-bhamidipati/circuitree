from math import ceil
from numba import vectorize as nb_vectorize
from numba import njit, prange, int32, char
from numba import types
import numpy as np
from typing import Iterable


__all__ = [
    "ceiling",
    "vround",
    "rgb2hex",
]

## vectorized functions


@nb_vectorize
def ceiling(x):
    return ceil(x)


@nb_vectorize
def vround(x):
    return round(x)


## Color utils


def rgb2hex(rgb):
    """Converts rgb colors to hex"""

    RGB = np.zeros((3,), dtype=np.uint8)
    for i, _c in enumerate(rgb[:3]):
        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.0:
            c = int(_c * 255)
        else:
            c = _c

        # Calculate new values
        RGB[i] = round(c)

    return "#{:02x}{:02x}{:02x}".format(*RGB)
