from math import ceil
from numpy import vectorize

__all__ = [
    "ceiling",
    "vround",
]

## vectorized integer operations


@vectorize
def ceiling(x):
    return ceil(x)


@vectorize
def vround(x):
    return round(x)
