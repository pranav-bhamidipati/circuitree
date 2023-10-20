from itertools import combinations
from typing import Sequence

__all__ = [
    "merge_overlapping_sets",
]


## set operations


def merge_overlapping_sets(sets: Sequence[set]) -> list[set]:
    """Given an iterable of non-empty sets, merges any sets that have non-empty
    intersection"""

    if any(len(s) == 0 for s in sets):
        raise ValueError("Sets must be non-empty")

    sets = list(sets)
    found_merge = True
    while found_merge:
        found_merge = False
        n_sets = len(sets)
        for i, j in combinations(range(n_sets), 2):
            set_i = sets[i]
            set_j = sets[j]
            if set_i & set_j:
                sets[i] |= sets.pop(j)
                found_merge = True
                break

    return sets
