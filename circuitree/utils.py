from itertools import combinations
from typing import Iterable

__all__ = [
    "merge_overlapping_sets",
]


## set operations


def merge_overlapping_sets(sets: Iterable[set]) -> list[set]:
    """Given an iterable of non-empty sets, merges any sets that have non-empty
    intersection"""

    sets = list(sets)
    if any(len(s) == 0 for s in sets):
        raise ValueError("Sets must be non-empty")

    sets = list(sets)
    merged = False
    while not merged:
        n_sets = len(sets)
        for i, j in combinations(range(n_sets), 2):
            if sets[i] & sets[j]:
                sets[i] |= sets.pop(j)
                break
        merged = True

    return sets
