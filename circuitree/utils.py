from itertools import combinations, count
from typing import Iterable

try:
    from gevent.event import Event
except ImportError:
    from threading import Event


__all__ = [
    "merge_overlapping_sets",
    "ManagedEvent",
    "AtomicCounter",
]


## Set operations
 

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


## Useful primitives for parallel tree search


class BackupContextManager:
    """Context manager for blocking all other threads while a backup is in progress"""

    def __init__(self, backup_not_in_progress: Event):
        self._backup_not_in_progress = backup_not_in_progress

    def __enter__(self):
        self._backup_not_in_progress.clear()

    def __exit__(self, *args):
        self._backup_not_in_progress.set()


class ManagedEvent(Event):
    """
    Event object that also provides a context manager.

    During a backup, the context manager will clear the event, blocking all
    threads until the backup is complete. Then the event is set again and
    threads are released.
    """

    def backup_context(self):

        return BackupContextManager(self)


class AtomicCounter:
    """A simple thread-safe counter. Uses itertools.count(), which is implemented in C
    as an atomic operation. Can be used, for example, to track the total number of
    elapsed iterations across threads.

    **REQUIRES CYPTHON**

    From StackOverflow user 'PhilMarsh':
        https://stackoverflow.com/a/71565358

    """

    def __init__(self):
        self._incs = count()
        self._accesses = count()

    def increment(self):
        next(self._incs)

    def value(self):
        return next(self._incs) - next(self._accesses)
