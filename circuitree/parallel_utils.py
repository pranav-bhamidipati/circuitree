"""Useful primitives for parallel tree search. Requires installing `circuitree` with
the `parallel` extra."""

from gevent.event import Event
from itertools import count

## Useful primitives for parallel tree search


class BackupContextManager:
    def __init__(self, backup_not_in_progress: Event):
        self._backup_not_in_progress = backup_not_in_progress

    def __enter__(self):
        self._backup_not_in_progress.clear()

    def __exit__(self, *args):
        self._backup_not_in_progress.set()


class ManagedEvent(Event):
    """Event object that also provides a context manager."""

    def backup_context(self):
        """During a backup, the context manager will clear the event, blocking all
        threads until the backup is complete. Then the event is set again and
        threads are released."""
        return BackupContextManager(self)


class AtomicCounter:
    """A simple thread-safe counter. Uses itertools.count(), which is implemented in C 
    as an atomic operation.

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
