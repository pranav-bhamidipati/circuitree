from collections import Counter
import datetime
from itertools import combinations, count
from typing import Iterable

try:
    from gevent.event import Event
    from gevent.lock import RLock
except ImportError:
    from threading import Event
    from threading import RLock


__all__ = [
    "merge_overlapping_sets",
    "BackupContext",
    "DatabaseBackupManager",
    "ThreadsafeCounter",
    "ThreadsafeCountTable",
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


class BackupContext:
    """Context manager for database backups"""

    def __init__(self, event: Event):
        self.event = event

    def __enter__(self):
        """Start a backup. Clears the event until the backup is complete."""
        self.event.clear()

    def __exit__(self, *args):
        """Finish a backup. Sets the event to allow other threads to proceed."""
        self.event.set()


class DatabaseBackupManager:
    """A class to manage backups of the Redis database where all simulation results are
    stored. Includes functions to monitor backup state and a context manager to be used
    while running the backup."""

    def __init__(
        self,
        database_info: dict,
        tz_offset: int,
        next_backup_in_seconds: int | float = 0,
    ):

        self.database_info = database_info

        # The event indicates whether a backup is *not* in progress.
        # By default, the event is set, allowing threads to proceed.
        self.event = Event()
        self.event.set()

        # By default, the time zone is Pacific and the next backup is scheduled for the
        # current time (i.e. the first backup will be triggered immediately)
        self.time_zone = datetime.timezone(datetime.timedelta(hours=tz_offset))
        self.next_backup_time = datetime.datetime.now(
            self.time_zone
        ) + datetime.timedelta(seconds=next_backup_in_seconds)

    def wait_until_finished(self):
        """Block until a backup is complete."""
        self.event.wait()

    def is_due(self):
        """Check if a backup is due."""
        return datetime.datetime.now(self.time_zone) >= self.next_backup_time

    def is_running(self):
        """Check if a backup is currently in progress."""
        return not self.event.is_set()

    def __call__(self):
        """Returns a context manager for backing up the database. Use as:
        with backup_manager() as backup:
            backup.backup_to_file(...)

        Other threads can check if a backup is in progress by calling
        backup_manager.is_running()."""

        if self.is_running():
            raise RuntimeError("Backup is already in progress.")

        return BackupContext(self.event)

    def schedule_next(self, backup_every: int):
        """Schedule the next backup to occur in `backup_every` seconds."""
        self.next_backup_time = datetime.datetime.now(
            self.time_zone
        ) + datetime.timedelta(seconds=backup_every)


class ThreadsafeCounter:
    """Increments an integer counter atomically across green threads. Uses a `gevent`
    lock for synchronization."""

    def __init__(self):
        self._lock = RLock()
        self._counter = 0

    def increment(self):
        with self._lock:
            self._counter += 1

    def value(self):
        with self._lock:
            return self._counter


class ThreadsafeCountTable:
    """A threadsafe wrapper around `collections.Counter`. Edits, reads, and increments
    are synchronized across green threads using a `gevent` lock."""

    def __init__(self):
        self._lock = RLock()
        self._counter = Counter()

    def __getitem__(self, key):
        with self._lock:
            return self._counter[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._counter[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._counter[key]

    def get_val_and_increment(self, key):
        with self._lock:
            val = self._counter[key]
            self._counter[key] += 1
            return val


class AtomicCounter:
    """[DEPRECATED] Increments a counter atomically.
    *********
    NOTE: Please use ThreadsafeCounter instead! To guarantee atomicity, the Python
    exec used must be CPython, and the GIL must be enabled. Going forward, the GIL
    will become optional (see PEP 703).
    *********

    Increments a counter atomically. Uses itertools.count(), which
    is implemented in C as an atomic operation.

    From StackOverflow user `PhilMarsh`:
        https://stackoverflow.com/a/71565358

    """

    def __init__(self):
        self._incs = count()
        self._accesses = count()

    def increment(self):
        next(self._incs)

    def value(self):
        return next(self._incs) - next(self._accesses)
