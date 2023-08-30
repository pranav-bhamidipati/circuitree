from math import ceil
from typing import Mapping
from numpy import vectorize
from abc import ABC

__all__ = [
    "DefaultMapping",
    "DefaultFactoryDict",
    "ceiling",
    "vround",
]


class DefaultMapping(ABC, Mapping):
    """Abstract base class for mappings that implement a __missing__ method.
    Can be used to check whether a class performs defaultdict-like behavior

    Example:
    >>> class MyDict(dict):
    >>>     def __missing__(self, key):
    >>>         return 0
    >>> ...
    >>> isinstance(MyDict(), DefaultMapping) # True
    >>> issubclass(MyDict, DefaultMapping) # True
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is DefaultMapping:
            if any("__missing__" in C.__dict__ for C in subclass.__mro__):
                return True
        return NotImplemented


class DefaultFactoryDict(dict):
    """Similar to a defaultdict, but the default value is a function of the key."""

    def __init__(self, *args, default_factory=None, **kwargs):
        if default_factory is None:
            raise ValueError("Must provide a default_factory")
        super().__init__(*args, **kwargs)
        self._default_factory = default_factory

    def __missing__(self, key):
        self[key] = self._default_factory(key)
        return self[key]


## vectorized integer operations


@vectorize
def ceiling(x):
    return ceil(x)


@vectorize
def vround(x):
    return round(x)
