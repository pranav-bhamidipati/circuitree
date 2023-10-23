from abc import ABC, abstractmethod
from typing import Any, Hashable, Iterable

from .utils import merge_overlapping_sets

__all__ = ["CircuitGrammar"]


class CircuitGrammar(ABC):
    def __init__(self, *args, **kwargs):
        self._non_serializable_attrs = ["_non_serializable_attrs"]

    @abstractmethod
    def get_actions(self, state: Hashable) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def do_action(self, state: Hashable, action: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_unique_state(self, state: Hashable) -> Any:
        raise NotImplementedError

    def has_pattern(self, state: Hashable, pattern: Hashable) -> bool:
        raise NotImplementedError

    def get_undo_actions(self, state: Hashable) -> Iterable[Any]:
        """Get the actions that can be undone from the given state."""
        raise NotImplementedError

    def undo_action(self, state: Hashable, action: Any) -> Hashable:
        """Undo one action from the given state."""
        raise NotImplementedError

    def get_dict_serializable(self) -> dict:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in self._non_serializable_attrs
        }

    def to_dict(self):
        return {
            "__grammar_cls__": self.__class__.__name__,
            **self.get_dict_serializable(),
        }

    def merge_overlapping_sets(self, sets: Iterable[set]) -> list[set]:
        return merge_overlapping_sets(sets)
