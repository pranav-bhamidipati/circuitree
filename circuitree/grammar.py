from abc import ABC, abstractmethod
from typing import Any, Hashable, Iterable

__all__ = ["CircuitGrammar"]


class CircuitGrammar(ABC):
    """Abstract class for Grammars, which define the rules for the circuit assembly
    "game". Each stage of circuit topology assembly is represented by a ``state``, which
    is a hashable ID (typically a string) for the current state. The state can be
    modified by applying an ``action``, and the game ends when a terminal state is
    reached.

    To search a particular space of circuits, you must use or create a subclass of this
    class that defines the rules of that particular assembly game - which actions can be
    taken from a given state, and how to apply them. Specifically, the following methods
    must be implemented:

    - ``get_actions``
    - ``do_action``
    - ``is_terminal``
    - ``get_unique_state``

    See the documentation for each of these methods for more details.

    """

    def __init__(self, *args, **kwargs):
        # Attributes that should not be serialized when saving to a JSON file
        self._non_serializable_attrs = ["_non_serializable_attrs"]

    @abstractmethod
    def get_actions(self, state: Hashable) -> Iterable[Any]:
        """Get the possible actions that can be taken from the current state. A terminal
        state should return an empty list. List entries should be unique.

        **Warning**: Actions should never form a cycle. That is, an action should always
        result in a new state, and multiple actions should never lead back to the same
        state. Many of the features of the package, including MCTS, assume that no loops
        are present.

        Args:
            state (Hashable): The current state of the game.

        Returns:
            Iterable[Any]: A list of actions.
        """
        pass

    @abstractmethod
    def do_action(self, state: Hashable, action: Any) -> Hashable:
        """Given a state and an action, return the new state that results. Should be
        deterministic. The resulting state ID does not need to be unique (for example, it
        may not take symmetries into account).

        Args:
            state (Hashable): The current state of the game.
            action (Any): The action to take.

        Returns:
            Hashable: The new state.
        """
        pass

    @abstractmethod
    def is_terminal(self, state: Hashable) -> bool:
        """Given a state, return whether it is a terminal state. A terminal state is one
        where no further actions can be taken.

        Args:
            state (Hashable): The current state.

        Returns:
            bool: Whether the state is terminal.
        """
        pass

    @abstractmethod
    def get_unique_state(self, state: Hashable) -> Hashable:
        """Given a state, return a unique representation of that state. 
        
        This is used to determine whether two ``state`` IDs represent the same topology. 
        For example, two assembly paths may lead to state IDs that are isomorphic 
        "mirror images" of each other. In this case, ``get_unique_state`` should return 
        the same ID for both.

        Args:
            state (Hashable): The current state.

        Returns:
            Hashable: A unique representation of the state.
        """
        pass

    def has_pattern(self, state: Hashable, pattern: Hashable) -> bool:
        """Optional method to check if a given state contains a particular sub-pattern. 
        This is useful for finding specific beneficial patterns, or motifs.

        Args:
            state (Hashable): The current state.
            pattern (Hashable): The pattern to search for.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            bool: Whether the pattern is present.
        """
        raise NotImplementedError

    def get_undo_actions(self, state: Hashable) -> Iterable[Any]:
        """(Experimental) Get all actions that can be undone from the given state."""
        raise NotImplementedError

    def undo_action(self, state: Hashable, action: Any) -> Hashable:
        """(Experimental) Undo one action from the given state."""
        raise NotImplementedError

    def _get_dict_serializable(self) -> dict:
        """Get a dictionary of all attributes that can be serialized. Attributes listed
        in ``_non_serializable_attrs`` will be excluded.

        Returns:
            dict: Dictionary of attribute names and values.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in self._non_serializable_attrs
        }

    def to_dict(self) -> dict:
        """Convert the grammar to a dictionary that can be serialized to JSON. Useful for
        saving the grammar to a file or for transferring it over a network.
        
        The ``__grammar_cls__`` key is added to the dictionary to indicate the name of the
        class that should be used to re-instantiate the grammar.

        Returns:
            dict: A dictionary representation of the grammar.
        """
        return {
            "__grammar_cls__": self.__class__.__name__,
            **self._get_dict_serializable(),
        }
