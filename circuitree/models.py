from collections import Counter
from copy import copy
from functools import cached_property, lru_cache
from itertools import chain, combinations, product, permutations
import networkx as nx
import numpy as np
from typing import Callable, Generator, Iterable, Literal, Optional

from .circuitree import CircuiTree
from .grammar import CircuitGrammar
from .utils import merge_overlapping_sets

__all__ = [
    "SimpleNetworkGrammar",
    "SimpleNetworkTree",
    "DimersGrammar",
    "DimerNetworkTree",
]

"""Classes for modeling different types of circuits."""


class SimpleNetworkGrammar(CircuitGrammar):
    """A Grammar class for pairwise interaction networks.

    This class models a circuit as a set of ``components`` and pairwise
    ``interactions``. This grammar is well-suited for simple biological networks such as
    basic transcriptional networks or signaling pathways where regulation between
    components is pairwise.

    Each ``state`` of circuit assembly is encoded as a string that contains the components
    and interactions in the circuit, and each ``action`` can either add a new interaction
    or terminate the assembly.

    * A component is abbreviated by an uppercase letter. For instance, Mdm2 and p53 would
      be ``M`` and ``P``, respectively). Multiple components are concatenated (``MP``).
    * Each type of interaction is abbreviated as a lowercase letter. For instance,
      activation and inhibition would be abbreviated as ``a`` and ``i``.
    * Each pairwise interaction is represented by three characters, the two components
      involved and the type of interaction. For instance, ``MPi`` means "MdM2 inhibits
      p53". Multiple interactions are separated by ``_``.
    * The ``state`` is a two-part string ``<components>::<interactions>``.
    * A terminal ``state`` starts with ``*``.

    For example, the ``state`` string ``*MP::MPi_PMa`` represents the MdM2-p53 negative
    feedback circuit, where MdM2 inhibits p53 and p53 activates MdM2. The ``*`` indicates
    that the circuit has been fully assembled.

    Args:
        components (str | Iterable[str]): _description_
        interactions (str | Iterable[str]): _description_
        max_interactions (int, optional): The maximum number of interactions
            allowed in a ``state``. Defaults to ``len(components) ** 2``.
        root (str, optional): The initial ``state``.
        fixed_components (Optional[str | Iterable[str]], optional): A list of components
            to ignore when computing uniqueness. In other words, these components are
            considered fixed and will not be permuted when computing unique states.
        cache_maxsize (int | None, optional): The maximum size of the LRU cache used to
            speed up the computation of unique states. Defaults to 128.

    Raises:
        ValueError: If the first character of any component or interaction type
            is not unique.

    Models a circuit as a set of ``components`` and with pairwise interactions. Each
    ``state`` of circuit assembly is encoded as a string with the following format:
    """

    def __init__(
        self,
        components: str | Iterable[str],
        interactions: str | Iterable[str],
        max_interactions: Optional[int] = None,
        root: Optional[str] = None,
        fixed_components: Optional[list[str]] = None,
        cache_maxsize: int | None = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if len(set(c[0].upper() for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0].lower() for c in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        self.root = root
        self.components = list(components)
        self.component_map = {c[0].upper(): c for c in self.components}
        self.interactions = list(interactions)
        self.interaction_map = {ixn[0].lower(): ixn for ixn in self.interactions}

        if max_interactions is None:
            self.max_interactions = len(self.components) ** 2  # all possible edges
        else:
            self.max_interactions = max_interactions

        self.fixed_components = list(fixed_components or [])

        # Allow user to specify a cache size for the get_interaction_recolorings method.
        # This method is called frequently during search, and evaluation can become a
        # bottleneck for large spaces. Caching the results of this method can
        # significantly speed up search, but cache size is limited by system memory.
        self.cache_maxsize = cache_maxsize
        self.get_interaction_recolorings: Callable[[str], list[str]] = lru_cache(
            maxsize=self.cache_maxsize
        )(self._get_interaction_recolorings)

        # Attributes that should not be serialized when saving the object to file
        self._non_serializable_attrs.extend(
            [
                "component_map",
                "interaction_map",
                "edge_options",
                "component_codes",
                "_recolor",
                "get_interaction_recolorings",
            ]
        )

    @property
    def _recolorable_components(self) -> list[str]:
        """List of single-character component codes to permute when checking for
        uniqueness."""
        return [
            c[0].upper()
            for c in self.components
            if c not in self.fixed_components
            and c[0].upper() not in self.fixed_components
        ]

    def __getstate__(self):
        result = copy(self.__dict__)

        # We need to remove the lru_cache object because it is not serializable. We
        # will re-initialize it in the __setstate__ method.
        result["get_interaction_recolorings"] = NotImplemented
        return result

    def __setstate__(self, state):
        self.__dict__ = state

        # Re-initialize the lru_cache object
        self.get_interaction_recolorings = lru_cache(maxsize=self.cache_maxsize)(
            self._get_interaction_recolorings
        )

    @cached_property
    def _edge_options(self):
        """List of possible interactions for each ordered pair of components."""
        return [
            [
                c1[0].upper() + c2[0].upper() + ixn[0].lower()
                for ixn in self.interactions
            ]
            for c1, c2 in product(self.components, self.components)
        ]

    @cached_property
    def component_codes(self) -> set[str]:
        """Set of single-character uppercase component codes."""
        return set(c[0].upper() for c in self.components)

    def get_actions(self, genotype: str) -> Iterable[str]:
        """Returns the actions that can be taken from a given ``state``, or ``genotype``.

        Possible actions are adding a new component, adding a new interaction, or
        terminating the assembly process. Checks to make sure the necessary components
        are present and the circuit would remain fully connected.

        Args:
            genotype (str): The current state of the circuit assembly.

        Returns:
            Iterable[str]: A list of valid actions that can be taken.
        """

        # If terminal already, no actions can be taken
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        # Get the components and interactions in the current genotype
        components_joined, interactions_joined = genotype.strip("*").split("::")
        components = set(components_joined)
        interactions = set(ixn[:2] for ixn in interactions_joined.split("_") if ixn)
        n_interactions = len(interactions)

        if n_interactions >= self.max_interactions:
            return actions

        # Add components if possible
        actions.extend([c for c in self.component_codes if c not in components])

        # No interactions yet, any pair of components can be connected
        if n_interactions == 0:
            possible_edge_options = [
                grp
                for grp in self._edge_options
                if grp and set(grp[0][:2]).issubset(components)
            ]
            return list(chain.from_iterable(possible_edge_options))

        # Otherwise, add valid interactions
        for action_group in self._edge_options:
            if action_group:
                c_ij = action_group[0][:2]
                c_i, c_j = c_ij

                # Make sure that the circuit would still be fully connected after adding
                # an interaction between c_i and c_j
                has_necessary_components = c_i in components and c_j in components
                connected_to_current_edges = (
                    c_i in interactions_joined or c_j in interactions_joined
                )
                no_existing_edge = c_ij not in interactions
                if (
                    has_necessary_components
                    and connected_to_current_edges
                    and no_existing_edge
                ):
                    actions.extend(action_group)

        return actions

    def do_action(self, genotype: str, action: str) -> str:
        """Applies the given action to the current circuit assembly state, or
        ``genotype``.

        The action can be:

        - A single character: Add a new component to the circuit.
        - Three characters: Add a new interaction between two components.
        - ``*terminate*``: Terminate the assembly process.

        Args:
            genotype (str): The current assembly state.
            action (str): The action to be applied.

        Returns:
            str: The new state (``genotype``) after applying the action to given state.

        Examples:
            >>> grammar = SimpleNetworkGrammar(...)
            >>> gen1 = "MP::"  # Initial state with two components
            >>> gen2 = grammar.do_action(gen1, "MPi")  # Add inhibition interaction
            >>> print(gen2)
            "MP::MPi"
            >>> gen3 = grammar.do_action(gen2, "Q")  # Add a new component
            >>> print(gen3)
            "MPQ::MPi"
            >>> gen4 = grammar.do_action(gen3, "*terminate*")  # Terminate the assembly
            >>> grammar.is_terminal(gen4)
            True
        """
        if action == "*terminate*":
            new_genotype = "*" + genotype
        else:
            components, interactions = genotype.split("::")
            if len(action) == 1:
                new_genotype = "".join([components, action, "::", interactions])
            elif len(action) == 3:
                if interactions:
                    delim = "_"
                else:
                    delim = ""
                new_genotype = "::".join([components, interactions + delim + action])
        return new_genotype

    def get_undo_actions(self, genotype: str) -> Iterable[str]:
        if genotype == self.root:
            return []
        if self.is_terminal(genotype):
            return ["*undo_terminate*"]

        undo_actions = []
        components, interactions_joined = genotype.split("::")

        # Can only remove an edge if it keeps the circuit connected
        if interactions_joined:
            interactions = interactions_joined.split("_")
            components_in_ixn = [set(ixn[:2]) for ixn in interactions_joined.split("_")]
            for i, ixn in enumerate(interactions):
                # If we remove this interaction, will the circuit remain connected?
                distinct_sets_without_ixn = merge_overlapping_sets(
                    components_in_ixn[:i] + components_in_ixn[i + 1 :]
                )

                # If there is just one distinct set of weakly connected components,
                # then the circuit will remain connected. If there are more than one,
                # then the circuit will be disconnected. If there are zero, then the
                # circuit will have no interactions (also allowed).
                if len(distinct_sets_without_ixn) < 2:
                    undo_actions.append(ixn)

        # Can only remove a component if it has no edges and we have more components
        # than the root circuit
        if len(components) - len(self.root.split("::")[0]):
            undo_actions.extend(set(components) - set(interactions_joined))

        return undo_actions

    def undo_action(self, genotype: str, action: str) -> str:
        if action == "*undo_terminate*":
            if genotype.startswith("*"):
                return genotype[1:]
            else:
                raise ValueError(
                    f"Cannot undo termination on a non-terminal genotype: {genotype}"
                )

        components, interactions_joined = genotype.split("::")
        if len(action) == 1:
            if action in components:
                components = components.replace(action, "")
        elif len(action) == 3:
            interactions = interactions_joined.split("_")
            interactions.remove(action)
            interactions_joined = "_".join(interactions)
        return "::".join([components, interactions_joined])

    @staticmethod
    def is_terminal(genotype: str) -> bool:
        """Checks if the given assembly state (``genotype``) is a terminal state. A
        terminal state starts with an asterisk ``*``.

        Args:
            genotype (str): An assembly state.

        Returns:
            bool: Whether the state is terminal.
        """
        return genotype.startswith("*")

    @cached_property
    def _recolor(self):
        """List of dictionaries representing all possible permutations of components"""
        return [
            dict(zip(self._recolorable_components, p))
            for p in permutations(self._recolorable_components)
        ]

    @staticmethod
    def _recolor_string(mapping: dict[str, str], string: str):
        """Recolors a state string (renames components) using a given mapping."""
        return "".join([mapping.get(c, c) for c in string])

    def _get_interaction_recolorings(self, interactions: str) -> list[str]:
        """Returns all possible recolorings of the interactions in a given state."""
        interaction_recolorings = []
        for mapping in self._recolor:
            recolored_interactions = sorted(
                [self._recolor_string(mapping, ixn) for ixn in interactions.split("_")]
            )
            interaction_recolorings.append("_".join(recolored_interactions).strip("_"))

        return interaction_recolorings

    @lru_cache
    def get_component_recolorings(self, components: str) -> list[str]:
        """Returns all possible recolorings of the components in a given state."""
        component_recolorings = []
        for mapping in self._recolor:
            recolored_components = "".join(
                sorted(self._recolor_string(mapping, components))
            )
            component_recolorings.append(recolored_components)

        return component_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        """Returns all possible recolorings (equivalent string representations) of a
        given state."""
        prefix = "*" if self.is_terminal(genotype) else ""
        components, interactions = genotype.strip("*").split("::")
        rcs = self.get_component_recolorings(components)
        ris = self.get_interaction_recolorings(interactions)
        recolorings = [f"{prefix}{rc}::{ri}" for rc, ri in zip(rcs, ris)]

        return recolorings

    def get_unique_state(self, genotype: str) -> str:
        """Returns a unique representation of a given state.

        This method is used to determine whether two ``state`` IDs represent the same
        topology, after accounting for component renaming and reording of the
        interactions. For example, different sequences of actions may lead to different
        ``state`` IDs that are isomorphic mirror images, or "recolorings" of each other.
        This method picks the alphabetically first representation over all possible
        recolorings.

        Args:
            genotype (str): The current state.

        Returns:
            str: The unique representation of the state.
        """
        return min(self.get_recolorings(genotype))

    def has_pattern(self, state: str, pattern: str):
        """Checks if a given state contains a particular sub-pattern.

        The ``pattern`` should be a string that contains only interactions (separated by
        underscores) and no components. This method checks if the interactions in the
        ``pattern`` are a subset of the interactions in the given ``state``.

        Args:
            state (str): The current state.
            pattern (str): The sub-pattern to search for.

        Returns:
            bool: Whether the state contains the given pattern.

        Raises:
            ValueError: If the pattern contains components or if the state does not
                contain both components and interactions.
        """

        if ("::" in pattern) or ("*" in pattern):
            raise ValueError(
                "Pattern code should only contain interactions, no components"
            )
        if "::" not in state:
            raise ValueError(
                "State code should contain both components and interactions"
            )

        interaction_code = state.split("::")[1]
        if not interaction_code:
            return False
        state_interactions = set(interaction_code.split("_"))

        for recoloring in self.get_interaction_recolorings(pattern):
            pattern_interactions = set(recoloring.split("_"))
            if pattern_interactions.issubset(state_interactions):
                return True
        return False

    @staticmethod
    def parse_genotype(genotype: str, nonterminal_ok: bool = False):
        """Parses a genotype string into its components, activations, and inhibitions.

        Args:
            genotype (str): The genotype string to parse.
            nonterminal_ok (bool, optional): Whether to allow non-terminal genotypes.
                Defaults to False.

        Returns:
            tuple[str, np.ndarray, np.ndarray]: The components, activations, and
                inhibitions in the genotype. The activations and inhibitions are
                represented as numpy arrays with two columns, where each row is a pair of
                indices representing the interacting components.

        Raises:
            ValueError: If the genotype is nonterminal and ``nonterminal_ok`` is False.
        """

        if not genotype.startswith("*") and not nonterminal_ok:
            raise ValueError(
                f"Assembly incomplete. Genotype {genotype} is not a terminal genotype."
            )
        components, interaction_codes = genotype.strip("*").split("::")
        component_indices = {c: i for i, c in enumerate(components)}

        interactions = [i for i in interaction_codes.split("_") if i]

        activations = []
        inhbitions = []
        for left, right, ixn in interactions:
            if ixn.lower() == "a":
                activations.append((component_indices[left], component_indices[right]))
            elif ixn.lower() == "i":
                inhbitions.append((component_indices[left], component_indices[right]))

        activations = np.array(activations, dtype=np.int_)
        inhbitions = np.array(inhbitions, dtype=np.int_)

        return components, activations, inhbitions


class SimpleNetworkTree(CircuiTree):
    """A convenience subclass of CircuiTree that uses SimpleNetworkGrammar by default.

    See CircuiTree and SimpleNetworkGrammar for more details.
    """

    def __init__(
        self,
        grammar: Optional[SimpleNetworkGrammar] = None,
        components: Iterable[Iterable[str]] = None,
        interactions: Iterable[str] = None,
        max_interactions: Optional[int] = None,
        root: Optional[str] = None,
        exploration_constant: float | None = None,
        seed: int = 2023,
        graph: nx.DiGraph | None = None,
        tree_shape: Optional[Literal["tree", "dag"]] = None,
        compute_unique: bool = True,
        fixed_components: Optional[list[str]] = None,
        **kwargs,
    ):
        if grammar is None:
            grammar = SimpleNetworkGrammar(
                components=components,
                interactions=interactions,
                max_interactions=max_interactions,
                root=root,
                fixed_components=fixed_components,
            )
        super().__init__(
            grammar=grammar,
            root=root,
            exploration_constant=exploration_constant,
            seed=seed,
            graph=graph,
            tree_shape=tree_shape,
            compute_unique=compute_unique,
            **kwargs,
        )


class DimersGrammar(CircuitGrammar):
    """**(Experimental)** A grammar class for the design space of dimerizing TF networks.
    Intended to represent dimerizing regulatory networks such as the MultiFate system.

    Each circuit consists of transcriptional ``components`` and ``regulators``.
    Components are TFs that may be regulated by other TFs, while regulators are TFs that
    are never regulated themselves. Any pair of components or regulators could
    potentially form a dimer to regulate a third component. Therefore, each interaction
    in this circuit is a "triplet" of two monomer TFs and a target TF.

    The state ID string for a circuit topology is similar to ``SimpleNetworkGrammar``,
    with some modifications.

    * Components and regulators are represented by single uppercase characters
    * Interactions are represented by a 4-character string
        - Characters 1-2 (uppercase): the dimerizing species (components and/or
          regulators). Order does not matter.
        - Character 3 (lowercase): the type of regulation upon binding (for example,
          ``a`` for activation)
        - Character 4 (uppercase): the target of regulation (a component)
    * Multiple interactions are separated by underscores ``_``
    * The ``state`` is a three-part string: ``<components>+<regulators>::<interactions>``
    * A terminal assembly state is denoted with a leading asterisk ``*``

    For example, the ``state`` string ``*AB+R::ARiB_BBaB`` represents a system of three
    TFs, ``A``, ``B``, and ``R`` where ``A-R`` heterodimers inhibit transcription of
    ``B`` and ``B-B`` homodimers activate transcription of ``B``. The asterisk ``*`` at
    the beginning means that the assembly process has finished.

    The number of distinct dimers that can regulate a given component is limited using
    the ``max_interactions_per_promoter`` argument (default of 2), and the total number
    of interactions in the circuit is limited by the ``max_interactions`` argument (no
    limit by default).
    """

    def __init__(
        self,
        components: Iterable[str],
        regulators: Iterable[str],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        max_interactions_per_promoter: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if len(set(c[0].upper() for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(r[0].upper() for r in regulators)) < len(regulators):
            raise ValueError("First character of each regulator must be unique")
        if len(set(ixn[0].lower() for ixn in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        self.components = list(components)
        self.regulators = list(regulators)
        self.interactions = list(interactions)

        if max_interactions is None:
            self.max_interactions = (
                len(components) * (len(components) + len(regulators)) ** 2
            )
        else:
            self.max_interactions = max_interactions
        self.max_interactions_per_promoter = max_interactions_per_promoter

        # The following attributes/cached properties should not be serialized when
        # saving the object to file
        self._non_serializable_attrs.extend(
            [
                "_component_codes",
                "_regulator_codes",
                "edge_options",
            ]
        )

    @cached_property
    def _component_codes(self) -> set[str]:
        return set(c[0].upper() for c in self.components)

    @cached_property
    def _regulator_codes(self) -> set[str]:
        return set(r[0].upper() for r in self.regulators)

    @cached_property
    def _interaction_codes(self) -> set[str]:
        return set(r[0].lower() for r in self.interactions)

    @property
    def dimer_options(self):
        """List of possible dimers (two-character codes) of components and regulators."""

        # Homodimers
        dimers = list(zip(self._component_codes, self._component_codes))

        # Component-component heterodimers
        dimers.extend(combinations(sorted(self._component_codes), 2))

        # Component-regulator heterodimers
        dimers.extend(product(self._regulator_codes, self._component_codes))

        # Order of the monomers doesn't matter, so OK to sort them
        dimers = ["".join(sorted(d)) for d in dimers]

        return sorted(dimers)

    @cached_property
    def edge_options(self):
        """List-of-list of all possible 4-character interactions. The first level
        of the list is the triplet (which dimer regulating which target), and the
        second level is the nature of the regulation."""
        return [
            [d + ixn + c for ixn in self._interaction_codes]
            for d, c in product(self.dimer_options, self._component_codes)
        ]

    @staticmethod
    def is_terminal(genotype: str) -> bool:
        """Checks if the given assembly state (``genotype``) is a terminal state. A
        terminal state starts with an asterisk ``*``.

        Args:
            genotype (str): An assembly state.

        Returns:
            bool: Whether the state is terminal.
        """
        return genotype.startswith("*")

    def get_actions(self, genotype: str) -> Iterable[str]:
        """Returns the actions that can be taken from a given ``state``, or ``genotype``.

        Possible actions include adding a new component/regulator, adding a new
        interaction between a dimer and a target component, or terminating the assembly
        process. Checks to make sure the necessary pieces are present and the circuit
        would remain fully connected after taking the action.

        Args:
            genotype (str): The current state of the circuit assembly.

        Returns:
            Iterable[str]: A list of valid actions that can be taken. Each action is a
            4-character code.
        """
        # If terminal already, no actions can be taken
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        # Get the components and interactions in the current genotype
        (
            components,
            regulators,
            interactions_joined,
        ) = self.get_genotype_parts(genotype)
        # components_joined, interactions_joined = genotype.strip("*").split("::")
        components_and_regulators = set(components + regulators)
        interaction_triplets = set(
            ixn[:2] + ixn[3] for ixn in interactions_joined.split("_") if ixn
        )
        n_interactions = len(interaction_triplets)

        # If we have reached the limit on interactions, only termination is an option
        if n_interactions >= self.max_interactions:
            return actions

        # We can add TFs not already in the genotype
        actions.extend([c for c in self._component_codes if c not in components])
        actions.extend(["+" + r for r in self._regulator_codes if r not in regulators])

        # If we have no interactions yet, don't need to check for connectedness
        if n_interactions == 0:
            possible_edge_options = [
                grp
                for grp in self.edge_options
                if grp and set(grp[:2] + grp[3]).issubset(components_and_regulators)
            ]
            return list(chain.from_iterable(possible_edge_options))

        # Otherwise, add all valid interactions
        currently_in_use = set(chain.from_iterable(interaction_triplets))
        n_regulators_per_promoter = Counter(
            ixn[3] for ixn in interactions_joined.split("_") if ixn
        )
        promoters_with_max_regulation = set(
            p
            for p in components
            if n_regulators_per_promoter[p] >= self.max_interactions_per_promoter
        )
        for group in self.edge_options:
            if group:
                dimer = group[0][:2]
                target = group[0][3]
                triplet = dimer + target

                # Check if the promoter has reached the limit for the # of regulators
                if target in promoters_with_max_regulation:
                    continue

                # Check if the dimer already regulates the target
                if triplet in interaction_triplets:
                    continue

                # Check if the necessary components/regulators exist
                if not set(triplet).issubset(components_and_regulators):
                    continue

                # Check if the proposed dimer + promoter would maintain connectedness
                if len(set(triplet) & currently_in_use) == 0:
                    continue

                actions.extend(group)

        return actions

    def do_action(self, genotype: str, action: str) -> str:
        action_len = len(action)
        components, regulators, interactions = self.get_genotype_parts(genotype)
        if action == "*terminate*":
            components = "*" + components
        elif action_len == 1:
            components = components + action
        elif action_len == 2 and action[0] == "+":
            regulators = regulators + action[1]
        elif action_len == 4:
            if interactions:
                interactions += f"_{action}"
            else:
                interactions = action
        else:
            raise ValueError(f"Invalid action: {action}")
        return components + "+" + regulators + "::" + interactions

    def get_unique_state(self, genotype: str) -> str:
        """Returns a unique representation of a given state.

        This method is used to determine whether two ``state`` IDs represent the same
        topology, after accounting for

        - Reording of the interactions
        - Renaming of components and regulators (swapping labels)

        For example, different sequences of actions may lead to different
        ``state`` IDs that are isomorphic mirror images, or "recolorings" of each other.
        This method picks the alphabetically first representation over all possible
        recolorings.

        Args:
            genotype (str): The current state.

        Returns:
            str: The unique representation of the state.
        """
        if genotype.startswith("*"):
            prefix = "*"
            genotype = genotype[1:]
        else:
            prefix = ""

        components, regulators = genotype.split("::")[0].split("+")

        # Sort components/regulators alphabetically
        components = "".join(sorted(components))
        regulators = "".join(sorted(regulators))

        # Compute the unique way to represent the set of interactions
        interactions_unique = self._get_interactions_unique(genotype)

        genotype_unique = f"{prefix}{components}+{regulators}::{interactions_unique}"
        return genotype_unique

    @lru_cache
    def _get_interactions_unique(self, genotype: str) -> str:
        return min(self._get_interaction_recolorings(genotype))

    @staticmethod
    def _get_interaction_recolorings(genotype: str) -> Generator[str]:
        preamble, interactions = genotype.split("::")
        components, regulators = preamble.strip("*").split("+")
        for component_recoloring in permutations(components):
            for regulator_recoloring in permutations(regulators):
                preamble_recoloring = "".join(
                    component_recoloring + regulator_recoloring
                )
                recoloring = str.maketrans(preamble, preamble_recoloring)
                recolored_interactions = interactions.translate(recoloring)
                recolored_and_sorted = DimersGrammar._sort_interactions(
                    recolored_interactions
                )
                yield recolored_and_sorted

    @staticmethod
    def _sort_interactions(interactions: str) -> str:
        interactions_arr = np.array(
            [list(triplet) for triplet in interactions.split("_")]
        )
        dimers_arr = np.sort(interactions_arr[:, :2], axis=1)
        interactions_arr = np.hstack([dimers_arr, interactions_arr[:, 2:]])
        interactions_arr = np.sort(interactions_arr, axis=0)
        sorted_interactions = "_".join("".join(row) for row in interactions_arr)
        return sorted_interactions

    @staticmethod
    def get_genotype_parts(genotype: str) -> tuple[str, str, str]:
        """Parses the state string (genotype) into its three parts: components,
        regulators, and interactions.

        Args:
            genotype (str): An assembly state.

        Returns:
            tuple[str, str, str]: the components, regulators, and interactions
            in the state.
        """
        components_and_regulators, interactions = genotype.strip("*").split("::")
        components, regulators = components_and_regulators.split("+")
        return components, regulators, interactions

    @staticmethod
    def parse_genotype(genotype: str):
        """Parses a genotype string into its components, regulators, activation
        interactions, and inhibition interactions.

        Args:
            genotype (str): The genotype string to parse.

        Returns:
            tuple[str, str, np.ndarray, np.ndarray]: The components, regulators,
            activations, and inhibitions. Components and regulators are provided as
            strings. The activations and inhibitions are represented as NumPy arrays with
            three columns, the indices of the two monomers involved in the interaction
            and the index of the target TF. The indices are determined by concatenating
            ``components`` and ``regulators``.
        """
        components_and_regulators, interaction_codes = genotype.strip("*").split("::")
        components, regulators = components_and_regulators.split("+")
        indices = {c: i for i, c in enumerate(components + regulators)}

        interactions = interaction_codes.split("_") if interaction_codes else []
        activations = []
        inhbitions = []
        for *monomers, regulation_type, target in interactions:
            m1, m2 = sorted(monomers)
            ixn_tuple = (
                indices[m1],
                indices[m2],
                indices[target],
            )
            if regulation_type.lower() == "a":
                activations.append(ixn_tuple)
            elif regulation_type.lower() == "i":
                inhbitions.append(ixn_tuple)
            else:
                raise ValueError(
                    f"Unknown regulation type {regulation_type} in {genotype}"
                )
        activations = np.array(activations, dtype=np.int_)
        inhbitions = np.array(inhbitions, dtype=np.int_)
        return components, regulators, activations, inhbitions

    @staticmethod
    def has_pattern(state, pattern):
        """Checks if a given state contains a particular sub-pattern.

        This method checks if the interactions in the given ``pattern`` are a subset of
        the interactions in the given ``state``, taking into account all possible
        renamings of components and regulators. The ``pattern`` should be a string that
        contains only interactions (separated by underscores).

        Args:
            state (str): The current state.
            pattern (str): The sub-pattern to search for.

        Returns:
            bool: Whether the state contains the given pattern.

        Raises:
            ValueError: If the pattern is empty or contains components/regulators or if
            the state does not contain both components and interactions.
        """

        if ("::" in pattern) or ("*" in pattern):
            raise ValueError(
                "pattern should only contain interactions, no components or regulators"
            )
        if "::" not in state or "+" not in state:
            raise ValueError(
                "state should contain components, regulators and interactions"
            )
        if pattern == "":
            raise ValueError("pattern should be non-empty")

        preamble, interactions = state.split("::")

        for ixn in pattern.split("_"):
            if any(char not in preamble for char in ixn[:2] + ixn[3]):
                raise ValueError(
                    "All components and regulators in the pattern should also be "
                    "present in the state."
                )

        if not interactions:
            return False

        state_triplets = set(ixn[:2] + ixn[3] for ixn in interactions.split("_"))
        for pattern_recoloring in DimersGrammar._get_interaction_recolorings(
            preamble + pattern
        ):
            pattern_triplets = set(
                ixn[:2] + ixn[3] for ixn in pattern_recoloring.split("_")
            )
            if pattern_triplets.issubset(state_triplets):
                return True
        return False


class DimerNetworkTree(CircuiTree):
    """A convenience subclass of CircuiTree that uses DimersGrammar by default.

    See CircuiTree and DimersGrammar for more details.
    """

    def __init__(
        self,
        components: Iterable[str],
        regulators: Iterable[str],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        max_interactions_per_promoter: int = 2,
        root: Optional[str] = None,
        exploration_constant: float | None = None,
        seed: int = 2023,
        graph: nx.DiGraph | None = None,
        tree_shape: Optional[Literal["tree", "dag"]] = None,
        compute_unique: bool = True,
        **kwargs,
    ):
        grammar = DimersGrammar(
            components=components,
            regulators=regulators,
            interactions=interactions,
            max_interactions=max_interactions,
            max_interactions_per_promoter=max_interactions_per_promoter,
            root=root,
        )
        super().__init__(
            grammar=grammar,
            root=root,
            exploration_constant=exploration_constant,
            seed=seed,
            graph=graph,
            tree_shape=tree_shape,
            compute_unique=compute_unique,
            **kwargs,
        )
