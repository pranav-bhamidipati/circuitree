from collections import Counter
from copy import copy
from functools import cached_property, lru_cache
from itertools import chain, product, permutations
import networkx as nx
import numpy as np
from typing import Callable, Iterable, Literal, Optional

from .circuitree import CircuiTree
from .grammar import CircuitGrammar

__all__ = [
    "SimpleNetworkGrammar",
    "SimpleNetworkTree",
    "DimersGrammar",
    "DimerNetworkTree",
]

"""Classes for modeling different types of circuits."""


class SimpleNetworkGrammar(CircuitGrammar):
    """A class implementing the grammar for simple network circuits. It provides
    all required methods for the CircuiTree and MultithreadedCircuiTree classes
    except the get_reward() method."""

    def __init__(
        self,
        components: Iterable[Iterable[str]],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        root: Optional[str] = None,
        cache_maxsize: int | None = 128,
        fixed_components: Optional[list[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if len(set(c[0] for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        self.root = root
        self.components = components
        self.component_map = {c[0]: c for c in self.components}
        self.interactions = interactions
        self.interaction_map = {ixn[0]: ixn for ixn in self.interactions}

        if max_interactions is None:
            self.max_interactions = len(self.components) ** 2  # all possible edges
        else:
            self.max_interactions = max_interactions

        self.fixed_components = fixed_components or []

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
    def recolorable_components(self) -> list[str]:
        return [c for c in self.components if c not in self.fixed_components]

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
    def edge_options(self):
        return [
            [c1[0] + c2[0] + ixn[0] for ixn in self.interactions]
            for c1, c2 in product(self.components, self.components)
        ]

    @cached_property
    def component_codes(self) -> set[str]:
        return set(c[0] for c in self.components)

    def get_actions(self, genotype: str) -> Iterable[str]:
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

        # If we have reached the limit on interactions, only termination is an option
        if n_interactions >= self.max_interactions:
            return actions

        # We can add at most one more component not already in the genotype
        if len(components) < len(self.components):
            actions.append(next(c for c in self.component_codes if c not in components))

        # If we have no interactions yet, don't need to check for connectedness
        elif n_interactions == 0:
            possible_edge_options = [
                grp
                for grp in self.edge_options
                if grp and set(grp[0][:2]).issubset(components)
            ]
            return list(chain.from_iterable(possible_edge_options))

        # Otherwise, add all valid interactions
        for action_group in self.edge_options:
            if action_group:
                c1_c2 = action_group[0][:2]
                c1, c2 = c1_c2

                # Add the interaction the necessary components are present, that edge
                # isn't already taken, and the added edge would be contiguous with the
                # existing edges (i.e. the circuit should be fully connected)
                has_necessary_components = c1 in components and c2 in components
                connected_to_current_edges = (
                    c1_c2[0] in interactions_joined or c1_c2[1] in interactions_joined
                )
                no_existing_edge = c1_c2 not in interactions
                if (
                    has_necessary_components
                    and connected_to_current_edges
                    and no_existing_edge
                ):
                    actions.extend(action_group)

        return actions

    def do_action(self, genotype: str, action: str) -> str:
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
                distinct_sets_without_ixn = self.merge_overlapping_sets(
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
        return genotype.startswith("*")

    @cached_property
    def _recolor(self):
        return [
            dict(zip(self.recolorable_components, p))
            for p in permutations(self.recolorable_components)
        ]

    @staticmethod
    def _recolor_string(mapping: dict[str, str], string: str):
        return "".join([mapping.get(c, c) for c in string])

    def _get_interaction_recolorings(self, interactions: str) -> list[str]:
        interaction_recolorings = []
        for mapping in self._recolor:
            recolored_interactions = sorted(
                [self._recolor_string(mapping, ixn) for ixn in interactions.split("_")]
            )
            interaction_recolorings.append("_".join(recolored_interactions).strip("_"))

        return interaction_recolorings

    @lru_cache
    def get_component_recolorings(self, components: str) -> list[str]:
        component_recolorings = []
        for mapping in self._recolor:
            recolored_components = "".join(
                sorted(self._recolor_string(mapping, components))
            )
            component_recolorings.append(recolored_components)

        return component_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        prefix = "*" if self.is_terminal(genotype) else ""
        components, interactions = genotype.strip("*").split("::")
        rcs = self.get_component_recolorings(components)
        ris = self.get_interaction_recolorings(interactions)
        recolorings = [f"{prefix}{rc}::{ri}" for rc, ri in zip(rcs, ris)]

        return recolorings

    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    def has_pattern(self, state: str, pattern: str):
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
    """
    SimpleNetworkTree
    =================
    Models a circuit as a set of components and strictly pairwise interactions between them.
    The circuit topology (also referred to as the "state" during search) is encoded using a
    string representation with the following rules:
        - Components are represented by a single uppercase character
        - Interactions are represented by a three-character string
            - The first two characters are the components involved. Order is assumed to
                matter.
            - The third character (lowercase) is the type of interaction. A common use case
                would have "a" for activation and "i" for inhibition.
        - Multiple interactions are separated by underscores ``_``
        - Components and interactions are separated by double colons ``::``
        - A terminal assembly is denoted with a leading asterisk ``*``

        For example, the following string represents a terminally assembled circuit
        that encodes an incoherent feed-forward loop:
            ``*ABC::ABa_ACa_BCi``
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
        compute_symmetries: bool = True,
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
            compute_unique=compute_symmetries,
            **kwargs,
        )


class DimersGrammar(CircuitGrammar):
    """A grammar class for circuits consisting of dimerizing molecules."""

    def __init__(
        self,
        components: Iterable[str],
        regulators: Iterable[str],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        max_interactions_per_promoter: int = 2,
        root: Optional[str] = None,
        cache_maxsize: int | None = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if len(set(c[0] for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in regulators)) < len(regulators):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        self.components = components
        self.component_map = {c[0]: c for c in self.components}
        self.regulators = regulators
        self.regulator_map = {r[0]: r for r in self.regulators}
        self.interactions = interactions
        self.interaction_map = {ixn[0]: ixn for ixn in self.interactions}

        self.max_interactions = max_interactions or np.inf
        self.max_interactions_per_promoter = max_interactions_per_promoter

        self.root = root

        # Allow user to specify a cache size for the get_interaction_recolorings method.
        # This method is called frequently during search, and evaluation can become a
        # bottleneck for large spaces. Caching the results of this method can
        # significantly speed up search, but cache size is limited by system memory.
        self.get_interaction_recolorings: Callable[[str], list[str]] = lru_cache(
            maxsize=cache_maxsize
        )(self._get_interaction_recolorings)

        # The following attributes/cached properties should not be serialized when
        # saving the object to file
        self._non_serializable_attrs.extend(
            [
                "component_map",
                "regulator_map",
                "interaction_map",
                "component_codes",
                "regulator_codes",
                "dimer_options",
                "edge_options",
                "_recolor_components",
                "_recolor_regulators",
                "get_interaction_recolorings",
            ]
        )

    @cached_property
    def component_codes(self) -> set[str]:
        return set(c[0] for c in self.components)

    @cached_property
    def regulator_codes(self) -> set[str]:
        return set(c[0] for c in self.regulators)

    @cached_property
    def dimer_options(self):
        dimers = set()
        monomers = self.components + self.regulators
        for c1, c2 in product(monomers, monomers):
            if c1 in self.regulators and c2 in self.regulators:
                continue
            dimers.add("".join(sorted([c1[0], c2[0]])))
        return list(dimers)

    @property
    def edges(self):
        return product(self.dimer_options, self.components)

    @cached_property
    def edge_options(self):
        return [
            [d + ixn[0] + c[0] for ixn in self.interactions]
            for d, c in product(self.dimer_options, self.components)
        ]

    @staticmethod
    def is_terminal(genotype: str) -> bool:
        return genotype.startswith("*")

    def get_actions(self, genotype: str) -> Iterable[str]:
        # If terminal already, no actions can be taken
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        # Get the components and interactions in the current genotype
        (
            components_joined,
            regulators_joined,
            interactions_joined,
        ) = self.get_genotype_parts(genotype)
        # components_joined, interactions_joined = genotype.strip("*").split("::")
        components = set(components_joined)
        regulators = set(regulators_joined)
        components_and_regulators = components | regulators
        interactions = set(
            ixn[:2] + ixn[3] for ixn in interactions_joined.split("_") if ixn
        )
        n_interactions = len(interactions)

        # If we have reached the limit on interactions, only termination is an option
        if n_interactions >= self.max_interactions:
            return actions

        # We can add a molecule not already in the genotype
        if len(components) < len(self.components):
            actions.append(next(c for c in self.component_codes if c not in components))
        if len(regulators) < len(self.regulators):
            actions.append(
                next("+" + r for r in self.regulator_codes if r not in regulators)
            )

        # If we have no interactions yet, don't need to check for connectedness
        elif n_interactions == 0:
            possible_edge_options = [
                options
                for (dimer, promoter), options in zip(self.edges, self.edge_options)
                if set(dimer + promoter).issubset(components_and_regulators)
            ]
            return list(chain.from_iterable(possible_edge_options))

        # Otherwise, add all valid interactions
        n_ixn_per_promoter = Counter(
            ixn[3] for ixn in interactions_joined.split("_") if ixn
        )
        for (dimer, promoter), edge_options in zip(self.edges, self.edge_options):
            if edge_options:
                # Add the interaction the necessary components are present, that edge
                # isn't already taken, and the added edge would be contiguous with the
                # existing edges (i.e. the circuit should be fully connected)
                edge = dimer + promoter
                edge_set = set(edge)
                has_necessary_components = edge_set.issubset(components_and_regulators)
                connected_to_current_edges = (
                    len(edge_set & components_and_regulators) > 0
                )
                no_existing_edge = edge not in interactions
                promoter_not_saturated = (
                    n_ixn_per_promoter[promoter] < self.max_interactions_per_promoter
                )
                if (
                    has_necessary_components
                    and connected_to_current_edges
                    and no_existing_edge
                    and promoter_not_saturated
                ):
                    actions.extend(edge_options)

        return actions

    # def get_actions(self, genotype: str) -> Iterable[str]:
    #     if self.is_terminal(genotype):
    #         return list()

    #     # Terminating assembly is always an option
    #     actions = ["*terminate*"]

    #     components, regulators, interactions_joined = self.get_genotype_parts(genotype)
    #     actions.extend(list(set(self.components) - set(components)))
    #     regulators_to_add = set(self.regulators) - set(regulators)
    #     actions.extend([f"+{r}" for r in regulators_to_add])
    #     n_bound = Counter(ixn[3] for ixn in interactions_joined.split("_") if ixn)
    #     for action_group in self.edge_options:
    #         if action_group:
    #             promoter = action_group[0][-1]
    #             if n_bound[promoter] < self.max_interactions_per_promoter:
    #                 actions.extend(action_group)

    #     return actions

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
            ixn_list = interactions.split("_") if interactions else []
            interactions = "_".join(ixn_list + [action])
        return components + "+" + regulators + "::" + interactions

    @staticmethod
    def get_unique_state(genotype: str) -> str:
        components_and_regulators, interactions = genotype.split("::")
        components, regulators = components_and_regulators.split("+")
        if components.startswith("*"):
            prefix = "*"
            components = components[1:]
        else:
            prefix = ""
        unique_components = "".join(sorted(components))
        unique_regulators = "".join(sorted(regulators))
        unique_interaction_list = []
        for ixn in interactions.split("_"):
            *monomers, logic, promoter = ixn
            unique_ixn = "".join(sorted(monomers)) + logic + promoter
            unique_interaction_list.append(unique_ixn)
        unique_interactions = "_".join(sorted(unique_interaction_list))
        unique_genotype = "".join(
            [
                prefix,
                unique_components,
                "+",
                unique_regulators,
                "::",
                unique_interactions,
            ]
        )
        return unique_genotype

    @staticmethod
    def get_genotype_parts(genotype: str):
        components_and_regulators, interactions = genotype.strip("*").split("::")
        components, regulators = components_and_regulators.split("+")
        return components, regulators, interactions

    @staticmethod
    def parse_genotype(genotype: str):
        components_and_regulators, interaction_codes = genotype.strip("*").split("::")
        components, regulators = components_and_regulators.split("+")
        component_indices = {c: i for i, c in enumerate(components + regulators)}

        interactions = interaction_codes.split("_") if interaction_codes else []
        activations = []
        inhbitions = []
        for *monomers, regulation_type, promoter in interactions:
            m1, m2 = sorted(monomers)
            ixn_tuple = (
                component_indices[m1],
                component_indices[m2],
                component_indices[promoter],
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

    @cached_property
    def _recolor_components(self):
        return [dict(zip(self.components, p)) for p in permutations(self.components)]

    @cached_property
    def _recolor_regulators(self):
        return [dict(zip(self.regulators, p)) for p in permutations(self.regulators)]

    @property
    def _recolorings(self):
        return (
            rc | rr
            for rc, rr in product(self._recolor_components, self._recolor_regulators)
        )

    @staticmethod
    def _recolor(mapping, code):
        return "".join([mapping.get(char, char) for char in code])

    def _get_interaction_recolorings(self, interactions: str) -> list[str]:
        interaction_recolorings = (
            "_".join(
                sorted([self._recolor(mapping, ixn) for ixn in interactions.split("_")])
            ).strip("_")
            for mapping in self._recolorings
        )
        return interaction_recolorings

    @lru_cache
    def get_component_recolorings(self, components: str) -> list[str]:
        component_recolorings = (
            "".join(sorted(self._recolor(mapping, components)))
            for mapping in self._recolor_components
        )
        return component_recolorings

    @lru_cache
    def get_regulator_recolorings(self, regulators: str) -> list[str]:
        regulator_recolorings = (
            "".join(sorted(self._recolor(mapping, regulators)))
            for mapping in self._recolor_components
        )
        return regulator_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        prefix = "*" if self.is_terminal(genotype) else ""

        components, regulators, interactions = self.get_genotype_parts(genotype)
        if regulators:
            return (
                f"{prefix}{c}+{r}::{i}"
                for c, r, i in zip(
                    self.get_component_recolorings(components),
                    self.get_regulator_recolorings(regulators),
                    self.get_interaction_recolorings(interactions),
                )
            )
        else:
            return (
                f"{prefix}{c}::{i}"
                for c, i in zip(
                    self.get_component_recolorings(components),
                    self.get_interaction_recolorings(interactions),
                )
            )

    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    @lru_cache
    def _pattern_recolorings(self, motif: str) -> list[set[str]]:
        if ("+" in motif) or ("::" in motif) or ("*" in motif):
            raise ValueError(
                "Motif code should only contain interactions, no components or "
                "regulators"
            )

        return [
            set(recoloring.split("_"))
            for recoloring in self.get_interaction_recolorings(motif)
        ]

    def has_pattern(self, state, motif):
        if ("::" in motif) or ("*" in motif):
            raise ValueError(
                "Motif code should only contain interactions, no components"
            )
        if "::" not in state:
            raise ValueError(
                "State code should contain both components and interactions"
            )

        interaction_code = state.split("::")[1]
        if not interaction_code:
            return False

        state_interactions = set(interaction_code.split("_"))
        for motif_interactions_set in self._pattern_recolorings(motif):
            if motif_interactions_set.issubset(state_interactions):
                return True
        return False


class DimerNetworkTree(CircuiTree):
    """
    DimerNetworkTree
    =================
    A CircuiTree for the design space of dimerizing TF networks. Intended to recapitulate
    the dimerization of zinc-finger proteins.

    Models a system of dimerizing transcription factors (e.g. zinc-fingers) that regulate
    each other's transcription. The circuit consists a set of ``components``, which
    represent transcription factors that are being regulated. Components can form homo-
    or heterodimers that bind to a component's promoter region and regulate
    transcription. There is also a set of ``regulators``, which can dimerize and regulate
    transcription but are not themselves regulated. Regulator-regulator homodimers and
    regulator-component heterodimers can act as TFs, but regulator-regulator homodimers
    are assumed to be inactive.

    The circuit topology (also referred to as the "state" during search) is encoded using
    a string representation (aka "genotype") with the following rules:
        - Components and regulators are represented by single uppercase characters
        - Interactions are represented by a 4-character string
            - Characters 1-2 (uppercase): the dimerizing species (components/regulators)
            - Character 3 (lowercase): the type of regulation upon binding
            - Character 4 (uppercase): the target of regulation (a component)
        - Components are separated from regulators by a ``+``
        - Components/regulators are separated from interactions by a ``::``
        - Interactions are separated from one another by underscores ``_``
        - A terminal assembly is denoted with a leading asterisk ``*``

        For example, the following string represents a 2-component MultiFate system that
        has not been fully assembled (lacks the terminal asterisk):

            ``AB+::AAa_BBa``

        While the following string represents a terminally assembled 2-component
        MultiFate system with a regulator L that flips the system into the A state:

            ``*AB+L::AAa_ALa_BBa_BLi``

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
        compute_symmetries: bool = True,
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
            compute_unique=compute_symmetries,
            **kwargs,
        )
