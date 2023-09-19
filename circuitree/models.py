from collections import Counter
from functools import cached_property, cache
from itertools import product, permutations
import numpy as np
from typing import Iterable, Optional
from .circuitree import CircuiTree

__all__ = ["SimpleNetworkTree", "DimerNetworkTree"]

"""Classes for modeling different types of circuits."""


class SimpleNetworkGrammar:
    """A class implementing the grammar for simple network circuits. Can be combined with
    a CircuiTree or MultithreadedCircuiTree class that implements the get_reward() method
    ."""

    def __init__(
        self,
        components: Iterable[Iterable[str]],
        interactions: Iterable[str],
        max_interactions: Optional[int] = None,
        **kwargs,
    ):
        if len(set(c[0] for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        super().__init__(**kwargs)

        self.components = components
        self.component_map = {c[0]: c for c in self.components}
        self.interactions = interactions
        self.interaction_map = {ixn[0]: ixn for ixn in self.interactions}

        if max_interactions is None:
            self.max_interactions = len(self.components) ** 2  # all possible edges
        else:
            self.max_interactions = max_interactions

        self._non_serializable_attrs.extend(
            ["component_map", "interaction_map", "edge_options", "_recolor"]
        )

    @cached_property
    def edge_options(self):
        return [
            [c1[0] + c2[0] + ixn[0] for ixn in self.interactions]
            for c1, c2 in product(self.components, self.components)
        ]

    def get_actions(self, genotype: str) -> Iterable[str]:
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        components, interactions_joined = genotype.strip("*").split("::")
        interactions = set(ixn[:2] for ixn in interactions_joined.split("_"))
        for component in self.components:
            c0 = component[0]
            if c0 not in components:
                actions.append(c0)
        for action_group in self.edge_options:
            if action_group:
                c1, c2, _ = action_group[0]
                if (
                    (c1 in components)
                    and (c2 in components)
                    and (c1 + c2) not in interactions
                ):
                    for action in action_group:
                        actions.append(action)

        return actions

    def do_action(self, genotype: str, action: str) -> str:
        if action == "*terminate*":
            new_genotype = "*" + genotype
        else:
            # Root node
            if genotype == ".":
                components = list()
                interactions = list()
            else:
                components, interactions = genotype.split("::")
            if len(action) == 1:
                new_genotype = "".join([components, action, "::", interactions])
            elif len(action) == 3:
                delim = ("", "_")[bool(interactions)]
                new_genotype = "::".join(
                    [components, delim.join([interactions, action])]
                )
        return new_genotype

    @staticmethod
    def is_terminal(genotype: str) -> bool:
        return genotype.startswith("*")

    @staticmethod
    def get_unique_state(genotype: str) -> str:
        prefix = "*" if genotype.startswith("*") else ""
        components, interactions = genotype.strip("*").split("::")
        unique_components = "".join(sorted(components))
        unique_interactions = "_".join(sorted(interactions.split("_")))
        return prefix + unique_components + "::" + unique_interactions

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


class SimpleNetworkTree(SimpleNetworkGrammar, CircuiTree):
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


class DimersMixin:
    """A mixin class for dimerizing network circuits. Can be combined with the CircuiTree
    or MultithreadedCircuiTree base classes."""

    @cached_property
    def dimer_options(self):
        dimers = set()
        monomers = self.components + self.regulators
        for c1, c2 in product(monomers, monomers):
            if c1 in self.regulators and c2 in self.regulators:
                continue
            dimers.add("".join(sorted([c1[0], c2[0]])))
        return list(dimers)

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
        if self.is_terminal(genotype):
            return list()

        # Terminating assembly is always an option
        actions = ["*terminate*"]

        components, regulators, interactions_joined = self.get_genotype_parts(genotype)
        actions.extend(list(set(self.components) - set(components)))
        regulators_to_add = set(self.regulators) - set(regulators)
        actions.extend([f"+{r}" for r in regulators_to_add])
        n_bound = Counter(ixn[3] for ixn in interactions_joined.split("_") if ixn)
        for action_group in self.edge_options:
            if action_group:
                promoter = action_group[0][-1]
                if n_bound[promoter] < self.n_binding_sites:
                    actions.extend(action_group)

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
            ixn_list = interactions.split("_") if interactions else []
            interactions = "_".join(ixn_list + [action])
        return components + "+" + regulators + "::" + interactions

    @staticmethod
    @cache
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
    def parse_genotype(genotype: str, nonterminal_ok: bool = False):
        if not genotype.startswith("*") and not nonterminal_ok:
            raise ValueError(
                f"Assembly incomplete. Genotype {genotype} is not a terminal genotype."
            )

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

    @cached_property
    def _recolorings(self):
        return [
            rc | rr
            for rc, rr in product(self._recolor_components, self._recolor_regulators)
        ]

    @staticmethod
    def _recolor(mapping, code):
        return "".join([mapping.get(char, char) for char in code])

    def get_interaction_recolorings(self, genotype: str) -> list[str]:
        *_, interactions = self.get_genotype_parts(genotype)
        interaction_recolorings = (
            "_".join(
                sorted([self._recolor(mapping, ixn) for ixn in interactions.split("_")])
            ).strip("_")
            for mapping in self._recolorings
        )
        return interaction_recolorings

    def get_component_recolorings(self, genotype: str) -> list[str]:
        components, *_ = self.get_genotype_parts(genotype)
        component_recolorings = (
            "".join(sorted(self._recolor(mapping, components)))
            for mapping in self._recolor_components
        )
        return component_recolorings

    def get_regulator_recolorings(self, genotype: str) -> list[str]:
        _, regulators, *_ = self.get_genotype_parts(genotype)
        regulator_recolorings = (
            "".join(sorted(self._recolor(mapping, regulators)))
            for mapping in self._recolor_components
        )
        return regulator_recolorings

    def get_recolorings(self, genotype: str) -> Iterable[str]:
        prefix = "*" if self.is_terminal(genotype) else ""

        _, regulators, *_ = self.get_genotype_parts(genotype)
        if regulators:
            return (
                f"{prefix}{c}+{r}::{i}"
                for c, r, i in zip(
                    self.get_component_recolorings(genotype),
                    self.get_regulator_recolorings(genotype),
                    self.get_interaction_recolorings(genotype),
                )
            )
        else:
            return (
                f"{prefix}{c}::{i}"
                for c, i in zip(
                    self.get_component_recolorings(genotype),
                    self.get_interaction_recolorings(genotype),
                )
            )

    @cache
    def get_unique_state(self, genotype: str) -> str:
        return min(self.get_recolorings(genotype))

    @cache
    def _motif_recolorings(self, motif: str) -> list[set[str]]:
        if ("+" in motif) or ("::" in motif) or ("*" in motif):
            raise ValueError(
                "Motif code should only contain interactions, no components or "
                "regulators"
            )

        return [
            set(recoloring.split("_"))
            for recoloring in self.get_interaction_recolorings(motif)
        ]

    def has_motif(self, state, motif):
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

        for motif_interactions_set in self._motif_recolorings(motif):
            if motif_interactions_set.issubset(state_interactions):
                return True
        return False


class DimerNetworkTree(DimersMixin, CircuiTree):
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
        n_binding_sites: int = 2,
        **kwargs,
    ):
        if len(set(c[0] for c in components)) < len(components):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in regulators)) < len(regulators):
            raise ValueError("First character of each component must be unique")
        if len(set(c[0] for c in interactions)) < len(interactions):
            raise ValueError("First character of each interaction must be unique")

        super().__init__(**kwargs)

        self.components = components
        self.component_map = {c[0]: c for c in self.components}
        self.regulators = regulators
        self.regulator_map = {r[0]: r for r in self.regulators}
        self.interactions = interactions
        self.interaction_map = {ixn[0]: ixn for ixn in self.interactions}

        self.n_binding_sites = n_binding_sites

        self._non_serializable_attrs.extend(
            ["component_map", "regulator_map", "interaction_map"]
        )
