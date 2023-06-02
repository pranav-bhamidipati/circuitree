from functools import cached_property
from itertools import product
import numpy as np
from typing import Iterable
from .circuitree import CircuiTree

__all__ = ["SimpleNetworkTree"]

""" 
Classes for modeling different types of circuits.

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


class SimpleNetworkTree(CircuiTree):
    def __init__(
        self, components: Iterable[Iterable[str]], interactions: Iterable[str], **kwargs
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
        components, interactions = genotype.split("::")
        prefix = ""
        if components.startswith("*"):
            prefix = "*"
            components = components[1:]

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
