from itertools import product
from typing import Optional, Iterable, Mapping
from .circuitree import CircuiTree

__all__ = ["SimpleNetworkTree"]


class SimpleNetworkTree(CircuiTree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_node_options(self):
        return tuple(c[0] for c in self.components)

    def _get_edge_options(self):
        return [
            [c1[0] + c2[0] + ixn[0] for ixn in self.interactions]
            for c1, c2 in product(self.components, self.components)
        ]

    def get_actions(self, genotype: str) -> Iterable[str]:
        if self.is_terminal(genotype):
            return list()

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

    def _do_action(self, genotype: str, action: str) -> str:
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

    def is_terminal(self, genotype: str) -> bool:
        return genotype.startswith("*")

    @staticmethod
    def get_edge_code(genotype: str) -> str:
        return genotype.strip("*").split("::")[-1]

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
