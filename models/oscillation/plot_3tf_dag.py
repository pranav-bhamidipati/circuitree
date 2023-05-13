from itertools import permutations

from datetime import date
from pathlib import Path
import pickle

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import pandas as pd

from oscillation import OscillationTree

import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from matplotlib import cm

save = True

components = ["A", "B", "C"]
interactions = ["activates", "inhibits"]
root = "A::"


enum = OscillationTree(components, interactions, default_root=root, tree_shape="dag")
enum.grow_tree()
D = enum.get_complexity_graph()

...


...

# class Enumerate3TF(CircuiTree):
#     @property
#     def _recolor(self):
#         return [dict(zip(self.components, p)) for p in permutations(self.components)]

#     @staticmethod
#     def _recolor_string(mapping, string):
#         return "".join([mapping.get(c, c) for c in string])

#     def _dag_action(self, genotype: str, action: str) -> str:
#         new_genotype = super()._dag_action(genotype, action)

#         components, interactions = new_genotype.split("::")
#         recolorings = [new_genotype]
#         for mapping in self._recolor:
#             recolored_components = "".join(
#                 sorted(self._recolor_string(mapping, components))
#             )
#             recolored_interactions = sorted(
#                 [self._recolor_string(mapping, ixn) for ixn in interactions.split("_")]
#             )
#             recolored_genotype = "::".join(
#                 [recolored_components, "_".join(recolored_interactions).strip("_")]
#             )
#             recolorings.append(recolored_genotype)
#         genotype_unique = min(recolorings)

#         return genotype_unique

