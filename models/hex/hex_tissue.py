from dataclasses import dataclass
from copy import deepcopy, copy
from functools import partial
from itertools import chain
from typing import Any, Iterable, Optional, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import trange

from reaction_graph import ReactionGraph, ReactionBank
from component import CircuitComponent
from utils import *
from hex_geometry import *

_DefaultHex = {
    "rows": 20,
    "cols": 20,
    "cell_type_ratio": "equal",
}


@dataclass(frozen=True)
class IntegrationResults:
    dt: float
    nt: int
    t: np.ndarray[np.float_]
    s_0: np.ndarray[np.float_]
    s_t: np.ndarray[np.float_]


@dataclass
class HexTissue:
    lattice: np.ndarray[float]
    adjacency: np.ndarray[float | int | bool]
    species: Sequence[str]
    reactions: Mapping[str, CircuitComponent]
    reaction_bank: ReactionBank
    cell_types: Sequence[str]
    interface_reactions: Optional[Mapping[str, CircuitComponent]] = None
    species_attrs: Optional[Mapping[str, CircuitComponent]] = None
    cell_type_masks: Optional[np.ndarray[bool]] = None
    cell_type_indices: Optional[np.ndarray[int]] = None
    cell_types_unique: Optional[Sequence[str]] = None
    type_adjacency: Optional[Sequence[Sequence[csr_matrix]]] = None
    parameters: Optional[dict[str, dict[str, float | int]]] = None
    initial_condition: Optional[dict[str, dict[str, float]]] = None
    rg: Optional[np.random.RandomState] = None
    seed: Optional[int] = 2023
    delim: str = "."
    species_to_idx: Optional[Mapping[str, int]] = None
    cell_type_to_idx: Optional[Mapping[str, int]] = None

    def __post_init__(self):

        if self.cell_type_indices is None:
            self.cell_type_indices = (
                pd.Series(self.cell_types).astype("category").cat.codes.values
            )

        if self.cell_types_unique is None:
            self.cell_types_unique = sorted(set(self.cell_types))

        if self.cell_type_masks is None:
            self.cell_type_masks = np.array(
                [ct == self.cell_types for ct in self.cell_types_unique]
            )

        self.n = self.lattice.shape[0]
        self.n_species = len(self.species)
        self.n_cell_types = len(self.cell_types_unique)

        if self.rg is None:
            self.rg = np.random.default_rng(self.seed)

        if getattr(self, "interface_reactions", None) is None:
            self.interface_reactions = dict()

        self.all_components: Mapping[str, CircuitComponent] = (
            self.species_attrs | self.reactions | self.interface_reactions
        )

    @staticmethod
    def _deg(*args, **kwargs):
        return deg_func_masked(*args, **kwargs)

    @staticmethod
    def _prod(*args, **kwargs):
        return prod_func_masked(*args, **kwargs)

    @staticmethod
    def _unary_reactions(*args, **kwargs):
        return unary_funcs(*args, **kwargs)

    @staticmethod
    def _rxn(*args, **kwargs):
        return reaction_func_masked(*args, **kwargs)

    @staticmethod
    def _interface_rxn(*args, **kwargs):
        return interface_reaction_func(*args, **kwargs)

    @classmethod
    def from_circuit_model(
        cls,
        model: ReactionGraph,
        reaction_bank: Optional[ReactionBank] = None,
        parameters: Optional[Mapping[str, Mapping[str, float | int]]] = None,
        initial: Optional[Mapping[str, Mapping[str, float]]] = None,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        hex_kwargs: Optional[dict] = None,
        seed: int = 2023,
        delim: str = ".",
        _interface_name: str = "_INTERFACE",
    ):

        if parameters is None:
            parameters = getattr(model, "_parameters")
        if initial is None:
            initial = getattr(model, "_initial", None)

        if hex_kwargs is None:
            hex_kwargs = {}
        hex_kwargs = deepcopy(_DefaultHex) | hex_kwargs

        # Read in the circuit model
        # Extract species and reactions, as well as their cell types
        cell_types_unique = sorted(set(c for _, c in model.nodes("compartment")))
        if _interface_name in cell_types_unique:
            cell_types_unique.remove(_interface_name)
        n_cell_types = len(cell_types_unique)

        species: list[str] = []
        species_attrs: dict[str, CircuitComponent] = {}
        reactions: dict[str, CircuitComponent] = {}
        interface_reactions: dict[str, CircuitComponent] = {}
        for n, a in model.nodes(data=True):
            component = CircuitComponent.from_attrs(a)
            if component.is_reaction:
                if component.compartment == "_INTERFACE":
                    interface_reactions[n] = component
                else:
                    reactions[n] = component
            else:
                species_attrs[n] = component

                # Append unique chemical species to list of species in the tissue
                if not component.name in species:
                    species.append(component.name)

        species = sorted(species)
        n_species = len(species)

        # Construct a lattice with adjacency matrix
        kwargs = deepcopy(hex_kwargs)
        if rows:
            kwargs["rows"] = rows
        if cols:
            kwargs["cols"] = cols

        rg = np.random.default_rng(seed)

        lattice, Adj = get_hex_grid_with_nnb_adjacency(**kwargs)
        Adj_mask = Adj != 0
        n = lattice.shape[0]

        cell_type_ratio = kwargs.get("cell_type_ratio", "equal")
        if cell_type_ratio == "equal" or isinstance(cell_type_ratio, Sequence):
            if isinstance(cell_type_ratio, str):
                _ct_ratio = np.ones(n_cell_types) / n_cell_types
            else:
                _ct_ratio = cell_type_ratio
            cell_types = rg.choice(cell_types_unique, n, p=_ct_ratio)
        elif isinstance(cell_type_ratio, Mapping):
            if set(cell_type_ratio.keys()) == (cell_types_unique):
                ratios = tuple(cell_type_ratio[c] for c in cell_types_unique)
                cell_types = rg.choice(cell_types_unique, n, p=ratios)
            elif "n_left" in cell_type_ratio:
                n_left = cell_type_ratio["n_left"]
                left_type = cell_type_ratio["left_type"]
                ratios = np.array([float(c != left_type) for c in cell_types_unique])
                ratios = ratios / ratios.sum()
                cell_types = [left_type] * n_left + rg.choice(
                    cell_types_unique, n - n_left, p=ratios
                ).tolist()
                cell_types = np.array(cell_types)

        cell_type_indices = pd.Series(cell_types).astype("category").cat.codes.values
        cell_type_masks = np.array(
            [cti == cell_type_indices for cti in range(n_cell_types)]
        )
        cell_type_adjacency: Mapping[tuple[int], csr_matrix] = {}
        for a in range(n_cell_types):
            a_mask = cell_type_indices == a
            for b in range(n_cell_types):
                b_mask = cell_type_indices == b
                ab_mask = np.logical_and.outer(a_mask, b_mask)
                ab_and_adj_mask = ab_mask & Adj_mask
                sparse_adj = csr_matrix(
                    (Adj[ab_and_adj_mask], ab_and_adj_mask.nonzero()), shape=(n, n)
                )
                cell_type_adjacency[(a, b)] = sparse_adj

        s0 = np.zeros((n, n_species), dtype=float)
        if initial is None:
            initial = {ct: {sp: 0.0 for sp in species} for ct in cell_types_unique}
        else:
            for ct, mask in zip(cell_types, cell_type_masks):
                for i, sp in enumerate(species):
                    init_val = initial[ct].get(sp, 0.0)
                    s0[mask, i] = init_val

        species_to_idx = {sp: i for sp, i in zip(species, np.argsort(species))}
        species_sorter = np.argsort(species)

        cell_type_to_idx = {
            ct: i for ct, i in zip(cell_types_unique, np.argsort(cell_types_unique))
        }
        cell_type_sorter = np.argsort(cell_types_unique)

        # Get the function to apply for each reaction
        if reaction_bank is None:
            reaction_bank = model._reaction_bank

        degradation_func, _ = reaction_bank["_degradation"]
        production_func, _ = reaction_bank["_production"]

        for name, component in species_attrs.items():
            component.cell_type_idx = cell_type_to_idx[component.compartment]
            component.ct_mask = cell_type_masks[component.cell_type_idx]
            component.species_idx = species_to_idx[component.name]

        for name, component in reactions.items():
            func, default_params = reaction_bank[component.reaction]
            component.ct_mask = cell_type_masks[cell_type_to_idx[component.compartment]]
            component._base_rxn_func = func

            lhs_names = [species_attrs[n].name for n in component.lhs_order]
            rhs_names = [species_attrs[n].name for n in component.rhs_order]
            component.lhs_indices = species_sorter[
                np.searchsorted(species, lhs_names, sorter=species_sorter)
            ]
            component.rhs_indices = species_sorter[
                np.searchsorted(species, rhs_names, sorter=species_sorter)
            ]

        for name, component in interface_reactions.items():
            func, default_params = reaction_bank[component.reaction]
            component._base_rxn_func = func

            component.lhs_which_cell = list(component.lhs_index)
            component.rhs_which_cell = list(component.rhs_index)

            lhs_cell_types = [species_attrs[n].compartment for n in component.lhs_order]
            rhs_cell_types = [species_attrs[n].compartment for n in component.rhs_order]
            lhs_names = [species_attrs[n].name for n in component.lhs_order]
            rhs_names = [species_attrs[n].name for n in component.rhs_order]

            component.lhs_indices = species_sorter[
                np.searchsorted(species, lhs_names, sorter=species_sorter)
            ]
            component.rhs_indices = species_sorter[
                np.searchsorted(species, rhs_names, sorter=species_sorter)
            ]

            cell_type_idx_0 = -1
            cell_type_idx_1 = -1
            iterator = chain(
                zip(component.lhs_index, lhs_cell_types),
                zip(component.rhs_index, rhs_cell_types),
            )
            for i, ct in iterator:
                if (cell_type_idx_0 >= 0) and (cell_type_idx_1 >= 0):
                    break
                elif i == 0:
                    cell_type_idx_0 = cell_type_to_idx[ct]
                elif i == 1:
                    cell_type_idx_1 = cell_type_to_idx[ct]

            component.cell_type_masks = (
                cell_type_masks[cell_type_idx_0],
                cell_type_masks[cell_type_idx_1],
            )
            component.interface_Adj = cell_type_adjacency[
                (cell_type_idx_0, cell_type_idx_1)
            ]

        tissue = cls(
            lattice=lattice,
            adjacency=Adj,
            species=species,
            reactions=reactions,
            reaction_bank=reaction_bank,
            cell_types=cell_types,
            species_attrs=species_attrs,
            interface_reactions=interface_reactions,
            cell_type_indices=cell_type_indices,
            cell_type_masks=cell_type_masks,
            cell_types_unique=cell_types_unique,
            type_adjacency=cell_type_adjacency,
            parameters=parameters,
            initial_condition=initial,
            rg=rg,
            delim=delim,
        )

        tissue.n_species = n_species
        tissue.species_to_idx = species_to_idx
        tissue.cell_type_to_idx = cell_type_to_idx

        tissue.update_parameters(parameters)

        return tissue

    def update_parameters(self, updates: Mapping[str, Mapping[str, float | int]]):

        degradation_func, _ = self.reaction_bank["_degradation"]
        production_func, _ = self.reaction_bank["_production"]

        new_parameters: dict[str, dict[str, float | int]] = getattr(
            self, "parameters", {}
        )
        new_parameters.update(updates)

        for name, component in self.species_attrs.items():
            component.unary_reactions = []
            deg_func = partial(
                self._deg,
                cell_type_mask=component.ct_mask,
                species_idx=component.species_idx,
                deg_rate=new_parameters[name]["_gamma"],
                deg_func=degradation_func,
            )
            component.unary_reactions.append(deg_func)

            if component.is_genetic:
                prod_func = partial(
                    self._prod,
                    cell_type_mask=component.ct_mask,
                    species_idx=component.species_idx,
                    prod_rate=new_parameters[name]["_alpha"],
                    prod_func=production_func,
                )
                component.unary_reactions.append(prod_func)

            component.reaction_func = partial(
                self._unary_reactions, unary_reactions=component.unary_reactions
            )

        for name, component in self.reactions.items():
            func, default_params = self.reaction_bank[component.reaction]
            component.param_vals = tuple(
                new_parameters[name][p] for p in default_params
            )
            component.reaction_func = partial(
                self._rxn,
                cell_type_mask=component.ct_mask,
                lhs_indices=component.lhs_indices,
                rhs_indices=component.rhs_indices,
                params=component.param_vals,
                rxn_func=component._base_rxn_func,
            )

        for name, component in self.interface_reactions.items():

            func, default_params = self.reaction_bank[component.reaction]
            component.param_vals = tuple(
                new_parameters[name][p] for p in default_params
            )
            component.reaction_func = partial(
                self._interface_rxn,
                interface_adjacency=component.interface_Adj,
                lhs_which_cell=component.lhs_which_cell,
                rhs_which_cell=component.rhs_which_cell,
                lhs_indices=component.lhs_indices,
                rhs_indices=component.rhs_indices,
                params=component.param_vals,
                rxn_func=component._base_rxn_func,
            )

        self.parameters = new_parameters

    def initialize(self, **kwargs):

        self.initialize_expression(**kwargs)
        self.initialize_integrator(**kwargs)

    def initialize_expression(
        self, initial_condition: Optional[dict[str, dict[str, float]]] = None, **kwargs
    ):

        if initial_condition is None:
            if self.initial_condition is None:
                raise ValueError("No `init` argument passed ")
            else:
                initial_condition = deepcopy(self.initial_condition)

        s0 = np.zeros((self.n, self.n_species), dtype=np.float_)
        for cti, cell_type in enumerate(self.cell_types_unique):
            for species, init_val in initial_condition[cell_type].items():
                s0[self.cell_type_masks[cti], self.species_to_idx[species]] = init_val

        self.s0 = s0
        self.s = s0.copy()

    def initialize_integrator(self, dt: float, method="euler", **kwargs):
        def _calculate_expression_velocity(s: np.ndarray[np.float_]):
            ds_dt = np.zeros_like(s)
            for term in self.all_components.values():
                ds_dt_term = term.reaction_func(s)
                ds_dt += ds_dt_term
            return ds_dt

        if method == "euler":

            def _integrator(s: np.ndarray[np.float_], dt: float):
                ds_dt = _calculate_expression_velocity(s)
                return np.maximum(0, s + ds_dt * dt)

        self._integrator = _integrator
        self.dt = dt

    def step_py(self, s: np.ndarray[np.float_]):
        return self._integrator(s, self.dt)

    def integrate(self, nt: int, progress: bool = True):

        s_t = np.zeros((nt, *self.s0.shape), dtype=np.float_)
        s = self.s0.copy()

        if progress:
            iterator = trange(1, nt, initial=1, total=nt)
        else:
            iterator = range(1, nt)

        s_t[0] = s
        for i in iterator:
            s = self.step_py(s)
            s_t[i] = s

        t = self.dt * np.arange(nt)

        return IntegrationResults(dt=self.dt, nt=nt, t=t, s_0=self.s0, s_t=s_t)

    def copy(self, deep=True):
        if deep:
            return deepcopy(self)
        else:
            return copy(self)
