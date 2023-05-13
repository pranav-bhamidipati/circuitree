from pathlib import Path

import numpy as np

from hex_tissue import HexTissue
from reaction_graph import ReactionGraph
from utils import *

import matplotlib.pyplot as plt
from viz import *

transceiver = ReactionGraph.from_toml(
    "../models/transceiver_circuit.toml",
    param_file="../models/transceiver_params.toml",
    initial_file="../models/transceiver_initial.toml",
)

tissue = HexTissue.from_circuit_model(
    transceiver,
    hex_kwargs=dict(
        rows=6,
        cols=20,
        periodic=(False, True),
        cell_type_ratio=dict(n_left=12, left_type="Sender"),
    ),
)

rxn_name, component = next(iter(tissue.interface_reactions.items()))
tissue.initialize(dt=0.001)

# print()
# _ = component.reaction_func(tissue.s0)
# Adj = component.interface_Adj
# print()

nt = 1000

res = tissue.integrate(nt)

save = True
animate = False
dpi = 300
save_dir = Path("../figures/tmp")

figsize = None
figsize = (5, 4)
prows = 4
pcols = 1

ct_cmap = plt.get_cmap("Set2")
cell_types = tissue.cell_type_indices

n_frames = 10
frames = get_frames(res.nt, n_frames)
suptitles = CallableAsEnumerable(
    lambda step: f"Time = {res.t[step]:.3f} ({step + 1}/{nt})"
)

fig, axs = plt.subplots(prows, pcols, figsize=figsize)
axs = np.array(axs).ravel().tolist()  # make sure it's a flat 1-d list

@animate_over_kwargs(fig=fig, ax_or_axs=axs, iter_kwargs=("var", "suptitle"))
def animate_hex(var, suptitle, axs, **kw):
    for i in range(prows * pcols):
        plot_hex_sheet(ax=axs[i], var=var[:, i], **kw)
        axs[i].set_title(f"{tissue.species[i]}")
    plt.suptitle(suptitle)

if save:
    animate_hex(
        save_dir=save_dir,
        filename="test.mp4",
        n_frames=n_frames,
        var=res.s_t,
        suptitle=suptitles,
        X=tissue.lattice,
        cell_types=cell_types,
        ctype_cmap=ct_cmap,
        fps=3,
    )

#     if save:
#         save_dir.mkdir(exist_ok=True)
#         path = save_dir.joinpath(f"TC_step{step}.png")
#         print(f"Writing to: {path.resolve().absolute()}")
#         plt.savefig(path, dpi=dpi)

# animate_hex = animate_over_kwargs(
#     plot_hex_species, fig, axs, iter_kwargs=("var", "suptitle")
# )
