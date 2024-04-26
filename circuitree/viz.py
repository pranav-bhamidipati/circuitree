from collections import Counter
import logging
import networkx as nx
import numpy as np
from typing import Iterable, Literal, Optional

from .circuitree import CircuiTree

MPL_BACKEND = "Agg"

try:
    import matplotlib

    matplotlib.use(MPL_BACKEND)

    from matplotlib.axis import Axis
    import matplotlib.pyplot as plt
    from matplotlib import collections as mpl_coll
    from matplotlib.patches import Rectangle

except ImportError:
    logging.warning(
        "Could not import matplotlib. Some functionality may be unavailable."
    )

__all__ = [
    "rgb2hex",
    "plot_network",
]


## Color utils


def rgb2hex(rgb):
    """Converts rgb colors to hex"""

    RGB = np.zeros((3,), dtype=np.uint8)
    for i, _c in enumerate(rgb[:3]):
        # Convert vals in [0., 1.] to [0, 255]
        if isinstance(_c, float) and (_c <= 1.0):
            c = int(_c * 255)
        else:
            c = _c

        # Calculate new values
        RGB[i] = round(c)

    return "#{:02x}{:02x}{:02x}".format(*RGB)


## Search graph visualization


def complexity_layout(
    tree: CircuiTree,
    dy: float = 0.5,
    aspect: float = 2.0,
) -> tuple[dict[str, int], dict[str, tuple[float, float]]]:
    """Returns the depth and xy coordinates of each node in the search graph when
    visualized based on complexity.

    In a complexity layout, a search graph is represented by a series of layers, where
    each layer contains nodes that have the same complexity, which is measured as the
    depth, or distance from the root node. Layers are separated in the y direction, and
    in the x direction, nodes in the same layer are sorted by the number of visits they
    have received. Terminal nodes are subsumed into their parent nonterminals.
    """

    # Convert the search graph to a complexity graph, where terminal nodes are subsumed
    # into their parent nonterminals.
    G = tree.to_complexity_graph(successes=False)

    # Get the depth of each node (distance from the root)
    depth_of_node = {}
    for i, layer in enumerate(nx.bfs_layers(tree.graph, sources=tree.root)):
        for n in layer:
            if n in G.nodes:
                depth_of_node[n] = i

    layer_counter = Counter(depth_of_node.values())
    max_layer_size = max(layer_counter.values(), default=0)

    if max_layer_size == 0:
        return np.array([], dtype=np.float64), {}

    # Get the complexity as the distance from the root to each node
    nodes = np.array(list(G.nodes))
    complexity = np.array([depth_of_node[n] for n in nodes])
    visits = np.array([G.nodes[n]["visits"] for n in nodes])
    order = np.lexsort((-visits, complexity))
    nodes = nodes[order]
    complexity = complexity[order]

    # Get y coordinates for each node based on complexity
    yvals = -dy * complexity
    height = dy * complexity.max(initial=0) + dy / 2

    # Evenly space nodes in the x direction within each layer. Sort nodes in the
    # x-direction by the number of visits they have received.
    width = height * aspect
    dx = width / max_layer_size
    xvals = np.zeros_like(yvals)
    for c, n_c in layer_counter.items():
        xvals_c = np.arange(n_c) * dx
        xvals_c = xvals_c - xvals_c.mean()
        xvals[complexity == c] = xvals_c

    pos = {n: (x, y) for n, x, y in zip(nodes, xvals, yvals)}
    return complexity, pos


def plot_complexity(
    tree: CircuiTree,
    complexity: Optional[dict[str, int]] = None,
    pos: Optional[dict[str, tuple[float, float]]] = None,
    vlim: tuple[int | float] = (None, None),
    vscale: Literal["log", "lin", "flat"] = "flat",
    figsize: tuple[float, float] = (6, 3),
    alpha: float = 0.25,
    lw: float = 1.0,
    complexity_labels: bool = True,
    plot_layers_as_blocks: bool = True,
    block_height: float = 0.075,
    block_clr: str = "lightsteelblue",
    marker_clr: str = "k",
    marker_size: float = 10,
    n_to_highlight: Optional[float] = None,
    highlight_min_visits: int = 1,
    highlight_clr: str = "tab:orange",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> None:

    # Convert the search graph to a complexity graph, where terminal nodes are subsumed
    # into their parent nonterminals.
    G = tree.to_complexity_graph(successes=False)

    if complexity is None or pos is None:
        complexity, pos = complexity_layout(tree, **kwargs)

    if len(pos) == 0:
        xvals = []
        yvals = []
    else:
        xvals, yvals = zip(*pos.values())
    xvals = np.array(xvals)
    yvals = np.array(yvals)

    # Get the limits for the number of visits - only show edges with visits in this range
    visits = np.array([v for *e, v in G.edges(data="visits")])
    if vlim[0] is None:
        vmin = max(visits.min(initial=1), 1)  # ignore zero visits
    else:
        vmin = vlim[0]
    if vlim[1] is None:
        vmax = visits.max(initial=1)
    else:
        vmax = vlim[1]

    # Decide how to normalize the edge weights - on a log/linear scale or flat
    if vscale == "log":
        log_vmin = np.log10(vmin)
        log_vrange = np.log10(vmax) - log_vmin
        weights = np.clip((np.log10(visits) - log_vmin) / log_vrange, 0, 1)
    elif vscale == "lin":
        vrange = vmax - vmin
        weights = np.clip((visits - vmin) / vrange, 0, 1)
    elif vscale == "flat":
        weights = (visits >= vmin).astype(float)

    else:
        raise ValueError(f"Invalid vscale: {vscale}. Must be 'flat', 'log', or 'lin'.")

    # Black edges with transparency (alpha) proportional to the weight
    edge_colors = np.zeros((len(weights), 4))
    edge_colors[:, 3] = alpha * weights

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    # Turn of all axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Turn off x axis ticks
    ax.set_xticks([])

    if complexity_labels:
        # Turn off y axis ticks but keep labels
        ax.set_yticks(np.unique(yvals)[::-1])
        ax.set_yticklabels([f"{d}" for d in np.unique(complexity)])
        ax.yaxis.set_ticks_position("none")
        ax.set_ylabel("Complexity")
    else:
        ax.set_yticks([])

    # set edge positions
    if plot_layers_as_blocks:
        ybias = 0.5 * block_height
    else:
        ybias = 0.0
    edge_pos = []
    for n1, n2 in G.edges:
        x1, y1 = pos[n1]
        x2, y2 = pos[n2]
        edge_pos.append(((x1, y1 - ybias), (x2, y2 + ybias)))
    edge_pos = np.array(edge_pos)

    # Draw edges
    edges = mpl_coll.LineCollection(
        edge_pos,
        colors=edge_colors,
        linewidths=lw,
        antialiaseds=(1,),
        linestyle="-",
    )
    edges.set_zorder(0)  # edges go behind nodes
    ax.add_collection(edges)

    # edges = nx.draw_networkx_edges(
    #     G,
    #     G_pos,
    #     ax=ax,
    #     edge_color=edge_colors,
    #     width=0.5,
    #     arrows=False,
    #     node_size=0,
    # )
    # edges.set_zorder(0)

    if plot_layers_as_blocks:
        # Plot a gray horizontal rectangle to represent each layer of nodes
        rects = []
        for c in np.unique(complexity):
            c_mask = complexity == c
            xmin = xvals[c_mask].min()
            xmax = xvals[c_mask].max()
            yval = yvals[c_mask][0]
            rect = Rectangle(
                (xmin, yval - 0.5 * block_height),
                width=xmax - xmin,
                height=block_height,
                color=block_clr,
                zorder=1,
            )
            rects.append(rect)
        layer_blocks = mpl_coll.PatchCollection(rects, match_original=True)
        ax.add_collection(layer_blocks)

    else:
        # Draw each node as a circle
        for n, (x, y) in pos.items():
            ax.scatter(x, y, color=marker_clr, s=marker_size, zorder=1)

    # for xmin_d, xmax_d, yval_d in zip(depth_xmins, depth_xmaxs, depth_yvals):
    #     plt.hlines(yval_d, xmin_d, xmax_d, color="gray", zorder=4, lw=1.0)

    if n_to_highlight is not None:

        # Get the states with the highest mean reward among states that were
        # visited at least "higlight_min_visits" times
        states_to_consider = (
            n
            for n in tree.terminal_states
            if tree.graph.nodes[n]["visits"] >= highlight_min_visits
        )
        highlight_states = sorted(
            states_to_consider,
            key=lambda n: (
                tree.graph.nodes[n].get("reward", 0)
                / tree.graph.nodes[n].get("visits", 1)
            ),
            reverse=True,
        )[:n_to_highlight]

        # Get the coordinates for the top states
        top_states_pos = []
        for state in highlight_states:
            nonterm = state.lstrip("*")
            if nonterm in pos:
                top_states_pos.append(pos[nonterm])
        top_states_pos = np.array(top_states_pos)

        # Plot a marker to highlight the top states
        if top_states_pos.size > 0:
            plt.scatter(
                *top_states_pos.T,
                color=highlight_clr,
                s=15,
                zorder=5,
                edgecolors="k",
                lw=0.3,
            )

    # axes_lims = xvals.min(), xvals.max(), yvals.min(), yvals.max()
    # ax.axis(axes_lims)
    ax.axis("equal")

    return fig, ax


## Plot a diagram of an N-component network (see models.SimpleNetworkTree)


def plot_network(
    names: Iterable[str],
    activations: np.ndarray[np.int_],
    inhibitions: np.ndarray[np.int_],
    cmap="tab10",
    node_shrink=0.9,
    node_lw=0.5,
    center=(0, 0),
    width=0.01,
    offset=0.4,
    auto_shrink=0.6,
    color="k",
    ec="k",
    lw=2,
    hw=0.15,
    padding=0.15,
    auto_padding: Optional[float] = None,
    fontsize=None,
    text_kwargs=None,
    plot_labels=True,
    ax: Optional[Axis] = None,
    colormap=None,
):
    """Plot an N-component network as a circular diagram.

    Names, activations, and inhibitions could be returned by e.g.
    models.SimpleNetworkTree.parse_genotype.

    """

    from matplotlib.patches import Circle

    if ax is None:
        ax = plt.gca()

    n_components = len(names)

    if n_components == 1:
        # Special case
        theta = np.array([np.pi / 2])
        x = np.array([0.0]) + center[0]
        y = np.array([0.0]) + center[1]
    elif n_components > 1:
        angle0 = np.pi / 2
        theta = np.linspace(angle0, angle0 + 2 * np.pi, n_components, endpoint=False)
        x = np.cos(theta) + center[0]
        y = np.sin(theta) + center[1]

    theta = theta % (2 * np.pi)
    xy = np.column_stack([x, y])

    if colormap is None:
        colormap = plt.get_cmap(cmap)
    colors = colormap(range(n_components))

    radius = node_shrink * _compute_radius(n_components)
    molecules = []
    for xy_i, c in zip(xy, colors):
        mlc = Circle(xy_i, radius=radius, fc=c, ec=ec, lw=node_lw)
        ax.add_patch(mlc)
        molecules.append(mlc)

    if auto_padding is None:
        auto_padding = padding

    kw = dict(
        radius=radius,
        width=width,
        offset=offset,
        shrink=auto_shrink,
        color=color,
        lw=lw,
        hw=hw,
        ax=ax,
    )
    if text_kwargs is None:
        text_kwargs = {}
    if fontsize is None:
        fontsize_kw = {}
    else:
        fontsize_kw = dict(fontsize=fontsize)
    text_kw = dict(color="w", fontsize=18) | text_kwargs | fontsize_kw

    # Plot activations
    interactions = activations.tolist() + inhibitions.tolist()
    for lhs, rhs in activations:
        if lhs == rhs:
            plot_autoactivation(x[lhs], y[lhs], theta[lhs], padding=auto_padding, **kw)
        else:
            displace = [rhs, lhs] in interactions
            plot_activation(
                x[lhs], y[lhs], x[rhs], y[rhs], displace=displace, padding=padding, **kw
            )

    # Plot inhibitions
    for lhs, rhs in inhibitions:
        if lhs == rhs:
            plot_autoinhibition(x[lhs], y[lhs], theta[lhs], padding=auto_padding, **kw)
        else:
            displace = [rhs, lhs] in interactions
            plot_inhibition(
                x[lhs], y[lhs], x[rhs], y[rhs], displace=displace, padding=padding, **kw
            )

    # Component labels
    if plot_labels:
        for mlc, label in zip(molecules, names):
            plt.text(*mlc.center, label, ha="center", va="center", **text_kw)

    # Remove axes
    plt.axis("off")
    ax.set_aspect("equal")

    return ax


def plot_activation(
    src_x,
    src_y,
    dst_x,
    dst_y,
    radius,
    *,
    color="k",
    width=0.01,
    padding=0.05,
    displace=False,
    hw=0.1,
    **kwargs,
):
    """Plot an activation arrow between two nodes"""

    src = np.array([src_x, src_y])
    dst = np.array([dst_x, dst_y])
    padding_director = (dst - src) / np.linalg.norm(dst - src)
    padding_vec = (1 + padding) * radius * padding_director
    padded_src = src + padding_vec
    padded_dst = dst - padding_vec

    # If there's an edge in the reverse direction, displace to the left
    if displace:
        displacement_director = np.array([-padding_director[1], padding_director[0]])
        padded_src += 0.75 * hw * displacement_director
        padded_dst += 0.75 * hw * displacement_director

    arrow = plt.arrow(
        *padded_src,
        *(padded_dst - padded_src),
        width=width,
        color=color,
        length_includes_head=True,
        head_width=hw,
        head_length=hw,
    )

    return arrow


def plot_inhibition(
    src_x,
    src_y,
    dst_x,
    dst_y,
    radius,
    *,
    color="k",
    width=0.01,
    hw=0.05,
    padding=0.05,
    displace=False,
    **kwargs,
):
    """Plot an activation arrow between two nodes"""

    src = np.array([src_x, src_y])
    dst = np.array([dst_x, dst_y])
    padding_director = (dst - src) / np.linalg.norm(dst - src)
    padding_vec = (1 + padding) * radius * padding_director
    padded_src = src + padding_vec
    padded_dst = dst - padding_vec

    # If there's an edge in the reverse direction, displace to the left
    if displace:
        displacement_director = np.array([-padding_director[1], padding_director[0]])
        padded_src += 0.75 * hw * displacement_director
        padded_dst += 0.75 * hw * displacement_director

    padded_src_x, padded_src_y = padded_src
    padded_dst_x, padded_dst_y = padded_dst

    head_director = np.array([padded_src_y - padded_dst_y, padded_dst_x - padded_src_x])
    head_director /= np.linalg.norm(head_director)
    head_center = np.array([padded_dst_x, padded_dst_y])
    head_src = head_center - 0.5 * hw * head_director
    head_dst = head_center + 0.5 * hw * head_director

    arrow = plt.arrow(
        padded_src_x,
        padded_src_y,
        padded_dst_x - padded_src_x,
        padded_dst_y - padded_src_y,
        width=width,
        color=color,
        length_includes_head=True,
        head_width=0,
        head_length=0,
    )

    # Inhibition arrow "head" is a bar plotted as a line segment (another arrow)
    head = plt.arrow(
        *head_src,
        *(head_dst - head_src),
        width=width,
        color=color,
        length_includes_head=True,
        head_width=0,
        head_length=0,
    )

    return head, arrow


def _autoregulation_arrow_arc(a: float, b: float, padding: float):
    """Returns the arc of an arrow from a circle to itself (the angle)"""

    if a >= (b + 1) or b >= (a + 1):
        raise ValueError("a and b constitute an invalid arrow")

    arg_angle = 2 * np.pi - 2 * np.arccos((a**2 + b**2 - 1) / (2 * a * b))
    return arg_angle * (1 - 0.5 * padding)


def _plot_autoregulation_arc(
    x,
    y,
    radius,
    theta,
    *,
    ax=None,
    offset=0.2,
    shrink=0.7,
    color="k",
    width=3,
    hw=0.1,
    padding=0.05,
):
    """Plot an activation arrow from a node to itself"""
    from matplotlib.patches import Arc

    center_to_node_director = np.array([np.cos(theta), np.sin(theta)])
    shift = 1 + offset
    head_length = 1.0 * hw

    arc_radius = shrink * radius
    arc_diameter = 2 * arc_radius
    arc_perimeter = np.pi * arc_diameter
    head_length_over_perimeter = head_length / arc_perimeter  # radians

    arc_angle = _autoregulation_arrow_arc(shift, shrink, padding)
    start_angle = (theta + 0.5 * arc_angle) % (2 * np.pi)
    start_angle = np.rad2deg(start_angle)

    end_angle = (theta - 0.5 * arc_angle) + 2 * head_length_over_perimeter
    end_angle %= 2 * np.pi
    end_angle = np.rad2deg(end_angle)

    dx, dy = shift * radius * center_to_node_director
    arc_center = np.array([x + dx, y + dy])
    arc = Arc(
        arc_center,
        arc_diameter,
        arc_diameter,
        theta1=end_angle,
        theta2=start_angle,
        ec=color,
        lw=width * 2 / 3,
    )

    end_angle = np.deg2rad(end_angle)
    end_point = arc_center + arc_radius * np.array(
        [np.cos(end_angle), np.sin(end_angle)]
    )

    ax.add_patch(arc)

    return arc_center, start_angle, end_angle, end_point, arc


def plot_autoactivation(
    x,
    y,
    theta,
    radius,
    *,
    offset=0.2,
    shrink=0.7,
    color="k",
    lw=3,
    hw=0.1,
    padding=0.05,
    ax=None,
    **kwargs,
):
    arc_center, start_angle, end_angle, end_point, arc = _plot_autoregulation_arc(
        x,
        y,
        radius,
        theta,
        ax=ax,
        offset=offset,
        shrink=shrink,
        color=color,
        width=lw,
        padding=padding,
    )

    # Arrow head points clockwise perpendicular to the arc at the endpoint
    arc_radius = shrink * radius
    end_angle += 2 * hw / (2 * np.pi * arc_radius)
    head_center = arc_center + arc_radius * np.array(
        [np.cos(end_angle), np.sin(end_angle)]
    )
    head_director = np.array([np.sin(end_angle), -np.cos(end_angle)])
    head_src = head_center - 0.5 * hw * head_director
    head_dxy = 1e-2 * head_director * radius
    head = plt.arrow(
        *head_src,
        *head_dxy,
        width=0,
        color=color,
        length_includes_head=False,
        head_width=hw,
        head_length=hw,
    )
    return arc, head


def plot_autoinhibition(
    x,
    y,
    theta,
    radius,
    *,
    width=0.01,
    offset=0.2,
    shrink=0.7,
    color="k",
    lw=3,
    hw=0.05,
    padding=0.05,
    ax=None,
    **kwargs,
):
    arc_center, start_angle, end_angle, end_point, arc = _plot_autoregulation_arc(
        x,
        y,
        radius,
        theta,
        ax=ax,
        offset=offset,
        shrink=shrink,
        color=color,
        width=lw,
        padding=padding,
    )

    head_director = np.array([np.cos(end_angle), np.sin(end_angle)])
    head_src = end_point - 0.75 * hw * head_director

    # Inhibition arrow "head" is a bar plotted as a line segment (another arrow)
    head = plt.arrow(
        *head_src,
        *(1.5 * hw * head_director),
        width=width,
        color=color,
        length_includes_head=True,
        head_width=0,
        head_length=0,
    )

    return arc, head


def _compute_radius(n_components: int):
    if n_components == 1:
        return _compute_radius(2)
    else:
        return 0.25 * np.sqrt(2 * (1 - np.cos(2 * np.pi / n_components)))
