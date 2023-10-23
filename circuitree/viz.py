import logging
import numpy as np
from typing import Iterable, Optional

MPL_BACKEND = "Agg"

try:
    import matplotlib

    matplotlib.use(MPL_BACKEND)

    from matplotlib.axis import Axis
    import matplotlib.pyplot as plt

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

    kw = dict(
        radius=radius,
        width=width,
        offset=offset,
        shrink=auto_shrink,
        color=color,
        lw=lw,
        hw=hw,
        padding=padding,
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
            plot_autoactivation(x[lhs], y[lhs], theta[lhs], **kw)
        else:
            displace = [rhs, lhs] in interactions
            plot_activation(x[lhs], y[lhs], x[rhs], y[rhs], displace=displace, **kw)

    # Plot inhibitions
    for lhs, rhs in inhibitions:
        if lhs == rhs:
            plot_autoinhibition(x[lhs], y[lhs], theta[lhs], **kw)
        else:
            displace = [rhs, lhs] in interactions
            plot_inhibition(x[lhs], y[lhs], x[rhs], y[rhs], displace=displace, **kw)

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
