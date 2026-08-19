"""Mesh wireframe and cell visualization.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from ._utils import add_colorbar, ensure_ax, to_numpy


def plot_mesh_2d(cell_centers_x, cell_centers_z, invert_yaxis=True, ax=None, **kwargs):
    """
    Plot 2D structured mesh grid lines from 1D cell-center coordinates.

    Uses ``vlines``/``hlines`` so the mesh lines stay within the mesh
    extent, earlier revisions used ``axvline``/``axhline`` which
    extended across the whole Axes and, combined with the forced
    ``set_xlim``/``set_ylim`` below, clobbered caller-supplied limits
    when the mesh was drawn on top of an existing field plot. When the
    caller passes ``ax``, we no longer force axis limits.

    Args:
        invert_yaxis: When True (default), flip the y-axis so depth
            increases downward, matches :func:`geobrain.vis.field.plot_field_2d`
            so a mesh overlay aligns with the field beneath it. Pass
            ``False`` for plan views where ``z`` is a horizontal coordinate.
    """
    cx = to_numpy(cell_centers_x)
    cz = to_numpy(cell_centers_z)
    ax_was_supplied = ax is not None
    fig, ax = ensure_ax(ax)

    def centers_to_edges(c):
        edges = np.empty(len(c) + 1)
        edges[1:-1] = (c[:-1] + c[1:]) / 2
        edges[0] = c[0] - (edges[1] - c[0])
        edges[-1] = c[-1] + (c[-1] - edges[-2])
        return edges

    xe = centers_to_edges(cx)
    ze = centers_to_edges(cz)

    defaults = dict(color="k", linewidth=0.5)
    defaults.update(kwargs)

    # Draw only within the mesh bounding box, not across the whole Axes.
    ax.vlines(xe, ze[0], ze[-1], **defaults)
    ax.hlines(ze, xe[0], xe[-1], **defaults)

    if not ax_was_supplied:
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ze[0], ze[-1])
        # Depth-down default, but only when we own the Axes. When the
        # caller supplies ``ax`` (overlay on an existing field plot) we
        # leave the orientation the underlying plot already chose, so the
        # mesh aligns with the field instead of re-flipping it.
        if invert_yaxis and not ax.yaxis_inverted():
            ax.invert_yaxis()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    return ax


def plot_mesh_quadtree(cell_bounds, levels=None, ax=None, **kwargs):
    """Plot quadtree/octree mesh cells as rectangles.

    Args:
        cell_bounds: EITHER a 2-D :class:`~geobrain.mesh.OctreeMesh`
            (recommended; its mesh-order ``cell_bounds()`` /``levels`` are
            read and permuted internally), OR a raw ``(n, 4)`` array in
            **x-first plot order** ``[x_lo, x_hi, z_lo, z_hi]``. NOTE the
            mesh contract is z-first: ``OctreeMesh.cell_bounds()`` returns
            ``[z_lo, z_hi, x_lo, x_hi]``: pass the mesh itself (or permute
            with ``bounds[:, [2, 3, 0, 1]]``) or the plot silently swaps
            depth and x under correct-looking axis labels.
        levels: optional per-cell refinement levels (colours the cells).
            Defaults to the mesh's own ``levels`` when a mesh is passed.
    """
    from geobrain.mesh import OctreeMesh

    if isinstance(cell_bounds, OctreeMesh):
        om = cell_bounds
        if om.n_dim != 2:
            raise ValueError(
                "plot_mesh_quadtree draws 2-D quadtree cells; got a "
                f"{om.n_dim}-D OctreeMesh (use vis.scene3d for 3-D)"
            )
        b = to_numpy(om.cell_bounds())            # mesh order [z0, z1, x0, x1]
        cell_bounds = b[:, [2, 3, 0, 1]]          # -> plot order [x0, x1, z0, z1]
        if levels is None:
            levels = om.levels
    cell_bounds = to_numpy(cell_bounds)
    fig, ax = ensure_ax(ax)

    patches = []
    for i in range(len(cell_bounds)):
        x0, x1, z0, z1 = cell_bounds[i]
        patches.append(Rectangle((x0, z0), x1 - x0, z1 - z0))

    if levels is not None:
        levels = to_numpy(levels)
        pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.5)
        pc.set_array(levels.astype(float))
        ax.add_collection(pc)
        add_colorbar(pc, ax, label="Level")
    else:
        pc = PatchCollection(patches, facecolor="none", edgecolor="k", linewidth=0.5, **kwargs)
        ax.add_collection(pc)

    ax.autoscale_view()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    return ax


def plot_mesh_triangles(vertices, cells, ax=None, **kwargs):
    """Plot triangular mesh wireframe.

    Args:
        vertices: ``(n_nodes, 2)`` array in **x-first plot order**, column 0
            is plotted as X, column 1 as Z. NOTE the platform mesh contract
            is z-first (``GeometryMesh``: column 0 = depth z), so
            platform-order node tables must be flipped first
            (``vertices[:, [1, 0]]``) or the plot silently swaps depth and x
            under correct-looking axis labels.
        cells: ``(n_tris, 3)`` node-index triples.
    """
    vertices, cells = to_numpy(vertices), to_numpy(cells)
    fig, ax = ensure_ax(ax)
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], cells)
    defaults = dict(color="k", linewidth=0.5)
    defaults.update(kwargs)
    ax.triplot(triang, **defaults)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    return ax


# The polygon variant with optional scalar coloring is
# :func:`geobrain.vis.field.plot_field_polygon`: a *field* plot, not a
# *mesh-structure* plot. This module hosts only mesh-geometry primitives.
