"""Coordinate-column permutation between world (x-first) and mesh axis order.

The platform mesh-axis order is **z first**, ``(z, x)`` in 2-D, ``(z, x, y)``
in 3-D (the ``(nz, nx[, ny])`` TensorMesh convention; see
:class:`~geobrain.mesh.capabilities.GeometryMesh`). External tools
(meshers, station files, most plotting) speak x-first ``(x[, y], z)``. The
mismatch is a silent-corruption trap: a transposed point cloud produces a
plausible-looking mesh with depth in the x column. These two helpers (and the
``axes=`` keyword on the :class:`UnstructuredMesh` builders) are the single
audited home for that permutation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from ..core.errors import GeoBrainError

__all__ = ["apply_axes_spec", "mesh_axes_to_xyz", "xyz_to_mesh_axes"]


def _check_points(points: torch.Tensor, who: str) -> int:
    if not isinstance(points, torch.Tensor) or points.dim() != 2 or (
        int(points.shape[1]) not in (2, 3)
    ):
        raise GeoBrainError(
            "points must be an (n, 2) or (n, 3) coordinate tensor",
            object_name=who, field="points",
            expected="(n, 2|3) Tensor",
            actual=tuple(points.shape) if isinstance(points, torch.Tensor)
            else type(points).__name__,
        )
    return int(points.shape[1])


def xyz_to_mesh_axes(points: torch.Tensor) -> torch.Tensor:
    """World-order columns → mesh-axis order.

    ``(x, z) → (z, x)`` in 2-D; ``(x, y, z) → (z, x, y)`` in 3-D. Returns a
    view-like permuted copy (``points[:, perm]``); dtype/device untouched.
    """
    n_dim = _check_points(points, "xyz_to_mesh_axes")
    perm = [1, 0] if n_dim == 2 else [2, 0, 1]
    return points[:, perm]


def mesh_axes_to_xyz(points: torch.Tensor) -> torch.Tensor:
    """Mesh-axis-order columns → world order (inverse of :func:`xyz_to_mesh_axes`).

    ``(z, x) → (x, z)`` in 2-D; ``(z, x, y) → (x, y, z)`` in 3-D.
    """
    n_dim = _check_points(points, "mesh_axes_to_xyz")
    perm = [1, 0] if n_dim == 2 else [1, 2, 0]
    return points[:, perm]


def _parse_axes_spec(axes: str, n_dim: int, who: str) -> list[tuple[str, float]]:
    """``"xy-z"`` → ``[("x", +1), ("y", +1), ("z", -1)]`` with full validation."""
    letters = ("z", "x") if n_dim == 2 else ("z", "x", "y")
    tokens: list[tuple[str, float]] = []
    sign = 1.0
    pending_sign = False
    for ch in axes:
        if ch == "-":
            if pending_sign:
                break  # double sign: fall through to the error below
            sign, pending_sign = -1.0, True
            continue
        tokens.append((ch, sign))
        sign, pending_sign = 1.0, False
    seen = [name for name, _ in tokens]
    if (
        pending_sign
        or len(tokens) != n_dim
        or sorted(seen) != sorted(letters)
    ):
        raise GeoBrainError(
            "axes must name each mesh axis exactly once, optionally sign-"
            "prefixed, e.g. 'zxy' (mesh order, default), 'xyz' (world "
            "order), 'xy-z' (world columns with elevation, negated into "
            "depth-down)",
            object_name=who, field="axes",
            expected=f"signed permutation of {''.join(letters)!r}",
            actual=axes,
        )
    return tokens


def apply_axes_spec(
    points: torch.Tensor, axes: str, who: str = "apply_axes_spec"
) -> torch.Tensor:
    """Reorder (and optionally negate) coordinate columns into mesh order.

    ``axes`` names, per INPUT column, the platform axis that column holds; a
    ``-`` prefix negates the column (the elevation → depth-down flip). The
    permutation and the sign flip happen in one audited step, so external
    frames map without hand-stacked columns:

    - ``"zxy"``: columns already ``(z, x, y)``; pass-through (builder default).
    - ``"xyz"``: world order ``(x, y, z_down)``.
    - ``"xy-z"``: world columns carrying ELEVATION (most GIS tools):
      ``(x, y, z_up)`` → ``(z_down, x, y)``.
    - 2-D: ``"zx"`` / ``"xz"`` / ``"x-z"`` analogously.

    Works for any ``(n, 2|3)`` coordinate tensor (vertices, electrodes,
    stations), the same spec that builds the mesh converts its survey.

    Args:
        points: ``(n, dim)`` coordinate table in the SOURCE frame.
        axes: signed spec naming which platform axis each input column
            holds (e.g. ``"xy-z"``; ``-`` negates the column).
        who: caller name woven into error messages.
    """
    n_dim = _check_points(points, who)
    tokens = _parse_axes_spec(axes, n_dim, who)
    order = ("z", "x") if n_dim == 2 else ("z", "x", "y")
    column_of = {name: idx for idx, (name, _) in enumerate(tokens)}
    sign_of = {name: sign for name, sign in tokens}
    out = points[:, [column_of[axis] for axis in order]]
    signs = [sign_of[axis] for axis in order]
    if any(sign < 0 for sign in signs):
        out = out * points.new_tensor(signs)
    return out


def resolve_axes_kwarg(points: torch.Tensor, axes: str, who: str) -> torch.Tensor:
    """Shared ``axes=`` handling for the UnstructuredMesh builders.

    ``"zxy"`` (default), columns already in mesh-axis order, pass through;
    ``"xyz"``: world-order columns; any signed permutation (e.g. ``"xy-z"``
    for world columns carrying elevation) via :func:`apply_axes_spec`.
    """
    if axes == "zxy":
        return points
    if axes == "xyz":
        return xyz_to_mesh_axes(points)
    return apply_axes_spec(points, axes, who)
