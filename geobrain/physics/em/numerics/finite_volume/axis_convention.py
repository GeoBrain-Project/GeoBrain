"""
Bridge the platform ``(nz, nx, ny)`` external axis convention to the Yee
finite-volume engine's internal ``(nz, ny, nx)`` layout.

The 3-D inductive-EM Yee finite-volume machinery, :mod:`curl_curl`,
:mod:`averaging`, :mod:`receiver_sampling`, :mod:`_sigma_jacobian`, and the
TEM3D edge/face dual-volume + source builders; is written throughout in the
``nz, ny, nx = mesh.shape`` frame: mesh axis-1 is **y**, axis-2 is **x**, and
``Ex`` edges (x-polarisation) run along axis-2 with x-fastest flat indexing.

The platform-wide 3-D axis convention is ``(nz, nx, ny)`` (axis-1 = x,
axis-2 = y; see the :class:`~geobrain.mesh.TensorMesh` docstring,
``mesh.dx = spacing[1]``, and the axis-convention documentation). The
cell-centred FV static family (DC3D / IP3D / SIP) presents that contract
directly; this module is the boundary bridge for the edge-element
inductive family.

Design: the Yee curl-curl operator is **numerically x↔y symmetric**
(``Z_B = -transpose(Z_A)`` to float64, proven), so rather than rewrite the Yee
stencil / edge ordering / Jacobian in lockstep, each 3-D inductive operator
(:class:`MT3D`, :class:`FDEM3D`, :class:`TEM3D`) presents the platform
``(nz, nx, ny)`` contract at its **public boundary** and calls
:func:`to_engine_mesh` + :func:`swap_xy_cell_field` to feed the engine the
algebraically-identical ``(nz, ny, nx)`` encoding of the same physical
scenario. Because the x↔y swap is applied consistently to BOTH the mesh
geometry AND the cell field, feeding the engine reduces exactly to the
already-validated ``(nz, ny, nx)`` computation of the same physics.

Physical source / receiver coordinates and output field-component labels
are layout-independent; the engine's "x" axis carries the user's physical x
data after the swap, so they pass through unchanged (no coordinate swap, no
output relabel). Operators emit responses at stations/receivers, never a
mesh-shaped field, so no output un-transpose is needed either.

Autograd: :func:`swap_xy_cell_field` is a plain ``torch`` transpose, so the
gradient of an operator output w.r.t. the user's ``(nz, nx, ny)`` ``sigma``
flows back through the swap correctly (the engine-frame gradient is transposed
back to the user frame automatically).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.mesh import TensorMesh
from geobrain.core.errors import GeoBrainError

__all__ = ["to_engine_mesh", "swap_xy_cell_field"]


def to_engine_mesh(mesh: TensorMesh) -> TensorMesh:
    """Swap the x (axis-1) and y (axis-2) axes of a 3-D structured mesh.

    Maps a platform ``(nz, nx, ny)`` :class:`TensorMesh` (axis-1 = x, axis-2 =
    y) to the ``(nz, ny, nx)`` mesh the Yee engine consumes internally (axis-1 =
    y, axis-2 = x). Preserves uniform-vs-non-uniform construction so the MT3D
    graded-mesh path keeps its per-cell ``cell_widths`` and the FDEM3D / TEM3D
    uniform paths keep their scalar ``spacing`` (and ``is_uniform`` stays True,
    which those operators require).

    Args:
        mesh: 3-D structured :class:`TensorMesh` in the platform ``(nz, nx, ny)``
            convention.

    Returns:
        The x↔y-swapped ``(nz, ny, nx)`` :class:`TensorMesh` for the engine.
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"to_engine_mesh requires a 3D TensorMesh, got n_dim={mesh.n_dim}",
            object_name="to_engine_mesh", field="mesh.n_dim",
            expected="n_dim == 3", actual=mesh.n_dim,
        )
    nz, nx, ny = mesh.shape
    # Origin swaps with its axes and MUST survive the transpose: dropping it
    # here silently re-anchored every origin-shifted mesh at zero for the
    # whole Yee stack (primary fields, source snapping, receiver anchors).
    # An explicit zero origin is byte-identical to passing None (TensorMesh
    # contract), so the zero-origin engine paths are unchanged bitwise.
    oz, ox, oy = mesh.origin
    if mesh.is_uniform:
        dz, dx, dy = mesh.spacing
        return TensorMesh(shape=(nz, ny, nx), spacing=(dz, dy, dx),
                          origin=(oz, oy, ox))
    wz, wx, wy = mesh.cell_widths
    return TensorMesh(shape=(nz, ny, nx), cell_widths=(wz, wy, wx),
                      origin=(oz, oy, ox))


def swap_xy_cell_field(field: torch.Tensor) -> torch.Tensor:
    """Transpose a ``(nz, nx, ny)`` cell field to the engine's ``(nz, ny, nx)``.

    Autograd-transparent (a ``torch`` transpose + ``contiguous``), so a gradient
    computed in the engine frame propagates back to the user's ``(nz, nx, ny)``
    field. The same function maps either direction (it is its own inverse).

    Args:
        field: Cell-centred tensor whose last two axes are the x (axis-1) and y
            (axis-2) axes to be swapped. Must be at least 3-D.

    Returns:
        The x↔y-swapped contiguous tensor.
    """
    if field.dim() < 3:
        raise GeoBrainError(
            "swap_xy_cell_field requires a tensor with at least 3 dims "
            f"(…, nx, ny); got {field.dim()}-D",
            object_name="swap_xy_cell_field", field="field.dim",
            expected=">= 3", actual=field.dim(),
        )
    return field.transpose(-2, -1).contiguous()
