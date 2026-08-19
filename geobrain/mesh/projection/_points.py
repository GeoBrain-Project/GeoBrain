"""Mesh → arbitrary-points interpolation (the data-space half of projection).

``interpolation_matrix(mesh, points)`` returns a sparse
``(n_points, n_cells)`` float64 matrix ``P`` with ``P @ field.reshape(-1)``
sampling a cell-centred field at arbitrary physical points, the
standard interpolation-matrix primitive, and the single
differentiable primitive under receiver/station sampling (gradients flow
through the FIELD; point positions are geometry and stay detached, per the
platform's mesh-geometry policy).

Methods:

- ``'linear'`` (auto on structured line-carrying meshes: TensorMesh /
  CylindricalMesh): per-axis bracketing on the cell-centre lines with
  tensor-product ``2**n_dim`` corner weights. Linear-exact; points beyond
  the centre lines clamp to the border (constant extrapolation).
- ``'idw'`` (auto otherwise) / ``'nearest'``: the shared k-NN builder on
  cell centres, works for every mesh.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings

import torch

from ...core.errors import GeoBrainError
from ._builders import _build_general_weights

__all__ = ["interpolation_matrix", "interpolate_to_points"]


def _structured_lines(mesh):
    center_lines = getattr(mesh, "center_lines", None)
    if center_lines is None:
        return None
    try:
        lines = center_lines()
    except Exception as exc:
        # A present-but-failing accessor downgrades the auto method from
        # 'linear' to the general k-NN path: visibly, never silently: a
        # swallowed failure here used to hide real accessor bugs behind a
        # quietly different interpolant.
        warnings.warn(
            f"{type(mesh).__name__}.center_lines() raised "
            f"{type(exc).__name__}: {exc}, falling back to the general "
            "(k-NN) interpolation path instead of 'linear'.",
            stacklevel=3,
        )
        return None
    return [t.to(torch.float64) for t in lines]


def interpolation_matrix(
    mesh,
    points: torch.Tensor,
    *,
    method: str | None = None,
    k_neighbors: int = 4,
    idw_power: float = 1.0,
) -> torch.Tensor:
    """Sparse ``(n_points, n_cells)`` sampling matrix at physical points.

    Args:
        mesh: any GeometryMesh (cell-centred field layout, flat C-order).
        points: ``(n_points, n_dim)`` coordinates in MESH-AXIS order
            (``(z, x[, y])`` Cartesian, ``(z, r)`` cylindrical).
        method: ``None`` auto-picks ``'linear'`` on structured
            line-carrying meshes and ``'idw'`` otherwise.
        k_neighbors / idw_power: k-NN controls for the idw path.
    """
    pts = torch.as_tensor(points, dtype=torch.float64)
    if pts.dim() != 2 or int(pts.shape[1]) != mesh.n_dim:
        raise GeoBrainError(
            "points must be (n_points, n_dim) in mesh-axis order",
            object_name="interpolation_matrix", field="points",
            expected=f"(n, {mesh.n_dim})", actual=tuple(pts.shape),
        )
    lines = _structured_lines(mesh)
    if method is None:
        method = "linear" if lines is not None else "idw"
    if method not in ("linear", "idw", "nearest"):
        raise GeoBrainError(
            "interpolation_matrix method must be 'linear', 'idw' or "
            "'nearest'",
            object_name="interpolation_matrix", field="method",
            expected="'linear' | 'idw' | 'nearest' | None",
            actual=method,
        )
    if method == "linear":
        if lines is None:
            raise GeoBrainError(
                "method='linear' needs a structured mesh with per-axis "
                "centre lines (TensorMesh / CylindricalMesh); use 'idw'",
                object_name="interpolation_matrix", field="method",
                expected="structured mesh", actual=type(mesh).__name__,
            )
        return _linear_point_matrix(lines, pts, mesh.n_cells)
    return _build_general_weights(
        mesh.cell_centers(), pts, method=method,
        k_neighbors=k_neighbors, idw_power=idw_power,
    )


def _linear_point_matrix(
    lines: list[torch.Tensor], pts: torch.Tensor, n_cells: int,
) -> torch.Tensor:
    """Tensor-product per-axis linear weights (border-clamped)."""
    n_dim = len(lines)
    n_p = int(pts.shape[0])
    idx_lo, idx_hi, w_hi = [], [], []
    for a, line in enumerate(lines):
        n_a = int(line.shape[0])
        p = pts[:, a].clamp(float(line[0]), float(line[-1]))
        if n_a == 1:
            i1 = torch.zeros(n_p, dtype=torch.long)
            i0 = i1
            t = torch.zeros(n_p, dtype=torch.float64)
        else:
            i1 = torch.clamp(torch.searchsorted(line, p), 1, n_a - 1)
            i0 = i1 - 1
            t = (p - line[i0]) / (line[i1] - line[i0])
        idx_lo.append(i0)
        idx_hi.append(i1)
        w_hi.append(t)

    strides = []
    s = 1
    for line in reversed(lines):
        strides.insert(0, s)
        s *= int(line.shape[0])

    rows_l, cols_l, vals_l = [], [], []
    for corner in range(2 ** n_dim):
        flat = torch.zeros(n_p, dtype=torch.long)
        w = torch.ones(n_p, dtype=torch.float64)
        for a in range(n_dim):
            hi = (corner >> a) & 1
            flat = flat + (idx_hi[a] if hi else idx_lo[a]) * strides[a]
            w = w * (w_hi[a] if hi else 1.0 - w_hi[a])
        rows_l.append(torch.arange(n_p))
        cols_l.append(flat)
        vals_l.append(w)
    return torch.sparse_coo_tensor(
        torch.stack([torch.cat(rows_l), torch.cat(cols_l)]),
        torch.cat(vals_l), size=(n_p, n_cells), dtype=torch.float64,
    ).coalesce()


def interpolate_to_points(mesh, points, field: torch.Tensor,
                          **kw) -> torch.Tensor:
    """One-shot ``interpolation_matrix(mesh, points, **kw) @ field`` (flattened).

    Same dtype policy as :class:`~geobrain.mesh.MeshProjection`: internal
    compute is float64, the result is cast back to ``field``'s dtype, and
    complex fields are refused loudly (the cast would silently discard the
    imaginary part, interpolate real and imaginary parts separately).
    """
    if field.is_complex():
        raise GeoBrainError(
            "interpolate_to_points supports real fields only: the float64 "
            "compute would silently discard the imaginary part.",
            object_name="interpolate_to_points", field="field",
            expected="a real-valued field", actual=str(field.dtype),
        )
    w = interpolation_matrix(mesh, points, **kw)
    return (w @ field.reshape(-1).to(torch.float64)).to(field.dtype)
