# pyright: reportPrivateImportUsage=false
"""
3D DC resistivity.

Cell-centered finite-volume Poisson assembly routed through the
mesh-agnostic :func:`assemble_poisson_fv` seam. DC3D runs on a structured
``TensorMesh`` and on an ``OctreeMesh`` (e.g. derived from a TensorMesh via
:meth:`OctreeMesh.from_tensor_mesh`). The survey is in PHYSICAL coordinates
(electrode positions in metres, per-field ``source_x``/``source_y``/``source_z``,
convention-independent, resolved onto the platform ``(nz, nx, ny)`` mesh with
physical x on axis-1 and y on axis-2); electrodes are
resolved to mesh cells at ``_forward`` time, structured via
``floor(coord / spacing)``, octree/unstructured via argmin over
``mesh.cell_centers()``. No base grid spacing is required, so the operator
works on any :class:`ConnectivityMesh`. The Poisson system is always solved
through the pluggable sparse-solve bridge :func:`sparse_linear_solve_with_adjoint`.
The default backend is CPU ``scipy.sparse.linalg.splu`` (float64); the
autograd-opaque factorisation is wrapped by the bridge adjoint. The whole
forward (FV assembly, dipole RHS, and receiver sampling) is device-preserving,
so installing a GPU :class:`~geobrain.core.adjoint.SparseFactorSolver` (e.g.
:class:`~geobrain.core.adjoint.PcgGpuSolver`, Jacobi-PCG on CUDA) runs DC3D
END-TO-END on the GPU. With the default scipy backend the solve is CPU-only, so
a non-CPU σ raises a clear error from the bridge naming the fix (install a GPU
backend). Same architectural pattern as :mod:`dc2d`, absorbs the physics improvements
(symmetric Dirichlet pin, multi-RHS readiness, full ``potential`` field
exposure) on the sparse-FV + adjoint-solve substrate.

Additions:

- :class:`ElectrodeArray`: frozen ``(x, y, z)`` electrode container.
- :class:`DipoleDipoleSurvey`: mesh-free ABMN-quadripole survey geometry;
  :meth:`DipoleDipoleSurvey.bind` produces a :class:`BoundDipoleDipoleSurvey`
  carrying the electrode→cell trilinear / nearest-cell interpolation weights.
- :func:`build_sigma_jacobian_dc3d`: closed-form sparse Jacobian of the
  cell-centered FV Poisson stencil with respect to σ. Returned 4-tuple feeds
  directly into ``SparseLinearSolveWithSigmaGrad``.
- :class:`DC3D` gains a ``use_closed_form_sigma_jacobian: bool = False``
  constructor flag. When ``True``, the σ-gradient flows through the
  closed-form jacobian + the splu σ-bridge rather than via autograd
  through ``A.values()``.

Governing equation:

    −∇ · (σ ∇φ) = q                                    in Ω
    ∂φ/∂n      = 0                                     at z = 0 (free surface)
    φ          = 0                                     at one pinned cell

7-point stencil with harmonic-mean σ on each interior face:

    σ_face = 2 σ_l σ_r / (σ_l + σ_r)

Each face contributes a symmetric ``±T = ±σ_face / spacing²`` block to
``A``. One far cell (default **bottom mid-x mid-y**) is pinned to
``φ = 0`` via a symmetric row+column replacement, removing the
singular Neumann nullspace without the over-constraint of the legacy
five-wall Dirichlet. Bottom mid-(x, y) preserves both x- and y-mirror
symmetry for surface-electrode surveys laid out symmetrically about
the mesh center.

Operator level: ``IMPLICIT_VJP``. The system is solved through the scipy
``splu`` bridge (``sparse_linear_solve_with_adjoint``), which factorises ``A``
on the CPU in float64, so DC3D is **CPU / float64 only**; non-CPU inputs are
rejected at ``_forward`` entry.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from geobrain.physics.em.config import EMExecutionConfig
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
    sparse_linear_solve_with_adjoint,
)
from ....mesh import TensorMesh
from ....mesh.capabilities import ConnectivityMesh, StructuredMesh
from ....mesh import canonicalize_cell_field, require_capable_mesh


@dataclass(frozen=True)
class DC3DSurvey:
    """
    4-electrode DC geometry on a 3D grid, in PHYSICAL coordinates.

    Electrode positions are physical coordinates in **metres** (floats),
    not grid indices. They are resolved to mesh cells inside
    :class:`DC3D._forward`: on a structured ``TensorMesh`` via
    ``floor(coord / spacing)`` (the cell containing the point; round is
    ambiguous at half-integer centres), on an ``OctreeMesh`` via argmin over
    ``mesh.cell_centers()``. This matches the physical-coordinate
    convention of :class:`DipoleDipoleSurvey` and removes any need for the
    octree to remember a base grid spacing.

    Electrode coordinates are physical metres in each named field
    (``source_x``/``source_y``/``source_z`` etc.), convention-independent.
    They resolve onto the platform ``(nz, nx, ny)`` mesh with physical x on
    axis-1 (spacing dx) and y on axis-2 (spacing dy); for a uniform mesh cell
    ``(iz, ix, iy)`` has centre ``((iz + 0.5)·dz, (ix + 0.5)·dx,
    (iy + 0.5)·dy)``.

    Attributes:
        source_z, source_x, source_y: physical coordinate (m) of the
            current injection electrode (positive current ``+I``).
        sink_z, sink_x, sink_y:        physical coordinate (m) of the
            sink (``−I``).
        rcv_z, rcv_x, rcv_y:           1-D float tensors of receiver
            coordinates (m).
        current:                       injected current magnitude (A).
    """

    source_z: float
    source_x: float
    source_y: float
    sink_z: float
    sink_x: float
    sink_y: float
    rcv_z: torch.Tensor
    rcv_x: torch.Tensor
    rcv_y: torch.Tensor
    current: float = 1.0

    def __post_init__(self) -> None:
        for name in ("rcv_z", "rcv_x", "rcv_y"):
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor):
                raise GeoBrainError(
                    f"DC3DSurvey.{name} must be a torch.Tensor",
                    object_name="DC3DSurvey", field=name,
                    expected=torch.Tensor, actual=type(t),
                )
            if t.ndim != 1:
                raise GeoBrainError(
                    f"DC3DSurvey.{name} must be 1D",
                    object_name="DC3DSurvey", field=name,
                    expected="1D Tensor", actual=tuple(t.shape),
                )
        n = self.rcv_z.numel()
        if not (self.rcv_x.numel() == n and self.rcv_y.numel() == n):
            raise GeoBrainError(
                "rcv_{z,x,y} must share length",
                object_name="DC3DSurvey", field="rcv_x/rcv_y",
                expected=n,
                actual=(self.rcv_x.numel(), self.rcv_y.numel()),
            )
        if self.current <= 0:
            raise GeoBrainError(
                "current must be positive",
                object_name="DC3DSurvey", field="current",
                expected="> 0", actual=self.current,
            )

    @property
    def n_rcv(self) -> int:
        return int(self.rcv_z.numel())


def _flat3(iz, ix, iy, nx: int, ny: int):
    """
    Row-major flat index under ``(nz, nx, ny)`` shape.

    Mirrors :func:`assemble_poisson_3d`'s convention
    ``flat = iy + ix*ny + iz*nx*ny`` (y-fastest, then x, then z).
    Accepts ints or 1-D long tensors and broadcasts elementwise.
    """
    return iy + ix * ny + iz * nx * ny


def _sparse_matvec_autograd(A_csr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Autograd-true sparse-CSR · dense matvec ``A x`` via manual gather/scatter.

    Equivalent to ``torch.sparse.mm(A, x[:, None])[:, 0]`` but the backward is a
    pure DENSE gather/scatter (``ScatterAddBackward``) instead of the sparse-add
    path ``torch.sparse.mm`` takes, that path requires a PyTorch built with MKL
    (unavailable in this env). The gradient w.r.t. ``A.values()`` is exactly
    ``x[col]`` per non-zero, so σ-autograd through ``A``'s values is preserved
    EXACTLY (used for the singularity-removal residual ``−(A(σ)−A(σ_ref)) u_p``
    where ``x = u_p`` is a detached constant and ``A``'s values carry σ).

    Args:
        A_csr: square sparse CSR tensor; ``A_csr.values()`` may require grad.
        x: dense ``(n,)`` vector (typically detached).

    Returns:
        ``(n,)`` dense ``A x`` with autograd attached through ``A_csr.values()``.
    """
    crow = A_csr.crow_indices().to(torch.long)
    col = A_csr.col_indices().to(torch.long)
    vals = A_csr.values()
    n = int(crow.numel() - 1)
    counts = (crow[1:] - crow[:-1]).to(torch.long)
    rows = torch.repeat_interleave(
        torch.arange(n, dtype=torch.long, device=A_csr.device), counts
    )
    out = torch.zeros(n, dtype=vals.dtype, device=A_csr.device)
    return out.scatter_add(0, rows, vals * x[col])


def assemble_dc3d_system(
    sigma: torch.Tensor,
    dx: float,
    dz: float,
    dy: float,
    *,
    dirichlet_idx: int | None = None,
    boundary: Literal["neumann", "robin"] = "neumann",
    source_center: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    Assemble the DC3D operator as a dense ``torch.Tensor``.

    Dense-return adapter (NOT a back-compat shim): internally calls
    :func:`_assemble_dc3d_sparse` (which delegates to ``assemble_poisson_3d``)
    and densifies via ``.to_dense()``. It exists so matrix-structure tests can
    assert symmetry / pin row+col on a dense matrix; :class:`DC3D._forward` uses
    the sparse path directly and never calls this. Public + tested, not vestigial.

    Args:
        sigma: ``(nz, nx, ny)`` conductivity tensor (S/m).
        dx, dz, dy: cell spacings (m).
        dirichlet_idx: cell flat index where ``φ = 0`` is enforced.
            ``None`` → default
            ``(nz - 1) * nx * ny + (nx // 2) * ny + ny // 2``.
        boundary: boundary condition on the outer (non-top) faces.
            ``"neumann"`` is the pure Neumann + Dirichlet pin formulation
            (the assembly-level form of the operator's
            ``boundary="pin"``); ``"robin"`` adds a diagonal Robin
            contribution ``∂φ/∂n + φ/r = 0`` on lateral and bottom faces
            (top remains Neumann, the free surface) and corresponds to
            the operator's ``boundary="robin"``. When ``"robin"``, the
            pin is dropped (Robin alone makes the system well-posed).
            The operator's third mode, ``boundary="mixed"``, never
            reaches this assembler; it routes through the mesh-agnostic
            ``assemble_poisson_fv`` seam instead.
        source_center: ``(x, y, z)`` reference point in physical
            coordinates from which the Robin distance ``r`` is measured.
            Required when ``boundary="robin"``. Coordinates are relative
            to the mesh's bottom-front-left corner (the implicit origin
            used by ``cell_widths``); for a uniform mesh, cell
            ``(iz, ix, iy)`` has centre ``((ix + 0.5) * dx,
            (iy + 0.5) * dy, (iz + 0.5) * dz)``.

    Returns:
        ``(n_cells, n_cells)`` dense ``A``.
    """
    return _assemble_dc3d_sparse(
        sigma, dx=dx, dz=dz, dy=dy,
        dirichlet_idx=dirichlet_idx,
        boundary=boundary, source_center=source_center,
    ).to_dense()


def _assemble_dc3d_sparse(
    sigma: torch.Tensor,
    dx: float,
    dz: float,
    dy: float,
    *,
    dirichlet_idx: int | None = None,
    boundary: Literal["neumann", "robin"] = "neumann",
    source_center: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """
    Assemble the DC3D operator as torch sparse CSR.

    Delegates the bulk of the work to the ``assemble_poisson_3d``
    (pure-Neumann 7-point stencil with harmonic-mean face conductivities),
    then applies the symmetric Dirichlet pin by zeroing entries in the
    pinned row and column of the CSR values vector while keeping the
    sparsity pattern intact. The diagonal entry at
    ``(dirichlet_idx, dirichlet_idx)`` is overwritten to 1.

    Signature mirrors :func:`assemble_dc3d_system` (sigma shape
    ``(nz, nx, ny)`` with spacings ``dx, dz, dy``) so the two are
    drop-in alternatives. The default pin matches the dense path's
    bottom-mid-(x, y) cell, preserving x/y-mirror symmetry for
    surveys laid out symmetrically about the mesh center.

    At uniform unit spacing (``dx = dy = dz = 1``) the values match
    the dense :func:`assemble_dc3d_system` to atol 1e-12; for
    non-unit spacing the proper finite-volume face transmissibility
    ``T = σ_face · A_face / d_normal`` is used (dense's
    ``T = σ_face / spacing²`` is only consistent at unit area). The
    forward path has since been switched onto this FV transmissibility
    via the ``assemble_poisson_fv`` seam.

    Args:
        sigma:           ``(nz, nx, ny)`` conductivity tensor.
        dx, dz, dy:      mesh spacings (in that order, matching dense).
        dirichlet_idx:   flat index of pinned cell. Default = bottom
            mid-(x, y) cell, matching :func:`assemble_dc3d_system`.

    Returns:
        ``torch.sparse_csr_tensor`` shape ``(n, n)`` with
        ``.values()`` carrying the autograd graph of ``sigma``.
    """
    from ..numerics.finite_volume.poisson import assemble_poisson_3d

    if sigma.ndim != 3:
        raise GeoBrainError(
            f"sigma must be 3D, got shape {tuple(sigma.shape)}",
            object_name="_assemble_dc3d_sparse",
            field="sigma",
            expected="3D tensor",
            actual=tuple(sigma.shape),
        )
    if boundary not in ("neumann", "robin"):
        raise GeoBrainError(
            f"boundary must be 'neumann' or 'robin', got {boundary!r}",
            object_name="_assemble_dc3d_sparse",
            field="boundary",
            expected="'neumann' | 'robin'",
            actual=boundary,
        )
    if boundary == "robin" and source_center is None:
        raise GeoBrainError(
            "boundary='robin' requires source_center=(x, y, z)",
            object_name="_assemble_dc3d_sparse",
            field="source_center",
            expected="(x, y, z) tuple",
            actual=None,
        )
    nz, nx, ny = sigma.shape
    n = nz * nx * ny
    if dirichlet_idx is None:
        # Bottom mid-(x, y), matching ``assemble_dc3d_system``.
        dirichlet_idx = (nz - 1) * nx * ny + (nx // 2) * ny + ny // 2
    if not (0 <= int(dirichlet_idx) < n):
        raise GeoBrainError(
            "dirichlet_idx out of range",
            object_name="_assemble_dc3d_sparse",
            field="dirichlet_idx",
            expected=f"[0, {n})",
            actual=dirichlet_idx,
        )
    pin = int(dirichlet_idx)

    # ``TensorMesh`` 3-D layout is ``(nz, nx, ny)`` with spacings
    # ``(dz, dx, dy)``: matching ``assemble_poisson_3d`` (flat index
    # ``iy + ix*ny + iz*nx*ny``). Delegating directly preserves the
    # flat-index ordering.
    mesh = TensorMesh(shape=(nz, nx, ny), spacing=(dz, dx, dy))
    A_csr = assemble_poisson_3d(mesh, sigma)

    crow = A_csr.crow_indices()
    col = A_csr.col_indices()
    vals = A_csr.values()

    if boundary == "robin":
        # Robin / mixed BC ``∂φ/∂n + φ/r = 0`` on the five outer
        # boundaries (x_min, x_max, y_min, y_max, z_max). Top z=0 face
        # stays Neumann (free surface: no current into air). For each
        # boundary cell, this adds ``σ · A_face / (h/2 + r_face)`` to
        # the diagonal A[c, c], where ``h/2`` is the cell half-width
        # in the boundary-face normal direction and ``r_face`` is the
        # distance from the face centre to ``source_center``.
        #
        # Derivation:
        #   Centered FD across the boundary face says
        #       σ_face (φ_face − φ_c) / (h/2) = −σ_face · φ_face / r,
        #   from which φ_face = φ_c / (1 + h / (2 r)). Combined with
        #   the Neumann path's missing-neighbour flux ``σ_c · A_face ·
        #   (φ_face − φ_c) / (h/2)`` gives the Robin diagonal contribution
        #       Δ A[c, c] = σ_c · A_face / (h/2 + r_face).
        #
        # Returns: per-cell diagonal additions as a flat tensor of
        # length n, padded with zero for non-boundary cells.
        robin_diag = _robin_diagonal_contribution(
            sigma, dx=dx, dy=dy, dz=dz,
            source_center=source_center,
        )

        # Add Robin diagonal to A.values() via a scatter-add on the
        # CSR values vector. The diagonal sparsity pattern is
        # preserved by ``assemble_poisson_3d`` (every cell has a
        # self-entry from at least one face), so we locate the
        # diagonal positions by scanning the CSR row segments.
        n_rows_csr = int(crow.numel() - 1)
        counts_csr = (crow[1:] - crow[:-1]).to(torch.long)
        row_idx_csr = torch.repeat_interleave(
            torch.arange(n_rows_csr, dtype=torch.long, device=vals.device),
            counts_csr,
        )
        on_diag_mask = (row_idx_csr == col)
        # Positions inside ``vals`` that correspond to diagonal entries
        # (in ascending order, one per row since the CSR is coalesced).
        diag_positions = torch.nonzero(on_diag_mask, as_tuple=True)[0]
        # Pull the per-cell Robin contributions in the same order.
        diag_cells = row_idx_csr[on_diag_mask]
        diag_adds = robin_diag.to(vals.dtype)[diag_cells]
        # Build the additive vector aligned with ``vals`` and add.
        add_vec = torch.zeros_like(vals).scatter_add(
            0, diag_positions, diag_adds,
        )
        new_vals = vals + add_vec
        # Robin BC alone anchors the field: drop the Dirichlet pin.
        return torch.sparse_csr_tensor(crow, col, new_vals, size=(n, n))

    # Reconstruct per-nnz row index from CSR ``crow``.
    counts = (crow[1:] - crow[:-1]).to(torch.long)
    n_rows = int(crow.numel() - 1)
    row_idx = torch.repeat_interleave(
        torch.arange(n_rows, dtype=torch.long, device=vals.device),
        counts,
    )

    in_pinned_row = row_idx == pin
    in_pinned_col = col == pin
    on_diag_pin = in_pinned_row & in_pinned_col

    # Symmetric Dirichlet pin: zero the pinned row + column, place 1
    # on the pin diagonal. Sparsity pattern is preserved (we only
    # rewrite values), so the CSR ``crow`` / ``col`` arrays still
    # describe a valid layout.
    new_vals = torch.where(
        on_diag_pin,
        torch.ones_like(vals),
        torch.where(in_pinned_row | in_pinned_col, torch.zeros_like(vals), vals),
    )

    return torch.sparse_csr_tensor(crow, col, new_vals, size=(n, n))


def _robin_diagonal_contribution(
    sigma: torch.Tensor,
    *,
    dx: float,
    dy: float,
    dz: float,
    source_center: tuple[float, float, float],
) -> torch.Tensor:
    """
    Robin BC diagonal contribution as a flat ``(n,)`` tensor.

    For each cell on one of the 5 outer boundaries (x_min, x_max,
    y_min, y_max, z_max) (top z=0 stays Neumann) adds
    ``σ_c · A_face / (h/2 + r_face)`` to the cell's diagonal entry.
    A cell at a corner gets contributions from multiple faces (they
    add). Non-boundary cells get zero.

    All operations are torch-native so autograd flows through ``σ``.

    Args:
        sigma: ``(nz, nx, ny)`` conductivity tensor.
        dx, dy, dz: cell spacings (uniform; non-uniform support would
            require switching to ``mesh.cell_widths`` here).
        source_center: ``(x, y, z)`` reference point used for ``r``.

    Returns:
        Flat ``(n,)`` tensor of diagonal additions, with the same dtype
        as ``sigma`` and autograd-attached through ``σ`` at boundary
        cells.
    """
    nz, nx, ny = sigma.shape
    device, dtype = sigma.device, sigma.dtype
    n = nz * nx * ny

    sx, sy, sz = (float(s) for s in source_center)
    half_dx = 0.5 * float(dx)
    half_dy = 0.5 * float(dy)
    half_dz = 0.5 * float(dz)

    # Cell-centre coordinates (uniform mesh; non-uniform would need
    # cumulative-width-based centres). Under the ``(nz, nx, ny)`` layout
    # axis-1 is x and axis-2 is y.
    ix = torch.arange(nx, device=device, dtype=dtype)
    iy = torch.arange(ny, device=device, dtype=dtype)
    iz = torch.arange(nz, device=device, dtype=dtype)
    xc = (ix + 0.5) * float(dx)   # (nx,)
    yc = (iy + 0.5) * float(dy)   # (ny,)
    zc = (iz + 0.5) * float(dz)   # (nz,)

    # Accumulate diagonal additions into a flat (n,) tensor that
    # preserves autograd through σ via the multiplication. Flat index under
    # ``(nz, nx, ny)`` is ``iy + ix*ny + iz*nx*ny`` (y-fastest).
    diag_add = torch.zeros(n, device=device, dtype=dtype)

    def _flat_idx(kk, iix, iiy):
        return (iiy + iix * ny + kk * nx * ny).reshape(-1)

    # --- x_min boundary (face at x = 0; cell ix=0) ---
    # Face centre: x_f = 0, y_f = yc[j], z_f = zc[k].
    # Build face-centre coords via broadcasting (nz, ny).
    yy = yc.view(1, ny).expand(nz, ny)
    zz = zc.view(nz, 1).expand(nz, ny)
    r_face = torch.sqrt((0.0 - sx) ** 2 + (yy - sy) ** 2 + (zz - sz) ** 2)
    area = float(dy) * float(dz)
    add_face = sigma[:, 0, :] * area / (half_dx + r_face)  # (nz, ny)
    kk = torch.arange(nz, device=device).view(nz, 1).expand(nz, ny)
    jj = torch.arange(ny, device=device).view(1, ny).expand(nz, ny)
    ix_const = torch.zeros_like(kk)
    flat = _flat_idx(kk, ix_const, jj)
    diag_add = diag_add.scatter_add(0, flat, add_face.reshape(-1))

    # --- x_max boundary (face at x = nx·dx; cell ix=nx-1) ---
    r_face = torch.sqrt((nx * float(dx) - sx) ** 2 + (yy - sy) ** 2 + (zz - sz) ** 2)
    add_face = sigma[:, -1, :] * area / (half_dx + r_face)
    ix_const = torch.full_like(kk, nx - 1)
    flat = _flat_idx(kk, ix_const, jj)
    diag_add = diag_add.scatter_add(0, flat, add_face.reshape(-1))

    # --- y_min boundary (face at y = 0; cell iy=0) ---
    xx = xc.view(1, nx).expand(nz, nx)
    zz = zc.view(nz, 1).expand(nz, nx)
    r_face = torch.sqrt((xx - sx) ** 2 + (0.0 - sy) ** 2 + (zz - sz) ** 2)
    area = float(dx) * float(dz)
    add_face = sigma[:, :, 0] * area / (half_dy + r_face)  # (nz, nx)
    kk = torch.arange(nz, device=device).view(nz, 1).expand(nz, nx)
    ii = torch.arange(nx, device=device).view(1, nx).expand(nz, nx)
    iy_const = torch.zeros_like(kk)
    flat = _flat_idx(kk, ii, iy_const)
    diag_add = diag_add.scatter_add(0, flat, add_face.reshape(-1))

    # --- y_max boundary (face at y = ny·dy; cell iy=ny-1) ---
    r_face = torch.sqrt((xx - sx) ** 2 + (ny * float(dy) - sy) ** 2 + (zz - sz) ** 2)
    add_face = sigma[:, :, -1] * area / (half_dy + r_face)
    iy_const = torch.full_like(kk, ny - 1)
    flat = _flat_idx(kk, ii, iy_const)
    diag_add = diag_add.scatter_add(0, flat, add_face.reshape(-1))

    # --- z_max boundary (face at z = nz·dz; cell iz=nz-1) ---
    # Top z=0 stays Neumann (free surface), so only the bottom z-face
    # contributes. Iterate over (nx, ny).
    xx = xc.view(nx, 1).expand(nx, ny)
    yy = yc.view(1, ny).expand(nx, ny)
    r_face = torch.sqrt((xx - sx) ** 2 + (yy - sy) ** 2 + (nz * float(dz) - sz) ** 2)
    area = float(dx) * float(dy)
    add_face = sigma[-1, :, :] * area / (half_dz + r_face)  # (nx, ny)
    ii = torch.arange(nx, device=device).view(nx, 1).expand(nx, ny)
    jj = torch.arange(ny, device=device).view(1, ny).expand(nx, ny)
    kk_const = torch.full_like(ii, nz - 1)
    flat = _flat_idx(kk_const, ii, jj)
    diag_add = diag_add.scatter_add(0, flat, add_face.reshape(-1))

    return diag_add


class DC3D(EMOperatorDiscovery, ForwardOperator):
    """
    3D DC resistivity forward operator.

    Device: float64. CPU by default (scipy sparse LU); GPU-capable by installing
    a GPU sparse backend (``from geobrain.core.adjoint import PcgGpuSolver,
    set_default_sparse_solver; set_default_sparse_solver(PcgGpuSolver())``), the
    forward is device-preserving and then runs end-to-end on CUDA. Note the
    PcgGpuSolver Jacobi-CG convergence caveat (weak preconditioner: iteration
    count grows with conditioning). With the default scipy backend a non-CPU
    input is rejected by the bridge with an actionable error.

    Inputs (ModelState):  ``sigma`` shape ``(nz, nx, ny)`` matching the mesh.

    Boundary condition (``boundary=`` constructor kwarg, one of three modes):
        ``"pin"`` (default), pure Neumann on the outer walls with a single
        Dirichlet pin removing the nullspace. On a SMALL truncated box the
        no-flux walls reflect the potential, biasing the field; the cure is
        LARGE padding. Assembly-level form: ``boundary="neumann"`` (+ the
        Dirichlet pin) in :func:`assemble_dc3d_system`.

        ``"mixed"``: adds a Dey–Morrison **mixed / Robin** absorbing term
        ``∂φ/∂n ≈ −α·φ`` on the outer (non-free-surface) faces, where ``α`` is
        the asymptotic decay rate of a point source over a half-space with a
        mirror image across the free surface (see
        :func:`~geobrain.physics.em.numerics.finite_volume.mixed_boundary.mixed_boundary_alpha`).
        ``α`` is evaluated from ONE fixed reference point, the centroid of all
        electrodes, so ``A`` stays source-independent and the LU factorisation
        is shared across right-hand sides. The Robin term makes ``A``
        non-singular, so the Dirichlet pin is DROPPED. Routed through the
        mesh-agnostic ``assemble_poisson_fv`` seam, so it works on BOTH the
        structured ``TensorMesh`` path and the unstructured / octree path. The
        free-surface (``z = 0``) faces stay Neumann (no current into air). Lets a
        SMALL box reach near the accuracy of a 2× padded Neumann box. Never
        reaches :func:`assemble_dc3d_system`.

        ``"robin"``: the legacy structured-only **diagonal Robin**
        ``∂φ/∂n + φ/r = 0`` on the lateral + bottom faces, with ``r`` measured
        from the source/sink midpoint (computed internally at forward time;
        the Dirichlet pin is dropped; Robin alone anchors the field).
        TensorMesh-only: the check fires at ``_forward`` time because the mesh
        arrives via ``ctx``. Assembly-level form: ``boundary="robin"`` (+
        ``source_center=``) in :func:`assemble_dc3d_system`. Prefer
        ``"mixed"``: the mesh-agnostic seam route.

        ``boundary="mixed"`` and ``boundary="robin"`` both forbid an explicit
        ``dirichlet_idx`` semantics-wise (the pin is dropped; ``"robin"``
        silently ignores it) and the closed-form σ-Jacobian (which models only
        the Neumann stencil).

    Singularity removal (``singularity_removal=`` / ``sigma_ref=`` kwargs):
        The discrete DC solution's dominant error is the ``1/r`` singularity at
        the electrodes, the worst-resolved feature on coarse / irregular tet
        meshes. ``singularity_removal=True`` splits ``u = u_p + u_s`` (Lowry /
        Coggon-style singularity removal, cast in
        cell-centred FV): ``u_p`` is the ANALYTIC potential of the electrode pair
        in a homogeneous half-space of conductivity ``σ_ref`` (mirror-image
        Green's function at the cell centres), and the SAME discrete system is
        solved only for the smooth correction driven by the conductivity contrast,
        ``A(σ) u_s = −(A(σ) − A(σ_ref)) u_p`` (the operator-difference RHS, exactly
        ``0`` where ``σ == σ_ref``). The singular near-field is carried exactly by
        ``u_p``; the receiver voltage is ``u_p + u_s`` at the receiver cells.
        ``σ_ref=None`` (default) → the
        DETACHED median of the forward σ (deterministic). ``u_p`` is detached, but
        ``(A(σ) − A(σ_ref)) u_p`` differentiates through ``A(σ)``'s σ-carrying values, so the
        σ-gradient of the whole split is EXACT. Composes with ``boundary="pin"``
        and ``boundary="mixed"``; mutually exclusive with the closed-form
        σ-Jacobian and the legacy ``boundary="robin"``. The electrode self-cell ``1/r``
        is regularised with the cell's volume-equivalent sphere radius
        ``r_eff = (3 V_cell/4π)^(1/3)`` (see
        :func:`~geobrain.physics.em.numerics.finite_volume.singularity_removal.primary_potential`).

    Outputs (ForwardOutput):
        - ``data["voltage"]``: receiver voltages, shape ``(n_rcv,)``.
        - ``fields["potential"]``: full 3D ``φ`` field, shape
          ``(nz, nx, ny)`` (diagnostic).
        - ``metadata["pin"]``: flat cell index of the Dirichlet pin
          (``None`` when ``boundary="mixed"``).
        - ``metadata["boundary"]``: the boundary mode (``"pin"`` |
          ``"mixed"`` | ``"robin"``).
        - ``metadata["singularity_removal"]``: whether the
          primary/secondary split ran.
        - ``metadata["sigma_ref"]``: the reference σ used for ``u_p``
          (``None`` when SR off).
        - ``metadata["secondary_primary_ratio"]``: ``‖u_s‖/‖u_p‖``
          diagnostic (``None`` when SR off); ≈ 0 on a homogeneous
          ``σ ≡ σ_ref`` model.

    Args:
        survey: 3-D galvanic acquisition.
        dirichlet_idx: node pinned to zero potential (``None`` = auto).
        config: :class:`~geobrain.physics.em.EMExecutionConfig` execution
            policy (closed-form sigma Jacobian).
        boundary: ``'pin'`` / ``'mixed'`` / ``'robin'`` boundary treatment.
        singularity_removal: split off the analytic primary potential.
        sigma_ref: reference conductivity for singularity removal.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        output_keys=("voltage",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (ConnectivityMesh,)

    def __init__(
        self,
        survey: DC3DSurvey,
        *,
        dirichlet_idx: int | None = None,
        config: EMExecutionConfig | None = None,
        boundary: Literal["pin", "mixed", "robin"] = "pin",
        singularity_removal: bool = False,
        sigma_ref: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, DC3DSurvey):
            raise GeoBrainError(
                "DC3D requires a DC3DSurvey",
                object_name="DC3D", field="survey",
                expected=DC3DSurvey, actual=type(survey),
            )
        if boundary not in ("pin", "mixed", "robin"):
            raise GeoBrainError(
                f"boundary must be 'pin', 'mixed', or 'robin', got {boundary!r}",
                object_name="DC3D", field="boundary",
                expected="'pin' | 'mixed' | 'robin'", actual=boundary,
            )
        self.survey = survey
        self._dirichlet_idx = dirichlet_idx
        # The single boundary mode (one of three). ``"pin"`` (default) keeps the
        # pure-Neumann 7-point stencil + a single Dirichlet pin removing the
        # nullspace. ``"mixed"`` adds the Dey–Morrison point-source Robin term
        # ``∂φ/∂n ≈ −α·φ`` on the outer (non-free-surface) faces (α from the
        # electrode-array centroid via :func:`mixed_boundary_alpha`), routed
        # through the mesh-agnostic ``assemble_poisson_fv`` seam (works on BOTH
        # the structured and the unstructured/octree path), and DROPS the pin
        # (the Robin term alone makes the system non-singular). ``"robin"`` is
        # the legacy structured-only diagonal Robin ``∂φ/∂n + φ/r = 0`` on the
        # lateral + bottom faces with ``r`` from the source/sink midpoint; it
        # too drops the pin. The structured-only check fires at ``_forward``
        # time because the mesh arrives via ``ctx``.
        self._boundary = boundary
        # ``boundary="robin"`` (legacy diagonal Robin) maps to the assembly-level
        # ``boundary="robin"`` keyword of :func:`assemble_dc3d_system` /
        # :func:`_assemble_dc3d_sparse`; ``"pin"`` and ``"mixed"`` both map to
        # the assembly-level ``"neumann"`` stencil (``"mixed"`` then adds its
        # Robin diagonals through the FV seam, ``"pin"`` keeps the Dirichlet pin).
        self._assembly_boundary: Literal["neumann", "robin"] = (
            "robin" if boundary == "robin" else "neumann"
        )
        # ``boundary="mixed"`` drops the Dirichlet pin, so an explicit
        # ``dirichlet_idx`` is contradictory: fail loudly rather than silently
        # ignore it. ``dirichlet_idx`` is honoured only in the default "pin" mode.
        # ``"robin"`` drops the pin too, but historically ignored an explicit
        # ``dirichlet_idx`` silently; that behaviour is preserved.
        if boundary == "mixed" and dirichlet_idx is not None:
            raise GeoBrainError(
                "boundary='mixed' drops the Dirichlet pin; do not also pass "
                "dirichlet_idx (it is honoured only for boundary='pin').",
                object_name="DC3D", field="dirichlet_idx",
                expected="None when boundary='mixed'", actual=dirichlet_idx,
            )
        # The closed-form σ-Jacobian models only the Neumann 7-point stencil,
        # not any Robin diagonal term, so it is incompatible with both
        # Robin-flavoured modes (``"mixed"`` and the legacy ``"robin"``).
        cfg = config if config is not None else EMExecutionConfig()
        self._use_closed_form_sigma_jacobian = cfg.use_closed_form_sigma_jacobian
        if boundary == "mixed" and self._use_closed_form_sigma_jacobian:
            raise GeoBrainError(
                "boundary='mixed' is incompatible with "
                "use_closed_form_sigma_jacobian=True (the closed-form σ-Jacobian "
                "models only the Neumann 7-point stencil, not the Robin diagonal "
                "term). Set one or the other.",
                object_name="DC3D",
                field="boundary / use_closed_form_sigma_jacobian",
                expected="boundary='pin' OR use_closed_form_sigma_jacobian=False",
                actual=("mixed", True),
            )
        if boundary == "robin" and self._use_closed_form_sigma_jacobian:
            raise GeoBrainError(
                "use_closed_form_sigma_jacobian=True is not supported "
                "with boundary='robin' (closed-form Jacobian is Neumann-only). "
                "Set one or the other.",
                object_name="DC3D",
                field="use_closed_form_sigma_jacobian / boundary",
                expected="boundary='pin' OR use_closed_form_sigma_jacobian=False",
                actual=(boundary, True),
            )
        # Singularity removal (primary/secondary potential split). When ``True``,
        # the analytic 1/r near-field of the electrode pair over a homogeneous
        # half-space of conductivity ``sigma_ref`` is carried by an analytic
        # primary potential ``u_p`` evaluated at the cell centres, and only the
        # smooth secondary correction ``u_s`` (``A u_s = b − A u_p``) is solved on
        # the discrete grid. ``sigma_ref=None`` → a DETACHED median of the forward
        # σ (deterministic, documented). See
        # :func:`~geobrain.physics.em.numerics.finite_volume.singularity_removal.primary_potential`.
        self._singularity_removal = bool(singularity_removal)
        if sigma_ref is not None and float(sigma_ref) <= 0.0:
            raise GeoBrainError(
                "sigma_ref must be a positive conductivity (S/m) or None",
                object_name="DC3D", field="sigma_ref",
                expected="> 0 or None", actual=sigma_ref,
            )
        self._sigma_ref = None if sigma_ref is None else float(sigma_ref)
        # SR composes with both boundary modes (pin and mixed) and is orthogonal
        # to the Robin/closed-form switches, BUT the closed-form σ-Jacobian path
        # builds its Jacobian against the ORIGINAL RHS b: it does not model the
        # ``b − A u_p`` residual source, so the two are mutually exclusive.
        if self._singularity_removal and self._use_closed_form_sigma_jacobian:
            raise GeoBrainError(
                "singularity_removal=True is incompatible with "
                "use_closed_form_sigma_jacobian=True (the closed-form σ-Jacobian "
                "is built against the original injection RHS, not the "
                "secondary-solve residual b − A u_p). Set one or the other.",
                object_name="DC3D",
                field="singularity_removal / use_closed_form_sigma_jacobian",
                expected="at most one of the two",
                actual=(True, True),
            )
        # The legacy structured-only diagonal Robin (``boundary='robin'``) is a
        # different mechanism that the connectivity SR helper does not target;
        # SR is offered on the mesh-agnostic Neumann+pin / mixed seam only.
        if self._singularity_removal and boundary == "robin":
            raise GeoBrainError(
                "singularity_removal=True is not supported with the legacy "
                "structured-only boundary='robin'; use boundary='pin' (default) "
                "or boundary='mixed' (both route through the FV seam SR targets).",
                object_name="DC3D", field="singularity_removal / boundary",
                expected="boundary='pin' or 'mixed'", actual=("SR", "robin"),
            )

    def _electrode_centroid_zxy(self) -> torch.Tensor:
        """Centroid of ALL electrodes (source, sink, receivers) as ``(z, x, y)``.

        The mixed / Robin reference point. Physical metres in the platform
        ``(z, x, y)`` column order, the same frame as ``cell_centers()`` /
        ``boundary_faces().centroid`` on a ``(nz, nx, ny)`` mesh (origin at the
        mesh corner). Averaging over every electrode keeps ``A``
        source-independent (one fixed reference) so the LU factorisation stays
        shared across right-hand sides.
        """
        zs = torch.cat([
            torch.tensor([self.survey.source_z, self.survey.sink_z],
                         dtype=torch.float64),
            self.survey.rcv_z.to(torch.float64),
        ])
        ys = torch.cat([
            torch.tensor([self.survey.source_y, self.survey.sink_y],
                         dtype=torch.float64),
            self.survey.rcv_y.to(torch.float64),
        ])
        xs = torch.cat([
            torch.tensor([self.survey.source_x, self.survey.sink_x],
                         dtype=torch.float64),
            self.survey.rcv_x.to(torch.float64),
        ])
        return torch.stack([zs.mean(), xs.mean(), ys.mean()])

    def _mixed_boundary_dict(self, mesh, *, pin: int | None) -> dict:
        """Robin boundary dict for :func:`assemble_poisson_fv` (mixed mode).

        Calls :func:`mixed_boundary_alpha` on ``mesh.boundary_faces()`` with the
        electrode-array centroid as the reference point (free surface at depth
        ``z = 0``, ``free_surface_axis=0``). Returns the ``{"robin_cells",
        "robin_alpha_area"}`` dict; ``pin`` is intentionally NOT added (mixed
        mode drops the Dirichlet pin; the Robin term anchors the field). Works
        identically on the structured and the unstructured/octree path because
        ``boundary_faces()`` is a :class:`ConnectivityMesh` method.
        """
        from ..numerics.finite_volume.mixed_boundary import mixed_boundary_alpha

        ref_zxy = self._electrode_centroid_zxy()
        robin_cells, robin_alpha_area = mixed_boundary_alpha(
            mesh.boundary_faces(), ref_zxy,
            free_surface_axis=0, free_surface_coord=0.0,
        )
        return {
            "robin_cells": robin_cells,
            "robin_alpha_area": robin_alpha_area,
        }

    @staticmethod
    def _nearest_cell(centers, z, x, y):
        """Flat octree-leaf index nearest the PHYSICAL electrode ``(z, x, y)``.

        ``centers`` is ``mesh.cell_centers()`` (``(n_cells, n_dim)``, platform
        ``(z, x, y)`` columns on a ``(nz, nx, ny)`` mesh, metres). The electrode
        coordinate is provided directly in metres by the survey, so no base grid
        spacing is needed: we simply argmin the squared distance over the leaf
        centres. This is exact on a refined octree because it compares against
        the actual leaf geometry rather than reconstructing a coordinate from any
        single spacing.
        """
        target = torch.tensor(
            [float(z), float(x), float(y)],
            dtype=torch.float64,
        )
        d2 = ((centers - target) ** 2).sum(dim=1)
        return int(torch.argmin(d2))

    def _resolve_sigma_ref(self, sigma_f64: torch.Tensor) -> float:
        """Reference half-space conductivity for the singularity-removal split.

        Returns the explicit ``sigma_ref`` constructor value when given,
        otherwise the DETACHED median of the (flat) forward σ, a deterministic,
        documented default. Detaching keeps ``σ_ref`` a fixed scalar so the
        primary potential ``u_p`` and the reference operator ``A(σ_ref)`` carry no
        autograd; the σ-gradient of the whole split flows through ``A(σ)`` in the
        operator-difference residual ``−(A(σ) − A(σ_ref)) u_p``.
        """
        if self._sigma_ref is not None:
            return float(self._sigma_ref)
        return float(sigma_f64.detach().reshape(-1).median())

    def _solve_with_singularity_removal(
        self,
        A_coo: torch.Tensor,
        A_ref_csr: torch.Tensor,
        *,
        cell_centers: torch.Tensor,
        cell_volumes: torch.Tensor,
        src_cell: int,
        snk_cell: int,
        sigma_ref: float,
        pin: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Primary/secondary potential split solve: returns ``(phi, u_p, ratio)``.

        Lowry/Coggon-style singularity removal (the form node-based ERT codes
        implement at mesh nodes, recast in cell-centred FV). The CONTINUOUS primary ``u_p`` (the homogeneous-half-
        space Green's function of the electrode pair at conductivity ``σ_ref``,
        evaluated at the cell centres) satisfies the FULL singular source
        ``−∇·(σ_ref ∇u_p) = q`` exactly. Subtracting that from the target equation
        ``−∇·(σ∇u) = q`` removes the source entirely and leaves the smooth
        secondary driven only by the conductivity contrast::

            −∇·(σ ∇u_s) = ∇·((σ − σ_ref) ∇u_p)

        i.e. the discrete operator-difference RHS ``b_s = −(A(σ) − A(σ_ref)) u_p``
       , with NO singular point source. Where ``σ == σ_ref`` the RHS is exactly
        ``0`` and ``u_s ≡ 0`` (the homogeneous case reduces to the analytic ``u_p``
        to machine precision); elsewhere ``u_s`` carries only the smooth anomaly
        response. The total field is ``phi = u_p + u_s``.

        NB the plain (singularity-removal-off) formula ``A u_s = b − A u_p`` is an
        algebraic IDENTITY (``u_p + u_s ≡ A⁻¹ b``) and would change nothing; the
        operator-difference form above is what actually carries the singular
        near-field analytically and is what the spec's "RHS ≈ 0 where σ == σ_ref"
        intent requires.

        Gradients: ``A(σ_ref)`` is a DETACHED constant (σ_ref is fixed); ``u_p`` is
        detached; ``A(σ)`` (both in the RHS operator-difference and handed to the
        solve) carries σ's autograd. So ``b_s`` differentiates through ``A(σ)`` and
        the σ-gradient of the whole split is EXACT.

        ``pin`` (when not ``None``) is the Dirichlet-pinned cell: its row in the
        secondary RHS is zeroed so the pinned-row unit equation holds for ``u_s``
        (the pin fixes the additive gauge of the secondary field). ``ratio =
        ‖u_s‖/‖u_p‖`` is a homogeneity diagnostic.

        ``A_coo`` arrives as the σ-carrying COO matrix (NOT a CSR). It is coalesced
        once and its dense ``values()`` tensor is shared by BOTH the residual
        matvec and the secondary solve, so the two σ-grads accumulate at a DENSE
        leaf rather than at a sparse autograd node (the latter would require a
        PyTorch built with MKL: unavailable here). ``A_ref_csr`` is a fully
        DETACHED constant-σ_ref operator, so it carries no graph.
        """
        from ..numerics.finite_volume.singularity_removal import primary_potential

        device = A_coo.device
        centers = cell_centers.to(device=device, dtype=torch.float64)
        src_pos = centers[src_cell]
        snk_pos = centers[snk_cell]
        u_p = primary_potential(
            centers,
            src_pos,
            snk_pos,
            current=float(self.survey.current),
            sigma_ref=float(sigma_ref),
            free_surface_axis=0,
            free_surface_coord=0.0,
            self_cells=(int(src_cell), int(snk_cell)),
            cell_volumes=cell_volumes.to(device=device, dtype=torch.float64),
        ).to(device=device, dtype=torch.float64)

        # Extract the σ-carrying COO triplet ONCE. ``vals`` is a DENSE tensor with
        # the σ-graph; sharing it between the matvec and the solve's CSR makes both
        # σ-grads accumulate at this dense leaf (a dense add) rather than at a
        # sparse autograd node (which needs an MKL build).
        A_co = A_coo.coalesce()
        idx = A_co.indices()
        vals = A_co.values()
        n = int(A_co.shape[0])

        # Operator-difference secondary RHS  b_s = -(A(σ) − A(σ_ref)) u_p, built
        # via the dense gather/scatter matvec ``A x = scatter_add(row, vals·x[col])``
        # (autograd through ``vals`` ⊃ σ; ``u_p`` is a detached constant; ``A(σ_ref)``
        # is detached). This carries σ's autograd solely through ``A(σ)``, exactly
        # what an exact gradient of the split requires.
        A_up = torch.zeros(n, dtype=vals.dtype, device=device).scatter_add(
            0, idx[0], vals * u_p[idx[1]]
        )
        Aref_up = _sparse_matvec_autograd(A_ref_csr, u_p)
        b_s = -(A_up - Aref_up)
        if pin is not None:
            # Keep the pinned-row unit equation: secondary RHS at the pin is 0 so
            # u_s[pin] = 0 (the pin anchors the additive gauge of u_s).
            b_s = b_s.clone()
            b_s[int(pin)] = 0.0
        # Rebuild the solve CSR from the SAME dense ``vals`` so the solve's σ-grad
        # also lands on ``vals`` (dense accumulation with the matvec branch).
        A_csr = torch.sparse_coo_tensor(idx, vals, (n, n)).to_sparse_csr()
        u_s = sparse_linear_solve_with_adjoint(A_csr, b_s)
        phi = u_p + u_s

        up_norm = float(torch.linalg.norm(u_p.detach()))
        us_norm = float(torch.linalg.norm(u_s.detach()))
        ratio = us_norm / up_norm if up_norm > 0.0 else float("inf")
        return phi, u_p, ratio

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma,) = state.fetch("sigma")
        # Physical conductivity must be finite and strictly positive. A sigma<=0
        # cell makes the Poisson system indefinite (still solvable, returning
        # finite but meaningless voltages); reject it up front (vectorized, the
        # two reductions are cheap and device-agnostic).
        if not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
            raise GeoBrainError(
                "DC3D requires a finite, strictly-positive conductivity field "
                "(sigma > 0 S/m); a non-physical sigma <= 0 (or non-finite) makes "
                "the Poisson system indefinite and yields meaningless voltages",
                object_name="DC3D", field="sigma",
                expected="finite and all > 0",
                actual="contains sigma <= 0 or non-finite entries",
            )
        # Device policy is enforced by the shared sparse-solve bridge
        # (``sparse_linear_solve_with_adjoint``), the single source of truth:
        # with the default scipy backend a non-CPU σ raises a clear, actionable
        # error there; with a GPU backend (e.g. PcgGpuSolver) installed the whole
        # DC3D forward: assembly, RHS, and receiver sampling all follow
        # ``sigma.device``: runs end-to-end on CUDA. No premature per-operator
        # guard here.
        mesh = require_capable_mesh(ctx, ConnectivityMesh)
        if mesh.n_dim != 3:
            raise GeoBrainError(
                "DC3D requires a 3D mesh",
                object_name="DC3D", field="mesh.n_dim",
                expected=3, actual=mesh.n_dim,
            )
        is_structured = mesh.declares(StructuredMesh)
        # The structured path reads scalar ``mesh.spacing`` for assembly AND for
        # ``floor(coord/spacing)`` electrode/receiver snapping + the Robin source
        # centre: it assumes a UNIFORM grid. Reject a graded TensorMesh here
        # rather than silently snapping electrodes against the per-axis mean; use
        # an unstructured ConnectivityMesh (FV path) for graded geometry.
        if is_structured and not mesh.is_uniform:
            raise GeoBrainError(
                "DC3D structured path requires a uniform TensorMesh; "
                "non-uniform/graded meshes are not supported on this path "
                "(use an unstructured ConnectivityMesh for graded geometry)",
                object_name="DC3D", field="mesh",
                expected="uniform TensorMesh", actual="non-uniform TensorMesh",
            )
        sigma_flat = canonicalize_cell_field(mesh, sigma, name="sigma",
                                             owner="DC3D")
        # The structured branch keeps the grid layout throughout.
        sigma = sigma_flat.reshape(mesh.shape) if is_structured else sigma_flat

        if is_structured:
            # --- structured path (platform ``(nz, nx, ny)`` axis convention) ---
            nz, nx, ny = mesh.shape
            n = nz * nx * ny
            # TensorMesh.spacing convention for 3D is (dz, dx, dy) under the
            # (nz, nx, ny) shape used by assemble_poisson_3d.
            dz, dx, dy = mesh.spacing
            device, dtype = sigma.device, sigma.dtype

            # ``sparse_linear_solve_with_adjoint`` (scipy splu path) requires
            # float64. Promote sigma → float64 for assembly + solve, then cast
            # the solution back to the caller's dtype so downstream code (and
            # tests using torch's default float32) sees the original contract.
            sigma_f64 = sigma if sigma.dtype == torch.float64 else sigma.to(torch.float64)

            pin = (
                self._dirichlet_idx
                if self._dirichlet_idx is not None
                else (nz - 1) * nx * ny + (nx // 2) * ny + ny // 2
            )

            # Resolve the survey's PHYSICAL electrode coordinates (metres) to
            # structured grid indices via ``floor(coord / spacing)`` (clamped to
            # the mesh extent). ``floor`` is the exact inverse of the cell-centre
            # map ``coord = (i + 0.5)·spacing`` (``floor(i + 0.5) == i``),
            # whereas ``round`` is ambiguous at the half-integer cell centres
            # (banker's rounding). Physical x binds to axis-1 (spacing dx, size
            # nx) and physical y to axis-2 (spacing dy, size ny) under the
            # platform ``mesh.spacing == (dz, dx, dy)`` / ``(nz, nx, ny)`` order.
            import math as _math

            def _idx(coord: float, spacing: float, n_axis: int) -> int:
                return int(
                    min(max(_math.floor(float(coord) / float(spacing)), 0), n_axis - 1)
                )

            src_iz = _idx(self.survey.source_z, dz, nz)
            src_iy = _idx(self.survey.source_y, dy, ny)
            src_ix = _idx(self.survey.source_x, dx, nx)
            snk_iz = _idx(self.survey.sink_z, dz, nz)
            snk_iy = _idx(self.survey.sink_y, dy, ny)
            snk_ix = _idx(self.survey.sink_x, dx, nx)

            # Source / sink physical coordinates (cell centres, uniform mesh).
            # Used only when boundary='robin' to anchor the Robin distance `r`.
            source_center: tuple[float, float, float] | None = None
            if self._boundary == "robin":
                sx = 0.5 * (src_ix + snk_ix + 1) * float(dx)
                sy = 0.5 * (src_iy + snk_iy + 1) * float(dy)
                sz = 0.5 * (src_iz + snk_iz + 1) * float(dz)
                source_center = (sx, sy, sz)

            def _assemble_struct_csr(sg3d: torch.Tensor) -> torch.Tensor:
                """Assemble the structured DC3D ``A`` (CSR) for a ``(nz,ny,nx)`` σ.

                Routes through whichever boundary the operator is configured for,
                the mesh-agnostic seam (``boundary='mixed'``) or the legacy
                Neumann+pin / diagonal-Robin assembler, so a constant-σ_ref
                re-assembly for singularity removal is byte-consistent with the
                σ-dependent ``A``.
                """
                if self._boundary == "mixed":
                    # Mesh-agnostic seam route: pure-Neumann FV stencil +
                    # Dey–Morrison Robin diagonals on the outer faces, NO pin (the
                    # Robin term makes A non-singular). ``assemble_poisson_fv`` on
                    # the structured mesh's own FaceRecords reproduces the legacy
                    # 7-point stencil; ``boundary_faces()`` shares the flat-index
                    # convention, so the robin cells line up with σ.
                    from ..numerics.finite_volume.poisson_fv import assemble_poisson_fv

                    # The seam wants a flat ``(n_cells,)`` σ. The TensorMesh
                    # ``(nz, nx, ny)`` row-major reshape is y-fastest: exactly the
                    # ``iy + ix*ny + iz*nx*ny`` flat-index convention of
                    # ``face_neighbors()`` / ``boundary_faces()``, so cells line up.
                    return assemble_poisson_fv(
                        mesh.face_neighbors(), sg3d.reshape(-1),
                        boundary=self._mixed_boundary_dict(mesh, pin=None),
                    ).to_sparse_csr()
                return _assemble_dc3d_sparse(
                    sg3d, dx=dx, dz=dz, dy=dy,
                    dirichlet_idx=pin,
                    boundary=self._assembly_boundary,
                    source_center=source_center,
                )

            A = _assemble_struct_csr(sigma_f64)

            # Source / sink current vector. Defensively zero the pin row if
            # an electrode collides with the pin (rare but possible).
            src_flat = _flat3(src_iz, src_ix, src_iy, nx, ny)
            snk_flat = _flat3(snk_iz, snk_ix, snk_iy, nx, ny)
            b = torch.zeros(n, dtype=torch.float64, device=device).scatter_add(
                0,
                torch.tensor([src_flat, snk_flat], device=device, dtype=torch.long),
                torch.tensor(
                    [+self.survey.current, -self.survey.current],
                    dtype=torch.float64, device=device,
                ),
            )
            # The pin only matters when it is actually present: dropped by both
            # boundary='robin' (legacy diagonal Robin) and boundary='mixed'
            # (seam Robin); kept only in the default 'pin' mode.
            has_pin_struct = self._boundary == "pin"
            if has_pin_struct and (src_flat == pin or snk_flat == pin):
                b = b.clone()
                b[pin] = 0.0

            sr_ratio: float | None = None
            sigma_ref_used: float | None = None
            if self._singularity_removal:
                # Primary/secondary split on the structured FV system. The
                # TensorMesh cell_centers()/cell_volumes() are in the SAME
                # x-fastest flat order as the assembler (``_flat3``), so the
                # source/sink flat indices index them directly. ``A_ref`` is the
                # same operator re-assembled at the constant σ_ref (detached).
                sigma_ref_used = self._resolve_sigma_ref(sigma_f64)
                sigma_ref_3d = torch.full_like(
                    sigma_f64, float(sigma_ref_used)
                ).detach()
                A_ref = _assemble_struct_csr(sigma_ref_3d)
                sr_pin = None if self._boundary == "mixed" else int(pin)
                # Pass the σ-matrix as COO (graph preserved through CSR→COO) so the
                # SR helper can share its dense values between matvec and solve.
                phi, _u_p, sr_ratio = self._solve_with_singularity_removal(
                    A.to_sparse_coo(),
                    A_ref,
                    cell_centers=mesh.cell_centers(),
                    cell_volumes=mesh.cell_volumes(),
                    src_cell=int(src_flat),
                    snk_cell=int(snk_flat),
                    sigma_ref=sigma_ref_used,
                    pin=sr_pin,
                )
            elif self._use_closed_form_sigma_jacobian:
                phi = _dc3d_solve_closed_form(
                    mesh, A, sigma_f64, b, dirichlet_idx=pin,
                )
            else:
                phi = sparse_linear_solve_with_adjoint(A, b)
            if phi.dtype != dtype:
                phi = phi.to(dtype)
            phi_3d = phi.view(nz, nx, ny)

            # Receivers: physical metres → structured indices via
            # ``floor(coord / spacing)`` (clamped), matching the source/sink
            # resolution above (``floor`` inverts the cell-centre map exactly).
            # x → axis-1 (dx, nx), y → axis-2 (dy, ny).
            rcv_z = (self.survey.rcv_z.to(torch.float64) / float(dz)).floor().clamp_(
                0, nz - 1
            ).to(device).to(torch.long)
            rcv_x = (self.survey.rcv_x.to(torch.float64) / float(dx)).floor().clamp_(
                0, nx - 1
            ).to(device).to(torch.long)
            rcv_y = (self.survey.rcv_y.to(torch.float64) / float(dy)).floor().clamp_(
                0, ny - 1
            ).to(device).to(torch.long)
            voltage = phi_3d[rcv_z, rcv_x, rcv_y]

            return ForwardOutput(
                data={"voltage": voltage},
                fields={"potential": phi_3d},  # full φ field, diagnostic, not an observation
                metadata={
                    "current": self.survey.current,
                    "n_rcv": self.survey.n_rcv,
                    "pin": None if self._boundary == "mixed" else pin,
                    "boundary": self._boundary,
                    "singularity_removal": self._singularity_removal,
                    "sigma_ref": sigma_ref_used,
                    "secondary_primary_ratio": sr_ratio,
                },
            )

        # --- ConnectivityMesh (octree / unstructured) path ---
        if self._boundary == "robin":
            raise GeoBrainError(
                "DC3D boundary='robin' is only supported on a TensorMesh "
                "(use boundary='mixed' for the mesh-agnostic absorbing boundary)",
                object_name="DC3D", field="boundary",
                expected="'pin' or 'mixed' on a non-TensorMesh", actual=self._boundary,
            )
        if self._use_closed_form_sigma_jacobian:
            raise GeoBrainError(
                "use_closed_form_sigma_jacobian is TensorMesh-only",
                object_name="DC3D", field="use_closed_form_sigma_jacobian",
                expected=False, actual=True,
            )
        from ..numerics.finite_volume.poisson_fv import assemble_poisson_fv

        device, dtype = sigma.device, sigma.dtype
        n = int(mesh.n_cells)
        sigma_f64 = sigma if sigma.dtype == torch.float64 else sigma.to(torch.float64)
        pin = self._dirichlet_idx if self._dirichlet_idx is not None else 0
        if self._boundary == "mixed":
            # Seam Robin (Dey–Morrison) on the unstructured/octree outer faces;
            # NO pin (the Robin term anchors the field). Identical seam call as
            # the structured branch: ``boundary_faces()`` is a ConnectivityMesh
            # method, so this path needs no special-casing per mesh type.
            boundary_dict = self._mixed_boundary_dict(mesh, pin=None)
            has_pin_unstr = False
        else:
            boundary_dict = {"dirichlet_idx": pin}
            has_pin_unstr = True
        A = assemble_poisson_fv(
            mesh.face_neighbors(), sigma_f64, boundary=boundary_dict,
        )
        centers = mesh.cell_centers().to(torch.float64)
        src_flat = self._nearest_cell(centers,
            self.survey.source_z, self.survey.source_x, self.survey.source_y)
        snk_flat = self._nearest_cell(centers,
            self.survey.sink_z, self.survey.sink_x, self.survey.sink_y)
        # If the source and sink map to the SAME octree leaf, the +I/−I scatter
        # cancels to an all-zero RHS and the solve silently returns φ ≡ 0. This
        # happens when the octree is too coarse to resolve the electrode
        # separation. Fail loudly. (Receivers coinciding is acceptable.)
        if src_flat == snk_flat:
            raise GeoBrainError(
                "DC3D source and sink electrodes map to the same octree leaf, "
                "the source/sink current would cancel to a zero RHS (silent "
                "phi=0). Refine the octree near the electrodes.",
                object_name="DC3D", field="survey",
                expected="source and sink in distinct octree leaves",
                actual="same leaf, octree too coarse for this electrode separation",
            )
        b = torch.zeros(n, dtype=torch.float64, device=device).scatter_add(
            0,
            torch.tensor([src_flat, snk_flat], device=device, dtype=torch.long),
            torch.tensor(
                [+self.survey.current, -self.survey.current],
                dtype=torch.float64, device=device,
            ),
        )
        if has_pin_unstr and (src_flat == pin or snk_flat == pin):
            b = b.clone()
            b[pin] = 0.0
        sr_ratio: float | None = None
        sigma_ref_used: float | None = None
        if self._singularity_removal:
            # Primary/secondary split on the unstructured/octree FV system. The
            # electrode self-cell 1/r singularity (the dominant coarse-tet error)
            # is carried analytically; only the smooth secondary is solved.
            # ``A_ref`` is the SAME seam operator (same boundary dict) re-assembled
            # at the constant σ_ref (detached). The σ-matrix is passed as the
            # native COO so the helper shares its dense values across matvec+solve.
            sigma_ref_used = self._resolve_sigma_ref(sigma_f64)
            sigma_ref_vec = torch.full_like(sigma_f64, float(sigma_ref_used)).detach()
            A_ref = assemble_poisson_fv(
                mesh.face_neighbors(), sigma_ref_vec, boundary=boundary_dict,
            ).to_sparse_csr()
            sr_pin = None if not has_pin_unstr else int(pin)
            phi, _u_p, sr_ratio = self._solve_with_singularity_removal(
                A,
                A_ref,
                cell_centers=centers,
                cell_volumes=mesh.cell_volumes(),
                src_cell=int(src_flat),
                snk_cell=int(snk_flat),
                sigma_ref=sigma_ref_used,
                pin=sr_pin,
            )
        else:
            phi = sparse_linear_solve_with_adjoint(A.to_sparse_csr(), b)
        if phi.dtype != dtype:
            phi = phi.to(dtype)
        rcv_flat = torch.tensor(
            [
                self._nearest_cell(centers, z, x, y)
                for z, x, y in zip(
                    self.survey.rcv_z.tolist(),
                    self.survey.rcv_x.tolist(),
                    self.survey.rcv_y.tolist(),
                )
            ],
            dtype=torch.long, device=device,
        )
        voltage = phi[rcv_flat]
        return ForwardOutput(
            data={"voltage": voltage},
            fields={"potential": phi},
            metadata={
                "current": self.survey.current,
                "n_rcv": self.survey.n_rcv,
                "pin": None if self._boundary == "mixed" else pin,
                "boundary": self._boundary,
                "singularity_removal": self._singularity_removal,
                "sigma_ref": sigma_ref_used,
                "secondary_primary_ratio": sr_ratio,
            },
        )


# ---------------------------------------------------------------------------
# EM3: closed-form σ-Jacobian for the 7-point FV Poisson stencil
# ---------------------------------------------------------------------------
#
# The closed-form Poisson-FV σ-Jacobian trio is a pure numerics quantity, so
# it lives in the numerics layer beside its curl-curl sibling
# (``build_sigma_jacobian_curl_curl``) rather than inside this operator file.
# ``build_sigma_jacobian_dc3d`` is re-exported here so ``from ...dc3d import
# build_sigma_jacobian_dc3d`` (used by tests and historically by ip) stays
# source-compatible; ``_dc3d_solve_closed_form`` is the closed-form-σ solve
# the forward path dispatches to.

from ..numerics.finite_volume._sigma_jacobian import (  # noqa: E402  re-export
    _dc3d_solve_closed_form,
    build_sigma_jacobian_dc3d as build_sigma_jacobian_dc3d,
)


# ---------------------------------------------------------------------------
# EM3: ElectrodeArray + DipoleDipoleSurvey
# ---------------------------------------------------------------------------
#
# Moved out to :mod:`_dc3d_survey` to keep this file focused on the operator
# + sparse assembly + closed-form σ-Jacobian. They are re-exported below so
# ``from geobrain.physics.em.static.dc3d import ElectrodeArray``
# (and ``DipoleDipoleSurvey``) keep working.

from ._dc3d_survey import (  # noqa: E402
    BoundDipoleDipoleSurvey,
    DipoleDipoleSurvey,
    ElectrodeArray,
)


# Re-stamp __module__ so the original public location of these classes is
# preserved for introspection and pickling.
ElectrodeArray.__module__ = __name__
DipoleDipoleSurvey.__module__ = __name__
BoundDipoleDipoleSurvey.__module__ = __name__
