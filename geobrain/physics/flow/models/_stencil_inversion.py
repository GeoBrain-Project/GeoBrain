"""
Differentiable MPFA-stencil rebuilds for adjoint **parameter inversion**, shared by
the thermal flow models.

The MPFA models bake ``perm`` into the Darcy stencil ``L`` and ``λ_rock`` into the
rock-conduction stencil ``L_rock = stencil((1−φ)·λ_rock·I)`` at ``__init__``, so
neither appears in the residual graph; the adjoint's autograd VJP would see a zero
gradient. This mixin provides differentiable rebuilds :meth:`build_L_perm` /
:meth:`build_L_rock` / :meth:`build_L_phi` (assembled from per-face MPFA-O coefficients
that stay differentiable w.r.t. the per-cell field), which the residual uses when a
``perm=`` / ``lam_rock=`` / ``porosity=`` field is passed, letting the adjoint capture
``∂R/∂(perm, λ_rock, φ)`` for permeability / thermal-conductivity / porosity inversion.
``λ_fluid`` enters only as a scalar multiplier of the porosity-weighted geometric
stencil ``L_phi`` (independent of ``λ``), so it needs no rebuild, the residual simply
overrides the stored float.

A model is 2-D or 3-D via the class attribute :attr:`_MPFA_DIM` (the 3-D model classes
set it to ``3``); the mixin picks the matching per-face MPFA-O stencil function, so a
single implementation serves both. On an **interior** (no-flow) model the rebuild is the
``(n_faces, n_cells)`` operator; on a **Dirichlet-bounded** model it is the
**ghost-augmented** ``(n_faces, n_cells + n_dir)`` operator, rebuilt from the stored
:attr:`dir_list` (the Dirichlet edge/face ids in ghost order) via the bc-stencil, the
column/row order is geometry-determined, so a rebuild at the stored field reproduces the
stored buffer bit-for-bit and the residual consumes it exactly like the stored augmented
``L`` (it already concatenates the ghost potentials). A Dirichlet model that fails to
store :attr:`dir_list` raises (rather than silently producing a mismatched interior
operator).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias, cast

import torch

from ....core import GeoBrainError
from ..errors import FlowCapabilityError
from ..discretization.transmissibility import expand_perm

FaceStencils: TypeAlias = dict[int, dict[int, torch.Tensor]]
FullStencilBuilder: TypeAlias = Callable[[object, torch.Tensor], FaceStencils]
BoundaryStencilBuilder: TypeAlias = Callable[
    [object, torch.Tensor, Sequence[int]],
    tuple[torch.Tensor, torch.Tensor, list[int]],
]


class _StencilGrid(Protocol):
    """Connectivity exposed by either a two- or three-dimensional MPFA grid."""

    @property
    def edge_cells(self) -> tuple[tuple[int, ...], ...]: ...

    @property
    def face_cells(self) -> tuple[tuple[int, ...], ...]: ...


class StencilInversionMixin:
    """Differentiable Darcy / rock-conduction MPFA-O stencil rebuilds for inversion.

    Requires the host model to provide ``self.grid``, ``self.n_cells``, ``self.phi``,
    and, when the model has Dirichlet pressure boundaries, ``self.p_bc`` (the
    Dirichlet flag) and ``self.dir_list`` (the Dirichlet edge/face ids in ghost order,
    so the augmented stencil can be rebuilt to match the stored operator). 2-D by
    default; a 3-D model sets the class attribute ``_MPFA_DIM = 3``."""

    _MPFA_DIM: int = 2
    grid: _StencilGrid
    n_cells: int
    phi: torch.Tensor

    def _stencil_full_fn(self) -> FullStencilBuilder:
        if self._MPFA_DIM == 3:
            from ..discretization.mpfa3d import mpfa_o_face_flux_stencils_3d_full
            return cast(FullStencilBuilder, mpfa_o_face_flux_stencils_3d_full)
        from ..discretization.mpfa import mpfa_o_face_flux_stencils_full
        return cast(FullStencilBuilder, mpfa_o_face_flux_stencils_full)

    def _stencil_bc_fn(self) -> BoundaryStencilBuilder:
        """The **ghost-augmented** (Dirichlet) per-face MPFA-O stencil function, matching
        the one the model used at ``__init__`` to build the stored operator."""
        if self._MPFA_DIM == 3:
            from ..discretization.mpfa3d import mpfa_o_face_flux_stencils_3d_bc
            return cast(BoundaryStencilBuilder, mpfa_o_face_flux_stencils_3d_bc)
        from ..discretization.mpfa import mpfa_o_face_flux_stencils_bc
        return cast(BoundaryStencilBuilder, mpfa_o_face_flux_stencils_bc)

    def _diff_stencil(self, tensor_field: torch.Tensor) -> torch.Tensor:
        """Differentiably assemble the MPFA-O stencil matrix from a ``(n_cells, d, d)``
        conductivity-tensor field.

        Interior (no-flow) model → the ``(n_faces, n_cells)`` operator assembled via
        ``index_put`` (row order = ``sorted(faces)`` = :attr:`face_lr`). Dirichlet-bounded
        model → the **ghost-augmented** ``(n_faces, n_cells + n_dir)`` operator from the
        bc-stencil with the stored :attr:`dir_list` (its dense ``L`` is itself
        differentiable in the field; row/column order is identical to the stored buffer).
        Both stay differentiable w.r.t. the field. The MPFA face set is geometry-determined,
        identical for perm and conductivity."""
        if getattr(self, "p_bc", None) is not None:
            dir_list = getattr(self, "dir_list", None)
            if dir_list is None:
                raise GeoBrainError(
                    "Dirichlet-bounded model is missing self.dir_list, needed to rebuild the "
                    "ghost-augmented MPFA stencil for inversion. The model must store the "
                    "Dirichlet edge/face ids (the bc-stencil's dir_list, ghost order) at "
                    "construction so the rebuilt operator matches the stored L.",
                    object_name=type(self).__name__, field="dir_list",
                    expected="stored Dirichlet edge/face id list", actual="None",
                )
            L, _, _ = self._stencil_bc_fn()(self.grid, tensor_field, dir_list)
            return L
        st = self._stencil_full_fn()(self.grid, tensor_field)
        faces = sorted(st)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[torch.Tensor] = []
        for fi, f in enumerate(faces):
            for c, t in st[f].items():
                rows.append(fi)
                cols.append(c)
                vals.append(t)
        idx = torch.tensor([rows, cols], device=tensor_field.device)
        return tensor_field.new_zeros(len(faces), self.n_cells).index_put(
            (idx[0], idx[1]), torch.stack(vals))

    def _diff_sparse_stencil(self, tensor_field: torch.Tensor) -> torch.Tensor:
        """Build the differentiable no-flow inversion stencil as sparse COO.

        Dirichlet ghost stencils retain their legacy dense differentiable
        implementation until PyTorch supports vmap-backed sparse backward for
        that augmented operator. The sparse contract fails before assembly for
        that combination instead of silently materialising a dense fallback.
        """

        if getattr(self, "p_bc", None) is not None:
            raise FlowCapabilityError(
                "Sparse differentiable MPFA inversion is not available with Dirichlet ghosts",
                object_name=type(self).__name__,
                field="p_bc",
                expected="no-flow boundary for sparse inversion",
                actual="Dirichlet boundary",
                hint="use the declared dense differentiable inversion path explicitly",
            )
        from ..discretization.mpfa import _pack_sparse_face_stencil

        stencils = self._stencil_full_fn()(self.grid, tensor_field)
        face_cells = (
            self.grid.face_cells if self._MPFA_DIM == 3 else self.grid.edge_cells
        )
        return _pack_sparse_face_stencil(
            stencils,
            face_cells,
            n_cells=self.n_cells,
            like=tensor_field,
        ).matrix

    def build_L_perm(self, perm: torch.Tensor) -> torch.Tensor:
        """Differentiable Darcy MPFA-O stencil from a per-cell ``perm`` field (for
        permeability inversion); ``(n_cells,)`` isotropic or ``(n_cells, d, d)``.
        Ghost-augmented on a Dirichlet-bounded model (see :meth:`_diff_stencil`)."""
        K = expand_perm(perm.unsqueeze(-1), self._MPFA_DIM) if perm.dim() == 1 else perm
        return self._diff_stencil(K)

    def build_sparse_L_perm(self, perm: torch.Tensor) -> torch.Tensor:
        """Sparse COO counterpart of :meth:`build_L_perm` for no-flow models."""

        K = expand_perm(perm.unsqueeze(-1), self._MPFA_DIM) if perm.dim() == 1 else perm
        return self._diff_sparse_stencil(K)

    def build_L_rock(
        self,
        lam_rock: float | torch.Tensor,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable rock-conduction stencil ``stencil((1−φ)·λ_rock·I)`` from a
        per-cell (or scalar) rock thermal conductivity, and optionally a differentiable
        porosity field ``porosity`` (defaults to ``self.phi``), for conductivity and/or
        **porosity** inversion (``λ_rock`` inversion passes ``porosity=None``)."""
        phi = self.phi if porosity is None else porosity
        eye = torch.eye(self._MPFA_DIM, dtype=phi.dtype, device=phi.device)
        tensor = ((1.0 - phi) * lam_rock).reshape(-1, 1, 1) * eye
        return self._diff_stencil(tensor)

    def build_sparse_L_rock(
        self,
        lam_rock: float | torch.Tensor,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sparse COO rock-conduction stencil for no-flow models."""

        phi = self.phi if porosity is None else porosity
        eye = torch.eye(self._MPFA_DIM, dtype=phi.dtype, device=phi.device)
        tensor = ((1.0 - phi) * lam_rock).reshape(-1, 1, 1) * eye
        return self._diff_sparse_stencil(tensor)

    def build_L_phi(self, porosity: torch.Tensor) -> torch.Tensor:
        """Differentiable porosity-weighted geometric conduction stencil
        ``stencil(φ·I)`` from a differentiable porosity field, the φ-dependence of the
        fluid-conduction path, for **porosity inversion** (the fluid conductivity
        ``Σ_α λ_α S̄_α`` multiplies this stencil in the residual)."""
        eye = torch.eye(self._MPFA_DIM, dtype=porosity.dtype, device=porosity.device)
        return self._diff_stencil(porosity.reshape(-1, 1, 1) * eye)

    def build_sparse_L_phi(self, porosity: torch.Tensor) -> torch.Tensor:
        """Sparse COO porosity-conduction stencil for no-flow models."""

        eye = torch.eye(self._MPFA_DIM, dtype=porosity.dtype, device=porosity.device)
        return self._diff_sparse_stencil(porosity.reshape(-1, 1, 1) * eye)


__all__ = ["StencilInversionMixin"]
