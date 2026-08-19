"""
PEC-pinned Whitney edge system: assemble + symmetric Dirichlet pin (+ solve).

The 3-D curl-curl operators (FDEM3D / MT3D / TEM3D) all assemble the same
``A = stiffness_coeff·K + mass_coeff·M`` over an :class:`EdgeAssemblyPlan`,
truncate it with the symmetric Dirichlet (PEC) pin, tangential E = 0 on the
outer boundary, and solve. Only the per-cell ``mass_coeff`` /
``stiffness_coeff`` (the physics: ``iωσ`` vs ``σ/Δt``, paired with ``1/μ₀``) and
the right-hand side differ; the assemble → pin → build-sparse → solve
orchestration was repeated verbatim in each operator. The leaf primitives
already live in :mod:`.assembly` and :mod:`.topology`; this module collapses the
editorial tail above them.

- :func:`assemble_pec_pinned_operator`: assemble + pin → one sparse COO matrix.
  TEM3D's time loop calls this directly so it can cache the splu factor across
  substeps that share a Δt.
- :func:`solve_pec_pinned_edge_system`: the one-shot convenience (build the pin
  plan if absent, zero the RHS boundary rows, solve) for the FDEM3D / MT3D
  per-frequency secondary solves (one factorisation per call).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

import torch

from geobrain.core.adjoint import sparse_linear_solve_with_adjoint

from .assembly import EdgeAssemblyPlan, assemble_edge_operator
from .topology import (
    PecPinPlan,
    apply_pec_pin,
    build_pec_pin_plan,
    pec_zero_rhs,
)


def assemble_pec_pinned_operator(
    plan: EdgeAssemblyPlan,
    pin_plan: PecPinPlan,
    *,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
) -> torch.Tensor:
    """Assemble the Whitney edge operator and apply the symmetric PEC pin.

    Assemble ``A = stiffness_coeff·K + mass_coeff·M`` over ``plan`` (the
    differentiable COO hot path), zero the boundary rows/columns and append the
    unit-diagonal pins from ``pin_plan``, and return the result as a sparse COO
    tensor. The σ/Δt-dependent coefficients carry the physics; the boundary
    truncation is the shared DC3D / edge-element seam. ``pin_plan`` is
    coefficient-independent, so callers in a frequency / timestep loop build it
    once (``build_pec_pin_plan(boundary_mask, torch.stack([plan.rows,
    plan.cols]))``) and reuse it here every iteration.

    Args:
        plan: the mesh's :class:`EdgeAssemblyPlan`.
        pin_plan: the mesh's :class:`PecPinPlan` (matching ``plan``'s pattern).
        mass_coeff: ``(n_cells,)`` float64/complex128 mass-term coefficient.
        stiffness_coeff: ``(n_cells,)`` float64/complex128 curl-curl coefficient.

    Returns:
        The pinned operator as a ``(n_edges, n_edges)`` sparse COO tensor
        (uncoalesced; the solve seam coalesces internally).
    """
    indices, values, shape = assemble_edge_operator(
        plan, mass_coeff=mass_coeff, stiffness_coeff=stiffness_coeff,
    )
    pin_indices, pin_values = apply_pec_pin(pin_plan, indices, values)
    return torch.sparse_coo_tensor(pin_indices, pin_values, shape)


def solve_pec_pinned_edge_system(
    plan: EdgeAssemblyPlan,
    boundary_mask: torch.Tensor,
    *,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
    rhs: torch.Tensor,
    pin_plan: PecPinPlan | None = None,
) -> torch.Tensor:
    """Assemble, PEC-pin, and solve one Whitney edge system.

    The one-shot tail of the FDEM3D / MT3D per-frequency secondary solves: build
    the :class:`PecPinPlan` from ``plan`` if not supplied, assemble and pin
    through :func:`assemble_pec_pinned_operator`, zero the boundary rows of
    ``rhs`` (:func:`pec_zero_rhs`, the symmetric counterpart of the matrix pin),
    and solve through the shared sparse implicit-adjoint seam. One factorisation
    per call; TEM3D's time loop instead reuses
    :func:`assemble_pec_pinned_operator` so it can cache the splu factor across
    substeps.

    Args:
        plan: the mesh's :class:`EdgeAssemblyPlan`.
        boundary_mask: ``(n_edges,)`` bool outer-boundary edges (the PEC pins).
        mass_coeff: ``(n_cells,)`` float64/complex128 mass-term coefficient.
        stiffness_coeff: ``(n_cells,)`` float64/complex128 curl-curl coefficient.
        rhs: ``(n_edges,)`` or ``(n_edges, k)`` right-hand side; multiple
            columns solve against ONE factorisation.
        pin_plan: optional precomputed :class:`PecPinPlan`; built from
            ``boundary_mask`` and ``plan``'s COO pattern when ``None``.

    Returns:
        The solution, same shape and dtype as ``rhs``.
    """
    if pin_plan is None:
        pin_plan = build_pec_pin_plan(
            boundary_mask, torch.stack([plan.rows, plan.cols]),
        )
    a_sp = assemble_pec_pinned_operator(
        plan, pin_plan, mass_coeff=mass_coeff, stiffness_coeff=stiffness_coeff,
    )
    rhs = pec_zero_rhs(boundary_mask, rhs)
    return sparse_linear_solve_with_adjoint(a_sp, rhs)
