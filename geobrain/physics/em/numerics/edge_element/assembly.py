"""
Global assembly of Whitney edge-element operators from an EdgeConnectivityMesh.

The seam mirrors the FV one (:func:`assemble_poisson_fv` consumes
``FaceRecords``): the mesh hands out RECORDS only (``cell_edges()`` ids +
signs, ``cell_nodes()``/``node_coords()``), and the operator algebra lives
here. A :class:`EdgeAssemblyPlan` precomputes everything σ-independent ONCE
(local geometric factors, flattened COO index pattern, sign outer products);
:func:`assemble_edge_operator` then turns per-cell coefficients into global COO
triplets with a single broadcast multiply, the differentiable, per-frequency /
per-timestep hot path of the curl-curl family

    A(σ, ω) = stiffness_coeff·K + mass_coeff·M,
    e.g. FDEM:  A = (1/μ)·K + iωσ·M.

Orientation: local matrices are built in the LOCAL edge directions
(first → second stored node); the mesh's ``cell_edges()`` signs ``s ∈ {±1}``
convert to the global canonical direction (small node id → large), so the
global stamp is ``A[E_e, E_f] += s_e s_f A_loc[e, f]``.

Duplicates: the returned COO triplets are NOT coalesced, each edge shared by
``k`` cells contributes ``k`` diagonal entries (and likewise for shared
off-diagonal pairs). This is deliberate: the core sparse-solve seam
(:func:`geobrain.core.adjoint.factorize_sparse` /
:func:`~geobrain.core.adjoint.sparse_linear_solve_with_adjoint`) coalesces COO
input internally via ``Tensor.coalesce()``, which is differentiable in the
values, so the coefficient gradient survives the duplicate-summing. Callers
needing unique entries can ``torch.sparse_coo_tensor(...).coalesce()``
themselves (``to_dense()`` also sums duplicates).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geobrain.core.errors import GeoBrainError
from geobrain.mesh import ensure_capable_mesh
from geobrain.mesh.capabilities import EdgeConnectivityMesh

from .whitney import barycentric_gradients, local_curl_curl, local_edge_mass

__all__ = [
    "EdgeAssemblyPlan",
    "assemble_edge_operator",
    "build_edge_assembly_plan",
    "edge_operator_matvec",
]

_COEFF_DTYPES = (torch.float64, torch.complex128)


@dataclass(frozen=True, eq=False)
class EdgeAssemblyPlan:
    """Precomputed, coefficient-independent Whitney assembly data for one mesh.

    **Experimental API**; may change in a minor release before it
    stabilizes. Part of the experimental edge
    (Whitney/Nédélec) branch.

    Built once by :func:`build_edge_assembly_plan`; reused across every
    :func:`assemble_edge_operator` / :func:`edge_operator_matvec` call (e.g.
    one plan per mesh, many frequencies/timesteps). ``eq=False``:
    equality/hash are by object identity, a generated field-wise ``__eq__``
    would call ``bool()`` on multi-element tensor comparisons and raise
    ``RuntimeError``.

    Attributes:
        mass_geom: ``(n_cells, 6, 6)`` float64, unit-σ Whitney mass factors
            (:func:`~geobrain.physics.em.numerics.edge_element.local_edge_mass`).
        stiffness_geom: ``(n_cells, 6, 6)`` float64, unit-coefficient
            curl-curl factors
            (:func:`~geobrain.physics.em.numerics.edge_element.local_curl_curl`).
        rows: ``(n_cells·36,)`` long, flattened global COO row ids, the
            ``(c, e, f) ↦ E[c, e]`` outer expansion of the ``cell_edges()`` ids.
        cols: ``(n_cells·36,)`` long, flattened global COO column ids,
            ``(c, e, f) ↦ E[c, f]``.
        sign_outer: ``(n_cells, 6, 6)`` float64, ``s_e·s_f`` orientation outer
            products from the ``cell_edges()`` signs.
        n_edges: number of global unique edges (matrix dimension).
    """

    mass_geom: torch.Tensor
    stiffness_geom: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    sign_outer: torch.Tensor
    n_edges: int

    @property
    def n_cells(self) -> int:
        """Number of tetrahedral cells the plan was built over."""
        return int(self.mass_geom.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Global operator shape ``(n_edges, n_edges)``."""
        return (self.n_edges, self.n_edges)


def build_edge_assembly_plan(mesh) -> EdgeAssemblyPlan:
    """Precompute the Whitney assembly plan for an ``EdgeConnectivityMesh``.

    **Experimental API**; may change in a minor release before it
    stabilizes. Part of the experimental edge
    (Whitney/Nédélec) branch.

    Capability membership is checked BY DECLARATION (instance attribute first,
    class fallback, the same rule as ``require_mesh``): only a 3-D
    simplex-built mesh declares
    :class:`~geobrain.mesh.capabilities.EdgeConnectivityMesh`.

    Args:
        mesh: a mesh declaring ``EdgeConnectivityMesh`` (e.g. a 3-D
            simplex-built :class:`~geobrain.mesh.unstructured.UnstructuredMesh`).

    Returns:
        A frozen :class:`EdgeAssemblyPlan`.

    Raises:
        GeoBrainError: when the mesh does not declare ``EdgeConnectivityMesh``.
    """
    ensure_capable_mesh(mesh, EdgeConnectivityMesh,
                        owner="build_edge_assembly_plan")
    g, volume = barycentric_gradients(mesh.node_coords(), mesh.cell_nodes())
    mass_geom = local_edge_mass(g, volume)
    stiffness_geom = local_curl_curl(g, volume)

    cell_edge_ids, cell_edge_signs = mesh.cell_edges()
    n_cells = int(cell_edge_ids.shape[0])
    n_edges = int(mesh.edge_records().nodes.shape[0])
    rows = cell_edge_ids[:, :, None].expand(n_cells, 6, 6).reshape(-1)
    cols = cell_edge_ids[:, None, :].expand(n_cells, 6, 6).reshape(-1)
    sign_outer = cell_edge_signs[:, :, None] * cell_edge_signs[:, None, :]
    return EdgeAssemblyPlan(
        mass_geom=mass_geom,
        stiffness_geom=stiffness_geom,
        rows=rows.contiguous(),
        cols=cols.contiguous(),
        sign_outer=sign_outer,
        n_edges=n_edges,
    )


def _check_coeff(plan: EdgeAssemblyPlan, coeff: torch.Tensor, name: str) -> None:
    """Shape/dtype gate for a per-cell coefficient tensor."""
    if not isinstance(coeff, torch.Tensor) or coeff.dim() != 1 or (
        int(coeff.shape[0]) != plan.n_cells
    ):
        raise GeoBrainError(
            f"{name} must be a 1-D (n_cells,) tensor",
            object_name="assemble_edge_operator", field=name,
            expected=(plan.n_cells,),
            actual=tuple(coeff.shape) if isinstance(coeff, torch.Tensor)
            else type(coeff).__name__,
        )
    if coeff.dtype not in _COEFF_DTYPES:
        raise GeoBrainError(
            f"{name} dtype must be float64 or complex128",
            object_name="assemble_edge_operator", field=f"{name}.dtype",
            expected="float64 or complex128", actual=coeff.dtype,
        )


def _signed_local_values(
    plan: EdgeAssemblyPlan,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
) -> torch.Tensor:
    """``(n_cells, 6, 6)`` signed local stamps: the shared differentiable core.

    ``sign_outer ⊙ (mass_coeff·M_geo + stiffness_coeff·K_geo)`` with the
    coefficients broadcast per cell; dtype follows torch promotion (float64
    unless either coefficient is complex128).
    """
    _check_coeff(plan, mass_coeff, "mass_coeff")
    _check_coeff(plan, stiffness_coeff, "stiffness_coeff")
    local = (
        mass_coeff.view(-1, 1, 1) * plan.mass_geom
        + stiffness_coeff.view(-1, 1, 1) * plan.stiffness_geom
    )
    return plan.sign_outer * local


def assemble_edge_operator(
    plan: EdgeAssemblyPlan,
    *,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Assemble ``A = stiffness_coeff·K + mass_coeff·M`` as COO triplets.

    **Experimental API**; may change in a minor release before it
    stabilizes. Part of the experimental edge
    (Whitney/Nédélec) branch.

    The differentiable hot path: gradients flow from the returned ``values``
    back to both coefficient tensors (and through them to σ, μ, …).

    Args:
        plan: the mesh's precomputed :class:`EdgeAssemblyPlan`.
        mass_coeff: ``(n_cells,)`` float64 or complex128 cell coefficient of
            the Whitney mass term (e.g. ``iωμ₀σ`` for the FDEM E-field system,
            with σ autograd-live).
        stiffness_coeff: ``(n_cells,)`` float64 or complex128 cell coefficient
            of the curl-curl term (e.g. ``1/μ``).

    Returns:
        ``(indices, values, shape)``: ``indices`` ``(2, nnz)`` long,
        ``values`` ``(nnz,)`` with ``nnz = n_cells·36``, and
        ``shape = (n_edges, n_edges)``. **Not coalesced**: entries sharing an
        ``(row, col)`` slot are duplicated, one per contributing cell. The core
        sparse-solve seam (``factorize_sparse`` /
        ``sparse_linear_solve_with_adjoint``) coalesces internally with the
        values-differentiable ``Tensor.coalesce()``, and ``to_dense()`` also
        sums duplicates; see the module docstring.
    """
    values = _signed_local_values(plan, mass_coeff, stiffness_coeff).reshape(-1)
    indices = torch.stack([plan.rows, plan.cols], dim=0)
    return indices, values, plan.shape


def _check_matvec_x(plan: EdgeAssemblyPlan, x: torch.Tensor) -> None:
    """Shape/dtype gate for the edge vector of a matvec."""
    if x.dim() != 1 or int(x.shape[0]) != plan.n_edges:
        raise GeoBrainError(
            "edge_operator_matvec: x must be 1-D (n_edges,)",
            object_name="edge_operator_matvec", field="x",
            expected=(plan.n_edges,), actual=tuple(x.shape),
        )
    if x.dtype not in _COEFF_DTYPES:
        raise GeoBrainError(
            "edge_operator_matvec: x dtype must be float64 or complex128",
            object_name="edge_operator_matvec", field="x.dtype",
            expected="float64 or complex128", actual=x.dtype,
        )


def _matvec_compute(
    plan: EdgeAssemblyPlan,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """The raw gather/``scatter_add`` ``A x`` kernel.

    The ONE op sequence both matvec entry points run,
    :class:`_EdgeOperatorMatvec`'s forward (under no-grad) and
    :func:`_edge_operator_matvec_autograd` (recorded), so the custom-Function
    output is bit-identical to the straight-through autograd reference.
    """
    values = _signed_local_values(plan, mass_coeff, stiffness_coeff).reshape(-1)
    out_dtype = torch.result_type(values, x)
    out = torch.zeros(plan.n_edges, dtype=out_dtype, device=x.device)
    return out.scatter_add(0, plan.rows, values.to(out_dtype) * x[plan.cols])


def _match_leaf_dtype(grad: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
    """Cast a backward grad to its leaf's dtype.

    Torch's complex-autograd convention stores ``∂L/∂Re(z) + i·∂L/∂Im(z)``
    for a real loss ``L``; a REAL leaf feeding a complex chain takes the real
    part of that grad (``Re(z)`` moves both ``Re`` and ``Im`` of downstream
    values, and the imaginary component of the Wirtinger grad cancels), the
    same reduction torch's built-in ops apply.
    """
    if grad.is_complex() and not leaf.is_complex():
        grad = grad.real
    return grad.to(leaf.dtype)


class _EdgeOperatorMatvec(torch.autograd.Function):
    """O(n_cells + n_edges) checkpointed ``A x`` (vs O(36·n_cells) per call
    for the straight-through tape).

    Recording :func:`_matvec_compute` directly saves three fresh
    ``(n_cells·36,)`` intermediates per call (the signed local stamps, their
    dtype-cast copy, and the gathered ``x[cols]``) plus references to the
    ``(n_cells, 6, 6)`` plan constants; per BE substep and per source in the
    TEM3D edge loop, the dominant (linear-in-``n_steps``) memory term. This
    Function saves ONLY the two ``(n_cells,)`` coefficients and ``x``
    (``n_edges``,); the σ-independent plan geometry is kept as a plain ``ctx``
    attribute reference, never a SavedVariable, never in the graph, and the
    gathers/einsums are recomputed in backward.

    Backward math (the 6×6 bilinear form, per cell ``c`` with edge table
    ``E = cell_edge_ids``, signs ``S[c,e,f] = s_e s_f`` and geometric factors
    ``M``/``K`` all REAL and symmetric in ``(e, f)``):

        y_r = Σ_{c,e,f : E[c,e]=r} S[c,e,f]·(m_c·M[c,e,f] + k_c·K[c,e,f])·x_{E[c,f]}

    is linear (holomorphic) in ``m``, ``k`` and ``x``, so under torch's
    convention (real loss; ``grad_in = Σ_i conj(∂y_i/∂in)·grad_out_i``):

        grad_m[c] = Σ_{e,f} (S⊙M)[c,e,f] · g[E[c,e]] · conj(x[E[c,f]])
        grad_k[c] = Σ_{e,f} (S⊙K)[c,e,f] · g[E[c,e]] · conj(x[E[c,f]])
        grad_x    = Aᴴ g = conj(A) g          (A = Aᵀ: M, K, S symmetric)
                  = the SAME kernel on conjugated coefficients.

    Real leaves in a complex chain take the real part
    (:func:`_match_leaf_dtype`). First-order only (``once_differentiable``):
    nothing in the EM stack double-backwards through a matvec, and failing
    loudly beats a silently wrong second-order graph.
    """

    @staticmethod
    def forward(ctx, mass_coeff, stiffness_coeff, x, plan):
        ctx.save_for_backward(mass_coeff, stiffness_coeff, x)
        ctx.plan = plan  # geometry by reference, not in the autograd graph
        return _matvec_compute(plan, mass_coeff, stiffness_coeff, x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        mass_coeff, stiffness_coeff, x = ctx.saved_tensors
        plan: EdgeAssemblyPlan = ctx.plan
        need_m, need_k, need_x = ctx.needs_input_grad[:3]
        grad_m = grad_k = grad_x = None
        if need_m or need_k:
            nc = plan.n_cells
            # Recomputed gathers in the (c, e, f) layout of plan.rows/cols:
            # pair[c,e,f] = grad_out[E[c,e]] · conj(x[E[c,f]]).
            pair = (
                grad_out[plan.rows].view(nc, 6, 6)
                * x[plan.cols].conj().view(nc, 6, 6)
            )
            if need_m:
                grad_m = _match_leaf_dtype(
                    (plan.sign_outer * plan.mass_geom * pair).sum(dim=(1, 2)),
                    mass_coeff,
                )
            if need_k:
                grad_k = _match_leaf_dtype(
                    (plan.sign_outer * plan.stiffness_geom * pair).sum(dim=(1, 2)),
                    stiffness_coeff,
                )
        if need_x:
            grad_x = _match_leaf_dtype(
                _matvec_compute(
                    plan, mass_coeff.conj(), stiffness_coeff.conj(), grad_out,
                ),
                x,
            )
        return grad_m, grad_k, grad_x, None


def edge_operator_matvec(
    plan: EdgeAssemblyPlan,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Matrix-free ``A x`` for the same operator :func:`assemble_edge_operator`
    builds, the gather/``scatter_add`` pattern (no torch ``sparse.mm``, whose
    backward is unavailable without MKL), the seam future iterative solvers
    plug into.

    Runs as the :class:`_EdgeOperatorMatvec` custom Function: forward output
    is bit-identical to the plain tape (same kernel), but only the two
    ``(n_cells,)`` coefficients and ``x`` are saved for backward, O(n_cells +
    n_edges) per call instead of the straight-through tape's O(36·n_cells);
    which keeps the per-substep-per-source RHS matvecs of the TEM3D edge loop
    from growing the graph linearly in ``36·n_cells·n_steps``
    (:func:`_edge_operator_matvec_autograd` keeps the plain tape for
    cross-checks).

    Args:
        plan: the mesh's precomputed :class:`EdgeAssemblyPlan`.
        mass_coeff: ``(n_cells,)`` float64/complex128 mass-term coefficient.
        stiffness_coeff: ``(n_cells,)`` float64/complex128 curl-curl coefficient.
        x: ``(n_edges,)`` float64/complex128 edge vector.

    Returns:
        ``A x`` of shape ``(n_edges,)``; dtype is the torch promotion of the
        coefficients' and ``x``'s dtypes. Differentiable in the coefficients
        and ``x`` (first order; see :class:`_EdgeOperatorMatvec`).
    """
    _check_matvec_x(plan, x)
    _check_coeff(plan, mass_coeff, "mass_coeff")
    _check_coeff(plan, stiffness_coeff, "stiffness_coeff")
    return _EdgeOperatorMatvec.apply(mass_coeff, stiffness_coeff, x, plan)


def _edge_operator_matvec_autograd(
    plan: EdgeAssemblyPlan,
    mass_coeff: torch.Tensor,
    stiffness_coeff: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """:func:`edge_operator_matvec` through the straight autograd tape.

    Same validation, same kernel, same numbers, but every intermediate is
    recorded by autograd. The bit-level/gradient/memory reference the tests
    pin the custom Function against; NOT for production time loops.
    """
    _check_matvec_x(plan, x)
    return _matvec_compute(plan, mass_coeff, stiffness_coeff, x)
