"""
Sparse Jacobian assembly via colored finite differences.

Two pieces:

1. :class:`JacobianSparsitySpec`, sparsity pattern + greedy distance-1
   column coloring. Built once per model (TPFA stencil + any extra
   well couplings) and reused across Newton iterations.

2. :func:`compute_sparse_jacobian`, central FD on residual evaluations
   batched per color. Returns a ``torch.sparse_coo_tensor``.

Pure torch (no scipy.sparse). The sparsity bookkeeping is in Python /
numpy because it is build-once and the pattern is tiny compared to the
residual workload it amortises across many Newton steps.

DOF ordering convention: **per-variable** blocks::

    [var0_cell0, var0_cell1, ..., var0_cellN, var1_cell0, ...]

Block-structured preconditioners that need per-cell ordering permute
internally.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch

from ....core import GeoBrainError
from ..errors import FlowCapabilityError


# Auto-batched FD threshold: below this dof count the sequential per-color loop
# is faster than torch.vmap's per-op batching-rule overhead (measured crossover
# between 4k dof: loop 0.13 s vs vmap ~10 s, and 100k dof, loop 45.3 s vs
# vmap 3.0 s on the 2-phase TPFA residual).
_BATCHED_FD_MIN_DOF = 20_000

_ColorProbe = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_RowProbe = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_BatchedProbe = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]


@dataclass
class JacobianSparsitySpec:
    """
    Sparsity pattern + column coloring.

    Attributes:
        n_dof:    total degrees of freedom ``n_cells * n_vars``.
        rows:     ``(nnz,) int64`` row indices of nonzeros.
        cols:     ``(nnz,) int64`` column indices.
        colors:   ``(n_dof,) int32`` greedy coloring; columns of the
            same color share no row, so they can be perturbed
            simultaneously in one FD residual evaluation.
        n_colors: number of distinct colors used.
    """

    n_dof: int
    rows: np.ndarray
    cols: np.ndarray
    colors: np.ndarray
    n_colors: int
    # Per-device vectorized probe/scatter index tensors (built lazily once,
    # see _color_probe_indices). Not part of the value identity of the spec.
    _probe_cache: dict[object, object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )


def _color_probe_indices(
    spec: "JacobianSparsitySpec", device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-color ``(probe_cols, entry_flat, entry_rows, entry_cols)`` long tensors.

    ``probe_cols`` are the columns perturbed together; the ``entry_*`` arrays
    address every nonzero owned by those columns, in the spec's storage order,
    so the assembled ``values`` layout is identical to the historical
    per-entry python scatter. Built once per (spec, device) and cached.
    """
    key = str(device)
    cached = spec._probe_cache.get(key)
    if cached is not None:
        return cast(list[_ColorProbe], cached)
    colors_of_entries = spec.colors[spec.cols]  # color of each nonzero's column
    per_color = []
    for k in range(spec.n_colors):
        cols_in_color = np.nonzero(spec.colors == k)[0]
        if cols_in_color.size == 0:
            continue
        mask = colors_of_entries == k
        entry_flat = np.nonzero(mask)[0]
        per_color.append(
            (
                torch.from_numpy(cols_in_color.astype(np.int64)).to(device),
                torch.from_numpy(entry_flat.astype(np.int64)).to(device),
                torch.from_numpy(spec.rows[mask].astype(np.int64)).to(device),
                torch.from_numpy(spec.cols[mask].astype(np.int64)).to(device),
            )
        )
    spec._probe_cache[key] = per_color
    return per_color


def _row_probe_indices(
    spec: "JacobianSparsitySpec", device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-row-color ``(probe_rows, entry_flat, entry_cols)`` long tensors.

    The mirror image of :func:`_color_probe_indices`, for the reverse-mode
    assembly: rows of one color share no column, so a single VJP seeded with
    ``λ = Σ e_i`` over those rows reads every nonzero they own, entry
    ``(i, j)`` lands at position ``j`` of ``Jᵀλ``, uncontested.

    The row coloring is derived here from the transposed pattern rather than
    reused from :attr:`JacobianSparsitySpec.colors`: the two coincide only
    when the pattern is structurally symmetric, which
    :func:`compute_sparsity_pattern` happens to guarantee but a
    caller-supplied spec need not. Built once per (spec, device) and cached.
    """
    key = ("rows", str(device))
    cached = spec._probe_cache.get(key)
    if cached is not None:
        return cast(list[_RowProbe], cached)
    # compute_coloring colors its second argument's index space, so passing
    # the pattern transposed colors the ROWS.
    row_colors, n_row_colors = compute_coloring(spec.cols, spec.rows, spec.n_dof)
    colors_of_entries = row_colors[spec.rows]  # color of each nonzero's row
    per_color: list[_RowProbe] = []
    for k in range(n_row_colors):
        rows_in_color = np.nonzero(row_colors == k)[0]
        if rows_in_color.size == 0:
            continue
        mask = colors_of_entries == k
        per_color.append(
            (
                torch.from_numpy(rows_in_color.astype(np.int64)).to(device),
                torch.from_numpy(np.nonzero(mask)[0].astype(np.int64)).to(device),
                torch.from_numpy(spec.cols[mask].astype(np.int64)).to(device),
            )
        )
    spec._probe_cache[key] = per_color
    return per_color


def _batched_probe_indices(
    spec: "JacobianSparsitySpec", device: torch.device
) -> _BatchedProbe:
    """Indices for the fully-batched FD path (built once per (spec, device)).

    Returns ``(probe_row_of_col, probe_cols, entry_flat, entry_rows, entry_cols,
    entry_probe_row, n_probes)`` where ``probe_row_of_col`` maps each perturbed
    column to its row in the (n_probes, n_dof) probe matrix, and ``entry_probe_row``
    maps every nonzero to the probe row of its column."""
    key = ("batched", str(device))
    cached = spec._probe_cache.get(key)
    if cached is not None:
        return cast(_BatchedProbe, cached)
    nonempty = [k for k in range(spec.n_colors) if np.any(spec.colors == k)]
    color_to_probe = {k: i for i, k in enumerate(nonempty)}
    probe_of_col = np.array([color_to_probe[int(c)] for c in spec.colors], dtype=np.int64)
    all_cols: np.ndarray = np.arange(spec.n_dof, dtype=np.int64)
    entry_probe_row = probe_of_col[spec.cols]
    out = (
        torch.from_numpy(probe_of_col).to(device),
        torch.from_numpy(all_cols).to(device),
        torch.from_numpy(np.arange(spec.rows.size, dtype=np.int64)).to(device),
        torch.from_numpy(spec.rows.astype(np.int64)).to(device),
        torch.from_numpy(spec.cols.astype(np.int64)).to(device),
        torch.from_numpy(entry_probe_row.astype(np.int64)).to(device),
        len(nonempty),
    )
    spec._probe_cache[key] = out
    return out


def _fd_batched_values(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    spec: JacobianSparsitySpec,
    h: torch.Tensor,
) -> torch.Tensor:
    """All-colors-at-once FD assembly via ``torch.vmap`` (bitwise-identical
    per-entry arithmetic to the per-color loop; verified by the oracle test)."""
    device, dtype = state.device, state.dtype
    (probe_of_col, all_cols, entry_flat, entry_rows, entry_cols, entry_probe_row, n_probes) = (
        _batched_probe_indices(spec, device)
    )
    H = torch.zeros(n_probes, spec.n_dof, device=device, dtype=dtype)
    H[probe_of_col, all_cols] = h
    r_plus = torch.vmap(residual_fn)(state.unsqueeze(0) + H).detach()
    r_minus = torch.vmap(residual_fn)(state.unsqueeze(0) - H).detach()
    values = torch.zeros(int(spec.rows.size), device=device, dtype=dtype)
    values[entry_flat] = (
        r_plus[entry_probe_row, entry_rows] - r_minus[entry_probe_row, entry_rows]
    ) / (2.0 * h[entry_cols])
    return values


def compute_sparsity_pattern(
    cell_neighbors: torch.Tensor,
    n_cells: int,
    n_vars: int,
    extra_couplings: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ``(rows, cols)`` of Jacobian nonzeros for a TPFA-stencil
    multi-variable per-cell residual.

    Args:
        cell_neighbors: ``(n_faces, 2) int64`` cell pairs from
            :attr:`FlowGrid.conn.neighbors`.
        n_cells: cell count.
        n_vars:  state-variables per cell. DOF ordering is per-variable
            (see module docstring).
        extra_couplings: additional ``(cell_a, cell_b)`` pairs (e.g.
            multi-perforation wells coupling otherwise disjoint cells).
    """
    if cell_neighbors.device.type != "cpu":
        raise FlowCapabilityError(
            "Jacobian sparsity construction requires explicit CPU topology metadata",
            object_name="compute_sparsity_pattern",
            field="cell_neighbors.device",
            expected="CPU topology metadata prepared before sparse assembly",
            actual=str(cell_neighbors.device),
        )
    if cell_neighbors.layout != torch.strided:
        raise FlowCapabilityError(
            "Jacobian sparsity construction requires strided topology metadata",
            object_name="compute_sparsity_pattern",
            field="cell_neighbors.layout",
            expected=str(torch.strided),
            actual=str(cell_neighbors.layout),
        )
    # The pattern/coloring is immutable, build-once CPU metadata.  Calling
    # ``.cpu()`` here used to hide a synchronous CUDA-to-host transfer; callers
    # now declare that boundary explicitly and this conversion is zero-copy.
    nbrs = cell_neighbors.detach().numpy()
    couplings: set[tuple[int, int]] = set()
    for c in range(n_cells):
        couplings.add((c, c))
    for cl, cr in nbrs:
        if cr >= 0:
            couplings.add((int(cl), int(cr)))
            couplings.add((int(cr), int(cl)))
    if extra_couplings:
        for a, b in extra_couplings:
            couplings.add((int(a), int(b)))
            couplings.add((int(b), int(a)))

    cell_pairs = np.array(sorted(couplings), dtype=np.int64)
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for vr in range(n_vars):
        for vc in range(n_vars):
            rows_list.append(vr * n_cells + cell_pairs[:, 0])
            cols_list.append(vc * n_cells + cell_pairs[:, 1])
    return np.concatenate(rows_list), np.concatenate(cols_list)


def compute_coloring(
    rows: np.ndarray,
    cols: np.ndarray,
    n_dof: int,
) -> tuple[np.ndarray, int]:
    """
    Greedy distance-1 column coloring.

    Two columns share a color iff no row touches both. Returns
    ``(colors[n_dof], n_colors)``.
    """
    col_to_rows: list[set[int]] = [set() for _ in range(n_dof)]
    row_to_cols: list[list[int]] = [[] for _ in range(n_dof)]
    for r, c in zip(rows, cols):
        col_to_rows[c].add(int(r))
        row_to_cols[r].append(int(c))

    colors: np.ndarray = np.full(n_dof, -1, dtype=np.int32)
    for c in range(n_dof):
        forbidden: set[int] = set()
        for r in col_to_rows[c]:
            for c2 in row_to_cols[r]:
                if c2 != c and colors[c2] >= 0:
                    forbidden.add(int(colors[c2]))
        k = 0
        while k in forbidden:
            k += 1
        colors[c] = k
    return colors, int(colors.max() + 1)


def make_sparsity_spec(
    cell_neighbors: torch.Tensor,
    n_cells: int,
    n_vars: int,
    extra_couplings: list[tuple[int, int]] | None = None,
) -> JacobianSparsitySpec:
    """Convenience: pattern + greedy coloring in one call."""
    rows, cols = compute_sparsity_pattern(
        cell_neighbors,
        n_cells,
        n_vars,
        extra_couplings,
    )
    n_dof = n_cells * n_vars
    colors, n_colors = compute_coloring(rows, cols, n_dof)
    return JacobianSparsitySpec(
        n_dof=n_dof,
        rows=rows,
        cols=cols,
        colors=colors,
        n_colors=n_colors,
    )


def compute_sparse_jacobian(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    spec: JacobianSparsitySpec,
    eps_rel: float = 1e-7,
    eps_abs: float = 1e-8,
    *,
    mode: str = "fd",
    batched: bool | None = None,
) -> torch.Tensor:
    """
    Assemble ``J = ∂r/∂x`` batched per structurally-orthogonal color.

    Two probe modes, both costing one probe per color (``n_colors`` ≈ the
    stencil bandwidth, independent of ``n_cells``):

    - ``mode="fd"`` (default): central finite differences::

          e_k    = sum of unit vectors of columns colored k
          r_±    = residual(x ± h_k · e_k)
          J[i,j] = (r_+[i] - r_-[i]) / (2 · h[j])  for each j in color k

      Cheap (no autograd graph) and accurate to O(h²). Used for the
      forward Newton iterations, where an approximate Jacobian only
      affects the step path, not the converged root.

    - ``mode="ad"``: exact, via one reverse-mode VJP per ROW color::

          λ_k    = sum of unit vectors of rows colored k
          J[i,j] = (Jᵀ λ_k)[j]  for the unique row i of color k owning
                   column j (distance-1 coloring ⇒ rows of a color share
                   no column)

      One probe per row color, **exact** (no step size), over a residual
      graph built once. Used for the implicit-FT adjoint cleanup so
      parameter gradients are exact even when the forward solve uses the
      cheaper FD Jacobian. Reverse mode rather than a JVP because a
      well-coupled residual is not forward-differentiable; see the
      comment on the branch.

    Returns a coalesced ``torch.sparse_coo_tensor`` on the same device and
    dtype as ``state``. The Jacobian is **detached**, Newton uses it for
    forward solves; the adjoint helper handles gradient flow through the
    converged state separately (the exact ``J`` only feeds the adjoint
    linear operator, not an autograd graph).
    """
    n_dof = spec.n_dof
    device, dtype = state.device, state.dtype

    values = torch.zeros(int(spec.rows.size), device=device, dtype=dtype)

    if mode == "ad":
        # Exact Jacobian by REVERSE mode: build the residual graph once, then
        # one VJP per row color. Rows of a color share no column, so
        # ``(Jᵀλ)[j] == J[i, j]`` for the unique row i of that color owning
        # column j: the transpose of the FD path's column probing.
        #
        # Forward mode is not usable here, and its failure is silent, which is
        # why this is spelled out: a well-coupled residual assembles its source
        # terms as a sparse COO block, ``torch.func.jvp`` raises on it outright
        # (no forward-AD rule for _sparse_coo_tensor_with_dims_and_tensors), and
        # ``torch.autograd.functional.jvp``, which emulates forward mode with a
        # double backward: returns WRONG numbers for it instead of raising.
        # Measured on a 2-well oil-water step: every well-free entry exact to
        # 1e-16, the four well-cell entries off by tens of percent, which fed a
        # correct forward solve and a badly wrong implicit-FT parameter
        # gradient. Reverse mode differentiates the same block exactly.
        # ``enable_grad`` because the forward Newton loop calls this under
        # ``torch.no_grad``.
        with torch.enable_grad():
            x0 = state.detach().requires_grad_(True)
            residual = residual_fn(x0)
            # A residual that does not depend differentiably on the state has a
            # zero Jacobian; leave the prepared zeros rather than raising.
            if residual.requires_grad:
                row_probe = _row_probe_indices(spec, device)
                last = len(row_probe) - 1
                for position, (probe_rows, entry_flat, entry_cols) in enumerate(row_probe):
                    lam = torch.zeros(n_dof, device=device, dtype=dtype)
                    lam.index_fill_(0, probe_rows, 1.0)
                    (row_block,) = torch.autograd.grad(
                        residual,
                        x0,
                        grad_outputs=lam,
                        retain_graph=position < last,
                        allow_unused=True,
                    )
                    if row_block is not None:
                        values[entry_flat] = row_block.detach()[entry_cols]
    elif mode == "fd":
        h = eps_rel * state.detach().abs() + eps_abs
        # Size-gated batching: ONE torch.vmap call evaluates every color's ± probe
        # simultaneously. vmap pays a per-op batching-rule overhead that DOMINATES
        # at small problems (measured: ~10 s/Jacobian at 4k dof vs 0.13 s looped)
        # but amortizes into a big win at scale (3.0 s vs 45.3 s at 100k dof), so
        # auto mode batches only above _BATCHED_FD_MIN_DOF. Falls back to the
        # sequential per-color loop if the residual is not vmap-compatible
        # (data-dependent python control flow, .item(), ...).
        use_batched = batched if batched is not None else n_dof >= _BATCHED_FD_MIN_DOF
        done = False
        if use_batched:
            try:
                values = _fd_batched_values(residual_fn, state.detach(), spec, h)
                done = True
            except Exception as exc:  # noqa: BLE001, deliberate fallback
                if batched is True:
                    raise
                import warnings as _warnings

                _warnings.warn(
                    f"batched FD Jacobian unavailable ({type(exc).__name__}: "
                    f"{exc}); using the sequential per-color loop",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not done:
            # Vectorized per-color probe/scatter indices, built ONCE per
            # (spec, device) and cached. The original implementation rebuilt an
            # nnz-sized python dict plus per-nonzero python scalar writes on
            # EVERY call: measured at >90% of the whole flow forward (M: 31 s
            # of 47 s). The gathers below produce the numerically IDENTICAL
            # entries (same elementwise arithmetic per nonzero).
            for probe_cols, entry_flat, entry_rows, entry_cols in _color_probe_indices(
                spec, device
            ):
                h_color = torch.zeros(n_dof, device=device, dtype=dtype)
                h_color.index_copy_(0, probe_cols, h[probe_cols])
                r_plus = residual_fn(state + h_color).detach()
                r_minus = residual_fn(state - h_color).detach()
                values[entry_flat] = (r_plus[entry_rows] - r_minus[entry_rows]) / (
                    2.0 * h[entry_cols]
                )
    else:
        raise GeoBrainError(f"compute_sparse_jacobian: mode must be 'fd' or 'ad', got {mode!r}")

    rows_t = torch.from_numpy(spec.rows).to(device=device)
    cols_t = torch.from_numpy(spec.cols).to(device=device)
    indices = torch.stack([rows_t, cols_t], dim=0)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_dof, n_dof),
        device=device,
        dtype=dtype,
    ).coalesce()


__all__ = [
    "JacobianSparsitySpec",
    "compute_coloring",
    "compute_sparse_jacobian",
    "compute_sparsity_pattern",
    "make_sparsity_spec",
]
