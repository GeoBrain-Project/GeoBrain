"""
Private sparse adjoint solver implementation.

scipy splu forward (CPU only). Backward via ``torch.autograd.Function``:
``grad_b`` from an adjoint solve, ``grad_A_values`` from the standard
``-conj(λ_i) · x_j`` formula projected onto the supplied sparsity pattern.
Float64 + complex128 supported; lower precisions rejected.



DESIGN BOUNDARY: this is the FACTOR-REUSE direct family (assembled systems,
one factorization behind the core.linalg seam, handle shared by forward and
backward). The matrix-free / iterative differentiable solve lives in
:mod:`geobrain.core.adjoint.matrix_free`: see the boundary note there for why
the two are deliberately separate.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations


import torch
from torch.autograd.function import once_differentiable

from ..errors import GeoBrainError
from ..linalg import seam as _seam
from ..linalg.seam import (
    ScipySpluSolver as ScipySpluSolver,
    SparseFactorSolver,
    _effective_default,
    _is_cpu_only_solver,
    default_sparse_solver_is_cpu_only as default_sparse_solver_is_cpu_only,
    set_default_sparse_solver as set_default_sparse_solver,
    sparse_solver as sparse_solver,
)


def __getattr__(name: str):
    """Forward the seam's MUTABLE globals dynamically: a plain re-export
    would snapshot the import-time binding and go stale after
    ``set_default_sparse_solver`` rebinds it (pinned by
    test_seam_ownership.test_mutable_default_state_is_single_and_forwarded).

    READ-ONLY forward: assigning ``_sparse_impl._DEFAULT_SOLVER = ...`` would
    plant a real module attribute that SHADOWS this hook (PEP 562 has no
    module ``__setattr__``), mutate state only via
    ``set_default_sparse_solver`` / the ``sparse_solver`` scope."""
    if name in ("_DEFAULT_SOLVER", "_SCOPED_SOLVER"):
        return getattr(_seam, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_ALLOWED_TORCH_DTYPES = (torch.float64, torch.complex128)

# The effective sparse backend resolves in two layers:
# # * ``_DEFAULT_SOLVER``: the PROCESS-global default (CPU scipy SuperLU), set
#   once for the whole program by :func:`set_default_sparse_solver`. A plain
#   module global: a genuine process-wide setting visible from every thread.


def _check_inputs(
    A_sparse: torch.Tensor,
    b: torch.Tensor,
    solver: SparseFactorSolver | None = None,
) -> None:
    """Validate (A_sparse, b): raises GeoBrainError on any violation."""
    if not isinstance(A_sparse, torch.Tensor):
        raise GeoBrainError(
            "sparse_linear_solve_with_adjoint: A_sparse must be a torch sparse tensor",
            object_name="sparse_linear_solve_with_adjoint",
            field="A_sparse",
            expected="torch sparse CSR or COO tensor",
            actual=type(A_sparse).__name__,
        )
    if A_sparse.layout not in (torch.sparse_csr, torch.sparse_coo):
        raise GeoBrainError(
            f"A_sparse layout must be sparse CSR or COO, got {A_sparse.layout}",
            object_name="sparse_linear_solve_with_adjoint",
            field="A_sparse.layout",
            expected="torch.sparse_csr or torch.sparse_coo",
            actual=str(A_sparse.layout),
        )
    if not isinstance(b, torch.Tensor):
        raise GeoBrainError(
            "b must be a torch.Tensor",
            object_name="sparse_linear_solve_with_adjoint",
            field="b",
            expected="torch.Tensor",
            actual=type(b).__name__,
        )
    if A_sparse.dtype not in _ALLOWED_TORCH_DTYPES:
        raise GeoBrainError(
            f"A_sparse dtype must be float64 or complex128, got {A_sparse.dtype}",
            object_name="sparse_linear_solve_with_adjoint",
            field="A_sparse.dtype",
            expected="float64 or complex128",
            actual=A_sparse.dtype,
        )
    if b.dtype != A_sparse.dtype:
        raise GeoBrainError(
            f"A_sparse and b dtype must match (got A={A_sparse.dtype}, b={b.dtype})",
            object_name="sparse_linear_solve_with_adjoint",
            field="b.dtype",
            expected=A_sparse.dtype,
            actual=b.dtype,
        )
    # Device guard, relaxed for pluggable backends: only the CPU-only scipy
    # backend forces CPU. A non-scipy SparseFactorSolver (e.g. a CUDA CG
    # backend) is trusted to handle its own device, so a CUDA matrix can reach
    # it. CPU inputs are unaffected: behaviour is identical for the default
    # scipy path (CPU passes, non-CPU still rejected).
    if _is_cpu_only_solver(solver) and (
        A_sparse.device.type != "cpu" or b.device.type != "cpu"
    ):
        raise GeoBrainError(
            "sparse_linear_solve_with_adjoint is CPU-only with the default "
            "scipy splu backend. Either move sigma to CPU, or install a GPU "
            "backend: from geobrain.core.adjoint import PcgGpuSolver, "
            "set_default_sparse_solver; set_default_sparse_solver(PcgGpuSolver()). "
            "(PcgGpuSolver is Jacobi-CG for REAL SPD systems; DC/IP/SIP, and is "
            "opt-in because its iterative convergence semantics differ from a "
            "direct solve.)",
            object_name="sparse_linear_solve_with_adjoint",
            field="device",
            expected="cpu",
            actual=f"A={A_sparse.device}, b={b.device}",
        )
    n_rows, n_cols = A_sparse.shape
    if n_rows != n_cols:
        raise GeoBrainError(
            f"A_sparse must be square, got shape {(n_rows, n_cols)}",
            object_name="sparse_linear_solve_with_adjoint",
            field="A_sparse.shape",
            expected="(n, n)",
            actual=(n_rows, n_cols),
        )
    if b.shape[0] != n_rows:
        raise GeoBrainError(
            f"b leading dim must match A_sparse rows (got b={tuple(b.shape)}, A rows={n_rows})",
            object_name="sparse_linear_solve_with_adjoint",
            field="b.shape",
            expected=f"({n_rows},) or ({n_rows}, k)",
            actual=tuple(b.shape),
        )


def _unpack_torch_sparse(A_sparse: torch.Tensor):
    """Return (indices_packed, values, shape) for CSR or COO input.

    ``indices_packed`` has shape ``(2, nnz)`` holding ``(row, col)`` pairs.
    For CSR inputs the row indices are expanded from ``crow_indices`` via
    ``repeat_interleave``; for COO inputs we coalesce first so duplicate
    entries are summed once before the autograd ``Function.apply`` boundary.
    """
    if A_sparse.layout == torch.sparse_csr:
        crow = A_sparse.crow_indices().to(torch.long)
        col = A_sparse.col_indices().to(torch.long)
        counts = (crow[1:] - crow[:-1]).to(torch.long)
        n = int(crow.numel() - 1)
        # Build the row-index arange on the SAME device as the CSR crow/col
        # indices so a CUDA CSR (e.g. a GPU-assembled DC3D operator) unpacks
        # without a cross-device index_select. The bridge is device-agnostic;
        # the backend decides where the solve runs.
        rows = torch.repeat_interleave(
            torch.arange(n, dtype=torch.long, device=crow.device), counts
        )
        indices = torch.stack([rows, col])
        return indices, A_sparse.values(), tuple(A_sparse.shape)
    if A_sparse.layout == torch.sparse_coo:
        coo = A_sparse.coalesce()
        return coo.indices().to(torch.long), coo.values(), tuple(coo.shape)
    raise GeoBrainError(
        f"Unsupported sparse layout: {A_sparse.layout}",
        object_name="sparse_linear_solve_with_adjoint",
        field="A_sparse.layout",
        expected="torch.sparse_csr or torch.sparse_coo",
        actual=str(A_sparse.layout),
    )


class _SparseLinearSolveFn(torch.autograd.Function):
    """Internal: scipy splu forward + adjoint-state backward.

    Forward factors ``A`` (built from the supplied ``(row, col, values)``
    triplets) with ``scipy.sparse.linalg.splu`` and solves ``A x = b``.
    Backward reuses the factor: one adjoint solve gives ``λ`` from which
    ``grad_b = λ`` and ``grad_A_values = -conj(λ_i) · x_j`` (real case
    drops the conjugate). Only the values gradient is reduced across RHS
    columns; ``grad_b`` keeps the original ``(n,)`` or ``(n, k)`` layout.

    Memory: the LU factor (holding the scipy ``splu`` handle) is stashed on
    ``ctx`` so backward can reuse it for the adjoint solve; it therefore
    stays alive as long as this autograd node is retained. A long-lived graph
    that runs many solves before a single ``.backward()`` (or one built with
    ``retain_graph=True``) thus holds every solve's factor simultaneously.
    That is the intended trade (reuse the forward factorization for the
    adjoint); free it by letting the graph be collected after backward, or use
    the ``factor=`` reuse path to share one factorization across solves.
    """

    @staticmethod
    def forward(ctx, A_indices_packed, A_values, A_shape, b, factor):
        """Solve ``A x = b`` against a pre-built :class:`SparseFactor`.

        ``A_indices_packed``: ``(2, nnz)`` long tensor of ``(row, col)`` pairs.
        ``A_values``:         ``(nnz,)`` values tensor (may require grad).
        ``A_shape``:          ``(n, n)`` Python tuple (autograd skips it).
        ``b``:                dense RHS, shape ``(n,)`` or ``(n, k)``.
        ``factor``:           a :class:`SparseFactor` (solver + handle) built from
                              this same ``A``, possibly REUSED across equal-matrix
                              solves (e.g. repeated-``dt`` substeps at fixed ``σ``).
                              The σ-gradient is still computed per call from this
                              node's ``A_values``, so reuse shares only the numeric
                              factorization, never the graph. Backend (scipy / GPU)
                              is whatever the factor's solver provides.
        """
        x = factor.solve(b, adjoint=False)
        ctx.save_for_backward(A_indices_packed, x)
        ctx.factor = factor
        ctx.is_complex = torch.is_complex(b)
        return x

    @staticmethod
    @once_differentiable  # first-order only: the factor is opaque to autograd
    def backward(ctx, grad_x):
        A_indices_packed, x = ctx.saved_tensors

        # Adjoint solve  Aᴴ λ = grad_x: delegated to the factor's solver
        # ('H'/'T' handled inside, so the bridge stays backend-agnostic).
        lam = ctx.factor.solve(grad_x, adjoint=True)

        # grad_b = λ: DO NOT sum across RHS columns; keep the b layout.
        grad_b = lam

        rows = A_indices_packed[0]
        cols = A_indices_packed[1]
        x_at_J = x[cols]
        lam_at_I = lam[rows]

        if ctx.is_complex:
            # PyTorch Wirtinger convention: conjugate on x, not λ.
            # Matches torch.linalg.solve's autograd backward for the dense
            # equivalent, verified by direct cross-check.
            contrib = -lam_at_I * x_at_J.conj()
        else:
            contrib = -lam_at_I * x_at_J

        # 2D RHS: sum the per-nnz contributions across the k RHS columns.
        # (Must sum the product, not the factors separately.)
        if contrib.ndim == 2:
            grad_A_values = contrib.sum(dim=-1)
        else:
            grad_A_values = contrib

        # Signature: (A_indices_packed, A_values, A_shape, b, lu)
        return None, grad_A_values, None, grad_b, None


def _values_fingerprint(
    values: torch.Tensor,
) -> tuple[int, float, float, float, float]:
    """Cheap O(nnz), allocation-light moment fingerprint of a values vector.

    ``(nnz, Σ Re(v), Σ Im(v), max |v|, Σ i·Re(v))``. The imaginary-part sum
    catches imaginary sign flips (``|3 - i| == |3 + i|``); the **position-weighted**
    sum ``Σ i·Re(v)`` makes the fingerprint ORDER-sensitive, so a permutation of
    the same values, e.g. ``diag(2,3,4,5)`` vs ``diag(5,4,3,2)``, which share the
    same ``nnz``/sum/max, no longer collides.

    This is a **best-effort tripwire**, not a cryptographic hash: a handful of
    torch-reduced moments (no host copy, so it stays cheap on the per-substep
    reuse-check hot path). The binding contract is that a factor is reused only
    for the *same numeric matrix*, legitimate reuse re-presents bit-identical
    values (the same tensor, or a deterministic re-assembly at the same σ/dt), so
    every moment matches exactly. The moments make an accidental stale-factor
    reuse very likely to be caught, but a deliberately crafted collision is not
    ruled out; callers must not rely on the fingerprint to license reusing a
    factor across genuinely different matrices.
    """
    v = values.detach()
    nnz = int(v.numel())
    if nnz == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    re = v.real
    re_sum = float(re.sum())
    im_sum = float(v.imag.sum()) if v.is_complex() else 0.0
    max_abs = float(torch.linalg.vector_norm(v, ord=float("inf")))
    # Order-sensitive moment: dot with [1..nnz] (torch reduction, no host copy).
    pos = torch.arange(1, nnz + 1, device=re.device, dtype=re.dtype)
    pos_re = float((pos * re).sum())
    return (nnz, re_sum, im_sum, max_abs, pos_re)


class SparseFactor:
    """A reusable factorization of a fixed sparse matrix, produced by a
    :class:`~geobrain.core.adjoint.SparseFactorSolver`.

    Build one with :func:`factorize_sparse` and pass it to
    :func:`sparse_solve_with_adjoint` (``factor=``) to skip re-factorization
    across solves that share the **same numeric matrix**, e.g. the many
    repeated-``dt`` backward-Euler substeps in a TEM time loop (a log substep
    schedule clamped by ``dt_max`` spends most of its steps at one ``dt``). The
    handle is valid only while ``A``'s values are unchanged; rebuild it when the
    matrix (σ, dt, …) changes, :func:`sparse_solve_with_adjoint` verifies this
    via ``values_fingerprint`` and rejects a stale factor.
    """

    __slots__ = ("_solver", "_handle", "shape", "values_fingerprint")

    def __init__(
        self,
        solver: SparseFactorSolver,
        handle,
        shape: tuple[int, int],
        values_fingerprint: tuple[int, float, float, float, float] | None = None,
    ) -> None:
        self._solver = solver
        self._handle = handle
        self.shape = shape
        # Fingerprint of the factored matrix's values (see
        # ``_values_fingerprint``); ``None`` disables the staleness check for
        # externally constructed factors.
        self.values_fingerprint = values_fingerprint

    def solve(self, b: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        """Solve ``A x = b`` (``adjoint=False``) or ``Aᴴ x = b`` against this factor."""
        return self._solver.solve(self._handle, b, adjoint=adjoint)


def factorize_sparse(
    A_sparse: torch.Tensor,
    *,
    solver: SparseFactorSolver | None = None,
) -> SparseFactor:
    """Factor ``A_sparse`` into a reusable :class:`SparseFactor`.

    Uses the supplied ``solver`` (default :class:`ScipySpluSolver`, CPU scipy
    SuperLU, float64/complex128). The factorization uses the DETACHED values;
    gradients still flow per-solve through the ``A`` passed to
    :func:`sparse_solve_with_adjoint`.
    """
    if A_sparse.layout not in (torch.sparse_csr, torch.sparse_coo):
        raise GeoBrainError(
            "factorize_sparse: A_sparse must be sparse CSR or COO",
            object_name="factorize_sparse", field="A_sparse.layout",
            expected="torch.sparse_csr or torch.sparse_coo",
            actual=str(A_sparse.layout),
        )
    if A_sparse.dtype not in _ALLOWED_TORCH_DTYPES:
        raise GeoBrainError(
            "factorize_sparse: dtype must be float64 or complex128",
            object_name="factorize_sparse", field="A_sparse.dtype",
            expected="float64 or complex128", actual=A_sparse.dtype,
        )
    solver = solver if solver is not None else _effective_default()
    # Device guard, relaxed for pluggable backends (see ``_check_inputs``): the
    # CPU-only scipy backend forces CPU; any other backend handles its own
    # device. CPU behaviour is unchanged for the default scipy path.
    if _is_cpu_only_solver(solver) and A_sparse.device.type != "cpu":
        raise GeoBrainError(
            "factorize_sparse is CPU-only with the default scipy splu backend. "
            "Either move sigma to CPU, or install a GPU backend: from "
            "geobrain.core.adjoint import PcgGpuSolver, set_default_sparse_solver; "
            "set_default_sparse_solver(PcgGpuSolver()).",
            object_name="factorize_sparse", field="device",
            expected="cpu", actual=str(A_sparse.device),
        )
    indices, values, shape = _unpack_torch_sparse(A_sparse)
    return SparseFactor(
        solver,
        solver.factor(indices, values, shape),
        shape,
        values_fingerprint=_values_fingerprint(values),
    )


def sparse_solve_with_adjoint(
    A_sparse: torch.Tensor,
    b: torch.Tensor,
    *,
    factor: SparseFactor | None = None,
    solver: SparseFactorSolver | None = None,
) -> torch.Tensor:
    """Public entrypoint: autograd-attached sparse solve.

    ``factor``: an optional pre-built :class:`SparseFactor` to reuse instead of
    re-factorizing ``A`` (must factor the same numeric ``A``).
    ``solver``: an optional :class:`SparseFactorSolver` backend used when no
    ``factor`` is supplied (default :class:`ScipySpluSolver`).
    """
    # Resolve the effective backend so the device guard can trust a non-scipy
    # solver: a supplied factor's own solver, else an explicit ``solver=``, else
    # the process-global default.
    effective_solver = (
        factor._solver if factor is not None
        else (solver if solver is not None else _effective_default())
    )
    _check_inputs(A_sparse, b, effective_solver)
    indices, values, shape = _unpack_torch_sparse(A_sparse)
    if factor is None:
        factor = factorize_sparse(A_sparse, solver=solver)
    else:
        if tuple(factor.shape) != tuple(shape):
            raise GeoBrainError(
                "sparse_solve_with_adjoint: factor shape mismatch",
                object_name="sparse_solve_with_adjoint", field="factor.shape",
                expected=tuple(shape), actual=tuple(factor.shape),
            )
        # Same shape is not enough: a factor of yesterday's values silently
        # returns the WRONG solution for today's matrix. Re-fingerprint the
        # incoming values (O(nnz), no copies) and reject a stale factor.
        if factor.values_fingerprint is not None:
            incoming = _values_fingerprint(values)
            if incoming != factor.values_fingerprint:
                raise GeoBrainError(
                    "sparse_solve_with_adjoint: stale factor: matrix values "
                    "changed since factorization, rebuild it with "
                    "factorize_sparse(A)",
                    object_name="sparse_solve_with_adjoint",
                    field="factor.values_fingerprint",
                    expected=factor.values_fingerprint,
                    actual=incoming,
                )
    return _SparseLinearSolveFn.apply(indices, values, shape, b, factor)
