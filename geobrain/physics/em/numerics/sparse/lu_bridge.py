# pyright: reportPrivateImportUsage=false
"""
scipy ``splu`` + closed-form-σ-Jacobian autograd bridge for EM solves.

the :func:`geobrain.core.sparse_linear_solve_with_adjoint` (E5)
already covers the canonical ``A x = b`` adjoint with autograd through
``A.values()`` and ``b``. It is the right tool when the caller is happy
to let the bridge rediscover the σ-dependence by graph-walking through
``A``'s torch values.

Some EM operators have a **closed-form** ``dA/dσ`` Jacobian, the matrix
entries depend on σ via a small, hand-derived formula, and want to skip
the per-A-entry autograd path entirely. That is what this module ports
from the reference implementation:

- :class:`SparseLinearSolveWithSigmaGrad`: ``dA_dsigma_vals`` is a plain
  numpy / complex array. Autograd through σ; no autograd through the
  Jacobian values themselves.

- :class:`SparseLinearSolveWithSigmaTorchJacGrad`: ``dA_dsigma_vals``
  is a torch tensor. Autograd through both σ and the Jacobian values,
  used by the SIP path where ``dA/dσ`` carries an ``η_eff(ω)`` factor.

The Wirtinger convention matches the adjoint helper: the
conjugate sits on ``x`` (not on ``λ``) when ``A`` is complex with real σ.
This is the convention PyTorch's complex autograd expects, and is
verified bit-identically against
:func:`sparse_linear_solve_with_adjoint` by the LU-bridge suite.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from geobrain.core.errors import GeoBrainError

__all__ = [
    "SparseLinearSolveWithSigmaGrad",
    "SparseLinearSolveWithSigmaTorchJacGrad",
    "sparse_solve_with_sigma",
    "sparse_solve_with_sigma_torch_jac",
]


_ALLOWED_TORCH_DTYPES = (torch.float64, torch.complex128)
_ALLOWED_NUMPY_DTYPES = (np.float64, np.complex128)


def _checked_splu(
    A_csr_data, A_csr_indices, A_csr_indptr, A_shape, *, object_name: str
):
    """``splu`` with guard rails: finite-values check + structured failure.

    A NaN/Inf in the assembled matrix (almost always a NaN/Inf σ upstream)
    must not surface as a misleading 'singular matrix' RuntimeError, and a
    genuinely singular system should name its usual root causes instead of
    scipy's bare message.
    """
    if not np.isfinite(A_csr_data).all():
        raise GeoBrainError(
            "sparse matrix values are not finite (NaN/Inf); check the "
            "property model (e.g. σ) used to assemble the system",
            object_name=object_name,
            field="A_csr.data",
            expected="all entries finite",
            actual="NaN/Inf present",
        )
    A_csr = sp.csr_matrix((A_csr_data, A_csr_indices, A_csr_indptr), shape=A_shape)
    try:
        return spla.splu(A_csr.tocsc())
    except RuntimeError as exc:
        raise GeoBrainError(
            f"sparse LU factorization failed ({exc}); common causes: "
            "non-positive σ entries, a missing Dirichlet pin, or a "
            "pure-Neumann (singular) system",
            object_name=object_name,
            field="A_csr",
            expected="non-singular sparse matrix",
            actual=str(exc),
        ) from exc


def _check_inputs(A_csr: sp.csr_matrix, b_torch: torch.Tensor) -> None:
    """
    Validate the matrix/RHS pair.

    Accepts ``(float64, float64)`` for the DC path and
    ``(complex128, complex128)`` for the FDEM/MT path. Mixed dtypes
    and lower precisions are rejected.
    """
    if not sp.issparse(A_csr):
        raise GeoBrainError(
            "A_csr must be a scipy sparse matrix",
            object_name="sparse_solve_with_sigma",
            field="A_csr",
            expected="scipy.sparse.csr_matrix",
            actual=type(A_csr).__name__,
        )
    if not isinstance(b_torch, torch.Tensor):
        raise GeoBrainError(
            "b_torch must be a torch.Tensor",
            object_name="sparse_solve_with_sigma",
            field="b_torch",
            expected="torch.Tensor",
            actual=type(b_torch).__name__,
        )
    if b_torch.dtype not in _ALLOWED_TORCH_DTYPES:
        raise GeoBrainError(
            "b_torch must be float64 or complex128",
            object_name="sparse_solve_with_sigma",
            field="b_torch.dtype",
            expected="float64 or complex128",
            actual=b_torch.dtype,
        )
    if b_torch.device.type != "cpu":
        raise GeoBrainError(
            "sparse_solve_with_sigma is CPU-only (scipy.sparse.linalg.splu)",
            object_name="sparse_solve_with_sigma",
            field="b_torch.device",
            expected="cpu",
            actual=str(b_torch.device),
        )
    if b_torch.dim() not in (1, 2):
        raise GeoBrainError(
            "b_torch must be 1D (n,) or 2D (n, k)",
            object_name="sparse_solve_with_sigma",
            field="b_torch.shape",
            expected="(n,) or (n, k)",
            actual=tuple(b_torch.shape),
        )
    if A_csr.dtype not in _ALLOWED_NUMPY_DTYPES:
        raise GeoBrainError(
            "A_csr must be float64 or complex128",
            object_name="sparse_solve_with_sigma",
            field="A_csr.dtype",
            expected="float64 or complex128",
            actual=str(A_csr.dtype),
        )
    a_is_complex = np.issubdtype(A_csr.dtype, np.complexfloating)
    b_is_complex = b_torch.is_complex()
    if a_is_complex != b_is_complex:
        raise GeoBrainError(
            "A_csr and b_torch must match in real/complex",
            object_name="sparse_solve_with_sigma",
            field="dtype",
            expected="float64+float64 or complex128+complex128",
            actual=f"A={A_csr.dtype}, b={b_torch.dtype}",
        )
    n = A_csr.shape[0]
    if A_csr.shape[1] != n:
        raise GeoBrainError(
            "A_csr must be square",
            object_name="sparse_solve_with_sigma",
            field="A_csr.shape",
            expected="(n, n)",
            actual=A_csr.shape,
        )
    if b_torch.shape[0] != n:
        raise GeoBrainError(
            "b_torch first dim must match A_csr.shape[0]",
            object_name="sparse_solve_with_sigma",
            field="b_torch.shape[0]",
            expected=n,
            actual=b_torch.shape[0],
        )


# ---------------------------------------------------------------------------
# sigma-path Function (numpy / scalar jac_vals)
# ---------------------------------------------------------------------------


class SparseLinearSolveWithSigmaGrad(torch.autograd.Function):
    """
    Differentiable ``x = A(σ)^{-1} b`` with closed-form σ-Jacobian.

    The σ-path uses the precomputed sparse Jacobian
    ``dA[A_rows[m], A_cols[m]] / dσ[sigma_indices[m]] = jac_vals[m]``
    and the adjoint-state identity

        grad_σ_k = - sum_{(i,j) in support(k)} λ[i] (dA[i,j]/dσ_k) x[j]    (real A)
        grad_σ_k = - Re[ sum_{(i,j) in support(k)} conj(λ[i]) (dA[i,j]/dσ_k) x[j] ]  (complex A, real σ)

    For the multi-RHS case (``b`` shape ``(n, K)``) the contribution sums
    over RHS columns. The minus sign is non-negotiable; getting it wrong
    silently flips gradient direction and inversion diverges.

    Both real (DC) and complex (FDEM / MT) ``A`` are supported. ``σ`` is
    always real (``float64``); ``jac_vals`` may be complex when ``A`` is
    complex (e.g. ``-iω`` on the diagonal for a curl-curl operator).
    """

    @staticmethod
    def forward(
        ctx,
        A_csr_data,
        A_csr_indices,
        A_csr_indptr,
        A_shape,
        b_torch,
        sigma_torch,
        jac_rows,
        jac_cols,
        jac_sig,
        jac_vals,
    ):
        lu = _checked_splu(
            A_csr_data, A_csr_indices, A_csr_indptr, A_shape,
            object_name="sparse_solve_with_sigma",
        )

        b_np = b_torch.detach().cpu().numpy()
        x_np = lu.solve(b_np)

        ctx.lu = lu
        ctx.b_device = b_torch.device
        ctx.b_dtype = b_torch.dtype
        ctx.b_is_complex = b_torch.is_complex()
        ctx.sigma_device = sigma_torch.device
        ctx.n_sigma = sigma_torch.shape[0]
        ctx.jac_rows = np.ascontiguousarray(jac_rows)
        ctx.jac_cols = np.ascontiguousarray(jac_cols)
        ctx.jac_sig = np.ascontiguousarray(jac_sig)
        # Cast jac_vals to A's dtype so contractions follow standard numpy
        # type promotion.
        ctx.jac_vals = np.ascontiguousarray(jac_vals, dtype=A_csr_data.dtype)
        ctx.x_np = np.ascontiguousarray(x_np)

        x_torch = torch.from_numpy(np.ascontiguousarray(x_np)).to(
            device=b_torch.device, dtype=b_torch.dtype
        )
        return x_torch

    @staticmethod
    def backward(ctx, grad_output):
        # 1) Adjoint solve. trans='H' is correct for both real (A^H = A^T)
        #    and complex (A^H = conj(A^T)).
        grad_np = np.ascontiguousarray(grad_output.detach().cpu().numpy())
        lam_np = ctx.lu.solve(grad_np, trans="H")
        grad_b = torch.from_numpy(np.ascontiguousarray(lam_np)).to(
            device=ctx.b_device, dtype=ctx.b_dtype
        )

        # 2) σ-path. For complex A with real σ the Wirtinger-correct
        #    accumulator is
        #        grad_σ_k = -Re[ sum_K sum_m conj(λ[i_m, K]) * jac_vals[m] * x[j_m, K] ]
        x_np = ctx.x_np
        rows = ctx.jac_rows
        cols = ctx.jac_cols
        sig = ctx.jac_sig
        vals = ctx.jac_vals

        lam_for_contraction = (
            np.conjugate(lam_np) if ctx.b_is_complex else lam_np
        )

        if x_np.ndim == 1:
            contribs = lam_for_contraction[rows] * vals * x_np[cols]
        else:
            n_rhs = x_np.shape[1]
            # Bound peak memory at ~64 MB of float64 per chunk so large
            # multi-RHS DC inversions do not allocate the full
            # (n_records, n_obs) intermediate.
            chunk = max(1, int(8_000_000 // max(rows.shape[0], 1)))
            chunk = min(chunk, n_rhs)
            contribs = np.zeros(rows.shape[0], dtype=lam_for_contraction.dtype)
            for s in range(0, n_rhs, chunk):
                e = min(s + chunk, n_rhs)
                contribs += (
                    lam_for_contraction[rows, s:e] * x_np[cols, s:e]
                ).sum(axis=1)
            contribs *= vals

        if ctx.b_is_complex:
            contribs = contribs.real

        grad_sigma_np = np.zeros(ctx.n_sigma, dtype=np.float64)
        # Scatter-add: many sparse-Jacobian entries can share the same σ idx.
        np.add.at(grad_sigma_np, sig, contribs)
        grad_sigma_np *= -1.0  # adjoint-state minus sign

        grad_sigma = torch.from_numpy(np.ascontiguousarray(grad_sigma_np)).to(
            device=ctx.sigma_device, dtype=torch.float64
        )

        # Ten inputs: A_csr_data, A_csr_indices, A_csr_indptr, A_shape,
        # b_torch, sigma_torch, jac_rows, jac_cols, jac_sig, jac_vals.
        return None, None, None, None, grad_b, grad_sigma, None, None, None, None


def sparse_solve_with_sigma(
    A_csr: sp.csr_matrix,
    b_torch: torch.Tensor,
    sigma_torch: torch.Tensor,
    jacobian: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> torch.Tensor:
    """
    Solve ``A(σ) x = b`` with autograd flowing to both ``b`` and ``σ``.

    Args:
        A_csr: square ``scipy.sparse.csr_matrix``, ``float64`` or ``complex128``.
            Already assembled at the current ``σ``.
        b_torch: right-hand side. ``torch.float64`` or ``torch.complex128``
            (must match ``A_csr.dtype``), shape ``(n,)`` or ``(n, k)``.
        sigma_torch: conductivity vector that built ``A``. ``torch.float64``
            (always real, even when ``A`` is complex), shape ``(n_sigma,)``.
        jacobian: sparse pattern as a 4-tuple
            ``(A_rows, A_cols, sigma_indices, dA_dsigma_vals)``. Each entry
            ``m`` says ``dA[A_rows[m], A_cols[m]] / dσ[sigma_indices[m]] ==
            dA_dsigma_vals[m]``. ``dA_dsigma_vals`` is real for real ``A`` and
            may be complex for complex ``A`` (e.g. ``-iω`` for a curl-curl /
            MT operator). Multiple entries with the same ``sigma_indices[m]``
            are summed.

    Returns:
        Solution ``x`` with the same shape and dtype as ``b_torch``.
    """
    _check_inputs(A_csr, b_torch)
    if not isinstance(sigma_torch, torch.Tensor):
        raise GeoBrainError(
            "sigma_torch must be a torch.Tensor",
            object_name="sparse_solve_with_sigma",
            field="sigma_torch",
            expected="torch.Tensor",
            actual=type(sigma_torch).__name__,
        )
    if sigma_torch.dtype != torch.float64:
        raise GeoBrainError(
            "sigma_torch must be float64",
            object_name="sparse_solve_with_sigma",
            field="sigma_torch.dtype",
            expected="float64",
            actual=sigma_torch.dtype,
        )
    if sigma_torch.dim() != 1:
        raise GeoBrainError(
            "sigma_torch must be 1D",
            object_name="sparse_solve_with_sigma",
            field="sigma_torch.shape",
            expected="(n_sigma,)",
            actual=tuple(sigma_torch.shape),
        )
    if sigma_torch.device.type != "cpu":
        raise GeoBrainError(
            "sparse_solve_with_sigma is CPU-only (scipy splu)",
            object_name="sparse_solve_with_sigma",
            field="sigma_torch.device",
            expected="cpu",
            actual=str(sigma_torch.device),
        )

    jac_rows, jac_cols, jac_sig, jac_vals = jacobian
    jac_rows = np.asarray(jac_rows, dtype=np.int64)
    jac_cols = np.asarray(jac_cols, dtype=np.int64)
    jac_sig = np.asarray(jac_sig, dtype=np.int64)
    jac_vals = np.asarray(jac_vals)
    if not (jac_rows.shape == jac_cols.shape == jac_sig.shape == jac_vals.shape):
        raise GeoBrainError(
            "jacobian arrays must all share the same shape",
            object_name="sparse_solve_with_sigma",
            field="jacobian",
            expected="(n_jac,) for all four",
            actual=(
                jac_rows.shape, jac_cols.shape, jac_sig.shape, jac_vals.shape,
            ),
        )

    A_csr_canon = A_csr if isinstance(A_csr, sp.csr_matrix) else A_csr.tocsr()
    return SparseLinearSolveWithSigmaGrad.apply(
        A_csr_canon.data,
        A_csr_canon.indices,
        A_csr_canon.indptr,
        A_csr_canon.shape,
        b_torch,
        sigma_torch,
        jac_rows,
        jac_cols,
        jac_sig,
        jac_vals,
    )


# ---------------------------------------------------------------------------
# sigma + torch-tracked-jac-vals path Function (SIP variant)
# ---------------------------------------------------------------------------


class SparseLinearSolveWithSigmaTorchJacGrad(torch.autograd.Function):
    """SIP-flavoured variant: ``jac_vals`` is a torch tensor with autograd.

    Used when the σ-Jacobian carries an upstream parameter (e.g. the
    Cole-Cole ``η_eff(ω)`` factor in SIP). The adjoint-state derivation
    is identical to :class:`SparseLinearSolveWithSigmaGrad`; in addition
    we accumulate gradients into ``jac_vals_torch``::

        grad_jac_vals[m] = -λ[rows[m]] σ[sig[m]] conj(x[cols[m]])     (summed over RHS)

    Note the asymmetry with the σ branch: σ is real so we project via
    ``Re``; ``jac_vals`` is complex so we keep the full Wirtinger-conjugate
    value (matching PyTorch's complex-autograd convention, same as the
    core ``sparse_linear_solve_with_adjoint`` helper).
    """

    @staticmethod
    def forward(
        ctx,
        A_csr_data,
        A_csr_indices,
        A_csr_indptr,
        A_shape,
        b_torch,
        sigma_torch,
        jac_rows,
        jac_cols,
        jac_sig,
        jac_vals_torch,
    ):
        lu = _checked_splu(
            A_csr_data, A_csr_indices, A_csr_indptr, A_shape,
            object_name="sparse_solve_with_sigma_torch_jac",
        )

        b_np = b_torch.detach().cpu().numpy()
        x_np = lu.solve(b_np)

        ctx.lu = lu
        ctx.b_device = b_torch.device
        ctx.b_dtype = b_torch.dtype
        ctx.b_is_complex = b_torch.is_complex()
        ctx.sigma_device = sigma_torch.device
        ctx.n_sigma = sigma_torch.shape[0]
        ctx.jac_device = jac_vals_torch.device
        ctx.jac_dtype = jac_vals_torch.dtype
        ctx.n_jac = jac_vals_torch.shape[0]
        ctx.jac_rows = np.ascontiguousarray(jac_rows)
        ctx.jac_cols = np.ascontiguousarray(jac_cols)
        ctx.jac_sig = np.ascontiguousarray(jac_sig)
        ctx.jac_vals_np = np.ascontiguousarray(
            jac_vals_torch.detach().cpu().numpy(), dtype=A_csr_data.dtype
        )
        ctx.sigma_np = np.ascontiguousarray(
            sigma_torch.detach().cpu().numpy(), dtype=np.float64
        )
        ctx.x_np = np.ascontiguousarray(x_np)

        x_torch = torch.from_numpy(np.ascontiguousarray(x_np)).to(
            device=b_torch.device, dtype=b_torch.dtype
        )
        return x_torch

    @staticmethod
    def backward(ctx, grad_output):
        # 1) Adjoint solve.
        grad_np = np.ascontiguousarray(grad_output.detach().cpu().numpy())
        lam_np = ctx.lu.solve(grad_np, trans="H")
        grad_b = torch.from_numpy(np.ascontiguousarray(lam_np)).to(
            device=ctx.b_device, dtype=ctx.b_dtype
        )

        x_np = ctx.x_np
        rows = ctx.jac_rows
        cols = ctx.jac_cols
        sig = ctx.jac_sig
        vals = ctx.jac_vals_np
        sigma_np = ctx.sigma_np
        is_complex = ctx.b_is_complex

        # σ-branch contraction: conj(λ[rows]) * x[cols], summed over RHS.
        lam_conj = np.conjugate(lam_np) if is_complex else lam_np
        if x_np.ndim == 1:
            core_sigma = lam_conj[rows] * x_np[cols]
        else:
            core_sigma = (lam_conj[rows] * x_np[cols]).sum(axis=1)

        sigma_contribs = core_sigma * vals
        if is_complex:
            sigma_contribs = sigma_contribs.real
        grad_sigma_np = np.zeros(ctx.n_sigma, dtype=np.float64)
        np.add.at(grad_sigma_np, sig, sigma_contribs)
        grad_sigma_np *= -1.0
        grad_sigma = torch.from_numpy(np.ascontiguousarray(grad_sigma_np)).to(
            device=ctx.sigma_device, dtype=torch.float64
        )

        # jac_vals branch: λ[rows] * conj(x[cols]), summed over RHS.
        x_for_jac = np.conjugate(x_np) if is_complex else x_np
        if x_np.ndim == 1:
            core_jac = lam_np[rows] * x_for_jac[cols]
        else:
            core_jac = (lam_np[rows] * x_for_jac[cols]).sum(axis=1)
        grad_jac_vals_np = -core_jac * sigma_np[sig]
        grad_jac_vals_np = np.ascontiguousarray(
            grad_jac_vals_np.astype(
                np.complex128 if ctx.jac_dtype.is_complex else np.float64,
                copy=False,
            )
        )
        grad_jac_vals = torch.from_numpy(grad_jac_vals_np).to(
            device=ctx.jac_device, dtype=ctx.jac_dtype
        )

        # Ten inputs.
        return (
            None, None, None, None,
            grad_b, grad_sigma,
            None, None, None, grad_jac_vals,
        )


def sparse_solve_with_sigma_torch_jac(
    A_csr: sp.csr_matrix,
    b_torch: torch.Tensor,
    sigma_torch: torch.Tensor,
    jac_rows: np.ndarray,
    jac_cols: np.ndarray,
    jac_sig: np.ndarray,
    jac_vals_torch: torch.Tensor,
) -> torch.Tensor:
    """
    Solve ``A(σ) x = b`` with autograd to ``b``, ``σ``, AND ``jac_vals_torch``.

    Variant of :func:`sparse_solve_with_sigma` where the σ-Jacobian
    *values* tensor is a torch tensor (typically ``complex128``), so
    upstream parameters that chain-rule into ``jac_vals`` (e.g. the
    Cole-Cole ``η_eff(ω)`` factor in the SIP operator) receive autograd
    gradients via the adjoint state.

    The Jacobian *indices* (``rows``, ``cols``, ``sig``) stay as numpy
    arrays; they describe the sparse pattern and do not depend on any
    learnable parameter.
    """
    _check_inputs(A_csr, b_torch)
    if not isinstance(sigma_torch, torch.Tensor):
        raise GeoBrainError(
            "sigma_torch must be a torch.Tensor",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="sigma_torch",
            expected="torch.Tensor",
            actual=type(sigma_torch).__name__,
        )
    if sigma_torch.dtype != torch.float64:
        raise GeoBrainError(
            "sigma_torch must be float64",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="sigma_torch.dtype",
            expected="float64",
            actual=sigma_torch.dtype,
        )
    if sigma_torch.dim() != 1:
        raise GeoBrainError(
            "sigma_torch must be 1D",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="sigma_torch.shape",
            expected="(n_sigma,)",
            actual=tuple(sigma_torch.shape),
        )
    if sigma_torch.device.type != "cpu":
        raise GeoBrainError(
            "sparse_solve_with_sigma_torch_jac is CPU-only (scipy splu)",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="sigma_torch.device",
            expected="cpu",
            actual=str(sigma_torch.device),
        )
    if not isinstance(jac_vals_torch, torch.Tensor):
        raise GeoBrainError(
            "jac_vals_torch must be a torch.Tensor",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="jac_vals_torch",
            expected="torch.Tensor",
            actual=type(jac_vals_torch).__name__,
        )
    if jac_vals_torch.dtype not in _ALLOWED_TORCH_DTYPES:
        raise GeoBrainError(
            "jac_vals_torch must be float64 or complex128",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="jac_vals_torch.dtype",
            expected="float64 or complex128",
            actual=jac_vals_torch.dtype,
        )
    if jac_vals_torch.dim() != 1:
        raise GeoBrainError(
            "jac_vals_torch must be 1D",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="jac_vals_torch.shape",
            expected="(n_jac,)",
            actual=tuple(jac_vals_torch.shape),
        )
    if jac_vals_torch.device.type != "cpu":
        raise GeoBrainError(
            "jac_vals_torch must be on CPU",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="jac_vals_torch.device",
            expected="cpu",
            actual=str(jac_vals_torch.device),
        )

    jac_rows = np.asarray(jac_rows, dtype=np.int64)
    jac_cols = np.asarray(jac_cols, dtype=np.int64)
    jac_sig = np.asarray(jac_sig, dtype=np.int64)
    if not (jac_rows.shape == jac_cols.shape == jac_sig.shape):
        raise GeoBrainError(
            "jac_rows, jac_cols, jac_sig must all share the same shape",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="jacobian indices",
            expected="(n_jac,) for all three",
            actual=(jac_rows.shape, jac_cols.shape, jac_sig.shape),
        )
    if jac_rows.shape[0] != jac_vals_torch.shape[0]:
        raise GeoBrainError(
            "jac_vals_torch length must match the index arrays",
            object_name="sparse_solve_with_sigma_torch_jac",
            field="jac_vals_torch.shape[0]",
            expected=jac_rows.shape[0],
            actual=jac_vals_torch.shape[0],
        )

    A_csr_canon = A_csr if isinstance(A_csr, sp.csr_matrix) else A_csr.tocsr()
    return SparseLinearSolveWithSigmaTorchJacGrad.apply(
        A_csr_canon.data,
        A_csr_canon.indices,
        A_csr_canon.indptr,
        A_csr_canon.shape,
        b_torch,
        sigma_torch,
        jac_rows,
        jac_cols,
        jac_sig,
        jac_vals_torch,
    )
