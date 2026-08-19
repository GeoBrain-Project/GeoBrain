"""
EM-specific sparse adjoint bridges.

the :func:`geobrain.core.sparse_linear_solve_with_adjoint` already
covers the basic ``A x = b`` case (autograd through ``A.values()`` and
``b``). Two EM-flavoured specialisations on top of that:

- :func:`sparse_solve_with_sigma`: closed-form ``dA/dσ`` Jacobian:
  the caller supplies a sparse pattern
  ``(A_rows, A_cols, sigma_indices, dA_dsigma_vals)`` so the bridge can
  accumulate ``grad_σ`` directly without rebuilding the whole linear
  system in autograd. Used by MT2D (TE-diagonal, TM-face-harmonic) and
  any other EM operator with a closed-form σ-Jacobian.

- :func:`sparse_solve_with_sigma_torch_jac`: same shape, but
  ``dA_dsigma_vals`` is now a torch tensor (typically complex). Lets
  upstream chain-rule parameters such as the Cole-Cole
  ``(η₀, τ, c)`` SIP triplet receive gradients via the same adjoint
  state. Used by SIP-flavoured operators.

The bridges live here (not in ``core/``) because they only make sense in
the EM-σ-parametrised setting: ``σ`` is real, ``A`` may be complex, and
the Wirtinger conjugate goes on ``x`` (matching the existing E5
``sparse_linear_solve_with_adjoint``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .lu_bridge import (
    SparseLinearSolveWithSigmaGrad,
    SparseLinearSolveWithSigmaTorchJacGrad,
    sparse_solve_with_sigma,
    sparse_solve_with_sigma_torch_jac,
)

__all__ = [
    "SparseLinearSolveWithSigmaGrad",
    "SparseLinearSolveWithSigmaTorchJacGrad",
    "sparse_solve_with_sigma",
    "sparse_solve_with_sigma_torch_jac",
]
