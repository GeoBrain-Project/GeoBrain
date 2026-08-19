"""
BDF2 implicit time-stepping orchestrator for diffusive 3D EM.

Solves the semi-discrete system

    M · dE/dt + K · E = J(t)

where ``M`` is a (sigma-weighted) mass matrix and ``K`` is a (curl-curl)
stiffness matrix, both real, sparse, ``(n, n)``. The forward step uses
implicit Euler (BDF1) for the first sub-step to bootstrap, then second-order
backward differentiation (BDF2) thereafter:

- BDF1 (step 1)::

      (M + dt · K) · E^1 = M · E^0 + dt · J(t^1)

- BDF2 (step n+1, n ≥ 1): uniform ``dt``::

      (1.5 M + dt · K) · E^(n+1) = M · (2 · E^n - 0.5 · E^(n-1)) + dt · J(t^(n+1))

  Variable step size is handled by the standard variable-coefficient BDF2
  with ``omega = dt_n / dt_{n-1}``::

      (alpha0 · M + dt_n · K) · E^(n+1)
          = -M · (alpha1 · E^n + alpha2 · E^(n-1)) + dt_n · J(t^(n+1))

      alpha0 = (1 + 2*omega) / (1 + omega)
      alpha1 = -(1 + omega)
      alpha2 = omega**2 / (1 + omega)

  For ``omega == 1`` these reduce to ``(1.5, -2, 0.5)``, the uniform case.

For TEM step-off response with a known initial condition (the steady-state
DC field at switch-off), this scheme is L-stable and second-order accurate
in ``dt``.

Implementation notes:
- Real-valued only. TEM is real in the time domain; complex frequency-domain
  solves live elsewhere.
- Sparse LU factorizations are cached by ``dt`` (with a tolerance hash) so
  that uniform, or piece-wise uniform, step sequences only refactorize
  once. This is critical because LU dominates per-step cost.
- No autograd integration. The TEM 3D operator will wrap this with a
  discrete-adjoint :class:`torch.autograd.Function` that re-uses the same
  cached factorizations on the backward sweep.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .....core.errors import GeoBrainError

__all__ = ["bdf2_stepwise"]


# Relative tolerance for treating two ``dt`` values as identical when keying
# the LU cache. 1e-12 is well below numerical step-size noise yet still
# distinguishes log-spaced sweeps that differ by even a single ULP after
# round-tripping through float64 arithmetic.
_DT_HASH_RTOL = 1e-12


def _dt_key(dt: float, scale: float) -> int:
    """
    Quantize ``dt`` to an integer key suitable for caching.

    Two ``dt`` values that agree to ``_DT_HASH_RTOL`` of ``scale`` share a key.
    ``scale`` is the largest ``|dt|`` seen in the schedule; using it as the
    quantization grid makes the tolerance scale-invariant.
    """
    if scale <= 0.0:
        scale = 1.0
    return int(round(dt / (scale * _DT_HASH_RTOL)))


def bdf2_stepwise(
    M_csr: sp.csr_matrix,
    K_csr: sp.csr_matrix,
    initial_field: np.ndarray,
    time_steps: np.ndarray,
    rhs_callable: Optional[Callable[[float], np.ndarray]] = None,
) -> np.ndarray:
    """
    Step ``M·dE/dt + K·E = J(t)`` forward using BDF1 + BDF2.

    Args:
        M_csr: Mass matrix (sigma-weighted), shape ``(n, n)``, real, sparse.
        K_csr: Stiffness matrix (curl-curl), shape ``(n, n)``, real, sparse.
        initial_field: ``E^0`` of length ``n``. Real-valued (complex inputs raise).
        time_steps: 1-D array of ``dt`` values, length ``n_steps``. Non-uniform
            stepping is fully supported; LU factorizations are cached per unique
            ``dt``.
        rhs_callable: Optional ``J(t) -> ndarray(n,)`` source. ``None`` (default)
            means a homogeneous problem (right-hand-side source is zero), typical
            for TEM step-off where the source is encoded purely in ``E^0``.

    Returns:
        Field history of shape ``(n_steps + 1, n)``. Index ``0`` is
        ``initial_field``; index ``k`` is ``E`` after the ``k``-th step.

    The first step uses implicit Euler::

        (M + dt · K) · E^1 = M · E^0 + dt · J(t^1)

    so that ``E^(-1)`` is never required. All subsequent steps use BDF2::

        (1.5 M + dt · K) · E^(n+1)
            = M · (2 · E^n - 0.5 · E^(n-1)) + dt · J(t^(n+1))

    Variable step sizes use the standard variable-coefficient BDF2; for
    uniform schedules this collapses exactly to the formula above.

    LU factorizations are cached by ``(alpha0, dt)`` with relative
    tolerance ``1e-12`` on each component. Switching schemes (BDF1 vs
    BDF2) re-keys the cache because the LHS coefficient on ``M`` changes
    from ``1.0`` to ``1.5`` (uniform) or whatever ``alpha0`` the variable
    formula produces.
    """
    if M_csr.shape != K_csr.shape:
        raise GeoBrainError(
            f"M and K must have the same shape; got {M_csr.shape} vs {K_csr.shape}",
            object_name="bdf2_stepwise",
            field="K_csr",
            expected=f"shape == M_csr.shape ({M_csr.shape})",
            actual=K_csr.shape,
        )
    n = M_csr.shape[0]
    if M_csr.shape[0] != M_csr.shape[1]:
        raise GeoBrainError(
            f"M must be square; got shape {M_csr.shape}",
            object_name="bdf2_stepwise",
            field="M_csr",
            expected="square (n, n)",
            actual=M_csr.shape,
        )

    E0 = np.asarray(initial_field)
    if E0.ndim != 1 or E0.shape[0] != n:
        raise GeoBrainError(
            f"initial_field must have shape ({n},), got {E0.shape}",
            object_name="bdf2_stepwise",
            field="initial_field",
            expected=f"({n},)",
            actual=E0.shape,
        )
    if np.iscomplexobj(E0):
        # A genuine *type* error (real dtype required): kept as TypeError;
        # GeoBrainError subclasses ValueError, not TypeError.
        raise TypeError(
            "bdf2_stepwise is real-valued only; complex initial_field is not supported"
        )
    E0 = E0.astype(np.float64, copy=False)

    dt_arr = np.asarray(time_steps, dtype=np.float64).ravel()
    if dt_arr.size == 0:
        return E0.reshape(1, n).copy()
    if np.any(dt_arr <= 0.0):
        raise GeoBrainError(
            "time_steps must be strictly positive",
            object_name="bdf2_stepwise",
            field="time_steps",
            expected="all dt > 0",
            actual=dt_arr,
        )

    M = M_csr.tocsr().astype(np.float64, copy=False)
    K = K_csr.tocsr().astype(np.float64, copy=False)

    n_steps = dt_arr.size
    E_all = np.zeros((n_steps + 1, n), dtype=np.float64)
    E_all[0] = E0

    # LU cache keyed by (alpha0_key, dt_key). Both axes are quantized with
    # the same relative tolerance so that uniform, or piece-wise uniform,
    # schedules only factorize once. A single ``dt`` reused under both BDF1
    # (alpha0 = 1) and BDF2 (alpha0 = 1.5) naturally gets two cache entries.
    dt_scale = float(np.max(np.abs(dt_arr)))
    lu_cache: dict[tuple[int, int], spla.SuperLU] = {}

    def _factor(alpha0: float, dt: float) -> spla.SuperLU:
        # Quantize alpha0 against unit scale (it lives in [1, 2] for our
        # range of practical step ratios), dt against the schedule maximum.
        key = (_dt_key(alpha0, 1.0), _dt_key(dt, dt_scale))
        cached = lu_cache.get(key)
        if cached is not None:
            return cached
        LHS = (alpha0 * M + dt * K).tocsc()
        lu = spla.splu(LHS)
        lu_cache[key] = lu
        return lu

    # Track absolute time so rhs_callable receives consistent times.
    t = 0.0

    # -- Step 1: implicit Euler bootstrap ---------------------------------
    dt = float(dt_arr[0])
    t += dt
    rhs = M @ E0
    if rhs_callable is not None:
        rhs = rhs + dt * np.asarray(rhs_callable(t), dtype=np.float64)
    lu = _factor(1.0, dt)
    E_prev_prev = E0
    E_prev = lu.solve(rhs)
    E_all[1] = E_prev
    dt_prev = dt

    # -- Steps 2..N: variable-coefficient BDF2 ----------------------------
    for step in range(1, n_steps):
        dt = float(dt_arr[step])
        t += dt
        omega = dt / dt_prev
        # Variable BDF2 coefficients; reduce to (1.5, -2, 0.5) for omega=1.
        alpha0 = (1.0 + 2.0 * omega) / (1.0 + omega)
        alpha1 = -(1.0 + omega)
        alpha2 = (omega * omega) / (1.0 + omega)
        rhs = -M @ (alpha1 * E_prev + alpha2 * E_prev_prev)
        if rhs_callable is not None:
            rhs = rhs + dt * np.asarray(rhs_callable(t), dtype=np.float64)
        lu = _factor(alpha0, dt)
        E_next = lu.solve(rhs)
        E_all[step + 1] = E_next
        E_prev_prev = E_prev
        E_prev = E_next
        dt_prev = dt

    return E_all
