"""NFVM transient physics solvers: multiphase / thermal displacement on the monotone
NFVM flux, built on the :mod:`.kernel` core.

``nfvm_two_phase`` (oil-water), ``nfvm_thermal_conduction``, and the coupled thermal
solvers ``nfvm_thermal_single_phase`` / ``nfvm_thermal_two_phase`` /
``nfvm_thermal_compositional``, plus the shared robustification helpers
``_newton_solve`` (singular-Jacobian → fail-loud) and ``_adaptive_march`` (adaptive
dt sub-stepping / pseudo-transient continuation).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Literal, NoReturn, TypeAlias

import torch

from ...config import FlowHistoryConfig
from ...errors import FlowContractError, FlowConvergenceError
from ...history import FlowHistoryWriter
from ...properties.relperm import RelPerm
from ...solvers.diagnostics import FlowConvergenceDiagnostics, convergence_diagnostics
from ..flux import scatter_boundary_outflow, scatter_internal_face_flux
from .kernel import (
    NFVMGeometry,
    _build_nfvm,
    _onesided,
    nfvm_flux,
)


Scalar: TypeAlias = float | int | torch.Tensor
TensorInput: TypeAlias = Scalar | Sequence[float] | Sequence[Sequence[float]]
WorkCounters: TypeAlias = dict[str, int]
ConvergenceStage: TypeAlias = Literal["flash", "nonlinear", "linear"]
ConvergenceReason: TypeAlias = Literal[
    "tolerance",
    "max_iterations",
    "line_search",
    "breakdown",
    "nonfinite",
    "invalid_state",
]
AdaptiveReason: TypeAlias = Literal["newton", "saturation"]
AdaptiveOutcome: TypeAlias = (
    tuple[torch.Tensor, bool, AdaptiveReason]
    | tuple[torch.Tensor, bool, AdaptiveReason, WorkCounters]
)
AdaptiveSolve: TypeAlias = Callable[[torch.Tensor, float], AdaptiveOutcome]
Perforation: TypeAlias = tuple[int, Scalar]
TwoPhaseWell: TypeAlias = tuple[int, Scalar, Scalar, Scalar]
SinglePhaseWell: TypeAlias = tuple[int, Scalar, Scalar, Scalar]
SinglePhaseRateWell: TypeAlias = (
    tuple[int, Scalar, Scalar, Scalar]
    | tuple[int, Scalar, Scalar, Scalar, Scalar | None]
)
SinglePhaseMultiperfWell: TypeAlias = tuple[Sequence[Perforation], Scalar, Scalar]
SinglePhaseNeumann: TypeAlias = tuple[int, Scalar, Scalar]
ThermalTwoPhaseWell: TypeAlias = tuple[int, Scalar, Scalar, Scalar, Scalar]
ThermalTwoPhaseRateWell: TypeAlias = (
    tuple[int, Scalar, Scalar, Scalar, Scalar]
    | tuple[int, Scalar, Scalar, Scalar, Scalar, Scalar | None]
)
ThermalTwoPhaseMultiperfWell: TypeAlias = tuple[
    Sequence[Perforation], Scalar, Scalar, Scalar, Literal["reservoir", "surface"]
]
ThermalTwoPhaseNeumann: TypeAlias = tuple[int, Scalar, Scalar, Scalar]
CompositionalWell: TypeAlias = tuple[int, Scalar, Scalar, TensorInput, Scalar]
CompositionalRateWell: TypeAlias = (
    tuple[int, Scalar, Scalar, TensorInput, Scalar]
    | tuple[int, Scalar, Scalar, TensorInput, Scalar, Scalar | None]
)
CompositionalMultiperfWell: TypeAlias = tuple[
    Sequence[Perforation], Scalar, TensorInput, Scalar
]
CompositionalNeumann: TypeAlias = tuple[int, Scalar, TensorInput, Scalar]

# A converged thermal two-phase state may exceed [0,1] by this much before it is treated
# as a non-physical over-injection (fail loud) rather than the documented mild monotonicity
# excursion that strongly anisotropic / two-point-fallback faces can produce at convergence.
_SAT_BOUND_TOL = 1e-2


class _NFVMTensorHistory:
    """Bounded adapter from NFVM tensor histories to the shared Flow writer."""

    def __init__(
        self,
        initial: torch.Tensor,
        *,
        nsteps: int,
        dt_s: float,
        config: FlowHistoryConfig | None,
        accepted_step_bound: int | None = None,
    ) -> None:
        self._dt_s = float(dt_s)
        self._time_s = 0.0
        self._legacy_output = config is None
        self._output_states = [initial.clone()]
        self._writer = FlowHistoryWriter(
            FlowHistoryConfig(mode="all") if config is None else config,
            accepted_step_bound=(nsteps if accepted_step_bound is None else accepted_step_bound),
        )
        self._writer.record_initial(time_s=0.0, state={"state": initial.clone()})

    @property
    def time_s(self) -> float:
        return self._time_s

    def append(
        self,
        state: torch.Tensor,
        *,
        dt_s: float | None = None,
        control: Mapping[str, object] | None = None,
        expose_output: bool = True,
        residual_evaluations: int = 0,
        jacobian_assemblies: int = 0,
        linear_solves: int = 0,
    ) -> None:
        accepted_dt = self._dt_s if dt_s is None else float(dt_s)
        self._time_s += accepted_dt
        self._writer.record_accepted(
            time_s=self._time_s,
            state={"state": state.clone()},
            dt_s=accepted_dt,
            control=control,
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=jacobian_assemblies,
            linear_solves=linear_solves,
        )
        if expose_output:
            self._output_states.append(state.clone())

    def record_output(self, state: torch.Tensor) -> None:
        """Preserve the historical macro-step tensor return by default."""
        self._output_states.append(state.clone())

    def stack(self) -> torch.Tensor:
        history = self._writer.finalize()
        states = (
            self._output_states
            if self._legacy_output
            else [entry["state"] for entry in history.states]
        )
        result = torch.stack(states)
        result.flow_history = history
        return result

    def reject(
        self,
        diagnostics: FlowConvergenceDiagnostics,
        *,
        residual_evaluations: int = 0,
        jacobian_assemblies: int = 0,
        linear_solves: int = 0,
    ) -> None:
        self._writer.record_rejected(
            diagnostics,
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=jacobian_assemblies,
            linear_solves=linear_solves,
        )


def _validate_newton_controls(
    *, max_iter: object, tol: object, object_name: str
) -> tuple[int, float]:
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise FlowContractError(
            "max_iter must be a positive integer",
            object_name=object_name,
            field="max_iter",
            expected="positive integer",
            actual=max_iter,
        )
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not math.isfinite(float(tol))
        or float(tol) <= 0
    ):
        raise FlowContractError(
            "tol must be positive and finite",
            object_name=object_name,
            field="tol",
            expected="finite value > 0",
            actual=tol,
        )
    return max_iter, float(tol)


def _raise_nfvm_convergence(
    *,
    object_name: str,
    reason: ConvergenceReason,
    iterations: int,
    max_iterations: int,
    residual_norm: float,
    tolerance: float,
    step_index: int | None = None,
    failed_cells: tuple[int, ...] = (),
    initial_residual_norm: float | None = None,
    residual_history: tuple[float, ...] | list[float] = (),
    message: str,
    stage: ConvergenceStage = "nonlinear",
    cause: Exception | None = None,
) -> NoReturn:
    history = tuple(float(value) for value in residual_history)
    if not history:
        history = (float(residual_norm),)
    record = convergence_diagnostics(
        stage=stage,
        converged=False,
        reason=reason,
        iterations=iterations,
        max_iterations=max_iterations,
        initial_residual_norm=(
            history[0] if initial_residual_norm is None else float(initial_residual_norm)
        ),
        residual_norm=residual_norm,
        residual_history=history,
        failed_cells=failed_cells,
        step_index=step_index,
    )
    error = FlowConvergenceError(
        message,
        object_name=object_name,
        field="convergence",
        expected=f"residual norm < {tolerance}",
        actual=record.to_dict(),
        diagnostics=record,
    )
    if cause is None:
        raise error
    raise error from cause


def _face_divergence(flux: torch.Tensor, left: int, right: int, n_cells: int) -> torch.Tensor:
    """Scatter one canonical left-to-right scalar NFVM face flux."""

    cells = torch.tensor([[left, right]], dtype=torch.long, device=flux.device)
    return scatter_internal_face_flux(flux.reshape(1), cells, n_cells)


def _boundary_divergence(flux: torch.Tensor, cell: int, n_cells: int) -> torch.Tensor:
    """Scatter one canonical cell-to-exterior scalar NFVM flux."""

    cells = torch.tensor([cell], dtype=torch.long, device=flux.device)
    return scatter_boundary_outflow(flux.reshape(1), cells, n_cells)


# --------------------------------------------------------------------------
# Multiphase NFVM: two-phase displacement on the monotone flux
# --------------------------------------------------------------------------
def nfvm_two_phase(
    geom: NFVMGeometry,
    relperm: RelPerm,
    wells: Sequence[TwoPhaseWell],
    *,
    mu_w: float = 1e-3,
    mu_o: float = 2e-3,
    phi: Scalar = 0.2,
    V: Scalar = 1.0,
    c: float = 1e-7,
    p_ref: float = 1e7,
    sw0: float = 0.05,
    p0: float = 1e7,
    dt: float = 8.64e4,
    nsteps: int = 12,
    scheme: str = "nmpfa",
    tol: float = 1e-8,
    max_iter: int = 40,
    history_config: FlowHistoryConfig | None = None,
) -> torch.Tensor:
    """Oil-water two-phase displacement on the **NFVM monotone flux** (closed domain
    + BHP wells), fully-implicit. The per-face geometric flux ``G_f = nfvm_flux(p)``
    is the monotone NTPFA/NMPFA two-point form; the phase mass flux is
    ``F_α = (ρ_α k_rα/μ_α)|_up · G_f`` (phase-potential upwinding). Because ``G_f``
    is monotone the saturation stays in ``[0, 1]`` on anisotropic / skewed grids
    where the linear MPFA-O flux would overshoot.

    ``geom`` a :class:`NFVMGeometry` (no Dirichlet ghosts → no-flow boundaries);
    ``relperm`` a :class:`RelPerm` (``kr_water``/``kr_oil`` of ``S_w``); ``wells`` a
    list ``(cell, WI, bhp, inj_sw)``. Slight compressibility ``c`` anchors the
    otherwise-singular pressure. Returns the ``(nsteps+1, n_cells)`` saturation
    history (``S_w`` per step). Differentiable in ``perm`` (via ``geom.K``)
    through the WHOLE march: each converged step is reattached with its exact
    implicit-function-theorem VJP (:func:`_ift_attach`) and steps chain without
    detaching, so a multi-step history gradient is the exact through-time
    gradient (FD-pinned by the NFVM gradient suite)."""
    max_iter, tol = _validate_newton_controls(
        max_iter=max_iter,
        tol=tol,
        object_name="nfvm_two_phase",
    )
    interior, _ = _build_nfvm(geom, scheme)
    n = geom.n
    dtype = geom._permeability_view().dtype
    Vc = V if isinstance(V, torch.Tensor) else torch.full((n,), float(V), dtype=dtype)
    phic = phi if isinstance(phi, torch.Tensor) else torch.full((n,), float(phi), dtype=dtype)

    def rho(p: torch.Tensor) -> torch.Tensor:
        return 1.0 + c * (p - p_ref)

    def residual(state: torch.Tensor, state_old: torch.Tensor) -> torch.Tensor:
        # raw S_w (the relperm clamps its own normalised saturation): clamping S_w
        # here would zero the accumulation derivative once a cell saturates → singular.
        p, sw = state[:n], state[n:]
        p_o, sw_o = state_old[:n], state_old[n:]
        so, so_o = 1 - sw, 1 - sw_o
        rw, ro = rho(p), rho(p)
        rw_o, ro_o = rho(p_o), rho(p_o)
        acc_w = Vc * phic * (rw * sw - rw_o * sw_o) / dt
        acc_o = Vc * phic * (ro * so - ro_o * so_o) / dt
        R_w, R_o = acc_w, acc_o
        mm_w = rw * relperm.kr_water(sw) / mu_w
        mm_o = ro * relperm.kr_oil(sw) / mu_o
        for L_disc, R_disc in interior:  # NFVM monotone geometric flux of p
            # The kernel exposes the shared canonical orientation: G > 0 means
            # left-to-right flow, so the upwind mobility is the left cell's.
            G = nfvm_flux(p, L_disc, R_disc, scheme)
            left, right = L_disc["left"], L_disc["right"]
            up = float(G.detach()) >= 0
            Fw = (mm_w[left] if up else mm_w[right]) * G
            Fo = (mm_o[left] if up else mm_o[right]) * G
            R_w = R_w + _face_divergence(Fw, left, right, n)
            R_o = R_o + _face_divergence(Fo, left, right, n)
        for cell, WI, bhp, inj_sw in wells:  # BHP wells (mass + fractional flow)
            dpw = p[cell] - bhp
            sw_w = sw[cell] if float(dpw.detach()) >= 0 else torch.as_tensor(inj_sw, dtype=dtype)
            Fw = WI * rho(p[cell]) * relperm.kr_water(sw_w) / mu_w * dpw
            Fo = WI * rho(p[cell]) * relperm.kr_oil(sw_w) / mu_o * dpw
            R_w = R_w + _boundary_divergence(Fw, cell, n)
            R_o = R_o + _boundary_divergence(Fo, cell, n)
        return torch.cat([R_w, R_o])

    state = torch.cat(
        [
            geom._permeability_view().new_full((n,), p0),
            geom._permeability_view().new_full((n,), sw0),
        ]
    )
    history = _NFVMTensorHistory(state[n:], nsteps=nsteps, dt_s=dt, config=history_config)
    for step in range(nsteps):
        work: WorkCounters = {
            "residual_evaluations": 0,
            "jacobian_assemblies": 0,
            "linear_solves": 0,
        }

        def evaluate(q: torch.Tensor, old_state: torch.Tensor) -> torch.Tensor:
            work["residual_evaluations"] += 1
            return residual(q, old_state)

        def assemble(q: torch.Tensor, old_state: torch.Tensor) -> torch.Tensor:
            work["jacobian_assemblies"] += 1
            return torch.autograd.functional.jacobian(
                lambda candidate: evaluate(candidate, old_state),
                q,
                vectorize=True,
            )

        def solve_linear(
            jacobian: torch.Tensor, residual_value: torch.Tensor
        ) -> torch.Tensor:
            work["linear_solves"] += 1
            return _newton_solve(jacobian, residual_value, "nfvm_two_phase")

        prev = state  # attached, the through-time link
        old = prev.detach()
        with torch.no_grad():  # the Newton march is grad-free; the
            s = old.clone()  # gradient comes from _ift_attach below
            converged = False
            residual_history: list[float] = []
            newton_iterations = 0
            for _it in range(max_iter):
                r = evaluate(s, old)
                rn = float(torch.linalg.vector_norm(r.detach(), ord=float("inf")))
                residual_history.append(rn)
                if rn < tol:
                    converged = True
                    newton_iterations = _it
                    break
                J = assemble(s, old)
                dp = solve_linear(J, r)
                newton_iterations = _it + 1
                alpha, found = 1.0, False
                for _ls in range(20):  # backtracking line search on |r|∞
                    try:
                        trial_residual = evaluate(s - alpha * dp, old)
                    except FlowContractError:
                        alpha *= 0.5
                        continue
                    if (
                        float(torch.linalg.vector_norm(trial_residual.detach(), ord=float("inf")))
                        < rn
                    ):
                        found = True
                        break
                    alpha *= 0.5
                if not found:  # no descent possible (Newton stalled)
                    break
                s = s - alpha * dp
        if not converged:  # fail loud; do not silently return (and clamp)
            _raise_nfvm_convergence(
                object_name="nfvm_two_phase",
                reason="line_search" if not found else "max_iterations",
                iterations=newton_iterations,
                max_iterations=max_iter,
                residual_norm=rn,
                tolerance=tol,
                step_index=step,
                residual_history=residual_history,
                message=(
                    "NFVM two-phase Newton solve did not converge; reduce dt, "
                    "the well rate, or the anisotropy"
                ),
            )
        sw_d = s[n:].detach()  # raw saturation (no clamp), fail loud if a
        if (
            float(sw_d.max()) > 1.0 + _SAT_BOUND_TOL or float(sw_d.min()) < -_SAT_BOUND_TOL
        ):  # converged state is
            failed = tuple(
                int(index)
                for index in torch.nonzero(
                    (sw_d < -_SAT_BOUND_TOL) | (sw_d > 1.0 + _SAT_BOUND_TOL),
                    as_tuple=False,
                ).flatten()
            )
            _raise_nfvm_convergence(
                object_name="nfvm_two_phase",
                reason="invalid_state",
                iterations=newton_iterations,
                max_iterations=max_iter,
                residual_norm=rn,
                tolerance=tol,
                step_index=step,
                failed_cells=failed,
                residual_history=residual_history,
                message="NFVM two-phase converged to a non-physical saturation",
            )
        state = _ift_attach(
            s,
            lambda q: residual(q, prev),
            "nfvm_two_phase",
            work=work,
        )
        history.append(
            state[n:],
            residual_evaluations=work["residual_evaluations"],
            jacobian_assemblies=work["jacobian_assemblies"],
            linear_solves=work["linear_solves"],
        )  # raw, guards above ensure S_w ∈ [0,1] on success
    return history.stack()


def nfvm_thermal_conduction(
    geom: NFVMGeometry,
    T0: torch.Tensor,
    *,
    rho_C: Scalar = 2.0e6,
    V: Scalar = 1.0,
    dt: float = 8.64e4,
    nsteps: int = 10,
    scheme: str = "nmpfa",
    tol: float = 1e-8,
    max_iter: int = 40,
    history_config: FlowHistoryConfig | None = None,
) -> torch.Tensor:
    """Transient heat **conduction** on the NFVM monotone flux: the thermal analog
    of the single-phase NFVM. ``geom.K`` is the bulk thermal-conductivity tensor (W/m·K)
    and ``geom.ghost_p`` the Dirichlet boundary temperatures (no ghosts ⇒ insulated /
    no-flux boundaries). The energy balance per cell is::

        ρ_C·V·(T − T|old)/Δt + Σ_f F^cond_f = 0 ,   F^cond_f = nfvm_flux(T)

    with ``F^cond`` the monotone NTPFA/NMPFA two-point Fourier flux, so the temperature
    respects the discrete maximum principle (stays within its initial / boundary range)
    on strongly-anisotropic conductivity where the *linear* MPFA-O conduction overshoots.
    ``ρ_C`` = bulk volumetric heat capacity ((1−φ)ρ_r C_r + φ ρ C_f) [J/(m³·K)]. Returns
    the ``(nsteps+1, n_cells)`` temperature history; differentiable in the conductivity
    through the whole march (per-step implicit VJP via :func:`_ift_attach`, steps chained
    without detaching; the multi-step gradient is the exact through-time gradient)."""
    max_iter, tol = _validate_newton_controls(
        max_iter=max_iter,
        tol=tol,
        object_name="nfvm_thermal_conduction",
    )
    interior, boundary = _build_nfvm(geom, scheme)
    n = geom.n
    dtype = geom._permeability_view().dtype
    Vc = V if isinstance(V, torch.Tensor) else torch.full((n,), float(V), dtype=dtype)
    rc = rho_C if isinstance(rho_C, torch.Tensor) else torch.full((n,), float(rho_C), dtype=dtype)
    ghost_pressures = geom._ghost_pressures_view()
    ghost = torch.stack(ghost_pressures) if ghost_pressures else None

    def residual(
        Tc: torch.Tensor, T_old: torch.Tensor, mu_half: bool = False
    ) -> torch.Tensor:
        # energy: ρ_C·V·(T−T_old)/Δt = ∇·(λ∇T); the NFVM flux ``F``
        # (left→right) is the *down-gradient* heat flux, so the discrete Laplacian
        # (heat into a cell) is −scatter(F): the conduction term enters the
        # residual with a minus sign.
        T = torch.cat([Tc, ghost]) if ghost is not None else Tc
        R = rc * Vc * (Tc - T_old) / dt
        for L_disc, R_disc in interior:
            F = nfvm_flux(T, L_disc, R_disc, scheme, mu_half=mu_half)
            R = R + _face_divergence(F, L_disc["left"], L_disc["right"], n)
        for L_disc in boundary:
            q, _ = _onesided(T, L_disc)
            R = R + _boundary_divergence(-q, L_disc["left"], n)
        return R

    T = T0.clone()
    history = _NFVMTensorHistory(T0, nsteps=nsteps, dt_s=dt, config=history_config)
    for step in range(nsteps):
        work: WorkCounters = {
            "residual_evaluations": 0,
            "jacobian_assemblies": 0,
            "linear_solves": 0,
        }

        def evaluate(
            q: torch.Tensor, old_state: torch.Tensor, *, mu_half: bool = False
        ) -> torch.Tensor:
            work["residual_evaluations"] += 1
            return residual(q, old_state, mu_half=mu_half)

        def assemble(
            q: torch.Tensor, old_state: torch.Tensor, *, mu_half: bool = False
        ) -> torch.Tensor:
            work["jacobian_assemblies"] += 1
            return torch.autograd.functional.jacobian(
                lambda candidate: evaluate(candidate, old_state, mu_half=mu_half),
                q,
                vectorize=True,
            )

        def solve_linear(
            jacobian: torch.Tensor, residual_value: torch.Tensor
        ) -> torch.Tensor:
            work["linear_solves"] += 1
            return _newton_solve(jacobian, residual_value, "nfvm_thermal_conduction")

        prev = T  # attached, the through-time link
        old = prev.detach()
        # avgMPFA init (linear μ≡½ implicit step) then nonlinear NFVM Newton, more
        # stable than a cold T_old start for the nonlinear flux. NOTE: on strongly
        # anisotropic conductivity where the conical basis falls back to two-point on
        # some faces, the transient can mildly violate the maximum principle (the
        # two-point fallback is not monotone); the steady NFVM and isotropic / moderate
        # anisotropy stay bounded.
        with torch.no_grad():  # grad-free march; gradient from _ift_attach
            s = old.clone()
            r0 = evaluate(s, old, mu_half=True)
            J0 = assemble(s, old, mu_half=True)
            s = s - solve_linear(J0, r0)
            converged = False
            rn = float("inf")
            residual_history: list[float] = []
            newton_iterations = 0
            failure_reason: ConvergenceReason = "max_iterations"
            for _it in range(max_iter):
                r = evaluate(s, old)
                rn = float(torch.linalg.vector_norm(r, ord=float("inf")))
                residual_history.append(rn)
                if rn < tol:
                    converged = True
                    newton_iterations = _it
                    break
                J = assemble(s, old)
                dT = solve_linear(J, r)
                newton_iterations = _it + 1
                alpha = 1.0
                found = False
                for _ls in range(20):
                    if (
                        float(
                            torch.linalg.vector_norm(
                                evaluate(s - alpha * dT, old), ord=float("inf")
                            )
                        )
                        < rn
                    ):
                        found = True
                        break
                    alpha *= 0.5
                if not found:
                    failure_reason = "line_search"
                    break
                s = s - alpha * dT
        if not converged:  # fail loud; do not silently return a
            _raise_nfvm_convergence(
                object_name="nfvm_thermal_conduction",
                reason=failure_reason,
                iterations=newton_iterations,
                max_iterations=max_iter,
                residual_norm=rn,
                tolerance=tol,
                step_index=step,
                residual_history=residual_history,
                message="NFVM thermal-conduction Newton solve did not converge",
            )
        T = _ift_attach(
            s,
            lambda q: residual(q, prev),
            "nfvm_thermal_conduction",
            work=work,
        )
        history.append(
            T,
            residual_evaluations=work["residual_evaluations"],
            jacobian_assemblies=work["jacobian_assemblies"],
            linear_solves=work["linear_solves"],
        )
    return history.stack()


def _newton_solve(
    A: torch.Tensor, b: torch.Tensor, fname: str
) -> torch.Tensor:
    """Dense Newton linear solve that turns a singular Jacobian into the module's fail-loud
    :class:`GeoBrainError` instead of a bare LAPACK ``LinAlgError``. The usual culprit is an
    ill-posed pressure problem: a closed, **incompressible** (``c=0``) domain driven by a net
    source / Neumann rate with no pressure datum has a rank-deficient pressure block (the
    pressure is defined only up to a constant and a net rate cannot fill an incompressible
    closed volume)."""
    exc: Exception | None = None
    x: torch.Tensor | None = None
    try:
        x = torch.linalg.solve(A, b)
    except torch.linalg.LinAlgError as e:
        exc = e
    # LAPACK's ``getrf`` only raises on an *exact* zero pivot; a numerically
    # rank-deficient Jacobian (whether it trips the pivot check is BLAS- and
    # rounding-dependent) otherwise yields a non-finite or residual-violating
    # solution instead of raising. Validate the solve explicitly so the
    # fail-loud singular contract is deterministic across environments/threads.
    if x is None:
        residual_norm = float(torch.linalg.vector_norm(b.detach()))
        _raise_nfvm_convergence(
            object_name=fname,
            stage="linear",
            reason="breakdown",
            iterations=0,
            max_iterations=1,
            residual_norm=residual_norm,
            tolerance=1.0e-6 * (residual_norm + 1.0e-30),
            message=(
                "singular Newton Jacobian: add compressibility, a pressure "
                "datum, or balance the net source"
            ),
            cause=exc,
        )
    singular = not bool(torch.isfinite(x).all())
    if not singular:
        with torch.no_grad():
            b_norm = torch.linalg.vector_norm(b)
            resid = torch.linalg.vector_norm(A @ x - b)
        singular = bool(resid > 1e-6 * (b_norm + 1e-30))
    if singular:
        residual_norm = float(torch.linalg.vector_norm(b.detach()))
        _raise_nfvm_convergence(
            object_name=fname,
            stage="linear",
            reason="breakdown",
            iterations=0,
            max_iterations=1,
            residual_norm=residual_norm,
            tolerance=1.0e-6 * (residual_norm + 1.0e-30),
            message=(
                "singular Newton Jacobian: add compressibility, a pressure "
                "datum, or balance the net source"
            ),
            cause=exc,
        )
    return x


def _ift_attach(
    s_star: torch.Tensor,
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    fname: str,
    *,
    work: WorkCounters | None = None,
) -> torch.Tensor:
    """Reattach a converged (grad-free) Newton state to the autograd graph with
    its exact per-step implicit-function-theorem gradient.

    ``s_star`` is the converged step state computed under ``torch.no_grad()``;
    ``residual_fn(q)`` evaluates the TRUE (non-``mu_half``) residual at ``q``,
    closing over everything else the step depends on, the model parameters
    (through the prebuilt NFVM stencils) and the PREVIOUS state, both still
    attached to their graphs. With ``J = ∂r/∂q`` at ``s_star`` (a detached
    constant) and ``δ = J⁻¹·r(s_star)``,

        out = s_star + (δ.detach() − δ)

    has the value of ``s_star`` EXACTLY (the correction is identically zero
    elementwise) while ``d out = −J⁻¹(∂r/∂θ·dθ + ∂r/∂prev·dprev)``, the
    implicit-function-theorem derivative of the fully-implicit step. Chaining
    steps WITHOUT detaching then yields the exact discrete through-time
    adjoint at one extra Jacobian + linear solve per step, with no Newton
    unrolling kept in memory.

    This replaces the old attached-Newton-unroll gradient, which was wrong in
    two independent ways: the ``old = state.detach()`` between steps severed
    the through-time chain (a multi-step history gradient reached only the
    final step), and whenever the linear ``mu_half`` avgMPFA INIT step already
    satisfied the nonlinear residual (common on mild problems, the Newton
    loop then converged at iteration 0 with no true update executed) the
    returned gradient was the IFT gradient of the WRONG (μ≡½ linearised)
    operator. The NFVM gradient suite pins all five
    transients against multi-step central finite differences."""
    jacobian_residual: Callable[[torch.Tensor], torch.Tensor]
    if work is not None:
        work["residual_evaluations"] += 1
    r = residual_fn(s_star)
    if work is not None:
        work["jacobian_assemblies"] += 1

        def counted_residual(state: torch.Tensor) -> torch.Tensor:
            work["residual_evaluations"] += 1
            return residual_fn(state)

        jacobian_residual = counted_residual
    else:
        jacobian_residual = residual_fn
    J = torch.autograd.functional.jacobian(jacobian_residual, s_star, vectorize=True)
    if work is not None:
        work["linear_solves"] += 1
    delta = _newton_solve(J, r, fname)
    return s_star + (delta.detach() - delta)


def _adaptive_march(
    solve_one: AdaptiveSolve,
    state0: torch.Tensor,
    nsteps: int,
    max_substeps: int,
    fname: str,
    *,
    dt_s: float,
    history_config: FlowHistoryConfig | None = None,
) -> torch.Tensor:
    """Explicit transient march with **adaptive dt sub-stepping** (pseudo-transient
    continuation), the robustification for the NFVM thermal monotone solvers.

    ``solve_one(state, dt_frac) -> (new_state, ok, reason)`` advances ``state`` by
    ``dt_frac`` of the macro step; ``ok=False`` (with ``reason`` ``"newton"`` /
    ``"saturation"``) means its Newton did not converge / the result is non-physical;
    it does NOT raise. Each macro step is marched over ``[0, 1]`` in fractions; a failed
    fraction is **halved** (and re-tried from the same sub-state) down to
    ``1/2**max_substeps`` before giving up fail-loud, and the fraction is grown back after
    a success. ``max_substeps=0`` ⇒ no sub-stepping (a failure raises immediately, the
    pre-robustification behaviour). Returns the ``(nsteps+1, …)`` macro-step history; every
    accepted sub-step is reattached by ``solve_one`` via :func:`_ift_attach` and the chain
    is NEVER detached, so the history carries the exact through-time gradient."""
    history = _NFVMTensorHistory(
        state0,
        nsteps=nsteps,
        dt_s=dt_s,
        config=history_config,
        accepted_step_bound=nsteps * max(1, 2**max_substeps),
    )
    state = state0
    min_frac = 0.5**max_substeps
    msg: dict[AdaptiveReason, str] = {
        "newton": "did not converge",
        "saturation": "converged to a non-physical saturation",
    }
    for step in range(nsteps):
        sub = state
        remaining, frac = 1.0, 1.0
        reason: AdaptiveReason = "newton"
        attempted_substeps = 0
        while remaining > 1e-12:
            f = frac if frac < remaining else remaining
            report_times = (
                history_config.report_times_s
                if history_config is not None and history_config.mode == "report"
                else ()
            )
            next_report = next(
                (report for report in report_times if report > history.time_s + 1e-12),
                None,
            )
            if next_report is not None:
                f = min(f, (next_report - history.time_s) / float(dt_s))
            attempted_substeps += 1
            outcome = solve_one(sub, f)
            if len(outcome) == 3:
                s, ok, reason = outcome
                work: WorkCounters = {
                    "residual_evaluations": 0,
                    "jacobian_assemblies": 0,
                    "linear_solves": 0,
                }
            else:
                s, ok, reason, work = outcome
            if ok:
                remaining -= f
                sub = s
                frac = min(1.0, frac * 2.0)
                history.append(
                    sub,
                    dt_s=f * float(dt_s),
                    control={"nfvm_solver": fname},
                    expose_output=False,
                    residual_evaluations=int(work["residual_evaluations"]),
                    jacobian_assemblies=int(work["jacobian_assemblies"]),
                    linear_solves=int(work["linear_solves"]),
                )
            else:
                rejected = convergence_diagnostics(
                    stage="nonlinear",
                    converged=False,
                    reason=("invalid_state" if reason == "saturation" else "max_iterations"),
                    iterations=int(work.get("nonlinear_iterations", 0)),
                    max_iterations=int(
                        work.get(
                            "max_nonlinear_iterations",
                            work.get("nonlinear_iterations", 0),
                        )
                    ),
                    initial_residual_norm=float("inf"),
                    residual_norm=float("inf"),
                    residual_history=(),
                    time_s=history.time_s,
                    step_index=step,
                )
                history.reject(
                    rejected,
                    residual_evaluations=int(work["residual_evaluations"]),
                    jacobian_assemblies=int(work["jacobian_assemblies"]),
                    linear_solves=int(work["linear_solves"]),
                )
                frac *= 0.5
                if frac < min_frac - 1e-15:
                    _raise_nfvm_convergence(
                        object_name=fname,
                        reason="invalid_state" if reason == "saturation" else "max_iterations",
                        iterations=max_substeps,
                        max_iterations=max_substeps,
                        residual_norm=float("inf"),
                        tolerance=0.0,
                        step_index=step,
                        message=(
                            f"{fname}: {msg[reason]} after {max_substeps} "
                            "adaptive sub-step halvings"
                        ),
                    )
        state = sub
        history.record_output(state)
    return history.stack()


# Cycle-safe tail re-export: _transient_thermal back-imports the Newton/
# adaptive infrastructure above, so this import must run after it exists.
from geobrain.physics.flow.discretization.nfvm._transient_thermal import (  # noqa: E402,F401  cycle-safe tail re-export
    nfvm_thermal_compositional,
    nfvm_thermal_single_phase,
    nfvm_thermal_two_phase,
)
