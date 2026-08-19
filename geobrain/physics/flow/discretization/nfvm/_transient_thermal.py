"""NFVM thermal transient marches: single-phase, two-phase and
compositional thermal solvers (split from transient.py; shares the
Newton/adaptive infrastructure defined there).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from __future__ import annotations
from collections.abc import Sequence
import torch
from .....core import GeoBrainError
from ...compositional.cubic_eos import CubicEOS
from ...config import FlowHistoryConfig
from ...errors import FlowContractError, FlowConvergenceError
from ...properties.relperm import RelPerm
from ..flux import scatter_boundary_outflow
from .kernel import (
    NFVMGeometry,
    _build_nfvm,
    _extend_to_ghosts,
    _onesided,
    nfvm_flux,
)
from geobrain.physics.flow.discretization.nfvm.transient import (
    AdaptiveOutcome,
    CompositionalMultiperfWell,
    CompositionalNeumann,
    CompositionalRateWell,
    CompositionalWell,
    Scalar,
    SinglePhaseMultiperfWell,
    SinglePhaseNeumann,
    SinglePhaseRateWell,
    SinglePhaseWell,
    TensorInput,
    ThermalTwoPhaseMultiperfWell,
    ThermalTwoPhaseNeumann,
    ThermalTwoPhaseRateWell,
    ThermalTwoPhaseWell,
    WorkCounters,
    _NFVMTensorHistory,
    _SAT_BOUND_TOL,
    _adaptive_march,
    _boundary_divergence,
    _face_divergence,
    _ift_attach,
    _newton_solve,
    _raise_nfvm_convergence,
    _validate_newton_controls,
)


def nfvm_thermal_single_phase(
    geom: NFVMGeometry,
    lam_bulk: Scalar,
    T0: torch.Tensor,
    p0: Scalar,
    *,
    source_mass: torch.Tensor | None = None,
    source_energy: torch.Tensor | None = None,
    rho_ref: float = 1000.0,
    mu: float = 1e-3,
    c_f: float = 1e-9,
    p_ref: float = 1e7,
    cp_f: float = 4200.0,
    cp_r: float = 1000.0,
    rho_r: float = 2650.0,
    T_ref: float = 300.0,
    alpha_T: float = 0.0,
    depth: torch.Tensor | None = None,
    gravity: float = 9.81,
    wells: Sequence[SinglePhaseWell] = (),
    rate_wells: Sequence[SinglePhaseRateWell] = (),
    multiperf_wells: Sequence[SinglePhaseMultiperfWell] = (),
    neumann: Sequence[SinglePhaseNeumann] = (),
    T_bc: TensorInput | None = None,
    depth_bc: TensorInput | None = None,
    phi: Scalar = 0.2,
    V: Scalar = 1.0,
    dt: float = 8.64e4,
    nsteps: int = 10,
    scheme: str = "nmpfa",
    tol: float = 1e-8,
    max_iter: int = 40,
    history_config: FlowHistoryConfig | None = None,
) -> torch.Tensor:
    """Single-phase **thermal** flow fully on the NFVM monotone flux, coupled
    mass + energy, the thermal analog of :func:`nfvm_two_phase`.

    Both transport operators use the monotone NTPFA/NMPFA two-point flux: the Darcy
    mass flux ``G_p = nfvm_flux(p)`` from ``geom`` (whose ``K`` is the permeability),
    and the Fourier conduction ``F_cond = nfvm_flux(T)`` from a sibling geometry whose
    ``K`` is the bulk thermal conductivity ``lam_bulk·I`` (same cells/faces, built
    internally). The energy flux is the advected specific enthalpy ``H_α|_up·F_mass``
    (``H = C_f T + p/ρ``, upwinded on the mass-flux sign, the same monotone ``G_p``)
    plus the monotone conduction. Per cell::

        mass:    V·φ·(ρ − ρ|old)/Δt + Σ_f (±F_mass) − q^m = 0
        energy:  V·(E − E|old)/Δt + Σ_f (±(H|_up F_mass + F_cond)) − q^e = 0
        E = (1−φ)ρ_r C_r T + φ ρ C_f T ,  ρ(p,T) = ρ_ref(1 + c_f(p−p_ref) − α_T(T−T_ref))

    Because both fluxes are monotone, temperature respects the discrete maximum
    principle on anisotropic / skewed grids where the linear MPFA-O thermal flux
    overshoots; on a K-orthogonal grid the scheme collapses to two-point and
    reproduces the TPFA thermal solution (≡ :class:`MPFAThermalSinglePhaseModel`).

    ``geom`` a :class:`NFVMGeometry` with ``K`` = permeability and **no Dirichlet
    ghosts** (no-flow mass + insulated heat, closed domain driven by the optional
    per-cell ``source_mass`` / ``source_energy``). Slight compressibility ``c_f``
    anchors the otherwise-singular pressure. Returns the ``(nsteps+1, 2·n_cells)``
    state history (each row ``[p ; T]``). **Differentiable through time** in ``perm``
    (``geom.K``) and the conductivity (``lam_bulk``): each converged step is reattached
    with its exact implicit-function-theorem VJP (:func:`_ift_attach`) and steps chain
    without detaching, so the multi-step history gradient is the exact through-time
    gradient at one extra Jacobian + linear solve per step (bounded memory, no Newton
    unrolling; FD-pinned by the NFVM gradient suite). Fails loud
    (raises) if the nonlinear monotone
    Newton stalls on a stiff step rather than returning a non-converged state. Optional
    **gravity**: pass ``depth`` (``(n,)`` +down [m]) for a hydrostatic head
    ``Φ = p − ρ g·depth`` (``gravity`` = g); ``depth=None`` ⇒ potential ``p``. Optional
    **BHP wells** ``wells = [(cell, WI, bhp, T_inj), ...]`` (produce the cell enthalpy,
    inject at ``T_inj``). **Rate wells** ``rate_wells = [(cell, WI, q, T_inj[, bhp_limit]), ...]``
    prescribe a volumetric rate (``q ≥ 0`` produces, ``q < 0`` injects), switching to BHP control
    if the implied bhp ``p − q·μ/WI`` would pass ``bhp_limit``. **Multi-perforation**
    ``multiperf_wells = [(perfs, q_target, T_inj), ...]`` (``perfs = [(cell, WI), ...]``) share one
    bhp from the total volumetric-rate constraint. **Dirichlet (p, T) boundaries**: build ``geom`` with mirror ghosts
    (``geom.ghost_p`` = the fixed pressures) and pass ``T_bc`` (the fixed temperatures, aligned
    with ``geom.ghost_p``); each ghost is a fixed-(p, T) reservoir the cell exchanges mass +
    enthalpy + heat with. Like :func:`solve_nfvm`, the boundary value is anchored at the mirror
    ghost (≈ half a cell beyond the face), a discretisation-convention difference from the MPFA
    bc-at-face. **Neumann** ``neumann = [(cell, q, T_inj), ...]`` prescribes a volumetric rate
    (``q ≥ 0`` removes the cell fluid, ``q < 0`` injects at ``T_inj``) as a source term.
    **Gravity with Dirichlet**: pass ``depth_bc`` (the ghost depths, aligned with ``geom.ghost_p``)
    so the ghost head carries the hydrostatic term ``Φ_bc = p_bc − ρ_bc g D_bc`` (a hydrostatic
    column then stays in exact equilibrium)."""
    max_iter, tol = _validate_newton_controls(
        max_iter=max_iter,
        tol=tol,
        object_name="nfvm_thermal_single_phase",
    )
    n = geom.n
    dtype = geom._permeability_view().dtype
    dim = geom._coordinates_view().shape[1]
    lam = (
        lam_bulk
        if isinstance(lam_bulk, torch.Tensor)
        else torch.full((n,), float(lam_bulk), dtype=dtype)
    )
    if lam.numel() == 1:  # 0-dim / length-1 conductivity → per-cell
        lam = lam.reshape(()).expand(n)  # (autograd-safe; reshape later copies the view)
    eye = torch.eye(dim, dtype=dtype)
    # conduction geometry: same cells/faces, K = bulk thermal conductivity (insulated)
    geom_cond = NFVMGeometry(
        geom._coordinates_view(),
        _extend_to_ghosts(lam, geom).reshape(-1, 1, 1) * eye,
        [list(records) for records in geom._face_records_view()],
        [],
        n,
    )
    interior_p, boundary_p = _build_nfvm(geom, scheme)
    interior_T, boundary_T = _build_nfvm(geom_cond, scheme)
    ghost_pressures = geom._ghost_pressures_view()
    p_bc = torch.stack(ghost_pressures) if ghost_pressures else None
    T_boundary: torch.Tensor | None = None
    depth_boundary: torch.Tensor | None = None
    if p_bc is not None:  # fixed-(p, T) boundaries
        if T_bc is None:
            raise GeoBrainError(
                "nfvm_thermal_single_phase: Dirichlet ghosts present (geom.ghost_p) but T_bc not given; "
                "a fixed-pressure boundary also needs a fixed temperature.",
                object_name="nfvm_thermal_single_phase",
                field="T_bc",
            )
        T_boundary = torch.as_tensor(T_bc, dtype=dtype)
        if depth is not None:  # gravity ⇒ the ghost potential carries a
            if depth_bc is None:  # head Φ_bc = p_bc − ρ_bc g D_bc, so the
                raise GeoBrainError(  # ghost needs its own depth (mirror point)
                    "nfvm_thermal_single_phase: gravity (depth) with Dirichlet ghosts needs depth_bc "
                    "(the ghost depths, aligned with geom.ghost_p).",
                    object_name="nfvm_thermal_single_phase",
                    field="depth_bc",
                )
            depth_boundary = (
                depth_bc.to(dtype)
                if isinstance(depth_bc, torch.Tensor)
                else torch.as_tensor(depth_bc, dtype=dtype)
            )
            if tuple(depth_boundary.shape) != (
                p_bc.shape[0],
            ):  # a wrong length would silently broadcast /
                raise GeoBrainError(  # cat-mismatch instead of a clean error
                    "nfvm_thermal_single_phase: depth_bc must be a per-ghost field aligned with geom.ghost_p.",
                    object_name="nfvm_thermal_single_phase",
                    field="depth_bc",
                    expected=f"({p_bc.shape[0]},)",
                    actual=tuple(depth_boundary.shape),
                )
    Vc = V if isinstance(V, torch.Tensor) else torch.full((n,), float(V), dtype=dtype)
    phic = phi if isinstance(phi, torch.Tensor) else torch.full((n,), float(phi), dtype=dtype)
    dep = depth.to(dtype) if depth is not None else None  # (n,) +down [m] ⇒ hydrostatic head
    if dep is not None and tuple(dep.shape) != (n,):  # a 0-dim / length-1 depth would
        raise GeoBrainError(  # silently broadcast to zero head
            "nfvm_thermal_single_phase: depth must be a per-cell (n,) field (+down).",
            object_name="nfvm_thermal_single_phase",
            field="depth",
            expected=f"({n},)",
            actual=tuple(dep.shape),
        )

    # multi-perforation wells: perforations [(cell, WI), ...] share one bhp from Σ_k WI_k (1/μ)
    # (p_k − bhp) = q_target (single-phase total volumetric rate). The total mobility 1/μ is the
    # same at every perforation, so the bhp is closed-form with no cross-flow fixed point; the
    # per-perforation enthalpy still upwinds (cell T on production, T_inj on injection).
    mp_cell_ids: list[int] = []
    mp_wi_values: list[Scalar] = []
    mp_well_ids: list[int] = []
    mp_specs = list(multiperf_wells)
    for wi, (perfs, _q, _tinj) in enumerate(mp_specs):
        for cell, WI in perfs:
            mp_cell_ids.append(int(cell))
            mp_wi_values.append(WI)
            mp_well_ids.append(wi)
    mp_pc: torch.Tensor | None
    if mp_cell_ids:
        mp_pc = torch.tensor(mp_cell_ids, dtype=torch.long)
        mp_pwi = torch.stack([torch.as_tensor(x, dtype=dtype) for x in mp_wi_values])
        mp_pw = torch.tensor(mp_well_ids, dtype=torch.long)
        mp_q = torch.stack([torch.as_tensor(s[1], dtype=dtype) for s in mp_specs])
        mp_tinj = torch.stack([torch.as_tensor(s[2], dtype=dtype) for s in mp_specs])
        mp_nw = len(mp_specs)
    else:
        mp_pc = None

    def rho(p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return rho_ref * (1.0 + c_f * (p - p_ref) - alpha_T * (T - T_ref))

    def residual(
        state: torch.Tensor, state_old: torch.Tensor, mu_half: bool = False
    ) -> torch.Tensor:
        p, T = state[:n], state[n:]
        p_o, T_o = state_old[:n], state_old[n:]
        r, r_o = rho(p, T), rho(p_o, T_o)
        E = (1.0 - phic) * rho_r * cp_r * T + phic * r * cp_f * T
        E_o = (1.0 - phic) * rho_r * cp_r * T_o + phic * r_o * cp_f * T_o
        H = cp_f * T + p / r  # specific enthalpy
        pot = p - r * gravity * dep if dep is not None else p  # Φ = p − ρ g D (hydrostatic head)
        if p_bc is not None:  # Dirichlet ghosts: an interior cell's MPFA
            assert T_boundary is not None
            p_aug = torch.cat([p, p_bc])  # stencil near the boundary references the
            T_aug = torch.cat([T, T_boundary])  # ghost, so every field lookup uses the
            r_aug = rho(p_aug, T_aug)  # augmented (cell+ghost) arrays
            H_aug = cp_f * T_aug + p_aug / r_aug
            if dep is not None:
                assert depth_boundary is not None
                pot_aug = p_aug - r_aug * gravity * torch.cat([dep, depth_boundary])
            else:
                pot_aug = p_aug
        else:
            pot_aug, T_aug, r_aug, H_aug = pot, T, r, H
        R_m = Vc * phic * (r - r_o) / dt
        R_e = Vc * (E - E_o) / dt
        for (Lp, Rp), (Lt, Rt) in zip(interior_p, interior_T):
            # Monotone Darcy flux: Φ[left] > Φ[right] means G > 0 (left→right).
            G = nfvm_flux(pot_aug, Lp, Rp, scheme, mu_half=mu_half)
            left, right = Lp["left"], Lp["right"]
            up = float(G.detach()) >= 0
            Fm = (r_aug[left] if up else r_aug[right]) / mu * G
            Hup = H_aug[left] if up else H_aug[right]
            Fc = nfvm_flux(T_aug, Lt, Rt, scheme, mu_half=mu_half)  # left→right
            Fe = Hup * Fm + Fc
            R_m = R_m + _face_divergence(Fm, left, right, n)
            R_e = R_e + _face_divergence(Fe, left, right, n)
        for cell, WI, bhp, T_inj in wells:  # BHP wells (mass + energy): produce at the
            dpw = p[cell] - bhp  # cell enthalpy, inject at T_inj
            T_w = T[cell] if float(dpw.detach()) >= 0 else torch.as_tensor(T_inj, dtype=dtype)
            rww = rho(p[cell], T_w)
            Fm = WI * rww / mu * dpw
            Hw = cp_f * T_w + p[cell] / rww
            R_m = R_m + _boundary_divergence(Fm, cell, n)
            R_e = R_e + _boundary_divergence(Hw * Fm, cell, n)
        for w in rate_wells:  # rate-controlled wells: prescribed volumetric
            cell, WI, q, T_inj = (
                w[0],
                w[1],
                w[2],
                w[3],
            )  # rate q; optional bhp_limit ⇒ BHP control switch
            qb = torch.as_tensor(q, dtype=dtype)
            prod = float(qb.detach()) >= 0
            T_w = T[cell] if prod else torch.as_tensor(T_inj, dtype=dtype)
            rww = rho(p[cell], T_w)
            viol = False
            if len(w) >= 5 and w[4] is not None:  # None ⇒ no limit (matches the MPFA _has contract)
                lim = torch.as_tensor(w[4], dtype=dtype)
                bhp_rate = p[cell] - qb * mu / WI  # single-phase total mobility = 1/μ
                viol = bool(bhp_rate.detach() < lim) if prod else bool(bhp_rate.detach() > lim)
            Fm = WI * rww / mu * (p[cell] - lim) if viol else rww * qb
            Hw = cp_f * T_w + p[cell] / rww
            R_m = R_m + _boundary_divergence(Fm, cell, n)
            R_e = R_e + _boundary_divergence(Hw * Fm, cell, n)
        if mp_pc is not None:  # multi-perforation wells (shared bhp)
            pc, pw, pWI = mp_pc, mp_pw, mp_pwi
            WL = pWI / mu  # single-phase total mobility 1/μ (uniform)
            num = p.new_zeros(mp_nw).scatter_add(0, pw, WL * p[pc])
            den = p.new_zeros(mp_nw).scatter_add(0, pw, WL)
            bhp = (num - mp_q) / den.clamp_min(
                1e-30
            )  # closed-form (no cross-flow fixed point: Λ uniform)
            dp = p[pc] - bhp[pw]
            up = dp.detach() >= 0  # produce (cell T) / inject (T_inj)
            T_w = torch.where(up, T[pc], mp_tinj[pw])
            rwk = rho(p[pc], T_w)
            Fm = pWI * rwk / mu * dp
            R_m = R_m + scatter_boundary_outflow(Fm, pc, n)
            R_e = R_e + scatter_boundary_outflow((cp_f * T_w + p[pc] / rwk) * Fm, pc, n)
        if p_bc is not None:  # Dirichlet (p, T) ghosts: one-sided cell→ghost
            for Lp_b, Lt_b in zip(boundary_p, boundary_T):
                cell, ghost = Lp_b["left"], Lp_b["right"]
                G = -_onesided(pot_aug, Lp_b)[0]  # match interior: G = −(one-sided flux)
                up = float(G.detach()) >= 0  # G > 0 ⇒ p_cell > p_bc ⇒ outflow (use cell)
                Fm = (r_aug[cell] if up else r_aug[ghost]) / mu * G
                Hup = H_aug[cell] if up else H_aug[ghost]
                Fc = -_onesided(T_aug, Lt_b)[0]  # one-sided conduction flux to the T_bc ghost
                R_m = R_m + _boundary_divergence(Fm, cell, n)
                R_e = R_e + _boundary_divergence(Hup * Fm + Fc, cell, n)
        for cell, q, T_inj in neumann:  # prescribed-rate (Neumann) source: q ≥ 0
            qb = torch.as_tensor(q, dtype=dtype)  # removes the cell fluid, q < 0 injects at T_inj
            T_b = T[cell] if float(qb.detach()) >= 0 else torch.as_tensor(T_inj, dtype=dtype)
            rb = rho(p[cell], T_b)
            Fm = rb * qb  # q volumetric ⇒ ρ·q mass rate
            Hb = cp_f * T_b + p[cell] / rb
            R_m = R_m + _boundary_divergence(Fm, cell, n)
            R_e = R_e + _boundary_divergence(Hb * Fm, cell, n)
        if source_mass is not None:
            R_m = R_m - source_mass
        if source_energy is not None:
            R_e = R_e - source_energy
        return torch.cat([R_m, R_e])

    inf = float("inf")
    # The mass (~φ·c·Δp) and energy (~ρ_C·ΔT) residuals differ in scale by ~1e8, so a
    # single inf-norm is all energy and under-resolves pressure. Normalise each block by
    # a FIXED capacity rate (CNV/MB-style): pore-volume mass rate and heat-capacity rate.
    # Fixed scales never collapse to round-off (unlike a step-start residual when a block
    # is momentarily balanced), which would otherwise pin the criterion and starve the
    # other block. ``scaled(r) = max(|R_m|∞/s_m, |R_e|∞/s_e)`` is then a dimensionless
    # mis-balance fraction; converge when it is below ``tol``.
    T_char = float(T0.abs().max()) + 1e-30
    rhoC_char = float((1.0 - phic).mean()) * rho_r * cp_r + float(phic.mean()) * rho_ref * cp_f
    s_m = float((Vc * phic).max()) * rho_ref / float(dt) + 1e-300
    s_e = float(Vc.max()) * rhoC_char * T_char / float(dt) + 1e-300

    def scaled(r: torch.Tensor) -> float:  # scalar convergence/line-search test
        r = r.detach()  # only, never on the state's autograd path
        return max(
            float(torch.linalg.vector_norm(r[:n], ord=inf)) / s_m,
            float(torch.linalg.vector_norm(r[n:], ord=inf)) / s_e,
        )

    state = torch.cat(
        [
            geom._permeability_view().new_full((n,), float(p0))
            if not isinstance(p0, torch.Tensor)
            else p0.to(dtype),
            T0.to(dtype),
        ]
    )
    history = _NFVMTensorHistory(state, nsteps=nsteps, dt_s=dt, config=history_config)
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
            return _newton_solve(jacobian, residual_value, "nfvm_thermal_single_phase")

        prev = state  # attached, the through-time link
        old = prev.detach()
        with torch.no_grad():  # grad-free march; gradient from _ift_attach
            s = old.clone()
            r0 = evaluate(s, old, mu_half=True)  # avgMPFA (linear) init
            J0 = assemble(s, old, mu_half=True)
            s = s - solve_linear(J0, r0)
            converged = False
            for _it in range(max_iter):
                r = evaluate(s, old)
                rn = scaled(r)
                if rn < tol:  # both blocks converged (capacity-normalised)
                    converged = True
                    break
                J = assemble(s, old)
                d = solve_linear(J, r)
                alpha, found = 1.0, False
                for _ls in range(25):  # backtracking on the scaled norm
                    if scaled(evaluate(s - alpha * d, old)) < rn:
                        found = True
                        break
                    alpha *= 0.5
                if not found:  # no descent possible (Newton stalled)
                    break
                s = s - alpha * d
        if not converged:  # fail loud; do not silently return a
            _raise_nfvm_convergence(
                object_name="nfvm_thermal_single_phase",
                reason="line_search" if not found else "max_iterations",
                iterations=max_iter,
                max_iterations=max_iter,
                residual_norm=rn,
                tolerance=tol,
                step_index=step,
                message="NFVM thermal single-phase Newton solve did not converge",
            )
        state = _ift_attach(
            s,
            lambda q: residual(q, prev),
            "nfvm_thermal_single_phase",
            work=work,
        )
        history.append(
            state,
            residual_evaluations=work["residual_evaluations"],
            jacobian_assemblies=work["jacobian_assemblies"],
            linear_solves=work["linear_solves"],
        )
    return history.stack()


def nfvm_thermal_two_phase(
    geom: NFVMGeometry,
    relperm: RelPerm,
    lam_rock: Scalar,
    T0: torch.Tensor,
    sw0: Scalar,
    p0: Scalar,
    *,
    wells: Sequence[ThermalTwoPhaseWell] = (),
    rate_wells: Sequence[ThermalTwoPhaseRateWell] = (),
    multiperf_wells: Sequence[ThermalTwoPhaseMultiperfWell] = (),
    neumann: Sequence[ThermalTwoPhaseNeumann] = (),
    sw_bc: TensorInput | None = None,
    T_bc: TensorInput | None = None,
    depth_bc: TensorInput | None = None,
    source_water: torch.Tensor | None = None,
    source_oil: torch.Tensor | None = None,
    source_energy: torch.Tensor | None = None,
    rho_w_ref: float = 1000.0,
    rho_o_ref: float = 800.0,
    mu_w: float = 1e-3,
    mu_o: float = 2e-3,
    c_w: float = 0.0,
    c_o: float = 0.0,
    p_ref: float = 1e7,
    alpha_w: float = 0.0,
    alpha_o: float = 0.0,
    cp_w: float = 4184.0,
    cp_o: float = 2000.0,
    cp_r: float = 1000.0,
    rho_r: float = 2650.0,
    lam_w: float = 0.6,
    lam_o: float = 0.15,
    T_ref: float = 300.0,
    depth: torch.Tensor | None = None,
    gravity: float = 9.81,
    phi: Scalar = 0.2,
    V: Scalar = 1.0,
    dt: float = 8.64e4,
    nsteps: int = 10,
    scheme: str = "nmpfa",
    tol: float = 1e-8,
    max_iter: int = 40,
    max_substeps: int = 0,
    history_config: FlowHistoryConfig | None = None,
) -> torch.Tensor:
    """Oil-water two-phase **thermal** flow fully on the NFVM monotone flux, coupled
    per-phase mass + energy, the two-phase analog of :func:`nfvm_thermal_single_phase`.

    Three monotone two-point operators, all built from the same cells/faces: the Darcy
    flux ``G_p = nfvm_flux(p)`` (``geom.K`` = permeability) and a **parallel-conductance**
    Fourier flux = a rock stencil ``nfvm_flux(T)`` with ``K = (1−φ)λ_rock·I`` plus a
    porosity-weighted geometric stencil ``nfvm_flux(T)`` with ``K = φ·I`` scaled by the
    face-averaged fluid conductivity ``λ_w S̄_w + λ_o S̄_o`` (exactly
    as :class:`MPFAThermalTwoPhaseModel`). Each phase mass flux
    ``F_α = (ρ_α k_rα/μ_α)|_up·G_p`` is phase-potential upwinded on ``G_p``; the energy
    flux is the advected per-phase enthalpy ``Σ_α H_α|_up F_α`` (``H_α = C_α T + p/ρ_α``,
    same upwind) plus the conduction. The monotone two-point flux keeps the saturation in
    ``[0,1]`` and the temperature within its source/initial range on **moderate**
    anisotropic / skewed grids; on a K-orthogonal grid the scheme collapses to two-point
    and reproduces :class:`MPFAThermalTwoPhaseModel` to machine precision. Two failure
    modes are guarded **fail-loud** (raise rather than return a wrong state): (a) the
    coupled monotone Newton may not converge under strong anisotropy / strong-well
    displacement; (b) because the accumulation keeps the **raw** ``S_w`` (to hold the
    Jacobian diagonal alive at saturation, unlike the clamped MPFA model), a rate that
    over-injects beyond the drainable pore volume converges to a non-physical ``S_w > 1``;
    this is caught by a post-convergence bound check (a mild ``<= 1%`` excursion on a
    strongly anisotropic two-point-fallback face is tolerated). Reduce ``dt`` / the rate /
    the anisotropy if it raises, or pass ``max_substeps > 0`` to **auto-subdivide** a failed
    macro step (adaptive dt sub-stepping / pseudo-transient continuation): a transient
    overshoot at a large ``dt`` is then resolved by sub-steps, while a genuine over-injection
    (beyond the pore volume) still fails loud even at the finest sub-step.

    ``geom`` a :class:`NFVMGeometry` with ``K`` = permeability; no Dirichlet ghosts ⇒ closed
    domain (no-flow + insulated). **Dirichlet (p, sw, T) boundaries**: build ``geom`` with
    mirror ghosts (``geom.ghost_p`` = the fixed pressures) and pass ``sw_bc`` / ``T_bc`` (the
    fixed saturations / temperatures, aligned with ``geom.ghost_p``), like :func:`solve_nfvm`
    the boundary value is anchored at the mirror ghost (≈ half a cell beyond the face), so this
    does **not** reproduce :class:`MPFAThermalTwoPhaseModel` ``dirichlet=…`` to machine
    precision (a boundary half-distance convention difference, not an error). Pass ``depth_bc``
    (ghost depths) for **gravity with Dirichlet**, the per-phase ghost head ``Φ_α,bc = p_bc −
    ρ_α,bc g D_bc``. **Neumann** ``neumann = [(cell, q, sw_inj, T_inj), ...]`` prescribes
    a total volumetric rate split by fractional flow (``q ≥ 0`` removes the cell fluid, ``q < 0``
    injects ``sw_inj`` at ``T_inj``) as a source term, a pure source, so it reproduces the MPFA
    Neumann to machine precision. ``relperm`` a :class:`RelPerm`; ``wells`` a list
    ``(cell, WI, bhp, inj_sw, T_inj)`` of BHP wells (each carrying mass + energy); ``rate_wells``
    ``[(cell, WI, q, inj_sw, T_inj[, bhp_limit])]`` are rate-controlled (total volumetric rate split
    by fractional flow, switching to BHP control past ``bhp_limit``); ``multiperf_wells``
    ``[(perfs, q_target, inj_sw, T_inj, mode), ...]`` (``perfs = [(cell, WI), ...]``,
    ``mode ∈ {reservoir, surface}``) are multi-perforation wells whose perforations share one bhp
    solved from the total-rate constraint, with cross-flow handled per perforation. State
    ``[p, S_w, T]``; returns the ``(nsteps+1, 3·n_cells)`` history. **Differentiable
    through time** in ``perm`` (``geom.K``) and ``λ_rock``: every accepted (sub-)step is
    reattached with its exact implicit-function-theorem VJP (:func:`_ift_attach`) and the
    chain is never detached; the multi-step history gradient is the exact through-time
    gradient (FD-pinned by the NFVM gradient suite); see
    :func:`nfvm_thermal_single_phase`. Raw ``S_w`` (relperm clamps internally)
    keeps the accumulation diagonal alive at saturation. Optional **gravity**: pass
    ``depth`` (``(n,)`` +down [m]) to give each phase its own potential
    ``Φ_α = p − ρ_α g·depth`` (``gravity`` = g), so buoyancy drives water/oil segregation
    (each phase's flux + enthalpy upwind independently); ``depth=None`` ⇒ shared potential
    ``p`` (no buoyancy). v1: closed domain + sources/wells; no capillarity."""
    max_iter, tol = _validate_newton_controls(
        max_iter=max_iter,
        tol=tol,
        object_name="nfvm_thermal_two_phase",
    )
    if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 0:
        raise FlowContractError(
            "max_substeps must be a non-negative integer",
            object_name="nfvm_thermal_two_phase",
            field="max_substeps",
            expected="integer >= 0",
            actual=max_substeps,
        )
    n = geom.n
    dtype = geom._permeability_view().dtype
    dim = geom._coordinates_view().shape[1]
    eye = torch.eye(dim, dtype=dtype)
    lr = (
        lam_rock
        if isinstance(lam_rock, torch.Tensor)
        else torch.full((n,), float(lam_rock), dtype=dtype)
    )
    phic = phi if isinstance(phi, torch.Tensor) else torch.full((n,), float(phi), dtype=dtype)
    Vc = V if isinstance(V, torch.Tensor) else torch.full((n,), float(V), dtype=dtype)
    dep = depth.to(dtype) if depth is not None else None  # (n,) +down [m] ⇒ buoyancy
    if dep is not None and tuple(dep.shape) != (n,):  # a 0-dim / length-1 depth would
        raise GeoBrainError(  # silently broadcast to zero buoyancy
            "nfvm_thermal_two_phase: depth must be a per-cell (n,) field (+down).",
            object_name="nfvm_thermal_two_phase",
            field="depth",
            expected=f"({n},)",
            actual=tuple(dep.shape),
        )
    face_records = [list(records) for records in geom._face_records_view()]
    geom_rock = NFVMGeometry(
        geom._coordinates_view(),
        _extend_to_ghosts((1.0 - phic) * lr, geom).reshape(-1, 1, 1) * eye,
        face_records,
        [],
        n,
    )
    geom_phi = NFVMGeometry(
        geom._coordinates_view(),
        _extend_to_ghosts(phic, geom).reshape(-1, 1, 1) * eye,
        face_records,
        [],
        n,
    )
    interior_p, boundary_p = _build_nfvm(geom, scheme)
    interior_rock, boundary_rock = _build_nfvm(geom_rock, scheme)
    interior_phi, boundary_phi = _build_nfvm(geom_phi, scheme)
    ghost_pressures = geom._ghost_pressures_view()
    p_bc = torch.stack(ghost_pressures) if ghost_pressures else None
    sw_boundary: torch.Tensor | None = None
    T_boundary: torch.Tensor | None = None
    depth_boundary: torch.Tensor | None = None
    if p_bc is not None:  # fixed-(p, sw, T) boundaries
        if sw_bc is None or T_bc is None:
            raise GeoBrainError(
                "nfvm_thermal_two_phase: Dirichlet ghosts present (geom.ghost_p) but sw_bc / T_bc not given; "
                "a fixed-pressure boundary also needs a fixed saturation and temperature.",
                object_name="nfvm_thermal_two_phase",
                field="sw_bc/T_bc",
            )
        sw_boundary = torch.as_tensor(sw_bc, dtype=dtype)
        T_boundary = torch.as_tensor(T_bc, dtype=dtype)
        if depth is not None:  # gravity ⇒ per-phase ghost head
            if depth_bc is None:  # Φ_α,bc = p_bc − ρ_α,bc g D_bc
                raise GeoBrainError(
                    "nfvm_thermal_two_phase: gravity (depth) with Dirichlet ghosts needs depth_bc "
                    "(the ghost depths, aligned with geom.ghost_p).",
                    object_name="nfvm_thermal_two_phase",
                    field="depth_bc",
                )
            depth_boundary = (
                depth_bc.to(dtype)
                if isinstance(depth_bc, torch.Tensor)
                else torch.as_tensor(depth_bc, dtype=dtype)
            )
            if tuple(depth_boundary.shape) != (
                p_bc.shape[0],
            ):  # a wrong length would silently broadcast /
                raise GeoBrainError(  # cat-mismatch instead of a clean error
                    "nfvm_thermal_two_phase: depth_bc must be a per-ghost field aligned with geom.ghost_p.",
                    object_name="nfvm_thermal_two_phase",
                    field="depth_bc",
                    expected=f"({p_bc.shape[0]},)",
                    actual=tuple(depth_boundary.shape),
                )

    # multi-perforation wells: perforations [(cell, WI), ...] SHARE one bhp, solved closed-form from
    # Σ_k WI_k Λ_k (p_k − bhp) = q_target (reservoir Λ = λ_w+λ_o / surface Σ_α ρ_α λ_α/ρ_α,ref).
    mp_cell_ids: list[int] = []
    mp_wi_values: list[Scalar] = []
    mp_well_ids: list[int] = []
    mp_specs = list(multiperf_wells)
    for wi, (perfs, _q, _isw, _tinj, mode) in enumerate(mp_specs):
        if mode not in ("reservoir", "surface"):
            raise GeoBrainError(
                "nfvm_thermal_two_phase: multiperf well mode must be 'reservoir' or 'surface'.",
                object_name="nfvm_thermal_two_phase",
                field="mode",
                actual=mode,
            )
        for cell, WI in perfs:
            mp_cell_ids.append(int(cell))
            mp_wi_values.append(WI)
            mp_well_ids.append(wi)
    mp_pc: torch.Tensor | None
    if mp_cell_ids:
        mp_pc = torch.tensor(mp_cell_ids, dtype=torch.long)
        mp_pwi = torch.stack([torch.as_tensor(x, dtype=dtype) for x in mp_wi_values])
        mp_pw = torch.tensor(mp_well_ids, dtype=torch.long)
        mp_q = torch.stack([torch.as_tensor(s[1], dtype=dtype) for s in mp_specs])
        mp_isw = torch.stack([torch.as_tensor(s[2], dtype=dtype) for s in mp_specs])
        mp_tinj = torch.stack([torch.as_tensor(s[3], dtype=dtype) for s in mp_specs])
        mp_surf = torch.tensor([s[4] == "surface" for s in mp_specs])
        mp_nw = len(mp_specs)
    else:
        mp_pc = None

    def rho_w(p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return rho_w_ref * (1.0 + c_w * (p - p_ref) - alpha_w * (T - T_ref))

    def rho_o(p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return rho_o_ref * (1.0 + c_o * (p - p_ref) - alpha_o * (T - T_ref))

    def residual(
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt_arg: float,
        mu_half: bool = False,
    ) -> torch.Tensor:
        p, sw, T = state[:n], state[n : 2 * n], state[2 * n :]
        p_o, sw_o, T_o = state_old[:n], state_old[n : 2 * n], state_old[2 * n :]
        so, so_o = 1.0 - sw, 1.0 - sw_o
        rw, ro = rho_w(p, T), rho_o(p, T)
        rw_o, ro_o = rho_w(p_o, T_o), rho_o(p_o, T_o)
        E = (1.0 - phic) * rho_r * cp_r * T + phic * (rw * sw * cp_w + ro * so * cp_o) * T
        E_o = (1.0 - phic) * rho_r * cp_r * T_o + phic * (
            rw_o * sw_o * cp_w + ro_o * so_o * cp_o
        ) * T_o
        H_w, H_o = cp_w * T + p / rw, cp_o * T + p / ro
        mm_w, mm_o = rw * relperm.kr_water(sw) / mu_w, ro * relperm.kr_oil(sw) / mu_o
        # per-phase potential Φ_α = p − ρ_α g D (D = depth, +down); buoyancy can flip the
        # two phases' flux directions independently (heavy water sinks, light oil rises). No
        # gravity ⇒ Φ_w = Φ_o = p ⇒ both phases share one flux (byte-identical to v1).
        phi_w_pot = p - rw * gravity * dep if dep is not None else p
        phi_o_pot = p - ro * gravity * dep if dep is not None else p
        if p_bc is not None:  # Dirichlet ghosts: an interior cell's MPFA
            assert sw_boundary is not None
            assert T_boundary is not None
            p_aug = torch.cat([p, p_bc])  # stencil near the boundary references the
            sw_aug = torch.cat([sw, sw_boundary])  # ghost, so every field lookup uses the
            T_aug = torch.cat([T, T_boundary])  # augmented (cell+ghost) arrays
            so_aug = 1.0 - sw_aug
            rw_aug, ro_aug = rho_w(p_aug, T_aug), rho_o(p_aug, T_aug)
            mm_w_aug = rw_aug * relperm.kr_water(sw_aug) / mu_w
            mm_o_aug = ro_aug * relperm.kr_oil(sw_aug) / mu_o
            H_w_aug, H_o_aug = cp_w * T_aug + p_aug / rw_aug, cp_o * T_aug + p_aug / ro_aug
            if dep is not None:  # per-phase ghost head (buoyancy)
                assert depth_boundary is not None
                D_aug = torch.cat([dep, depth_boundary])
                pot_w_aug = p_aug - rw_aug * gravity * D_aug
                pot_o_aug = p_aug - ro_aug * gravity * D_aug
            else:
                pot_w_aug = pot_o_aug = p_aug
        else:
            sw_aug, so_aug, T_aug = sw, so, T
            mm_w_aug, mm_o_aug, H_w_aug, H_o_aug = mm_w, mm_o, H_w, H_o
            pot_w_aug, pot_o_aug = phi_w_pot, phi_o_pot
        R_w = Vc * phic * (rw * sw - rw_o * sw_o) / dt_arg
        R_o = Vc * phic * (ro * so - ro_o * so_o) / dt_arg
        R_e = Vc * (E - E_o) / dt_arg
        for (Lp, Rp), (Lk, Rk), (Lf, Rf) in zip(interior_p, interior_rock, interior_phi):
            left, right = Lp["left"], Lp["right"]
            if dep is not None:  # phase-potential flux per phase
                Gw = nfvm_flux(pot_w_aug, Lp, Rp, scheme, mu_half=mu_half)
                Go = nfvm_flux(pot_o_aug, Lp, Rp, scheme, mu_half=mu_half)
            else:
                Gw = Go = nfvm_flux(
                    pot_w_aug, Lp, Rp, scheme, mu_half=mu_half
                )  # shared (monotone Darcy)
            up_w = float(Gw.detach()) >= 0
            up_o = float(Go.detach()) >= 0
            Fw = (mm_w_aug[left] if up_w else mm_w_aug[right]) * Gw
            Fo = (mm_o_aug[left] if up_o else mm_o_aug[right]) * Go
            Hw_up = H_w_aug[left] if up_w else H_w_aug[right]
            Ho_up = H_o_aug[left] if up_o else H_o_aug[right]
            F_rock = nfvm_flux(T_aug, Lk, Rk, scheme, mu_half=mu_half)  # rock conduction
            G_phi = nfvm_flux(T_aug, Lf, Rf, scheme, mu_half=mu_half)  # porosity-weighted geometric
            lam_face = lam_w * 0.5 * (sw_aug[left] + sw_aug[right]) + lam_o * 0.5 * (
                so_aug[left] + so_aug[right]
            )
            Fe = Hw_up * Fw + Ho_up * Fo + F_rock + lam_face * G_phi
            R_w = R_w + _face_divergence(Fw, left, right, n)
            R_o = R_o + _face_divergence(Fo, left, right, n)
            R_e = R_e + _face_divergence(Fe, left, right, n)
        for cell, WI, bhp, inj_sw, T_inj in wells:  # BHP wells (mass + energy)
            dpw = p[cell] - bhp
            prod = float(dpw.detach()) >= 0
            sw_w = sw[cell] if prod else torch.as_tensor(inj_sw, dtype=dtype)
            T_w = T[cell] if prod else torch.as_tensor(T_inj, dtype=dtype)
            rww, row = rho_w(p[cell], T_w), rho_o(p[cell], T_w)
            Fw = WI * rww * relperm.kr_water(sw_w) / mu_w * dpw
            Fo = WI * row * relperm.kr_oil(sw_w) / mu_o * dpw
            Hw_w, Ho_w = cp_w * T_w + p[cell] / rww, cp_o * T_w + p[cell] / row
            R_w = R_w + _boundary_divergence(Fw, cell, n)
            R_o = R_o + _boundary_divergence(Fo, cell, n)
            R_e = R_e + _boundary_divergence(Hw_w * Fw + Ho_w * Fo, cell, n)
        for w in rate_wells:  # rate-controlled wells: total volumetric rate q
            cell, WI, q, inj_sw, T_inj = (
                w[0],
                w[1],
                w[2],
                w[3],
                w[4],
            )  # split by fractional flow; optional bhp_limit
            qb = torch.as_tensor(q, dtype=dtype)  # ⇒ switches to BHP control when the implied bhp
            prod = float(qb.detach()) >= 0  # would pass the limit (rate then under-delivered)
            sw_w = sw[cell] if prod else torch.as_tensor(inj_sw, dtype=dtype)
            T_w = T[cell] if prod else torch.as_tensor(T_inj, dtype=dtype)
            rww, row = rho_w(p[cell], T_w), rho_o(p[cell], T_w)
            lw, lo = relperm.kr_water(sw_w) / mu_w, relperm.kr_oil(sw_w) / mu_o
            lt = (lw + lo).clamp_min(1e-30)
            viol = False
            if len(w) >= 6 and w[5] is not None:  # bhp limit (max injector / min producer); None ⇒
                lim = torch.as_tensor(
                    w[5], dtype=dtype
                )  # no limit (matches the MPFA _has contract)
                bhp_rate = p[cell] - qb / (WI * lt)
                viol = bool(bhp_rate.detach() < lim) if prod else bool(bhp_rate.detach() > lim)
            if viol:  # BHP-limited: q_actual = WI·λ·(p − bhp_limit) ≤ q
                dpl = p[cell] - lim
                Fw, Fo = WI * rww * lw * dpl, WI * row * lo * dpl
            else:
                Fw, Fo = rww * (lw / lt) * qb, row * (lo / lt) * qb
            Hw_w, Ho_w = cp_w * T_w + p[cell] / rww, cp_o * T_w + p[cell] / row
            R_w = R_w + _boundary_divergence(Fw, cell, n)
            R_o = R_o + _boundary_divergence(Fo, cell, n)
            R_e = R_e + _boundary_divergence(Hw_w * Fw + Ho_w * Fo, cell, n)
        if mp_pc is not None:  # multi-perforation wells: perforations share one
            pc, pw, pWI = mp_pc, mp_pw, mp_pwi  # bhp solved from the total-rate constraint
            surf, isw_p, tinj_p = mp_surf[pw], mp_isw[pw], mp_tinj[pw]

            def mp_state(
                up: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:  # per-perforation upwind
                return torch.where(up, sw[pc], isw_p), torch.where(up, T[pc], tinj_p)

            def mp_bhp(
                sw_w: torch.Tensor, T_w: torch.Tensor
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:  # closed-form shared bhp + the Σ(WI·Λ) denominator
                rwk, rok = rho_w(p[pc], T_w), rho_o(p[pc], T_w)
                lwk, lok = relperm.kr_water(sw_w) / mu_w, relperm.kr_oil(sw_w) / mu_o
                Lam = torch.where(surf, rwk * lwk / rho_w_ref + rok * lok / rho_o_ref, lwk + lok)
                WL = pWI * Lam
                num = p.new_zeros(mp_nw).scatter_add(0, pw, WL * p[pc])
                den = p.new_zeros(mp_nw).scatter_add(0, pw, WL)
                return (num - mp_q) / den.clamp_min(1e-30), rwk, rok, lwk, lok

            with torch.no_grad():  # cross-flow: bhp ↔ per-perf flow direction → a
                up = (mp_q >= 0)[pw]  # fixed point on the (detached) upwind pattern
                for _ in range(12):
                    bhp, *_ = mp_bhp(*mp_state(up))
                    up_new = (p[pc] - bhp[pw]) >= 0
                    if bool(torch.equal(up_new, up)):
                        break
                    up = up_new
            sw_w, T_w = mp_state(up)  # differentiable final pass on the frozen pattern
            bhp, rwk, rok, lwk, lok = mp_bhp(sw_w, T_w)
            dp = p[pc] - bhp[pw]
            Fw_p, Fo_p = pWI * rwk * lwk * dp, pWI * rok * lok * dp
            Hw_p, Ho_p = cp_w * T_w + p[pc] / rwk, cp_o * T_w + p[pc] / rok
            R_w = R_w + scatter_boundary_outflow(Fw_p, pc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_p, pc, n)
            R_e = R_e + scatter_boundary_outflow(Hw_p * Fw_p + Ho_p * Fo_p, pc, n)
        if p_bc is not None:  # Dirichlet (p, sw, T) ghosts: one-sided cell→ghost
            for Lp_b, Lk_b, Lf_b in zip(boundary_p, boundary_rock, boundary_phi):
                cell, ghost = Lp_b["left"], Lp_b["right"]
                Gw = -_onesided(pot_w_aug, Lp_b)[0]  # per-phase boundary flux (buoyancy can flip
                Go = -_onesided(pot_o_aug, Lp_b)[
                    0
                ]  # the phases independently; shared when no gravity)
                up_w, up_o = float(Gw.detach()) >= 0, float(Go.detach()) >= 0
                Fw = (mm_w_aug[cell] if up_w else mm_w_aug[ghost]) * Gw
                Fo = (mm_o_aug[cell] if up_o else mm_o_aug[ghost]) * Go
                Hw_up = H_w_aug[cell] if up_w else H_w_aug[ghost]
                Ho_up = H_o_aug[cell] if up_o else H_o_aug[ghost]
                F_rock = -_onesided(T_aug, Lk_b)[0]
                G_phi = -_onesided(T_aug, Lf_b)[0]
                lam_face = lam_w * 0.5 * (sw_aug[cell] + sw_aug[ghost]) + lam_o * 0.5 * (
                    so_aug[cell] + so_aug[ghost]
                )
                Fe = Hw_up * Fw + Ho_up * Fo + F_rock + lam_face * G_phi
                R_w = R_w + _boundary_divergence(Fw, cell, n)
                R_o = R_o + _boundary_divergence(Fo, cell, n)
                R_e = R_e + _boundary_divergence(Fe, cell, n)
        for cell, q, sw_inj, T_inj in neumann:  # prescribed-rate (Neumann) source: total
            qb = torch.as_tensor(q, dtype=dtype)  # volumetric rate split by fractional flow;
            out = float(qb.detach()) >= 0  # q ≥ 0 removes cell fluid, q < 0 injects sw_inj@T_inj
            sw_b = sw[cell] if out else torch.as_tensor(sw_inj, dtype=dtype)
            T_b = T[cell] if out else torch.as_tensor(T_inj, dtype=dtype)
            rwb, rob = rho_w(p[cell], T_b), rho_o(p[cell], T_b)
            lw, lo = relperm.kr_water(sw_b) / mu_w, relperm.kr_oil(sw_b) / mu_o
            tot = (lw + lo).clamp_min(1e-30)
            Fw, Fo = rwb * (lw / tot) * qb, rob * (lo / tot) * qb
            Hwb, Hob = cp_w * T_b + p[cell] / rwb, cp_o * T_b + p[cell] / rob
            R_w = R_w + _boundary_divergence(Fw, cell, n)
            R_o = R_o + _boundary_divergence(Fo, cell, n)
            R_e = R_e + _boundary_divergence(Hwb * Fw + Hob * Fo, cell, n)
        if source_water is not None:
            R_w = R_w - source_water
        if source_oil is not None:
            R_o = R_o - source_oil
        if source_energy is not None:
            R_e = R_e - source_energy
        return torch.cat([R_w, R_o, R_e])

    inf = float("inf")
    # capacity-rate normalisation (CNV/MB-style; fixed scales never collapse) for the three
    # blocks of widely different magnitude (water, oil mass vs energy); the scales are ∝ 1/dt,
    # so a sub-step (dt_arg = frac·dt) rescales them through dt_arg.
    T_char = float(T0.abs().max()) + 1e-30
    rhoC_char = (
        float((1.0 - phic).mean()) * rho_r * cp_r
        + float(phic.mean()) * (rho_w_ref * cp_w + rho_o_ref * cp_o) * 0.5
    )
    sw_num = float((Vc * phic).max()) * rho_w_ref + 1e-300
    so_num = float((Vc * phic).max()) * rho_o_ref + 1e-300
    se_num = float(Vc.max()) * rhoC_char * T_char + 1e-300

    def scaled(r: torch.Tensor, dt_arg: float) -> float:  # convergence test only
        r = r.detach()  # never on the state's autograd path
        return max(
            float(torch.linalg.vector_norm(r[:n], ord=inf)) * dt_arg / sw_num,
            float(torch.linalg.vector_norm(r[n : 2 * n], ord=inf)) * dt_arg / so_num,
            float(torch.linalg.vector_norm(r[2 * n :], ord=inf)) * dt_arg / se_num,
        )

    def solve_one(st: torch.Tensor, frac: float) -> AdaptiveOutcome:
        """Advance ``st`` by ``frac * dt`` and report retry metadata."""
        work: WorkCounters = {
            "residual_evaluations": 0,
            "jacobian_assemblies": 0,
            "linear_solves": 0,
            "nonlinear_iterations": 0,
            "max_nonlinear_iterations": max_iter,
        }

        def evaluate(
            q: torch.Tensor,
            old_state: torch.Tensor,
            step_dt: float,
            *,
            mu_half: bool = False,
        ) -> torch.Tensor:
            work["residual_evaluations"] += 1
            return residual(q, old_state, step_dt, mu_half=mu_half)

        def assemble(
            q: torch.Tensor,
            old_state: torch.Tensor,
            step_dt: float,
            *,
            mu_half: bool = False,
        ) -> torch.Tensor:
            work["jacobian_assemblies"] += 1
            return torch.autograd.functional.jacobian(
                lambda candidate: evaluate(candidate, old_state, step_dt, mu_half=mu_half),
                q,
                vectorize=True,
            )

        def solve_linear(
            jacobian: torch.Tensor, residual_value: torch.Tensor
        ) -> torch.Tensor:
            work["linear_solves"] += 1
            return _newton_solve(jacobian, residual_value, "nfvm_thermal_two_phase")

        dt_sub = frac * float(dt)
        old = st.detach()
        with torch.no_grad():  # grad-free march; gradient from _ift_attach
            s = old.clone()
            r0 = evaluate(s, old, dt_sub, mu_half=True)  # avgMPFA (linear) init
            J0 = assemble(s, old, dt_sub, mu_half=True)
            d0 = solve_linear(J0, r0)
            rn0, a0 = scaled(r0, dt_sub), 1.0
            init_found = False
            domain_rejected = False
            valid_init_candidate = False
            for _ls in range(25):  # line-search the avgMPFA init (it can overshoot)
                try:
                    trial_residual = evaluate(s - a0 * d0, old, dt_sub, mu_half=True)
                except FlowContractError:
                    domain_rejected = True
                    a0 *= 0.5
                    continue
                valid_init_candidate = True
                if scaled(trial_residual, dt_sub) < rn0:
                    init_found = True
                    break
                a0 *= 0.5
            if not init_found and not valid_init_candidate:
                return (
                    s,
                    False,
                    "saturation" if domain_rejected else "newton",
                    work,
                )
            s = s - a0 * d0
            converged = False
            for _it in range(max_iter):
                work["nonlinear_iterations"] = _it
                r = evaluate(s, old, dt_sub)
                rn = scaled(r, dt_sub)
                if rn < tol:
                    converged = True
                    break
                work["nonlinear_iterations"] = _it + 1
                J = assemble(s, old, dt_sub)
                d = solve_linear(J, r)
                alpha, found = 1.0, False
                domain_rejected = False
                for _ls in range(25):
                    try:
                        trial_residual = evaluate(s - alpha * d, old, dt_sub)
                    except FlowContractError:
                        domain_rejected = True
                        alpha *= 0.5
                        continue
                    if scaled(trial_residual, dt_sub) < rn:
                        found = True
                        break
                    alpha *= 0.5
                if not found:  # no descent possible (Newton stalled)
                    break
                s = s - alpha * d
        if not converged:
            return (
                s,
                False,
                "saturation" if domain_rejected else "newton",
                work,
            )
        sw_d = s[n : 2 * n]  # bound check only, grad-free already
        if (
            float(sw_d.max()) > 1.0 + _SAT_BOUND_TOL or float(sw_d.min()) < -_SAT_BOUND_TOL
        ):  # gross over-injection
            return s, False, "saturation", work
        return (
            _ift_attach(
                s,
                lambda q: residual(q, st, dt_sub),
                "nfvm_thermal_two_phase",
                work=work,
            ),
            True,
            "newton",
            work,
        )

    p_init = (
        p0.to(dtype)
        if isinstance(p0, torch.Tensor)
        else geom._permeability_view().new_full((n,), float(p0))
    )
    sw_init = (
        sw0.to(dtype)
        if isinstance(sw0, torch.Tensor)
        else geom._permeability_view().new_full((n,), float(sw0))
    )  # non-uniform IC ⇒ segregation
    state0 = torch.cat([p_init, sw_init, T0.to(dtype)])
    return _adaptive_march(
        solve_one,
        state0,
        nsteps,
        max_substeps,
        "nfvm_thermal_two_phase",
        dt_s=dt,
        history_config=history_config,
    )


def nfvm_thermal_compositional(
    geom: NFVMGeometry,
    eos: CubicEOS,
    T0: torch.Tensor,
    p0: Scalar,
    z0: TensorInput,
    *,
    source_mol: torch.Tensor | None = None,
    source_energy: torch.Tensor | None = None,
    mu_l: float = 5e-4,
    mu_v: float = 2e-5,
    swl: float = 0.0,
    sgr: float = 0.0,
    n_l: float = 2.0,
    n_v: float = 2.0,
    cp_components: Scalar = 2100.0,
    cp_r: float = 1000.0,
    rho_r: float = 2650.0,
    lam_l: float = 0.13,
    lam_v: float = 0.03,
    lam_rock: Scalar = 3.0,
    depth: torch.Tensor | None = None,
    gravity: float = 9.81,
    viscosity: str = "constant",
    wells: Sequence[CompositionalWell] = (),
    rate_wells: Sequence[CompositionalRateWell] = (),
    multiperf_wells: Sequence[CompositionalMultiperfWell] = (),
    neumann: Sequence[CompositionalNeumann] = (),
    z_bc: TensorInput | None = None,
    T_bc: TensorInput | None = None,
    depth_bc: TensorInput | None = None,
    phi: Scalar = 0.2,
    V: Scalar = 1.0,
    dt: float = 8.64e4,
    nsteps: int = 10,
    scheme: str = "nmpfa",
    tol: float = 1e-8,
    max_iter: int = 40,
    max_substeps: int = 0,
    history_config: FlowHistoryConfig | None = None,
) -> torch.Tensor:
    """EOS-flash compositional **thermal** flow fully on the NFVM monotone flux, per-
    component molar balance + energy, the compositional analog of
    :func:`nfvm_thermal_two_phase`.

    A vapor-liquid flash runs at the per-cell temperature ``T`` (a primary variable), so
    K-values, Z-factors and molar densities respond to ``T`` (a constant-Cp
    calorific model: per-phase molar heat capacity ``cm_p = Σ_i C_i x_{i,p} M_i``, molar
    enthalpy ``h_p = cm_p T + p/ρ_p``). The Darcy molar phase flux
    ``q_p = (ρ_p k_rp/μ_p)|_up·G_p`` (``G_p = nfvm_flux(p)`` monotone, phase-potential
    upwound) carries the upwind composition ``F_i = q_l x_{i,f} + q_v y_{i,f}``; the energy
    flux is the advected molar enthalpy ``Σ_p h_p|_up q_p`` plus the parallel-conductance
    Fourier flux (rock stencil + porosity-weighted fluid stencil scaled by
    ``λ_l S_l,f + λ_v S_v,f``). On a K-orthogonal grid the scheme collapses to two-point and
    reproduces :class:`MPFAThermalCompositionalModel` to machine precision.

    ``geom`` a :class:`NFVMGeometry` with ``K`` = permeability and no ghosts (no-flow +
    insulated, closed domain). State ``[p, z_1..z_{nc-1}, T]``; ``z0`` the ``(nc,)`` feed
    composition; returns the ``(nsteps+1, n_cells·nc + n_cells)`` history. **Differentiable
    through time** (every accepted (sub-)step reattached with its exact implicit-function-
    theorem VJP via :func:`_ift_attach`, chain never detached, FD-pinned by
    the NFVM gradient suite). Fails loud (raises) if the coupled flash + monotone Newton
    does not converge; pass ``max_substeps > 0`` to **auto-subdivide** a failed macro step
    (adaptive dt sub-stepping, a flash crash / divergence on an over-aggressive ``dt`` is
    caught and retried on smaller sub-steps). v1: closed domain + per-component molar source
    ``source_mol`` ``(n_cells, nc)`` + energy source. Optional **gravity** (pass ``depth``,
    ``(n,)`` +down [m]): each phase gets its own potential
    ``Φ_p = p − ρ_p·M_p·g·depth`` (``ρ_p`` molar density, ``M_p = Σ_i comp_{p,i} M_i`` phase
    molar mass ⇒ ``ρ_p·M_p`` mass density), so the lighter phase rises. Optional
    ``viscosity='lbc'`` ⇒ Lohrenz-Bray-Clark per-cell phase viscosity (else constant
    ``mu_l``/``mu_v``). Optional **BHP wells** ``wells = [(cell, WI, bhp, z_inj, T_inj), ...]``
    (production draws the cell fluid by phase molar mobility; injection injects the feed
    ``z_inj`` at ``T_inj``, the injectant flashed at the cell pressure). **Rate wells** ``rate_wells
    = [(cell, WI, q, z_inj, T_inj[, bhp_limit]), ...]`` prescribe a total molar rate (split by molar
    fractional flow; injection injects ``z_inj``), switching to BHP control past ``bhp_limit``.
    **Multi-perforation** ``multiperf_wells = [(perfs, q_target, z_inj, T_inj), ...]`` share one bhp
    from the total molar-rate constraint, cross-flow handled per perforation (injectant flashed per
    perforation). **Dirichlet (p, z, T)
    boundaries**: build ``geom`` with mirror ghosts (``geom.ghost_p`` = the fixed pressures) and
    pass ``z_bc`` / ``T_bc`` (the fixed feed compositions / temperatures, aligned with
    ``geom.ghost_p``; the ghost feed is flashed at ``(p_bc, T_bc)``). Like :func:`solve_nfvm`
    the bc is anchored at the mirror ghost (≈ half a cell beyond the face), a discretisation-
    convention difference from the MPFA bc-at-face. **Neumann** ``neumann = [(cell, q, z_inj,
    T_inj), ...]`` prescribes a total molar rate (``q ≥ 0`` removes the molar fractional-flow
    composition, ``q < 0`` injects feed ``z_inj`` at ``T_inj``) as a source, a pure source, so it
    reproduces the MPFA Neumann to machine precision. Pass ``depth_bc`` (ghost depths) for
    **gravity with Dirichlet**, the per-phase ghost head ``Φ_p,bc = p_bc − ρ_p,bc M_p,bc g D_bc``."""
    max_iter, tol = _validate_newton_controls(
        max_iter=max_iter,
        tol=tol,
        object_name="nfvm_thermal_compositional",
    )
    if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 0:
        raise FlowContractError(
            "max_substeps must be a non-negative integer",
            object_name="nfvm_thermal_compositional",
            field="max_substeps",
            expected="integer >= 0",
            actual=max_substeps,
        )
    from ...compositional.flash import flash, require_flash_converged
    from ...compositional.mpfa_model import _R_GAS as RG
    from ...compositional.viscosity import lbc_viscosity

    n = geom.n
    dtype = geom._permeability_view().dtype
    dim = geom._coordinates_view().shape[1]
    eye = torch.eye(dim, dtype=dtype)
    nc = eos.mixture.molar_mass_kg_mol.shape[0]
    dep = depth.to(dtype) if depth is not None else None  # (n,) +down [m] ⇒ buoyancy
    if dep is not None and tuple(dep.shape) != (n,):
        raise GeoBrainError(
            "nfvm_thermal_compositional: depth must be a per-cell (n,) field (+down).",
            object_name="nfvm_thermal_compositional",
            field="depth",
            expected=f"({n},)",
            actual=tuple(dep.shape),
        )
    lr = (
        lam_rock
        if isinstance(lam_rock, torch.Tensor)
        else torch.full((n,), float(lam_rock), dtype=dtype)
    )
    phic = phi if isinstance(phi, torch.Tensor) else torch.full((n,), float(phi), dtype=dtype)
    Vc = V if isinstance(V, torch.Tensor) else torch.full((n,), float(V), dtype=dtype)
    cp_comp = (
        cp_components.to(dtype)
        if isinstance(cp_components, torch.Tensor)
        else torch.full((nc,), float(cp_components), dtype=dtype)
    )
    mw = cp_comp * eos.mixture.molar_mass_kg_mol
    face_records = [list(records) for records in geom._face_records_view()]
    geom_rock = NFVMGeometry(
        geom._coordinates_view(),
        _extend_to_ghosts((1.0 - phic) * lr, geom).reshape(-1, 1, 1) * eye,
        face_records,
        [],
        n,
    )
    geom_phi = NFVMGeometry(
        geom._coordinates_view(),
        _extend_to_ghosts(phic, geom).reshape(-1, 1, 1) * eye,
        face_records,
        [],
        n,
    )
    interior_p, boundary_p = _build_nfvm(geom, scheme)
    interior_rock, boundary_rock = _build_nfvm(geom_rock, scheme)
    interior_phi, boundary_phi = _build_nfvm(geom_phi, scheme)
    ghost_pressures = geom._ghost_pressures_view()
    p_bc = torch.stack(ghost_pressures) if ghost_pressures else None
    z_boundary: torch.Tensor | None = None
    T_boundary: torch.Tensor | None = None
    depth_boundary: torch.Tensor | None = None
    if p_bc is not None:  # fixed-(p, z, T) boundaries
        if z_bc is None or T_bc is None:
            raise GeoBrainError(
                "nfvm_thermal_compositional: Dirichlet ghosts present (geom.ghost_p) but z_bc / T_bc not given; "
                "a fixed-pressure boundary also needs a fixed feed composition and temperature.",
                object_name="nfvm_thermal_compositional",
                field="z_bc/T_bc",
            )
        z_boundary = torch.as_tensor(z_bc, dtype=dtype)
        z_boundary = (
            z_boundary.expand(p_bc.shape[0], nc)
            if z_boundary.dim() == 1
            else z_boundary
        )  # (g, nc) feed per ghost
        T_boundary = torch.as_tensor(T_bc, dtype=dtype)
        if depth is not None:  # gravity ⇒ per-phase ghost head
            if depth_bc is None:  # Φ_p,bc = p_bc − ρ_p,bc M_p,bc g D_bc
                raise GeoBrainError(
                    "nfvm_thermal_compositional: gravity (depth) with Dirichlet ghosts needs depth_bc "
                    "(the ghost depths, aligned with geom.ghost_p).",
                    object_name="nfvm_thermal_compositional",
                    field="depth_bc",
                )
            depth_boundary = (
                depth_bc.to(dtype)
                if isinstance(depth_bc, torch.Tensor)
                else torch.as_tensor(depth_bc, dtype=dtype)
            )
            if tuple(depth_boundary.shape) != (
                p_bc.shape[0],
            ):  # a wrong length would silently broadcast /
                raise GeoBrainError(  # cat-mismatch instead of a clean error
                    "nfvm_thermal_compositional: depth_bc must be a per-ghost field aligned with geom.ghost_p.",
                    object_name="nfvm_thermal_compositional",
                    field="depth_bc",
                    expected=f"({p_bc.shape[0]},)",
                    actual=tuple(depth_boundary.shape),
                )

    # multi-perforation wells: perforations [(cell, WI), ...] share one bhp from Σ_k WI_k Λ_k
    # (p_k − bhp) = q_target (total MOLAR rate); Λ_k = total molar mobility (cell on production /
    # the injectant's on injection), so cross-flow needs a per-perforation upwind fixed point.
    mp_cell_ids: list[int] = []
    mp_wi_values: list[Scalar] = []
    mp_well_ids: list[int] = []
    mp_specs = list(multiperf_wells)
    for wi, (perfs, _q, _z, _tinj) in enumerate(mp_specs):
        for cell, WI in perfs:
            mp_cell_ids.append(int(cell))
            mp_wi_values.append(WI)
            mp_well_ids.append(wi)
    mp_pc: torch.Tensor | None
    if mp_cell_ids:
        mp_pc = torch.tensor(mp_cell_ids, dtype=torch.long)
        mp_pwi = torch.stack([torch.as_tensor(x, dtype=dtype) for x in mp_wi_values])
        mp_pw = torch.tensor(mp_well_ids, dtype=torch.long)
        mp_q = torch.stack([torch.as_tensor(s[1], dtype=dtype) for s in mp_specs])
        mp_z = torch.stack([torch.as_tensor(s[2], dtype=dtype) for s in mp_specs])  # (nw, nc) feed
        mp_tinj = torch.stack([torch.as_tensor(s[3], dtype=dtype) for s in mp_specs])
        mp_nw = len(mp_specs)
    else:
        mp_pc = None

    def phase_state(
        p: torch.Tensor, z: torch.Tensor, T: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:  # flash → (V, x, y, S_l, ρ_l, ρ_v, D) molar
        res = flash(eos, p, T, z)
        require_flash_converged(res, object_name="nfvm_thermal_compositional")
        Vf = res.V
        x, y = res.x, res.y
        Z_l = eos.compressibility(*eos.mixture_ab(x, *eos.ab_components(p, T))[:2], root="liquid")
        Z_v = eos.compressibility(*eos.mixture_ab(y, *eos.ab_components(p, T))[:2], root="vapor")
        v_l, v_v = Z_l * RG * T / p, Z_v * RG * T / p
        Dm = (1.0 - Vf) * v_l + Vf * v_v
        return Vf, x, y, 1.0 - (Vf * v_v) / Dm, 1.0 / v_l, 1.0 / v_v, Dm

    def mobilities(
        S_l: torch.Tensor, mul: Scalar, muv: Scalar
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Corey kr / μ
        se = ((S_l - swl) / (1.0 - swl - sgr + 1e-30)).clamp(0.0, 1.0)
        return se.pow(n_l) / mul, (1.0 - se).pow(n_v) / muv

    def phase_visc(
        p: torch.Tensor,
        T: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rho_l: torch.Tensor,
        rho_v: torch.Tensor,
    ) -> tuple[Scalar, Scalar]:  # constant or per-cell LBC at T
        if viscosity != "lbc":
            return mu_l, mu_v
        Z_l, Z_v = p / (rho_l * RG * T), p / (rho_v * RG * T)
        return (lbc_viscosity(eos.mixture, p, T, x, Z_l), lbc_viscosity(eos.mixture, p, T, y, Z_v))

    def unpack(
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_red = state[n : n * nc].reshape(n, nc - 1)
        z = torch.cat([z_red, 1.0 - z_red.sum(-1, keepdim=True)], dim=-1)
        return state[:n], z, state[n * nc :]

    def residual(
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt_arg: float,
        mu_half: bool = False,
    ) -> torch.Tensor:
        p, z, T = unpack(state)
        p_o, z_o, T_o = unpack(state_old)
        V, x, y, S_l, rho_l, rho_v, Dm = phase_state(p, z, T)
        _, x_o, y_o, S_l_o, rho_l_o, rho_v_o, Dm_o = phase_state(p_o, z_o, T_o)
        N = (phic / Dm).unsqueeze(-1) * z
        N_o = (phic / Dm_o).unsqueeze(-1) * z_o
        R = Vc.unsqueeze(-1) * (N - N_o) / dt_arg  # (n, nc) per-component molar balance
        cm_l, cm_v = (x * mw).sum(-1), (y * mw).sum(-1)
        h_l, h_v = cm_l * T + p / rho_l, cm_v * T + p / rho_v
        mu_l_c, mu_v_c = phase_visc(p, T, x, y, rho_l, rho_v)
        mob_l, mob_v = mobilities(S_l, mu_l_c, mu_v_c)
        mm_l, mm_v = rho_l * mob_l, rho_v * mob_v
        S_v = 1.0 - S_l
        # per-phase potential Φ_p = p − ρ_p·M_p·g·D (ρ_p·M_p = mass density); buoyancy lets the
        # lighter (vapor) phase rise. No gravity ⇒ Φ_l = Φ_v = p ⇒ one shared flux.
        if dep is not None:
            molar_mass = eos.mixture.molar_mass_kg_mol
            M_l, M_v = (x * molar_mass).sum(-1), (y * molar_mass).sum(-1)
            gz = gravity * dep
            phi_l_pot, phi_v_pot = p - rho_l * M_l * gz, p - rho_v * M_v * gz
        else:
            phi_l_pot = phi_v_pot = p
        if p_bc is not None:  # Dirichlet ghosts: flash the feed z_bc at the
            assert z_boundary is not None
            assert T_boundary is not None
            _, x_b, y_b, Sl_b, rl_b, rv_b, Dm_b = phase_state(
                p_bc, z_boundary, T_boundary
            )  # boundary (p_bc, T_bc); an
            cm_l_b, cm_v_b = (x_b * mw).sum(-1), (y_b * mw).sum(-1)  # interior stencil near the
            mu_l_b, mu_v_b = phase_visc(
                p_bc, T_boundary, x_b, y_b, rl_b, rv_b
            )  # boundary references the ghost,
            mob_l_b, mob_v_b = mobilities(Sl_b, mu_l_b, mu_v_b)  # so all lookups use _aug arrays
            p_aug, T_aug = torch.cat([p, p_bc]), torch.cat([T, T_boundary])
            x_aug, y_aug = torch.cat([x, x_b]), torch.cat([y, y_b])
            mm_l_aug, mm_v_aug = (
                torch.cat([mm_l, rl_b * mob_l_b]),
                torch.cat([mm_v, rv_b * mob_v_b]),
            )
            h_l_aug = torch.cat([h_l, cm_l_b * T_boundary + p_bc / rl_b])
            h_v_aug = torch.cat([h_v, cm_v_b * T_boundary + p_bc / rv_b])
            S_l_aug, S_v_aug = torch.cat([S_l, Sl_b]), torch.cat([S_v, 1.0 - Sl_b])
            if dep is not None:  # per-phase ghost head Φ_p,bc = p_bc − ρ_p M_p g D_bc
                assert depth_boundary is not None
                molar_mass = eos.mixture.molar_mass_kg_mol
                M_l_b, M_v_b = (x_b * molar_mass).sum(-1), (y_b * molar_mass).sum(-1)
                pot_l_aug = torch.cat(
                    [phi_l_pot, p_bc - rl_b * M_l_b * gravity * depth_boundary]
                )
                pot_v_aug = torch.cat(
                    [phi_v_pot, p_bc - rv_b * M_v_b * gravity * depth_boundary]
                )
            else:
                pot_l_aug = pot_v_aug = p_aug
        else:
            p_aug, T_aug, x_aug, y_aug = p, T, x, y
            mm_l_aug, mm_v_aug, h_l_aug, h_v_aug = mm_l, mm_v, h_l, h_v
            S_l_aug, S_v_aug, pot_l_aug, pot_v_aug = S_l, S_v, phi_l_pot, phi_v_pot
        E = (1.0 - phic) * rho_r * cp_r * T + phic * (S_l * rho_l * cm_l + S_v * rho_v * cm_v) * T
        cm_l_o, cm_v_o = (x_o * mw).sum(-1), (y_o * mw).sum(-1)
        E_o = (1.0 - phic) * rho_r * cp_r * T_o + phic * (
            S_l_o * rho_l_o * cm_l_o + (1.0 - S_l_o) * rho_v_o * cm_v_o
        ) * T_o
        R_e = Vc * (E - E_o) / dt_arg
        for (Lp, Rp), (Lk, Rk), (Lf, Rf) in zip(interior_p, interior_rock, interior_phi):
            left, right = Lp["left"], Lp["right"]
            if dep is not None:  # per-phase potential flux (buoyancy)
                Gl = nfvm_flux(pot_l_aug, Lp, Rp, scheme, mu_half=mu_half)
                Gv = nfvm_flux(pot_v_aug, Lp, Rp, scheme, mu_half=mu_half)
            else:
                Gl = Gv = nfvm_flux(pot_l_aug, Lp, Rp, scheme, mu_half=mu_half)
            ul = float(Gl.detach()) >= 0
            uv = float(Gv.detach()) >= 0
            ql = (mm_l_aug[left] if ul else mm_l_aug[right]) * Gl
            qv = (mm_v_aug[left] if uv else mm_v_aug[right]) * Gv
            F = ql * (x_aug[left] if ul else x_aug[right]) + qv * (
                y_aug[left] if uv else y_aug[right]
            )
            adv = (h_l_aug[left] if ul else h_l_aug[right]) * ql + (
                h_v_aug[left] if uv else h_v_aug[right]
            ) * qv
            F_rock = nfvm_flux(T_aug, Lk, Rk, scheme, mu_half=mu_half)
            G_phi = nfvm_flux(T_aug, Lf, Rf, scheme, mu_half=mu_half)
            lam_face = lam_l * 0.5 * (S_l_aug[left] + S_l_aug[right]) + lam_v * 0.5 * (
                S_v_aug[left] + S_v_aug[right]
            )
            F_e = adv + F_rock + lam_face * G_phi
            R = R.index_add(0, torch.tensor([left, right]), torch.stack([F, -F]))
            R_e = R_e + _face_divergence(F_e, left, right, n)
        for cell, WI, bhp, z_inj, T_inj in wells:  # BHP wells (component molar + energy)
            dpw = p[cell] - bhp
            if float(dpw.detach()) >= 0:  # PRODUCTION: cell fluid, per-phase molar
                q_l = WI * (rho_l * mob_l)[cell] * dpw  # mobility; component split x_cell/y_cell
                q_v = WI * (rho_v * mob_v)[cell] * dpw
                Fw = q_l * x[cell] + q_v * y[cell]  # (nc,)
                Ew = h_l[cell] * q_l + h_v[cell] * q_v
            else:  # INJECTION: specified feed z_inj at T_inj
                zi, Ti = torch.as_tensor(z_inj, dtype=dtype), torch.as_tensor(T_inj, dtype=dtype)
                _, xi, yi, Sli, rli, rvi, Dfi = phase_state(
                    p[cell : cell + 1], zi.reshape(1, nc), Ti.reshape(1)
                )
                mu_li, mu_vi = phase_visc(p[cell : cell + 1], Ti.reshape(1), xi, yi, rli, rvi)
                mob_li, mob_vi = mobilities(Sli, mu_li, mu_vi)
                q_inj = (WI * (rli * mob_li + rvi * mob_vi) * dpw)[0]  # total injectant molar rate
                h_feed = (zi * mw).sum() * Ti + p[cell] * Dfi[
                    0
                ]  # feed molar enthalpy (ρ_feed = 1/D_feed)
                Fw, Ew = q_inj * zi, h_feed * q_inj
            R = R.index_add(0, torch.tensor([cell]), Fw.reshape(1, nc))
            R_e = R_e + _boundary_divergence(Ew, cell, n)
        for w in rate_wells:  # rate-controlled wells: total molar rate q split
            cell, WI, q, z_inj, T_inj = (
                w[0],
                w[1],
                w[2],
                w[3],
                w[4],
            )  # by molar fractional flow; optional bhp_limit
            qb = torch.as_tensor(q, dtype=dtype)  # ⇒ switches to BHP control past the limit
            prod = float(qb.detach()) >= 0
            lam_c = (mm_l[cell] + mm_v[cell]).clamp_min(1e-30)  # cell total molar mobility
            if prod:
                lam = lam_c
            else:  # injectant flash → its total molar mobility + feed
                zi, Ti = torch.as_tensor(z_inj, dtype=dtype), torch.as_tensor(T_inj, dtype=dtype)
                _, xi, yi, Sli, rli, rvi, Dfi = phase_state(
                    p[cell : cell + 1], zi.reshape(1, nc), Ti.reshape(1)
                )
                mu_li, mu_vi = phase_visc(p[cell : cell + 1], Ti.reshape(1), xi, yi, rli, rvi)
                mob_li, mob_vi = mobilities(Sli, mu_li, mu_vi)
                lam = (rli * mob_li + rvi * mob_vi).clamp_min(1e-30)[0]
            viol = False
            if len(w) >= 6 and w[5] is not None:  # None ⇒ no limit (matches the MPFA _has contract)
                lim = torch.as_tensor(w[5], dtype=dtype)
                bhp_rate = p[cell] - qb / (WI * lam)
                viol = bool(bhp_rate.detach() < lim) if prod else bool(bhp_rate.detach() > lim)
            q_eff = (
                WI * lam * (p[cell] - lim) if viol else qb
            )  # total molar rate (target or bhp-limited)
            if prod:
                q_l, q_v = q_eff * mm_l[cell] / lam_c, q_eff * mm_v[cell] / lam_c
                Fw, Ew = q_l * x[cell] + q_v * y[cell], h_l[cell] * q_l + h_v[cell] * q_v
            else:
                Fw = q_eff * zi
                Ew = q_eff * ((zi * mw).sum() * Ti + p[cell] * Dfi[0])
            R = R.index_add(0, torch.tensor([cell]), Fw.reshape(1, nc))
            R_e = R_e + _boundary_divergence(Ew, cell, n)
        if mp_pc is not None:  # multi-perforation wells: perforations share one
            pc, pw, pWI = (
                mp_pc,
                mp_pw,
                mp_pwi,
            )  # bhp; cross-flow ⇒ per-perforation upwind fixed point
            zinj_p, tinj_p = mp_z[pw], mp_tinj[pw]
            _, xi, yi, Sli, rli, rvi, Dfi = phase_state(
                p[pc], zinj_p, tinj_p
            )  # injectant flash at each perforation
            mu_li, mu_vi = phase_visc(p[pc], tinj_p, xi, yi, rli, rvi)
            mob_li, mob_vi = mobilities(Sli, mu_li, mu_vi)
            lam_i = rli * mob_li + rvi * mob_vi  # injectant total molar mobility (per perforation)
            lam_c = (mm_l + mm_v)[pc]  # cell total molar mobility (per perforation)

            def mp_bhp(
                up: torch.Tensor,
            ) -> torch.Tensor:  # cell mobility on production, injectant on injection
                WL = pWI * torch.where(up, lam_c, lam_i)
                num = p.new_zeros(mp_nw).scatter_add(0, pw, WL * p[pc])
                den = p.new_zeros(mp_nw).scatter_add(0, pw, WL)
                return (num - mp_q) / den.clamp_min(1e-30)

            with torch.no_grad():  # cross-flow fixed point on the (detached) pattern
                up = (mp_q >= 0)[pw]
                for _ in range(12):
                    up_new = (p[pc] - mp_bhp(up)[pw]) >= 0
                    if bool(torch.equal(up_new, up)):
                        break
                    up = up_new
            dp = p[pc] - mp_bhp(up)[pw]  # differentiable final pass on the frozen pattern
            q_l, q_v = (
                pWI * mm_l[pc] * dp,
                pWI * mm_v[pc] * dp,
            )  # production: cell molar fractional flow
            F_prod = q_l.unsqueeze(-1) * x[pc] + q_v.unsqueeze(-1) * y[pc]
            E_prod = h_l[pc] * q_l + h_v[pc] * q_v
            q_inj = pWI * lam_i * dp  # injection: feed z_inj (flashed for its enthalpy)
            h_feed = (zinj_p * mw).sum(-1) * tinj_p + p[pc] * Dfi
            upc = up.unsqueeze(-1)
            R = R.index_add(0, pc, torch.where(upc, F_prod, q_inj.unsqueeze(-1) * zinj_p))
            R_e = R_e + scatter_boundary_outflow(torch.where(up, E_prod, q_inj * h_feed), pc, n)
        if p_bc is not None:  # Dirichlet (p, z, T) ghosts: one-sided cell→ghost
            for Lp_b, Lk_b, Lf_b in zip(boundary_p, boundary_rock, boundary_phi):
                cell, ghost = Lp_b["left"], Lp_b["right"]
                Gl = -_onesided(pot_l_aug, Lp_b)[0]  # per-phase boundary flux (buoyancy can flip
                Gv = -_onesided(pot_v_aug, Lp_b)[
                    0
                ]  # the phases independently; shared when no gravity)
                ul, uv = float(Gl.detach()) >= 0, float(Gv.detach()) >= 0
                ql = (mm_l_aug[cell] if ul else mm_l_aug[ghost]) * Gl
                qv = (mm_v_aug[cell] if uv else mm_v_aug[ghost]) * Gv
                F = ql * (x_aug[cell] if ul else x_aug[ghost]) + qv * (
                    y_aug[cell] if uv else y_aug[ghost]
                )
                adv = (h_l_aug[cell] if ul else h_l_aug[ghost]) * ql + (
                    h_v_aug[cell] if uv else h_v_aug[ghost]
                ) * qv
                F_rock = -_onesided(T_aug, Lk_b)[0]
                G_phi = -_onesided(T_aug, Lf_b)[0]
                lam_face = lam_l * 0.5 * (S_l_aug[cell] + S_l_aug[ghost]) + lam_v * 0.5 * (
                    S_v_aug[cell] + S_v_aug[ghost]
                )
                R = R.index_add(0, torch.tensor([cell]), F.reshape(1, nc))
                R_e = R_e + _boundary_divergence(adv + F_rock + lam_face * G_phi, cell, n)
        for cell, q, z_inj, T_inj in neumann:  # prescribed molar-rate (Neumann) source:
            qb = torch.as_tensor(q, dtype=dtype)  # q ≥ 0 removes the molar fractional-flow
            if float(qb.detach()) >= 0:  # composition, q < 0 injects feed z_inj@T_inj
                mmt = (mm_l[cell] + mm_v[cell]).clamp_min(1e-30)
                comp = (mm_l[cell] * x[cell] + mm_v[cell] * y[cell]) / mmt
                h_nb = (mm_l[cell] * h_l[cell] + mm_v[cell] * h_v[cell]) / mmt
            else:
                zi, Ti = torch.as_tensor(z_inj, dtype=dtype), torch.as_tensor(T_inj, dtype=dtype)
                _, _, _, _, _, _, Dfi = phase_state(
                    p[cell : cell + 1], zi.reshape(1, nc), Ti.reshape(1)
                )
                comp, h_nb = zi, (zi * mw).sum() * Ti + p[cell] * Dfi[0]
            R = R.index_add(0, torch.tensor([cell]), (qb * comp).reshape(1, nc))
            R_e = R_e + _boundary_divergence(qb * h_nb, cell, n)
        if source_mol is not None:
            R = R - source_mol
        if source_energy is not None:
            R_e = R_e - source_energy
        return torch.cat([R.reshape(-1), R_e])

    p_init = (
        p0
        if isinstance(p0, torch.Tensor)
        else geom._permeability_view().new_full((n,), float(p0))
    )
    z_init = z0.to(dtype) if isinstance(z0, torch.Tensor) else torch.as_tensor(z0, dtype=dtype)
    z_init = z_init.expand(n, nc) if z_init.dim() == 1 else z_init
    T_init = T0.to(dtype)
    with torch.no_grad():  # capacity-rate scales (molar vs energy ~1e6)
        Dm0 = phase_state(p_init, z_init, T_init)[6]
    inf = float("inf")
    T_char = float(T_init.abs().max()) + 1e-30
    rhoC_char = float((1.0 - phic).mean()) * rho_r * cp_r + float(phic.mean()) * float(
        (1.0 / Dm0).mean()
    ) * float(mw.mean())
    smol_num = float((Vc * phic).max()) * float((1.0 / Dm0).max()) + 1e-300
    se_num = float(Vc.max()) * rhoC_char * T_char + 1e-300

    def scaled(r: torch.Tensor, dt_arg: float) -> float:  # molar vs energy blocks
        r = r.detach()
        return max(
            float(torch.linalg.vector_norm(r[: n * nc], ord=inf)) * dt_arg / smol_num,
            float(torch.linalg.vector_norm(r[n * nc :], ord=inf)) * dt_arg / se_num,
        )

    def solve_one(st: torch.Tensor, frac: float) -> AdaptiveOutcome:
        """Advance ``st`` by ``frac * dt`` and report retry metadata."""
        work: WorkCounters = {
            "residual_evaluations": 0,
            "jacobian_assemblies": 0,
            "linear_solves": 0,
            "nonlinear_iterations": 0,
            "max_nonlinear_iterations": max_iter,
        }

        def evaluate(
            q: torch.Tensor,
            old_state: torch.Tensor,
            step_dt: float,
            *,
            mu_half: bool = False,
        ) -> torch.Tensor:
            work["residual_evaluations"] += 1
            return residual(q, old_state, step_dt, mu_half=mu_half)

        def assemble(
            q: torch.Tensor,
            old_state: torch.Tensor,
            step_dt: float,
            *,
            mu_half: bool = False,
        ) -> torch.Tensor:
            work["jacobian_assemblies"] += 1
            return torch.autograd.functional.jacobian(
                lambda candidate: evaluate(candidate, old_state, step_dt, mu_half=mu_half),
                q,
                vectorize=True,
            )

        def solve_linear(
            jacobian: torch.Tensor, residual_value: torch.Tensor
        ) -> torch.Tensor:
            work["linear_solves"] += 1
            return _newton_solve(jacobian, residual_value, "nfvm_thermal_compositional")

        try:
            dt_sub = frac * float(dt)
            old = st.detach()
            with torch.no_grad():  # grad-free march; gradient from _ift_attach
                s = old.clone()
                r0 = evaluate(s, old, dt_sub, mu_half=True)  # avgMPFA init
                J0 = assemble(s, old, dt_sub, mu_half=True)
                d0 = solve_linear(J0, r0)
                rn0, a0 = scaled(r0, dt_sub), 1.0
                for _ls in range(25):
                    trial_residual = evaluate(s - a0 * d0, old, dt_sub, mu_half=True)
                    if scaled(trial_residual, dt_sub) < rn0:
                        break
                    a0 *= 0.5
                s = s - a0 * d0
                converged = False
                for _it in range(max_iter):
                    work["nonlinear_iterations"] = _it
                    r = evaluate(s, old, dt_sub)
                    rn = scaled(r, dt_sub)
                    if rn < tol:
                        converged = True
                        break
                    work["nonlinear_iterations"] = _it + 1
                    J = assemble(s, old, dt_sub)
                    d = solve_linear(J, r)
                    alpha, found = 1.0, False
                    for _ls in range(25):
                        trial_residual = evaluate(s - alpha * d, old, dt_sub)
                        if scaled(trial_residual, dt_sub) < rn:
                            found = True
                            break
                        alpha *= 0.5
                    if not found:
                        break
                    s = s - alpha * d
            if not converged:
                return s, False, "newton", work
            return (
                _ift_attach(
                    s,
                    lambda q: residual(q, st, dt_sub),
                    "nfvm_thermal_compositional",
                    work=work,
                ),
                True,
                "newton",
                work,
            )
        except FlowConvergenceError:
            # sub-step ⇒ sub-step smaller (or fail loud)
            return st, False, "newton", work

    state0 = torch.cat([p_init, z_init[:, : nc - 1].reshape(-1), T_init])
    return _adaptive_march(
        solve_one,
        state0,
        nsteps,
        max_substeps,
        "nfvm_thermal_compositional",
        dt_s=dt,
        history_config=history_config,
    )

