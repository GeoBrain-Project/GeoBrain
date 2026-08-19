"""
Empirical / regression rock-physics models (``ComponentModel`` layer).

Pure-math ``nn.Module`` implementations, the operator-style wrappers
(``*Operator``, ``PropertyTransform``) that adapt these to the ``ModelState``
contract are colocated in this module.

Models:
    Gardner:           density from P-velocity, ρ = a · vp^b
    Castagna:          mudrock line, vs = a · vp + b
    GreenbergCastagna: multi-mineral vs predictor via Hill-averaging
    Han:               vp/vs/ρ regression for shaly sandstones
    MacBeth:           pressure-sensitivity of dry-frame moduli

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..moduli import check_bounded
from ..empirical import (
    bulk_density,
    castagna_mudrock_velocity,
    castagna_shear_velocity,
    ehrenberg_porosity,
    gardner_density,
    greenberg_castagna_shear_velocity,
    han_shaly_sandstone,
    hillis_velocity,
    hjelstuen_velocity,
    japsen_velocity,
    krief_dry_moduli,
    macbeth_dry_moduli,
    ramm_porosity,
    raymer_hunt_gardner_velocity,
    scherbaum_velocity,
    sclater_depth_from_porosity,
    sclater_porosity_from_depth,
    st_peter_velocities,
    storvoll_velocity,
    wyllie_time_average_velocity,
)
from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    PropertyTransform,
)
from ._categories import EmpiricalModel
from ._registry import register


# --- Gardner ----------------------------------------------------------------


@register("Gardner", aliases=["gardner"])
class Gardner(EmpiricalModel):
    """
    Gardner et al. (1974) empirical density-velocity relation.

    ``ρ = a · vp^b``  with default ``(a, b) = (310.0, 0.25)`` for ``vp``
    in m/s and ``ρ`` in SI kg/m³ (``a_SI = 1000 · a_gcm3``; the historical
    g/cm³ coefficient was 0.31). (SI-EXEMPT: historical note)
    """

    def __init__(self, a: float = 310.0, b: float = 0.25) -> None:
        super().__init__()
        if a <= 0:
            raise GeoBrainError(
                "Gardner coefficient a must be positive",
                object_name="Gardner",
                field="a",
                expected="> 0",
                actual=a,
            )
        self.a = float(a)
        self.b = float(b)

    def forward(self, vp: Tensor) -> Tensor:
        return gardner_density(vp, self.a, self.b)


# --- Castagna ---------------------------------------------------------------


@register("Castagna", aliases=["castagna"])
class Castagna(EmpiricalModel):
    """
    Castagna et al. (1985) mudrock line.

    ``vs = a · vp + b``  with default ``(a, b) = (0.804, -856)`` m/s.
    """

    def __init__(self, a: float = 0.804, b: float = -856.0) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)

    def forward(self, vp: Tensor) -> Tensor:
        return castagna_shear_velocity(vp, self.a, self.b)


# --- Greenberg-Castagna -----------------------------------------------------


_GC_COEFFS = {
    "sand": (0.0, 0.80416, -0.85588),
    "shale": (0.0, 0.76969, -0.86735),
    "lime": (-0.05509, 1.01677, -1.03049),
    "dolo": (0.0, 0.58321, -0.07775),
}


def _gc_poly(c: tuple[float, float, float], x: Tensor) -> Tensor:
    return c[0] * x.pow(2) + c[1] * x + c[2]


@register("GreenbergCastagna", aliases=["greenberg_castagna"])
class GreenbergCastagna(EmpiricalModel):
    """
    Greenberg & Castagna (1992) multi-mineral vs predictor.

    Per-lithology Castagna polynomial vs(vp), then Hill-averaged using
    rock-volume mineral fractions. Inputs are velocity (m/s) and four
    non-negative fractions (need not sum to 1, normalised internally).
    Output: vs in m/s.
    """

    def forward(
        self,
        vp: Tensor, f_sand: Tensor, f_shale: Tensor, f_lime: Tensor, f_dolo: Tensor,
    ) -> Tensor:
        return greenberg_castagna_shear_velocity(
            vp, f_sand, f_shale, f_lime, f_dolo
        )


# --- Han --------------------------------------------------------------------


@register("Han", aliases=["han"])
class Han(EmpiricalModel):
    """
    Han (1986) brine-saturated shaly-sandstone regression.

    Defaults: 40 MPa effective pressure, brine fluid (ρ_fl = 1000 kg/m³),
    quartz mineral (ρ_min = 2650 kg/m³).

    ``vp = (a + b·φ + c·C) · 1000`` m/s
    ``vs = (a' + b'·φ + c'·C) · 1000`` m/s
    ``ρ  = ρ_min · (1 − φ) + ρ_fl · φ`` kg/m³
    """

    def __init__(
        self,
        *,
        vp_coefs: tuple[float, float, float] = (5.59, -6.93, -2.18),
        vs_coefs: tuple[float, float, float] = (3.52, -4.91, -1.89),
        rho_mineral: float = 2650.0,
        rho_fluid: float = 1000.0,
    ) -> None:
        super().__init__()
        if len(vp_coefs) != 3 or len(vs_coefs) != 3:
            raise GeoBrainError(
                "Han vp_coefs and vs_coefs must each be 3 numbers",
                object_name="Han",
                field="vp_coefs/vs_coefs",
                expected="(a, b_phi, b_clay)",
                actual=(vp_coefs, vs_coefs),
            )
        for name, value in (("rho_mineral", rho_mineral), ("rho_fluid", rho_fluid)):
            if value <= 0:
                raise GeoBrainError(
                    f"Han {name} must be positive",
                    object_name="Han",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        self.vp_coefs = tuple(float(c) for c in vp_coefs)
        self.vs_coefs = tuple(float(c) for c in vs_coefs)
        self.rho_mineral = float(rho_mineral)
        self.rho_fluid = float(rho_fluid)

    def forward(self, phi: Tensor, clay: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return han_shaly_sandstone(
            phi,
            clay,
            vp_coefficients=self.vp_coefs,
            vs_coefficients=self.vs_coefs,
            mineral_density=self.rho_mineral,
            fluid_density=self.rho_fluid,
        )


# --- MacBeth pressure sensitivity -------------------------------------------


@register("MacBeth", aliases=["macbeth"])
class MacBeth(EmpiricalModel):
    """
    MacBeth (2004) pressure-sensitivity of dry-frame moduli.

    ``K_dry(P) = K_∞ − ΔK · exp(−P/P_K)``
    ``μ_dry(P) = μ_∞ − Δμ · exp(−P/P_μ)``

    At P=0 the rock is most compliant; as P grows the moduli approach
    ``(K_∞, μ_∞)``. Returns ``(K_dry, μ_dry)`` in Pa.
    """

    def __init__(
        self,
        *,
        K_inf: float = 24.0e9,
        mu_inf: float = 18.0e9,
        dK: float = 9.0e9,
        dmu: float = 6.0e9,
        P_K: float = 20.0e6,
        P_mu: float = 25.0e6,
    ) -> None:
        super().__init__()
        for name, value in (("K_inf", K_inf), ("mu_inf", mu_inf)):
            if value <= 0:
                raise GeoBrainError(
                    f"MacBeth {name} must be positive",
                    object_name="MacBeth",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        for name, value in (("dK", dK), ("dmu", dmu)):
            if value < 0:
                raise GeoBrainError(
                    f"MacBeth {name} must be non-negative",
                    object_name="MacBeth",
                    field=name,
                    expected=">= 0",
                    actual=value,
                )
        for name, value in (("P_K", P_K), ("P_mu", P_mu)):
            if value <= 0:
                raise GeoBrainError(
                    f"MacBeth {name} must be positive",
                    object_name="MacBeth",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        # K_dry(P=0) and μ_dry(P=0) must stay strictly positive.
        if K_inf - dK <= 0:
            raise GeoBrainError(
                "MacBeth K_inf must exceed dK so K_dry(0) > 0",
                object_name="MacBeth",
                field="K_inf - dK",
                expected="> 0",
                actual=K_inf - dK,
            )
        if mu_inf - dmu <= 0:
            raise GeoBrainError(
                "MacBeth mu_inf must exceed dmu so mu_dry(0) > 0",
                object_name="MacBeth",
                field="mu_inf - dmu",
                expected="> 0",
                actual=mu_inf - dmu,
            )
        self.K_inf = float(K_inf)
        self.mu_inf = float(mu_inf)
        self.dK = float(dK)
        self.dmu = float(dmu)
        self.P_K = float(P_K)
        self.P_mu = float(P_mu)

    def forward(self, P_eff: Tensor) -> tuple[Tensor, Tensor]:
        return macbeth_dry_moduli(
            P_eff,
            k_infinite=self.K_inf,
            mu_infinite=self.mu_inf,
            delta_k=self.dK,
            delta_mu=self.dmu,
            pressure_scale_k=self.P_K,
            pressure_scale_mu=self.P_mu,
        )


# --- Velocity-porosity (Raymer-Hunt-Gardner, Wyllie) -----------------------


@register("RaymerHuntGardner", aliases=["RHG", "raymer_hunt_gardner"])
class RaymerHuntGardner(EmpiricalModel):
    """``Vp = (1-φ)² Vp_mat + φ Vp_fl`` (m/s)."""

    def forward(
        self, phi: Tensor,
        Vp_mat: Tensor | float = 5500.0, Vp_fl: Tensor | float = 1500.0,
    ) -> Tensor:
        return raymer_hunt_gardner_velocity(phi, Vp_mat, Vp_fl)


@register("WyllieTimeAverage", aliases=["Wyllie", "wyllie"])
class WyllieTimeAverage(EmpiricalModel):
    """``1/Vp = φ/Vp_fl + (1-φ)/Vp_mat`` time-average (m/s)."""

    def forward(
        self, phi: Tensor,
        Vp_mat: Tensor | float = 5500.0, Vp_fl: Tensor | float = 1500.0,
    ) -> Tensor:
        return wyllie_time_average_velocity(phi, Vp_mat, Vp_fl)


# --- Vp-Vs (Castagna mudrock line; distinct from Castagna a,b form) ---------


@register("CastagnaMudrock", aliases=["Mudrock", "mudrock"])
class CastagnaMudrock(EmpiricalModel):
    """Castagna mudrock line with public velocities in m/s.

    The quoted coefficients use km/s only inside the canonical SI kernel.
    """

    def forward(self, Vp: Tensor) -> Tensor:
        return castagna_mudrock_velocity(Vp)


# --- Porosity-modulus -------------------------------------------------------


@register("Krief", aliases=["krief"])
class Krief(EmpiricalModel):
    """Krief porosity-modulus: ``M_dry = M_0 · (1 − a₀)`` with ``a₀ = 1 − (1−φ)^(3/(1−φ))``."""

    def forward(
        self, phi: Tensor, K0: Tensor, G0: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return krief_dry_moduli(phi, K0, G0)


# --- Compaction trends ------------------------------------------------------


@register("Storvoll", aliases=["storvoll"], tags=["compaction"])
class Storvoll(EmpiricalModel):
    """Storvoll velocity-depth: ``Vp = Z/1.76 + 2600`` (m/s)."""

    def forward(self, Z: Tensor) -> Tensor:
        return storvoll_velocity(Z)


@register("Japsen", aliases=["japsen"], tags=["compaction"])
class Japsen(EmpiricalModel):
    """Japsen 4-segment piecewise-linear velocity-depth trend (m/s)."""

    def forward(self, Z: Tensor) -> Tensor:
        return japsen_velocity(Z)


@register("Hillis", aliases=["hillis"], tags=["compaction"])
class Hillis(EmpiricalModel):
    """Hillis transit-time-derived ``Vp(Z)`` (m/s; Z in m)."""

    def forward(self, Z: Tensor) -> Tensor:
        return hillis_velocity(Z)


@register("Scherbaum", aliases=["scherbaum"], tags=["compaction"])
class Scherbaum(EmpiricalModel):
    """``Vp = 2325 + 0.51·Z`` (Z in m, Vp in m/s)."""

    def forward(self, Z: Tensor) -> Tensor:
        return scherbaum_velocity(Z)


@register("Hjelstuen", aliases=["hjelstuen"], tags=["compaction"])
class Hjelstuen(EmpiricalModel):
    """``Vp = (1.87 + 0.55·Z_km) · 1000`` (m/s)."""

    def forward(self, Z: Tensor) -> Tensor:
        return hjelstuen_velocity(Z)


# --- Porosity-depth ---------------------------------------------------------


@register("Ehrenberg", aliases=["ehrenberg"], tags=["porosity-depth"])
class Ehrenberg(EmpiricalModel):
    """``φ = −0.0922·Z_km + 0.48``."""

    def forward(self, Z: Tensor) -> Tensor:
        return ehrenberg_porosity(Z)


@register("RammPorosity", aliases=["ramm"], tags=["porosity-depth"])
class RammPorosity(EmpiricalModel):
    """Ramm & Bjørlykke regional porosity-depth (Norwegian shelf)."""

    def __init__(self, region: str = "haltenbanken") -> None:
        super().__init__()
        self.region = region

    def forward(self, Z: Tensor) -> Tensor:
        return ramm_porosity(Z, region=self.region)


@register("Sclater", aliases=["sclater"], tags=["porosity-depth"])
class Sclater(EmpiricalModel):
    """Sclater-Christie compaction with public depth in metres."""

    def forward(self, phi: Tensor) -> Tensor:
        return sclater_depth_from_porosity(phi)

    def inverse(self, Z: Tensor) -> Tensor:
        return sclater_porosity_from_depth(Z)


# --- Pressure-velocity ------------------------------------------------------


@register("StPeter", aliases=["st_peter"], tags=["pressure-velocity"])
class StPeter(EmpiricalModel):
    """St. Peter ``Vp/Vs`` in m/s from effective pressure in Pa."""

    def __init__(self, sample: int = 1) -> None:
        super().__init__()
        if sample not in (1, 2):
            raise GeoBrainError(
                "StPeter sample must be 1 or 2",
                object_name="StPeter", field="sample",
                expected="1 or 2", actual=sample,
            )
        self.sample = int(sample)

    def forward(self, Pe: Tensor) -> tuple[Tensor, Tensor]:
        return st_peter_velocities(Pe, sample=self.sample)


# --- Bulk density -----------------------------------------------------------


@register("DensityModel", aliases=["Density", "BulkDensity", "density"])
class DensityModel(EmpiricalModel):
    """``ρ_bulk = (1−φ)·ρ_matrix + φ·ρ_fluid``."""

    def forward(
        self, phi: Tensor,
        rho_matrix: Tensor | float = 2650.0, rho_fluid: Tensor | float = 1000.0,
    ) -> Tensor:
        return bulk_density(phi, rho_matrix, rho_fluid)

    def porosity_from_density(
        self, rho_bulk: Tensor,
        rho_matrix: Tensor | float = 2650.0, rho_fluid: Tensor | float = 1000.0,
    ) -> Tensor:
        denominator = rho_matrix - rho_fluid
        if isinstance(denominator, Tensor) and bool(torch.any(denominator == 0.0)):
            raise GeoBrainError(
                "matrix and fluid densities must differ",
                object_name="DensityModel",
                field="rho_matrix/rho_fluid",
                expected="unequal densities",
                actual="equal density",
            )
        return (rho_matrix - rho_bulk) / denominator


__all__ = [
    "Castagna", "CastagnaOperator", "CastagnaMudrock",
    "DensityModel",
    "Ehrenberg",
    "Gardner", "GardnerOperator",
    "GreenbergCastagna", "GreenbergCastagnaOperator",
    "Han", "HanOperator",
    "Hillis", "Hjelstuen",
    "Japsen",
    "Krief",
    "MacBeth", "MacBethOperator",
    "RaymerHuntGardner", "RammPorosity",
    "Scherbaum", "Sclater", "Storvoll", "StPeter",
    "WyllieTimeAverage",
]


# ============================================================================
# Operator wrappers (PropertyTransform)
#
# Merged from rock/gardner.py, rock/empirical.py, rock/pressure.py.
# These thin wrappers adapt the ComponentModel classes above to the
# ModelState / ForwardContext contract used by the operator graph.
# ============================================================================

# Private aliases so operator __init__ can reference the ComponentModel
# without name collisions.
_GardnerModel = Gardner
_HanModel = Han
_CastagnaModel = Castagna
_GCModel = GreenbergCastagna
_MacBethModel = MacBeth


class GardnerOperator(PropertyTransform):
    """Gardner ``ρ = a · vp^b`` operator (state.vp → state.rho)."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp",),
        output_keys=("rho",),
    )

    def __init__(self, a: float = 310.0, b: float = 0.25) -> None:
        super().__init__()
        self._model = _GardnerModel(a=a, b=b)
        self.a = self._model.a
        self.b = self._model.b

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (vp,) = state.fetch("vp")
        rho = self._model(vp)
        return state.with_tensors(rho=rho)

    def extra_repr(self) -> str:
        return f"a={self.a}, b={self.b}"


class HanOperator(PropertyTransform):
    """
    Han 1986 regression for brine-saturated shaly sandstones.

    Inputs (ModelState):  ``phi``, ``clay``.
    Outputs (ModelState): ``vp`` (m/s), ``vs`` (m/s), ``rho`` (kg/m³).
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("phi", "clay"),
        output_keys=("vp", "vs", "rho"),
    )

    def __init__(
        self,
        *,
        vp_coefs: tuple[float, float, float] = (5.59, -6.93, -2.18),
        vs_coefs: tuple[float, float, float] = (3.52, -4.91, -1.89),
        rho_mineral: float = 2650.0,
        rho_fluid: float = 1000.0,
    ) -> None:
        super().__init__()
        self._model = _HanModel(
            vp_coefs=vp_coefs,
            vs_coefs=vs_coefs,
            rho_mineral=rho_mineral,
            rho_fluid=rho_fluid,
        )
        self.vp_coefs = self._model.vp_coefs
        self.vs_coefs = self._model.vs_coefs
        self.rho_mineral = self._model.rho_mineral
        self.rho_fluid = self._model.rho_fluid

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        phi, clay = state.fetch("phi", "clay")
        if phi.shape != clay.shape:
            raise GeoBrainError(
                "phi and clay must share shape",
                object_name="Han",
                field="clay",
                expected=tuple(phi.shape),
                actual=tuple(clay.shape),
            )
        check_bounded(phi, "phi", owner="Han")
        check_bounded(clay, "clay", owner="Han")
        vp, vs, rho = self._model(phi, clay)
        return state.with_tensors(vp=vp, vs=vs, rho=rho)


class CastagnaOperator(PropertyTransform):
    """Castagna 1985 mudrock line ``vs = a · vp + b``."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp",),
        output_keys=("vs",),
    )

    def __init__(self, a: float = 0.804, b: float = -856.0) -> None:
        super().__init__()
        self._model = _CastagnaModel(a=a, b=b)
        self.a = self._model.a
        self.b = self._model.b

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (vp,) = state.fetch("vp")
        return state.with_tensors(vs=self._model(vp))

    def extra_repr(self) -> str:
        return f"vs = {self.a} * vp + {self.b}"


class GreenbergCastagnaOperator(PropertyTransform):
    """
    Greenberg & Castagna 1992 multi-mineral vs predictor.

    Inputs (ModelState):
        ``vp``:                               m/s
        ``f_sand, f_shale, f_lime, f_dolo``: non-negative mineral fractions

    Output (ModelState): ``vs`` (m/s).
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "f_sand", "f_shale", "f_lime", "f_dolo"),
        output_keys=("vs",),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model = _GCModel()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        vp, f_s, f_sh, f_l, f_d = state.fetch(
            "vp", "f_sand", "f_shale", "f_lime", "f_dolo",
        )
        for name, t in zip(
            ("f_sand", "f_shale", "f_lime", "f_dolo"), (f_s, f_sh, f_l, f_d),
        ):
            if t.shape != vp.shape:
                raise GeoBrainError(
                    "GreenbergCastagna inputs must share shape",
                    object_name="GreenbergCastagna",
                    field=name,
                    expected=tuple(vp.shape),
                    actual=tuple(t.shape),
                )
        for name, t in (
            ("f_sand", f_s), ("f_shale", f_sh), ("f_lime", f_l), ("f_dolo", f_d),
        ):
            if (t < 0).any():
                raise GeoBrainError(
                    f"GreenbergCastagna {name} must be non-negative",
                    object_name="GreenbergCastagna",
                    field=name,
                    expected=">= 0",
                    actual=float(t.min()),
                )
        f_sum = f_s + f_sh + f_l + f_d
        if (f_sum <= 0).any():
            raise GeoBrainError(
                "GreenbergCastagna mineral fractions must have positive sum",
                object_name="GreenbergCastagna",
                field="fractions",
                expected="sum > 0",
                actual=float(f_sum.min()),
            )
        vs = self._model(vp, f_s, f_sh, f_l, f_d)
        return state.with_tensors(vs=vs)


class MacBethOperator(PropertyTransform):
    """
    MacBeth 2004 pressure-sensitivity law for dry-frame moduli.

    See :class:`MacBeth` for the physics. This wrapper expects
    ``state.P_effective`` (Pa) and writes ``state.K_dry, state.mu_dry`` (Pa).
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("P_effective",),
        output_keys=("K_dry", "mu_dry"),
    )

    def __init__(
        self,
        *,
        K_inf: float = 24.0e9,
        mu_inf: float = 18.0e9,
        dK: float = 9.0e9,
        dmu: float = 6.0e9,
        P_K: float = 20.0e6,
        P_mu: float = 25.0e6,
    ) -> None:
        super().__init__()
        self._model = _MacBethModel(
            K_inf=K_inf, mu_inf=mu_inf, dK=dK, dmu=dmu, P_K=P_K, P_mu=P_mu,
        )
        # Expose attrs for backward-compat introspection.
        self.K_inf = self._model.K_inf
        self.mu_inf = self._model.mu_inf
        self.dK = self._model.dK
        self.dmu = self._model.dmu
        self.P_K = self._model.P_K
        self.P_mu = self._model.P_mu

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (P_eff,) = state.fetch("P_effective")
        if (P_eff < 0).any():
            raise GeoBrainError(
                "MacBeth P_effective must be non-negative",
                object_name="MacBeth",
                field="P_effective",
                expected=">= 0",
                actual=float(P_eff.min()),
            )
        K_dry, mu_dry = self._model(P_eff)
        return state.with_tensors(K_dry=K_dry, mu_dry=mu_dry)
