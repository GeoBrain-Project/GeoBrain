"""
Fluid ``ComponentModel`` implementations (mixing + substitution + EOS)
+ operator wrappers.

Implementations are split across private submodules grouped by phase-type
and physical family:

- :mod:`_fluid_mixing`:      Wood, Brie
- :mod:`_fluid_batzlewang`: BatzleWangBrine/Gas/OilDead/OilLive + unified BatzleWang
- :mod:`_fluid_gassmann`:    Gassmann/GassmannInverse/GassmannFluidSub +
  BiotHF/BiotDispersion + GeertsmaSmitHF/LF + BrownKorringa* + MavkoJizba
- :mod:`_fluid_co2`:         CO2Properties + LiveOil + CO2Brine

Public symbols are re-exported below so historical imports
(``from geobrain.physics.rock.models.fluid import Gassmann``,
``from geobrain.physics.rock.models import Gassmann``) continue to work.

Operator wrappers (merged from rock/gassmann.py, rock/fluid.py):
    GassmannOperator, WoodOperator, BrieOperator, BatzleWangBrineOperator, BatzleWangGasOperator,
    BatzleWangOilDeadOperator, BatzleWangOilLiveOperator

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    PropertyTransform,
)
from ._fluid_batzlewang import (
    BatzleWang,
    BatzleWangBrine,
    BatzleWangGas,
    BatzleWangOilDead,
    BatzleWangOilLive,
)
from ._fluid_co2 import CO2Brine, CO2Properties, LiveOil
from ._fluid_gassmann import (
    BiotDispersion,
    BiotHF,
    BrownKorringaDry2Sat,
    BrownKorringaSat2Dry,
    BrownKorringaSub,
    Gassmann,
    GassmannFluidSub,
    GassmannInverse,
    GeertsmaSmitHF,
    GeertsmaSmitLF,
    MavkoJizba,
)
from ._fluid_mixing import Brie, Wood
from ..moduli import check_bounded

# ComponentModel names for re-stamping __module__
_COMPONENT_MODEL_NAMES = [
    "BatzleWang",
    "BatzleWangBrine", "BatzleWangGas", "BatzleWangOilDead", "BatzleWangOilLive",
    "BiotDispersion", "BiotHF",
    "Brie",
    "BrownKorringaDry2Sat", "BrownKorringaSat2Dry", "BrownKorringaSub",
    "CO2Brine", "CO2Properties",
    "Gassmann", "GassmannFluidSub", "GassmannInverse",
    "GeertsmaSmitHF", "GeertsmaSmitLF",
    "LiveOil",
    "MavkoJizba",
    "Wood",
]

# Re-stamp ``__module__`` so the public class repr / introspection
# reports this aggregator module rather than the private submodule a
# class physically lives in. Several tests pin
# ``Cls.__module__ == "geobrain.physics.rock.models.fluid"`` to
# distinguish the functional-layer classes from their operator-style
# namesakes (e.g. ``rock.Gassmann`` vs ``rock.models.Gassmann``).
for _name in _COMPONENT_MODEL_NAMES:
    _cls = globals()[_name]
    _cls.__module__ = __name__
del _name, _cls


# ============================================================================
# Operator wrappers (PropertyTransform)
#
# Merged from rock/gassmann.py and rock/fluid.py.
# ============================================================================

# Private aliases
_GassmannModel = Gassmann
_WoodModel = Wood
_BrieModel = Brie
_BWBrineModel = BatzleWangBrine
_BWGasModel = BatzleWangGas
_BWOilDeadModel = BatzleWangOilDead
_BWOilLiveModel = BatzleWangOilLive


class GassmannOperator(PropertyTransform):
    """Gassmann fluid substitution as a ``PropertyTransform``."""

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        K_fluid: float | None = 2.25e9,
        rho_mineral: float = 2650.0,
        rho_fluid: float | None = 1000.0,
        mu_dry: float = 20.0e9,
        fluid_in_state: bool = False,
    ) -> None:
        super().__init__()
        self._model = _GassmannModel(
            K_mineral=K_mineral, rho_mineral=rho_mineral, mu_dry=mu_dry,
        )
        # When fluid_in_state=False the scalar fluid params must be positive.
        # The constructor-scalar mode is what the historical API exposed;
        # validate it here at the operator boundary.
        if not fluid_in_state:
            if K_fluid is None or rho_fluid is None:
                raise GeoBrainError(
                    "Gassmann needs scalar K_fluid and rho_fluid unless fluid_in_state=True",
                    object_name="Gassmann",
                    field="K_fluid/rho_fluid",
                    expected="positive scalars",
                    actual=(K_fluid, rho_fluid),
                )
            for name, value in (("K_fluid", K_fluid), ("rho_fluid", rho_fluid)):
                if value <= 0:
                    raise GeoBrainError(
                        f"Gassmann {name} must be positive",
                        object_name="Gassmann",
                        field=name,
                        expected="> 0",
                        actual=value,
                    )

        self.K_mineral = self._model.K_mineral
        self.rho_mineral = self._model.rho_mineral
        self.mu_dry = self._model.mu_dry
        self.fluid_in_state = bool(fluid_in_state)
        self.K_fluid = float(K_fluid) if (K_fluid is not None and not fluid_in_state) else None
        self.rho_fluid = float(rho_fluid) if (rho_fluid is not None and not fluid_in_state) else None

        trainable = (
            ("K_dry", "phi", "K_fluid", "rho_fluid") if fluid_in_state else ("K_dry", "phi")
        )
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=trainable,
            output_keys=("vp", "rho"),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K_dry, phi = state.fetch("K_dry", "phi")
        if K_dry.shape != phi.shape:
            raise GeoBrainError(
                "K_dry and phi must have matching shape",
                object_name="Gassmann",
                field="phi",
                expected=tuple(K_dry.shape),
                actual=tuple(phi.shape),
            )

        # Honor pore-fluid moduli carried in the ModelState whenever they are
        # present: produced by an upstream WoodOperator/BrieOperator mix (whose
        # ``output_keys`` are exactly ``K_fluid``/``rho_fluid``) or passed
        # directly for fluid-substitution monitoring. The state is the
        # data-flow source of truth; the constructor scalars are only a
        # fallback for when no fluid field is supplied. Silently substituting
        # the scalars when the state *does* carry fluid moduli would both use
        # the wrong fluid (e.g. default brine instead of the Wood-mixed CO₂/
        # brine) and zero the ``K_fluid``/``rho_fluid`` → ``vp`` gradient that a
        # 4D fluid-substitution inversion depends on. ``fluid_in_state=True``
        # additionally *requires* the fields (declared in ``trainable_inputs``)
        # for a strict, introspectable contract.
        k_in = "K_fluid" in state.tensors
        rho_in = "rho_fluid" in state.tensors
        if k_in != rho_in:
            # Exactly one of the pair present: fail loudly rather than silently
            # dropping the lone field and reverting to constructor scalars
            # (which would produce wrong physics with no signal). Fluid moduli
            # travel as a pair (WoodOperator/BrieOperator emit both); a half-specified
            # state is a contract violation upstream.
            raise GeoBrainError(
                f"Gassmann received only {'K_fluid' if k_in else 'rho_fluid'} "
                "in the ModelState; K_fluid and rho_fluid must be supplied "
                "together or neither (omit both to use constructor scalars).",
                object_name="Gassmann",
                field="K_fluid" if k_in else "rho_fluid",
                expected="both K_fluid and rho_fluid, or neither",
                actual=f"only {'K_fluid' if k_in else 'rho_fluid'}",
            )
        has_state_fluid = k_in and rho_in
        if self.fluid_in_state or has_state_fluid:
            K_fluid, rho_fluid_kgm3 = state.fetch("K_fluid", "rho_fluid")
            if K_fluid.shape != K_dry.shape:
                raise GeoBrainError(
                    "K_fluid must match K_dry shape",
                    object_name="Gassmann",
                    field="K_fluid",
                    expected=tuple(K_dry.shape),
                    actual=tuple(K_fluid.shape),
                )
            if rho_fluid_kgm3.shape != K_dry.shape:
                raise GeoBrainError(
                    "rho_fluid must match K_dry shape",
                    object_name="Gassmann",
                    field="rho_fluid",
                    expected=tuple(K_dry.shape),
                    actual=tuple(rho_fluid_kgm3.shape),
                )
        else:
            K_fluid = self.K_fluid
            rho_fluid_kgm3 = self.rho_fluid

        vp, rho = self._model(K_dry, phi, K_fluid, rho_fluid_kgm3)
        return state.with_tensors(vp=vp, rho=rho)


# --- Wood / Brie operators --------------------------------------------------


class WoodOperator(PropertyTransform):
    """Wood harmonic mixing operator."""

    def __init__(
        self,
        *,
        K_water: float | None = 2.25e9,
        K_other: float = 0.10e9,
        rho_water: float | None = 1000.0,
        rho_other: float = 200.0,
        fluid_in_state: bool = False,
    ) -> None:
        super().__init__()
        if not fluid_in_state:
            for name, value in (("K_water", K_water), ("rho_water", rho_water)):
                if value is None or value <= 0:
                    raise GeoBrainError(
                        f"Wood {name} must be positive (or set fluid_in_state=True)",
                        object_name="Wood",
                        field=name,
                        expected="> 0",
                        actual=value,
                    )
        self._model = _WoodModel(K_other=K_other, rho_other=rho_other)
        self.K_water = float(K_water) if (K_water is not None and not fluid_in_state) else None
        self.K_other = self._model.K_other
        self.rho_water = float(rho_water) if (rho_water is not None and not fluid_in_state) else None
        self.rho_other = self._model.rho_other
        self.fluid_in_state = bool(fluid_in_state)

        trainable = ("Sw", "K_fluid", "rho_fluid") if fluid_in_state else ("Sw",)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=trainable,
            output_keys=("K_fluid", "rho_fluid"),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (Sw,) = state.fetch("Sw")
        check_bounded(Sw, "Sw", owner="Wood")
        if self.fluid_in_state:
            K_water, rho_water = state.fetch("K_fluid", "rho_fluid")
        else:
            K_water = self.K_water
            rho_water = self.rho_water
        K_fluid, rho_fluid = self._model(Sw, K_water, rho_water)
        return state.with_tensors(K_fluid=K_fluid, rho_fluid=rho_fluid)


class BrieOperator(PropertyTransform):
    """Brie 1995 empirical patchy-saturation mixing operator."""

    def __init__(
        self,
        *,
        K_water: float | None = 2.25e9,
        K_other: float = 0.10e9,
        rho_water: float | None = 1000.0,
        rho_other: float = 200.0,
        exponent: float = 3.0,
        fluid_in_state: bool = False,
    ) -> None:
        super().__init__()
        if not fluid_in_state:
            for name, value in (("K_water", K_water), ("rho_water", rho_water)):
                if value is None or value <= 0:
                    raise GeoBrainError(
                        f"Brie {name} must be positive (or set fluid_in_state=True)",
                        object_name="Brie",
                        field=name,
                        expected="> 0",
                        actual=value,
                    )
        self._model = _BrieModel(
            K_other=K_other, rho_other=rho_other, exponent=exponent,
        )
        self.K_water = float(K_water) if (K_water is not None and not fluid_in_state) else None
        self.K_other = self._model.K_other
        self.rho_water = float(rho_water) if (rho_water is not None and not fluid_in_state) else None
        self.rho_other = self._model.rho_other
        self.exponent = self._model.exponent
        self.fluid_in_state = bool(fluid_in_state)

        trainable = ("Sw", "K_fluid", "rho_fluid") if fluid_in_state else ("Sw",)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=trainable,
            output_keys=("K_fluid", "rho_fluid"),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (Sw,) = state.fetch("Sw")
        check_bounded(Sw, "Sw", owner="Brie")
        if self.fluid_in_state:
            K_water, rho_water = state.fetch("K_fluid", "rho_fluid")
        else:
            K_water = self.K_water
            rho_water = self.rho_water
        K_fluid, rho_fluid = self._model(Sw, K_water, rho_water)
        return state.with_tensors(K_fluid=K_fluid, rho_fluid=rho_fluid)


# --- Batzle-Wang EOS family operators ----------------------------------------


def _validate_TP_shape(T, P, op_name: str) -> None:
    if T.shape != P.shape:
        raise GeoBrainError(
            "temperature_C and pressure_MPa must share shape",
            object_name=op_name,
            field="pressure_MPa",
            expected=tuple(T.shape),
            actual=tuple(P.shape),
        )


class BatzleWangBrineOperator(PropertyTransform):
    """Brine ``(K, ρ)`` vs ``(T, P, S)`` operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("temperature_C", "pressure_MPa"),
        output_keys=("K_fluid", "rho_fluid"),
    )

    def __init__(self, *, salinity: float = 0.035) -> None:
        super().__init__()
        self._model = _BWBrineModel(salinity=salinity)
        self.salinity = self._model.salinity

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        T, P = state.fetch("temperature_C", "pressure_MPa")
        _validate_TP_shape(T, P, "BatzleWangBrine")
        K_fluid, rho_fluid = self._model(T, P)
        return state.with_tensors(K_fluid=K_fluid, rho_fluid=rho_fluid)


class BatzleWangGasOperator(PropertyTransform):
    """Hydrocarbon-gas ``(K, ρ)`` vs ``(T, P)`` operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("temperature_C", "pressure_MPa"),
        output_keys=("K_fluid", "rho_fluid"),
    )

    def __init__(self, *, gas_gravity: float = 0.65) -> None:
        super().__init__()
        self._model = _BWGasModel(gas_gravity=gas_gravity)
        self.gas_gravity = self._model.gas_gravity

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        T, P = state.fetch("temperature_C", "pressure_MPa")
        _validate_TP_shape(T, P, "BatzleWangGas")
        K_fluid, rho_fluid = self._model(T, P)
        return state.with_tensors(K_fluid=K_fluid, rho_fluid=rho_fluid)


class BatzleWangOilDeadOperator(PropertyTransform):
    """Dead crude-oil ``(K, ρ)`` vs ``(T, P, ρ₀)`` operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("temperature_C", "pressure_MPa"),
        output_keys=("K_fluid", "rho_fluid"),
    )

    def __init__(self, *, reference_density: float = 0.85) -> None:
        super().__init__()
        self._model = _BWOilDeadModel(reference_density=reference_density)
        self.rho_0 = self._model.rho_0

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        T, P = state.fetch("temperature_C", "pressure_MPa")
        _validate_TP_shape(T, P, "BatzleWangOilDead")
        K_fluid, rho_fluid = self._model(T, P)
        return state.with_tensors(K_fluid=K_fluid, rho_fluid=rho_fluid)


class BatzleWangOilLiveOperator(PropertyTransform):
    """Live crude-oil ``(K, ρ)`` vs ``(T, P, ρ₀, GOR, γ_G)`` operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("temperature_C", "pressure_MPa"),
        output_keys=("K_fluid", "rho_fluid"),
    )

    def __init__(
        self,
        *,
        reference_density: float = 0.85,
        gas_oil_ratio: float = 85.0,
        gas_gravity: float = 0.65,
    ) -> None:
        super().__init__()
        self._model = _BWOilLiveModel(
            reference_density=reference_density,
            gas_oil_ratio=gas_oil_ratio,
            gas_gravity=gas_gravity,
        )
        self.rho_0 = self._model.rho_0
        self.R_G = self._model.R_G
        self.G = self._model.G

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        T, P = state.fetch("temperature_C", "pressure_MPa")
        _validate_TP_shape(T, P, "BatzleWangOilLive")
        K_fluid, rho_fluid = self._model(T, P)
        return state.with_tensors(K_fluid=K_fluid, rho_fluid=rho_fluid)


__all__ = [
    "BatzleWang",
    "BatzleWangBrine", "BatzleWangBrineOperator",
    "BatzleWangGas", "BatzleWangGasOperator",
    "BatzleWangOilDead", "BatzleWangOilDeadOperator",
    "BatzleWangOilLive", "BatzleWangOilLiveOperator",
    "BiotDispersion", "BiotHF",
    "Brie", "BrieOperator",
    "BrownKorringaDry2Sat", "BrownKorringaSat2Dry", "BrownKorringaSub",
    "CO2Brine", "CO2Properties",
    "Gassmann", "GassmannFluidSub", "GassmannInverse", "GassmannOperator",
    "GeertsmaSmitHF", "GeertsmaSmitLF",
    "LiveOil",
    "MavkoJizba",
    "Wood", "WoodOperator",
]
