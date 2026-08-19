"""
Effective-medium ``ComponentModel`` implementations + operator wrappers.

Implementations are split across private submodules grouped by sub-family:

- :mod:`_effective_bounds`:     Voigt/Reuss/VRH/HashinShtrikman/CriticalPorosity
- :mod:`_effective_inclusion`: DEM/KusterToksoz/SC/XuWhite/EshelbyCheng/
  SwissCheese/DiluteCrack/SCDilute/SCFlex/PQ
- :mod:`_effective_hudson`:     HudsonStiffness/Random/Ortho/Cone crack family
- :mod:`_effective_misc`:       MTAverage/OConnellBudiansky/OConnellBudianskyFl

Public symbols are re-exported below so historical imports
(``from geobrain.physics.rock.models.effective import HudsonRandom``,
``from geobrain.physics.rock.models import HudsonRandom``) continue to work.

Models:
    VRH:                   Voigt-Reuss-Hill average for a 2-component mixture
    HashinShtrikman:       Hashin-Shtrikman bounds, Hill-averaged
    Voigt / Reuss:         Iso-strain / iso-stress bounds
    CriticalPorosity:      Nur et al. (1998) critical-porosity model
    DEM:                   Differential Effective Medium (Berryman 1992 ODE)
    KusterToksoz:          Sparse spherical-inclusion closed form
    SelfConsistent:        Berryman 1980 self-consistent EMT
    XuWhite:               Two-pore-shape (oblate-spheroid) clastic rock model
    EshelbyCheng:          Cracked-isotropic VTI stiffness
    SwissCheese:           Dilute spherical pores
    DiluteCrack:           Walsh (1965) non-interacting random cracks
    SCDilute / SCFlex:     Self-consistent two-phase extensions
    PQ:                    Berryman ellipsoidal P, Q strain-concentration factors
    HudsonStiffness:       Aligned penny-shaped cracks (VTI/HTI 6×6)
    HudsonRandom:          Randomly oriented cracks (isotropic)
    HudsonOrtho:           Three orthogonal aligned-crack sets (orthorhombic)
    HudsonCone:            Cone-distributed crack normals (VTI 6×6)
    MTAverage:             Modified Mori-Tanaka three-phase average
    OConnellBudiansky*:    Dry / fluid-saturated penny-shaped cracks

Operator wrappers (merged from rock/effective.py):
    VRHOperator, HashinShtrikmanOperator, DEMOperator, KusterToksozOperator, SelfConsistentOperator, XuWhiteOperator

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
from ._effective_bounds import (
    VRH,
    CriticalPorosity,
    HashinShtrikman,
    Reuss,
    Voigt,
)
from ._effective_hudson import (
    HudsonCone,
    HudsonOrtho,
    HudsonRandom,
    HudsonStiffness,
)
from ._effective_inclusion import (
    DEM,
    PQ,
    DiluteCrack,
    EshelbyCheng,
    KusterToksoz,
    SCDilute,
    SCFlex,
    SelfConsistent,
    SwissCheese,
    XuWhite,
)
from ._effective_misc import (
    MTAverage,
    OConnellBudiansky,
    OConnellBudianskyFl,
)

# ComponentModel names used as __all__ for re-stamping __module__
_COMPONENT_MODEL_NAMES = [
    "CriticalPorosity",
    "DEM",
    "DiluteCrack",
    "EshelbyCheng",
    "HashinShtrikman",
    "HudsonCone",
    "HudsonOrtho",
    "HudsonRandom",
    "HudsonStiffness",
    "KusterToksoz",
    "MTAverage",
    "OConnellBudiansky",
    "OConnellBudianskyFl",
    "PQ",
    "Reuss",
    "SCDilute",
    "SCFlex",
    "SelfConsistent",
    "SwissCheese",
    "VRH",
    "Voigt",
    "XuWhite",
]

# Re-stamp ``__module__`` so the public class repr / introspection
# reports this aggregator module rather than the private submodule a
# class physically lives in. Several tests pin
# ``Cls.__module__ == "geobrain.physics.rock.models.effective"`` to
# distinguish the functional-layer classes from their operator-style
# namesakes (e.g. ``rock.VRH`` vs ``rock.models.VRH``).
for _name in _COMPONENT_MODEL_NAMES:
    _cls = globals()[_name]
    _cls.__module__ = __name__
del _name, _cls


# ============================================================================
# Operator wrappers (PropertyTransform)
#
# Merged from rock/effective.py. These thin wrappers adapt the
# ComponentModel classes above to the ModelState / ForwardContext contract.
# ============================================================================

# Private aliases so operator __init__ can reference the ComponentModel
_VRHModel = VRH
_HSModel = HashinShtrikman
_DEMModel = DEM
_KTModel = KusterToksoz
_SCModel = SelfConsistent
_XWModel = XuWhite


class VRHOperator(PropertyTransform):
    """Voigt-Reuss-Hill average operator."""

    def __init__(self, output_name: str = "K_eff") -> None:
        super().__init__()
        if not output_name:
            raise GeoBrainError(
                "VRH output_name must be a non-empty string",
                object_name="VRH",
                field="output_name",
                expected="non-empty string",
                actual=output_name,
            )
        self._model = _VRHModel()
        self.output_name = output_name
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=("K1", "K2", "f1"),
            output_keys=(output_name,),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K1, K2, f1 = state.fetch("K1", "K2", "f1")
        if not (K1.shape == K2.shape == f1.shape):
            raise GeoBrainError(
                "K1, K2 and f1 must have matching shape",
                object_name="VRH",
                field="K1/K2/f1",
                expected=tuple(K1.shape),
                actual=(tuple(K2.shape), tuple(f1.shape)),
            )
        K_eff = self._model(K1, K2, f1)
        return state.with_tensors(**{self.output_name: K_eff})


class HashinShtrikmanOperator(PropertyTransform):
    """Hashin-Shtrikman bounds (Hill-averaged) operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("K1", "mu1", "K2", "mu2", "f1"),
        output_keys=("K_HS", "mu_HS"),
    )

    def __init__(self) -> None:
        super().__init__()
        self._model = _HSModel()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K1, mu1, K2, mu2, f1 = state.fetch("K1", "mu1", "K2", "mu2", "f1")
        for name, t in zip(("mu1", "K2", "mu2", "f1"), (mu1, K2, mu2, f1)):
            if t.shape != K1.shape:
                raise GeoBrainError(
                    "HashinShtrikman inputs must share shape",
                    object_name="HashinShtrikman",
                    field=name,
                    expected=tuple(K1.shape),
                    actual=tuple(t.shape),
                )
        K_HS, mu_HS = self._model(K1, mu1, K2, mu2, f1)
        return state.with_tensors(K_HS=K_HS, mu_HS=mu_HS)


class DEMOperator(PropertyTransform):
    """Differential Effective Medium operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("phi",),
        output_keys=("K_eff", "mu_eff"),
    )

    def __init__(
        self,
        *,
        K_host: float = 37.0e9,
        mu_host: float = 44.0e9,
        K_inc: float = 2.25e9,
        mu_inc: float = 0.0,
        n_steps: int = 50,
        tolerance: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self._model = _DEMModel(
            K_host=K_host,
            mu_host=mu_host,
            K_inc=K_inc,
            mu_inc=mu_inc,
            n_steps=n_steps,
            tolerance=tolerance,
        )
        self.K_host = self._model.K_host
        self.mu_host = self._model.mu_host
        self.K_inc = self._model.K_inc
        self.mu_inc = self._model.mu_inc
        self.n_steps = self._model.n_steps
        self.tolerance = self._model.tolerance

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (phi,) = state.fetch("phi")
        K_eff, mu_eff = self._model(phi)
        return state.with_tensors(K_eff=K_eff, mu_eff=mu_eff)


class KusterToksozOperator(PropertyTransform):
    """Kuster-Toksöz operator (sparse spherical inclusions)."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("K_mineral", "mu_mineral", "phi"),
        output_keys=("K_KT", "mu_KT"),
    )

    def __init__(
        self,
        *,
        K_inclusion: float = 2.25e9,
        mu_inclusion: float = 0.0,
    ) -> None:
        super().__init__()
        self._model = _KTModel(K_inclusion=K_inclusion, mu_inclusion=mu_inclusion)
        self.K_inclusion = self._model.K_inclusion
        self.mu_inclusion = self._model.mu_inclusion

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K_m, mu_m, phi = state.fetch("K_mineral", "mu_mineral", "phi")
        for name, t in (("mu_mineral", mu_m), ("phi", phi)):
            if t.shape != K_m.shape:
                raise GeoBrainError(
                    "KusterToksoz inputs must share shape",
                    object_name="KusterToksoz",
                    field=name,
                    expected=tuple(K_m.shape),
                    actual=tuple(t.shape),
                )
        if (phi < 0).any() or (phi >= 1).any():
            raise GeoBrainError(
                "KusterToksoz phi must lie in [0, 1)",
                object_name="KusterToksoz",
                field="phi",
                expected="[0, 1)",
                actual=(float(phi.min()), float(phi.max())),
            )
        K_KT, mu_KT = self._model(K_m, mu_m, phi)
        return state.with_tensors(K_KT=K_KT, mu_KT=mu_KT)


class SelfConsistentOperator(PropertyTransform):
    """Self-consistent EMT operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("K1", "mu1", "K2", "mu2", "f1"),
        output_keys=("K_SC", "mu_SC"),
    )

    def __init__(self, *, n_iter: int = 60, tolerance: float = 1.0e-3) -> None:
        super().__init__()
        self._model = _SCModel(n_iter=n_iter, tolerance=tolerance)
        self.n_iter = self._model.n_iter
        self.tolerance = self._model.tolerance

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K1, mu1, K2, mu2, f1 = state.fetch("K1", "mu1", "K2", "mu2", "f1")
        for name, t in (("mu1", mu1), ("K2", K2), ("mu2", mu2), ("f1", f1)):
            if t.shape != K1.shape:
                raise GeoBrainError(
                    "SelfConsistent inputs must share shape",
                    object_name="SelfConsistent",
                    field=name,
                    expected=tuple(K1.shape),
                    actual=tuple(t.shape),
                )
        if (f1 < 0).any() or (f1 > 1).any():
            raise GeoBrainError(
                "SelfConsistent f1 must lie in [0, 1]",
                object_name="SelfConsistent",
                field="f1",
                expected="[0, 1]",
                actual=(float(f1.min()), float(f1.max())),
            )
        K_SC, mu_SC = self._model(K1, mu1, K2, mu2, f1)
        return state.with_tensors(K_SC=K_SC, mu_SC=mu_SC)


class XuWhiteOperator(PropertyTransform):
    """Xu-White two-pore-shape clastic operator."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("V_sh", "phi"),
        output_keys=("K_dry", "mu_dry", "rho_dry"),
    )

    def __init__(
        self,
        *,
        K_quartz: float = 37.0e9,
        mu_quartz: float = 44.0e9,
        rho_quartz: float = 2650.0,
        K_clay: float = 21.0e9,
        mu_clay: float = 7.0e9,
        rho_clay: float = 2580.0,
        alpha_sand: float = 0.12,
        alpha_clay: float = 0.03,
        n_steps: int = 20,
        tolerance: float = 5.0e-2,
    ) -> None:
        super().__init__()
        self._model = _XWModel(
            K_quartz=K_quartz,
            mu_quartz=mu_quartz,
            rho_quartz=rho_quartz,
            K_clay=K_clay,
            mu_clay=mu_clay,
            rho_clay=rho_clay,
            alpha_sand=alpha_sand,
            alpha_clay=alpha_clay,
            n_steps=n_steps,
            tolerance=tolerance,
        )
        # Expose attrs for backward-compat introspection.
        for attr in (
            "K_quartz",
            "mu_quartz",
            "rho_quartz",
            "K_clay",
            "mu_clay",
            "rho_clay",
            "alpha_sand",
            "alpha_clay",
            "n_steps",
            "tolerance",
        ):
            setattr(self, attr, getattr(self._model, attr))

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        V_sh, phi = state.fetch("V_sh", "phi")
        if V_sh.shape != phi.shape:
            raise GeoBrainError(
                "XuWhite V_sh and phi must share shape",
                object_name="XuWhite",
                field="V_sh/phi",
                expected=tuple(phi.shape),
                actual=tuple(V_sh.shape),
            )
        if (V_sh < 0).any() or (V_sh > 1).any():
            raise GeoBrainError(
                "XuWhite V_sh must lie in [0, 1]",
                object_name="XuWhite",
                field="V_sh",
                expected="[0, 1]",
                actual=(float(V_sh.min()), float(V_sh.max())),
            )
        if (phi < 0).any() or (phi >= 1).any():
            raise GeoBrainError(
                "XuWhite phi must lie in [0, 1)",
                object_name="XuWhite",
                field="phi",
                expected="[0, 1)",
                actual=(float(phi.min()), float(phi.max())),
            )
        if (V_sh + phi > 1.0).any():
            raise GeoBrainError(
                "XuWhite V_sh + phi must not exceed 1",
                object_name="XuWhite",
                field="V_sh + phi",
                expected="<= 1",
                actual=float((V_sh + phi).max()),
            )
        K_dry, mu_dry, rho_dry = self._model(V_sh, phi)
        return state.with_tensors(K_dry=K_dry, mu_dry=mu_dry, rho_dry=rho_dry)


__all__ = [
    "CriticalPorosity",
    "DEM",
    "DEMOperator",
    "DiluteCrack",
    "EshelbyCheng",
    "HashinShtrikman",
    "HashinShtrikmanOperator",
    "HudsonCone",
    "HudsonOrtho",
    "HudsonRandom",
    "HudsonStiffness",
    "KusterToksoz",
    "KusterToksozOperator",
    "MTAverage",
    "OConnellBudiansky",
    "OConnellBudianskyFl",
    "PQ",
    "Reuss",
    "SCDilute",
    "SCFlex",
    "SelfConsistent",
    "SelfConsistentOperator",
    "SwissCheese",
    "VRH",
    "VRHOperator",
    "Voigt",
    "XuWhite",
    "XuWhiteOperator",
]
