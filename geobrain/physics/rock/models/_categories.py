"""
Category-specific base classes for rock-physics models.

Each concrete model inherits from one of these category bases (which in
turn inherits from :class:`ComponentModel`). The category serves three
purposes:

1. **Type discrimination**: an ``EffectiveModel`` cannot be used where a
   ``FluidModel`` is expected, even if both implement ``forward``.
2. **Workflow hooks**: each category declares the standard method
   :class:`RockPhysicsWorkflow` calls during orchestration (e.g.
   ``FluidModel.compute_fluid_mix``).
3. **Registry inference**: :class:`RockPhysicsRegistry` infers a
   model's category from its base class, so the user doesn't need to
   pass ``category="..."`` on every ``@register`` decoration.

Note: AVO reflectivity classes are **not** under
:class:`AVOModel` here; they live in :mod:`geobrain.physics.wave.reflectivity`
because reflection-coefficient math is seismic, not rock-physics.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from ._base import ComponentModel


class EffectiveModel(ComponentModel):
    """
    Effective-medium models (mixing minerals / inclusions / pores).

    Examples: Voigt, Reuss, VRH, Hashin–Shtrikman, DEM, Self-Consistent,
    Critical Porosity, Hudson cracks, Kuster–Toksöz.
    """


class FluidModel(ComponentModel):
    """
    Fluid models (mixing, substitution, equation-of-state).

    Examples: Gassmann, Wood, Brie, Batzle–Wang (brine / gas / oil /
    CO₂), Brown–Korringa, Biot dispersion, Mavko–Jizba.
    """

    def compute_fluid_mix(self, fl1_K, fl2_K, fl1_rho, fl2_rho, Sw, **params):
        """
        Workflow hook for two-fluid mixing.

        Default raises ``NotImplementedError``; override for any model
        that wants to participate in workflow saturation handling.

        Args:
            fl1_K, fl2_K: Bulk moduli of phases 1 / 2 (Pa, SI).
            fl1_rho, fl2_rho: Densities of phases 1 / 2 (kg/m³, SI).
            Sw: Volume fraction of phase 1 (e.g. water saturation).

        Returns:
            ``(K_eff, rho_eff)`` of the mix.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compute_fluid_mix(). "
            "Override this method for workflow integration."
        )


class GranularModel(ComponentModel):
    """
    Granular-medium models (contact mechanics, friable / cemented sands).

    Examples: Hertz-Mindlin, Soft Sand, Stiff Sand, Contact Cement,
    Walton, MUHS, PCM, Thomas-Stieber, shaly-sand variants.

    Both entry points are SI: ``forward(P_eff)`` (the operator /
    ``PropertyTransform`` path) and ``compute_dry_rock(K0, G0, phi, ..., P=...)``
    (the geomodel-workflow hook used by the composite / soft-sand /
    stiff-sand builders) both take moduli and effective pressure in Pa and
    return moduli in Pa, both paths share one SI unit convention (no
    GPa/MPa variant exists for either).
    """


class EmpiricalModel(ComponentModel):
    """
    Empirical / regression rock-physics relations.

    Examples: Han, Castagna mudrock, Greenberg-Castagna, Gardner,
    Krief, Wyllie time-average, Raymer-Hunt-Gardner, compaction trends.
    """


class AnisotropyModel(ComponentModel):
    """
    Anisotropy parameterizations and transforms.

    Examples: Thomsen parameters, Backus averaging, Bond rotation,
    velocity-azimuth fits (HTI / VTI), Thomsen-Tsvankin (orthorhombic).
    """


class PermeabilityModel(ComponentModel):
    """
    Permeability-from-porosity correlations.

    Examples: Kozeny-Carman, Owolabi, Panda-Lake, Revil, Fredrich,
    Bloch, Bernabe.
    """


class ResistivityModel(ComponentModel):
    """
    Resistivity / formation-factor relations.

    Examples: Archie's law, Indonesia equation, Simandoux, Waxman-Smits.
    """


__all__ = [
    "AnisotropyModel",
    "EffectiveModel",
    "EmpiricalModel",
    "FluidModel",
    "GranularModel",
    "PermeabilityModel",
    "ResistivityModel",
]
