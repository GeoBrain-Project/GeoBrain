"""GeoBrain Rock physics public API.

The public surface is deliberately small: canonical tensor kernels live in
named scientific modules, while terminal ``ForwardOperator`` facades provide
stable schemas, capability reports, resource preflight, and Agent discovery.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .anisotropy import hudson_phase_properties, sayers_kachanov_phase_properties
from .elastic import bulk_modulus_from_velocities, velocities_from_moduli
from .empirical import gardner_density
from .fluid_eos import batzle_wang_brine
from .fluid_substitution import (
    gassmann_saturated_properties,
    gassmann_vp_from_dry_frame,
)
from .granular import hertz_mindlin_moduli
from .inclusions import self_consistent_moduli
from .mixtures import wood_fluid_mix
from .operators import (
    ROCK_OPERATOR_TYPES,
    ArchieResistivity,
    BatzleWangBrine,
    BiotHighFrequency,
    Gardner,
    Gassmann,
    HertzMindlin,
    Hudson,
    KozenyCarman,
    ModuliFromVelocities,
    RockPhysicsTemplate,
    SayersKachanov,
    SelfConsistent,
    VelocitiesFromModuli,
    WoodFluidMix,
    get_rock_operator,
)
from .petrophysics import archie_resistivity, kozeny_carman_permeability
from .poroelastic import biot_high_frequency_limits
from .qi import rock_physics_template

__all__ = [
    "ArchieResistivity",
    "BatzleWangBrine",
    "BiotHighFrequency",
    "Gardner",
    "Gassmann",
    "HertzMindlin",
    "Hudson",
    "KozenyCarman",
    "ModuliFromVelocities",
    "ROCK_OPERATOR_TYPES",
    "RockPhysicsTemplate",
    "SayersKachanov",
    "SelfConsistent",
    "VelocitiesFromModuli",
    "WoodFluidMix",
    "archie_resistivity",
    "batzle_wang_brine",
    "biot_high_frequency_limits",
    "bulk_modulus_from_velocities",
    "gardner_density",
    "gassmann_saturated_properties",
    "gassmann_vp_from_dry_frame",
    "get_rock_operator",
    "hertz_mindlin_moduli",
    "hudson_phase_properties",
    "kozeny_carman_permeability",
    "rock_physics_template",
    "sayers_kachanov_phase_properties",
    "self_consistent_moduli",
    "velocities_from_moduli",
    "wood_fluid_mix",
]
