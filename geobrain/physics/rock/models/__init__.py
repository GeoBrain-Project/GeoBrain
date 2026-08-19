"""
Concrete rock-physics ``ComponentModel`` implementations.

Pure-math layer. Each model inherits from one of the category bases in
this package (``AnisotropyModel``, ``FluidModel``, …, re-exported at the
package root) and registers itself with :func:`~geobrain.physics.rock.models.register`.

**Status: the platform's extended rock-physics research library
(addressable tier, kept permanently).** The registry and its ~100
registered models exist for research breadth. Nothing here appears in
:data:`geobrain.physics.rock.__all__`, so nothing here is covered by the
``ROCK_SURFACE`` freeze or reachable by Agent discovery; every module path
is addressable-tier (import-smoked and registry-tested by the
architecture and rock test suites).

The public rock surface is a different pair of layers, and neither of them
delegates to this package:

- **canonical SI tensor kernels**: the named scientific modules one level up
  (:mod:`~geobrain.physics.rock.empirical`,
  :mod:`~geobrain.physics.rock.anisotropy`,
  :mod:`~geobrain.physics.rock.granular`, and siblings). These are the only
  scientific implementations the platform retains as canon; self-contained.
- **terminal ``ForwardOperator`` facades**: :mod:`~geobrain.physics.rock.operators`,
  carrying the schema / capability report / resource preflight / discovery
  contract.

Module names here deliberately mirror the kernel modules one level up
(``empirical``, ``anisotropy``, ``granular``, …) because both cover the same
physics from the two sides of that cutover. When editing, confirm which layer
you are in.

To wrap a model from this package into the operator graph, use
:class:`geobrain.physics.rock.models.RockPhysicsTransform`: kept off the
public surface; for new work prefer
:class:`~geobrain.physics.rock.operators.RockForwardOperator`.

## Categories

``empirical``, ``granular``, ``fluid``, ``effective``, ``anisotropy``,
``permeability``, ``resistivity``; the category bases are
re-exported at this package root. Query the registry for what is
actually registered rather than consulting a hand-maintained inventory (an
earlier one in this docstring had drifted well out of date)::

    from geobrain.physics.rock.models import list_categories, list_models

    list_categories()
    list_models("granular")

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

# =========================================================================
# Registry machinery: the model library is ONE self-contained subtree;
# the package root is its single import door (machinery modules are
# private by design).
# =========================================================================
from ._base import BaseModel, ComponentModel, CompositeModel, RockPhysicsTransform
from ._categories import (
    AnisotropyModel,
    EffectiveModel,
    EmpiricalModel,
    FluidModel,
    GranularModel,
    PermeabilityModel,
    ResistivityModel,
)
from ._registry import (
    create_model,
    get_model,
    get_model_metadata,
    list_aliases,
    list_categories,
    list_models,
    register,
)
from ._types import (
    BRIE_E,
    CN,
    EPS,
    F,
    P,
    PHI_C,
    PHI_C_CEMENTED,
    PI,
    Tensor,
    TensorLike,
    as_tensor,
    ensure_same_device,
)

from .anisotropy import (
    Backus, BackusOperator as BackusOperator,
    BondTransform, BrownKorringa,
    BrownKorringaOperator as BrownKorringaOperator,
    Hudson, HudsonOperator as HudsonOperator,
    SayersKachanov, SayersKachanovOperator as SayersKachanovOperator,
    Thomsen, ThomsenOperator as ThomsenOperator, ThomsenTsvankin,
    VelocityAzimuthHTI, VelocityAzimuthVTI,
)
from .effective import (
    CriticalPorosity, DEM, DEMOperator as DEMOperator, DiluteCrack, EshelbyCheng,
    HashinShtrikman, HashinShtrikmanOperator as HashinShtrikmanOperator,
    HudsonCone, HudsonOrtho, HudsonRandom, HudsonStiffness,
    KusterToksoz, KusterToksozOperator as KusterToksozOperator,
    MTAverage, OConnellBudiansky, OConnellBudianskyFl, PQ, Reuss, SCDilute,
    SCFlex, SelfConsistent, SelfConsistentOperator as SelfConsistentOperator,
    SwissCheese, VRH, VRHOperator as VRHOperator, Voigt,
    XuWhite, XuWhiteOperator as XuWhiteOperator,
)
from .empirical import (
    Castagna, CastagnaOperator as CastagnaOperator,
    CastagnaMudrock, DensityModel, Ehrenberg, Gardner,
    GardnerOperator as GardnerOperator,
    GreenbergCastagna, GreenbergCastagnaOperator as GreenbergCastagnaOperator,
    Han, HanOperator as HanOperator,
    Hillis, Hjelstuen, Japsen, Krief, MacBeth, MacBethOperator as MacBethOperator,
    RammPorosity, RaymerHuntGardner, Scherbaum, Sclater, StPeter, Storvoll,
    WyllieTimeAverage,
)
from .fluid import (
    BatzleWang, BatzleWangBrine,
    BatzleWangBrineOperator as BatzleWangBrineOperator,
    BatzleWangGas, BatzleWangGasOperator as BatzleWangGasOperator,
    BatzleWangOilDead, BatzleWangOilDeadOperator as BatzleWangOilDeadOperator,
    BatzleWangOilLive, BatzleWangOilLiveOperator as BatzleWangOilLiveOperator,
    BiotDispersion, BiotHF, Brie, BrieOperator as BrieOperator,
    BrownKorringaDry2Sat, BrownKorringaSat2Dry, BrownKorringaSub,
    CO2Brine, CO2Properties,
    Gassmann, GassmannFluidSub, GassmannInverse,
    GassmannOperator as GassmannOperator,
    GeertsmaSmitHF, GeertsmaSmitLF,
    LiveOil, MavkoJizba, Wood, WoodOperator as WoodOperator,
)
from .granular import (
    ConstantCement, ContactCement, ContactCementFull, Digby, Diluting,
    HertzMindlin, HertzMindlinOperator as HertzMindlinOperator, MUHS, PCM,
    ShalySand, SiltyShale, SoftSand, SoftSandOperator as SoftSandOperator,
    StiffSand, StiffSandOperator as StiffSandOperator,
    ThomasStieber, VPCM, Walton, WaltonOperator as WaltonOperator,
    hm_moduli_v05 as hm_moduli_v05,
    soft_sand_math as soft_sand_math,
)
from .permeability import (
    Bernabe, Bloch, Fredrich, KozenyCarman, KozenyCarmanPercolation,
    Owolabi, PandaLake, PandaLakeCem, PermLogs, Revil,
)
from .resistivity import ArchieResistivity
from .composite import RockPhysicsWorkflow

# NOTE: The ``*Op`` operator-wrapper classes (GardnerOperator, GassmannOperator, …) are
# imported above so they remain reachable from this module's namespace for
# the top-level ``rock`` package, which binds each to its short alias
# (``rock.Gardner`` -> ``GardnerOperator``). They are deliberately NOT in
# ``__all__``: the only public operator name is ``rock.<Name>`` (the alias),
# and ``rock.models.<Name>`` stays the relation/``ComponentModel`` model.
__all__ = [
    "ArchieResistivity",
    "Backus",
    "BatzleWang",
    "BatzleWangBrine",
    "BatzleWangGas",
    "BatzleWangOilDead",
    "BatzleWangOilLive",
    "Bernabe", "BiotDispersion", "BiotHF", "Bloch", "BondTransform",
    "Brie", "BrownKorringa",
    "BrownKorringaDry2Sat", "BrownKorringaSat2Dry", "BrownKorringaSub",
    "CO2Brine", "CO2Properties",
    "Castagna", "CastagnaMudrock",
    "ConstantCement", "ContactCement", "ContactCementFull",
    "CriticalPorosity",
    "DEM", "DensityModel", "Digby", "DiluteCrack", "Diluting",
    "Ehrenberg", "EshelbyCheng",
    "Fredrich",
    "Gardner",
    "Gassmann", "GassmannFluidSub", "GassmannInverse",
    "GeertsmaSmitHF", "GeertsmaSmitLF",
    "GreenbergCastagna",
    "Han",
    "HashinShtrikman",
    "HertzMindlin",
    "Hillis", "Hjelstuen", "Hudson",
    "HudsonCone", "HudsonOrtho", "HudsonRandom", "HudsonStiffness",
    "Japsen",
    "KozenyCarman", "KozenyCarmanPercolation", "Krief",
    "KusterToksoz",
    "LiveOil",
    "MTAverage", "MUHS", "MacBeth", "MavkoJizba",
    "OConnellBudiansky", "OConnellBudianskyFl", "Owolabi",
    "PCM", "PQ", "PandaLake", "PandaLakeCem", "PermLogs",
    "RammPorosity", "RaymerHuntGardner", "Reuss", "Revil",
    "SCDilute", "SCFlex", "SayersKachanov",
    "Scherbaum", "Sclater",
    "SelfConsistent",
    "ShalySand", "SiltyShale", "SoftSand",
    "StPeter", "StiffSand", "Storvoll", "SwissCheese",
    "ThomasStieber", "Thomsen", "ThomsenTsvankin",
    "VPCM",
    "VRH", "VelocityAzimuthHTI", "VelocityAzimuthVTI", "Voigt",
    "Walton", "Wood", "WyllieTimeAverage",
    "XuWhite",
    "RockPhysicsWorkflow",
]
