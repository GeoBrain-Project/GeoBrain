"""GeoBrain Electromagnetics: DC resistivity, MT, CSEM, FDEM, and TEM forward modelling.

Provides finite-element and semi-analytic forward operators for five
electromagnetic methods: DC resistivity (2D/3D), magnetotellurics (1D/2D/3D),
controlled-source EM (CSEM 1D), frequency-domain EM (FDEM 3D, HEM), and
time-domain EM (TEM 1D/3D, VTEM). Shared building blocks (data classes,
material properties, survey definitions) live in ``materials``, ``surveys``,
and ``results``; numerical
primitives (finite-volume, Hankel transforms, sparse solvers, time-stepping)
live in ``numerics/``.

Architecture:
    em/
    ├── materials.py        # Material parameter descriptors
    ├── surveys.py          # Shared source, receiver, and survey records
    ├── results.py          # Field components and complex result storage
    ├── static/             # DC resistivity (2D/3D), induced polarization
    ├── natural_source/     # Magnetotellurics (MT 1D/2D/3D)
    ├── frequency_domain/   # CSEM 1D, FDEM 3D, helicopter EM (HEM)
    ├── time_domain/        # TEM 1D/3D, VTEM
    └── numerics/           # Finite-volume, Hankel, sparse solvers, time-stepping

Public surface (semver-relevant) policy
---------------------------------------
``__all__`` advertises the reviewed operators, surveys, sources, and shared
contracts. Matrix-assembly / quadrature internals
(``assemble_dc2d_system``, ``assemble_dc3d_system``, ``assemble_mt2d_system``,
``mt2d_jacobian``, ``wavenumber_quadrature``) are NOT advertised at the root;
their canonical home is the leaf module (e.g.
``from geobrain.physics.em.static.dc2d import assemble_dc2d_system``).

Quick Start:
    >>> import torch
    >>> from geobrain.physics.em import MT1D, MT1DSurvey
    >>> survey = MT1DSurvey(frequencies=torch.tensor([0.01, 0.1, 1.0]),
    ...                   layer_thickness=torch.tensor([100.0, 200.0]))
    >>> op = MT1D(survey)   # layer resistivities are the model, passed at forward()

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

# =========================================================================
# Natural Source: Magnetotellurics
# =========================================================================
from .config import EMExecutionConfig
from .natural_source.mt1d import MT1D, MT1DSurvey
from .natural_source.mt2d import MT2D, MT2DSurvey
from .natural_source.mt3d import MT3D, MT3DStation, MT3DSurvey

# =========================================================================
# Static: DC Resistivity & Induced Polarization
# =========================================================================
from .static.dc2d import DC2D, DC2DSurvey
from .static.dc25d import DC25D, DC25DSurvey
from .static.dc3d import (
    DC3D,
    DC3DSurvey,
    BoundDipoleDipoleSurvey,
    DipoleDipoleSurvey,
)
from .static.ip import (
    IP2D,
    IP3D,
    IPChargeabilityModel,
    IPSimulator,
)
from .static.self_potential import SelfPotential2D, SelfPotentialSurvey
from .static.sip import SIP, SIPColeColeModel, SIPSurvey

# =========================================================================
# Frequency Domain: CSEM, FDEM, helicopter EM
# =========================================================================
from .frequency_domain import (
    CSEM1D,
    CSEMReceiver,
    FDEM3D,
    FDEM3DReceiver,
    FDEM3DSurvey,
    FDEMCyl,
    FDEMCylSurvey,
    HEM,
    HEMSurvey,
    MagneticDipoleSource,
    MarineCSEM1DSurvey,
    VMDSource,
)

# =========================================================================
# Time Domain: TEM (1D/3D), VTEM (airborne)
# =========================================================================
from .time_domain import (
    CentralLoopSource,
    TEM1D,
    TEM1DReceiver,
    TEM1DSurvey,
    WaveformTEM1D,
    TEM3D,
    TEM3DReceiver,
    TEM3DSurvey,
    VTEM,
    VTEMSurvey,
)
from .materials import Conductivity, Permeability, Permittivity, Resistivity
from .results import ComplexData, FieldComponent
from .surveys import (
    EMReceiver,
    EMSurvey,
    EMSource,
    FrequencyDomainSurvey,
    GalvanicSurvey,
    TimeDomainSurvey,
    TimeWaveform,
)


# Advertised root API (semver-relevant; frozen by the per-family governance test).
__all__ = [
    # Static: DC / IP / SIP / SP
    "DC2D",
    "DC2DSurvey",
    "DC25D",
    "DC25DSurvey",
    "DC3D",
    "DC3DSurvey",
    "DipoleDipoleSurvey",
    "EMExecutionConfig",
    "BoundDipoleDipoleSurvey",
    "SelfPotential2D",
    "SelfPotentialSurvey",
    "IP2D",
    "IP3D",
    "IPChargeabilityModel",
    "IPSimulator",
    "SIP",
    "SIPColeColeModel",
    "SIPSurvey",
    # Natural source: MT
    "MT1D",
    "MT2D",
    "MT2DSurvey",
    "MT3D",
    "MT3DStation",
    "MT3DSurvey",
    "MT1DSurvey",
    # Frequency domain: CSEM / FDEM / HEM
    "CSEM1D",
    "CSEMReceiver",
    "FDEM3D",
    "FDEM3DReceiver",
    "FDEM3DSurvey",
    "FDEMCyl",
    "FDEMCylSurvey",
    "HEM",
    "HEMSurvey",
    "MagneticDipoleSource",
    "MarineCSEM1DSurvey",
    "VMDSource",
    # Time domain: TEM / VTEM
    "CentralLoopSource",
    "TEM1D",
    "WaveformTEM1D",
    "TEM1DReceiver",
    "TEM1DSurvey",
    "TEM3D",
    "TEM3DReceiver",
    "TEM3DSurvey",
    "VTEM",
    "VTEMSurvey",
    # Canonical shared SI records
    "ComplexData",
    "Conductivity",
    "EMReceiver",
    "EMSource",
    "EMSurvey",
    "FieldComponent",
    "FrequencyDomainSurvey",
    "GalvanicSurvey",
    "Permeability",
    "Permittivity",
    "Resistivity",
    "TimeDomainSurvey",
    "TimeWaveform",
]
