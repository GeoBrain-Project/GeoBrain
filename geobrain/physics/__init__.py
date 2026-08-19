"""GeoBrain Physics: multi-discipline forward-modelling engine.

The physics package groups five self-contained families of geophysical
forward operators.  Each sub-package owns its own data classes, solvers,
and operator wrappers. Cross-family imports are governed by the layer
contracts (enforced by the architecture governance suite): the single
permitted edge is ``flow -> rock`` (flow's CO2/brine PVT reuses the
retained rock registry); every other family pair is import-free, so a
family can be used without pulling in the others.

Architecture:
    physics/
    ├── wave/         # Seismic: acoustic & elastic FDTD, AVO, Helmholtz
    ├── rock/         # Rock physics: SI kernels + operator facades + the
    │                 #   retained registry library (97 registered models)
    ├── em/           # DC, IP/SIP, MT, CSEM, FDEM, TEM
    ├── potential/    # Gravity & magnetics forward / processing
    ├── flow/         # Reservoir multiphase flow simulation
    └── _families.py  # Private family metadata shared by discovery tooling

Quick Start:
    >>> from geobrain.physics.rock import Gardner
    >>> from geobrain.physics.wave import Acoustic2D
    >>> from geobrain.physics.em import MT1D

Family Overview:
    family              scope
    ──────────────────  ──────────────────────────────────────────────
    wave                Acoustic/elastic wave propagation, AVO
    rock                Constitutive relations (Gardner, Gassmann, VRH, ...)
    em    DC/MT/CSEM/FDEM/TEM
    potential    Gravity, magnetic forward & processing
    flow                Reservoir multiphase (single-phase, oil-water, black-oil)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
