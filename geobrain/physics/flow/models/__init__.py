"""
Per-fluid-system flow kernels.

Each model exposes:

- ``state_size() -> int``
- ``initial_state(p0) -> Tensor``
- ``residual(state, state_old, dt, **kw) -> Tensor``
- ``jacobian(state, state_old, dt, **kw) -> Tensor``
- ``state_split(state) -> dict[str, Tensor]``

Used by :class:`~geobrain.physics.flow.solvers.NewtonSolver` and (in
later turns) the time-stepping / adjoint operators.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .black_oil import BlackOilModel
from .black_oil_varswitch import BlackOilVarSwitchModel
from .mpfa_single_phase import MPFASinglePhaseModel
from .mpfa_thermal_single_phase import MPFAThermalSinglePhaseModel, MPFAThermalSinglePhaseModel3D
from .mpfa_thermal_three_phase import MPFAThermalThreePhaseModel, MPFAThermalThreePhaseModel3D
from .mpfa_thermal_two_phase import MPFAThermalTwoPhaseModel, MPFAThermalTwoPhaseModel3D
from .mpfa_three_phase import MPFAThreePhaseModel, MPFAThreePhaseModel3D
from .mpfa_two_phase import MPFATwoPhaseModel, MPFATwoPhaseModel3D
from .oil_water import OilWaterModel
from .single_phase import SinglePhaseModel
from .thermal_single_phase import ThermalSinglePhaseModel

__all__ = (
    "BlackOilModel",
    "BlackOilVarSwitchModel",
    "MPFASinglePhaseModel",
    "MPFAThermalSinglePhaseModel",
    "MPFAThermalSinglePhaseModel3D",
    "MPFAThermalThreePhaseModel",
    "MPFAThermalThreePhaseModel3D",
    "MPFAThermalTwoPhaseModel",
    "MPFAThermalTwoPhaseModel3D",
    "MPFAThreePhaseModel",
    "MPFAThreePhaseModel3D",
    "MPFATwoPhaseModel",
    "MPFATwoPhaseModel3D",
    "OilWaterModel",
    "SinglePhaseModel",
    "ThermalSinglePhaseModel",
)
