"""Curated public facade for GeoBrain subsurface flow.

Advanced kernels, properties, solvers, compositional models, and external-unit
adapters remain available from their responsibility-specific subpackages.  The
root surface contains only stable construction records, operator facades,
structured diagnostics/errors, and Agent discovery.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .capabilities import (
    FlowCapabilityReport,
    FlowUnsupportedCombination,
    discover_flow_capabilities,
)
from .config import FlowExecutionConfig, FlowHistoryConfig
from .contracts import FlowFieldSpec, FlowModelSchema, FlowResidualBlockSpec
from .errors import (
    FlowCapabilityError,
    FlowContractError,
    FlowConvergenceError,
    FlowResourceError,
)
from .grid import CartGrid
from .operators import (
    FlowEvolutionOperator,
    ParametricFlowOperator,
    TransientFlowOperator,
    WellObservationOperator,
)
from .properties import Rock
from .resources import FlowResourceEstimate, FlowResourceRequest
from .solvers.diagnostics import FlowConvergenceDiagnostics
from .timestep import AdaptiveTimeStepper, TimeStepScheduler
from .wells import (
    BHPControl,
    Perforation,
    RateControl,
    Well,
    WellGroup,
    WellRateKind,
)

__all__ = (
    "AdaptiveTimeStepper",
    "BHPControl",
    "CartGrid",
    "FlowCapabilityError",
    "FlowCapabilityReport",
    "FlowContractError",
    "FlowConvergenceDiagnostics",
    "FlowConvergenceError",
    "FlowEvolutionOperator",
    "FlowExecutionConfig",
    "FlowFieldSpec",
    "FlowHistoryConfig",
    "FlowModelSchema",
    "FlowResidualBlockSpec",
    "FlowResourceError",
    "FlowResourceEstimate",
    "FlowResourceRequest",
    "FlowUnsupportedCombination",
    "ParametricFlowOperator",
    "Perforation",
    "RateControl",
    "Rock",
    "TimeStepScheduler",
    "TransientFlowOperator",
    "Well",
    "WellGroup",
    "WellObservationOperator",
    "WellRateKind",
    "discover_flow_capabilities",
)
