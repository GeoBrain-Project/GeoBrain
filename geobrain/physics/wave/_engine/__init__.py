"""Cohesive internal contracts for Wave propagation implementations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .contracts import (
    CompiledAcquisition,
    ExecutionAccounting,
    PropagationRequest,
    PropagationResult,
    WaveBackendProtocol,
    WaveEquationDeclaration,
    WaveEquationProtocol,
    WaveMemoryGuarantees,
    WaveMemoryProtocol,
)
from .geometry import compile_acquisition
from .backends import EagerWaveBackend
from .resources import (
    ResourceEstimate,
    allocate_with_budget,
    autograd_resource_estimate_supported,
    estimate_resources,
    runtime_calibration_registry,
)
from .results import assemble_forward_output
from .sampling import inject_sources, sample_receivers

__all__ = [
    "CompiledAcquisition",
    "EagerWaveBackend",
    "ExecutionAccounting",
    "PropagationRequest",
    "PropagationResult",
    "ResourceEstimate",
    "WaveBackendProtocol",
    "WaveEquationDeclaration",
    "WaveEquationProtocol",
    "WaveMemoryGuarantees",
    "WaveMemoryProtocol",
    "allocate_with_budget",
    "autograd_resource_estimate_supported",
    "assemble_forward_output",
    "compile_acquisition",
    "estimate_resources",
    "inject_sources",
    "sample_receivers",
    "runtime_calibration_registry",
]
