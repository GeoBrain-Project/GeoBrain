"""Flow wells: one package, two well-modelling formulations.

The split axis is the well-modelling FORMULATION:

- :mod:`.explicit`: wells as schema-declared
  SOURCE TERMS outside the reservoir residual: ``Perforation`` (SI Peaceman
  index), ``WellControl`` (``BHPControl`` / ``RateControl``), ``Well`` +
  ``WellGroup`` producing sparse ``FlowSourceTerms`` phase-mass blocks.
  Keeps the kernel residual a pure function of state, autograd-friendly,
  unit-testable in isolation, model-agnostic.
- :mod:`.implicit`: wells COUPLED INTO the
  residual with one implicit BHP degree of freedom per well
  (:class:`WellSystem`, :class:`SparseBorderedJacobian`), built ON TOP of
  the explicit primitives. Rate-controlled BHP lives here as an augmented
  unknown.

The package root re-exports both formulations, so every well name is
importable directly from ``geobrain.physics.flow.wells``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .explicit import (
    BHPControl,
    FlowSourceTerms,
    Perforation,
    RateControl,
    Well,
    WellGroup,
    WellRateKind,
    WellRateReport,
    WellStandardConditions,
    compute_well_index,
    controlled_rate,
    source_block,
    validate_well_control,
    well_control_residual,
)
from .implicit import SparseBorderedJacobian, WellSystem

__all__ = [
    "BHPControl",
    "FlowSourceTerms",
    "Perforation",
    "RateControl",
    "SparseBorderedJacobian",
    "Well",
    "WellGroup",
    "WellRateKind",
    "WellRateReport",
    "WellStandardConditions",
    "WellSystem",
    "compute_well_index",
    "controlled_rate",
    "source_block",
    "validate_well_control",
    "well_control_residual",
]
