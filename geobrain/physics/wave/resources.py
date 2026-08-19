"""Resource-estimation door for the wave family (family-template module).

Every physics family exposes its resource estimation under ``resources.py``;
wave's estimator lives with the private engine (it needs the propagation
plan), so this module is the ADDRESSABLE DOOR re-exporting the public-facing
pieces:

- :class:`ResourceEstimate`: the record the packed operators' preflight
  returns (peak bytes by category, work counters).
- :func:`autograd_resource_estimate_supported`: whether calibrated
  autograd estimation is available for a device type.
- :func:`runtime_calibration_registry`: the recorded runtime calibrations
  backing those estimates.

Deliberately NOT in ``geobrain.physics.wave.__all__``: the frozen wave
surface exposes estimates through the operators' own preflight methods;
this door exists for family-template symmetry and direct inspection.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from ._engine.resources import (
    ResourceEstimate,
    autograd_resource_estimate_supported,
    runtime_calibration_registry,
)

__all__ = [
    "ResourceEstimate",
    "autograd_resource_estimate_supported",
    "runtime_calibration_registry",
]
