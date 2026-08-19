"""Experimental fail-loud native CUDA backend for GeoBrain Wave.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .capabilities import NativeCapabilityDecision, probe_native_capability
from .dispatch import NativeWaveBackend
from .loader import is_available, probe

__all__ = [
    "NativeCapabilityDecision",
    "NativeWaveBackend",
    "is_available",
    "probe",
    "probe_native_capability",
]
