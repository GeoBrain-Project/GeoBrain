"""Execution backends for the internal Wave engine.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .eager import EagerWaveBackend
from .native import NativeWaveBackend

__all__ = ["EagerWaveBackend", "NativeWaveBackend"]
