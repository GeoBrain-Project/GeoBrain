"""
Time-stepping primitives for transient EM solves (BDF2, schedule).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .bdf2 import bdf2_stepwise
from .schedule import build_log_substep_schedule

__all__ = ["bdf2_stepwise", "build_log_substep_schedule"]
