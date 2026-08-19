"""Shared execution status for decision analyses.

This module establishes the result vocabulary only. Cancellation checks
and partial execution live in the bounded-execution engine.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from enum import Enum


class DecisionRunStatus(str, Enum):
    """Terminal status of a decision-analysis run."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"


__all__ = ["DecisionRunStatus"]
