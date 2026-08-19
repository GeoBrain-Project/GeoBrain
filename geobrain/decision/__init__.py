"""Decision support with explicit decision and information currencies.

The package separates value of perfect information, Bayes-action utility and
accuracy gain, mutual information in nats, and closed-loop orchestration.

Module layout::

    decision/
    ├── voi.py                # Value of (perfect) information
    ├── accuracy.py           # Bayes-action utility / expected accuracy gain
    ├── mutual_information.py # Mutual information (nats)
    ├── eoi.py        # Spatial expected overall improvement maps
    ├── closed_loop.py        # Acquisition-decision closed-loop orchestration
    ├── protocols.py          # Decision-side structural protocols
    ├── status.py             # Run status records
    └── _metadata.py          # Private shared result-metadata helpers

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .voi import ValueOfInformation, VOIResult
from .accuracy import (
    DecisionAccuracyResult,
    expected_accuracy_gain,
    expected_utility_gain,
)
from .mutual_information import (
    MutualInformationEstimator,
    MutualInformationResult,
)
from .status import DecisionRunStatus
from .protocols import CancellationCheck, EnsembleUpdater, HistoryPolicy
from .eoi import (
    SpatialDecisionAccuracy,
)
from .closed_loop import ClosedLoopManager, ClosedLoopRunResult, ClosedLoopStep

__all__ = [
    'ValueOfInformation',
    'VOIResult',
    'expected_accuracy_gain',
    'expected_utility_gain',
    'MutualInformationEstimator',
    'MutualInformationResult',
    'DecisionRunStatus',
    'CancellationCheck',
    'EnsembleUpdater',
    'HistoryPolicy',
    'SpatialDecisionAccuracy',
    'DecisionAccuracyResult',
    'ClosedLoopManager',
    'ClosedLoopRunResult',
    'ClosedLoopStep',
]
