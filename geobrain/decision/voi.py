"""
Value of Information (VOI) analysis.

Computes the Value of Perfect Information (VOPI): the upper bound on what
additional data is worth before a decision must be made:

    VOI = E_m[max_d obj(d, m)] - max_d E_m[obj(d, m)]

Only requires a decision-model objective matrix [n_decisions, n_samples].

Design note (accepted platform deviation): the decision metrics in this module
are non-differentiable (argmax / max over decisions), so they are computed
directly on the objective tensor with no autograd graph. This is intentional;
VOI is a scoring layer over a *frozen* posterior ensemble, not a forward model.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from geobrain.core.errors import GeoBrainError

logger = logging.getLogger(__name__)


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class VOIResult:
    """
    Result of Value of Information analysis.

    Attributes:
        voi: NET value of (perfect) information, the gross VOPI after the
            ``data_cost`` is subtracted and the result floored at 0
            (``max(0, voi_gross - data_cost)``). Equal to :attr:`voi_gross`
            when ``data_cost == 0``.
        voi_gross: GROSS Value of Perfect Information, before ``data_cost`` and
            before the non-negativity floor: ``preposterior_value -
            prior_optimal_value`` (always ``>= 0`` by Jensen). This is the
            data-cost-independent score of the acquisition itself.
        prior_optimal_value: Expected value of the best decision under
            current information (max_d E_m[obj]).
        prior_optimal_decision: The decision achieving prior_optimal_value.
        preposterior_value: Expected value with perfect information
            (E_m[max_d obj]).
        per_sample_best_values: Best objective per sample [n_samples].
        per_sample_best_decisions: Decision index achieving best per sample.
        decision_values: Full objective matrix [n_decisions, n_samples].
        total_time: Wall-clock time (seconds).
        config: Configuration dict.
    """
    voi: float
    prior_optimal_value: float
    prior_optimal_decision: Any
    preposterior_value: float
    per_sample_best_values: torch.Tensor
    per_sample_best_decisions: list[Any]
    decision_values: torch.Tensor
    total_time: float
    config: dict[str, Any]
    voi_gross: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary string (see :meth:`to_dict` for a
        JSON-safe mapping)."""
        lines = [
            "=== Value of Information Result ===",
            f"VOI (net, perfect info) : {self.voi:.4f}",
            f"VOI (gross, pre-cost)   : {self.voi_gross:.4f}",
            f"Prior optimal value     : {self.prior_optimal_value:.4f}",
            f"Prior optimal decision  : {self.prior_optimal_decision}",
            f"Preposterior value      : {self.preposterior_value:.4f}",
            f"Decisions evaluated     : {self.decision_values.shape[0]}",
            f"Ensemble size           : {self.decision_values.shape[1]}",
            f"Total time              : {self.total_time:.2f} s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary (numpy arrays)."""
        return {
            'voi': self.voi,
            'voi_gross': self.voi_gross,
            'prior_optimal_value': self.prior_optimal_value,
            'prior_optimal_decision': self.prior_optimal_decision,
            'preposterior_value': self.preposterior_value,
            'per_sample_best_values': self.per_sample_best_values.cpu().numpy(),
            'per_sample_best_decisions': self.per_sample_best_decisions,
            'decision_values': self.decision_values.cpu().numpy(),
            'total_time': self.total_time,
            'config': self.config,
        }


# =============================================================================
# VOI Analyzer
# =============================================================================

class ValueOfInformation:
    """
    Value of Perfect Information (VOPI) calculator.

    Args:
        objective_fn: Callable ``(ensemble, decisions) -> Tensor`` that returns
            an objective matrix of shape ``[n_decisions, n_samples]``.
        decisions: List of decision alternatives (any hashable type).
        device: Torch device.

    Example:
        >>> voi_calc = ValueOfInformation(
        ...     objective_fn=my_obj_fn,
        ...     decisions=['A', 'B', 'C'],
        ... )
        >>> result = voi_calc.compute(ensemble)
        >>> print(result.summary())
    """

    def __init__(
        self,
        objective_fn: Callable[[torch.Tensor, Sequence[Any]], torch.Tensor],
        decisions: Sequence[Any],
        device: torch.device | None = None,
    ):
        if len(decisions) < 2:
            raise GeoBrainError(
                "VOI analysis needs at least 2 decisions",
                object_name="ValueOfInformation", field="decisions",
                expected=">= 2 decision alternatives", actual=len(decisions),
            )
        self.objective_fn = objective_fn
        self.decisions = decisions
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ensemble: torch.Tensor,
        data_cost: float = 0.0,
        verbose: bool = False,
    ) -> VOIResult:
        """
        Compute the Value of Perfect Information.

        Args:
            ensemble: Posterior samples, shape ``[n_samples, dim]``.
            data_cost: Cost of acquiring perfect information.  The net VOI
                is ``max(0, voi_gross - data_cost)``; ``voi_gross`` on the
                result is the cost-independent gross value.
            verbose: Log progress.

        Returns:
            VOIResult with VOI value and supporting diagnostics.
        """
        t_start = time.perf_counter()

        if verbose:
            logger.info(
                "Computing VOI: %d decisions, %d samples",
                len(self.decisions), ensemble.shape[0],
            )

        # Build objective matrix [n_decisions, n_samples]
        obj_matrix = self.objective_fn(ensemble, self.decisions)

        if obj_matrix.shape != (len(self.decisions), ensemble.shape[0]):
            raise GeoBrainError(
                "objective_fn returned the wrong shape",
                object_name="ValueOfInformation", field="objective_fn",
                expected=[len(self.decisions), ensemble.shape[0]],
                actual=list(obj_matrix.shape),
            )
        # A non-finite objective silently corrupts the max/argmax below (a NaN
        # entry poisons the reduction and yields a garbage decision index), so
        # reject it up front rather than return a meaningless VOI.
        if not bool(torch.isfinite(obj_matrix).all()):
            raise GeoBrainError(
                "objective_fn returned a non-finite value (NaN/Inf); check the "
                "decision model / ensemble used to build the objective matrix",
                object_name="ValueOfInformation", field="objective_fn",
                expected="all-finite objective matrix", actual="NaN/Inf present",
            )

        if self.device is not None:
            obj_matrix = obj_matrix.to(self.device)

        # --- Prior optimal: max_d E_m[obj(d, m)] ---
        expected = obj_matrix.mean(dim=1)            # [n_decisions]
        prior_best_idx = int(expected.argmax())
        prior_value = float(expected[prior_best_idx])
        prior_decision = self.decisions[prior_best_idx]

        # --- Preposterior: E_m[max_d obj(d, m)] ---
        best_per_sample_vals, best_per_sample_idx = obj_matrix.max(dim=0)
        preposterior_value = float(best_per_sample_vals.mean())
        per_sample_decisions = [
            self.decisions[int(idx)] for idx in best_per_sample_idx
        ]

        # --- VOI ---
        voi_gross = preposterior_value - prior_value
        voi = max(0.0, voi_gross - data_cost)

        elapsed = time.perf_counter() - t_start

        if verbose:
            logger.info(
                "VOI = %.4f (gross=%.4f, preposterior=%.4f, prior_optimal=%.4f, "
                "data_cost=%.4f)",
                voi, voi_gross, preposterior_value, prior_value, data_cost,
            )

        return VOIResult(
            voi=voi,
            voi_gross=voi_gross,
            prior_optimal_value=prior_value,
            prior_optimal_decision=prior_decision,
            preposterior_value=preposterior_value,
            per_sample_best_values=best_per_sample_vals,
            per_sample_best_decisions=per_sample_decisions,
            decision_values=obj_matrix,
            total_time=elapsed,
            config={
                'n_decisions': len(self.decisions),
                'n_samples': ensemble.shape[0],
                'data_cost': data_cost,
            },
        )
