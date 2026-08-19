"""
Experimental variogram + auto-fit.

Provides

- :class:`ExperimentalVariogram`: holds the binned ``(lags, gammas,
  pairs)`` triple plus optional direction info, and offers
  :meth:`fit_model` / :meth:`fit_model_with_diagnostics`;
- the diagnostics dataclasses :class:`VariogramFitAttempt`,
  :class:`VariogramFitDiagnostics`, :class:`VariogramFitResult`;
- the underlying weighted-least-squares fit routine that tries
  ``scipy.optimize.minimize`` with L-BFGS-B first, then falls back to
  Nelder-Mead.

``kind="auto"`` fits Spherical, Exponential, and Gaussian, then picks
the lowest-cost model.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, cast

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_bool_array, as_float_array, as_int_array
from .covariance import CovarianceModel
from .variogram_kernel import VariogramKernel

__all__ = [
    "ExperimentalVariogram",
    "VariogramFitAttempt",
    "VariogramFitDiagnostics",
    "VariogramFitResult",
]


# ----------------------------------------------------------------------
# Diagnostics dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VariogramFitAttempt:
    """One optimizer attempt during variogram fitting.

    Attributes:
        kind / optimizer: model type and optimizer tried.
        success / accepted: solver success and acceptance flags.
        cost: objective value reached.
        message: solver message.
    """

    kind: str
    optimizer: str
    success: bool
    accepted: bool
    cost: float | None
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "optimizer": self.optimizer,
            "success": self.success,
            "accepted": self.accepted,
            "cost": self.cost,
            "message": self.message,
        }


@dataclass(frozen=True)
class VariogramFitDiagnostics:
    """Per-fit diagnostics: set of attempts + selected kernel kind.

    Attributes:
        requested_kind / selected_kind: asked vs chosen model type.
        attempts: individual :class:`VariogramFitAttempt` rows.
        fallback_used: whether a fallback model was selected.
        warnings: fit warnings.
    """

    requested_kind: str
    selected_kind: str | None
    attempts: tuple[VariogramFitAttempt, ...] = ()
    fallback_used: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        if self.selected_kind is None:
            return False
        return any(
            a.kind == self.selected_kind and a.accepted for a in self.attempts
        )


@dataclass(frozen=True)
class VariogramFitResult:
    """Fitted CovarianceModel paired with its diagnostics.

    Attributes:
        model: the fitted covariance model.
        diagnostics: the full fit story.
    """

    model: CovarianceModel
    diagnostics: VariogramFitDiagnostics


@dataclass(frozen=True)
class _SingleFitResult:
    model: CovarianceModel | None
    attempts: tuple[VariogramFitAttempt, ...] = ()
    warnings: tuple[str, ...] = ()
    fallback_used: bool = False


# ----------------------------------------------------------------------
# ExperimentalVariogram
# ----------------------------------------------------------------------


class ExperimentalVariogram:
    """
    Binned experimental variogram.

    Attributes:
        lags:   ``(n_lags,)`` mean lag distances per bin (empty bins
            carry the nominal bin centre).
        gammas: ``(n_lags,)`` semivariance values.
        pairs:  ``(n_lags,)`` number of point pairs per bin.
        direction: optional ``(azimuth, dip)`` tuple if directional.
    """

    _KIND_MAP = {
        "spherical": VariogramKernel.SPHERICAL,
        "exponential": VariogramKernel.EXPONENTIAL,
        "gaussian": VariogramKernel.GAUSSIAN,
    }

    def __init__(
        self,
        lags: object,
        gammas: object,
        pairs: object,
        direction: tuple[float, float] | None = None,
    ) -> None:
        lags_arr = as_float_array(lags)
        gammas_arr = as_float_array(gammas)
        pairs_arr = as_int_array(pairs)
        if not (lags_arr.shape == gammas_arr.shape == pairs_arr.shape):
            raise GeoBrainError(
                "lags / gammas / pairs must share shape",
                object_name="ExperimentalVariogram", field="shapes",
                expected="same shape",
                actual=(lags_arr.shape, gammas_arr.shape, pairs_arr.shape),
            )
        self.lags = lags_arr
        self.gammas = gammas_arr
        self.pairs = pairs_arr
        self.direction = direction

    # ------------------------------------------------------------------

    def fit_model(
        self,
        kind: str = "auto",
        *,
        strict: bool = False,
    ) -> CovarianceModel:
        """Fit a theoretical model and return it. See :meth:`fit_model_with_diagnostics`."""
        return self.fit_model_with_diagnostics(kind=kind, strict=strict).model

    def fit_model_with_diagnostics(
        self,
        kind: str = "auto",
        *,
        strict: bool = False,
    ) -> VariogramFitResult:
        """
        Fit and return a :class:`VariogramFitResult` with diagnostics.

        Args:
            kind: ``"auto"`` (try spherical / exponential / gaussian, pick
                the lowest cost) or one of those three names.
            strict: if true, raise on any optimizer warning / fallback /
                non-success even when a usable model was produced.
        """
        nonempty = as_bool_array(self.pairs > 0)
        lag_vals = as_float_array(self.lags[nonempty])
        gamma_vals = as_float_array(self.gammas[nonempty])
        pair_counts = as_int_array(self.pairs[nonempty])

        if lag_vals.size == 0:
            msg = "no experimental lag bin has pairs; returning a nugget-only model"
            diagnostics = VariogramFitDiagnostics(
                requested_kind=kind,
                selected_kind="nugget",
                attempts=(VariogramFitAttempt(
                    kind="nugget", optimizer="none", success=True,
                    accepted=True, cost=0.0, message=msg,
                ),),
                fallback_used=False,
                warnings=(msg,),
            )
            self._finalize_diagnostics(diagnostics, strict)
            return VariogramFitResult(CovarianceModel(nugget=0.0), diagnostics)

        weights = as_float_array(pair_counts / np.maximum(gamma_vals**2, 1e-20))
        sill_est = float(np.max(gamma_vals))
        range_est = float(np.max(lag_vals))

        if kind == "auto":
            return self._fit_auto(
                kind, lag_vals, gamma_vals, weights, sill_est, range_est, strict
            )
        if kind in self._KIND_MAP:
            return self._fit_named(
                kind, lag_vals, gamma_vals, weights, sill_est, range_est, strict
            )
        raise GeoBrainError(
            f"unknown kind {kind!r}",
            object_name="ExperimentalVariogram.fit_model", field="kind",
            expected="'auto', 'spherical', 'exponential', or 'gaussian'",
            actual=kind,
        )

    # ------------------------------------------------------------------

    def _fit_auto(
        self,
        kind: str,
        lags: FloatArray,
        gammas: FloatArray,
        weights: FloatArray,
        sill_est: float,
        range_est: float,
        strict: bool,
    ) -> VariogramFitResult:
        best: _SingleFitResult | None = None
        best_kind: str | None = None
        best_cost = np.inf
        attempts: list[VariogramFitAttempt] = []
        for kname, ktype in self._KIND_MAP.items():
            single = self._fit_single(
                kname, ktype, lags, gammas, weights, sill_est, range_est
            )
            attempts.extend(single.attempts)
            if single.model is None:
                continue
            cost = self._wls_cost(single.model, lags, gammas, weights)
            if cost < best_cost:
                best_cost = cost
                best = single
                best_kind = kname

        if best is None or best.model is None:
            diagnostics = VariogramFitDiagnostics(
                requested_kind=kind,
                selected_kind=None,
                attempts=tuple(attempts),
                fallback_used=False,
                warnings=("auto variogram fitting did not produce a model",),
            )
            raise GeoBrainError(
                self._format_fit_failure(diagnostics),
                object_name="ExperimentalVariogram.fit_model",
                field="kind", expected="convergent fit", actual=kind,
            )

        diagnostics = VariogramFitDiagnostics(
            requested_kind=kind,
            selected_kind=best_kind,
            attempts=tuple(attempts),
            fallback_used=best.fallback_used,
            warnings=best.warnings,
        )
        self._finalize_diagnostics(diagnostics, strict)
        return VariogramFitResult(best.model, diagnostics)

    def _fit_named(
        self,
        kind: str,
        lags: FloatArray,
        gammas: FloatArray,
        weights: FloatArray,
        sill_est: float,
        range_est: float,
        strict: bool,
    ) -> VariogramFitResult:
        single = self._fit_single(
            kind, self._KIND_MAP[kind], lags, gammas, weights, sill_est, range_est
        )
        diagnostics = VariogramFitDiagnostics(
            requested_kind=kind,
            selected_kind=kind if single.model is not None else None,
            attempts=single.attempts,
            fallback_used=single.fallback_used,
            warnings=single.warnings,
        )
        if single.model is None:
            raise GeoBrainError(
                self._format_fit_failure(diagnostics),
                object_name="ExperimentalVariogram.fit_model",
                field="kind", expected="convergent fit", actual=kind,
            )
        self._finalize_diagnostics(diagnostics, strict)
        return VariogramFitResult(single.model, diagnostics)

    # ------------------------------------------------------------------

    def _fit_single(
        self,
        kind_name: str,
        kernel_kind: int,
        lags: FloatArray,
        gammas: FloatArray,
        weights: FloatArray,
        sill_est: float,
        range_est: float,
    ) -> _SingleFitResult:
        minimize = cast(
            Callable[..., Any],
            getattr(import_module("scipy.optimize"), "minimize"),
        )

        def objective(params: object) -> float:
            arr = as_float_array(params)
            nugget = max(0.0, float(arr[0]))
            contribution = max(1e-12, float(arr[1]))
            range_val = max(1e-10, float(arr[2]))
            model = CovarianceModel(
                nugget,
                [VariogramKernel(kernel_kind, contribution, (range_val, range_val, range_val))],
            )
            return float(np.sum(weights * (model.variogram(lags) - gammas) ** 2))

        nugget0 = max(0.0, float(gammas[0]) * 0.5) if gammas.size > 0 else 0.0
        contribution0 = max(sill_est - nugget0, 0.01)
        range0 = range_est * 0.4
        min_lag = float(lags[0]) if lags.size > 0 else 1.0
        bounds = [
            (0.0, sill_est * 2.0),
            (0.0, sill_est * 3.0),
            (min_lag * 0.5, range_est * 2.0),
        ]

        def make_model(params: object) -> CovarianceModel:
            arr = as_float_array(params)
            nugget = max(0.0, float(arr[0]))
            contribution = max(1e-12, float(arr[1]))
            range_val = max(1e-10, float(arr[2]))
            return CovarianceModel(
                nugget,
                [VariogramKernel(kernel_kind, contribution, (range_val, range_val, range_val))],
            )

        def run(
            optimizer: str,
            *,
            optimizer_bounds: list[tuple[float, float]] | None,
        ) -> tuple[Any | None, VariogramFitAttempt]:
            try:
                result = minimize(
                    objective,
                    [nugget0, contribution0, range0],
                    method=optimizer,
                    bounds=optimizer_bounds,
                    options={"maxiter": 2000},
                )
            except Exception as exc:                       # pragma: no cover
                return None, VariogramFitAttempt(
                    kind=kind_name, optimizer=optimizer,
                    success=False, accepted=False,
                    cost=None, message=str(exc),
                )
            cost = self._optimizer_cost(result)
            success = bool(getattr(result, "success", False))
            accepted = success or (cost is not None and cost < 1e15)
            message = str(getattr(result, "message", ""))
            return (
                result if accepted else None,
                VariogramFitAttempt(
                    kind=kind_name, optimizer=optimizer,
                    success=success, accepted=accepted,
                    cost=cost, message=message,
                ),
            )

        attempts: list[VariogramFitAttempt] = []
        result, att = run("L-BFGS-B", optimizer_bounds=bounds)
        attempts.append(att)
        if result is not None:
            wmsgs: tuple[str, ...] = ()
            if att.accepted and not att.success:
                wmsgs = (
                    f"L-BFGS-B variogram fit did not report success for "
                    f"{kind_name}, but produced a finite cost.",
                )
            return _SingleFitResult(
                model=make_model(getattr(result, "x")),
                attempts=tuple(attempts),
                warnings=wmsgs,
            )

        fallback_warnings: list[str] = [
            f"L-BFGS-B variogram fit failed for {kind_name}; falling back "
            f"to Nelder-Mead. Original message: {att.message}"
        ]
        result, att = run("Nelder-Mead", optimizer_bounds=None)
        attempts.append(att)
        if result is None:
            fallback_warnings.append(
                f"Nelder-Mead variogram fit failed for {kind_name}; "
                "no model was accepted."
            )
            return _SingleFitResult(
                model=None,
                attempts=tuple(attempts),
                warnings=tuple(fallback_warnings),
                fallback_used=True,
            )

        params = as_float_array(getattr(result, "x"))
        params[0] = float(np.clip(params[0], 0.0, sill_est * 2.0))
        params[1] = float(np.clip(params[1], 1e-12, sill_est * 3.0))
        params[2] = float(np.clip(params[2], min_lag * 0.5, range_est * 2.0))
        if att.accepted and not att.success:
            fallback_warnings.append(
                f"Nelder-Mead variogram fit did not report success for "
                f"{kind_name}, but produced a finite cost."
            )
        return _SingleFitResult(
            model=make_model(params),
            attempts=tuple(attempts),
            warnings=tuple(fallback_warnings),
            fallback_used=True,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _optimizer_cost(result: Any) -> float | None:
        try:
            cost = float(getattr(result, "fun"))
        except (TypeError, ValueError, AttributeError):
            return None
        return cost if np.isfinite(cost) else None

    @staticmethod
    def _wls_cost(
        model: CovarianceModel,
        lags: FloatArray,
        gammas: FloatArray,
        weights: FloatArray,
    ) -> float:
        predicted = model.variogram(lags)
        return float(np.sum(weights * (predicted - gammas) ** 2))

    @staticmethod
    def _finalize_diagnostics(
        diagnostics: VariogramFitDiagnostics, strict: bool
    ) -> None:
        if strict and (
            not diagnostics.success
            or diagnostics.fallback_used
            or diagnostics.warnings
        ):
            raise GeoBrainError(
                ExperimentalVariogram._format_fit_failure(diagnostics),
                object_name="ExperimentalVariogram.fit_model",
                field="strict", expected="clean fit", actual="warnings present",
            )
        for message in diagnostics.warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _format_fit_failure(diagnostics: VariogramFitDiagnostics) -> str:
        parts = [
            "strict variogram fit failed"
            if diagnostics.success
            else "variogram fit failed",
            f"requested_kind={diagnostics.requested_kind!r}",
            f"selected_kind={diagnostics.selected_kind!r}",
            f"fallback_used={diagnostics.fallback_used}",
        ]
        if diagnostics.warnings:
            parts.append(f"warnings={list(diagnostics.warnings)!r}")
        return "; ".join(parts)

    def __repr__(self) -> str:
        dir_str = f", direction={self.direction}" if self.direction else ""
        return f"ExperimentalVariogram({self.lags.size} lags{dir_str})"
