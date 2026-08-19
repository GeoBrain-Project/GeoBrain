"""Deterministic isolated execution for geostatistical realizations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast, overload

import numpy as np

from ...errors import GeomodelContractError, GeomodelNumericsError
from .execution import SimulationExecutionConfig

__all__ = ["RealizationRun", "RealizationWorkerResult", "resolve_n_jobs", "run_realizations"]

_HAS_THREADPOOLCTL = importlib.util.find_spec("threadpoolctl") is not None
_WORKER_BLAS_LIMITER: Any | None = None


@dataclass(frozen=True, slots=True)
class RealizationWorkerResult:
    """Owned result emitted by one realization worker."""

    index: int
    seed: int
    result: object
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RealizationRun:
    """Ordered worker output plus the backend actually used."""

    results: tuple[RealizationWorkerResult, ...]
    diagnostics: Mapping[str, object]


def _init_single_thread_blas() -> None:
    """Process-pool initializer: keep BLAS at one thread for worker lifetime."""
    global _WORKER_BLAS_LIMITER
    from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]

    _WORKER_BLAS_LIMITER = threadpool_limits(limits=1)


def _blas_limit() -> Any:
    if not _HAS_THREADPOOLCTL:
        return nullcontext()
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1)


def resolve_n_jobs(n_jobs: int | str, n_tasks: int) -> int:
    """Resolve the legacy worker selector to a positive bounded count."""
    cpu_count = os.cpu_count() or 1
    if isinstance(n_jobs, str):
        if n_jobs != "auto":
            raise GeomodelContractError(
                "n_jobs string must be 'auto'",
                object_name="run_realizations",
                field="n_jobs",
                expected="'auto' or int",
                actual=n_jobs,
            )
        return max(1, min(n_tasks, cpu_count))
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise GeomodelContractError(
            "n_jobs must be an integer or 'auto'",
            object_name="run_realizations",
            field="n_jobs",
            expected=">=1, -1, or 'auto'",
            actual=n_jobs,
        )
    requested = int(n_jobs)
    if requested == 0:
        raise GeomodelContractError(
            "n_jobs must be >= 1, -1, or 'auto'",
            object_name="run_realizations",
            field="n_jobs",
            expected=">=1, -1, or 'auto'",
            actual=n_jobs,
        )
    if requested < 0:
        requested = cpu_count
    return max(1, min(n_tasks, requested))


def _owned_result(value: object) -> object:
    if isinstance(value, np.ndarray):
        owned = np.array(value, copy=True, order="C")
        owned.setflags(write=False)
        return owned
    return value


def _owned_diagnostics(values: Mapping[str, object]) -> Mapping[str, object]:
    try:
        payload = json.dumps(dict(values), allow_nan=False, sort_keys=True)
        decoded = cast(object, json.loads(payload))
    except (TypeError, ValueError) as exc:
        raise GeomodelNumericsError(
            "realization worker returned non-JSON diagnostics",
            object_name="run_realizations",
            field="diagnostics",
            expected="strict JSON object",
            actual=type(values).__name__,
        ) from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise GeomodelNumericsError(
            "realization worker diagnostics must be a JSON object",
            object_name="run_realizations",
            field="diagnostics",
            expected="mapping with string keys",
            actual=type(values).__name__,
        )
    return cast(Mapping[str, object], decoded)


def _worker_identity_error(
    value: object,
    *,
    field: Literal["index", "seed"],
) -> GeomodelNumericsError:
    value_type = type(value)
    actual_value = (
        value
        if value is None or value_type in (bool, int, float, str)
        else "<non-scalar>"
    )
    try:
        type_name = value_type.__name__
    except Exception:
        type_name = "<unknown>"
    return GeomodelNumericsError(
        f"realization worker returned invalid {field}",
        object_name="run_realizations",
        field=f"worker return.{field}",
        expected="exact non-boolean integer",
        actual={"type": type_name, "value": actual_value},
    )


def _probe_worker_comparison(
    value: object,
    expected: int,
    *,
    field: Literal["index", "seed"],
) -> None:
    """Wrap hostile comparison behavior before object-result normalization."""
    try:
        bool(cast(Any, value) != expected)
    except (GeomodelContractError, GeomodelNumericsError):
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise _worker_identity_error(value, field=field) from exc


def _worker_integer(value: object, *, field: Literal["index", "seed"]) -> int:
    """Coerce one worker identity field without leaking boundary failures."""

    try:
        resolved = int(cast(Any, value))
    except (GeomodelContractError, GeomodelNumericsError):
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise _worker_identity_error(value, field=field) from exc
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        cause = TypeError(f"worker {field} must be an exact non-boolean integer")
        raise _worker_identity_error(value, field=field) from cause
    return resolved


def _coerce_worker_result(
    value: object,
    *,
    expected_index: int,
    expected_seed: int,
) -> RealizationWorkerResult:
    if isinstance(value, RealizationWorkerResult):
        _probe_worker_comparison(value.index, expected_index, field="index")
        _probe_worker_comparison(value.seed, expected_seed, field="seed")
        result = RealizationWorkerResult(
            _worker_integer(value.index, field="index"),
            _worker_integer(value.seed, field="seed"),
            value.result,
            value.diagnostics,
        )
    elif isinstance(value, tuple) and len(value) == 4:
        index, seed, owned, diagnostics = value
        if not isinstance(diagnostics, Mapping):
            raise GeomodelNumericsError(
                "realization worker returned invalid diagnostics",
                object_name="run_realizations",
                field="worker return",
                expected="(index, seed, result, mapping)",
                actual=type(diagnostics).__name__,
            )
        result = RealizationWorkerResult(
            _worker_integer(index, field="index"),
            _worker_integer(seed, field="seed"),
            owned,
            diagnostics,
        )
    else:
        raise GeomodelNumericsError(
            "realization worker returned an invalid result",
            object_name="run_realizations",
            field="worker return",
            expected="(index, seed, result, diagnostics)",
            actual=type(value).__name__,
        )
    if result.index != expected_index or result.seed != expected_seed:
        raise GeomodelNumericsError(
            "realization worker returned a mismatched index or seed",
            object_name="run_realizations",
            field="worker return",
            expected={"index": expected_index, "seed": expected_seed},
            actual={"index": result.index, "seed": result.seed},
        )
    return RealizationWorkerResult(
        result.index,
        result.seed,
        _owned_result(result.result),
        _owned_diagnostics(result.diagnostics),
    )


def _call_worker(
    worker: Callable[[int, int], object],
    item: tuple[int, int],
) -> object:
    return worker(*item)


def _execute(
    worker: Callable[[int, int], object],
    assignments: Sequence[tuple[int, int]],
    *,
    backend: Literal["serial", "process", "thread"],
    workers: int,
) -> list[object]:
    call = partial(_call_worker, worker)
    try:
        if backend == "serial":
            with _blas_limit():
                return [call(item) for item in assignments]
        if backend == "process":
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_init_single_thread_blas,
            ) as pool:
                return list(pool.map(call, assignments))
        with _blas_limit():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(call, assignments))
    except GeomodelNumericsError:
        raise
    except Exception as exc:
        raise GeomodelNumericsError(
            f"simulation realization worker failed: {exc}",
            object_name="run_realizations",
            field="worker",
            expected="successful isolated realization",
            actual=type(exc).__name__,
        ) from exc


def _actual_backend(execution: SimulationExecutionConfig, task_count: int) -> tuple[Literal["serial", "process", "thread"], int]:
    if task_count < 2 or execution.worker_backend == "serial":
        return "serial", 1
    workers = min(execution.workers, task_count)
    process_available = _HAS_THREADPOOLCTL and "fork" in multiprocessing.get_all_start_methods()
    if execution.worker_backend == "process" and process_available:
        return "process", workers
    if execution.worker_backend == "process":
        return "thread", workers
    return "thread", workers


def _run_configured(
    worker: Callable[[int, int], object],
    seeds: Sequence[int],
    execution: SimulationExecutionConfig,
) -> RealizationRun:
    seed_tuple = tuple(int(seed) for seed in seeds)
    if len(seed_tuple) != execution.n_realizations:
        raise GeomodelContractError(
            "execution realization count must match the derived seed count",
            object_name="run_realizations",
            field="seeds",
            expected=execution.n_realizations,
            actual=len(seed_tuple),
        )
    backend, workers = _actual_backend(execution, len(seed_tuple))
    assignments = tuple(enumerate(seed_tuple))
    raw_results = _execute(worker, assignments, backend=backend, workers=workers)
    results = tuple(
        _coerce_worker_result(value, expected_index=index, expected_seed=seed)
        for value, (index, seed) in zip(raw_results, assignments, strict=True)
    )
    return RealizationRun(
        tuple(sorted(results, key=lambda item: item.index)),
        {
            "worker_backend": backend,
            "workers": workers,
            "requested_worker_backend": execution.worker_backend,
            "requested_workers": execution.workers,
        },
    )


def _legacy_worker(worker: Callable[[int], np.ndarray], index: int, seed: int) -> tuple[int, int, np.ndarray, dict[str, object]]:
    return index, seed, worker(seed), {}


@overload
def run_realizations(
    worker: Callable[[int, int], object],
    seeds: Sequence[int],
    execution: SimulationExecutionConfig,
) -> RealizationRun: ...


@overload
def run_realizations(
    worker: Callable[[int], np.ndarray],
    seeds: Sequence[int],
    execution: int | str = 1,
) -> list[np.ndarray]: ...


def run_realizations(
    worker: Callable[..., object],
    seeds: Sequence[int],
    execution: SimulationExecutionConfig | int | str = 1,
) -> RealizationRun | list[np.ndarray]:
    """Execute isolated indexed workers, retaining the legacy seed-only adapter."""
    if isinstance(execution, SimulationExecutionConfig):
        configured_worker = cast(Callable[[int, int], object], worker)
        return _run_configured(configured_worker, seeds, execution)
    legacy_workers = resolve_n_jobs(execution, len(seeds))
    if len(seeds) == 0:
        return []
    backend: Literal["serial", "process", "thread"] = "serial"
    if legacy_workers > 1:
        backend = "process" if _HAS_THREADPOOLCTL and "fork" in multiprocessing.get_all_start_methods() else "thread"
    legacy_config = SimulationExecutionConfig(
        n_realizations=max(1, len(seeds)),
        workers=legacy_workers,
        worker_backend=backend,
    )
    adapted = partial(_legacy_worker, cast(Callable[[int], np.ndarray], worker))
    outcome = _run_configured(adapted, seeds, legacy_config)
    return [cast(np.ndarray, item.result) for item in outcome.results]
