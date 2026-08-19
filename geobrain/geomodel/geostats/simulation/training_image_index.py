"""Immutable exact catalogue for multi-point training-image queries.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from ...cache_keys import TrainingImageCacheKey
from ...frames import GeoFrame, PropertyMetadata
from ...errors import GeomodelContractError, GeomodelResourceError

from .execution import SimulationExecutionConfig
from .results import SimulationEnsemble, SimulationRealization

__all__ = ["TrainingImageIndex", "TrainingImageSelection", "TrainingImageSpec"]


def _readonly(values: object, dtype: np.dtype[object] | type[object]) -> np.ndarray:
    owned = np.array(values, dtype=dtype, copy=True, order="C")
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


@dataclass(frozen=True, slots=True)
class TrainingImageSpec:
    """Owned values, missingness, axes, property identity, and fingerprint.

    Attributes:
        values: the training-image array.
        property: its :class:`PropertyMetadata`.
        axis_names: axis labels.
        missing_mask: mask of unknown cells.
        fingerprint: content hash.
    """

    values: np.ndarray
    property: PropertyMetadata
    axis_names: tuple[str, ...]
    missing_mask: np.ndarray | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.property, PropertyMetadata):
            raise GeomodelContractError(
                "training image requires PropertyMetadata",
                object_name=type(self).__name__, field="property",
                expected="PropertyMetadata", actual=type(self.property).__name__,
            )
        try:
            values = np.array(self.values, dtype=np.float64, copy=True, order="C")
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "training-image values must be numeric",
                object_name=type(self).__name__, field="values",
                expected="finite rank-2 or rank-3 numeric array", actual=type(self.values).__name__,
            ) from exc
        axes = tuple(self.axis_names)
        canonical = ("x", "y") if values.ndim == 2 else ("x", "y", "z")
        if values.ndim not in (2, 3) or set(axes) != set(canonical) or len(axes) != values.ndim:
            raise GeomodelContractError(
                "training-image axes must name each physical axis exactly once",
                object_name=type(self).__name__, field="axis_names",
                expected=canonical, actual=axes,
            )
        mask = (
            np.zeros(values.shape, dtype=np.bool_)
            if self.missing_mask is None
            else np.array(self.missing_mask, dtype=np.bool_, copy=True, order="C")
        )
        if mask.shape != values.shape:
            raise GeomodelContractError(
                "training-image missing mask must match values",
                object_name=type(self).__name__, field="missing_mask",
                expected=tuple(values.shape), actual=tuple(mask.shape),
            )
        active = values[~mask]
        self.property.validate_values(active, object_name=type(self).__name__)
        key = TrainingImageCacheKey.from_array(
            values,
            property=self.property,
            axis_names=axes,
            missing_mask=mask,
        )
        if self.fingerprint is not None and self.fingerprint != key.digest:
            raise GeomodelContractError(
                "training-image fingerprint does not match its content",
                object_name=type(self).__name__, field="fingerprint",
                expected=key.digest, actual=self.fingerprint,
            )
        object.__setattr__(self, "values", _readonly(values, np.float64))
        object.__setattr__(self, "missing_mask", _readonly(mask, np.bool_))
        object.__setattr__(self, "axis_names", axes)
        object.__setattr__(self, "fingerprint", key.digest)


@dataclass(frozen=True, slots=True)
class TrainingImageSelection:
    """Stable candidate ordering by exact score then source rank.

    Attributes:
        candidate_ids: selected pattern ids.
        scores: match scores.
        source_ranks: provenance ranks of the patterns.
    """

    candidate_ids: np.ndarray
    scores: np.ndarray
    source_ranks: np.ndarray

    def __post_init__(self) -> None:
        candidate_ids = _readonly(self.candidate_ids, np.int64)
        scores = _readonly(self.scores, np.float64)
        ranks = _readonly(self.source_ranks, np.int64)
        if candidate_ids.ndim != 1 or scores.shape != candidate_ids.shape or ranks.shape != candidate_ids.shape:
            raise GeomodelContractError(
                "training-image selection arrays must be aligned vectors",
                object_name=type(self).__name__, field="candidate_ids/scores/source_ranks",
                expected="equal rank-1 shapes", actual=(candidate_ids.shape, scores.shape, ranks.shape),
            )
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "source_ranks", ranks)


@dataclass(frozen=True, slots=True)
class TrainingImageIndex:
    """Exact deterministic catalogue shared by MPS algorithm views.

    Attributes:
        spec: the indexed :class:`TrainingImageSpec`.
        template: search-template offsets.
        candidate_ids / source_ranks: indexed pattern table.
        features: pattern feature matrix.
        cache_key: content key for index reuse.
    """

    spec: TrainingImageSpec
    template: np.ndarray
    candidate_ids: np.ndarray
    source_ranks: np.ndarray
    features: np.ndarray
    cache_key: TrainingImageCacheKey

    @classmethod
    def build(
        cls,
        spec: TrainingImageSpec,
        template: object,
        budget_bytes: int | None = None,
    ) -> "TrainingImageIndex":
        if not isinstance(spec, TrainingImageSpec):
            raise GeomodelContractError(
                "training-image index requires TrainingImageSpec",
                object_name=cls.__name__, field="spec",
                expected="TrainingImageSpec", actual=type(spec).__name__,
            )
        offsets = np.array(template, dtype=np.int64, copy=True, order="C")
        if offsets.ndim != 2 or offsets.shape[1] != spec.values.ndim:
            raise GeomodelContractError(
                "training-image template must contain one integer offset per axis",
                object_name=cls.__name__, field="template",
                expected=f"(n, {spec.values.ndim}) integer array", actual=tuple(offsets.shape),
            )
        cell_count = int(spec.values.size)
        required = cell_count * (offsets.shape[0] * 8 + 24)
        if budget_bytes is not None and required > budget_bytes:
            raise GeomodelResourceError(
                "training-image catalogue exceeds the configured budget",
                object_name=cls.__name__, field="budget_bytes",
                expected=f">= {required}", actual=budget_bytes,
            )
        axis_positions = tuple(spec.axis_names.index(axis) for axis in (("x", "y") if spec.values.ndim == 2 else ("x", "y", "z")))
        array_offsets = offsets[:, np.argsort(np.asarray(axis_positions))]
        candidate_ids: list[int] = []
        ranks: list[int] = []
        feature_rows: list[list[float]] = []
        mask = cast(np.ndarray, spec.missing_mask)
        for rank, centre in enumerate(np.ndindex(spec.values.shape)):
            row: list[float] = []
            valid = not bool(mask[centre])
            for offset in array_offsets:
                location = tuple(int(centre[axis] + offset[axis]) for axis in range(spec.values.ndim))
                if any(location[axis] < 0 or location[axis] >= spec.values.shape[axis] for axis in range(spec.values.ndim)):
                    valid = False
                    break
                if bool(mask[location]):
                    valid = False
                    break
                row.append(float(spec.values[location]))
            if valid:
                candidate_ids.append(rank)
                ranks.append(rank)
                feature_rows.append(row)
        features = np.asarray(feature_rows, dtype=np.float64).reshape(-1, offsets.shape[0])
        key = TrainingImageCacheKey.from_array(
            spec.values,
            property=spec.property,
            axis_names=spec.axis_names,
            missing_mask=spec.missing_mask,
        )
        return cls(
            spec,
            _readonly(offsets, np.int64),
            _readonly(candidate_ids, np.int64),
            _readonly(ranks, np.int64),
            _readonly(features, np.float64),
            key,
        )

    def query(
        self,
        event: object,
        *,
        mode: Literal["exact", "exhaustive"] = "exact",
        max_candidates: int | None = None,
    ) -> TrainingImageSelection:
        if mode not in ("exact", "exhaustive"):
            raise GeomodelContractError(
                "training-image query mode is unsupported",
                object_name=type(self).__name__, field="mode",
                expected="exact or exhaustive", actual=mode,
            )
        values = np.asarray(event, dtype=np.float64)
        if values.shape != (self.template.shape[0],) or not np.isfinite(values).all():
            raise GeomodelContractError(
                "training-image event must match the template",
                object_name=type(self).__name__, field="event",
                expected=(self.template.shape[0],), actual=tuple(values.shape),
            )
        if self.spec.property.kind == "categorical":
            scores = np.mean(self.features != values[None, :], axis=1, dtype=np.float64)
        else:
            scores = np.sqrt(np.mean((self.features - values[None, :]) ** 2, axis=1))
        order = np.lexsort((self.source_ranks, scores))
        if max_candidates is not None:
            if isinstance(max_candidates, bool) or max_candidates < 1:
                raise GeomodelContractError(
                    "maximum candidate count must be positive",
                    object_name=type(self).__name__, field="max_candidates",
                    expected=">= 1", actual=max_candidates,
                )
            order = order[: int(max_candidates)]
        return TrainingImageSelection(
            self.candidate_ids[order],
            scores[order],
            self.source_ranks[order],
        )


def assemble_mps_ensemble(
    algorithm: str,
    property: PropertyMetadata,
    execution: SimulationExecutionConfig,
    seeds: tuple[int, ...] | list[int],
    frames: tuple[GeoFrame, ...],
    fingerprint: str,
) -> SimulationEnsemble:
    """Wrap legacy numerical kernels in the common immutable result contract."""
    realizations = tuple(
        SimulationRealization(
            index,
            int(seed),
            frame,
            {
                "training_image_fingerprint": fingerprint,
                "accounting": {
                    "distance_checks": 0,
                    "index_queries": 0,
                    "index_rebuilds": 0,
                    "append_writes": int(frame.geometry.npoints),
                    "pool_rebuilds": 0,
                    "candidate_comparisons": 0,
                },
            },
        )
        for index, (seed, frame) in enumerate(zip(seeds, frames, strict=True))
    )
    return SimulationEnsemble(
        property,
        realizations,
        execution,
        {
            "algorithm": algorithm,
            "property": property.to_dict(),
            "training_image_fingerprint": fingerprint,
            "index_accounting_status": "kernel_specific_counters_not_instrumented",
        },
    )
