"""Deterministic allocation-free EM resource estimation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import sys
from typing import Literal, NoReturn, cast

from .errors import EMContractError, EMResourceError


_MESH_FORMULATIONS = frozenset(
    {
        "edge_fem",
        "layered_1d",
        "structured_fem",
        "structured_mimetic",
        "structured_yee",
        "triangular_fem",
    }
)
_ITEMSIZE = {"complex128": 16, "complex64": 8, "float32": 4, "float64": 8}
_INDEX_BYTES = 8
_DEVICES = frozenset({"cpu", "cuda", "mps"})
_LAYOUTS = frozenset({"cartesian", "paired"})
_RECORDINGS = frozenset({"checkpoint_recompute", "gate_states", "output_only"})
_ASSUMPTIONS = (
    "assembly bounds a dense CSR matrix with 8-byte indices plus two value/index pairs per declared mesh entity",
    "sparse LU bounds dense L/U triangles with 8-byte indices, row pointers, and row/column permutations",
    "live state stores three forward vectors per source and six when gradients require an adjoint",
    "peak is the sum of all reported byte partitions",
)
_PARTITIONS = (
    "assembly_bytes",
    "factor_bytes",
    "live_state_bytes",
    "recording_bytes",
    "output_bytes",
)


def _contract_error(
    message: str,
    *,
    object_name: str,
    field: str,
    expected: object,
    actual: object,
) -> NoReturn:
    """Raise a stable request/estimate contract error."""
    actual_value = (
        actual if type(actual) in (str, int, bool) or actual is None else type(actual).__qualname__
    )
    raise EMContractError(
        message,
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual_value,
        hint="provide a canonical non-negative EM resource declaration",
        details={
            "field": field,
            "received_type": type(actual).__qualname__,
            "remediation": "provide a canonical non-negative EM resource declaration",
        },
    )


def _checked_add(*values: int, term: str) -> int:
    """Add addressable byte/count values or fail before allocation."""
    total = 0
    for value in values:
        if value > sys.maxsize - total:
            _overflow(term, values)
        total += value
    return total


def _checked_mul(*values: int, term: str) -> int:
    """Multiply addressable byte/count values or fail before allocation."""
    total = 1
    for value in values:
        if value != 0 and total > sys.maxsize // value:
            _overflow(term, values)
        total *= value
    return total


def _overflow(term: str, operands: tuple[int, ...]) -> NoReturn:
    """Map arithmetic overflow to the public resource error contract."""
    raise EMResourceError(
        "EM resource estimate exceeds addressable integer range",
        object_name="estimate_em_resources",
        field=term,
        expected=f"value <= {sys.maxsize}",
        actual="overflow",
        hint="reduce mesh, source, receiver, or sample counts",
        details={
            "limit_bytes": sys.maxsize,
            "operands": operands,
            "term": term,
            "remediation": "reduce mesh, source, receiver, or sample counts",
        },
    )


@dataclass(frozen=True, slots=True)
class EMResourceRequest:
    """Pure scalar inputs for conservative EM resource preflight."""

    mesh_formulation: str
    n_cells: int
    n_edges: int
    n_faces: int
    n_sources: int
    n_receivers: int
    n_samples: int
    dtype: str
    device: str
    receiver_layout: Literal["cartesian", "paired"]
    recording: Literal["output_only", "gate_states", "checkpoint_recompute"]
    requires_gradient: bool

    def __post_init__(self) -> None:
        """Reject type tricks, unsupported enums, and ambiguous combinations."""
        name = type(self).__name__
        for field in (
            "n_cells",
            "n_edges",
            "n_faces",
            "n_sources",
            "n_receivers",
            "n_samples",
        ):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value <= sys.maxsize:
                _contract_error(
                    "invalid EM resource request dimension",
                    object_name=name,
                    field=field,
                    expected=f"integer in [0, {sys.maxsize}]",
                    actual=value,
                )
        for field, allowed in (
            ("mesh_formulation", _MESH_FORMULATIONS),
            ("dtype", frozenset(_ITEMSIZE)),
            ("device", _DEVICES),
            ("receiver_layout", _LAYOUTS),
            ("recording", _RECORDINGS),
        ):
            value = getattr(self, field)
            if type(value) is not str or value not in allowed:
                _contract_error(
                    "invalid EM resource request selection",
                    object_name=name,
                    field=field,
                    expected=tuple(sorted(allowed)),
                    actual=value,
                )
        if type(self.requires_gradient) is not bool:
            _contract_error(
                "invalid EM resource gradient request",
                object_name=name,
                field="requires_gradient",
                expected="bool",
                actual=self.requires_gradient,
            )
        if self.receiver_layout == "paired" and self.n_sources != self.n_receivers:
            _contract_error(
                "paired EM resources require equal source and receiver counts",
                object_name=name,
                field="n_receivers",
                expected=self.n_sources,
                actual=self.n_receivers,
            )
        if self.recording == "checkpoint_recompute" and not self.requires_gradient:
            _contract_error(
                "checkpoint recomputation requires gradients",
                object_name=name,
                field="recording",
                expected="requires_gradient=true",
                actual=self.recording,
            )


@dataclass(frozen=True, slots=True)
class EMResourceEstimate:
    """Conservative byte estimate partitioned by allocation purpose."""

    assembly_bytes: int
    factor_bytes: int
    live_state_bytes: int
    recording_bytes: int
    output_bytes: int
    peak_bytes: int
    factorization_count: int
    linear_solve_count: int
    dominant_term: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require exact non-negative accounting and deeply owned assumptions."""
        name = type(self).__name__
        for field in (*_PARTITIONS, "peak_bytes", "factorization_count", "linear_solve_count"):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value <= sys.maxsize:
                _contract_error(
                    "invalid EM resource estimate",
                    object_name=name,
                    field=field,
                    expected=f"integer in [0, {sys.maxsize}]",
                    actual=value,
                )
        expected_peak = sum(cast(int, getattr(self, field)) for field in _PARTITIONS)
        if self.peak_bytes != expected_peak or expected_peak > sys.maxsize:
            _contract_error(
                "invalid EM resource partition sum",
                object_name=name,
                field="peak_bytes",
                expected=expected_peak,
                actual=self.peak_bytes,
            )
        expected_dominant = "none"
        if self.peak_bytes:
            expected_dominant = max(
                _PARTITIONS,
                key=lambda field: cast(int, getattr(self, field)),
            )
        if type(self.dominant_term) is not str or self.dominant_term != expected_dominant:
            _contract_error(
                "invalid EM dominant resource term",
                object_name=name,
                field="dominant_term",
                expected=expected_dominant,
                actual=self.dominant_term,
            )
        value = self.assumptions
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            _contract_error(
                "invalid EM resource assumptions",
                object_name=name,
                field="assumptions",
                expected="sequence of non-empty strings",
                actual=value,
            )
        assumptions = tuple(cast(Sequence[object], value))
        if any(type(item) is not str or not item.strip() for item in assumptions):
            _contract_error(
                "invalid EM resource assumptions",
                object_name=name,
                field="assumptions",
                expected="sequence of non-empty strings",
                actual=value,
            )
        object.__setattr__(self, "assumptions", cast(tuple[str, ...], assumptions))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe estimate."""
        return {
            "assembly_bytes": self.assembly_bytes,
            "factor_bytes": self.factor_bytes,
            "live_state_bytes": self.live_state_bytes,
            "recording_bytes": self.recording_bytes,
            "output_bytes": self.output_bytes,
            "peak_bytes": self.peak_bytes,
            "factorization_count": self.factorization_count,
            "linear_solve_count": self.linear_solve_count,
            "dominant_term": self.dominant_term,
            "assumptions": list(self.assumptions),
        }


def _matrix_storage_bounds(
    degrees_of_freedom: int,
    entities: int,
    itemsize: int,
) -> tuple[int, int]:
    """Bound assembly and sparse LU storage by fully dense CSR structures."""
    if degrees_of_freedom == 0:
        return 0, 0
    value_and_index_bytes = _checked_add(
        itemsize,
        _INDEX_BYTES,
        term="value_and_index_bytes",
    )
    dense_entries = _checked_mul(
        degrees_of_freedom,
        degrees_of_freedom,
        term="dense_matrix_entries",
    )
    row_count = _checked_add(degrees_of_freedom, 1, term="matrix_row_count")
    dense_matrix_bytes = _checked_mul(
        dense_entries,
        value_and_index_bytes,
        term="dense_matrix_bytes",
    )
    row_pointer_bytes = _checked_mul(
        row_count,
        _INDEX_BYTES,
        term="row_pointer_bytes",
    )
    entity_bytes = _checked_mul(
        entities,
        2,
        value_and_index_bytes,
        term="mesh_entity_bytes",
    )
    assembly_bytes = _checked_add(
        dense_matrix_bytes,
        row_pointer_bytes,
        entity_bytes,
        term="assembly_bytes",
    )

    # Dense lower and upper triangles together contain at most n * (n + 1)
    # stored values. Each value has one 8-byte column index; two CSR row-pointer
    # arrays and row/column permutation vectors are bounded separately.
    factor_entries = _checked_mul(
        degrees_of_freedom,
        row_count,
        term="factor_entries",
    )
    factor_entry_bytes = _checked_mul(
        factor_entries,
        value_and_index_bytes,
        term="factor_entry_bytes",
    )
    factor_row_pointer_bytes = _checked_mul(
        2,
        row_count,
        _INDEX_BYTES,
        term="factor_row_pointer_bytes",
    )
    permutation_bytes = _checked_mul(
        2,
        degrees_of_freedom,
        _INDEX_BYTES,
        term="factor_permutation_bytes",
    )
    factor_bytes = _checked_add(
        factor_entry_bytes,
        factor_row_pointer_bytes,
        permutation_bytes,
        term="factor_bytes",
    )
    return assembly_bytes, factor_bytes


def estimate_em_resources(request: EMResourceRequest) -> EMResourceEstimate:
    """Return a deterministic conservative estimate without allocating tensors."""
    if type(request) is not EMResourceRequest:
        _contract_error(
            "invalid EM resource request object",
            object_name="estimate_em_resources",
            field="request",
            expected="exact EMResourceRequest",
            actual=request,
        )
    itemsize = _ITEMSIZE[request.dtype]
    entities = _checked_add(request.n_cells, request.n_edges, request.n_faces, term="mesh_entities")
    degrees_of_freedom = max(request.n_cells, request.n_edges, request.n_faces)
    assembly_bytes, factor_bytes = _matrix_storage_bounds(
        degrees_of_freedom,
        entities,
        itemsize,
    )
    live_vector_count = 6 if request.requires_gradient else 3
    live_state_bytes = _checked_mul(
        degrees_of_freedom,
        request.n_sources,
        itemsize,
        live_vector_count,
        term="live_state_bytes",
    )
    if request.recording == "output_only":
        recording_bytes = 0
    elif request.recording == "gate_states":
        recording_bytes = _checked_mul(
            degrees_of_freedom,
            request.n_sources,
            request.n_samples,
            itemsize,
            term="recording_bytes",
        )
    else:
        checkpoint_count = 0 if request.n_samples == 0 else math.isqrt(request.n_samples - 1) + 1
        recording_bytes = _checked_mul(
            degrees_of_freedom,
            request.n_sources,
            checkpoint_count,
            itemsize,
            term="recording_bytes",
        )
    if request.receiver_layout == "paired":
        output_elements = _checked_mul(request.n_sources, request.n_samples, term="output_elements")
    else:
        output_elements = _checked_mul(
            request.n_sources,
            request.n_receivers,
            request.n_samples,
            term="output_elements",
        )
    output_bytes = _checked_mul(output_elements, itemsize, term="output_bytes")
    peak_bytes = _checked_add(
        assembly_bytes,
        factor_bytes,
        live_state_bytes,
        recording_bytes,
        output_bytes,
        term="peak_bytes",
    )
    has_work = degrees_of_freedom > 0 and request.n_sources > 0
    factorization_count = request.n_samples if has_work else 0
    linear_solve_count = (
        _checked_mul(request.n_sources, request.n_samples, term="linear_solve_count")
        if has_work
        else 0
    )
    dominant_term = "none"
    if peak_bytes:
        values = {
            "assembly_bytes": assembly_bytes,
            "factor_bytes": factor_bytes,
            "live_state_bytes": live_state_bytes,
            "recording_bytes": recording_bytes,
            "output_bytes": output_bytes,
        }
        dominant_term = max(_PARTITIONS, key=lambda field: values[field])
    return EMResourceEstimate(
        assembly_bytes=assembly_bytes,
        factor_bytes=factor_bytes,
        live_state_bytes=live_state_bytes,
        recording_bytes=recording_bytes,
        output_bytes=output_bytes,
        peak_bytes=peak_bytes,
        factorization_count=factorization_count,
        linear_solve_count=linear_solve_count,
        dominant_term=dominant_term,
        assumptions=_ASSUMPTIONS,
    )


def enforce_resource_budget(
    estimate: EMResourceEstimate,
    budget_bytes: int | None,
) -> None:
    """Reject an invalid or insufficient budget before any allocation."""
    if type(estimate) is not EMResourceEstimate:
        _contract_error(
            "invalid EM resource estimate object",
            object_name="enforce_resource_budget",
            field="estimate",
            expected="exact EMResourceEstimate",
            actual=estimate,
        )
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes < 0):
        _contract_error(
            "invalid EM resource budget",
            object_name="enforce_resource_budget",
            field="budget_bytes",
            expected="None or non-negative integer",
            actual=budget_bytes,
        )
    if budget_bytes is not None and estimate.peak_bytes > budget_bytes:
        raise EMResourceError(
            "EM resource estimate exceeds the explicit budget",
            object_name="enforce_resource_budget",
            field="budget_bytes",
            expected=f">= {estimate.peak_bytes}",
            actual=budget_bytes,
            hint="increase the budget or reduce mesh, source, receiver, or sample counts",
            details={
                "budget_bytes": budget_bytes,
                "dominant_term": estimate.dominant_term,
                "required_bytes": estimate.peak_bytes,
                "remediation": "increase the budget or reduce problem dimensions",
            },
        )


__all__ = [
    "EMResourceEstimate",
    "EMResourceRequest",
    "enforce_resource_budget",
    "estimate_em_resources",
]
