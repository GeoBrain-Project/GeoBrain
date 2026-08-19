"""Typed records and protocols for the internal Wave propagation engine.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import torch

from ..config import WaveSimulationConfig
from ..errors import WaveContractError


class RimSetter(Protocol):
    """Restore the named saved exterior band on one reconstructed field."""

    def __call__(self, field: torch.Tensor, name: str) -> torch.Tensor: ...


class InverseStep2D(Protocol):
    """Reversible 2-D equation step."""

    def __call__(
        self,
        state: Sequence[torch.Tensor],
        coefficients: Mapping[str, torch.Tensor],
        dt: float,
        dx: float,
        dz: float,
        set_rim: RimSetter,
    ) -> Sequence[torch.Tensor]: ...

class InverseStep3D(Protocol):
    """Reversible 3-D equation step in public reversed-spacing order."""

    def __call__(
        self,
        state: Sequence[torch.Tensor],
        coefficients: Mapping[str, torch.Tensor],
        dt: float,
        dy: float,
        dx: float,
        dz: float,
        set_rim: RimSetter,
    ) -> Sequence[torch.Tensor]: ...


InverseStep = InverseStep2D | InverseStep3D


def _contract_error(
    field: str,
    expected: object,
    actual: object,
    *,
    object_name: str = "PropagationRequest",
) -> WaveContractError:
    """Build a consistently attributed internal-contract error."""
    return WaveContractError(
        "invalid internal Wave engine contract",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _owned_mapping(
    mapping: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Freeze a copied mapping while retaining its live tensors exactly."""
    return MappingProxyType(dict(mapping))


def _unique_strings(value: tuple[str, ...]) -> bool:
    """Return whether a tuple is non-empty, string-only, and unique."""
    return bool(value) and all(type(item) is str and item for item in value) and (
        len(set(value)) == len(value)
    )


@dataclass(frozen=True, slots=True)
class WaveEquationDeclaration:
    """Immutable equation identity and storage declarations."""

    identifier: str
    dimension: int
    required_model_fields: tuple[str, ...]
    model_units: Mapping[str, str]
    state_fields: tuple[str, ...]
    cpml_fields: tuple[str, ...]
    declared_components: tuple[str, ...]
    source_component: str
    source_injection: Literal["additive"]

    def __post_init__(self) -> None:
        """Own declaration containers and reject internally inconsistent schemas."""
        tuple_fields = (
            self.required_model_fields,
            self.state_fields,
            self.declared_components,
        )
        if (
            type(self.identifier) is not str
            or not self.identifier
            or isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension not in (2, 3)
            or any(not _unique_strings(value) for value in tuple_fields)
            or any(type(item) is not str or not item for item in self.cpml_fields)
            or len(set(self.cpml_fields)) != len(self.cpml_fields)
            or set(self.state_fields) & set(self.cpml_fields)
            or type(self.source_component) is not str
            or self.source_component not in self.declared_components
            or self.source_injection != "additive"
        ):
            raise WaveContractError(
                "invalid Wave equation declaration",
                object_name=type(self).__name__,
                field="declaration",
                expected="consistent immutable equation declarations",
                actual=self.identifier,
            )
        units = dict(self.model_units)
        if set(units) != set(self.required_model_fields) or any(
            type(name) is not str or type(unit) is not str or not unit
            for name, unit in units.items()
        ):
            raise WaveContractError(
                "invalid Wave equation model units",
                object_name=type(self).__name__,
                field="model_units",
                expected=f"units for {self.required_model_fields}",
                actual=tuple(units),
            )
        object.__setattr__(self, "required_model_fields", tuple(self.required_model_fields))
        object.__setattr__(self, "state_fields", tuple(self.state_fields))
        object.__setattr__(self, "cpml_fields", tuple(self.cpml_fields))
        object.__setattr__(self, "declared_components", tuple(self.declared_components))
        object.__setattr__(self, "model_units", MappingProxyType(units))


@runtime_checkable
class WaveEquationProtocol(Protocol):
    """Dimension-neutral equation surface consumed by the eager traversal."""

    @property
    def declaration(self) -> WaveEquationDeclaration: ...

    @property
    def inverse_step(self) -> InverseStep | None: ...

    def cfl_limit(
        self, model: Mapping[str, torch.Tensor], spacing: tuple[float, ...]
    ) -> float: ...

    def prepare_model(
        self,
        model: Mapping[str, torch.Tensor],
        *,
        dt: float,
        spacing: tuple[float, ...],
    ) -> Mapping[str, torch.Tensor]: ...

    def initialize_state(
        self,
        n_shot: int,
        mesh_shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Sequence[torch.Tensor]: ...

    def advance(
        self,
        state: Sequence[torch.Tensor],
        coefficients: Mapping[str, torch.Tensor],
        *,
        boundary: object,
        dt: float,
        spacing: tuple[float, ...],
    ) -> Sequence[torch.Tensor]: ...

    def inject_sources(
        self,
        state: Sequence[torch.Tensor],
        source_indices: torch.Tensor,
        source_shot_index: torch.Tensor,
        amplitudes: torch.Tensor,
        *,
        dt: float,
    ) -> Sequence[torch.Tensor]: ...

    def sample_receivers(
        self,
        state: Sequence[torch.Tensor],
        receiver_indices: torch.Tensor,
        receiver_shot_index: torch.Tensor,
        components: tuple[str, ...],
    ) -> Mapping[str, torch.Tensor]: ...

    def snapshot_fields(
        self, state: Sequence[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]: ...

    def illumination_fields(
        self, state: Sequence[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]: ...


@runtime_checkable
class WaveBackendProtocol(Protocol):
    """Minimum backend execution surface."""

    @property
    def name(self) -> str: ...

    def execute(self, request: PropagationRequest) -> PropagationResult: ...


@dataclass(frozen=True, slots=True)
class WaveMemoryGuarantees:
    """Immutable guarantees made by a memory execution strategy."""

    strategy: str
    supports_autograd: bool
    preserves_forward_values: bool

    def __post_init__(self) -> None:
        """Validate exact, immutable execution guarantees."""
        if (
            type(self.strategy) is not str
            or not self.strategy
            or type(self.supports_autograd) is not bool
            or type(self.preserves_forward_values) is not bool
        ):
            raise WaveContractError(
                "invalid Wave memory guarantees",
                object_name=type(self).__name__,
                field="guarantees",
                expected="strategy string and boolean guarantees",
                actual=self.strategy,
            )


@runtime_checkable
class WaveMemoryProtocol(Protocol):
    """Minimum memory-strategy execution and guarantees surface."""

    @property
    def guarantees(self) -> WaveMemoryGuarantees: ...

    def execute(
        self, request: PropagationRequest, backend: WaveBackendProtocol
    ) -> PropagationResult: ...


@dataclass(frozen=True, slots=True)
class CompiledAcquisition:
    """Packed acquisition indices compiled for execution."""

    source_indices: torch.Tensor
    receiver_indices: torch.Tensor
    source_shot_index: torch.Tensor
    receiver_shot_index: torch.Tensor
    n_shot: int
    nt: int
    dt: float
    t0: float
    survey_fingerprint: str
    mesh_shape: tuple[int, ...]
    spacing: tuple[float, ...]

    def __post_init__(self) -> None:
        """Own integer geometry and immutable CPU shot maps."""
        object_name = type(self).__name__
        for field, value in (
            ("source_indices", self.source_indices),
            ("receiver_indices", self.receiver_indices),
            ("source_shot_index", self.source_shot_index),
            ("receiver_shot_index", self.receiver_shot_index),
        ):
            if not isinstance(value, torch.Tensor):
                raise _contract_error(
                    field,
                    "torch.Tensor",
                    type(value).__name__,
                    object_name=object_name,
                )
            if value.dtype is not torch.int64:
                raise _contract_error(
                    field, "torch.int64", value.dtype, object_name=object_name
                )
        if (
            self.source_indices.ndim != 2
            or self.receiver_indices.ndim != 2
            or self.source_indices.shape[1] not in (2, 3)
            or self.receiver_indices.shape[1] != self.source_indices.shape[1]
        ):
            raise _contract_error(
                "indices",
                "source/receiver shape (n, 2) or (n, 3)",
                (tuple(self.source_indices.shape), tuple(self.receiver_indices.shape)),
                object_name=object_name,
            )
        if self.source_indices.device != self.receiver_indices.device:
            raise _contract_error(
                "indices",
                "source and receiver indices on one execution device",
                (self.source_indices.device, self.receiver_indices.device),
                object_name=object_name,
            )
        if (
            self.source_shot_index.device.type != "cpu"
            or self.receiver_shot_index.device.type != "cpu"
            or self.source_shot_index.ndim != 1
            or self.receiver_shot_index.ndim != 1
            or self.source_shot_index.shape[0] != self.source_indices.shape[0]
            or self.receiver_shot_index.shape[0] != self.receiver_indices.shape[0]
        ):
            raise _contract_error(
                "shot_index",
                "owned CPU vectors matching packed geometry",
                (
                    tuple(self.source_shot_index.shape),
                    tuple(self.receiver_shot_index.shape),
                ),
                object_name=object_name,
            )
        if (
            isinstance(self.n_shot, bool)
            or not isinstance(self.n_shot, int)
            or self.n_shot <= 0
            or isinstance(self.nt, bool)
            or not isinstance(self.nt, int)
            or self.nt <= 0
            or type(self.survey_fingerprint) is not str
            or not self.survey_fingerprint
        ):
            raise _contract_error(
                "acquisition",
                "positive shot/time counts and survey fingerprint",
                (self.n_shot, self.nt, self.survey_fingerprint),
                object_name=object_name,
            )
        dimension = int(self.source_indices.shape[1])
        if (
            type(self.mesh_shape) is not tuple
            or len(self.mesh_shape) != dimension
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.mesh_shape
            )
        ):
            raise _contract_error(
                "mesh_shape",
                f"{dimension} positive integer dimensions",
                self.mesh_shape,
                object_name=object_name,
            )
        if type(self.spacing) is not tuple or len(self.spacing) != dimension:
            raise _contract_error(
                "spacing",
                f"{dimension} positive finite values",
                self.spacing,
                object_name=object_name,
            )
        try:
            spacing = tuple(float(value) for value in self.spacing)
        except (TypeError, ValueError, OverflowError) as exc:
            raise _contract_error(
                "spacing",
                f"{dimension} positive finite values",
                self.spacing,
                object_name=object_name,
            ) from exc
        if any(not math.isfinite(value) or value <= 0.0 for value in spacing):
            raise _contract_error(
                "spacing",
                f"{dimension} positive finite values",
                self.spacing,
                object_name=object_name,
            )
        if isinstance(self.dt, bool) or not isinstance(self.dt, (int, float)):
            raise _contract_error(
                "dt", "positive finite number", self.dt, object_name=object_name
            )
        if isinstance(self.t0, bool) or not isinstance(self.t0, (int, float)):
            raise _contract_error(
                "t0", "finite number", self.t0, object_name=object_name
            )
        try:
            dt = float(self.dt)
        except (ValueError, OverflowError) as exc:
            raise _contract_error(
                "dt",
                "positive finite number",
                self.dt,
                object_name=object_name,
            ) from exc
        try:
            t0 = float(self.t0)
        except (ValueError, OverflowError) as exc:
            raise _contract_error(
                "t0", "finite number", self.t0, object_name=object_name
            ) from exc
        if not math.isfinite(dt) or dt <= 0.0:
            raise _contract_error(
                "dt", "positive finite float", self.dt, object_name=object_name
            )
        if not math.isfinite(t0):
            raise _contract_error(
                "t0", "finite float", self.t0, object_name=object_name
            )
        for field, indices in (
            ("source_indices", self.source_indices),
            ("receiver_indices", self.receiver_indices),
        ):
            if indices.device.type != "meta" and bool((indices < 0).any()):
                raise _contract_error(
                    field,
                    "non-negative platform cell indices",
                    "negative index",
                    object_name=object_name,
                )
            if indices.device.type != "meta":
                for axis, size in enumerate(self.mesh_shape):
                    if bool((indices[:, axis] >= size).any()):
                        raise _contract_error(
                            field,
                            f"indices within mesh_shape {self.mesh_shape}",
                            f"out-of-bounds axis {axis}",
                            object_name=object_name,
                        )
        expected_shots = torch.arange(self.n_shot, dtype=torch.int64)
        for field, shot_index in (
            ("source_shot_index", self.source_shot_index),
            ("receiver_shot_index", self.receiver_shot_index),
        ):
            actual_shots = torch.unique(shot_index, sorted=True)
            if not torch.equal(actual_shots, expected_shots):
                raise _contract_error(
                    field,
                    f"exact contiguous shot domain 0..{self.n_shot - 1}",
                    actual_shots.tolist(),
                    object_name=object_name,
                )
        object.__setattr__(self, "source_indices", self.source_indices.clone())
        object.__setattr__(self, "receiver_indices", self.receiver_indices.clone())
        object.__setattr__(
            self, "source_shot_index", self.source_shot_index.detach().clone()
        )
        object.__setattr__(
            self, "receiver_shot_index", self.receiver_shot_index.detach().clone()
        )
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        object.__setattr__(self, "spacing", spacing)

    @property
    def n_source(self) -> int:
        """Number of packed sources."""
        return int(self.source_indices.shape[0])

    @property
    def n_trace(self) -> int:
        """Number of packed traces."""
        return int(self.receiver_indices.shape[0])


@dataclass(frozen=True, slots=True)
class PropagationRequest:
    """Validated immutable propagation request."""

    equation: WaveEquationProtocol
    backend: WaveBackendProtocol
    memory: WaveMemoryProtocol
    acquisition: CompiledAcquisition
    model: Mapping[str, torch.Tensor]
    wavelets: torch.Tensor
    config: WaveSimulationConfig
    components: tuple[str, ...]
    output_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate request compatibility without normalizing live tensors."""
        if not isinstance(self.acquisition, CompiledAcquisition):
            raise _contract_error(
                "acquisition", "CompiledAcquisition", type(self.acquisition).__name__
            )
        if not isinstance(self.config, WaveSimulationConfig):
            raise _contract_error(
                "config", "WaveSimulationConfig", type(self.config).__name__
            )
        for field, value, protocol in (
            ("equation", self.equation, WaveEquationProtocol),
            ("backend", self.backend, WaveBackendProtocol),
            ("memory", self.memory, WaveMemoryProtocol),
        ):
            try:
                conforms = isinstance(value, protocol)
            except Exception as exc:
                raise _contract_error(
                    field, protocol.__name__, type(value).__name__
                ) from exc
            if not conforms:
                raise _contract_error(field, protocol.__name__, type(value).__name__)

        try:
            declaration = self.equation.declaration
            inverse_step = self.equation.inverse_step
            cfl_limit = self.equation.cfl_limit
            prepare_model = self.equation.prepare_model
            initialize_state = self.equation.initialize_state
            advance = self.equation.advance
            inject_sources = self.equation.inject_sources
            sample_receivers = self.equation.sample_receivers
            snapshot_fields = self.equation.snapshot_fields
            illumination_fields = self.equation.illumination_fields
        except Exception as exc:
            raise _contract_error(
                "equation",
                "accessible equation declaration and execution members",
                type(self.equation).__name__,
            ) from exc
        if (
            not isinstance(declaration, WaveEquationDeclaration)
            or not callable(cfl_limit)
            or not callable(prepare_model)
            or not callable(initialize_state)
            or not callable(advance)
            or not callable(inject_sources)
            or not callable(sample_receivers)
            or not callable(snapshot_fields)
            or not callable(illumination_fields)
            or (inverse_step is not None and not callable(inverse_step))
        ):
            raise _contract_error(
                "equation",
                (
                    "WaveEquationDeclaration, dimension-neutral execution methods, "
                    "and callable-or-null inverse_step"
                ),
                type(self.equation).__name__,
            )
        try:
            backend_name = self.backend.name
            backend_execute = self.backend.execute
        except Exception as exc:
            raise _contract_error(
                "backend",
                "accessible backend name and execute",
                type(self.backend).__name__,
            ) from exc
        if (
            type(backend_name) is not str
            or not backend_name
            or not callable(backend_execute)
        ):
            raise _contract_error(
                "backend",
                "non-empty name and callable execute",
                type(self.backend).__name__,
            )
        try:
            guarantees = self.memory.guarantees
            memory_execute = self.memory.execute
        except Exception as exc:
            raise _contract_error(
                "memory",
                "accessible memory guarantees and execute",
                type(self.memory).__name__,
            ) from exc
        if not isinstance(guarantees, WaveMemoryGuarantees) or not callable(
            memory_execute
        ):
            raise _contract_error(
                "memory",
                "WaveMemoryGuarantees and callable execute",
                type(self.memory).__name__,
            )
        if backend_name != self.config.backend.name:
            raise _contract_error(
                "backend",
                f"backend selected by config ({self.config.backend.name!r})",
                backend_name,
            )
        if guarantees.strategy != self.config.memory.strategy:
            raise _contract_error(
                "memory",
                f"strategy selected by config ({self.config.memory.strategy!r})",
                guarantees.strategy,
            )
        if declaration.dimension != int(self.acquisition.source_indices.shape[1]):
            raise _contract_error(
                "acquisition",
                f"{declaration.dimension}-D acquisition",
                int(self.acquisition.source_indices.shape[1]),
            )
        if not isinstance(self.model, MappingABC):
            raise _contract_error("model", "mapping of live tensors", type(self.model).__name__)
        copied_model = dict(self.model)
        for name, tensor in copied_model.items():
            if type(name) is not str or not isinstance(tensor, torch.Tensor):
                raise _contract_error(name, "named torch.Tensor", type(tensor).__name__)
        for name in declaration.required_model_fields:
            if name not in copied_model:
                raise _contract_error(name, "required live model tensor", "missing")
        reference = copied_model[declaration.required_model_fields[0]]
        reference_shape = tuple(reference.shape)
        for name in declaration.required_model_fields:
            tensor = copied_model[name]
            if (
                tensor.dtype is not reference.dtype
                or tensor.device != reference.device
                or tuple(tensor.shape) != reference_shape
            ):
                raise _contract_error(
                    name,
                    (
                        f"dtype={reference.dtype}, device={reference.device}, "
                        f"shape={reference_shape}"
                    ),
                    (
                        f"dtype={tensor.dtype}, device={tensor.device}, "
                        f"shape={tuple(tensor.shape)}"
                    ),
                )
            if tuple(tensor.shape) != self.acquisition.mesh_shape:
                raise _contract_error(
                    name,
                    f"spatial shape={self.acquisition.mesh_shape}",
                    f"shape={tuple(tensor.shape)}",
                )
        if not isinstance(self.wavelets, torch.Tensor):
            raise _contract_error("wavelets", "torch.Tensor", type(self.wavelets).__name__)
        expected_wavelet_shape = (self.acquisition.n_source, self.acquisition.nt)
        if (
            tuple(self.wavelets.shape) != expected_wavelet_shape
            or self.wavelets.dtype is not reference.dtype
            or self.wavelets.device != reference.device
        ):
            raise _contract_error(
                "wavelets",
                (
                    f"shape={expected_wavelet_shape}, dtype={reference.dtype}, "
                    f"device={reference.device}"
                ),
                (
                    f"shape={tuple(self.wavelets.shape)}, dtype={self.wavelets.dtype}, "
                    f"device={self.wavelets.device}"
                ),
            )
        if (
            self.acquisition.source_indices.device != reference.device
            or self.acquisition.receiver_indices.device != reference.device
        ):
            raise _contract_error(
                "acquisition",
                f"execution indices on model device {reference.device}",
                (
                    self.acquisition.source_indices.device,
                    self.acquisition.receiver_indices.device,
                ),
            )
        if (
            type(self.components) is not tuple
            or not _unique_strings(self.components)
            or any(
                component not in declaration.declared_components
                for component in self.components
            )
        ):
            raise _contract_error(
                "components",
                f"unique tuple drawn from {declaration.declared_components}",
                self.components,
            )
        if self.components != self.config.output.components:
            raise _contract_error(
                "components",
                f"exact config output components {self.config.output.components}",
                self.components,
            )
        if type(self.output_indices) is not tuple or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.acquisition.nt
            for index in self.output_indices
        ) or len(set(self.output_indices)) != len(self.output_indices):
            raise _contract_error(
                "output_indices",
                f"unique integer tuple within [0, {self.acquisition.nt})",
                self.output_indices,
            )
        configured_indices = self.config.output.snapshot_indices
        if self.config.output.snapshot_policy == "selected":
            if (
                not configured_indices
                or len(set(configured_indices)) != len(configured_indices)
                or any(
                    index >= self.acquisition.nt
                    for index in configured_indices
                )
            ):
                raise _contract_error(
                    "snapshot_indices",
                    (
                        "unique non-empty integer tuple within "
                        f"[0, {self.acquisition.nt}) for selected snapshots"
                    ),
                    configured_indices,
                )
        elif configured_indices:
            raise _contract_error(
                "snapshot_indices",
                (
                    "empty tuple unless snapshot_policy is 'selected'"
                ),
                configured_indices,
            )
        if self.output_indices != configured_indices:
            raise _contract_error(
                "output_indices",
                f"exact configured snapshot indices {configured_indices}",
                self.output_indices,
            )
        object.__setattr__(self, "model", _owned_mapping(copied_model))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "output_indices", tuple(self.output_indices))


@dataclass(frozen=True, slots=True)
class ExecutionAccounting:
    """Observed propagation work and retained-state accounting."""

    forward_steps: int
    recomputed_steps: int
    complete_forwards: int
    saved_state_bytes: int
    peak_live_state_bytes: int

    def __post_init__(self) -> None:
        """Reject negative or non-integral accounting observations."""
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.forward_steps,
                self.recomputed_steps,
                self.complete_forwards,
                self.saved_state_bytes,
                self.peak_live_state_bytes,
            )
        ):
            raise WaveContractError(
                "invalid execution accounting",
                object_name=type(self).__name__,
                field="accounting",
                expected="non-negative integers",
                actual="invalid value",
            )


class ExecutionTelemetry:
    """Mutable private observations with immutable accounting snapshots.

    Propagation returns before checkpoint recomputation occurs.  Strategies
    therefore record work into this private telemetry object and callers inspect
    a fresh :class:`ExecutionAccounting` through
    :meth:`PropagationResult.accounting_snapshot` after backward.  Previously
    published snapshots never mutate.

    ``peak_live_state_bytes`` is an exact high-water mark for tensor storages
    explicitly owned or observed by the selected strategy.  It deliberately
    excludes PyTorch-internal backward-replay temporaries: public saved-tensor
    hooks do not compose with non-reentrant checkpoint's own hooks, and relying
    on private framework hook stacks would make this platform contract
    unstable.
    """

    __slots__ = (
        "_forward_steps",
        "_nt",
        "_peak_live_state_bytes",
        "_recomputing",
        "_recomputed_steps",
        "_saved_state_bytes",
        "_storage_keys",
    )

    def __init__(self, nt: int) -> None:
        if isinstance(nt, bool) or not isinstance(nt, int) or nt <= 0:
            raise WaveContractError(
                "invalid execution telemetry",
                object_name=type(self).__name__,
                field="nt",
                expected="positive integer",
                actual=nt,
            )
        self._nt = nt
        self._forward_steps = 0
        self._recomputed_steps = 0
        self._saved_state_bytes = 0
        self._peak_live_state_bytes = 0
        self._recomputing = False
        self._storage_keys: set[tuple[str, int, int]] = set()

    def record_advance(self) -> None:
        """Record one actual equation advance in forward or backward replay."""
        if self._recomputing:
            self._recomputed_steps += 1
        else:
            self._forward_steps += 1

    @contextmanager
    def recompute_region(self) -> Iterator[None]:
        """Mark advances in a public checkpoint ``context_fn`` replay region."""
        previous = self._recomputing
        self._recomputing = True
        try:
            yield
        finally:
            self._recomputing = previous

    def record_recomputed_advance(self) -> None:
        """Record one explicit custom-VJP replay advance."""
        self._recomputed_steps += 1

    @staticmethod
    def _storage_identity(tensor: torch.Tensor) -> tuple[tuple[str, int, int], int]:
        storage = tensor.untyped_storage()
        size = int(storage.nbytes())
        pointer = int(storage.data_ptr())
        if pointer == 0:
            pointer = id(storage)
        return (str(tensor.device), pointer, size), size

    def observe_saved_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Count one strategy-owned saved storage, deduplicating aliases."""
        key, size = self._storage_identity(tensor)
        if key not in self._storage_keys:
            self._storage_keys.add(key)
            self._saved_state_bytes += size
        self._peak_live_state_bytes = max(
            self._peak_live_state_bytes, self._saved_state_bytes
        )
        return tensor

    def observe_live_state(self, state: Sequence[torch.Tensor]) -> None:
        """Update the logical peak from unique currently-live state storages."""
        live: dict[tuple[str, int, int], int] = {}
        for tensor in state:
            key, size = self._storage_identity(tensor)
            live[key] = size
        self._peak_live_state_bytes = max(
            self._peak_live_state_bytes,
            self._saved_state_bytes
            + sum(
                size
                for key, size in live.items()
                if key not in self._storage_keys
            ),
        )

    def snapshot(self) -> ExecutionAccounting:
        """Return immutable accounting values observed at this lifecycle point."""
        complete_forwards = self._forward_steps // self._nt
        return ExecutionAccounting(
            forward_steps=self._forward_steps,
            recomputed_steps=self._recomputed_steps,
            complete_forwards=complete_forwards,
            saved_state_bytes=self._saved_state_bytes,
            peak_live_state_bytes=self._peak_live_state_bytes,
        )


class EagerStrategyBackendProtocol(WaveBackendProtocol, Protocol):
    """Internal eager operations available to selected memory strategies."""

    def prepare(
        self,
        request: PropagationRequest,
        telemetry: ExecutionTelemetry,
    ) -> tuple[Any, tuple[torch.Tensor, ...], tuple[int, ...]]: ...

    def run_segment(
        self,
        context: Any,
        state: Sequence[torch.Tensor],
        wavelets: torch.Tensor,
        *,
        time_start: int,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]: ...

    def assemble(
        self,
        request: PropagationRequest,
        context: Any,
        state: tuple[torch.Tensor, ...],
        records: torch.Tensor,
        collections: Mapping[str, torch.Tensor],
        telemetry: ExecutionTelemetry,
        *,
        diagnostics: Mapping[str, object] | None = None,
    ) -> PropagationResult: ...


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Internal packed result returned by propagation execution."""

    traces: torch.Tensor
    fields: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, object]
    accounting: ExecutionAccounting
    complete: bool
    _telemetry: ExecutionTelemetry | None = None

    def __post_init__(self) -> None:
        """Own result containers without altering live result tensors."""
        object_name = type(self).__name__
        if not isinstance(self.traces, torch.Tensor):
            raise _contract_error(
                "traces",
                "torch.Tensor",
                type(self.traces).__name__,
                object_name=object_name,
            )
        if not isinstance(self.fields, MappingABC):
            raise _contract_error(
                "fields",
                "mapping of named tensors",
                type(self.fields).__name__,
                object_name=object_name,
            )
        if not isinstance(self.diagnostics, MappingABC):
            raise _contract_error(
                "diagnostics",
                "mapping with non-empty string keys",
                type(self.diagnostics).__name__,
                object_name=object_name,
            )
        try:
            copied_fields = dict(self.fields)
            copied_diagnostics = dict(self.diagnostics)
        except Exception as exc:
            raise _contract_error(
                "fields/diagnostics",
                "readable mappings",
                "mapping access failed",
                object_name=object_name,
            ) from exc
        for name, tensor in copied_fields.items():
            if type(name) is not str or not name or not isinstance(tensor, torch.Tensor):
                raise _contract_error(
                    "fields",
                    "non-empty string keys with torch.Tensor values",
                    name,
                    object_name=object_name,
                )
        if any(type(name) is not str or not name for name in copied_diagnostics):
            raise _contract_error(
                "diagnostics",
                "non-empty string keys",
                tuple(copied_diagnostics),
                object_name=object_name,
            )
        if not isinstance(self.accounting, ExecutionAccounting):
            raise _contract_error(
                "accounting",
                "ExecutionAccounting",
                type(self.accounting).__name__,
                object_name=object_name,
            )
        if type(self.complete) is not bool:
            raise _contract_error(
                "complete", "bool", type(self.complete).__name__, object_name=object_name
            )
        if self._telemetry is not None and not isinstance(
            self._telemetry, ExecutionTelemetry
        ):
            raise _contract_error(
                "_telemetry",
                "ExecutionTelemetry or None",
                type(self._telemetry).__name__,
                object_name=object_name,
            )
        object.__setattr__(self, "fields", MappingProxyType(copied_fields))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(copied_diagnostics)
        )

    def accounting_snapshot(self) -> ExecutionAccounting:
        """Return final observations when called after any requested backward."""
        return (
            self.accounting
            if self._telemetry is None
            else self._telemetry.snapshot()
        )

__all__ = [
    "CompiledAcquisition",
    "EagerStrategyBackendProtocol",
    "ExecutionAccounting",
    "ExecutionTelemetry",
    "PropagationRequest",
    "PropagationResult",
    "WaveBackendProtocol",
    "WaveEquationDeclaration",
    "WaveEquationProtocol",
    "WaveMemoryGuarantees",
    "WaveMemoryProtocol",
]
