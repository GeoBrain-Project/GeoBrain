"""Immutable configuration records for Wave simulation contracts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn, cast

from .errors import WaveContractError


_VALIDATION_HINT = "correct the field value and retry"
_MAX_FINITE_FLOAT_INTEGER = int(sys.float_info.max)


def _safe_actual(value: object) -> object:
    """Preserve diagnostics without repr-converting an unrepresentably large integer."""
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) > _MAX_FINITE_FLOAT_INTEGER
    ):
        return f"<integer with {value.bit_length()} bits>"
    try:
        repr(value)
    except (OverflowError, ValueError):
        if isinstance(value, int):
            return f"<integer with {value.bit_length()} bits>"
        return f"<unrepresentable {type(value).__qualname__}>"
    return value


def _invalid(
    field: str, expected: object, actual: object, *, object_name: str
) -> NoReturn:
    """Raise the common structured error used by closed Wave config schemas."""
    raise WaveContractError(
        "invalid Wave configuration value",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=_safe_actual(actual),
        hint=_VALIDATION_HINT,
    )


def _require_exact_or_partial_keys(
    payload: Mapping[str, object], *, allowed: set[str], object_name: str
) -> dict[str, object]:
    """Validate mapping input while allowing omitted fields to use dataclass defaults."""
    if not isinstance(payload, Mapping):
        _invalid("payload", "mapping", type(payload).__name__, object_name=object_name)
    if not all(isinstance(key, str) for key in payload):
        _invalid("keys", "string keys", [repr(key) for key in payload], object_name=object_name)
    keys = set(payload)
    unknown = sorted(keys - allowed)
    if unknown:
        _invalid(unknown[0], sorted(allowed), sorted(keys), object_name=object_name)
    return dict(payload)


def _is_int(value: object) -> bool:
    """Return whether a value is a non-boolean integer."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    """Return whether a value is a finite non-boolean numeric scalar."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _tuple_of_strings(
    value: object, *, field_name: str, object_name: str, unique: bool = False
) -> tuple[str, ...]:
    """Detach a sequence of strings from caller-owned mutable storage."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(field_name, "sequence of strings", value, object_name=object_name)
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        _invalid(field_name, "sequence of non-empty strings", value, object_name=object_name)
    if unique and len(set(result)) != len(result):
        _invalid(field_name, "unique ordered strings", value, object_name=object_name)
    return result


def _tuple_of_indices(value: object, *, object_name: str) -> tuple[int, ...]:
    """Detach and validate non-negative snapshot indices."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(
            "snapshot_indices", "sequence of non-negative integers", value, object_name=object_name
        )
    result = tuple(value)
    if any(not _is_int(item) or item < 0 for item in result):
        _invalid(
            "snapshot_indices", "sequence of non-negative integers", value, object_name=object_name
        )
    if len(set(result)) != len(result):
        _invalid(
            "snapshot_indices",
            "unique ordered non-negative integers",
            value,
            object_name=object_name,
        )
    return result


@dataclass(frozen=True, slots=True)
class WaveDiscretizationConfig:
    """Finite-difference quality and stability policy.

    Attributes:
        fd_order: spatial finite-difference order.
        strict_cfl: fail (True) vs warn on CFL violation.
        min_points_per_wavelength: dispersion quality bound.
        quality_policy: how quality violations are handled.
    """

    fd_order: int = 4
    strict_cfl: bool = True
    min_points_per_wavelength: float = 5.0
    quality_policy: Literal["error", "degraded"] = "error"

    def __post_init__(self) -> None:
        """Validate finite-difference values at construction time."""
        object_name = type(self).__name__
        if not _is_int(self.fd_order) or self.fd_order < 2 or self.fd_order % 2:
            _invalid("fd_order", "even integer >= 2", self.fd_order, object_name=object_name)
        if not isinstance(self.strict_cfl, bool):
            _invalid("strict_cfl", "boolean", self.strict_cfl, object_name=object_name)
        if (
            not _is_finite_number(self.min_points_per_wavelength)
            or self.min_points_per_wavelength <= 0
        ):
            _invalid(
                "min_points_per_wavelength",
                "positive finite number",
                self.min_points_per_wavelength,
                object_name=object_name,
            )
        if self.quality_policy not in ("error", "degraded"):
            _invalid(
                "quality_policy",
                "'error' or 'degraded'",
                self.quality_policy,
                object_name=object_name,
            )
        object.__setattr__(self, "min_points_per_wavelength", float(self.min_points_per_wavelength))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe finite-difference configuration object."""
        return {
            "fd_order": self.fd_order,
            "strict_cfl": self.strict_cfl,
            "min_points_per_wavelength": self.min_points_per_wavelength,
            "quality_policy": self.quality_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WaveDiscretizationConfig:
        """Parse a partial, closed-schema finite-difference configuration."""
        values = _require_exact_or_partial_keys(
            payload,
            allowed={"fd_order", "strict_cfl", "min_points_per_wavelength", "quality_policy"},
            object_name=cls.__name__,
        )
        return cls(
            fd_order=cast(int, values.get("fd_order", 4)),
            strict_cfl=cast(bool, values.get("strict_cfl", True)),
            min_points_per_wavelength=cast(
                float, values.get("min_points_per_wavelength", 5.0)
            ),
            quality_policy=cast(
                Literal["error", "degraded"], values.get("quality_policy", "error")
            ),
        )


@dataclass(frozen=True, slots=True)
class WaveBoundaryConfig:
    """Boundary absorption and free-surface policy.

    Attributes:
        kind: boundary type (``'pml'`` / ``'habc'`` ...).
        layers: absorbing-layer thickness [cells].
        free_surface: free surface at the top when True.
        target_reflection: designed boundary reflection coefficient.
        profile_order / kappa_max / alpha_max: CPML profile parameters.
    """

    kind: Literal["cpml", "none"] = "cpml"
    layers: int = 30
    free_surface: bool = False
    target_reflection: float = 1.0e-4
    profile_order: float = 2.0
    kappa_max: float = 1.0
    alpha_max: float = 0.0

    def __post_init__(self) -> None:
        """Validate boundary-policy values without calculating numerical coefficients."""
        object_name = type(self).__name__
        if self.kind not in ("cpml", "none"):
            _invalid("kind", "'cpml' or 'none'", self.kind, object_name=object_name)
        if not _is_int(self.layers) or self.layers < 0:
            _invalid("layers", "non-negative integer", self.layers, object_name=object_name)
        if self.kind == "cpml" and self.layers <= 0:
            _invalid("layers", "positive integer for CPML", self.layers, object_name=object_name)
        if not isinstance(self.free_surface, bool):
            _invalid("free_surface", "boolean", self.free_surface, object_name=object_name)
        if not _is_finite_number(self.target_reflection) or not 0 < self.target_reflection < 1:
            _invalid(
                "target_reflection",
                "finite number strictly between 0 and 1",
                self.target_reflection,
                object_name=object_name,
            )
        if not _is_finite_number(self.profile_order) or self.profile_order <= 0:
            _invalid(
                "profile_order",
                "positive finite number",
                self.profile_order,
                object_name=object_name,
            )
        if not _is_finite_number(self.kappa_max) or self.kappa_max < 1:
            _invalid("kappa_max", "finite number >= 1", self.kappa_max, object_name=object_name)
        if not _is_finite_number(self.alpha_max) or self.alpha_max < 0:
            _invalid(
                "alpha_max", "non-negative finite number", self.alpha_max, object_name=object_name
            )
        for name in ("target_reflection", "profile_order", "kappa_max", "alpha_max"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe boundary configuration object."""
        return {
            "kind": self.kind,
            "layers": self.layers,
            "free_surface": self.free_surface,
            "target_reflection": self.target_reflection,
            "profile_order": self.profile_order,
            "kappa_max": self.kappa_max,
            "alpha_max": self.alpha_max,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WaveBoundaryConfig:
        """Parse a partial, closed-schema boundary configuration."""
        values = _require_exact_or_partial_keys(
            payload,
            allowed={
                "kind",
                "layers",
                "free_surface",
                "target_reflection",
                "profile_order",
                "kappa_max",
                "alpha_max",
            },
            object_name=cls.__name__,
        )
        return cls(
            kind=cast(Literal["cpml", "none"], values.get("kind", "cpml")),
            layers=cast(int, values.get("layers", 30)),
            free_surface=cast(bool, values.get("free_surface", False)),
            target_reflection=cast(float, values.get("target_reflection", 1.0e-4)),
            profile_order=cast(float, values.get("profile_order", 2.0)),
            kappa_max=cast(float, values.get("kappa_max", 1.0)),
            alpha_max=cast(float, values.get("alpha_max", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class WaveMemoryConfig:
    """Wavefield retention and checkpointing policy.

    Attributes:
        strategy: ``'full'`` / ``'checkpoint'`` / ``'recursive'`` /
            ``'boundary'`` wavefield-memory strategy.
        checkpoint_segments: segment count for uniform checkpointing.
        recursive_leaf_steps: leaf size for treeverse bisection.
        budget_bytes: optional memory budget for auto-selection.
    """

    strategy: Literal["full", "checkpoint", "recursive", "boundary"] = "full"
    checkpoint_segments: int = 4
    recursive_leaf_steps: int = 16
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        """Validate resource policy without choosing a runtime execution plan."""
        object_name = type(self).__name__
        if self.strategy not in ("full", "checkpoint", "recursive", "boundary"):
            _invalid(
                "strategy",
                "'full', 'checkpoint', 'recursive', or 'boundary'",
                self.strategy,
                object_name=object_name,
            )
        if not _is_int(self.checkpoint_segments) or self.checkpoint_segments <= 0:
            _invalid(
                "checkpoint_segments",
                "positive integer",
                self.checkpoint_segments,
                object_name=object_name,
            )
        if not _is_int(self.recursive_leaf_steps) or self.recursive_leaf_steps <= 0:
            _invalid(
                "recursive_leaf_steps",
                "positive integer",
                self.recursive_leaf_steps,
                object_name=object_name,
            )
        if self.budget_bytes is not None and (
            not _is_int(self.budget_bytes) or self.budget_bytes <= 0
        ):
            _invalid(
                "budget_bytes",
                "positive integer or null",
                self.budget_bytes,
                object_name=object_name,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe memory configuration object."""
        return {
            "strategy": self.strategy,
            "checkpoint_segments": self.checkpoint_segments,
            "recursive_leaf_steps": self.recursive_leaf_steps,
            "budget_bytes": self.budget_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WaveMemoryConfig:
        """Parse a partial, closed-schema memory configuration."""
        values = _require_exact_or_partial_keys(
            payload,
            allowed={"strategy", "checkpoint_segments", "recursive_leaf_steps", "budget_bytes"},
            object_name=cls.__name__,
        )
        return cls(
            strategy=cast(
                Literal["full", "checkpoint", "recursive", "boundary"],
                values.get("strategy", "full"),
            ),
            checkpoint_segments=cast(int, values.get("checkpoint_segments", 4)),
            recursive_leaf_steps=cast(int, values.get("recursive_leaf_steps", 16)),
            budget_bytes=cast(int | None, values.get("budget_bytes")),
        )


@dataclass(frozen=True, slots=True)
class WaveBackendConfig:
    """Execution backend selection and reproducibility policy.

    Attributes:
        name: kernel backend id (``'eager'`` / ``'native'`` ...; the
            same id-string convention as ``flow.linear_solver`` and
            ``em``'s ``EMExecutionConfig.linear_solver``).
        deterministic: request bitwise-deterministic execution.
    """

    name: Literal["eager", "native"] = "eager"
    deterministic: bool = True

    def __post_init__(self) -> None:
        """Validate backend names independently from backend availability."""
        object_name = type(self).__name__
        if self.name not in ("eager", "native"):
            _invalid("name", "'eager' or 'native'", self.name, object_name=object_name)
        if not isinstance(self.deterministic, bool):
            _invalid("deterministic", "boolean", self.deterministic, object_name=object_name)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe backend configuration object."""
        return {"name": self.name, "deterministic": self.deterministic}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WaveBackendConfig:
        """Parse a partial, closed-schema backend configuration."""
        values = _require_exact_or_partial_keys(
            payload, allowed={"name", "deterministic"}, object_name=cls.__name__
        )
        return cls(
            name=cast(Literal["eager", "native"], values.get("name", "eager")),
            deterministic=cast(bool, values.get("deterministic", True)),
        )


@dataclass(frozen=True, slots=True)
class WaveOutputConfig:
    """Requested wave components, snapshots, and gradient retention policy.

    Attributes:
        components: which field components to record.
        snapshot_policy / snapshot_indices: wavefield snapshot control.
        retain_field_gradients: keep gradients on snapshots.
        illumination: also accumulate illumination compensation.
    """

    components: tuple[str, ...] = ("pressure",)
    snapshot_policy: Literal["none", "final", "selected", "energy"] = "none"
    snapshot_indices: tuple[int, ...] = ()
    retain_field_gradients: bool = False
    illumination: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Own mutable input sequences and validate output policy values."""
        object_name = type(self).__name__
        object.__setattr__(
            self,
            "components",
            _tuple_of_strings(
                self.components, field_name="components", object_name=object_name, unique=True
            ),
        )
        object.__setattr__(
            self,
            "snapshot_indices",
            _tuple_of_indices(self.snapshot_indices, object_name=object_name),
        )
        object.__setattr__(
            self,
            "illumination",
            _tuple_of_strings(
                self.illumination,
                field_name="illumination",
                object_name=object_name,
                unique=True,
            ),
        )
        if not self.components:
            _invalid(
                "components",
                "non-empty sequence of unique non-empty strings",
                self.components,
                object_name=object_name,
            )
        if self.snapshot_policy not in ("none", "final", "selected", "energy"):
            _invalid(
                "snapshot_policy",
                "'none', 'final', 'selected', or 'energy'",
                self.snapshot_policy,
                object_name=object_name,
            )
        if self.snapshot_policy == "selected" and not self.snapshot_indices:
            _invalid(
                "snapshot_indices",
                "non-empty sequence when snapshot_policy is 'selected'",
                self.snapshot_indices,
                object_name=object_name,
            )
        if self.snapshot_policy != "selected" and self.snapshot_indices:
            _invalid(
                "snapshot_indices",
                "empty sequence unless snapshot_policy is 'selected'",
                self.snapshot_indices,
                object_name=object_name,
            )
        if not isinstance(self.retain_field_gradients, bool):
            _invalid(
                "retain_field_gradients",
                "boolean",
                self.retain_field_gradients,
                object_name=object_name,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe output configuration object."""
        return {
            "components": list(self.components),
            "snapshot_policy": self.snapshot_policy,
            "snapshot_indices": list(self.snapshot_indices),
            "retain_field_gradients": self.retain_field_gradients,
            "illumination": list(self.illumination),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WaveOutputConfig:
        """Parse a partial, closed-schema output configuration."""
        values = _require_exact_or_partial_keys(
            payload,
            allowed={
                "components",
                "snapshot_policy",
                "snapshot_indices",
                "retain_field_gradients",
                "illumination",
            },
            object_name=cls.__name__,
        )
        return cls(
            components=cast(tuple[str, ...], values.get("components", ("pressure",))),
            snapshot_policy=cast(
                Literal["none", "final", "selected", "energy"],
                values.get("snapshot_policy", "none"),
            ),
            snapshot_indices=cast(tuple[int, ...], values.get("snapshot_indices", ())),
            retain_field_gradients=cast(
                bool, values.get("retain_field_gradients", False)
            ),
            illumination=cast(tuple[str, ...], values.get("illumination", ())),
        )


@dataclass(frozen=True, slots=True)
class WaveSimulationConfig:
    """Complete immutable configuration for a Wave simulation request.

    Attributes:
        discretization: :class:`WaveDiscretizationConfig`.
        boundary: :class:`WaveBoundaryConfig`.
        memory: :class:`WaveMemoryConfig`.
        backend: :class:`WaveBackendConfig`.
        output: :class:`WaveOutputConfig`.
    """

    discretization: WaveDiscretizationConfig = field(default_factory=WaveDiscretizationConfig)
    boundary: WaveBoundaryConfig = field(default_factory=WaveBoundaryConfig)
    memory: WaveMemoryConfig = field(default_factory=WaveMemoryConfig)
    backend: WaveBackendConfig = field(default_factory=WaveBackendConfig)
    output: WaveOutputConfig = field(default_factory=WaveOutputConfig)

    def __post_init__(self) -> None:
        """Ensure direct construction receives configuration records, not mutable payloads."""
        expected_types = {
            "discretization": WaveDiscretizationConfig,
            "boundary": WaveBoundaryConfig,
            "memory": WaveMemoryConfig,
            "backend": WaveBackendConfig,
            "output": WaveOutputConfig,
        }
        for name, expected_type in expected_types.items():
            value = getattr(self, name)
            if not isinstance(value, expected_type):
                _invalid(
                    name,
                    expected_type.__name__,
                    type(value).__name__,
                    object_name=type(self).__name__,
                )

    def to_dict(self) -> dict[str, object]:
        """Return the nested JSON-safe Wave simulation configuration."""
        return {
            "discretization": self.discretization.to_dict(),
            "boundary": self.boundary.to_dict(),
            "memory": self.memory.to_dict(),
            "backend": self.backend.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> WaveSimulationConfig:
        """Parse a partial nested configuration and reject every unknown key."""
        values = _require_exact_or_partial_keys(
            payload,
            allowed={"discretization", "boundary", "memory", "backend", "output"},
            object_name=cls.__name__,
        )
        discretization = WaveDiscretizationConfig()
        if "discretization" in values:
            raw_discretization = values["discretization"]
            if not isinstance(raw_discretization, Mapping):
                _invalid(
                    "discretization",
                    "mapping",
                    type(raw_discretization).__name__,
                    object_name=cls.__name__,
                )
            discretization = WaveDiscretizationConfig.from_dict(raw_discretization)

        boundary = WaveBoundaryConfig()
        if "boundary" in values:
            raw_boundary = values["boundary"]
            if not isinstance(raw_boundary, Mapping):
                _invalid(
                    "boundary",
                    "mapping",
                    type(raw_boundary).__name__,
                    object_name=cls.__name__,
                )
            boundary = WaveBoundaryConfig.from_dict(raw_boundary)

        memory = WaveMemoryConfig()
        if "memory" in values:
            raw_memory = values["memory"]
            if not isinstance(raw_memory, Mapping):
                _invalid(
                    "memory",
                    "mapping",
                    type(raw_memory).__name__,
                    object_name=cls.__name__,
                )
            memory = WaveMemoryConfig.from_dict(raw_memory)

        backend = WaveBackendConfig()
        if "backend" in values:
            raw_backend = values["backend"]
            if not isinstance(raw_backend, Mapping):
                _invalid(
                    "backend",
                    "mapping",
                    type(raw_backend).__name__,
                    object_name=cls.__name__,
                )
            backend = WaveBackendConfig.from_dict(raw_backend)

        output = WaveOutputConfig()
        if "output" in values:
            raw_output = values["output"]
            if not isinstance(raw_output, Mapping):
                _invalid(
                    "output",
                    "mapping",
                    type(raw_output).__name__,
                    object_name=cls.__name__,
                )
            output = WaveOutputConfig.from_dict(raw_output)

        return cls(
            discretization=discretization,
            boundary=boundary,
            memory=memory,
            backend=backend,
            output=output,
        )


__all__ = [
    "WaveBackendConfig",
    "WaveBoundaryConfig",
    "WaveDiscretizationConfig",
    "WaveMemoryConfig",
    "WaveOutputConfig",
    "WaveSimulationConfig",
]
