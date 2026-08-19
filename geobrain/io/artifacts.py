"""Checksum-verified references and safe codecs for GeoBrain artifacts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Callable, NoReturn, TypeAlias, Union, cast

import numpy as np
import torch
from numpy.lib.npyio import NpzFile

from geobrain.core import ErrorCode, GeoBrainError


JsonValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]

_FORMATS = {"json", "npz", "pt"}  # each tag has a save/load codec pair
_CHECKSUM_PATTERN = re.compile(r"[0-9a-f]{64}")
_TORCH_DTYPES = {
    str(value).removeprefix("torch.")
    for value in vars(torch).values()
    if isinstance(value, torch.dtype)
}


class _FrozenSequence(tuple[object, ...]):
    """Tuple-backed metadata sequence that compares like its source JSON list."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    __hash__ = tuple.__hash__


def _raise_invalid(
    message: str,
    *,
    object_name: str,
    field: str,
    expected: object,
    actual: object,
) -> NoReturn:
    raise GeoBrainError(
        message,
        code=ErrorCode.ARTIFACT_INVALID,
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _is_json_value(value: object, active: set[int] | None = None) -> bool:
    if active is None:
        active = set()
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, _FrozenSequence)):
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(_is_json_value(item, active) for item in value)
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return False
        active.add(identity)
        try:
            return all(
                isinstance(key, str) and _is_json_value(item, active)
                for key, item in value.items()
            )
        finally:
            active.remove(identity)
    return False


def _copy_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, _FrozenSequence)):
        return [_copy_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    raise TypeError(f"value is not JSON-safe: {type(value)!r}")


def _freeze_json(value: JsonValue) -> object:
    if isinstance(value, list):
        return _FrozenSequence(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType(
            {name: _freeze_json(item) for name, item in value.items()}
        )
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, (list, _FrozenSequence)):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        return {str(name): _thaw_json(item) for name, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"frozen value is not JSON-safe: {type(value)!r}")


def _validate_metadata(
    metadata: object,
    *,
    object_name: str,
) -> dict[str, JsonValue]:
    if not isinstance(metadata, Mapping):
        _raise_invalid(
            "artifact metadata must be a JSON-safe object",
            object_name=object_name,
            field="metadata",
            expected="mapping[str, JsonValue]",
            actual=type(metadata),
        )
    copied = dict(metadata)
    if not _is_json_value(copied):
        _raise_invalid(
            "artifact metadata must be a JSON-safe object",
            object_name=object_name,
            field="metadata",
            expected="mapping[str, JsonValue] with finite numbers",
            actual=type(metadata),
        )
    return cast(dict[str, JsonValue], _copy_json(copied))


def _validate_units(units: object, *, object_name: str) -> str | dict[str, str] | None:
    if units is None:
        return None
    if isinstance(units, str):
        if units.strip():
            return units
    elif isinstance(units, Mapping) and bool(units) and all(
        isinstance(name, str)
        and bool(name.strip())
        and isinstance(unit, str)
        and bool(unit.strip())
        for name, unit in units.items()
    ):
        return dict(cast(Mapping[str, str], units))
    _raise_invalid(
        "artifact units must be a non-empty string, a string mapping, or null",
        object_name=object_name,
        field="units",
        expected="non-empty string, dict[str, non-empty str], or null",
        actual=units,
    )


def _is_canonical_dtype(dtype: str) -> bool:
    if dtype in _TORCH_DTYPES:
        return True
    try:
        return str(np.dtype(dtype)) == dtype
    except TypeError:
        return False


@dataclass(frozen=True)
class ArtifactRef:
    """JSON-round-trippable reference to a checksum-verified artifact file.

    Attributes:
        id: caller-chosen stable artifact identifier.
        path: file location the artifact was written to.
        format: payload format tag (``json`` / ``npz`` / ``pt``).
        checksum: content SHA-256.
        shape / dtype: array facts when applicable.
        units: recorded unit string or per-key mapping.
        metadata: JSON-safe extras.
    """

    id: str
    path: str
    format: str
    checksum: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    units: str | dict[str, str] | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            _raise_invalid(
                "artifact ID must be a non-empty string",
                object_name="ArtifactRef",
                field="id",
                expected="non-empty string",
                actual=self.id,
            )
        if (
            not isinstance(self.path, str)
            or not self.path.strip()
            or not Path(self.path).is_absolute()
        ):
            _raise_invalid(
                "artifact path must be a non-empty absolute path",
                object_name="ArtifactRef",
                field="path",
                expected="non-empty absolute path string",
                actual=self.path,
            )
        if not isinstance(self.format, str) or self.format not in _FORMATS:
            _raise_invalid(
                "artifact format is not supported",
                object_name="ArtifactRef",
                field="format",
                expected=sorted(_FORMATS),
                actual=self.format,
            )
        if (
            not isinstance(self.checksum, str)
            or _CHECKSUM_PATTERN.fullmatch(self.checksum) is None
        ):
            _raise_invalid(
                "artifact checksum must be a lowercase SHA-256 digest",
                object_name="ArtifactRef",
                field="checksum",
                expected="64 lowercase hexadecimal characters",
                actual=self.checksum,
            )
        if self.shape is not None and (
            not isinstance(self.shape, tuple)
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0
                for item in self.shape
            )
        ):
            _raise_invalid(
                "artifact shape must contain positive integer dimensions",
                object_name="ArtifactRef",
                field="shape",
                expected="tuple of positive integers or null",
                actual=self.shape,
            )
        if self.dtype is not None and (
            not isinstance(self.dtype, str) or not _is_canonical_dtype(self.dtype)
        ):
            _raise_invalid(
                "artifact dtype must be canonical",
                object_name="ArtifactRef",
                field="dtype",
                expected="canonical NumPy or PyTorch dtype string or null",
                actual=self.dtype,
            )

        validated_units = _validate_units(self.units, object_name="ArtifactRef")
        validated_metadata = _validate_metadata(self.metadata, object_name="ArtifactRef")
        frozen_units = (
            MappingProxyType(validated_units)
            if isinstance(validated_units, dict)
            else validated_units
        )
        frozen_metadata = MappingProxyType(
            {
                name: _freeze_json(value)
                for name, value in validated_metadata.items()
            }
        )
        object.__setattr__(self, "units", frozen_units)
        object.__setattr__(self, "metadata", frozen_metadata)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return an ordinary JSON-safe dictionary with no immutable wrappers."""
        units: JsonValue
        if isinstance(self.units, Mapping):
            units = {
                str(name): str(value) for name, value in self.units.items()
            }
        else:
            units = self.units
        return {
            "id": self.id,
            "path": self.path,
            "format": self.format,
            "checksum": self.checksum,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "units": units,
            "metadata": {
                name: _thaw_json(value) for name, value in self.metadata.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArtifactRef:
        """Build a reference from a dictionary with an exact, closed key schema."""
        if not isinstance(payload, Mapping):
            _raise_invalid(
                "artifact reference must be an object",
                object_name="ArtifactRef",
                field="payload",
                expected="mapping",
                actual=type(payload),
            )
        allowed = {"id", "path", "format", "checksum", "shape", "dtype", "units", "metadata"}
        required = {"id", "path", "format", "checksum"}
        if not all(isinstance(key, str) for key in payload):
            _raise_invalid(
                "artifact reference keys must be strings",
                object_name="ArtifactRef",
                field="keys",
                expected=sorted(allowed),
                actual=[repr(key) for key in payload],
            )
        keys = cast(set[str], set(payload))
        unknown = sorted(keys - allowed)
        missing = sorted(required - keys)
        if unknown or missing:
            _raise_invalid(
                "artifact reference keys do not match the schema",
                object_name="ArtifactRef",
                field=unknown[0] if unknown else missing[0],
                expected=sorted(allowed),
                actual=sorted(keys),
            )

        raw_id = payload["id"]
        raw_path = payload["path"]
        raw_format = payload["format"]
        raw_checksum = payload["checksum"]
        for name, value in (
            ("id", raw_id),
            ("path", raw_path),
            ("format", raw_format),
            ("checksum", raw_checksum),
        ):
            if not isinstance(value, str):
                _raise_invalid(
                    f"artifact {name} must be a string",
                    object_name="ArtifactRef",
                    field=name,
                    expected="string",
                    actual=type(value),
                )

        raw_shape = payload.get("shape")
        if raw_shape is not None and not isinstance(raw_shape, list):
            _raise_invalid(
                "artifact shape must be a JSON array or null",
                object_name="ArtifactRef",
                field="shape",
                expected="array or null",
                actual=type(raw_shape),
            )
        if isinstance(raw_shape, list) and any(
            not isinstance(item, int) or isinstance(item, bool) for item in raw_shape
        ):
            _raise_invalid(
                "artifact shape entries must be integers",
                object_name="ArtifactRef",
                field="shape",
                expected="array of integers or null",
                actual=raw_shape,
            )
        shape = None if raw_shape is None else tuple(cast(list[int], raw_shape))

        raw_dtype = payload.get("dtype")
        if raw_dtype is not None and not isinstance(raw_dtype, str):
            _raise_invalid(
                "artifact dtype must be a string or null",
                object_name="ArtifactRef",
                field="dtype",
                expected="string or null",
                actual=type(raw_dtype),
            )
        raw_units = payload.get("units")
        units = _validate_units(raw_units, object_name="ArtifactRef")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            _raise_invalid(
                "artifact metadata must be an object",
                object_name="ArtifactRef",
                field="metadata",
                expected="object",
                actual=type(raw_metadata),
            )
        metadata = _validate_metadata(raw_metadata, object_name="ArtifactRef")
        return cls(
            id=cast(str, raw_id),
            path=cast(str, raw_path),
            format=cast(str, raw_format),
            checksum=cast(str, raw_checksum),
            shape=shape,
            dtype=raw_dtype,
            units=units,
            metadata=metadata,
        )


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _atomic_target(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_checksum(ref: ArtifactRef) -> bytes:
    path = Path(ref.path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _raise_invalid(
            "artifact file cannot be read for checksum verification",
            object_name="ArtifactRef",
            field="path",
            expected="readable artifact file",
            actual=str(exc),
        )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ref.checksum:
        _raise_invalid(
            "artifact checksum does not match its reference",
            object_name="ArtifactRef",
            field="checksum",
            expected=ref.checksum,
            actual=actual,
        )
    return payload


def _verified_payload(ref: ArtifactRef, expected_format: str, object_name: str) -> bytes:
    if not isinstance(ref, ArtifactRef):
        _raise_invalid(
            "artifact load requires an ArtifactRef",
            object_name=object_name,
            field="ref",
            expected="ArtifactRef",
            actual=type(ref),
        )
    payload = _verify_checksum(ref)
    if ref.format != expected_format:
        _raise_invalid(
            "artifact format does not match the requested codec",
            object_name=object_name,
            field="format",
            expected=expected_format,
            actual=ref.format,
        )
    return payload


def _prototype_ref(
    path: str | Path,
    *,
    artifact_id: str,
    artifact_format: str,
    units: str | dict[str, str] | None,
    metadata: Mapping[str, JsonValue] | None,
) -> ArtifactRef:
    return ArtifactRef(
        id=artifact_id,
        path=str(path),
        format=artifact_format,
        checksum="0" * 64,
        units=units,
        metadata={} if metadata is None else metadata,
    )


def save_json_artifact(
    path: str | Path,
    payload: JsonValue,
    *,
    artifact_id: str,
    units: str | dict[str, str] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ArtifactRef:
    """Atomically save deterministic JSON and return its verified reference.

    Args:
        path: output file location.
        payload: JSON-safe value to write.
        artifact_id: stable identifier recorded in the returned ref.
        units: unit string (or per-key mapping) recorded alongside.
        metadata: JSON-safe extras recorded alongside.
    """
    prototype = _prototype_ref(
        path,
        artifact_id=artifact_id,
        artifact_format="json",
        units=units,
        metadata=metadata,
    )
    if not _is_json_value(payload):
        _raise_invalid(
            "JSON artifact payload is not JSON-safe",
            object_name="save_json_artifact",
            field="payload",
            expected="JsonValue with finite numbers",
            actual=payload,
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    target = Path(prototype.path)
    with _atomic_target(target) as temporary:
        temporary.write_text(encoded, encoding="utf-8")
    return ArtifactRef(
        id=prototype.id,
        path=prototype.path,
        format="json",
        checksum=sha256_file(target),
        units=prototype.units,
        metadata=prototype.metadata,
    )


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_artifact(ref: ArtifactRef) -> JsonValue:
    """Load JSON only after its checksum has been verified."""
    encoded = _verified_payload(ref, "json", "load_json_artifact")
    try:
        payload: object = json.loads(
            encoded.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _raise_invalid(
            "JSON artifact could not be decoded",
            object_name="load_json_artifact",
            field="payload",
            expected="valid JSON",
            actual=str(exc),
        )
    if not _is_json_value(payload):
        _raise_invalid(
            "JSON artifact payload is not JSON-safe",
            object_name="load_json_artifact",
            field="payload",
            expected="JsonValue with finite numbers",
            actual=payload,
        )
    return _copy_json(payload)


def _validate_array_mapping(
    arrays: object,
    *,
    object_name: str,
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        _raise_invalid(
            "NPZ artifact must contain a non-empty string-to-array mapping",
            object_name=object_name,
            field="arrays",
            expected="non-empty mapping[str, numpy.ndarray]",
            actual=type(arrays),
        )
    validated: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name.strip():
            _raise_invalid(
                "NPZ array names must be non-empty strings",
                object_name=object_name,
                field="arrays",
                expected="non-empty string keys",
                actual=name,
            )
        if name in {"file", "allow_pickle"}:
            _raise_invalid(
                "NPZ array name conflicts with a NumPy codec argument",
                object_name=object_name,
                field=f"arrays.{name}",
                expected="array name other than 'file' or 'allow_pickle'",
                actual=name,
            )
        if not isinstance(value, np.ndarray):
            _raise_invalid(
                "NPZ values must be NumPy arrays",
                object_name=object_name,
                field=f"arrays.{name}",
                expected="numpy.ndarray",
                actual=type(value),
            )
        if value.dtype.hasobject:
            _raise_invalid(
                "NPZ object arrays would require unsafe pickle loading",
                object_name=object_name,
                field=f"arrays.{name}.dtype",
                expected="non-object NumPy dtype",
                actual=str(value.dtype),
            )
        if any(dimension <= 0 for dimension in value.shape):
            _raise_invalid(
                "NPZ array dimensions must be positive",
                object_name=object_name,
                field=f"arrays.{name}.shape",
                expected="positive dimensions",
                actual=value.shape,
            )
        validated[name] = value
    return validated


def _array_details(arrays: Mapping[str, np.ndarray]) -> dict[str, JsonValue]:
    return {
        name: {"shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
        for name in sorted(arrays)
    }


def _metadata_with_details(
    metadata: Mapping[str, JsonValue] | None,
    details: dict[str, JsonValue],
    *,
    object_name: str,
) -> dict[str, JsonValue]:
    copied = _validate_metadata({} if metadata is None else metadata, object_name=object_name)
    if "arrays" in copied:
        _raise_invalid(
            "artifact metadata reserves 'arrays' for per-key shape and dtype details",
            object_name=object_name,
            field="metadata.arrays",
            expected="field absent",
            actual=copied["arrays"],
        )
    copied["arrays"] = details
    return copied


def _write_npz(stream: BinaryIO, arrays: Mapping[str, np.ndarray]) -> None:
    """Call NumPy's variadic archive writer through its runtime signature."""
    writer = cast(Callable[..., None], np.savez_compressed)
    writer(stream, **arrays)


def save_npz_artifact(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    artifact_id: str,
    units: str | dict[str, str] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ArtifactRef:
    """Atomically save non-object arrays in a compressed NPZ archive.

    Args:
        path: output file location.
        arrays: ``{name: ndarray}`` mapping written as ``.npz``.
        artifact_id: stable identifier recorded in the returned ref.
        units: unit string (or per-key mapping) recorded alongside.
        metadata: JSON-safe extras recorded alongside.
    """
    validated = _validate_array_mapping(arrays, object_name="save_npz_artifact")
    details = _array_details(validated)
    ref_metadata = (
        _metadata_with_details(
            metadata, details, object_name="save_npz_artifact"
        )
        if len(validated) != 1
        else _validate_metadata(
            {} if metadata is None else metadata, object_name="save_npz_artifact"
        )
    )
    prototype = _prototype_ref(
        path,
        artifact_id=artifact_id,
        artifact_format="npz",
        units=units,
        metadata=ref_metadata,
    )
    target = Path(prototype.path)
    with _atomic_target(target) as temporary:
        with temporary.open("wb") as stream:
            _write_npz(stream, validated)
    only = next(iter(validated.values())) if len(validated) == 1 else None
    return ArtifactRef(
        id=prototype.id,
        path=prototype.path,
        format="npz",
        checksum=sha256_file(target),
        shape=None if only is None else tuple(only.shape),
        dtype=None if only is None else str(only.dtype),
        units=prototype.units,
        metadata=prototype.metadata,
    )


def load_npz_artifact(ref: ArtifactRef) -> dict[str, np.ndarray]:
    """Load an NPZ archive without pickle, after checksum verification."""
    encoded = _verified_payload(ref, "npz", "load_npz_artifact")
    try:
        archive = np.load(io.BytesIO(encoded), allow_pickle=False)
        if not isinstance(archive, NpzFile):
            _raise_invalid(
                "NPZ artifact is not an NPZ archive",
                object_name="load_npz_artifact",
                field="payload",
                expected="NPZ archive containing named arrays",
                actual=type(archive),
            )
        try:
            payload = {name: archive[name] for name in archive.files}
        finally:
            archive.close()
    except GeoBrainError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        _raise_invalid(
            "NPZ artifact could not be decoded safely",
            object_name="load_npz_artifact",
            field="payload",
            expected="NPZ archive containing non-object arrays",
            actual=str(exc),
        )
    return _validate_array_mapping(payload, object_name="load_npz_artifact")


def _validate_tensor_mapping(
    tensors: object,
    *,
    object_name: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or not tensors:
        _raise_invalid(
            "PT artifact must contain a non-empty string-to-tensor mapping",
            object_name=object_name,
            field="tensors",
            expected="non-empty mapping[str, torch.Tensor]",
            actual=type(tensors),
        )
    validated: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if not isinstance(name, str) or not name.strip():
            _raise_invalid(
                "PT tensor names must be non-empty strings",
                object_name=object_name,
                field="tensors",
                expected="non-empty string keys",
                actual=name,
            )
        if not isinstance(value, torch.Tensor):
            _raise_invalid(
                "PT artifact values must be tensors",
                object_name=object_name,
                field=f"tensors.{name}",
                expected="torch.Tensor",
                actual=type(value),
            )
        if any(dimension <= 0 for dimension in value.shape):
            _raise_invalid(
                "PT tensor dimensions must be positive",
                object_name=object_name,
                field=f"tensors.{name}.shape",
                expected="positive dimensions",
                actual=value.shape,
            )
        validated[name] = value
    return validated


def _tensor_details(tensors: Mapping[str, torch.Tensor]) -> dict[str, JsonValue]:
    return {
        name: {
            "shape": list(tensors[name].shape),
            "dtype": str(tensors[name].dtype).removeprefix("torch."),
        }
        for name in sorted(tensors)
    }


def save_tensor_artifact(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    artifact_id: str,
    units: str | dict[str, str] | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> ArtifactRef:
    """Atomically save a tensor-only PT dictionary.

    Args:
        path: output file location.
        tensors: ``{name: Tensor}`` mapping (saved via ``torch.save``).
        artifact_id: stable identifier recorded in the returned ref.
        units: unit string (or per-key mapping) recorded alongside.
        metadata: JSON-safe extras recorded alongside.
    """
    validated = _validate_tensor_mapping(tensors, object_name="save_tensor_artifact")
    details = _tensor_details(validated)
    ref_metadata = (
        _metadata_with_details(
            metadata, details, object_name="save_tensor_artifact"
        )
        if len(validated) != 1
        else _validate_metadata(
            {} if metadata is None else metadata, object_name="save_tensor_artifact"
        )
    )
    prototype = _prototype_ref(
        path,
        artifact_id=artifact_id,
        artifact_format="pt",
        units=units,
        metadata=ref_metadata,
    )
    target = Path(prototype.path)
    with _atomic_target(target) as temporary:
        torch.save(dict(validated), temporary)
    only = next(iter(validated.values())) if len(validated) == 1 else None
    return ArtifactRef(
        id=prototype.id,
        path=prototype.path,
        format="pt",
        checksum=sha256_file(target),
        shape=None if only is None else tuple(only.shape),
        dtype=(
            None
            if only is None
            else str(only.dtype).removeprefix("torch.")
        ),
        units=prototype.units,
        metadata=prototype.metadata,
    )


def load_tensor_artifact(ref: ArtifactRef) -> dict[str, torch.Tensor]:
    """Load a tensor-only PT dictionary safely, on CPU, after checksum verification."""
    path = io.BytesIO(_verified_payload(ref, "pt", "load_tensor_artifact"))
    try:
        payload: object = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        _raise_invalid(
            "PT artifact could not be decoded safely",
            object_name="load_tensor_artifact",
            field="payload",
            expected="weights-only PT payload",
            actual=str(exc),
        )
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        raise GeoBrainError(
            "PT artifact must contain a string-to-tensor dictionary",
            code=ErrorCode.ARTIFACT_INVALID,
            object_name="load_tensor_artifact",
            field="payload",
            expected="dict[str, torch.Tensor]",
            actual=type(payload),
        )
    return cast(dict[str, torch.Tensor], payload)


__all__ = [
    "ArtifactRef",
    "JsonValue",
    "load_json_artifact",
    "load_npz_artifact",
    "load_tensor_artifact",
    "save_json_artifact",
    "save_npz_artifact",
    "save_tensor_artifact",
    "sha256_file",
]
