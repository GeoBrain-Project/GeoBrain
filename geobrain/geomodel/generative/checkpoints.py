"""Schema-validated model cards and checksum-first checkpoint loading.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import torch

from ...core.errors import ErrorCode
from ..errors import GeomodelContractError

__all__ = ["ModelCard", "load_verified_state_dict"]

_REQUIRED_CARD_FIELDS = {
    "schema", "architecture", "class_vocabulary", "components",
    "provider_versions", "framework_versions", "preprocessing",
    "training_data", "intended_use", "limitations", "metrics", "citation",
}
_REQUIRED_TRAINING_DATA_FIELDS = {"identity", "licence", "redistribution"}


def _artifact_error(field: str, expected: object, actual: object) -> GeomodelContractError:
    return GeomodelContractError(
        "invalid Geomodel model-card or checkpoint artifact",
        object_name="ModelCard", field=field, expected=expected, actual=actual,
        code=ErrorCode.ARTIFACT_INVALID,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Verified manifest of a generative-model checkpoint set.

    Attributes:
        schema: card schema tag.
        payload: the validated card content; per-component records carry
            the checkpoint SHA-256 and tensor manifest that
            :func:`load_verified_state_dict` enforces.
    """

    schema: str
    payload: Mapping[str, object]

    @classmethod
    def load(cls, path: str | Path) -> "ModelCard":
        resolved = Path(path)
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _artifact_error("path", "readable strict JSON model card", str(resolved)) from exc
        if not isinstance(value, dict):
            raise _artifact_error("root", "JSON object", type(value).__name__)
        missing = sorted(_REQUIRED_CARD_FIELDS - set(value))
        if missing:
            raise _artifact_error("required", sorted(_REQUIRED_CARD_FIELDS), {"missing": missing})
        if value.get("schema") != "geobrain.geomodel-model-card/1.0":
            raise _artifact_error("schema", "geobrain.geomodel-model-card/1.0", value.get("schema"))
        training = value.get("training_data")
        if not isinstance(training, dict) or _REQUIRED_TRAINING_DATA_FIELDS - set(training):
            raise _artifact_error("training_data", sorted(_REQUIRED_TRAINING_DATA_FIELDS), training)
        components = value.get("components")
        if not isinstance(components, dict) or not components:
            raise _artifact_error("components", "non-empty object", components)
        for name, component in components.items():
            if not isinstance(name, str) or not isinstance(component, dict):
                raise _artifact_error("components", "named component objects", components)
            checksum = component.get("sha256")
            if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise _artifact_error(f"components.{name}.sha256", "lowercase SHA-256", checksum)
            tensors = component.get("tensors")
            if not isinstance(tensors, dict):
                raise _artifact_error(f"components.{name}.tensors", "tensor manifest object", tensors)
        canonical = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
        return cls(cast(str, canonical["schema"]), MappingProxyType(canonical))

    @property
    def components(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.payload["components"])


def load_verified_state_dict(
    checkpoint_path: str | Path,
    card: ModelCard,
    component: str,
    map_location: str | torch.device,
) -> Mapping[str, torch.Tensor]:
    """Load a checkpoint state dict only after verifying it against its card.

    The checkpoint file's SHA-256 must match the digest the
    :class:`ModelCard` records for ``component``; loading is
    ``weights_only`` (no pickled code execution) and the result must be a
    ``{name: Tensor}`` mapping that also satisfies the card's per-tensor
    manifest. Any mismatch raises instead of returning partial state.

    Args:
        checkpoint_path: ``.pt`` file to load.
        card: the model card carrying the expected component records.
        component: which card component this checkpoint claims to be.
        map_location: device mapping forwarded to :func:`torch.load`.

    Returns:
        The verified ``{parameter name: tensor}`` state dict.

    Raises:
        GeomodelContractError: on a digest, type, or manifest mismatch, or
            an unreadable file, with the offending field in the payload.
    """
    if not isinstance(card, ModelCard):
        raise _artifact_error("card", "ModelCard", type(card).__name__)
    component_record = card.components.get(component)
    if not isinstance(component_record, Mapping):
        raise _artifact_error("component", sorted(card.components), component)
    path = Path(checkpoint_path)
    expected_sha = component_record.get("sha256")
    try:
        actual_sha = _sha256(path)
    except OSError as exc:
        raise _artifact_error("checkpoint_path", "readable checkpoint", str(path)) from exc
    if actual_sha != expected_sha:
        raise _artifact_error("sha256", expected_sha, actual_sha)
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise _artifact_error("checkpoint", "safe weights-only state dict", type(exc).__name__) from exc
    if not isinstance(loaded, Mapping) or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in loaded.items()):
        raise _artifact_error("checkpoint", "mapping[str, Tensor]", type(loaded).__name__)
    manifest = component_record.get("tensors")
    assert isinstance(manifest, Mapping)
    if set(loaded) != set(manifest):
        raise _artifact_error("checkpoint.keys", sorted(manifest), sorted(loaded))
    for name, tensor in loaded.items():
        record = manifest[name]
        if not isinstance(record, Mapping):
            raise _artifact_error(f"tensors.{name}", "shape/dtype object", record)
        expected_shape = tuple(record.get("shape", ()))
        expected_dtype = record.get("dtype")
        if tuple(tensor.shape) != expected_shape or str(tensor.dtype).removeprefix("torch.") != expected_dtype:
            raise _artifact_error(
                f"tensors.{name}",
                {"shape": list(expected_shape), "dtype": expected_dtype},
                {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")},
            )
    return MappingProxyType(dict(loaded))
