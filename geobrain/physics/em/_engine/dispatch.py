"""Value fingerprints and cache-key builders for inductive dispatch.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import struct
from typing import Protocol, cast

import torch

from geobrain.core import GeoBrainError
from geobrain.mesh import TensorMesh
from geobrain.physics.em.errors import EMContractError
from geobrain.physics.em.numerics.mesh_path import resolve_inductive_mesh_path

from .contracts import AssemblyCacheKey, exact_float_token


class _HashWriter(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _frame(hasher: _HashWriter, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    hasher.update(struct.pack("<Q", len(label_bytes)))
    hasher.update(label_bytes)
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}|{tuple(value.shape)}".encode("ascii")
    raw = cast(bytes, value.view(torch.uint8).numpy().tobytes(order="C"))
    return struct.pack("<Q", len(header)) + header + raw


def material_fingerprint(material: torch.Tensor) -> str:
    """SHA-256 over exact dtype, shape, and C-order tensor bytes."""
    if not isinstance(material, torch.Tensor) or material.layout != torch.strided:
        raise EMContractError(
            "material fingerprint requires a dense tensor",
            details={"received_type": type(material).__qualname__},
            object_name="material_fingerprint",
            field="material",
            expected="dense torch.Tensor",
            actual=type(material).__qualname__,
        )
    hasher = hashlib.sha256()
    _frame(hasher, "material", _tensor_bytes(material))
    return hasher.hexdigest()


def _record_tensors(record: object) -> tuple[tuple[str, torch.Tensor], ...]:
    if not is_dataclass(record):
        return ()
    result: list[tuple[str, torch.Tensor]] = []
    for item in fields(record):
        value = getattr(record, item.name)
        if isinstance(value, torch.Tensor):
            result.append((item.name, value))
    return tuple(result)


def mesh_fingerprint(mesh: object) -> str:
    """Stable SHA-256 covering inductive mesh topology and exact geometry."""
    n_dim = getattr(mesh, "n_dim", None)
    n_cells = getattr(mesh, "n_cells", None)
    if type(n_dim) is not int or n_dim <= 0 or type(n_cells) is not int or n_cells <= 0:
        raise EMContractError(
            "mesh geometry fingerprint requires positive dimension and cell count",
            details={"received_type": type(mesh).__qualname__},
            object_name="mesh_fingerprint",
            field="mesh",
            expected="mesh geometry with positive n_dim and n_cells",
            actual=type(mesh).__qualname__,
        )
    hasher = hashlib.sha256()
    _frame(hasher, "type", type(mesh).__qualname__.encode("utf-8"))
    _frame(hasher, "counts", repr((n_dim, n_cells)).encode("ascii"))

    if isinstance(mesh, TensorMesh):
        _frame(hasher, "shape", repr(mesh.shape).encode("ascii"))
        _frame(hasher, "origin", b"".join(struct.pack("<d", x) for x in mesh.origin))
        for axis, widths in enumerate(mesh.cell_widths):
            _frame(hasher, f"cell_widths[{axis}]", _tensor_bytes(widths))
        return hasher.hexdigest()

    for method_name in ("cell_centers", "cell_volumes"):
        method = getattr(mesh, method_name, None)
        if not callable(method):
            raise EMContractError(
                "mesh geometry fingerprint requires cell geometry",
                details={"method": method_name},
                object_name="mesh_fingerprint",
                field="mesh",
                expected=f"callable {method_name}()",
                actual=type(method).__qualname__,
            )
        value = method()
        if not isinstance(value, torch.Tensor):
            raise EMContractError(
                "mesh geometry method must return a tensor",
                details={"method": method_name},
                object_name="mesh_fingerprint",
                field=method_name,
                expected="torch.Tensor",
                actual=type(value).__qualname__,
            )
        _frame(hasher, method_name, _tensor_bytes(value))

    for method_name in ("node_coords", "cell_nodes"):
        method = getattr(mesh, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
        except GeoBrainError:  # optional capability method may reject this instance
            continue
        if isinstance(value, torch.Tensor):
            _frame(hasher, method_name, _tensor_bytes(value))

    for method_name in ("face_neighbors", "boundary_faces", "edge_records"):
        method = getattr(mesh, method_name, None)
        if not callable(method):
            continue
        try:
            record = method()
        except GeoBrainError:  # optional capability method may reject this instance
            continue
        for name, value in _record_tensors(record):
            _frame(hasher, f"{method_name}.{name}", _tensor_bytes(value))

    return hasher.hexdigest()


def build_assembly_cache_key(
    *,
    formulation_version: str,
    mesh: object,
    material: torch.Tensor,
    boundary: str,
    sample_value: float,
    backend: str,
    requires_gradient: bool,
) -> AssemblyCacheKey:
    """Build a complete matrix key from exact runtime values."""
    return AssemblyCacheKey(
        formulation_version=formulation_version,
        mesh_fingerprint=mesh_fingerprint(mesh),
        material_version=material_fingerprint(material),
        boundary=boundary,
        sample_value=exact_float_token(sample_value),
        dtype=str(material.dtype),
        device=str(material.device),
        backend=backend,
        requires_gradient=requires_gradient,
    )


__all__ = [
    "build_assembly_cache_key",
    "material_fingerprint",
    "mesh_fingerprint",
    "resolve_inductive_mesh_path",
]
