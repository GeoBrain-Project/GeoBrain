"""
Declarative, versioned, exact-round-trip serialization for the core mesh types.

Every core mesh (:class:`~geobrain.mesh.tensor.TensorMesh`,
:class:`~geobrain.mesh.octree.OctreeMesh`,
:class:`~geobrain.mesh.unstructured.UnstructuredMesh`,
:class:`~geobrain.mesh.cylindrical.CylindricalMesh`) grows an instance
method ``to_dict() -> dict`` producing a JSON-native payload, and this module
provides the inverse :func:`mesh_from_dict` factory that dispatches on the dict's
``"type"`` tag to rebuild the mesh. The round-trip is *exact*::

    mesh_from_dict(json.loads(json.dumps(m.to_dict())))  ==  m   # geometry-equal

This is **purely additive**; it reads the existing public/SoA geometry and
feeds it back through the canonical constructors, so no existing behaviour,
geometry, or the ``(nz, nx, ny)`` axis convention is touched.

Every payload carries a stable ``"type"`` string and an integer ``"version"``
(:data:`MESH_SCHEMA_VERSION`); the factory rejects an unknown type or an
unsupported version with a :class:`GeoBrainError` rather than mis-parsing it.

Tensor encoding
    Each tensor becomes a small JSON-native envelope ``{"dtype", "shape",
    "data"}`` where ``data`` is ``tensor.tolist()`` (plain nested Python
    int/float/bool lists). The dtype is recorded so float64 vs int64 (``long``)
    is restored exactly, and the shape is recorded so an empty or multi-column
    tensor, e.g. a ``(0, n_dim)`` boundary-normal on a mesh with no boundary,
    round-trips to the same shape (``torch.tensor([])`` alone collapses to
    ``(0,)``). float64 survives ``tolist`` → ``json`` → ``tolist`` bit-exactly
    (a Python ``float`` is an IEEE double and its ``repr`` round-trips), so the
    reconstruction is byte-for-byte identical geometry.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import fields as _dc_fields
from typing import Any

import torch

from ..core.errors import GeoBrainError
from .capabilities import BoundaryRecords, FaceRecords
from .cylindrical import CylindricalMesh
from .octree import OctreeMesh
from .tensor import TensorMesh
from .unstructured import UnstructuredMesh

# Schema version stamped into every mesh dict. Bump ONLY on an incompatible
# schema change; the factory rejects any other version so an old reader never
# silently mis-parses a newer payload (and vice-versa).
MESH_SCHEMA_VERSION = 1

# torch dtype <-> stable string name. Mesh geometry is canonicalized to CPU
# float64 (coordinates / widths / areas / normals / centroids) or torch.long
# (cell / level / node indices); those two are all that actually appear, but
# float32 / bool are mapped too for defensiveness. Note torch.long IS
# torch.int64, so a single "int64" entry covers both.
_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.int64: "int64",
    torch.bool: "bool",
}
_NAME_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_NAME.items()}

# Tensor-field names of the FaceRecords / BoundaryRecords SoAs: derived from
# the dataclasses so the schema tracks them: a field added to a record shows up
# here (and in the payloads) without touching this module.
_FACE_FIELDS = tuple(f.name for f in _dc_fields(FaceRecords))
_BOUNDARY_FIELDS = tuple(f.name for f in _dc_fields(BoundaryRecords))


# ----------------------------------------------------------------------
# tensor <-> JSON-native envelope
# ----------------------------------------------------------------------
def _ser_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a CPU tensor to a JSON-native ``{dtype, shape, data}`` envelope."""
    dtype = t.dtype
    if dtype not in _DTYPE_TO_NAME:
        raise GeoBrainError(
            "cannot serialize tensor dtype",
            object_name="mesh.serialization", field="dtype",
            expected=f"one of {sorted(_DTYPE_TO_NAME.values())}",
            actual=str(dtype),
        )
    return {
        "dtype": _DTYPE_TO_NAME[dtype],
        "shape": list(t.shape),
        "data": t.detach().cpu().tolist(),
    }


def _de_tensor(d: Any) -> torch.Tensor:
    """Inverse of :func:`_ser_tensor`: rebuild the exact CPU tensor."""
    if not isinstance(d, dict) or "dtype" not in d or "data" not in d:
        raise GeoBrainError(
            "malformed tensor envelope",
            object_name="mesh.serialization", field="tensor",
            expected="{dtype, shape, data} dict",
            actual=type(d).__name__,
        )
    name = d["dtype"]
    if name not in _NAME_TO_DTYPE:
        raise GeoBrainError(
            "unknown tensor dtype in payload",
            object_name="mesh.serialization", field="dtype",
            expected=f"one of {sorted(_NAME_TO_DTYPE)}", actual=name,
        )
    t = torch.tensor(d["data"], dtype=_NAME_TO_DTYPE[name])
    # Reshape to the recorded shape so an empty (0, n_dim) tensor is restored
    # with its columns intact (bare torch.tensor([]) would give (0,)).
    if "shape" in d:
        t = t.reshape([int(s) for s in d["shape"]])
    return t


def _ser_face_records(fr: Any) -> dict[str, Any]:
    return {name: _ser_tensor(getattr(fr, name)) for name in _FACE_FIELDS}


def _de_face_records(d: Any):  # type: ignore[no-untyped-def]

    return FaceRecords(**{name: _de_tensor(d[name]) for name in _FACE_FIELDS})


def _ser_boundary_records(br: Any) -> dict[str, Any]:
    return {name: _ser_tensor(getattr(br, name)) for name in _BOUNDARY_FIELDS}


def _de_boundary_records(d: Any):  # type: ignore[no-untyped-def]

    return BoundaryRecords(**{name: _de_tensor(d[name]) for name in _BOUNDARY_FIELDS})


# ----------------------------------------------------------------------
# per-type reconstruction (dispatched by mesh_from_dict)
# ----------------------------------------------------------------------
def _check_version(d: dict[str, Any]) -> None:
    version = d.get("version")
    if version != MESH_SCHEMA_VERSION:
        raise GeoBrainError(
            "unsupported mesh schema version",
            object_name="mesh_from_dict", field="version",
            expected=MESH_SCHEMA_VERSION, actual=version,
        )


def _tensor_from_dict(d: dict[str, Any]):  # type: ignore[no-untyped-def]

    _check_version(d)
    shape = tuple(int(s) for s in d["shape"])
    cell_widths = [_de_tensor(w) for w in d["cell_widths"]]
    # ``origin`` was added after the initial schema; a pre-origin payload has no
    # "origin" key, so default it to a zero origin (the mesh's legacy geometry),
    # old dicts still round-trip. ``None`` -> zeros in the constructor.
    origin = d.get("origin")
    # A uniform mesh MUST be rebuilt via ``spacing=`` (not ``cell_widths=``):
    # TensorMesh.__eq__ compares the ``is_uniform`` flag, and a mesh rebuilt from
    # cell_widths would be flagged non-uniform and compare unequal to the
    # original. The constant per-axis spacing is exactly cell_widths[axis][0].
    if bool(d.get("uniform", False)):
        spacing = [float(w[0]) for w in cell_widths]
        return TensorMesh(shape, spacing=spacing, origin=origin)
    return TensorMesh(shape, cell_widths=cell_widths, origin=origin)


def _octree_from_dict(d: dict[str, Any]):  # type: ignore[no-untyped-def]

    _check_version(d)
    return OctreeMesh(
        _de_tensor(d["centers"]),
        _de_tensor(d["half_widths"]),
        n_dim=int(d["n_dim"]),
        levels=_de_tensor(d["levels"]),
    )


def _unstructured_from_dict(d: dict[str, Any]):  # type: ignore[no-untyped-def]

    _check_version(d)
    boundary = d.get("boundary_faces")
    node_coords = d.get("node_coords")
    cell_nodes = d.get("cell_nodes")
    cell_markers = d.get("cell_markers")
    boundary_markers = d.get("boundary_markers")
    return UnstructuredMesh(
        _de_tensor(d["cell_centers"]),
        _de_tensor(d["cell_volumes"]),
        _de_face_records(d["faces"]),
        boundary_faces=(_de_boundary_records(boundary) if boundary is not None else None),
        node_coords=(_de_tensor(node_coords) if node_coords is not None else None),
        cell_nodes=(_de_tensor(cell_nodes) if cell_nodes is not None else None),
        cell_markers=(_de_tensor(cell_markers) if cell_markers is not None else None),
        boundary_markers=(
            _de_tensor(boundary_markers) if boundary_markers is not None else None
        ),
    )


def _cylindrical_from_dict(d: dict[str, Any]):  # type: ignore[no-untyped-def]

    _check_version(d)
    return CylindricalMesh(
        tuple(int(s) for s in d["shape"]),
        cell_widths=(_de_tensor(d["wz"]), _de_tensor(d["wr"])),
        origin=(float(d["z0"]), 0.0),
    )


_DISPATCH = {
    "TensorMesh": _tensor_from_dict,
    "OctreeMesh": _octree_from_dict,
    "UnstructuredMesh": _unstructured_from_dict,
    "CylindricalMesh": _cylindrical_from_dict,
}


def mesh_from_dict(d: dict[str, Any]):  # type: ignore[no-untyped-def]
    """Reconstruct a mesh from a :meth:`Mesh.to_dict` payload.

    Dispatches on the payload's ``"type"`` tag to the matching mesh constructor,
    reversing :meth:`~geobrain.mesh.tensor.TensorMesh.to_dict` /
    :meth:`~geobrain.mesh.octree.OctreeMesh.to_dict` /
    :meth:`~geobrain.mesh.unstructured.UnstructuredMesh.to_dict` /
    :meth:`~geobrain.mesh.cylindrical.CylindricalMesh.to_dict`. The dict
    must be exactly what one of those produced (optionally after a
    ``json.dumps`` / ``json.loads`` round-trip); the result is geometry-equal to
    the original mesh (``==`` for TensorMesh; identical ``cell_centers`` /
    ``cell_volumes`` / connectivity records for the others).

    Args:
        d: A mesh payload with a ``"type"`` tag and a ``"version"`` int.

    Returns:
        The reconstructed :class:`~geobrain.mesh.base.Mesh`.

    Raises:
        GeoBrainError: if ``d`` is not a dict, carries an unknown ``"type"``, or
            an unsupported ``"version"``.
    """
    if not isinstance(d, dict):
        raise GeoBrainError(
            "mesh_from_dict expects a dict",
            object_name="mesh_from_dict", field="d",
            expected="dict", actual=type(d).__name__,
        )
    mesh_type = d.get("type")
    if mesh_type not in _DISPATCH:
        raise GeoBrainError(
            "unknown mesh type tag",
            object_name="mesh_from_dict", field="type",
            expected=f"one of {sorted(_DISPATCH)}", actual=mesh_type,
        )
    return _DISPATCH[mesh_type](d)


__all__ = ["mesh_from_dict", "MESH_SCHEMA_VERSION"]
