"""UBC-GIF 3D tensor mesh / model file bindings (pure text, no extra deps).

Format (GRAV3D / MAG3D family):

- **Mesh file**: line 1 ``NE NN NZ`` (cell counts east / north / down);
  line 2 ``E0 N0 Z0`` = the **south-west-TOP** corner with ``Z0`` as
  elevation; lines 3-5 the per-axis cell widths (east, north, then z listed
  from the top DOWNWARD), each supporting the ``count*width`` repeat
  syntax; ``!`` starts a comment.
- **Model file**: one value per cell in a single column ordered z fastest
  (top -> down), then easting, then northing slowest. This matches
  the reference reader (Fortran reshape to ``(nz, nx, ny)``), the
  cross-validation oracle the io test suite pins this reader against.

Platform mapping: GeoBrain mesh axes are ``(z, x, y)`` with z as
depth-DOWN, so the UBC top-down z ordering aligns with the platform z
index as-is (no flip, unlike z-up codes) and the model column is exactly
the Fortran flatten of the platform ``(nz, nx, ny)`` array. Easting maps
to platform x, northing to y, and the top elevation ``Z0`` enters the
platform origin as ``-Z0`` (depth of the top face).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import os
from typing import NamedTuple, Sequence

import numpy as np
import torch

from geobrain.core.errors import GeoBrainError
from geobrain.mesh import TensorMesh


class UBCMesh3D(NamedTuple):
    """Parsed UBC-GIF 3D tensor mesh.

    Attributes:
        widths_east: ``(NE,)`` cell widths along easting [m].
        widths_north: ``(NN,)`` cell widths along northing [m].
        widths_z_down: ``(NZ,)`` cell widths from the TOP going down [m].
        origin_enu: ``(E0, N0, Z0)`` south-west-top corner; ``Z0`` is the
            top ELEVATION (up-positive), exactly as the file stores it.
    """

    widths_east: np.ndarray
    widths_north: np.ndarray
    widths_z_down: np.ndarray
    origin_enu: tuple[float, float, float]


def _expand_width_line(line: str, who: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for segment in line.split():
        if "*" in segment:
            count_text, width_text = segment.split("*", 1)
            chunks.append(np.full(int(count_text), float(width_text)))
        else:
            chunks.append(np.array([float(segment)]))
    if not chunks:
        raise GeoBrainError(
            "empty cell-width line in UBC mesh file",
            object_name=who,
            field="widths",
        )
    return np.concatenate(chunks)


def _content_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def read_ubc_mesh(path: str | os.PathLike) -> UBCMesh3D:
    """Read a UBC-GIF 3D tensor mesh file."""
    with open(path, encoding="utf-8") as handle:
        lines = _content_lines(handle.read())
    if len(lines) < 5:
        raise GeoBrainError(
            f"UBC mesh file needs 5 content lines, found {len(lines)}",
            object_name="read_ubc_mesh",
            field="path",
            actual=len(lines),
        )
    counts = [int(float(tok)) for tok in lines[0].split()]
    origin = tuple(float(tok) for tok in lines[1].split())
    if len(counts) != 3 or len(origin) != 3:
        raise GeoBrainError(
            "UBC mesh header must carry 3 counts and 3 origin coordinates",
            object_name="read_ubc_mesh",
            field="header",
            actual=(counts, origin),
        )
    widths = [
        _expand_width_line(lines[2 + axis], "read_ubc_mesh") for axis in range(3)
    ]
    for n_expected, axis_widths, name in zip(
        counts, widths, ("east", "north", "z")
    ):
        if len(axis_widths) != n_expected:
            raise GeoBrainError(
                f"UBC mesh {name} widths disagree with the header count",
                object_name="read_ubc_mesh",
                field=name,
                expected=n_expected,
                actual=len(axis_widths),
            )
    return UBCMesh3D(
        widths_east=widths[0],
        widths_north=widths[1],
        widths_z_down=widths[2],
        origin_enu=(origin[0], origin[1], origin[2]),
    )


def write_ubc_mesh(path: str | os.PathLike, mesh: UBCMesh3D) -> None:
    """Write a UBC-GIF 3D tensor mesh file (plain widths, no ``*`` runs).

    Args:
        path: output UBC-GIF mesh file.
        mesh: the :class:`UBCMesh3D` description to serialise.
    """
    e0, n0, z0 = mesh.origin_enu
    lines = [
        f"{len(mesh.widths_east)} {len(mesh.widths_north)} "
        f"{len(mesh.widths_z_down)}",
        f"{e0:.10g} {n0:.10g} {z0:.10g}",
        " ".join(f"{w:.10g}" for w in np.asarray(mesh.widths_east)),
        " ".join(f"{w:.10g}" for w in np.asarray(mesh.widths_north)),
        " ".join(f"{w:.10g}" for w in np.asarray(mesh.widths_z_down)),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def ubc_mesh_to_tensor_mesh(mesh: UBCMesh3D) -> TensorMesh:
    """Convert to the platform ``TensorMesh``: axes ``(z_down, x=east,
    y=north)``, origin z = ``-Z0`` (depth of the UBC top elevation)."""
    e0, n0, z0 = mesh.origin_enu
    widths = (mesh.widths_z_down, mesh.widths_east, mesh.widths_north)
    return TensorMesh(
        shape=tuple(len(w) for w in widths),
        cell_widths=tuple(
            torch.as_tensor(np.asarray(w), dtype=torch.float64) for w in widths
        ),
        origin=(-z0, e0, n0),
    )


def tensor_mesh_to_ubc_mesh(mesh: TensorMesh) -> UBCMesh3D:
    """Inverse of :func:`ubc_mesh_to_tensor_mesh` (3-D meshes only)."""
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "UBC export requires a 3D TensorMesh",
            object_name="tensor_mesh_to_ubc_mesh",
            field="mesh",
            expected=3,
            actual=mesh.n_dim,
        )
    wz, wx, wy = (np.asarray(w, dtype=float) for w in mesh.cell_widths)
    z0, x0, y0 = (float(v) for v in mesh.origin)
    return UBCMesh3D(
        widths_east=wx,
        widths_north=wy,
        widths_z_down=wz,
        origin_enu=(x0, y0, -z0),
    )


def _counts_of(mesh: UBCMesh3D) -> tuple[int, int, int]:
    return (
        len(mesh.widths_z_down),
        len(mesh.widths_east),
        len(mesh.widths_north),
    )


def read_ubc_model(path: str | os.PathLike, mesh: UBCMesh3D) -> np.ndarray:
    """Read a UBC-GIF model column into a platform-ordered ``(nz, nx, ny)``
    array (z index 0 at the top, easting as x, northing as y).

    Args:
        path: UBC-GIF model file to read.
        mesh: the :class:`UBCMesh3D` giving cell count and ordering.
    """
    with open(path, encoding="utf-8") as handle:
        content = _content_lines(handle.read())
    tokens = [token for line in content for token in line.split()]
    values = np.array([float(token) for token in tokens], dtype=float)
    nz, nx, ny = _counts_of(mesh)
    if values.size != nz * nx * ny:
        raise GeoBrainError(
            "UBC model size disagrees with the mesh cell count",
            object_name="read_ubc_model",
            field="path",
            expected=nz * nx * ny,
            actual=values.size,
        )
    return np.ascontiguousarray(values.reshape((nz, nx, ny), order="F"))


def write_ubc_model(
    path: str | os.PathLike, mesh: UBCMesh3D, model: Sequence | torch.Tensor | np.ndarray
) -> None:
    """Write a platform-ordered ``(nz, nx, ny)`` array as a UBC model column.

    Args:
        path: output UBC-GIF model file.
        mesh: the :class:`UBCMesh3D` giving cell count and ordering.
        model: per-cell values in the platform ``(nz, nx, ny)`` layout.
    """
    if isinstance(model, torch.Tensor):
        model = model.detach().cpu().numpy()
    array = np.asarray(model, dtype=float)
    nz, nx, ny = _counts_of(mesh)
    if array.shape != (nz, nx, ny):
        raise GeoBrainError(
            "model shape must be the platform (nz, nx, ny) for this mesh",
            object_name="write_ubc_model",
            field="model",
            expected=(nz, nx, ny),
            actual=tuple(array.shape),
        )
    column = array.reshape(-1, order="F")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(f"{v:.10g}" for v in column) + "\n")


__all__ = [
    "UBCMesh3D",
    "read_ubc_mesh",
    "write_ubc_mesh",
    "ubc_mesh_to_tensor_mesh",
    "tensor_mesh_to_ubc_mesh",
    "read_ubc_model",
    "write_ubc_model",
]
