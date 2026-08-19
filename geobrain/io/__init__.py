"""
File-format adapters for seismic / well / HDF5 data.

Each format binding lives in a sibling module (``segy``, ``hdf5``, ``las``,
…) and depends on an **optional third-party library** (segyio, h5py, lasio).
Importing ``geobrain.io`` itself does **not** import any of those, the
``__getattr__`` hook below only resolves a function when it is actually
called.

    >>> from geobrain.io import read_segy           # OK even without segyio
    >>> traces, dt, meta = read_segy("shot.sgy")    # imports segyio here

If the dependency is missing, the user gets a helpful error pointing at
``pip install 'geobrain[io]'``.

Conventions:
    - Readers return ``(numpy.ndarray, *metadata)``: never tensors.
      Callers convert via ``torch.as_tensor`` once they know the dtype.
    - Writers accept ``numpy.ndarray`` (or anything ``numpy.asarray`` can
      handle). They detach + cpu + numpy on tensor inputs.

Module layout::

    io/
    ├── segy.py / hdf5.py / las.py / vtk.py / edi.py / ubc.py
    │                   # One optional-dependency format binding per module
    │                   #   (segyio, h5py, lasio, pyvista+meshio, (,) )
    ├── artifacts.py    # JSON artifact save/load + ArtifactRef (no optional deps)
    ├── dataset.py      # GeoDataset / SimpleTensorDataset torch containers
    └── _common.py      # Private shared reader/writer plumbing

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import importlib
from typing import Any

from geobrain.io.artifacts import (
    ArtifactRef,
    load_json_artifact,
    load_npz_artifact,
    load_tensor_artifact,
    save_json_artifact,
    save_npz_artifact,
    save_tensor_artifact,
)

# NamedTuple return types: imported eagerly so callers can type-check
# ``isinstance(result, SegYData)`` without triggering the optional deps.
from geobrain.io.segy import SegYData
from geobrain.io.las import LASData
from geobrain.io.hdf5 import HDF5Data

# Pure-text industry formats: no optional dependency, imported eagerly.
from geobrain.io.edi import (
    EDIData,
    apparent_resistivity_ohm_m,
    impedance_phase_rad,
    read_edi,
)
from geobrain.io.ubc import (
    UBCMesh3D,
    read_ubc_mesh,
    read_ubc_model,
    tensor_mesh_to_ubc_mesh,
    ubc_mesh_to_tensor_mesh,
    write_ubc_mesh,
    write_ubc_model,
)

# Plain-tensor torch Datasets. (The composable property-transform pipeline
# lives in ``geobrain.core.transforms``: per-channel property prep is a
# preprocessing concern, not an IO one.)
from geobrain.io.dataset import GeoDataset, SimpleTensorDataset


# Symbol → (submodule, attribute, optional-dependency-name).
_LAZY_MAP: dict[str, tuple[str, str, str]] = {
    "read_segy":  ("geobrain.io.segy",  "read_segy",  "segyio"),
    "write_segy": ("geobrain.io.segy",  "write_segy", "segyio"),
    "read_hdf5":  ("geobrain.io.hdf5",  "read_hdf5",  "h5py"),
    "write_hdf5": ("geobrain.io.hdf5",  "write_hdf5", "h5py"),
    "read_las":   ("geobrain.io.las",   "read_las",   "lasio"),
    "mesh_to_pyvista":        ("geobrain.io.vtk", "mesh_to_pyvista",        "pyvista"),
    "write_tensormesh_vtk":    ("geobrain.io.vtk", "write_tensormesh_vtk",    "pyvista"),
    "write_points_vtk":        ("geobrain.io.vtk", "write_points_vtk",        "pyvista"),
    "write_octree_points_vtk": ("geobrain.io.vtk", "write_octree_points_vtk", "pyvista"),
    "write_unstructured_vtk": ("geobrain.io.vtk", "write_unstructured_vtk", "meshio"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_MAP:
        raise AttributeError(f"module 'geobrain.io' has no attribute {name!r}")
    module_path, attr, dep = _LAZY_MAP[name]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the optional dependency '{dep}'. "
            f"Install with: pip install '{dep}'   (or  pip install geobrain[io])"
        ) from exc
    return getattr(mod, attr)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = list(_LAZY_MAP) + [
    "SegYData",
    "LASData",
    "HDF5Data",
    "EDIData",
    "read_edi",
    "apparent_resistivity_ohm_m",
    "impedance_phase_rad",
    "UBCMesh3D",
    "read_ubc_mesh",
    "write_ubc_mesh",
    "read_ubc_model",
    "write_ubc_model",
    "ubc_mesh_to_tensor_mesh",
    "tensor_mesh_to_ubc_mesh",
    "SimpleTensorDataset",
    "GeoDataset",
    "ArtifactRef",
    "load_json_artifact",
    "load_npz_artifact",
    "load_tensor_artifact",
    "save_json_artifact",
    "save_npz_artifact",
    "save_tensor_artifact",
]
