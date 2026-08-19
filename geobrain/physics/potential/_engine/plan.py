"""Immutable value-fingerprinted execution plans for Potential kernels.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Protocol

import torch

from geobrain.core import ErrorCode
from geobrain.mesh.capabilities import PrismGeometryMesh

from ..errors import PotentialCapabilityError, PotentialContractError


_FINGERPRINT_SCHEMA = b"geobrain.potential.prism-plan/1.0\x00"
_GRAVITY_COMPONENTS = frozenset(
    {
        "gx",
        "gy",
        "gz",
        "gxx",
        "gxy",
        "gxz",
        "gyy",
        "gyz",
        "gzz",
    }
)
_MAGNETIC_COMPONENTS = frozenset(
    {
        "tmi",
        "bx",
        "by",
        "bz",
        "bxx",
        "bxy",
        "bxz",
        "byy",
        "byz",
        "bzz",
    }
)
_SUPPORTED_COMPONENTS = _GRAVITY_COMPONENTS | _MAGNETIC_COMPONENTS
_SUPPORTED_DTYPES = frozenset({torch.float32, torch.float64})
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda"})


class _Digest(Protocol):
    def update(self, payload: bytes, /) -> None: ...


def prism_mesh_cell_bounds_m(mesh) -> torch.Tensor:
    """Convert one 3-D prism-capable mesh to Potential prism coordinates.

    The engine consumes exactly the :class:`PrismGeometryMesh` contract,
    per-cell axis-aligned boxes via ``cell_bounds()``, so any declaring
    mesh drives it: the core ``TensorMesh`` and the graded ``OctreeMesh``
    (whose leaves ARE prisms) alike. Core mesh rows are ordered
    ``(z_lo, z_hi, x_lo, x_hi, y_lo, y_hi)`` with depth positive down.
    Potential geometry is ordered ``(x_min, x_max, y_min, y_max, z_min,
    z_max)`` with elevation positive up.  The bridge is deliberately
    structural: coordinate magnitudes never influence the permutation or
    vertical sign.
    """
    declared = getattr(mesh, "mesh_capabilities", frozenset())
    if PrismGeometryMesh not in set(declared):
        raise PotentialContractError(
            "Potential prism geometry requires a prism-capable mesh.",
            object_name="prism_mesh_cell_bounds_m",
            field="mesh",
            expected="mesh declaring PrismGeometryMesh "
            "(core TensorMesh or OctreeMesh)",
            actual=type(mesh).__name__,
            hint="Provide a prism-capable mesh from ForwardContext.",
        )
    if mesh.n_dim != 3:
        raise PotentialContractError(
            "Potential prism geometry requires a 3-D mesh.",
            object_name="prism_mesh_cell_bounds_m",
            field="mesh",
            expected="mesh.n_dim == 3",
            actual=mesh.n_dim,
            hint="Use Gravity2D for a 2-D mesh or supply a 3-D prism mesh.",
        )
    bounds = mesh.cell_bounds()
    return torch.stack(
        (
            bounds[:, 2],
            bounds[:, 3],
            bounds[:, 4],
            bounds[:, 5],
            -bounds[:, 1],
            -bounds[:, 0],
        ),
        dim=1,
    ).contiguous()


def _canonical_device(device: object) -> torch.device:
    if not isinstance(device, torch.device):
        raise PotentialContractError(
            "Plan device must be a torch.device.",
            object_name="PrismKernelPlan",
            field="device",
            expected="torch.device('cpu') or an available CUDA device",
            actual=device,
            hint="Construct and pass an explicit torch.device matching both geometry tensors.",
        )
    if device.type not in _SUPPORTED_DEVICE_TYPES:
        raise PotentialCapabilityError(
            "Plan device type is unsupported.",
            object_name="PrismKernelPlan",
            field="device",
            expected=["cpu", "cuda"],
            actual=device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Use CPU or an available CUDA device for Potential execution.",
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise PotentialCapabilityError(
                "CUDA plan construction was requested but CUDA is unavailable.",
                object_name="PrismKernelPlan",
                field="device",
                expected="an available CUDA runtime",
                actual=device,
                code=ErrorCode.DEVICE_UNAVAILABLE,
                hint="Use device=torch.device('cpu') or enable a CUDA runtime.",
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        return torch.device("cuda", index)
    if device.index is not None:
        raise PotentialContractError(
            "CPU plan device cannot carry an index.",
            object_name="PrismKernelPlan",
            field="device",
            expected="torch.device('cpu')",
            actual=device,
            hint="Use the canonical unindexed CPU device.",
        )
    return torch.device("cpu")


def _validated_dtype(dtype: object) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise PotentialContractError(
            "Plan dtype must be a torch.dtype.",
            object_name="PrismKernelPlan",
            field="dtype",
            expected=["torch.float32", "torch.float64"],
            actual=dtype,
            hint="Pass dtype=torch.float32 or dtype=torch.float64 explicitly.",
        )
    if dtype not in _SUPPORTED_DTYPES:
        raise PotentialCapabilityError(
            "Plan dtype is unsupported.",
            object_name="PrismKernelPlan",
            field="dtype",
            expected=["torch.float32", "torch.float64"],
            actual=dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Use float32 or float64 consistently for plan geometry and execution.",
        )
    return dtype


def _validated_tensor(
    value: object,
    *,
    field_name: str,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PotentialContractError(
            "Plan geometry must be a torch.Tensor.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected=f"non-empty (n, {width}) strided {dtype} tensor on {device}",
            actual=value,
            hint=f"Provide packed {field_name} with exactly {width} columns.",
        )
    if value.layout != torch.strided:
        raise PotentialCapabilityError(
            "Plan geometry requires a strided tensor layout.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected="torch.strided",
            actual=value.layout,
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
            hint="Materialize geometry in a strided tensor before building the plan.",
        )
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != width:
        raise PotentialContractError(
            "Plan geometry has the wrong shape.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected=["n>0", width],
            actual=list(value.shape),
            code=ErrorCode.SHAPE_MISMATCH,
            hint=f"Provide a non-empty tensor shaped (n, {width}).",
        )
    if value.dtype != dtype:
        raise PotentialContractError(
            "Plan geometry dtype does not match the requested dtype.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected=dtype,
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Construct both geometry tensors in the exact requested dtype.",
        )
    if value.device != device:
        raise PotentialContractError(
            "Plan geometry device does not match the requested device.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected=device,
            actual=value.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Construct both geometry tensors on the exact requested device.",
        )
    if value.requires_grad:
        raise PotentialContractError(
            "Plan geometry cannot own gradient history.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected="requires_grad=False",
            actual=True,
            hint="Pass validated non-gradient canonical geometry; no silent detach is performed.",
        )
    if not bool(torch.isfinite(value).all()):
        raise PotentialContractError(
            "Plan geometry must be finite.",
            object_name="PrismKernelPlan",
            field=field_name,
            expected="all finite values",
            actual="contains non-finite values",
            hint="Replace NaN or infinite geometry values before building the plan.",
        )
    return value.detach().clone(memory_format=torch.contiguous_format)


def _validated_components(components: object) -> tuple[str, ...]:
    if not isinstance(components, tuple) or not components:
        raise PotentialContractError(
            "Plan components must be a non-empty tuple.",
            object_name="PrismKernelPlan",
            field="components",
            expected="non-empty ordered tuple of unique supported component names",
            actual=components,
            hint="Pass component names in the exact requested execution order.",
        )
    if any(not isinstance(component, str) or not component for component in components):
        raise PotentialContractError(
            "Plan component names must be non-empty strings.",
            object_name="PrismKernelPlan",
            field="components",
            expected="non-empty string names",
            actual=components,
            hint="Replace blank or non-string component entries with supported names.",
        )
    if len(set(components)) != len(components):
        raise PotentialContractError(
            "Plan components must be unique.",
            object_name="PrismKernelPlan",
            field="components",
            expected="unique ordered component names",
            actual=components,
            hint="Remove duplicate components while preserving the requested order.",
        )
    component_set = frozenset(components)
    unsupported = tuple(
        component for component in components if component not in _SUPPORTED_COMPONENTS
    )
    if unsupported:
        raise PotentialCapabilityError(
            "Plan contains unsupported public components.",
            object_name="PrismKernelPlan",
            field="components",
            expected=sorted(_SUPPORTED_COMPONENTS),
            actual=unsupported,
            hint="Select only supported public gravity or magnetic component names.",
        )
    if not (component_set <= _GRAVITY_COMPONENTS or component_set <= _MAGNETIC_COMPONENTS):
        raise PotentialCapabilityError(
            "Plan components cannot mix gravity and magnetic families.",
            object_name="PrismKernelPlan",
            field="components",
            expected={
                "gravity": sorted(_GRAVITY_COMPONENTS),
                "magnetic": sorted(_MAGNETIC_COMPONENTS),
            },
            actual=components,
            hint="Select one non-empty component tuple entirely from one operator family.",
        )
    return components


def _validate_bounds(bounds: torch.Tensor) -> None:
    for lower_column, upper_column in ((0, 1), (2, 3), (4, 5)):
        if not bool((bounds[:, lower_column] < bounds[:, upper_column]).all()):
            raise PotentialContractError(
                "Every prism lower bound must be strictly below its upper bound.",
                object_name="PrismKernelPlan",
                field="cell_bounds_m",
                expected="xmin<xmax, ymin<ymax, and zmin<zmax for every cell",
                actual="contains equal or reversed bounds",
                hint="Correct the ordered (xmin,xmax,ymin,ymax,zmin,zmax) cell bounds.",
            )


def _validate_observations_outside_prisms(
    observations: torch.Tensor,
    bounds: torch.Tensor,
) -> None:
    # Vectorized in observation chunks: the scalar double loop was
    # O(n_obs * n_cells) PYTHON iterations with per-element tensor sync,
    # unusable at the family's documented target scale. The chunked test
    # reports the SAME first offender (observation-major order, then the
    # lowest offending cell index) with the identical error payload.
    n_obs = int(observations.shape[0])
    chunk = 4096
    for start in range(0, n_obs, chunk):
        obs = observations[start:start + chunk]                       # (c, 3)
        inside = (
            (bounds[None, :, 0] <= obs[:, None, 0])
            & (obs[:, None, 0] <= bounds[None, :, 1])
            & (bounds[None, :, 2] <= obs[:, None, 1])
            & (obs[:, None, 1] <= bounds[None, :, 3])
            & (bounds[None, :, 4] <= obs[:, None, 2])
            & (obs[:, None, 2] <= bounds[None, :, 5])
        )                                                             # (c, n_cells)
        offending = inside.any(dim=1)
        if bool(offending.any()):
            local_obs = int(offending.nonzero()[0, 0])
            cell_index = int(inside[local_obs].nonzero()[0, 0])
            raise PotentialContractError(
                "Every observation must be strictly outside every source prism.",
                object_name="PrismKernelPlan",
                field="observations_m",
                expected="observations outside all closed prism bounds",
                actual={
                    "observation_index": start + local_obs,
                    "cell_index": cell_index,
                },
                hint="Move the observation away from the prism interior and singular boundary.",
            )


def _update_length_prefixed(digest: _Digest, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    array = value.detach().to(device="cpu").contiguous().numpy()
    little_endian = array.astype(array.dtype.newbyteorder("<"), copy=False)
    payload: bytes = little_endian.tobytes(order="C")
    return payload


def _fingerprint(
    observations: torch.Tensor,
    bounds: torch.Tensor,
    components: tuple[str, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> str:
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_SCHEMA)
    _update_length_prefixed(digest, str(dtype).encode("utf-8"))
    _update_length_prefixed(digest, str(device).encode("utf-8"))
    for tensor_name, tensor in ((b"observations_m", observations), (b"cell_bounds_m", bounds)):
        _update_length_prefixed(digest, tensor_name)
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        _update_length_prefixed(digest, _tensor_bytes(tensor))
    digest.update(struct.pack("<Q", len(components)))
    for component in components:
        _update_length_prefixed(digest, component.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PrismKernelPlan:
    """Immutable validated geometry and canonical value identity."""

    fingerprint: str
    n_observations: int
    n_cells: int
    components: tuple[str, ...]
    dtype: torch.dtype
    device: torch.device
    _observations_m: torch.Tensor = field(repr=False)
    _cell_bounds_m: torch.Tensor = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        observations_m: torch.Tensor,
        cell_bounds_m: torch.Tensor,
        components: tuple[str, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> PrismKernelPlan:
        """Validate and own geometry before computing its canonical SHA-256."""
        validated_dtype = _validated_dtype(dtype)
        validated_device = _canonical_device(device)
        validated_components = _validated_components(components)
        observations = _validated_tensor(
            observations_m,
            field_name="observations_m",
            width=3,
            dtype=validated_dtype,
            device=validated_device,
        )
        bounds = _validated_tensor(
            cell_bounds_m,
            field_name="cell_bounds_m",
            width=6,
            dtype=validated_dtype,
            device=validated_device,
        )
        _validate_bounds(bounds)
        _validate_observations_outside_prisms(observations, bounds)
        fingerprint = _fingerprint(
            observations,
            bounds,
            validated_components,
            validated_dtype,
            validated_device,
        )
        plan = cls.__new__(cls)
        object.__setattr__(plan, "fingerprint", fingerprint)
        object.__setattr__(plan, "n_observations", observations.shape[0])
        object.__setattr__(plan, "n_cells", bounds.shape[0])
        object.__setattr__(plan, "components", validated_components)
        object.__setattr__(plan, "dtype", validated_dtype)
        object.__setattr__(plan, "device", validated_device)
        object.__setattr__(plan, "_observations_m", observations)
        object.__setattr__(plan, "_cell_bounds_m", bounds)
        return plan

    @property
    def observations_m(self) -> torch.Tensor:
        """Return a detached contiguous clone of the owned observations."""
        return self._observations_m.detach().clone(memory_format=torch.contiguous_format)

    @property
    def cell_bounds_m(self) -> torch.Tensor:
        """Return a detached contiguous clone of the owned cell bounds."""
        return self._cell_bounds_m.detach().clone(memory_format=torch.contiguous_format)


# Alias kept for existing imports (the prism_ name is canonical).
tensor_mesh_cell_bounds_m = prism_mesh_cell_bounds_m

__all__ = [
    "PrismKernelPlan",
    "prism_mesh_cell_bounds_m",
    "tensor_mesh_cell_bounds_m",
]
