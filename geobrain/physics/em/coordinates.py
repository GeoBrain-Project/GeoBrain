"""Origin-aware public EM coordinates over platform tensor meshes.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

from geobrain.mesh import TensorMesh
from geobrain.physics.em.errors import EMContractError


Axis = Literal["z", "x", "y"]
AxisLocation = Literal["node", "center"]

_AXIS_INDEX: dict[Axis, int] = {"z": 0, "x": 1, "y": 2}


def _coordinate_error(
    message: str,
    *,
    field: str,
    expected: object,
    actual: object,
    details: dict[str, object],
) -> None:
    raise EMContractError(
        message,
        object_name="EMCoordinateFrame",
        field=field,
        expected=expected,
        actual=actual,
        details=details,
        hint="use public (x, y, z) metres with z positive downward",
    )


@dataclass(frozen=True, slots=True)
class EMCoordinateFrame:
    """The sole public EM coordinate frame."""

    public_order: tuple[Literal["x"], Literal["y"], Literal["z"]]
    vertical_positive: Literal["down"]

    def __post_init__(self) -> None:
        """Own and validate the exact right-handed public frame."""
        order = tuple(self.public_order)
        if order != ("x", "y", "z"):
            _coordinate_error(
                "EM public coordinate order must be exactly (x, y, z)",
                field="public_order",
                expected=["x", "y", "z"],
                actual=list(order),
                details={
                    "field": "public_order",
                    "received": list(order),
                    "remediation": "supply coordinates in (x, y, z) order",
                },
            )
        if self.vertical_positive != "down":
            _coordinate_error(
                "EM public z must be positive downward",
                field="vertical_positive",
                expected="down",
                actual=self.vertical_positive,
                details={
                    "field": "vertical_positive",
                    "received": str(self.vertical_positive),
                    "remediation": "select the canonical downward-positive frame",
                },
            )
        object.__setattr__(
            self, "public_order", cast(tuple[Literal["x"], Literal["y"], Literal["z"]], order)
        )


def mesh_axis_coordinates(
    mesh: TensorMesh,
    axis: Axis,
    location: AxisLocation,
) -> torch.Tensor:
    """Return physical node or centre coordinates for one public mesh axis.

    ``TensorMesh`` stores axes as ``(z, x, y)``. Geometry is constructed from
    the complete width vector, so nonuniform spacing and nonzero origins are
    preserved without a mean-spacing approximation.
    """
    if axis not in _AXIS_INDEX:
        raise EMContractError(
            "unsupported EM mesh axis",
            object_name="mesh_axis_coordinates",
            field="axis",
            expected=["z", "x", "y"],
            actual=str(axis),
            details={
                "field": "axis",
                "received": str(axis),
                "supported": ["z", "x", "y"],
                "remediation": "select one public mesh axis",
            },
        )
    if location not in ("node", "center"):
        raise EMContractError(
            "unsupported EM mesh-axis location",
            object_name="mesh_axis_coordinates",
            field="location",
            expected=["node", "center"],
            actual=str(location),
            details={
                "field": "location",
                "received": str(location),
                "supported": ["node", "center"],
                "remediation": "select node or center coordinates",
            },
        )

    axis_index = _AXIS_INDEX[axis]
    if axis_index >= mesh.n_dim:
        raise EMContractError(
            "mesh does not carry the requested EM axis",
            object_name="mesh_axis_coordinates",
            field="axis",
            expected=f"axis available on {mesh.n_dim}D TensorMesh",
            actual=axis,
            details={
                "axis": axis,
                "field": "axis",
                "mesh_dimension": mesh.n_dim,
                "remediation": "use an axis present on the mesh",
            },
        )
    widths = mesh.cell_widths[axis_index]
    if widths.ndim != 1 or int(widths.numel()) == 0:
        raise EMContractError(
            "EM mesh spacing must be a non-empty vector",
            object_name="mesh_axis_coordinates",
            field=f"mesh.cell_widths[{axis_index}]",
            expected="non-empty 1-D tensor",
            actual=list(widths.shape),
            details={
                "axis": axis,
                "field": f"mesh.cell_widths[{axis_index}]",
                "remediation": "provide one positive width per mesh cell",
            },
        )
    if not bool(torch.isfinite(widths).all()) or not bool((widths > 0).all()):
        raise EMContractError(
            "EM mesh spacing must be positive and finite",
            object_name="mesh_axis_coordinates",
            field=f"mesh.cell_widths[{axis_index}]",
            expected="all finite values > 0",
            actual="invalid spacing",
            details={
                "axis": axis,
                "field": f"mesh.cell_widths[{axis_index}]",
                "remediation": "provide positive finite cell widths",
            },
        )
    origin = float(mesh.origin[axis_index])
    origin_tensor = widths.new_tensor(origin)
    cumulative = torch.cumsum(widths, dim=0)
    if location == "node":
        return torch.cat((origin_tensor.reshape(1), origin_tensor + cumulative))
    return origin_tensor + cumulative - 0.5 * widths


__all__ = ["EMCoordinateFrame", "mesh_axis_coordinates"]
