"""Immutable strict receiver projections for Yee and edge formulations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from typing import Literal, NoReturn, cast

import torch

from geobrain.mesh import TensorMesh
from geobrain.core._require import ensure_capable_mesh
from geobrain.mesh.capabilities import StructuredMesh
from geobrain.physics.em.coordinates import mesh_axis_coordinates
from geobrain.physics.em.errors import EMCapabilityError, EMContractError


class ReceiverLayout(str, Enum):
    """Supported source-to-receiver projection layouts."""

    CARTESIAN = "cartesian"
    PAIRED = "paired"


_YEE_CHANNELS: dict[
    str,
    tuple[
        Literal["E", "B"],
        tuple[Literal["node", "center"], Literal["node", "center"], Literal["node", "center"]],
    ],
] = {
    "ex": ("E", ("center", "node", "node")),
    "ey": ("E", ("node", "center", "node")),
    "ez": ("E", ("node", "node", "center")),
    "bx": ("B", ("node", "center", "center")),
    "hx": ("B", ("node", "center", "center")),
    "by": ("B", ("center", "node", "center")),
    "hy": ("B", ("center", "node", "center")),
    "bz": ("B", ("center", "center", "node")),
    "hz": ("B", ("center", "center", "node")),
}

_TET_LOCAL_EDGES = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    dtype=torch.long,
)


def _exact_tetra_barycentric(
    vertices: torch.Tensor,
    position: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return exact binary64 barycentric coordinates using rational fallback.

    The fallback runs only for float64 locations numerically close to a face.
    It preserves the sign of a one-ULP outside point on skew tetrahedra, where
    a floating solve may round a small negative coordinate to ``-0.0``.
    """

    def vector(values: torch.Tensor) -> tuple[Fraction, Fraction, Fraction]:
        return cast(
            tuple[Fraction, Fraction, Fraction],
            tuple(Fraction.from_float(float(value)) for value in values),
        )

    def subtract(
        left: tuple[Fraction, Fraction, Fraction],
        right: tuple[Fraction, Fraction, Fraction],
    ) -> tuple[Fraction, Fraction, Fraction]:
        return cast(
            tuple[Fraction, Fraction, Fraction],
            tuple(a - b for a, b in zip(left, right, strict=True)),
        )

    def determinant(
        a: tuple[Fraction, Fraction, Fraction],
        b: tuple[Fraction, Fraction, Fraction],
        c: tuple[Fraction, Fraction, Fraction],
    ) -> Fraction:
        return (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - b[0] * (a[1] * c[2] - a[2] * c[1])
            + c[0] * (a[1] * b[2] - a[2] * b[1])
        )

    v0, v1, v2, v3 = (vector(row) for row in vertices)
    point = vector(position)
    e1 = subtract(v1, v0)
    e2 = subtract(v2, v0)
    e3 = subtract(v3, v0)
    point_from_v0 = subtract(point, v0)
    denominator = determinant(e1, e2, e3)
    if denominator == 0:
        raise ArithmeticError("exact tetrahedron determinant is zero")
    numerators = (
        determinant(subtract(v1, point), subtract(v2, point), subtract(v3, point)),
        determinant(point_from_v0, e2, e3),
        determinant(e1, point_from_v0, e3),
        determinant(e1, e2, point_from_v0),
    )
    values = tuple(float(value / denominator) for value in numerators)
    return cast(tuple[float, float, float, float], values)


def _contract_error(
    message: str,
    *,
    object_name: str,
    field: str,
    expected: object,
    actual: object,
    details: dict[str, object],
) -> NoReturn:
    raise EMContractError(
        message,
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
        details=details,
        hint=str(details.get("remediation", "provide a valid receiver projection")),
    )


def _positive_count(value: object, field: str, object_name: str) -> int:
    if type(value) is not int or value <= 0:
        _contract_error(
            "receiver projection counts must be positive exact integers",
            object_name=object_name,
            field=field,
            expected="positive int",
            actual=value,
            details={
                "field": field,
                "received_type": type(value).__qualname__,
                "remediation": "provide a positive exact integer",
            },
        )
    return value


def _freeze_int_rows(value: Iterable[Iterable[object]], field: str) -> tuple[tuple[int, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a nested integer sequence") from error
    for row in rows:
        if not row or any(type(item) is not int or item < 0 for item in row):
            raise TypeError(f"{field} rows must contain non-negative exact integers")
    return cast(tuple[tuple[int, ...], ...], rows)


def _freeze_float_rows(
    value: Iterable[Iterable[object]], field: str
) -> tuple[tuple[float, ...], ...]:
    try:
        raw_rows = tuple(tuple(row) for row in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be a nested finite-number sequence") from error
    if any(not row or any(type(item) not in (int, float) for item in row) for row in raw_rows):
        raise TypeError(f"{field} rows must contain finite numbers")
    rows = tuple(tuple(float(cast(int | float, item)) for item in row) for row in raw_rows)
    if any(not row or any(not math.isfinite(item) for item in row) for row in rows):
        raise TypeError(f"{field} rows must contain finite numbers")
    return rows


def _freeze_sign_rows(
    value: Iterable[Iterable[object]],
) -> tuple[tuple[int, ...], ...]:
    try:
        rows = tuple(tuple(row) for row in value)
    except (TypeError, ValueError) as error:
        raise TypeError("orientation_signs must be a nested integer sequence") from error
    if any(
        not row or any(type(item) is not int or item not in (-1, 1) for item in row) for row in rows
    ):
        raise TypeError("orientation_signs must contain only -1 or 1")
    return cast(tuple[tuple[int, ...], ...], rows)


@dataclass(frozen=True, slots=True)
class ReceiverProjection:
    """Common immutable receiver-plan metadata."""

    formulation: Literal["yee", "edge_fem", "layered"]
    layout: ReceiverLayout
    channel: str
    receiver_count: int
    source_count: int
    output_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate exact formulation, layout, channel, counts, and shape."""
        if type(self.formulation) is not str or self.formulation not in (
            "yee",
            "edge_fem",
            "layered",
        ):
            raise TypeError("unsupported receiver formulation")
        try:
            layout = ReceiverLayout(self.layout)
        except (TypeError, ValueError):
            raise TypeError("receiver layout must be cartesian or paired")
        object.__setattr__(self, "layout", layout)
        if type(self.channel) is not str or not self.channel:
            raise TypeError("receiver channel must be a non-empty string")
        receiver_count = _positive_count(self.receiver_count, "receiver_count", type(self).__name__)
        source_count = _positive_count(self.source_count, "source_count", type(self).__name__)
        output_shape = tuple(self.output_shape)
        if any(type(item) is not int or item <= 0 for item in output_shape):
            raise TypeError("output_shape must contain positive exact integers")
        expected_shape = (
            (source_count, receiver_count)
            if layout is ReceiverLayout.CARTESIAN
            else (source_count,)
        )
        if layout is ReceiverLayout.PAIRED and source_count != receiver_count:
            raise TypeError("paired receiver projection requires equal counts")
        if output_shape != expected_shape:
            raise TypeError(f"output_shape must be {expected_shape!r} for {layout.value} layout")
        object.__setattr__(self, "output_shape", output_shape)


@dataclass(frozen=True, slots=True)
class YeeReceiverProjection(ReceiverProjection):
    """A concrete multilinear Yee projection plan."""

    formulation: Literal["yee"]
    dof_indices: tuple[tuple[int, ...], ...]
    interpolation_weights: tuple[tuple[float, ...], ...]
    paired_source_indices: tuple[int, ...] | None

    def __post_init__(self) -> None:
        ReceiverProjection.__post_init__(self)
        if self.formulation != "yee":
            raise TypeError("YeeReceiverProjection.formulation must be yee")
        if self.channel not in _YEE_CHANNELS:
            raise TypeError("YeeReceiverProjection.channel must be a Yee channel")
        indices = _freeze_int_rows(self.dof_indices, "dof_indices")
        weights = _freeze_float_rows(self.interpolation_weights, "interpolation_weights")
        if len(indices) != self.receiver_count or len(weights) != self.receiver_count:
            raise TypeError("Yee plan rows must match receiver_count")
        for index_row, weight_row in zip(indices, weights, strict=True):
            if len(index_row) != len(weight_row):
                raise TypeError("Yee index and weight row lengths must match")
            if any(weight < 0.0 for weight in weight_row) or not math.isclose(
                sum(weight_row), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise TypeError("Yee interpolation weights must form a partition")
        paired = None if self.paired_source_indices is None else tuple(self.paired_source_indices)
        if paired is not None and any(type(item) is not int for item in paired):
            raise TypeError("paired_source_indices must contain exact integers")
        expected_paired = (
            None if self.layout is ReceiverLayout.CARTESIAN else tuple(range(self.source_count))
        )
        if paired != expected_paired:
            raise TypeError("paired_source_indices do not match receiver layout")
        object.__setattr__(self, "dof_indices", indices)
        object.__setattr__(self, "interpolation_weights", weights)
        object.__setattr__(self, "paired_source_indices", paired)


@dataclass(frozen=True, slots=True)
class EdgeReceiverProjection(ReceiverProjection):
    """A concrete containing-tetra Whitney/Nédélec projection plan."""

    formulation: Literal["edge_fem"]
    element_indices: tuple[int, ...]
    local_edge_dof_indices: tuple[tuple[int, ...], ...]
    basis_weights: tuple[tuple[float, ...], ...]
    orientation_signs: tuple[tuple[int, ...], ...]
    paired_source_indices: tuple[int, ...] | None

    def __post_init__(self) -> None:
        ReceiverProjection.__post_init__(self)
        if self.formulation != "edge_fem":
            raise TypeError("EdgeReceiverProjection.formulation must be edge_fem")
        if self.channel not in _YEE_CHANNELS:
            raise TypeError("EdgeReceiverProjection.channel must be an EM field channel")
        elements = tuple(self.element_indices)
        if len(elements) != self.receiver_count or any(
            type(item) is not int or item < 0 for item in elements
        ):
            raise TypeError("element_indices must match receiver_count")
        edge_ids = _freeze_int_rows(self.local_edge_dof_indices, "local_edge_dof_indices")
        weights = _freeze_float_rows(self.basis_weights, "basis_weights")
        signs = _freeze_sign_rows(self.orientation_signs)
        if not (len(edge_ids) == len(weights) == len(signs) == self.receiver_count):
            raise TypeError("edge plan rows must match receiver_count")
        for ids, basis, orientation in zip(edge_ids, weights, signs, strict=True):
            if len(ids) != 6 or len(orientation) != 6 or len(basis) != 18:
                raise TypeError("each tetrahedral edge plan requires 6 edges and 18 basis values")
        paired = None if self.paired_source_indices is None else tuple(self.paired_source_indices)
        if paired is not None and any(type(item) is not int for item in paired):
            raise TypeError("paired_source_indices must contain exact integers")
        expected_paired = (
            None if self.layout is ReceiverLayout.CARTESIAN else tuple(range(self.source_count))
        )
        if paired != expected_paired:
            raise TypeError("paired_source_indices do not match receiver layout")
        object.__setattr__(self, "element_indices", elements)
        object.__setattr__(self, "local_edge_dof_indices", edge_ids)
        object.__setattr__(self, "basis_weights", weights)
        object.__setattr__(self, "orientation_signs", signs)
        object.__setattr__(self, "paired_source_indices", paired)


def _validate_request(
    positions_m: torch.Tensor,
    *,
    layout: ReceiverLayout | str,
    n_sources: object,
    geometry_device: torch.device,
    object_name: str,
) -> tuple[ReceiverLayout, int]:
    if not isinstance(positions_m, torch.Tensor):
        _contract_error(
            "receiver positions must be a tensor",
            object_name=object_name,
            field="positions_m",
            expected="torch.Tensor",
            actual=type(positions_m).__qualname__,
            details={
                "field": "positions_m",
                "remediation": "provide a floating (n_receivers, 3) tensor",
            },
        )
    if positions_m.ndim != 2 or int(positions_m.shape[1]) != 3 or int(positions_m.shape[0]) == 0:
        _contract_error(
            "receiver positions must have shape (n_receivers, 3)",
            object_name=object_name,
            field="positions_m.shape",
            expected="(n_receivers, 3) with n_receivers > 0",
            actual=list(positions_m.shape),
            details={
                "field": "positions_m.shape",
                "received_shape": list(positions_m.shape),
                "remediation": "provide one public (x, y, z) row per receiver",
            },
        )
    if positions_m.dtype not in (torch.float32, torch.float64):
        _contract_error(
            "receiver positions require float32 or float64",
            object_name=object_name,
            field="positions_m.dtype",
            expected=["float32", "float64"],
            actual=str(positions_m.dtype),
            details={
                "field": "positions_m.dtype",
                "received": str(positions_m.dtype),
                "remediation": "construct receiver positions in a floating dtype",
            },
        )
    if positions_m.device != geometry_device:
        raise EMCapabilityError(
            "receiver positions and mesh geometry must share a device",
            object_name=object_name,
            field="positions_m.device",
            expected=str(geometry_device),
            actual=str(positions_m.device),
            details={
                "field": "positions_m.device",
                "geometry_device": str(geometry_device),
                "position_device": str(positions_m.device),
                "remediation": "construct positions on the mesh geometry device",
            },
        )
    if positions_m.requires_grad:
        raise EMCapabilityError(
            "receiver projection geometry is non-differentiable",
            object_name=object_name,
            field="positions_m.requires_grad",
            expected=False,
            actual=True,
            details={
                "field": "positions_m.requires_grad",
                "remediation": "pass fixed receiver geometry without gradients",
            },
        )
    finite_rows = torch.isfinite(positions_m).all(dim=1)
    if not bool(finite_rows.all()):
        receiver_index = int(torch.nonzero(~finite_rows, as_tuple=False)[0, 0])
        _contract_error(
            "receiver coordinates must be finite",
            object_name=object_name,
            field=f"positions_m[{receiver_index}]",
            expected="three finite coordinates",
            actual="non-finite coordinate",
            details={
                "field": f"positions_m[{receiver_index}]",
                "receiver_index": receiver_index,
                "remediation": "replace NaN or Inf with a finite physical coordinate",
            },
        )
    try:
        resolved_layout = ReceiverLayout(layout)
    except (TypeError, ValueError) as error:
        raise EMCapabilityError(
            "unsupported receiver layout",
            object_name=object_name,
            field="layout",
            expected=["cartesian", "paired"],
            actual=str(layout),
            details={
                "field": "layout",
                "received": str(layout),
                "supported_values": ["cartesian", "paired"],
                "remediation": "select cartesian or paired",
            },
        ) from error
    source_count = _positive_count(n_sources, "n_sources", object_name)
    receiver_count = int(positions_m.shape[0])
    if resolved_layout is ReceiverLayout.PAIRED and source_count != receiver_count:
        raise EMCapabilityError(
            "paired receiver layout requires one receiver per source",
            object_name=object_name,
            field="n_sources",
            expected=receiver_count,
            actual=source_count,
            details={
                "field": "n_sources",
                "layout": "paired",
                "receiver_count": receiver_count,
                "source_count": source_count,
                "remediation": "make source and receiver counts equal or use cartesian",
            },
        )
    return resolved_layout, source_count


def _bracket(
    coordinates: torch.Tensor,
    value: torch.Tensor,
    *,
    receiver_index: int,
    axis: str,
) -> tuple[int, int, float]:
    lines = coordinates.to(dtype=value.dtype)
    lower = float(lines[0])
    upper = float(lines[-1])
    scalar = float(value)
    count = int(lines.numel())
    if scalar < lower or scalar > upper:
        _contract_error(
            "receiver lies outside the component interpolation extent",
            object_name="build_yee_receiver_projection",
            field=f"positions_m[{receiver_index}].{axis}",
            expected=[lower, upper],
            actual=scalar,
            details={
                "axis": axis,
                "bounds": [lower, upper],
                "coordinate": scalar,
                "field": f"positions_m[{receiver_index}].{axis}",
                "receiver_index": receiver_index,
                "remediation": "move the receiver inside the closed component support",
            },
        )
    if count == 1:
        return 0, 0, 0.0
    if scalar == upper:
        lower_index = count - 2
        return lower_index, count - 1, 1.0
    upper_index = int(torch.searchsorted(lines, value, right=True))
    lower_index = upper_index - 1
    upper_index = lower_index + 1
    fraction = float((value - lines[lower_index]) / (lines[upper_index] - lines[lower_index]))
    return lower_index, upper_index, fraction


def _yee_offsets(mesh: TensorMesh, channel: str) -> tuple[int, int, int, int]:
    nz, nx, ny = mesh.shape
    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_fx = (nx + 1) * ny * nz
    n_fy = nx * (ny + 1) * nz
    if channel == "ex":
        return 0, nx, ny + 1, nz + 1
    if channel == "ey":
        return n_ex, nx + 1, ny, nz + 1
    if channel == "ez":
        return n_ex + n_ey, nx + 1, ny + 1, nz
    if channel in ("bx", "hx"):
        return 0, nx + 1, ny, nz
    if channel in ("by", "hy"):
        return n_fx, nx, ny + 1, nz
    return n_fx + n_fy, nx, ny, nz + 1


def build_yee_receiver_projection(
    mesh: TensorMesh,
    positions_m: torch.Tensor,
    channel: str,
    layout: ReceiverLayout | str,
    n_sources: int,
) -> YeeReceiverProjection:
    """Resolve strict multilinear Yee stencils from public ``(x,y,z)`` points."""
    ensure_capable_mesh(mesh, StructuredMesh, owner="build_yee_receiver_projection")
    if mesh.n_dim != 3:
        _contract_error(
            "Yee receiver projection requires a 3D TensorMesh",
            object_name="build_yee_receiver_projection",
            field="mesh",
            expected="3D TensorMesh",
            actual=type(mesh).__qualname__,
            details={
                "field": "mesh",
                "remediation": "use the Yee builder only with a 3D TensorMesh",
            },
        )
    normalized_channel = channel.value if hasattr(channel, "value") else channel
    if type(normalized_channel) is not str or normalized_channel not in _YEE_CHANNELS:
        _contract_error(
            "unsupported Yee receiver channel",
            object_name="build_yee_receiver_projection",
            field="channel",
            expected=sorted(_YEE_CHANNELS),
            actual=str(normalized_channel),
            details={
                "field": "channel",
                "received": str(normalized_channel),
                "remediation": "select one Yee E, B, or H component",
            },
        )
    channel_name = normalized_channel
    resolved_layout, source_count = _validate_request(
        positions_m,
        layout=layout,
        n_sources=n_sources,
        geometry_device=mesh.cell_widths[0].device,
        object_name="build_yee_receiver_projection",
    )
    _, flavours = _YEE_CHANNELS[channel_name]
    x_lines = mesh_axis_coordinates(mesh, "x", flavours[0])
    y_lines = mesh_axis_coordinates(mesh, "y", flavours[1])
    z_lines = mesh_axis_coordinates(mesh, "z", flavours[2])
    base_offset, x_count, y_count, _z_count = _yee_offsets(mesh, channel_name)

    indices: list[tuple[int, ...]] = []
    weights: list[tuple[float, ...]] = []
    for receiver_index in range(int(positions_m.shape[0])):
        position = positions_m[receiver_index]
        ix0, ix1, wx = _bracket(
            x_lines,
            position[0],
            receiver_index=receiver_index,
            axis="x",
        )
        iy0, iy1, wy = _bracket(
            y_lines,
            position[1],
            receiver_index=receiver_index,
            axis="y",
        )
        iz0, iz1, wz = _bracket(
            z_lines,
            position[2],
            receiver_index=receiver_index,
            axis="z",
        )
        row_indices: list[int] = []
        row_weights: list[float] = []
        for z_index, z_weight in ((iz0, 1.0 - wz), (iz1, wz)):
            for y_index, y_weight in ((iy0, 1.0 - wy), (iy1, wy)):
                for x_index, x_weight in ((ix0, 1.0 - wx), (ix1, wx)):
                    row_indices.append(
                        base_offset + x_index + y_index * x_count + z_index * x_count * y_count
                    )
                    row_weights.append(x_weight * y_weight * z_weight)
        indices.append(tuple(row_indices))
        weights.append(tuple(row_weights))

    receiver_count = int(positions_m.shape[0])
    output_shape = (
        (source_count, receiver_count)
        if resolved_layout is ReceiverLayout.CARTESIAN
        else (source_count,)
    )
    return YeeReceiverProjection(
        formulation="yee",
        layout=resolved_layout,
        channel=channel_name,
        receiver_count=receiver_count,
        source_count=source_count,
        output_shape=output_shape,
        dof_indices=tuple(indices),
        interpolation_weights=tuple(weights),
        paired_source_indices=(
            None if resolved_layout is ReceiverLayout.CARTESIAN else tuple(range(source_count))
        ),
    )


def build_edge_receiver_projection(
    mesh: object,
    positions_m: torch.Tensor,
    channel: str,
    layout: ReceiverLayout | str,
    n_sources: int,
) -> EdgeReceiverProjection:
    """Locate containing tetrahedra and evaluate concrete Whitney bases."""
    required = ("node_coords", "cell_nodes", "cell_edges")
    if any(not callable(getattr(mesh, name, None)) for name in required):
        _contract_error(
            "edge receiver projection requires tetrahedral edge connectivity",
            object_name="build_edge_receiver_projection",
            field="mesh",
            expected=list(required),
            actual=type(mesh).__qualname__,
            details={
                "field": "mesh",
                "remediation": "provide a tetrahedral EdgeConnectivityMesh",
            },
        )
    normalized_channel = channel.value if hasattr(channel, "value") else channel
    if type(normalized_channel) is not str or normalized_channel not in _YEE_CHANNELS:
        _contract_error(
            "unsupported edge receiver channel",
            object_name="build_edge_receiver_projection",
            field="channel",
            expected=sorted(_YEE_CHANNELS),
            actual=str(normalized_channel),
            details={
                "field": "channel",
                "received": str(normalized_channel),
                "remediation": "select one edge E, B, or H component",
            },
        )
    node_coords = cast(torch.Tensor, getattr(mesh, "node_coords")())
    cell_nodes = cast(torch.Tensor, getattr(mesh, "cell_nodes")())
    cell_edge_ids, cell_edge_signs = cast(
        tuple[torch.Tensor, torch.Tensor], getattr(mesh, "cell_edges")()
    )
    resolved_layout, source_count = _validate_request(
        positions_m,
        layout=layout,
        n_sources=n_sources,
        geometry_device=node_coords.device,
        object_name="build_edge_receiver_projection",
    )
    coords = node_coords.to(torch.float64)
    cells = cell_nodes.to(torch.long)
    vertices = coords[cells]
    cell_origins = vertices[:, 0]
    local_matrices = (vertices[:, 1:] - cell_origins[:, None, :]).transpose(1, 2).contiguous()
    determinants = torch.linalg.det(local_matrices)
    if not bool(torch.isfinite(determinants).all()) or bool((determinants == 0).any()):
        _contract_error(
            "edge receiver projection cannot locate points in degenerate cells",
            object_name="build_edge_receiver_projection",
            field="mesh.cell_nodes",
            expected="finite nonzero tetrahedral volumes",
            actual="degenerate cell",
            details={
                "field": "mesh.cell_nodes",
                "remediation": "repair the tetrahedral mesh",
            },
        )
    # Public receivers are always (x, y, z); the core mesh is canonical
    # (z, x, y). Keep the bridge explicit at this boundary.
    positions_f64 = positions_m.to(torch.float64)
    positions_zxy = positions_f64[:, (2, 0, 1)]
    relative_positions = positions_zxy[:, None, :] - cell_origins[None, :, :]
    nonzero_barycentric = torch.linalg.solve(
        local_matrices.unsqueeze(0),
        relative_positions.unsqueeze(-1),
    ).squeeze(-1)

    # Neumaier-compensated ``1 - λ1 - λ2 - λ3`` preserves the sign of a
    # one-ULP excursion across the face opposite local vertex 0. A plain
    # reduction can round that residual to zero and silently admit the point.
    lambda_zero = torch.ones(
        nonzero_barycentric.shape[:-1],
        dtype=torch.float64,
        device=nonzero_barycentric.device,
    )
    compensation = torch.zeros_like(lambda_zero)
    for term in (-nonzero_barycentric).unbind(dim=-1):
        total = lambda_zero + term
        compensation = compensation + torch.where(
            lambda_zero.abs() >= term.abs(),
            (lambda_zero - total) + term,
            (term - total) + lambda_zero,
        )
        lambda_zero = total
    lambda_zero = lambda_zero + compensation
    barycentric = torch.cat((lambda_zero.unsqueeze(-1), nonzero_barycentric), dim=-1)
    inside = ((barycentric >= 0.0) & (barycentric <= 1.0)).all(dim=2)

    # A float64 solve can erase the sign of a one-ULP face excursion on a skew
    # tetrahedron. Reclassify only numerically face-adjacent pairs with exact
    # rational arithmetic over the represented binary64 coordinates.
    condition = torch.linalg.cond(local_matrices)
    uncertainty = (
        64.0 * torch.finfo(torch.float64).eps * torch.maximum(condition, torch.ones_like(condition))
    )
    near_face = (
        (barycentric.abs() <= uncertainty[None, :, None])
        | ((1.0 - barycentric).abs() <= uncertainty[None, :, None])
    ).any(dim=2)
    for receiver_index, cell_index in torch.nonzero(near_face, as_tuple=False):
        receiver = int(receiver_index)
        cell = int(cell_index)
        try:
            exact = _exact_tetra_barycentric(
                vertices[cell],
                positions_zxy[receiver],
            )
        except ArithmeticError:
            _contract_error(
                "edge receiver projection cannot locate points in degenerate cells",
                object_name="build_edge_receiver_projection",
                field="mesh.cell_nodes",
                expected="finite nonzero tetrahedral volumes",
                actual="exactly degenerate cell",
                details={
                    "field": "mesh.cell_nodes",
                    "remediation": "repair the tetrahedral mesh",
                },
            )
        barycentric[receiver, cell] = barycentric.new_tensor(exact)
        inside[receiver, cell] = all(0.0 <= value <= 1.0 for value in exact)

    identity = torch.eye(3, dtype=torch.float64, device=vertices.device).expand(
        int(vertices.shape[0]),
        -1,
        -1,
    )
    inverse_local = torch.linalg.solve(local_matrices, identity)
    gradients = torch.cat(
        (-inverse_local.sum(dim=1, keepdim=True), inverse_local),
        dim=1,
    )

    element_indices: list[int] = []
    edge_rows: list[tuple[int, ...]] = []
    basis_rows: list[tuple[float, ...]] = []
    sign_rows: list[tuple[int, ...]] = []
    local_edges = _TET_LOCAL_EDGES.to(cells.device)
    for receiver_index in range(int(positions_m.shape[0])):
        matches = torch.nonzero(inside[receiver_index], as_tuple=False).reshape(-1)
        coordinate = [float(item) for item in positions_m[receiver_index]]
        if int(matches.numel()) == 0:
            _contract_error(
                "edge receiver point location is outside",
                object_name="build_edge_receiver_projection",
                field=f"positions_m[{receiver_index}]",
                expected="at least one containing tetrahedron",
                actual=0,
                details={
                    "candidate_count": 0,
                    "coordinate": coordinate,
                    "field": f"positions_m[{receiver_index}]",
                    "receiver_index": receiver_index,
                    "remediation": "move the receiver onto or inside the tetrahedral mesh",
                },
            )
        # Shared faces/edges are legal physical receiver locations. Preserve the
        # former nearest-cell ownership rule, restricted to true containing
        # cells, with the ascending cell index as a deterministic tie-break.
        candidate_centres = vertices[matches].mean(dim=1)
        distance_squared = ((candidate_centres - positions_zxy[receiver_index]) ** 2).sum(dim=1)
        cell_index = int(matches[int(torch.argmin(distance_squared))])
        lambdas = barycentric[receiver_index, cell_index]
        cell_gradients = gradients[cell_index]
        basis = (
            lambdas[local_edges[:, 0], None] * cell_gradients[local_edges[:, 1]]
            - lambdas[local_edges[:, 1], None] * cell_gradients[local_edges[:, 0]]
        )
        basis = basis[:, (1, 2, 0)]  # core (z,x,y) vectors -> public (x,y,z)
        element_indices.append(cell_index)
        edge_rows.append(tuple(int(item) for item in cell_edge_ids[cell_index]))
        basis_rows.append(tuple(float(item) for item in basis.reshape(-1)))
        sign_rows.append(tuple(int(item) for item in cell_edge_signs[cell_index]))

    receiver_count = int(positions_m.shape[0])
    output_shape = (
        (source_count, receiver_count)
        if resolved_layout is ReceiverLayout.CARTESIAN
        else (source_count,)
    )
    return EdgeReceiverProjection(
        formulation="edge_fem",
        layout=resolved_layout,
        channel=normalized_channel,
        receiver_count=receiver_count,
        source_count=source_count,
        output_shape=output_shape,
        element_indices=tuple(element_indices),
        local_edge_dof_indices=tuple(edge_rows),
        basis_weights=tuple(basis_rows),
        orientation_signs=tuple(sign_rows),
        paired_source_indices=(
            None if resolved_layout is ReceiverLayout.CARTESIAN else tuple(range(source_count))
        ),
    )


__all__ = [
    "EdgeReceiverProjection",
    "ReceiverLayout",
    "ReceiverProjection",
    "YeeReceiverProjection",
    "build_edge_receiver_projection",
    "build_yee_receiver_projection",
]
