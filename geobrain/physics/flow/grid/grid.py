"""Canonical SI Cartesian grids and finite-volume connectivity for Flow.

Coordinates use ``(x, y, z)`` columns in metres, ``z`` is positive downward,
and cell ids are x-fastest: ``ix + iy*nx + iz*nx*ny``. The explicit
``TensorMesh`` bridge maps to the unchanged core ``(z, x, y)`` axis order; it
never accepts a unit selector because both representations are SI.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from ....mesh import TensorMesh
from .._defaults import DEVICE, DTYPE
from ..errors import FlowContractError
from .topology import OrientedFaceTopology, cartesian_oriented_topology


@dataclass(frozen=True, slots=True)
class ConnList:
    """Interior connection metrics in canonical SI units.

    ``neighbors`` contains oriented ``(left, right)`` cell ids. Areas are m²
    and half distances are m. All tensors share one device; floating tensors
    share one dtype.
    """

    neighbors: torch.Tensor
    face_area: torch.Tensor
    half_dist_l: torch.Tensor
    half_dist_r: torch.Tensor
    is_boundary: torch.Tensor

    @property
    def n_faces(self) -> int:
        return int(self.neighbors.shape[0])

    def clone(self) -> "ConnList":
        """Return a storage-independent public snapshot."""

        return ConnList(
            neighbors=self.neighbors.clone(),
            face_area=self.face_area.clone(),
            half_dist_l=self.half_dist_l.clone(),
            half_dist_r=self.half_dist_r.clone(),
            is_boundary=self.is_boundary.clone(),
        )

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "ConnList":
        """Explicitly copy the connection metrics to ``device``/``dtype``."""

        target_dtype = self.face_area.dtype if dtype is None else dtype
        return ConnList(
            neighbors=self.neighbors.to(device=device).clone(),
            face_area=self.face_area.to(device=device, dtype=target_dtype).clone(),
            half_dist_l=self.half_dist_l.to(device=device, dtype=target_dtype).clone(),
            half_dist_r=self.half_dist_r.to(device=device, dtype=target_dtype).clone(),
            is_boundary=self.is_boundary.to(device=device).clone(),
        )


class FlowGrid(nn.Module):  # type: ignore[misc,unused-ignore]  # isolated strict import boundary
    """Base finite-volume grid surface consumed by Flow models."""

    _cell_centers_m: torch.Tensor
    _cell_volumes_m3: torch.Tensor
    _conn_neighbors: torch.Tensor
    _conn_face_area: torch.Tensor
    _conn_half_dist_l: torch.Tensor
    _conn_half_dist_r: torch.Tensor
    _conn_is_boundary: torch.Tensor
    _topology_face_cells: torch.Tensor
    _topology_boundary_cells: torch.Tensor
    _topology_boundary_normals: torch.Tensor
    _topology_incidence: torch.Tensor

    def __init__(
        self,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self._device = torch.device(device)
        self._dtype = dtype
        self.register_buffer(
            "_cell_centers_m",
            torch.empty(0, 3, dtype=dtype, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_cell_volumes_m3",
            torch.empty(0, dtype=dtype, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_conn_neighbors",
            torch.empty((0, 2), dtype=torch.int64, device=self.device),
            persistent=False,
        )
        for name in (
            "_conn_face_area",
            "_conn_half_dist_l",
            "_conn_half_dist_r",
        ):
            self.register_buffer(
                name,
                torch.empty(0, dtype=dtype, device=self.device),
                persistent=False,
            )
        self.register_buffer(
            "_conn_is_boundary",
            torch.empty(0, dtype=torch.bool, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_topology_face_cells",
            torch.empty((0, 2), dtype=torch.int64, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_topology_boundary_cells",
            torch.empty(0, dtype=torch.int64, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_topology_boundary_normals",
            torch.empty((0, 3), dtype=dtype, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_topology_incidence",
            torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=dtype, device=self.device),
                size=(0, 0),
                device=self.device,
                dtype=dtype,
            ).coalesce(),
            persistent=False,
        )
        self._has_connections = False
        self._has_topology = False

    @property
    def device(self) -> torch.device:
        """Read-only device shared by every grid tensor."""

        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Read-only floating dtype shared by every grid metric."""

        return self._dtype

    @property
    def coordinate_columns(self) -> tuple[str, str, str]:
        """Read-only canonical coordinate-column order."""

        return ("x", "y", "z")

    @property
    def z_positive_down(self) -> bool:
        """Read-only depth-direction convention."""

        return True

    def _build_geometry(self) -> None:
        raise NotImplementedError

    def _build_connections(self) -> None:
        raise NotImplementedError

    @property
    def cell_centers_m(self) -> torch.Tensor:
        """Storage-independent xyz cell-centre snapshot in metres."""

        return self._cell_centers_m.clone()

    @property
    def cell_volumes_m3(self) -> torch.Tensor:
        """Storage-independent cell-volume snapshot in cubic metres."""

        return self._cell_volumes_m3.clone()

    def _cell_centers_view(self) -> torch.Tensor:
        """Return the zero-copy family-internal cell-centre view."""

        return self._cell_centers_m

    def _cell_volumes_view(self) -> torch.Tensor:
        """Return the zero-copy family-internal cell-volume view."""

        return self._cell_volumes_m3

    def _connection_metrics(self) -> ConnList | None:
        """Return the zero-copy family-internal connection view."""

        if not self._has_connections:
            return None
        return ConnList(
            neighbors=self._conn_neighbors,
            face_area=self._conn_face_area,
            half_dist_l=self._conn_half_dist_l,
            half_dist_r=self._conn_half_dist_r,
            is_boundary=self._conn_is_boundary,
        )

    @property
    def conn(self) -> ConnList | None:
        """Return storage-independent connection metrics for public callers."""

        internal = self._connection_metrics()
        return None if internal is None else internal.clone()

    def _oriented_topology(self) -> OrientedFaceTopology | None:
        """Return the zero-copy family-internal topology view."""

        if not self._has_topology:
            return None
        return OrientedFaceTopology(
            face_cells=self._topology_face_cells,
            boundary_cells=self._topology_boundary_cells,
            boundary_normals=self._topology_boundary_normals,
            incidence=self._topology_incidence,
        )

    @property
    def topology(self) -> OrientedFaceTopology | None:
        """Return a storage-independent oriented-topology snapshot."""

        internal = self._oriented_topology()
        return None if internal is None else internal.clone()

    def _set_connection_metrics(self, connection: ConnList) -> None:
        self._conn_neighbors = connection.neighbors
        self._conn_face_area = connection.face_area
        self._conn_half_dist_l = connection.half_dist_l
        self._conn_half_dist_r = connection.half_dist_r
        self._conn_is_boundary = connection.is_boundary
        self._has_connections = True

    def _set_oriented_topology(self, topology: OrientedFaceTopology) -> None:
        self._topology_face_cells = topology.face_cells
        self._topology_boundary_cells = topology.boundary_cells
        self._topology_boundary_normals = topology.boundary_normals
        self._topology_incidence = topology.incidence
        self._has_topology = True

    @property
    def n_cells(self) -> int:
        return int(self._cell_volumes_m3.shape[0])

    @property
    def n_faces(self) -> int:
        connection = self._connection_metrics()
        return connection.n_faces if connection is not None else 0

    def build_transmissibility(self, perm: torch.Tensor) -> torch.Tensor:
        """Return SI TPFA geometric transmissibility in m³.

        ``perm`` is isotropic ``[cell]`` or diagonal ``[cell, xyz]`` in m².
        The operation remains differentiable in ``perm`` and never caches a
        graph-bound result.
        """

        connection = self._connection_metrics()
        if connection is None:
            raise FlowContractError(
                "FlowGrid connections have not been built",
                object_name=type(self).__name__,
                field="conn",
                expected="ConnList",
                actual=None,
            )
        if perm.device != self.device or perm.dtype != self.dtype:
            raise FlowContractError(
                "permeability dtype/device must match the grid",
                object_name=type(self).__name__,
                field="perm",
                expected={"dtype": str(self.dtype), "device": str(self.device)},
                actual={"dtype": str(perm.dtype), "device": str(perm.device)},
            )
        neighbors = connection.neighbors
        if perm.ndim == 1:
            if perm.shape[0] != self.n_cells:
                raise FlowContractError(
                    "perm length must equal n_cells",
                    object_name=type(self).__name__,
                    field="perm",
                    expected=(self.n_cells,),
                    actual=tuple(perm.shape),
                )
            perm_left = perm[neighbors[:, 0]]
            perm_right = perm[neighbors[:, 1]]
        elif perm.ndim == 2 and perm.shape == (self.n_cells, 3):
            deltas = (
                self._cell_centers_m[neighbors[:, 1]] - self._cell_centers_m[neighbors[:, 0]]
            ).abs()
            axis = deltas.argmax(dim=1)
            perm_left = perm[neighbors[:, 0], axis]
            perm_right = perm[neighbors[:, 1], axis]
        else:
            raise FlowContractError(
                "perm must be [cell] or [cell, xyz]",
                object_name=type(self).__name__,
                field="perm",
                expected=f"({self.n_cells},) or ({self.n_cells}, 3)",
                actual=tuple(perm.shape),
            )
        half_left = perm_left * connection.face_area / connection.half_dist_l
        half_right = perm_right * connection.face_area / connection.half_dist_r
        denominator = half_left + half_right
        nonzero = denominator != 0
        safe_denominator = torch.where(
            nonzero,
            denominator,
            torch.ones_like(denominator),
        )
        return torch.where(
            nonzero,
            (half_left * half_right) / safe_denominator,
            torch.zeros_like(denominator),
        )


class CartGrid(FlowGrid):
    """Structured Cartesian grid with immutable SI geometry metadata.

    Args:
        nx / ny / nz: cell counts per axis.
        dx_m / dy_m / dz_m: cell sizes [m] (scalar or per-cell tensors).
        origin_m: grid origin [m].
        device / dtype: tensor placement of the grid geometry.
    """

    _dx_m: torch.Tensor
    _dy_m: torch.Tensor
    _dz_m: torch.Tensor

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx_m: float | torch.Tensor,
        dy_m: float | torch.Tensor,
        dz_m: float | torch.Tensor,
        *,
        origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if dtype not in {torch.float32, torch.float64}:
            raise FlowContractError(
                "CartGrid dtype must be a supported real floating dtype",
                object_name="CartGrid",
                field="dtype",
                expected=(str(torch.float32), str(torch.float64)),
                actual=str(dtype),
            )
        resolved_device = torch.device(device)
        super().__init__(device=resolved_device, dtype=dtype)
        for name, count in (("nx", nx), ("ny", ny), ("nz", nz)):
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise FlowContractError(
                    "CartGrid cell count must be a positive int",
                    object_name="CartGrid",
                    field=name,
                    expected="positive int",
                    actual=count,
                )
        self._nx, self._ny, self._nz = nx, ny, nz
        self._origin_m = self._validate_origin(origin_m)
        self.register_buffer(
            "_dx_m",
            self._spacing(dx_m, nx, "dx_m"),
            persistent=False,
        )
        self.register_buffer(
            "_dy_m",
            self._spacing(dy_m, ny, "dy_m"),
            persistent=False,
        )
        self.register_buffer(
            "_dz_m",
            self._spacing(dz_m, nz, "dz_m"),
            persistent=False,
        )
        self._build_geometry()
        self._build_connections()
        connection = self._connection_metrics()
        assert connection is not None
        self._set_oriented_topology(
            cartesian_oriented_topology(
                nx=nx,
                ny=ny,
                nz=nz,
                face_cells=connection.neighbors,
                dtype=dtype,
                device=resolved_device,
            )
        )

    @property
    def dx_m(self) -> torch.Tensor:
        """Storage-independent x-width snapshot in metres."""

        return self._dx_m.clone()

    @property
    def dy_m(self) -> torch.Tensor:
        """Storage-independent y-width snapshot in metres."""

        return self._dy_m.clone()

    @property
    def dz_m(self) -> torch.Tensor:
        """Storage-independent z-width snapshot in metres."""

        return self._dz_m.clone()

    def _axis_widths_view(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero-copy family-internal x/y/z width views."""

        return self._dx_m, self._dy_m, self._dz_m

    @property
    def nx(self) -> int:
        """Read-only number of cells along x."""

        return self._nx

    @property
    def ny(self) -> int:
        """Read-only number of cells along y."""

        return self._ny

    @property
    def nz(self) -> int:
        """Read-only number of cells along z."""

        return self._nz

    @property
    def origin_m(self) -> tuple[float, float, float]:
        """Read-only xyz origin in metres."""

        return self._origin_m

    @staticmethod
    def _require_derived_tensor(
        field: str,
        value: torch.Tensor,
        *,
        positive: bool = False,
    ) -> None:
        valid = bool(torch.isfinite(value).all())
        if positive:
            valid = valid and bool((value > 0).all())
        if not valid:
            raise FlowContractError(
                "CartGrid derived geometry must be finite and physically valid",
                object_name="CartGrid",
                field=field,
                expected="all finite" + (" and > 0" if positive else ""),
                actual="invalid derived entries present",
            )

    def _build_topology(self) -> None:
        connection = self._connection_metrics()
        assert connection is not None
        self._set_oriented_topology(
            cartesian_oriented_topology(
                nx=self.nx,
                ny=self.ny,
                nz=self.nz,
                face_cells=connection.neighbors,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> "CartGrid":
        """Move/cast immutable geometry atomically with its cached topology."""

        candidate_dx = fn(self._dx_m.clone())
        candidate_dy = fn(self._dy_m.clone())
        candidate_dz = fn(self._dz_m.clone())
        if not all(
            isinstance(value, torch.Tensor) for value in (candidate_dx, candidate_dy, candidate_dz)
        ):
            raise TypeError("CartGrid tensor transform must return tensors")
        target_dtype = candidate_dx.dtype
        target_device = candidate_dx.device
        if target_dtype not in {torch.float32, torch.float64}:
            raise FlowContractError(
                "CartGrid dtype must be a supported real floating dtype",
                object_name="CartGrid",
                field="dtype",
                expected=(str(torch.float32), str(torch.float64)),
                actual=str(target_dtype),
            )
        if any(
            value.dtype != target_dtype or value.device != target_device
            for value in (candidate_dy, candidate_dz)
        ):
            raise FlowContractError(
                "CartGrid tensor transform must preserve one dtype/device",
                object_name="CartGrid",
                field="dtype/device",
                expected={"dtype": str(target_dtype), "device": str(target_device)},
                actual="mixed transformed geometry",
            )
        for field, value, expected_shape in (
            ("dx_m", candidate_dx, (self.nx,)),
            ("dy_m", candidate_dy, (self.ny,)),
            ("dz_m", candidate_dz, (self.nz,)),
        ):
            if value.shape != expected_shape:
                raise FlowContractError(
                    "CartGrid tensor transform must preserve spacing shape",
                    object_name="CartGrid",
                    field=field,
                    expected=expected_shape,
                    actual=tuple(value.shape),
                )
            if value.requires_grad:
                raise FlowContractError(
                    "CartGrid spacing is immutable metadata and cannot require gradients",
                    object_name="CartGrid",
                    field=f"{field}.requires_grad",
                    expected=False,
                    actual=True,
                )
            if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
                raise FlowContractError(
                    "CartGrid tensor transform must preserve positive finite spacing",
                    object_name="CartGrid",
                    field=field,
                    expected="all finite and > 0",
                    actual="invalid transformed entries present",
                )

        # Preflight the target representation before mutating registered buffers.
        original_dx, original_dy, original_dz = self._dx_m, self._dy_m, self._dz_m
        original_dtype, original_device = self.dtype, self.device
        try:
            self._dx_m, self._dy_m, self._dz_m = candidate_dx, candidate_dy, candidate_dz
            self._dtype, self._device = target_dtype, target_device
            self._build_geometry()
            self._build_connections()
            self._build_topology()
        except Exception:
            self._dx_m, self._dy_m, self._dz_m = original_dx, original_dy, original_dz
            self._dtype, self._device = original_dtype, original_device
            self._build_geometry()
            self._build_connections()
            self._build_topology()
            raise

        # Geometry has no parameters or child modules; all registered buffers
        # were rebuilt directly on the target and form one consistent snapshot.
        return self

    @staticmethod
    def _validate_origin(origin_m: object) -> tuple[float, float, float]:
        if not isinstance(origin_m, tuple) or len(origin_m) != 3:
            raise FlowContractError(
                "CartGrid origin_m must contain exactly three xyz values",
                object_name="CartGrid",
                field="origin_m",
                expected="tuple[float, float, float]",
                actual=origin_m,
            )
        converted: list[float] = []
        for axis, value in enumerate(origin_m):
            try:
                number = float(value) if not isinstance(value, bool) else float("nan")
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            if not math.isfinite(number):
                raise FlowContractError(
                    "CartGrid origin entries must be finite",
                    object_name="CartGrid",
                    field=f"origin_m[{axis}]",
                    expected="finite float",
                    actual=value,
                )
            converted.append(number)
        return (converted[0], converted[1], converted[2])

    def _spacing(
        self,
        value: float | torch.Tensor,
        count: int,
        name: str,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device != self.device:
                raise FlowContractError(
                    "CartGrid spacing tensor device does not match declared device",
                    object_name="CartGrid",
                    field=f"{name}.device",
                    expected=str(self.device),
                    actual=str(value.device),
                )
            if value.dtype != self.dtype:
                raise FlowContractError(
                    "CartGrid spacing tensor dtype does not match declared dtype",
                    object_name="CartGrid",
                    field=f"{name}.dtype",
                    expected=str(self.dtype),
                    actual=str(value.dtype),
                )
            if value.requires_grad:
                raise FlowContractError(
                    "CartGrid spacing is immutable metadata and cannot require gradients",
                    object_name="CartGrid",
                    field=f"{name}.requires_grad",
                    expected=False,
                    actual=True,
                )
            if value.ndim == 0:
                spacing = value.expand(count)
            elif value.ndim == 1 and value.numel() == 1:
                spacing = value.expand(count)
            elif value.ndim == 1 and value.numel() == count:
                spacing = value
            else:
                raise FlowContractError(
                    "CartGrid spacing tensor has the wrong shape",
                    object_name="CartGrid",
                    field=name,
                    expected=(count,),
                    actual=tuple(value.shape),
                )
        else:
            try:
                number = float(value) if not isinstance(value, bool) else float("nan")
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            spacing = torch.full((count,), number, dtype=self.dtype, device=self.device)
        if not bool(torch.isfinite(spacing).all()) or not bool((spacing > 0).all()):
            raise FlowContractError(
                "CartGrid spacing must be finite and positive",
                object_name="CartGrid",
                field=name,
                expected="all finite and > 0",
                actual="invalid entries present",
            )
        return spacing.clone()

    def _build_geometry(self) -> None:
        ox_m, oy_m, oz_m = self.origin_m
        # Subtract the half-width before adding a large origin.  The former
        # ``origin + cumsum - half`` ordering overflowed even when the final
        # centre remained representable.
        centers_x = self._axis_centers(self._dx_m, ox_m)
        centers_y = self._axis_centers(self._dy_m, oy_m)
        centers_z = self._axis_centers(self._dz_m, oz_m)
        grid_z, grid_y, grid_x = torch.meshgrid(centers_z, centers_y, centers_x, indexing="ij")
        cell_centers = torch.stack(
            (grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)), dim=1
        )
        widths_z, widths_y, widths_x = torch.meshgrid(
            self._dz_m, self._dy_m, self._dx_m, indexing="ij"
        )
        width_products_xy = widths_x * widths_y
        direct_volumes = width_products_xy * widths_z
        finfo = torch.finfo(self.dtype)
        unstable_products = (
            ~torch.isfinite(width_products_xy)
            | (width_products_xy <= 0)
            | ((width_products_xy < finfo.tiny) & (width_products_xy > 0))
            | ~torch.isfinite(direct_volumes)
            | (direct_volumes <= 0)
        )
        if bool(unstable_products.any()):
            stable_volumes = torch.exp(
                torch.log(widths_x) + torch.log(widths_y) + torch.log(widths_z)
            )
            direct_volumes = torch.where(unstable_products, stable_volumes, direct_volumes)
        cell_volumes = direct_volumes.reshape(-1)
        self._require_derived_tensor("cell_centers_m", cell_centers)
        self._require_derived_tensor("cell_volumes_m3", cell_volumes, positive=True)
        self._cell_centers_m = cell_centers
        self._cell_volumes_m3 = cell_volumes

    @staticmethod
    def _axis_centers(widths: torch.Tensor, origin_m: float) -> torch.Tensor:
        """Return centres while preserving ordinary-path rounding.

        The direct cumulative expression is fast and retains the established
        arithmetic for normal reservoir grids.  If its edge sum overflows,
        the recurrence advances from one centre to the next; positive widths
        make any recurrence overflow a true unrepresentable centre rather
        than a removable intermediate overflow.
        """

        direct = origin_m + (torch.cumsum(widths, 0) - 0.5 * widths)
        if bool(torch.isfinite(direct).all()):
            return direct
        centres = [widths.new_tensor(origin_m) + 0.5 * widths[0]]
        for index in range(1, widths.numel()):
            centres.append(centres[-1] + 0.5 * widths[index - 1] + 0.5 * widths[index])
        return torch.stack(centres)

    def _build_connections(self) -> None:
        neighbor_parts: list[torch.Tensor] = []
        area_parts: list[torch.Tensor] = []
        left_parts: list[torch.Tensor] = []
        right_parts: list[torch.Tensor] = []
        nx, ny, nz = self.nx, self.ny, self.nz

        if nx > 1:
            ix, iy, iz = torch.meshgrid(
                torch.arange(nx - 1, device=self.device),
                torch.arange(ny, device=self.device),
                torch.arange(nz, device=self.device),
                indexing="ij",
            )
            left = ix + iy * nx + iz * nx * ny
            right = left + 1
            neighbor_parts.append(torch.stack((left.reshape(-1), right.reshape(-1)), 1))
            area_parts.append((self._dy_m[iy] * self._dz_m[iz]).reshape(-1))
            left_parts.append((0.5 * self._dx_m[ix]).reshape(-1))
            right_parts.append((0.5 * self._dx_m[ix + 1]).reshape(-1))
        if ny > 1:
            ix, iy, iz = torch.meshgrid(
                torch.arange(nx, device=self.device),
                torch.arange(ny - 1, device=self.device),
                torch.arange(nz, device=self.device),
                indexing="ij",
            )
            left = ix + iy * nx + iz * nx * ny
            right = left + nx
            neighbor_parts.append(torch.stack((left.reshape(-1), right.reshape(-1)), 1))
            area_parts.append((self._dx_m[ix] * self._dz_m[iz]).reshape(-1))
            left_parts.append((0.5 * self._dy_m[iy]).reshape(-1))
            right_parts.append((0.5 * self._dy_m[iy + 1]).reshape(-1))
        if nz > 1:
            ix, iy, iz = torch.meshgrid(
                torch.arange(nx, device=self.device),
                torch.arange(ny, device=self.device),
                torch.arange(nz - 1, device=self.device),
                indexing="ij",
            )
            left = ix + iy * nx + iz * nx * ny
            right = left + nx * ny
            neighbor_parts.append(torch.stack((left.reshape(-1), right.reshape(-1)), 1))
            area_parts.append((self._dx_m[ix] * self._dy_m[iy]).reshape(-1))
            left_parts.append((0.5 * self._dz_m[iz]).reshape(-1))
            right_parts.append((0.5 * self._dz_m[iz + 1]).reshape(-1))

        if neighbor_parts:
            neighbors = torch.cat(neighbor_parts).to(torch.int64)
            face_area = torch.cat(area_parts)
            half_dist_l = torch.cat(left_parts)
            half_dist_r = torch.cat(right_parts)
        else:
            neighbors = torch.empty((0, 2), dtype=torch.int64, device=self.device)
            face_area = torch.empty(0, dtype=self.dtype, device=self.device)
            half_dist_l = torch.empty(0, dtype=self.dtype, device=self.device)
            half_dist_r = torch.empty(0, dtype=self.dtype, device=self.device)
        self._require_derived_tensor("conn.face_area", face_area, positive=True)
        self._require_derived_tensor("conn.half_dist_l", half_dist_l, positive=True)
        self._require_derived_tensor("conn.half_dist_r", half_dist_r, positive=True)
        if neighbors.numel():
            center_deltas = (
                self._cell_centers_m[neighbors[:, 1]] - self._cell_centers_m[neighbors[:, 0]]
            )
            self._require_derived_tensor("conn.center_delta", center_deltas)
            self._require_derived_tensor(
                "conn.center_distance",
                center_deltas.abs().amax(dim=1),
                positive=True,
            )
        self._set_connection_metrics(
            ConnList(
                neighbors=neighbors,
                face_area=face_area,
                half_dist_l=half_dist_l,
                half_dist_r=half_dist_r,
                is_boundary=torch.zeros(neighbors.shape[0], dtype=torch.bool, device=self.device),
            )
        )

    def ijk_to_global(self, i: int, j: int = 0, k: int = 0) -> int:
        """Map a zero-based ``(i, j, k)`` index to the x-fastest cell id."""

        if not (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            raise FlowContractError(
                "ijk index is outside the Cartesian grid",
                object_name="CartGrid",
                field="ijk",
                expected=f"0<=i<{self.nx}, 0<=j<{self.ny}, 0<=k<{self.nz}",
                actual=(i, j, k),
            )
        return i + j * self.nx + k * self.nx * self.ny

    @classmethod
    def from_tensor_mesh(cls, mesh: TensorMesh) -> "CartGrid":
        """Map a 3-D core ``(z, x, y)`` SI mesh into Flow xyz/x-fastest form."""

        if not isinstance(mesh, TensorMesh):
            raise FlowContractError(
                "from_tensor_mesh expects a TensorMesh",
                object_name="CartGrid.from_tensor_mesh",
                field="mesh",
                expected="TensorMesh",
                actual=type(mesh).__name__,
            )
        if mesh.n_dim != 3:
            raise FlowContractError(
                "from_tensor_mesh requires a 3-D (z, x, y) TensorMesh",
                object_name="CartGrid.from_tensor_mesh",
                field="mesh",
                expected="n_dim == 3",
                actual=mesh.n_dim,
            )
        nz, nx, ny = mesh.shape
        widths_z, widths_x, widths_y = mesh.cell_widths
        origin_z, origin_x, origin_y = mesh.origin
        return cls(
            nx,
            ny,
            nz,
            widths_x,
            widths_y,
            widths_z,
            origin_m=(origin_x, origin_y, origin_z),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    def to_tensor_mesh(self) -> TensorMesh:
        """Map immutable SI metadata to a core ``(z, x, y)`` TensorMesh."""

        origin_x, origin_y, origin_z = self.origin_m
        return TensorMesh(
            shape=(self.nz, self.nx, self.ny),
            cell_widths=(
                self._dz_m.detach().cpu().to(torch.float64),
                self._dx_m.detach().cpu().to(torch.float64),
                self._dy_m.detach().cpu().to(torch.float64),
            ),
            origin=(origin_z, origin_x, origin_y),
        )


__all__ = ["CartGrid", "ConnList", "FlowGrid"]
