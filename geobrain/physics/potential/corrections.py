"""Strict-SI gravity corrections with explicit transformation signs.

Free-air and terrain functions return conventional positive terms to add to
observed gravity; Bouguer functions return the positive mass-attraction
magnitude to subtract. Formula units are never mixed with display units:
every result is in ``m/s²``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import overload

import torch

from geobrain.core import ErrorCode

from ._engine import build_sensitivity
from ._engine.plan import PrismKernelPlan
from .config import PotentialExecutionConfig
from .errors import PotentialContractError, PotentialNumericsError


from .helpers import G_SI as _GRAVITATIONAL_CONSTANT_SI  # single-sourced (identical value)
FREE_AIR_GRADIENT_M_PER_S2_PER_M: float = 3.086e-6
_SUPPORTED_DTYPES = frozenset({torch.float32, torch.float64})


def _error(
    message: str,
    *,
    field: str,
    expected: object,
    actual: object,
    hint: str,
    code: ErrorCode | None = None,
) -> PotentialContractError:
    return PotentialContractError(
        message,
        object_name="gravity_correction",
        field=field,
        expected=expected,
        actual=actual,
        hint=hint,
        code=code,
    )


def _validate_python_scalar(
    value: object,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise _error(
            f"{field} must be a finite Python float.",
            field=field,
            expected="finite float; integers and booleans are forbidden",
            actual=value,
            hint=f"Provide {field} explicitly in SI units as a float.",
        )
    if positive and value <= 0.0:
        raise _error(
            f"{field} must be positive.",
            field=field,
            expected="> 0",
            actual=value,
            hint=f"Provide a positive SI value for {field}.",
        )
    if nonnegative and value < 0.0:
        raise _error(
            f"{field} cannot be negative.",
            field=field,
            expected=">= 0",
            actual=value,
            hint=f"Provide a non-negative SI value for {field}.",
        )
    return value


def _validate_tensor(
    value: object,
    *,
    field: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _error(
            f"{field} must be a torch.Tensor.",
            field=field,
            expected="finite strided float32 or float64 tensor",
            actual=value,
            hint=f"Provide {field} as an SI tensor without implicit conversion.",
        )
    if value.layout != torch.strided:
        raise _error(
            f"{field} requires a strided tensor layout.",
            field=field,
            expected="torch.strided",
            actual=value.layout,
            hint=f"Materialize {field} as a strided tensor.",
        )
    if ndim is not None and value.ndim != ndim:
        raise _error(
            f"{field} has the wrong rank.",
            field=field,
            expected=f"{ndim}-D tensor",
            actual=list(value.shape),
            hint=f"Reshape {field} to the documented rank.",
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if shape is not None and tuple(value.shape) != shape:
        raise _error(
            f"{field} has the wrong shape.",
            field=field,
            expected=list(shape),
            actual=list(value.shape),
            hint=f"Match {field} to the topographic edge grid.",
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if value.numel() == 0:
        raise _error(
            f"{field} cannot be empty.",
            field=field,
            expected="at least one value",
            actual=list(value.shape),
            hint=f"Provide non-empty {field} data.",
        )
    if value.dtype not in _SUPPORTED_DTYPES:
        raise _error(
            f"{field} dtype is unsupported.",
            field=field,
            expected=["torch.float32", "torch.float64"],
            actual=value.dtype,
            hint=f"Use float32 or float64 for {field} without implicit casting.",
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )
    if value.device.type not in {"cpu", "cuda"}:
        raise _error(
            f"{field} device is unsupported.",
            field=field,
            expected=["cpu", "cuda"],
            actual=value.device,
            hint=f"Move {field} explicitly to CPU or an available CUDA device.",
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    if not bool(torch.isfinite(value).all()):
        raise _error(
            f"{field} must be finite.",
            field=field,
            expected="all finite SI values",
            actual="contains non-finite values",
            hint=f"Replace NaN or infinite values in {field}.",
        )
    if nonnegative and not bool((value >= 0).all()):
        raise _error(
            f"{field} cannot be negative.",
            field=field,
            expected="all values >= 0",
            actual="contains negative values",
            hint=f"Provide non-negative SI values for {field}.",
        )
    if positive and not bool((value > 0).all()):
        raise _error(
            f"{field} must be positive.",
            field=field,
            expected="all values > 0",
            actual="contains zero or negative values",
            hint=f"Provide positive SI values for {field}.",
        )
    return value


@overload
def _checked_correction_result(value: float, *, field: str) -> float: ...


@overload
def _checked_correction_result(value: torch.Tensor, *, field: str) -> torch.Tensor: ...


def _checked_correction_result(
    value: float | torch.Tensor,
    *,
    field: str,
) -> float | torch.Tensor:
    finite = (
        bool(torch.isfinite(value).all())
        if isinstance(value, torch.Tensor)
        else math.isfinite(value)
    )
    if not finite:
        actual: object
        if isinstance(value, torch.Tensor):
            actual = {"shape": list(value.shape), "dtype": value.dtype, "device": value.device}
        else:
            actual = {"status": "non-finite scalar result"}
        raise PotentialNumericsError(
            "Gravity correction produced a non-finite result after valid input checks.",
            object_name="gravity_correction",
            field=field,
            expected="finite SI correction",
            actual=actual,
            code=ErrorCode.EXECUTION_FAILED,
            hint="Use representable SI magnitudes or a higher precision dtype.",
        )
    return value


@overload
def free_air_correction(*, elevation_m: float) -> float: ...


@overload
def free_air_correction(*, elevation_m: torch.Tensor) -> torch.Tensor: ...


def free_air_correction(*, elevation_m: float | torch.Tensor) -> float | torch.Tensor:
    """Return the free-air correction to add, in ``m/s²``.

    Elevation is positive upward, so positive elevations receive a positive
    conventional reduction of ``3.086e-6 m/s²`` per metre.
    """
    if isinstance(elevation_m, torch.Tensor):
        elevation = _validate_tensor(elevation_m, field="elevation_m")
        return elevation * FREE_AIR_GRADIENT_M_PER_S2_PER_M
    elevation_value = _validate_python_scalar(elevation_m, field="elevation_m")
    return FREE_AIR_GRADIENT_M_PER_S2_PER_M * elevation_value


@overload
def bouguer_slab(*, density_kg_per_m3: float, thickness_m: float) -> float: ...


@overload
def bouguer_slab(
    *,
    density_kg_per_m3: torch.Tensor,
    thickness_m: float | torch.Tensor,
) -> torch.Tensor: ...


@overload
def bouguer_slab(  # type: ignore[overload-cannot-match, unused-ignore]
    *,
    density_kg_per_m3: float,
    thickness_m: torch.Tensor,
) -> torch.Tensor: ...


def bouguer_slab(
    *,
    density_kg_per_m3: float | torch.Tensor,
    thickness_m: float | torch.Tensor,
) -> float | torch.Tensor:
    """Return the positive infinite-slab magnitude ``2πGρh`` in ``m/s²``.

    Args:
        density_kg_per_m3: slab density [kg/m^3].
        thickness_m: slab thickness [m].
    """
    if not isinstance(density_kg_per_m3, torch.Tensor) and not isinstance(
        thickness_m, torch.Tensor
    ):
        density = _validate_python_scalar(
            density_kg_per_m3,
            field="density_kg_per_m3",
            positive=True,
        )
        thickness = _validate_python_scalar(
            thickness_m,
            field="thickness_m",
            nonnegative=True,
        )
        scalar_result = 2.0 * math.pi * _GRAVITATIONAL_CONSTANT_SI * density * thickness
        return _checked_correction_result(scalar_result, field="bouguer_slab")

    reference = density_kg_per_m3 if isinstance(density_kg_per_m3, torch.Tensor) else thickness_m
    if not isinstance(reference, torch.Tensor):  # pragma: no cover - guarded upstream
        raise AssertionError("correction reference resolution lost its tensor")
    reference = _validate_tensor(reference, field="correction_reference")
    if isinstance(density_kg_per_m3, torch.Tensor):
        density_tensor = _validate_tensor(
            density_kg_per_m3,
            field="density_kg_per_m3",
            positive=True,
        )
    else:
        density = _validate_python_scalar(
            density_kg_per_m3,
            field="density_kg_per_m3",
            positive=True,
        )
        density_tensor = torch.as_tensor(density, dtype=reference.dtype, device=reference.device)
    if isinstance(thickness_m, torch.Tensor):
        thickness_tensor = _validate_tensor(
            thickness_m,
            field="thickness_m",
            nonnegative=True,
        )
    else:
        thickness = _validate_python_scalar(
            thickness_m,
            field="thickness_m",
            nonnegative=True,
        )
        thickness_tensor = torch.as_tensor(
            thickness,
            dtype=reference.dtype,
            device=reference.device,
        )
    if (
        density_tensor.dtype != thickness_tensor.dtype
        or density_tensor.device != thickness_tensor.device
    ):
        raise _error(
            "Bouguer inputs must share exact dtype and device.",
            field="density_kg_per_m3/thickness_m",
            expected={"dtype": thickness_tensor.dtype, "device": thickness_tensor.device},
            actual={"dtype": density_tensor.dtype, "device": density_tensor.device},
            hint="Move and cast Bouguer tensors explicitly before evaluation.",
        )
    tensor_result = (
        2.0 * math.pi * _GRAVITATIONAL_CONSTANT_SI * density_tensor * thickness_tensor
    )
    return _checked_correction_result(tensor_result, field="bouguer_slab")


def bouguer_spherical_cap(
    *,
    density_kg_per_m3: float,
    thickness_m: float,
    earth_radius_m: float,
    cap_surface_radius_m: float,
) -> float:
    """Return finite spherical-cap attraction in ``m/s²``.

    ``earth_radius_m`` is the mean-sea-level radius, ``thickness_m`` is the
    station elevation and cap thickness, and ``cap_surface_radius_m`` is the
    sea-level arc length from the station axis to the cap edge.  Subtracting
    the infinite-slab value from this result gives the signed Bullard-B
    correction.  The standard Hayford-Bowie outer radius is ``166_735 m``.

    The closed form integrates the vertical attraction of a uniform spherical
    shell sector.  It is algebraically equivalent to the finite-cap solution
    in LaFehr (1991), *An exact solution for the gravity curvature (Bullard B)
    correction*, Geophysics 56(8), 1179-1184.

    Args:
        density_kg_per_m3: cap density [kg/m^3].
        thickness_m: cap thickness [m].
        earth_radius_m: reference Earth radius [m].
        cap_surface_radius_m: cap surface radius [m].
    """
    density = _validate_python_scalar(
        density_kg_per_m3,
        field="density_kg_per_m3",
        positive=True,
    )
    thickness = _validate_python_scalar(
        thickness_m,
        field="thickness_m",
        nonnegative=True,
    )
    radius = _validate_python_scalar(
        earth_radius_m,
        field="earth_radius_m",
        positive=True,
    )
    cap_surface_radius = _validate_python_scalar(
        cap_surface_radius_m,
        field="cap_surface_radius_m",
        positive=True,
    )
    if thickness >= radius:
        raise _error(
            "Spherical-cap thickness must be smaller than Earth radius.",
            field="thickness_m",
            expected=f"< {radius}",
            actual=thickness,
            hint="Use a physical layer thickness smaller than earth_radius_m.",
        )
    if cap_surface_radius >= math.pi * radius:
        raise _error(
            "Spherical-cap surface radius must define less than a full great-circle arc.",
            field="cap_surface_radius_m",
            expected=f"< {math.pi * radius}",
            actual=cap_surface_radius,
            hint="Provide a positive finite cap arc length smaller than pi * earth_radius_m.",
        )
    if thickness == 0.0:
        return 0.0

    half_angle = cap_surface_radius / radius
    if not 0.0 < half_angle < math.pi:
        raise _error(
            "Spherical-cap angle is not representable in the open physical interval.",
            field="cap_surface_radius_m",
            expected="a representable arc angle strictly between 0 and pi radians",
            actual=cap_surface_radius,
            hint="Increase an underflow-scale cap radius or reduce it below pi * earth_radius_m.",
        )
    cosine = math.cos(half_angle)
    sine = math.sin(half_angle)

    # The primitive difference below loses every significant bit when h/R is
    # below roughly sqrt(machine epsilon), and R + h may equal R exactly.  In
    # that regime the first radial-order term is both stable and more accurate:
    # F'(1) = 1 + sin(alpha / 2), with the omitted relative term O(h / R).
    relative_thickness = thickness / radius
    if relative_thickness <= math.sqrt(math.ulp(1.0)):
        attraction_length = thickness * (1.0 + math.sin(half_angle / 2.0))
    else:
        station_radius = radius + thickness
        inner_radius_ratio = radius / station_radius

        def dimensionless_primitive(radial_ratio: float) -> float:
            axial_offset = radial_ratio - cosine
            distance = math.hypot(axial_offset, sine)
            return (
                radial_ratio**3 / 3.0
                + radial_ratio * radial_ratio * distance
                - 2.0 * distance**3 / 3.0
                - cosine
                * (
                    axial_offset * distance
                    + sine * sine * math.asinh(axial_offset / sine)
                )
            )

        attraction_length = station_radius * (
            dimensionless_primitive(1.0) - dimensionless_primitive(inner_radius_ratio)
        )
    result = 2.0 * math.pi * _GRAVITATIONAL_CONSTANT_SI * density * attraction_length
    return _checked_correction_result(result, field="bouguer_spherical_cap")


def _validate_terrain_inputs(
    *,
    observations_m: object,
    topo_x_edges_m: object,
    topo_y_edges_m: object,
    topo_elevation_m: object,
    density_kg_per_m3: object,
    execution: object,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    PotentialExecutionConfig,
]:
    observations = _validate_tensor(observations_m, field="observations_m", ndim=2)
    if observations.shape[1] != 3:
        raise _error(
            "Terrain observations must have Cartesian x,y,z columns.",
            field="observations_m",
            expected=["n>0", 3],
            actual=list(observations.shape),
            hint="Provide elevation-positive-up station coordinates in metres.",
            code=ErrorCode.SHAPE_MISMATCH,
        )
    x_edges = _validate_tensor(topo_x_edges_m, field="topo_x_edges_m", ndim=1)
    y_edges = _validate_tensor(topo_y_edges_m, field="topo_y_edges_m", ndim=1)
    if x_edges.numel() < 2 or y_edges.numel() < 2:
        raise _error(
            "Terrain edge vectors require at least two entries.",
            field="topo_edges_m",
            expected="at least two strictly increasing values per axis",
            actual={"x": x_edges.numel(), "y": y_edges.numel()},
            hint="Provide at least one terrain cell on each horizontal axis.",
        )
    if not bool((x_edges[1:] > x_edges[:-1]).all()) or not bool((y_edges[1:] > y_edges[:-1]).all()):
        raise _error(
            "Terrain edges must be strictly increasing.",
            field="topo_edges_m",
            expected="strictly increasing x and y edges",
            actual="contains equal or reversed edges",
            hint="Sort and deduplicate terrain grid edges before correction.",
        )
    topography = _validate_tensor(
        topo_elevation_m,
        field="topo_elevation_m",
        ndim=2,
        shape=(y_edges.numel() - 1, x_edges.numel() - 1),
    )
    tensors = (observations, x_edges, y_edges, topography)
    if any(tensor.dtype != observations.dtype for tensor in tensors[1:]) or any(
        tensor.device != observations.device for tensor in tensors[1:]
    ):
        raise _error(
            "Terrain geometry must share exact dtype and device.",
            field="terrain_geometry",
            expected={"dtype": observations.dtype, "device": observations.device},
            actual=[{"dtype": tensor.dtype, "device": tensor.device} for tensor in tensors],
            hint="Move and cast all terrain geometry explicitly before correction.",
        )
    if isinstance(density_kg_per_m3, torch.Tensor):
        density = _validate_tensor(
            density_kg_per_m3,
            field="density_kg_per_m3",
            positive=True,
        )
        if density.ndim != 0:
            raise _error(
                "Terrain density tensor must be scalar.",
                field="density_kg_per_m3",
                expected="0-D tensor",
                actual=list(density.shape),
                hint="Provide one scalar terrain density in kg/m^3.",
                code=ErrorCode.SHAPE_MISMATCH,
            )
        if density.dtype != observations.dtype or density.device != observations.device:
            raise _error(
                "Terrain density must share geometry dtype and device.",
                field="density_kg_per_m3",
                expected={"dtype": observations.dtype, "device": observations.device},
                actual={"dtype": density.dtype, "device": density.device},
                hint="Move and cast density explicitly before correction.",
            )
    else:
        density_value = _validate_python_scalar(
            density_kg_per_m3,
            field="density_kg_per_m3",
            positive=True,
        )
        density = torch.as_tensor(
            density_value,
            dtype=observations.dtype,
            device=observations.device,
        )
    if not isinstance(execution, PotentialExecutionConfig):
        raise _error(
            "Terrain execution must be PotentialExecutionConfig.",
            field="execution",
            expected="PotentialExecutionConfig",
            actual=execution,
            hint="Provide an explicit bounded execution policy.",
        )
    return observations, x_edges, y_edges, topography, density, execution


def terrain_correction_prism(
    *,
    observations_m: torch.Tensor,
    topo_x_edges_m: torch.Tensor,
    topo_y_edges_m: torch.Tensor,
    topo_elevation_m: torch.Tensor,
    density_kg_per_m3: float | torch.Tensor,
    execution: PotentialExecutionConfig,
) -> torch.Tensor:
    """Return positive Hammer terrain-correction magnitudes in ``m/s²``.

    Each terrain cell is the elevation difference between the station plane
    and the actual topographic surface.  The cell containing a station's
    horizontal projection is omitted because its prism touches the receiver;
    a finite inner-zone model must be supplied by a higher-resolution terrain
    representation instead of silently regularizing the singular geometry.

    Args:
        observations_m: ``(n, 3)`` station coordinates [m].
        topo_x_edges_m / topo_y_edges_m: topography cell edges [m].
        topo_elevation_m: topography elevations per cell [m].
        density_kg_per_m3: correction density [kg/m^3].
        execution: tiling/budget policy for the prism sums.
    """
    observations, x_edges, y_edges, topography, density, execution = _validate_terrain_inputs(
        observations_m=observations_m,
        topo_x_edges_m=topo_x_edges_m,
        topo_y_edges_m=topo_y_edges_m,
        topo_elevation_m=topo_elevation_m,
        density_kg_per_m3=density_kg_per_m3,
        execution=execution,
    )
    ny, nx = topography.shape
    x_min = x_edges[:-1].repeat(ny)
    x_max = x_edges[1:].repeat(ny)
    y_min = y_edges[:-1].repeat_interleave(nx)
    y_max = y_edges[1:].repeat_interleave(nx)
    elevations = topography.reshape(-1)
    corrections: list[torch.Tensor] = []
    for observation in observations:
        station_z = observation[2]
        active = elevations != station_z
        contains_horizontal_projection = (
            (observation[0] >= x_min)
            & (observation[0] <= x_max)
            & (observation[1] >= y_min)
            & (observation[1] <= y_max)
        )
        active = active & ~contains_horizontal_projection
        if not bool(active.any()):
            corrections.append(
                torch.zeros((), dtype=observations.dtype, device=observations.device)
            )
            continue
        active_elevations = elevations[active]
        station_plane = torch.full_like(active_elevations, station_z)
        bounds = torch.stack(
            (
                x_min[active],
                x_max[active],
                y_min[active],
                y_max[active],
                torch.minimum(station_plane, active_elevations),
                torch.maximum(station_plane, active_elevations),
            ),
            dim=1,
        )
        plan = PrismKernelPlan.build(
            observations_m=observation.unsqueeze(0),
            cell_bounds_m=bounds,
            components=("gz",),
            dtype=observations.dtype,
            device=observations.device,
        )
        sensitivity = build_sensitivity(
            plan=plan,
            execution=execution,
            model_contract="rho",
            projection=None,
        )
        # Hill prisms are real positive mass above the station plane. Valley
        # prisms represent slab mass that is absent below the station plane,
        # so they carry negative equivalent density. Both therefore produce
        # positive Hammer corrections before summation; taking abs only after
        # summation would incorrectly cancel mixed hill/valley cells.
        relative_elevation_sign = torch.sign(active_elevations - station_plane)
        density_model = relative_elevation_sign * density
        raw_vertical = sensitivity.apply({"rho": density_model})["gz"][0]
        corrections.append(raw_vertical)
    return torch.stack(corrections)


__all__ = [
    "FREE_AIR_GRADIENT_M_PER_S2_PER_M",
    "bouguer_slab",
    "bouguer_spherical_cap",
    "free_air_correction",
    "terrain_correction_prism",
]
