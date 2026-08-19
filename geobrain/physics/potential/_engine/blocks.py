"""Authoritative unit-property SI kernels for rectangular-prism blocks.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core import ErrorCode

from ..errors import (
    PotentialCapabilityError,
    PotentialContractError,
    PotentialNumericsError,
)


from ..helpers import G_SI as _GRAVITATIONAL_CONSTANT_SI  # single-sourced (identical value)
_VACUUM_PERMEABILITY_OVER_FOUR_PI_SI = 1.0e-7
_AXES = "xyz"
_OTHER_AXES = ((1, 2), (0, 2), (0, 1))
_GRAVITY_COMPONENTS = frozenset(
    {
        "gx",
        "gy",
        "gz",
        "gxx",
        "gxy",
        "gxz",
        "gyx",
        "gyy",
        "gyz",
        "gzx",
        "gzy",
        "gzz",
    }
)
_MAGNETIC_COMPONENTS = frozenset(
    f"b{field_axis}_m{magnetization_axis}" for field_axis in _AXES for magnetization_axis in _AXES
)
_MAGNETIC_GRADIENT_COMPONENTS = frozenset(
    f"db{field_axis}_d{coordinate_axis}_m{magnetization_axis}"
    for field_axis in _AXES
    for coordinate_axis in _AXES
    for magnetization_axis in _AXES
)
_SUPPORTED_COMPONENTS = _GRAVITY_COMPONENTS | _MAGNETIC_COMPONENTS | _MAGNETIC_GRADIENT_COMPONENTS
_SUPPORTED_DTYPES = frozenset({torch.float32, torch.float64})
_SUPPORTED_DEVICE_TYPES = frozenset({"cpu", "cuda"})


def _validate_components(components: object) -> tuple[str, ...]:
    if not isinstance(components, tuple) or not components:
        raise PotentialContractError(
            "Prism block components must be a non-empty tuple.",
            object_name="evaluate_prism_block",
            field="components",
            expected="non-empty ordered tuple of unique internal component names",
            actual=components,
            hint="Pass internal gravity, magnetic-basis, or magnetic-gradient-basis names.",
        )
    if any(not isinstance(component, str) or not component for component in components):
        raise PotentialContractError(
            "Prism block component names must be non-empty strings.",
            object_name="evaluate_prism_block",
            field="components",
            expected="non-empty string component names",
            actual=components,
            hint="Replace blank or non-string entries with supported internal names.",
        )
    if len(set(components)) != len(components):
        raise PotentialContractError(
            "Prism block components must be unique.",
            object_name="evaluate_prism_block",
            field="components",
            expected="unique ordered internal component names",
            actual=components,
            hint="Remove duplicate names while preserving request order.",
        )
    unsupported = tuple(
        component for component in components if component not in _SUPPORTED_COMPONENTS
    )
    if unsupported:
        raise PotentialCapabilityError(
            "Prism block components contain unsupported or public projection names.",
            object_name="evaluate_prism_block",
            field="components",
            expected=sorted(_SUPPORTED_COMPONENTS),
            actual=unsupported,
            hint="Resolve public magnetic projections to internal Cartesian bases before evaluation.",
        )
    return components


def _validate_tensor(
    value: object,
    *,
    field_name: str,
    width: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PotentialContractError(
            "Prism block geometry must be a torch.Tensor.",
            object_name="evaluate_prism_block",
            field=field_name,
            expected=f"non-empty (n, {width}) strided float32 or float64 tensor",
            actual=value,
            hint=f"Provide {field_name} with exactly {width} columns.",
        )
    if value.layout != torch.strided:
        raise PotentialCapabilityError(
            "Prism block geometry requires a strided tensor layout.",
            object_name="evaluate_prism_block",
            field=field_name,
            expected="torch.strided",
            actual=value.layout,
            hint="Materialize geometry in a strided tensor before evaluation.",
        )
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != width:
        raise PotentialContractError(
            "Prism block geometry has the wrong shape.",
            object_name="evaluate_prism_block",
            field=field_name,
            expected=["n>0", width],
            actual=list(value.shape),
            code=ErrorCode.SHAPE_MISMATCH,
            hint=f"Provide a non-empty tensor shaped (n, {width}).",
        )
    if value.dtype not in _SUPPORTED_DTYPES:
        raise PotentialCapabilityError(
            "Prism block geometry dtype is unsupported.",
            object_name="evaluate_prism_block",
            field=field_name,
            expected=["torch.float32", "torch.float64"],
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Use float32 or float64 geometry without implicit casting.",
        )
    if value.device.type not in _SUPPORTED_DEVICE_TYPES:
        raise PotentialCapabilityError(
            "Prism block geometry device type is unsupported.",
            object_name="evaluate_prism_block",
            field=field_name,
            expected=["cpu", "cuda"],
            actual=value.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Use CPU or an available CUDA device.",
        )
    if not bool(torch.isfinite(value).all()):
        raise PotentialContractError(
            "Prism block geometry must be finite.",
            object_name="evaluate_prism_block",
            field=field_name,
            expected="all finite values",
            actual="contains non-finite values",
            hint="Replace NaN and infinite coordinates before evaluation.",
        )
    return value


def _validate_geometry(
    observations_m: object, cell_bounds_m: object
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = _validate_tensor(observations_m, field_name="observations_m", width=3)
    bounds = _validate_tensor(cell_bounds_m, field_name="cell_bounds_m", width=6)
    if observations.dtype != bounds.dtype:
        raise PotentialContractError(
            "Prism block geometry tensors must have the same dtype.",
            object_name="evaluate_prism_block",
            field="cell_bounds_m",
            expected=observations.dtype,
            actual=bounds.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Construct observations and bounds in one exact supported dtype.",
        )
    if observations.device != bounds.device:
        raise PotentialContractError(
            "Prism block geometry tensors must be on the same device.",
            object_name="evaluate_prism_block",
            field="cell_bounds_m",
            expected=observations.device,
            actual=bounds.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Move geometry explicitly to one CPU or CUDA device before evaluation.",
        )
    lower = bounds[:, (0, 2, 4)]
    upper = bounds[:, (1, 3, 5)]
    if not bool((lower < upper).all()):
        raise PotentialContractError(
            "Every prism lower bound must be strictly below its upper bound.",
            object_name="evaluate_prism_block",
            field="cell_bounds_m",
            expected="xmin<xmax, ymin<ymax, and zmin<zmax for every cell",
            actual="contains equal or reversed bounds",
            hint="Correct ordered (xmin,xmax,ymin,ymax,zmin,zmax) cell bounds.",
        )
    inside_closed = (
        (observations[:, None, 0] >= bounds[None, :, 0])
        & (observations[:, None, 0] <= bounds[None, :, 1])
        & (observations[:, None, 1] >= bounds[None, :, 2])
        & (observations[:, None, 1] <= bounds[None, :, 3])
        & (observations[:, None, 2] >= bounds[None, :, 4])
        & (observations[:, None, 2] <= bounds[None, :, 5])
    )
    if bool(inside_closed.any()):
        first_pair = torch.nonzero(inside_closed, as_tuple=False)[0]
        raise PotentialContractError(
            "Every observation must be strictly outside every source prism.",
            object_name="evaluate_prism_block",
            field="observations_m",
            expected="observations outside all closed prism bounds",
            actual={
                "observation_index": int(first_pair[0].item()),
                "cell_index": int(first_pair[1].item()),
            },
            hint="Move the observation away from the prism interior and singular boundary.",
        )
    return observations, bounds


def _stable_log_plus(
    primary: torch.Tensor,
    transverse_first: torch.Tensor,
    transverse_second: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``log(radius + primary)`` without subtractive cancellation.

    On a negative-axis extension the omitted zero-radius logarithm is common
    to both bounds and cancels in the alternating corner sum.
    """
    positive = primary >= 0
    if bool(positive.all()):
        return torch.log(radius + primary)

    def negative_branch() -> torch.Tensor:
        transverse_square = transverse_first.square() + transverse_second.square()
        safe_transverse_square = torch.where(
            transverse_square == 0, torch.ones_like(radius), transverse_square
        )
        value = torch.log(safe_transverse_square)
        del safe_transverse_square, transverse_square
        negative_log_denominator = torch.log(radius - primary)
        value.sub_(negative_log_denominator)
        return value

    if bool((~positive).all()):
        return negative_branch()

    negative_value = negative_branch()
    positive_argument = torch.where(positive, radius + primary, torch.ones_like(radius))
    positive_value = torch.log(positive_argument)
    del positive_argument
    return torch.where(positive, positive_value, negative_value)


def _stable_log_derivative(
    numerator: torch.Tensor,
    primary: torch.Tensor,
    other_transverse: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``numerator / (radius * (radius + primary))`` stably."""
    positive = primary >= 0
    positive_denominator = torch.where(
        positive, radius * (radius + primary), torch.ones_like(radius)
    )
    transverse_square = numerator.square() + other_transverse.square()
    regular_negative = (~positive) & (transverse_square != 0)
    negative_denominator = torch.where(
        regular_negative, radius * transverse_square, torch.ones_like(radius)
    )
    positive_value = numerator / positive_denominator
    negative_value = numerator * (radius - primary) / negative_denominator
    negative_value = torch.where(regular_negative, negative_value, torch.zeros_like(radius))
    return torch.where(positive, positive_value, negative_value)


def _corner_sum(value: torch.Tensor) -> torch.Tensor:
    """Reduce eight corner views without allocating a weighted corner block."""
    return (
        value[:, :, 0, 0, 0]
        - value[:, :, 0, 0, 1]
        - value[:, :, 0, 1, 0]
        + value[:, :, 0, 1, 1]
        - value[:, :, 1, 0, 0]
        + value[:, :, 1, 0, 1]
        + value[:, :, 1, 1, 0]
        - value[:, :, 1, 1, 1]
    )


def _sorted_third_key(first: int, second: int, third: int) -> tuple[int, int, int]:
    ordered = sorted((first, second, third))
    return ordered[0], ordered[1], ordered[2]


def _requested_dependencies(
    requested: tuple[str, ...],
) -> tuple[
    dict[str, int],
    dict[str, tuple[int, int]],
    dict[str, tuple[int, int, int]],
]:
    accelerations: dict[str, int] = {}
    second_derivatives: dict[str, tuple[int, int]] = {}
    third_derivatives: dict[str, tuple[int, int, int]] = {}
    for component in requested:
        if len(component) == 2:
            accelerations[component] = _AXES.index(component[1])
        elif component.startswith("g"):
            first = _AXES.index(component[1])
            second = _AXES.index(component[2])
            second_derivatives[component] = min(first, second), max(first, second)
        elif component.startswith("b"):
            first = _AXES.index(component[1])
            second = _AXES.index(component[4])
            second_derivatives[component] = min(first, second), max(first, second)
        else:
            first = _AXES.index(component[2])
            second = _AXES.index(component[5])
            third = _AXES.index(component[8])
            third_derivatives[component] = _sorted_third_key(first, second, third)
    return accelerations, second_derivatives, third_derivatives


def _checked_kernel(component: str, value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise PotentialNumericsError(
            "Prism block evaluation produced a non-finite selected kernel.",
            object_name="evaluate_prism_block",
            field=component,
            expected="finite unit-property SI kernel values",
            actual={
                "status": "contains non-finite values",
                "shape": list(value.shape),
                "dtype": value.dtype,
                "device": value.device,
            },
            code=ErrorCode.EXECUTION_FAILED,
            hint=(
                "Use float64 where appropriate and translate the same SI geometry "
                "to a numerically representable local coordinate frame; avoid "
                "extreme length scales without clamping or unit changes."
            ),
        )
    return value


def evaluate_prism_block(
    observations_m: torch.Tensor,
    cell_bounds_m: torch.Tensor,
    components: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Evaluate requested unit-property SI kernels for one prism block.

    Results are shaped ``(n_observations, n_cells)``. Gravity acceleration
    kernels are m/s² per kg/m³, gravity tensors are s⁻² per kg/m³,
    magnetic bases are T per A/m, and magnetic-gradient bases are
    T/m per A/m.
    """
    requested = _validate_components(components)
    observations, bounds = _validate_geometry(observations_m, cell_bounds_m)
    acceleration_dependencies, second_dependencies, third_dependencies = _requested_dependencies(
        requested
    )
    acceleration_axes = sorted(set(acceleration_dependencies.values()))
    second_keys = sorted(set(second_dependencies.values()))
    third_keys = sorted(set(third_dependencies.values()))

    x_distance = observations[:, None, 0, None] - bounds[None, :, (0, 1)]
    y_distance = observations[:, None, 1, None] - bounds[None, :, (2, 3)]
    z_distance = observations[:, None, 2, None] - bounds[None, :, (4, 5)]
    x = x_distance[:, :, :, None, None]
    y = y_distance[:, :, None, :, None]
    z = z_distance[:, :, None, None, :]
    radius = torch.sqrt(x.square() + y.square() + z.square())
    coordinates = (x, y, z)
    log_cache: dict[int, torch.Tensor] = {}
    atan_cache: dict[int, torch.Tensor] = {}
    second_cache: dict[tuple[int, int], torch.Tensor] = {}
    third_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    log_uses = {axis: 0 for axis in range(3)}
    atan_uses = {axis: 0 for axis in range(3)}
    for axis in acceleration_axes:
        first_transverse, second_transverse = _OTHER_AXES[axis]
        log_uses[first_transverse] += 1
        log_uses[second_transverse] += 1
        atan_uses[axis] += 1
    for first, second in second_keys:
        if first == second:
            atan_uses[first] += 1
        else:
            log_uses[3 - first - second] += 1

    def log_term(axis: int) -> torch.Tensor:
        if axis not in log_cache:
            first_transverse, second_transverse = _OTHER_AXES[axis]
            log_cache[axis] = _stable_log_plus(
                coordinates[axis],
                coordinates[first_transverse],
                coordinates[second_transverse],
                radius,
            )
        return log_cache[axis]

    def atan_term(axis: int) -> torch.Tensor:
        if axis not in atan_cache:
            first_transverse, second_transverse = _OTHER_AXES[axis]
            atan_cache[axis] = torch.atan2(
                coordinates[first_transverse] * coordinates[second_transverse],
                coordinates[axis] * radius,
            )
        return atan_cache[axis]

    def consume_log(axis: int) -> torch.Tensor:
        value = log_term(axis)
        log_uses[axis] -= 1
        if log_uses[axis] == 0:
            del log_cache[axis]
        return value

    def consume_atan(axis: int) -> torch.Tensor:
        value = atan_term(axis)
        atan_uses[axis] -= 1
        if atan_uses[axis] == 0:
            del atan_cache[axis]
        return value

    def acceleration(axis: int) -> torch.Tensor:
        first_transverse, second_transverse = _OTHER_AXES[axis]
        geometric = _corner_sum(coordinates[first_transverse] * consume_log(second_transverse))
        geometric = geometric + _corner_sum(
            coordinates[second_transverse] * consume_log(first_transverse)
        )
        geometric = geometric - _corner_sum(coordinates[axis] * consume_atan(axis))
        return _GRAVITATIONAL_CONSTANT_SI * geometric

    def second_derivative(key: tuple[int, int]) -> torch.Tensor:
        if key not in second_cache:
            first, second = key
            if first == second:
                second_cache[key] = _corner_sum(-consume_atan(first))
            else:
                remaining_axis = 3 - first - second
                second_cache[key] = _corner_sum(consume_log(remaining_axis))
        return second_cache[key]

    def primitive_third_derivative(key: tuple[int, int, int]) -> torch.Tensor:
        if key in third_cache:
            return third_cache[key]
        if key == (0, 0, 1):
            value = _corner_sum(_stable_log_derivative(x, z, y, radius))
        elif key == (0, 0, 2):
            value = _corner_sum(_stable_log_derivative(x, y, z, radius))
        elif key == (0, 1, 1):
            value = _corner_sum(_stable_log_derivative(y, z, x, radius))
        elif key == (0, 1, 2):
            value = _corner_sum(torch.reciprocal(radius))
        elif key == (0, 2, 2):
            value = _corner_sum(_stable_log_derivative(z, y, x, radius))
        elif key == (1, 1, 2):
            value = _corner_sum(_stable_log_derivative(y, x, z, radius))
        else:
            value = _corner_sum(_stable_log_derivative(z, x, y, radius))
        third_cache[key] = value
        return value

    def third_derivative(key: tuple[int, int, int]) -> torch.Tensor:
        if key == (0, 0, 0):
            return -(primitive_third_derivative((0, 1, 1)) + primitive_third_derivative((0, 2, 2)))
        if key == (1, 1, 1):
            return -(primitive_third_derivative((0, 0, 1)) + primitive_third_derivative((1, 2, 2)))
        if key == (2, 2, 2):
            return -(primitive_third_derivative((0, 0, 2)) + primitive_third_derivative((1, 1, 2)))
        return primitive_third_derivative(key)

    acceleration_values = {axis: acceleration(axis) for axis in acceleration_axes}
    second_values = {key: second_derivative(key) for key in second_keys}
    third_values = {key: third_derivative(key) for key in third_keys}

    result: dict[str, torch.Tensor] = {}
    for component in requested:
        if component in acceleration_dependencies:
            value = acceleration_values[acceleration_dependencies[component]]
        elif component in second_dependencies:
            value = second_values[second_dependencies[component]]
            if component.startswith("g"):
                value = _GRAVITATIONAL_CONSTANT_SI * value
            else:
                value = _VACUUM_PERMEABILITY_OVER_FOUR_PI_SI * value
        else:
            value = (
                _VACUUM_PERMEABILITY_OVER_FOUR_PI_SI * third_values[third_dependencies[component]]
            )
        result[component] = _checked_kernel(component, value)
    return result


__all__ = ["evaluate_prism_block"]
