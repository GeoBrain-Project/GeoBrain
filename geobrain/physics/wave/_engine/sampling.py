"""Dimension-parametric Wave source injection and receiver sampling.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..errors import WaveContractError
from .contracts import CompiledAcquisition


def _sampling_error(field: str, expected: object, actual: object) -> WaveContractError:
    """Build a structured dimension-parametric sampling error."""
    return WaveContractError(
        "invalid Wave sampling input",
        object_name="sampling",
        field=field,
        expected=expected,
        actual=actual,
    )


def _index_tuple(
    shot_index: torch.Tensor,
    cell_indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build same-device advanced indices for one packed operation."""
    return (
        shot_index.to(device=device),
        *(cell_indices[:, axis].to(device=device) for axis in range(cell_indices.shape[1])),
    )


def _requested_components(components: object) -> tuple[str, ...]:
    """Own and validate component names before mapping membership checks."""
    if isinstance(components, (str, bytes, bytearray)) or not isinstance(
        components, Sequence
    ):
        raise _sampling_error(
            "components", "unique non-empty string sequence", type(components).__name__
        )
    requested = tuple(components)
    if (
        not requested
        or any(type(component) is not str or not component for component in requested)
        or len(set(requested)) != len(requested)
    ):
        raise _sampling_error(
            "components", "unique non-empty string sequence", requested
        )
    return requested


def _validate_packed_bounds(
    *,
    shot_index: torch.Tensor,
    cell_indices: torch.Tensor,
    state_shape: tuple[int, ...],
    shot_field: str,
    cell_field: str,
) -> None:
    """Validate every packed shot/cell index before advanced indexing."""
    n_shot = state_shape[0]
    if bool(((shot_index < 0) | (shot_index >= n_shot)).any()):
        raise _sampling_error(
            shot_field, f"indices within [0, {n_shot})", shot_index.tolist()
        )
    for axis, size in enumerate(state_shape[1:]):
        axis_indices = cell_indices[:, axis]
        if bool(((axis_indices < 0) | (axis_indices >= size)).any()):
            raise _sampling_error(
                cell_field,
                f"axis {axis} indices within [0, {size})",
                f"out-of-bounds value on axis {axis}",
            )


def inject_sources(
    state: torch.Tensor,
    acquisition: CompiledAcquisition,
    amplitudes: torch.Tensor,
) -> torch.Tensor:
    """Add packed source amplitudes to shot-batched state."""
    if not isinstance(state, torch.Tensor):
        raise _sampling_error("state", "torch.Tensor", type(state).__name__)
    if not isinstance(acquisition, CompiledAcquisition):
        raise _sampling_error(
            "acquisition", "CompiledAcquisition", type(acquisition).__name__
        )
    if not isinstance(amplitudes, torch.Tensor):
        raise _sampling_error(
            "amplitudes", "torch.Tensor", type(amplitudes).__name__
        )
    spatial_rank = int(acquisition.source_indices.shape[1])
    if state.ndim != spatial_rank + 1 or state.shape[0] != acquisition.n_shot:
        raise _sampling_error(
            "state",
            f"shape (n_shot={acquisition.n_shot}, {spatial_rank} spatial axes)",
            tuple(state.shape),
        )
    if acquisition.source_indices.device != state.device:
        raise _sampling_error(
            "acquisition",
            f"source indices on state device {state.device}",
            acquisition.source_indices.device,
        )
    if (
        tuple(amplitudes.shape) != (acquisition.n_source,)
        or amplitudes.dtype is not state.dtype
        or amplitudes.device != state.device
    ):
        raise _sampling_error(
            "amplitudes",
            (
                f"shape ({acquisition.n_source},), dtype={state.dtype}, "
                f"device={state.device}"
            ),
            (
                tuple(amplitudes.shape),
                amplitudes.dtype,
                amplitudes.device,
            )
        )
    if state.device.type == "meta":
        raise _sampling_error(
            "state", "materialized execution tensor", state.device
        )
    _validate_packed_bounds(
        shot_index=acquisition.source_shot_index,
        cell_indices=acquisition.source_indices,
        state_shape=tuple(state.shape),
        shot_field="source_shot_index",
        cell_field="source_indices",
    )
    indices = _index_tuple(
        acquisition.source_shot_index,
        acquisition.source_indices,
        device=state.device,
    )
    return state.index_put(indices, amplitudes, accumulate=True)


def sample_receivers(
    states: Mapping[str, torch.Tensor],
    acquisition: CompiledAcquisition,
    *,
    components: Sequence[str],
) -> torch.Tensor:
    """Gather packed receiver traces in original order."""
    if not isinstance(acquisition, CompiledAcquisition):
        raise _sampling_error(
            "acquisition", "CompiledAcquisition", type(acquisition).__name__
        )
    if not isinstance(states, Mapping):
        raise _sampling_error("states", "mapping of component tensors", type(states).__name__)
    requested = _requested_components(components)
    validated: list[tuple[str, torch.Tensor]] = []
    reference: torch.Tensor | None = None
    for component in requested:
        if component not in states:
            raise _sampling_error(component, "component in state mapping", "missing")
        state = states[component]
        if not isinstance(state, torch.Tensor):
            raise _sampling_error(component, "torch.Tensor", type(state).__name__)
        spatial_rank = int(acquisition.receiver_indices.shape[1])
        if state.ndim != spatial_rank + 1 or state.shape[0] != acquisition.n_shot:
            raise _sampling_error(
                component,
                f"shape (n_shot={acquisition.n_shot}, {spatial_rank} spatial axes)",
                tuple(state.shape),
            )
        if acquisition.receiver_indices.device != state.device:
            raise _sampling_error(
                "acquisition",
                f"receiver indices on state device {state.device}",
                acquisition.receiver_indices.device,
            )
        if reference is None:
            reference = state
        elif (
            state.dtype is not reference.dtype
            or state.device != reference.device
            or tuple(state.shape) != tuple(reference.shape)
        ):
            raise _sampling_error(
                component,
                (
                    f"dtype={reference.dtype}, device={reference.device}, "
                    f"shape={tuple(reference.shape)}"
                ),
                (
                    f"dtype={state.dtype}, device={state.device}, "
                    f"shape={tuple(state.shape)}"
                ),
            )
        validated.append((component, state))
    assert reference is not None
    if reference.device.type == "meta":
        raise _sampling_error(
            "states", "materialized execution tensors", reference.device
        )
    _validate_packed_bounds(
        shot_index=acquisition.receiver_shot_index,
        cell_indices=acquisition.receiver_indices,
        state_shape=tuple(reference.shape),
        shot_field="receiver_shot_index",
        cell_field="receiver_indices",
    )
    indices = _index_tuple(
        acquisition.receiver_shot_index,
        acquisition.receiver_indices,
        device=reference.device,
    )
    gathered = [
        state[indices]
        for _, state in validated
    ]
    return torch.stack(gathered, dim=-1)


__all__ = ["inject_sources", "sample_receivers"]
