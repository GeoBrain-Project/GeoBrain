"""Immutable SI rock-property contract.

All live tensors must already share one floating dtype and device.  Unit
conversion belongs to :mod:`geobrain.physics.flow.adapters`, never this
production property kernel.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..errors import FlowContractError


def _floating_tensor(value: object, *, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise FlowContractError(
            f"{field} must be a floating tensor",
            object_name="Rock",
            field=field,
            expected="floating torch.Tensor in canonical SI units",
            actual=type(value).__name__,
        )
    if value.ndim != 1:
        raise FlowContractError(
            f"{field} must provide one value per cell",
            object_name="Rock",
            field=field,
            expected="shape [cell]",
            actual=tuple(value.shape),
        )
    if not bool(torch.isfinite(value).all()):
        raise FlowContractError(
            f"{field} must be finite",
            object_name="Rock",
            field=field,
            expected="finite values",
            actual="contains NaN or infinity",
        )
    return value


@dataclass(frozen=True, slots=True)
class Rock:
    """Per-cell canonical SI rock properties.

    Args:
        permeability_m2: Strictly positive permeability in square metres,
            shaped ``[cell]`` for isotropy or ``[cell, 3]`` for a diagonal
            tensor.
        porosity: Reference porosity fraction, strictly inside ``(0, 1)`` and
            shaped ``[cell]``.
        compressibility_pa_inv: Optional non-negative pore-volume
            compressibility in ``Pa⁻¹``, shaped ``[cell]``.
        reference_pressure_pa: Reference pressure in Pa, required exactly when
            compressibility is provided and shaped ``[cell]``.

    Input tensors are retained by identity so autograd connections are never
    hidden by an implicit copy, cast, or device transfer.
    """

    permeability_m2: torch.Tensor
    porosity: torch.Tensor
    compressibility_pa_inv: torch.Tensor | None = None
    reference_pressure_pa: torch.Tensor | None = None

    def __post_init__(self) -> None:
        permeability = self.permeability_m2
        if not isinstance(permeability, torch.Tensor) or not permeability.is_floating_point():
            raise FlowContractError(
                "permeability_m2 must be a floating tensor",
                object_name="Rock",
                field="permeability_m2",
                expected="floating [cell] or [cell, 3] tensor in m²",
                actual=type(permeability).__name__,
            )
        if permeability.ndim not in (1, 2) or (
            permeability.ndim == 2 and permeability.shape[1] != 3
        ):
            raise FlowContractError(
                "permeability_m2 has an invalid shape",
                object_name="Rock",
                field="permeability_m2",
                expected="[cell] or [cell, 3]",
                actual=tuple(permeability.shape),
            )
        if not bool(torch.isfinite(permeability).all()) or bool((permeability <= 0).any()):
            raise FlowContractError(
                "permeability_m2 must be positive and finite",
                object_name="Rock",
                field="permeability_m2",
                expected="> 0 m²",
                actual="contains a non-positive or non-finite value",
            )

        n_cells = permeability.shape[0]
        if isinstance(self.porosity, (int, float)) and not isinstance(self.porosity, bool):
            if not math.isfinite(float(self.porosity)):
                raise FlowContractError(
                    "porosity must be finite",
                    object_name="Rock",
                    field="porosity",
                    expected="finite fraction",
                    actual=self.porosity,
                )
            object.__setattr__(
                self,
                "porosity",
                permeability.new_full((n_cells,), float(self.porosity)),
            )
        porosity = _floating_tensor(self.porosity, field="porosity")
        if porosity.shape != (n_cells,):
            raise FlowContractError(
                "porosity must align with permeability cells",
                object_name="Rock",
                field="porosity",
                expected=(n_cells,),
                actual=tuple(porosity.shape),
            )
        if bool(((porosity <= 0) | (porosity >= 1)).any()):
            raise FlowContractError(
                "porosity must lie strictly inside (0, 1)",
                object_name="Rock",
                field="porosity",
                expected="0 < porosity < 1",
                actual=(
                    float(porosity.detach().min()),
                    float(porosity.detach().max()),
                ),
            )

        for field in ("compressibility_pa_inv", "reference_pressure_pa"):
            value = getattr(self, field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    raise FlowContractError(
                        f"{field} must be finite",
                        object_name="Rock",
                        field=field,
                        expected="finite SI value",
                        actual=value,
                    )
                object.__setattr__(
                    self,
                    field,
                    permeability.new_full((n_cells,), float(value)),
                )

        optional = {
            "compressibility_pa_inv": self.compressibility_pa_inv,
            "reference_pressure_pa": self.reference_pressure_pa,
        }
        if self.compressibility_pa_inv is not None and self.reference_pressure_pa is None:
            raise FlowContractError(
                "compressibility requires a reference pressure",
                object_name="Rock",
                field="compressibility_pa_inv/reference_pressure_pa",
                expected="reference_pressure_pa whenever compressibility is provided",
                actual=tuple(name for name, value in optional.items() if value is not None),
            )

        tensors = [permeability, porosity]
        for field, value in optional.items():
            if value is None:
                continue
            tensor = _floating_tensor(value, field=field)
            if tensor.shape != (n_cells,):
                raise FlowContractError(
                    f"{field} must align with permeability cells",
                    object_name="Rock",
                    field=field,
                    expected=(n_cells,),
                    actual=tuple(tensor.shape),
                )
            tensors.append(tensor)

        dtype = permeability.dtype
        device = permeability.device
        if any(value.dtype != dtype for value in tensors):
            raise FlowContractError(
                "rock tensors must share one dtype",
                object_name="Rock",
                field="dtype",
                expected=str(dtype),
                actual=tuple(str(value.dtype) for value in tensors),
            )
        if any(value.device != device for value in tensors):
            raise FlowContractError(
                "rock tensors must share one device",
                object_name="Rock",
                field="device",
                expected=str(device),
                actual=tuple(str(value.device) for value in tensors),
            )
        if self.compressibility_pa_inv is not None and bool(
            (self.compressibility_pa_inv < 0).any()
        ):
            raise FlowContractError(
                "compressibility_pa_inv must be non-negative",
                object_name="Rock",
                field="compressibility_pa_inv",
                expected=">= 0 Pa⁻¹",
                actual=float(self.compressibility_pa_inv.detach().min()),
            )

    @property
    def n_cells(self) -> int:
        return int(self.permeability_m2.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self.permeability_m2.dtype

    @property
    def device(self) -> torch.device:
        return self.permeability_m2.device

    def porosity_at_pressure(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        """Return porosity at ``pressure_pa`` without moving or casting it."""

        if (
            not isinstance(pressure_pa, torch.Tensor)
            or not pressure_pa.is_floating_point()
            or pressure_pa.shape != (self.n_cells,)
            or pressure_pa.dtype != self.dtype
            or pressure_pa.device != self.device
        ):
            raise FlowContractError(
                "pressure_pa must align with the rock tensor metadata",
                object_name="Rock.porosity_at_pressure",
                field="pressure_pa",
                expected=(str(self.dtype), str(self.device), (self.n_cells,)),
                actual=(
                    type(pressure_pa).__name__,
                    str(getattr(pressure_pa, "dtype", None)),
                    str(getattr(pressure_pa, "device", None)),
                    tuple(getattr(pressure_pa, "shape", ())),
                ),
            )
        if not bool(torch.isfinite(pressure_pa).all()):
            raise FlowContractError(
                "pressure_pa must be finite",
                object_name="Rock.porosity_at_pressure",
                field="pressure_pa",
                expected="finite Pa",
                actual="contains NaN or infinity",
            )
        if self.compressibility_pa_inv is None:
            return self.porosity
        assert self.reference_pressure_pa is not None
        result = self.porosity * (
            1.0 + self.compressibility_pa_inv * (pressure_pa - self.reference_pressure_pa)
        )
        if bool(((result <= 0) | (result >= 1)).any()):
            raise FlowContractError(
                "compressible-rock porosity left its physical domain",
                object_name="Rock.porosity_at_pressure",
                field="porosity",
                expected="0 < porosity < 1",
                actual=(
                    float(result.detach().min()),
                    float(result.detach().max()),
                ),
            )
        return result

    def pore_volume(self, pressure_pa: torch.Tensor, cell_volume_m3: torch.Tensor) -> torch.Tensor:
        """Return pore volume in cubic metres."""

        if (
            not isinstance(cell_volume_m3, torch.Tensor)
            or cell_volume_m3.shape != (self.n_cells,)
            or cell_volume_m3.dtype != self.dtype
            or cell_volume_m3.device != self.device
            or not bool(torch.isfinite(cell_volume_m3).all())
            or bool((cell_volume_m3 <= 0).any())
        ):
            raise FlowContractError(
                "cell_volume_m3 must be positive and align with the rock",
                object_name="Rock.pore_volume",
                field="cell_volume_m3",
                expected=(str(self.dtype), str(self.device), (self.n_cells,), "> 0 m³"),
                actual=(
                    str(getattr(cell_volume_m3, "dtype", None)),
                    str(getattr(cell_volume_m3, "device", None)),
                    tuple(getattr(cell_volume_m3, "shape", ())),
                ),
            )
        return self.porosity_at_pressure(pressure_pa) * cell_volume_m3

    def __repr__(self) -> str:
        porosity = self.porosity.detach()
        return (
            f"Rock(n_cells={self.n_cells}, porosity∈"
            f"[{porosity.min().item():.3f},{porosity.max().item():.3f}], SI=True)"
        )


__all__ = ["Rock"]
