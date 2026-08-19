"""PVT models: base interface + analytical + tabulated.

All implementations are ``nn.Module`` subclasses so PVT parameters
(viscosities, compressibilities, FVF anchor points) can become
``nn.Parameter`` for inversion without changing the surface API.

Production units are canonical SI: pressure [Pa], density [kg/m³], viscosity
[Pa·s], compressibility [Pa⁻¹], and dimensionless formation-volume factor.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn

from .._defaults import DEVICE, DTYPE
from ..errors import FlowContractError


@dataclass(frozen=True, slots=True)
class PropertyTable:
    """Immutable piecewise-linear table with an explicit bounds policy."""

    coordinates: torch.Tensor
    values: torch.Tensor
    bounds_policy: Literal["error", "constant"] = "error"

    def __post_init__(self) -> None:
        coordinates = self.coordinates
        values = self.values
        if (
            not isinstance(coordinates, torch.Tensor)
            or not coordinates.is_floating_point()
            or coordinates.ndim != 1
            or coordinates.numel() < 2
        ):
            raise FlowContractError(
                "PropertyTable coordinates must be a floating vector",
                object_name="PropertyTable",
                field="coordinates",
                expected="floating [knot] tensor with at least two knots",
                actual=(type(coordinates).__name__, tuple(getattr(coordinates, "shape", ()))),
            )
        if (
            not isinstance(values, torch.Tensor)
            or not values.is_floating_point()
            or values.ndim < 1
            or values.shape[0] != coordinates.numel()
        ):
            raise FlowContractError(
                "PropertyTable values must align with coordinates",
                object_name="PropertyTable",
                field="values",
                expected=f"floating [{coordinates.numel()}, ...] tensor",
                actual=(type(values).__name__, tuple(getattr(values, "shape", ()))),
            )
        if coordinates.dtype != values.dtype:
            raise FlowContractError(
                "PropertyTable tensors must share one dtype",
                object_name="PropertyTable",
                field="dtype",
                expected=str(coordinates.dtype),
                actual=str(values.dtype),
            )
        if coordinates.device != values.device:
            raise FlowContractError(
                "PropertyTable tensors must share one device",
                object_name="PropertyTable",
                field="device",
                expected=str(coordinates.device),
                actual=str(values.device),
            )
        if not bool(torch.isfinite(coordinates).all()) or not bool(torch.isfinite(values).all()):
            raise FlowContractError(
                "PropertyTable tensors must be finite",
                object_name="PropertyTable",
                field="coordinates/values",
                expected="finite values",
                actual="contains NaN or infinity",
            )
        if not bool((coordinates[1:] > coordinates[:-1]).all()):
            raise FlowContractError(
                "PropertyTable coordinates must be strictly increasing",
                object_name="PropertyTable",
                field="coordinates",
                expected="strictly increasing",
                actual=coordinates.detach().cpu().tolist(),
            )
        if self.bounds_policy not in ("error", "constant"):
            raise FlowContractError(
                "PropertyTable bounds_policy is invalid",
                object_name="PropertyTable",
                field="bounds_policy",
                expected=("error", "constant"),
                actual=self.bounds_policy,
            )
        # ``torch.searchsorted`` requires a contiguous boundary vector for its
        # fast path.  Normalizing strides here preserves dtype, device, values,
        # and the autograd graph while avoiding a hidden copy on every query.
        object.__setattr__(self, "coordinates", coordinates.contiguous())
        object.__setattr__(self, "values", values.contiguous())

    def interpolate(self, query: torch.Tensor) -> torch.Tensor:
        """Interpolate without silently copying, moving, or casting ``query``."""

        if (
            not isinstance(query, torch.Tensor)
            or not query.is_floating_point()
            or query.dtype != self.coordinates.dtype
            or query.device != self.coordinates.device
        ):
            raise FlowContractError(
                "PropertyTable query metadata does not match the table",
                object_name="PropertyTable.interpolate",
                field="query",
                expected=(str(self.coordinates.dtype), str(self.coordinates.device)),
                actual=(
                    type(query).__name__,
                    str(getattr(query, "dtype", None)),
                    str(getattr(query, "device", None)),
                ),
            )
        if not bool(torch.isfinite(query).all()):
            raise FlowContractError(
                "PropertyTable query must be finite",
                object_name="PropertyTable.interpolate",
                field="query",
                expected="finite coordinates",
                actual="contains NaN or infinity",
            )
        lower, upper = self.coordinates[0], self.coordinates[-1]
        outside = (query < lower) | (query > upper)
        if self.bounds_policy == "error" and bool(outside.any()):
            failed = torch.nonzero(outside.reshape(-1), as_tuple=False).reshape(-1)
            raise FlowContractError(
                "PropertyTable query is outside its declared bounds",
                object_name="PropertyTable.interpolate",
                field="query",
                expected=(float(lower), float(upper)),
                actual={"flat_indices": failed.detach().cpu().tolist()},
            )
        bounded = torch.clamp(query, min=lower, max=upper)
        index = torch.searchsorted(self.coordinates, bounded).clamp(1, self.coordinates.numel() - 1)
        x0 = self.coordinates[index - 1]
        x1 = self.coordinates[index]
        y0 = self.values[index - 1]
        y1 = self.values[index]
        weight = (bounded - x0) / (x1 - x0)
        if self.values.ndim > 1:
            weight = weight.reshape(*weight.shape, *((1,) * (self.values.ndim - 1)))
        return y0 + weight * (y1 - y0)


class PVT(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Abstract pressure-volume-temperature model.

    Subclasses must implement :meth:`density`, :meth:`viscosity`,
    :meth:`fvf`. Each takes and returns a ``torch.Tensor``
    (autograd-friendly).
    """

    def density(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def viscosity(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fvf(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rs(self, p: torch.Tensor) -> torch.Tensor:
        """Solution gas-oil ratio ``R_s(p)`` [standard m³ / stock-tank m³].

        Defaults to **zero**, dead oil, no dissolved gas. A live-oil PVT
        (:class:`PVTLiveOil`) overrides this; the black-oil gas balance always
        carries the dissolved-gas term ``R_s·S_o/B_o`` but it vanishes for any
        dead-oil phase, so dead-oil behaviour is recovered exactly.
        """
        return torch.zeros_like(p)


# --- analytical (constant-compressibility) PVT ----------------------------


class PVTAnalytic(PVT):
    """
    Analytical constant-compressibility PVT (linearised form).

    Density:    ``ρ(p) = ρ_ref · (1 + c · (p − p_ref))``
    FVF:        ``B(p) = B_ref · (1 − c · (p − p_ref))``
    Viscosity:  ``μ(p) = μ_ref · (1 − c_v · (p − p_ref))``

    First-order Taylor expansion of the exact ``ρ = ρ_ref · exp(c·Δp)``
    form, accurate for water and dead oil at typical reservoir
    pressure excursions (≲ a few hundred bars). For high-pressure
    cases where the linearisation breaks down, use :class:`PVTTable`
    with a tabulated PVDO / PVTW / PVDG deck.
    """

    def __init__(
        self,
        density_ref_kg_m3: float,
        viscosity_ref_pa_s: float,
        formation_volume_factor_ref: float,
        reference_pressure_pa: float = 27_579_029.172672,
        compressibility_pa_inv: float = 0.0,
        viscosibility_pa_inv: float = 0.0,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        values = {
            "density_ref_kg_m3": density_ref_kg_m3,
            "viscosity_ref_pa_s": viscosity_ref_pa_s,
            "formation_volume_factor_ref": formation_volume_factor_ref,
            "reference_pressure_pa": reference_pressure_pa,
            "compressibility_pa_inv": compressibility_pa_inv,
            "viscosibility_pa_inv": viscosibility_pa_inv,
        }
        for field, value in values.items():
            finite = isinstance(value, (int, float)) and not isinstance(value, bool)
            finite = finite and math.isfinite(float(value))
            nonnegative = field in {"compressibility_pa_inv", "viscosibility_pa_inv"}
            valid = finite and (float(value) >= 0 if nonnegative else float(value) > 0)
            if not valid:
                raise FlowContractError(
                    f"{field} is outside its physical domain",
                    object_name="PVTAnalytic",
                    field=field,
                    expected=">= 0" if nonnegative else "> 0 in canonical SI units",
                    actual=value,
                )
        for k, v in values.items():
            self.register_buffer(
                k,
                torch.tensor(float(v), device=self.device, dtype=self.dtype),
            )

    def _validate_pressure(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(pressure_pa, torch.Tensor)
            or not pressure_pa.is_floating_point()
            or pressure_pa.dtype != self.dtype
            or pressure_pa.device != self.device
        ):
            raise FlowContractError(
                "pressure tensor metadata must match PVTAnalytic",
                object_name="PVTAnalytic",
                field="pressure_pa.dtype/device",
                expected=(str(self.dtype), str(self.device)),
                actual=(
                    type(pressure_pa).__name__,
                    str(getattr(pressure_pa, "dtype", None)),
                    str(getattr(pressure_pa, "device", None)),
                ),
            )
        if not bool(torch.isfinite(pressure_pa).all()) or bool((pressure_pa <= 0).any()):
            raise FlowContractError(
                "pressure_pa must be positive and finite",
                object_name="PVTAnalytic",
                field="pressure_pa",
                expected="> 0 Pa",
                actual="contains a non-positive or non-finite value",
            )
        return pressure_pa

    def density(self, p: torch.Tensor) -> torch.Tensor:
        p = self._validate_pressure(p)
        dp = p - self.reference_pressure_pa
        factor = 1.0 + self.compressibility_pa_inv * dp
        if bool((factor <= 0).any()):
            raise FlowContractError(
                "density linearization is outside its positive domain",
                object_name="PVTAnalytic.density",
                field="pressure_pa",
                expected="1 + compressibility_pa_inv * dp > 0",
                actual="non-positive density factor",
            )
        return self.density_ref_kg_m3 * factor

    def viscosity(self, p: torch.Tensor) -> torch.Tensor:
        p = self._validate_pressure(p)
        dp = p - self.reference_pressure_pa
        factor = 1.0 - self.viscosibility_pa_inv * dp
        if bool((factor <= 0).any()):
            raise FlowContractError(
                "viscosity linearization is outside its positive domain",
                object_name="PVTAnalytic.viscosity",
                field="pressure_pa",
                expected="1 - viscosibility_pa_inv * dp > 0",
                actual="non-positive viscosity factor",
            )
        return self.viscosity_ref_pa_s * factor

    def fvf(self, p: torch.Tensor) -> torch.Tensor:
        p = self._validate_pressure(p)
        dp = p - self.reference_pressure_pa
        factor = 1.0 - self.compressibility_pa_inv * dp
        if bool((factor <= 0).any()):
            raise FlowContractError(
                "FVF linearization is outside its positive domain",
                object_name="PVTAnalytic.fvf",
                field="pressure_pa",
                expected="1 - compressibility_pa_inv * dp > 0",
                actual="non-positive formation-volume factor",
            )
        return self.formation_volume_factor_ref * factor


# --- live-oil PVT (dissolved gas, bubble point) ---------------------------


class PVTLiveOil(PVT):
    """
    Analytical **live-oil** PVT with dissolved gas and a bubble point.

    Below the bubble point ``p_b`` the oil is *saturated*: as pressure rises the
    solution gas-oil ratio ``R_s`` increases, the oil swells (``B_o`` rises) and
    thins (``μ_o`` falls). Above ``p_b`` the oil is *undersaturated*: ``R_s`` is
    fixed at its maximum and the oil compresses (``B_o`` falls) and stiffens
    (``μ_o`` rises) like a dead oil. ``B_o`` therefore peaks at ``p_b``, the
    classic live-oil signature.

        ``R_s(p) = R_s,max · min(p/p_b, 1)``                [standard m³/m³]
        ``B_o(p) = B_o,ref·(1 + a·R_s)·exp(−c_o·Δp_+)``     [m³/m³]
        ``μ_o(p) = μ_o,ref·(1 − b·R_s)·(1 + c_v·Δp_+)``     [Pa·s]
        ``ρ_o(p) = (ρ_o,sc + ρ_g,sc·R_s) / B_o``            [kg/m³]

    where ``(·)_+ = max(·, 0)`` is the undersaturated pressure excess. The
    undersaturated ``B_o`` uses the exponential
    (rather than linear) compressibility form so it stays strictly positive at
    any pressure. The ``R_s`` kink at ``p_b`` (saturated→undersaturated)
    is physically real and matches standard PVTO behaviour; it is the only
    point where ``∂R_s/∂p`` is discontinuous.

    The dissolved-gas mass ``ρ_g,sc·R_s`` makes reservoir live oil lighter
    than its stock-tank density would suggest once it swells, fed straight into
    the black-oil gravity term and the free/dissolved gas balance.
    """

    def __init__(
        self,
        surface_oil_density_kg_m3: float = 849.0,
        surface_gas_density_kg_m3: float = 0.96,
        reference_viscosity_pa_s: float = 2.0e-3,
        reference_formation_volume_factor: float = 1.05,
        solution_gas_oil_ratio_max_m3_m3: float = 106.8646,
        bubble_pressure_pa: float = 20_684_271.879504,
        swelling_per_solution_ratio: float = 1.4036e-3,
        oil_compressibility_pa_inv: float = 1.4503773773e-9,
        viscosibility_per_solution_ratio: float = 4.4916e-3,
        viscosibility_pa_inv: float = 2.9007547546e-9,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        values = {
            "surface_oil_density_kg_m3": surface_oil_density_kg_m3,
            "surface_gas_density_kg_m3": surface_gas_density_kg_m3,
            "reference_viscosity_pa_s": reference_viscosity_pa_s,
            "reference_formation_volume_factor": reference_formation_volume_factor,
            "solution_gas_oil_ratio_max_m3_m3": solution_gas_oil_ratio_max_m3_m3,
            "bubble_pressure_pa": bubble_pressure_pa,
            "swelling_per_solution_ratio": swelling_per_solution_ratio,
            "oil_compressibility_pa_inv": oil_compressibility_pa_inv,
            "viscosibility_per_solution_ratio": viscosibility_per_solution_ratio,
            "viscosibility_pa_inv": viscosibility_pa_inv,
        }
        nonnegative = {
            "swelling_per_solution_ratio",
            "oil_compressibility_pa_inv",
            "viscosibility_per_solution_ratio",
            "viscosibility_pa_inv",
        }
        for field, value in values.items():
            finite = isinstance(value, (int, float)) and not isinstance(value, bool)
            finite = finite and math.isfinite(float(value))
            valid = finite and (float(value) >= 0 if field in nonnegative else float(value) > 0)
            if not valid:
                raise FlowContractError(
                    f"{field} is outside its physical domain",
                    object_name="PVTLiveOil",
                    field=field,
                    expected=">= 0" if field in nonnegative else "> 0 in canonical SI units",
                    actual=value,
                )
        viscosity_reduction = float(viscosibility_per_solution_ratio) * float(
            solution_gas_oil_ratio_max_m3_m3
        )
        if viscosity_reduction >= 1.0:
            raise FlowContractError(
                "solution-ratio viscosity reduction must remain below one",
                object_name="PVTLiveOil",
                field="viscosibility_per_solution_ratio",
                expected="coefficient * maximum solution ratio < 1",
                actual=viscosity_reduction,
            )
        for field, value in values.items():
            self.register_buffer(
                field,
                torch.tensor(float(value), device=self.device, dtype=self.dtype),
            )

    def _validate_pressure(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(pressure_pa, torch.Tensor)
            or not pressure_pa.is_floating_point()
            or pressure_pa.dtype != self.dtype
            or pressure_pa.device != self.device
        ):
            raise FlowContractError(
                "pressure tensor metadata must match PVTLiveOil",
                object_name="PVTLiveOil",
                field="pressure_pa.dtype/device",
                expected=(str(self.dtype), str(self.device)),
                actual=(
                    type(pressure_pa).__name__,
                    str(getattr(pressure_pa, "dtype", None)),
                    str(getattr(pressure_pa, "device", None)),
                ),
            )
        if not bool(torch.isfinite(pressure_pa).all()) or bool((pressure_pa <= 0).any()):
            raise FlowContractError(
                "pressure_pa must be positive and finite",
                object_name="PVTLiveOil",
                field="pressure_pa",
                expected="> 0 Pa",
                actual="contains a non-positive or non-finite value",
            )
        return pressure_pa

    def _validate_solution_ratio(
        self,
        ratio_m3_m3: torch.Tensor,
        *,
        pressure_pa: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not isinstance(ratio_m3_m3, torch.Tensor)
            or not ratio_m3_m3.is_floating_point()
            or ratio_m3_m3.dtype != pressure_pa.dtype
            or ratio_m3_m3.device != pressure_pa.device
        ):
            raise FlowContractError(
                "solution ratio metadata must match pressure",
                object_name="PVTLiveOil",
                field="solution_gas_oil_ratio_m3_m3.dtype/device",
                expected=(str(pressure_pa.dtype), str(pressure_pa.device)),
                actual=(
                    type(ratio_m3_m3).__name__,
                    str(getattr(ratio_m3_m3, "dtype", None)),
                    str(getattr(ratio_m3_m3, "device", None)),
                ),
            )
        if not bool(torch.isfinite(ratio_m3_m3).all()) or bool((ratio_m3_m3 < 0).any()):
            raise FlowContractError(
                "solution ratio must be non-negative and finite",
                object_name="PVTLiveOil",
                field="solution_gas_oil_ratio_m3_m3",
                expected=">= 0 standard m³ / stock-tank m³",
                actual="contains a negative or non-finite value",
            )
        return ratio_m3_m3

    def rs(self, p: torch.Tensor) -> torch.Tensor:
        """Saturated solution ratio [standard m³ / stock-tank m³]."""
        p = self._validate_pressure(p)
        # Saturated below p_b (rises with p), capped at rs_max above (undersaturated).
        return self.solution_gas_oil_ratio_max_m3_m3 * (p / self.bubble_pressure_pa).clamp(max=1.0)

    def bubble_pressure(self, rs: torch.Tensor) -> torch.Tensor:
        """Pressure at which a given solution GOR is saturated (``R_s,sat(p_b)=R_s``)."""
        if not isinstance(rs, torch.Tensor):
            raise FlowContractError(
                "solution ratio must be a floating tensor",
                object_name="PVTLiveOil.bubble_pressure",
                field="solution_gas_oil_ratio_m3_m3",
                expected="floating tensor",
                actual=type(rs).__name__,
            )
        probe = torch.ones((), dtype=self.dtype, device=self.device)
        rs = self._validate_solution_ratio(rs, pressure_pa=probe)
        return self.bubble_pressure_pa * (rs / self.solution_gas_oil_ratio_max_m3_m3).clamp(max=1.0)

    def fvf(self, p: torch.Tensor, rs: torch.Tensor | None = None) -> torch.Tensor:
        """Oil FVF. With ``rs`` given (an *undersaturated* solution GOR < R_s,sat)
        the oil compresses above its own bubble point ``p_b(rs)``; default
        ``rs=None`` uses the saturated R_s,sat(p)."""
        p = self._validate_pressure(p)
        if rs is None:
            rs = self.rs(p)
        else:
            rs = self._validate_solution_ratio(rs, pressure_pa=p)
        dp_under = (p - self.bubble_pressure(rs)).clamp(min=0.0)  # >0 above this oil's bubble point
        # exp form keeps B_o > 0 at any pressure (a linear 1−c_o·dp would go negative).
        return (
            self.reference_formation_volume_factor
            * (1.0 + self.swelling_per_solution_ratio * rs)
            * torch.exp(-self.oil_compressibility_pa_inv * dp_under)
        )

    def viscosity(self, p: torch.Tensor, rs: torch.Tensor | None = None) -> torch.Tensor:
        p = self._validate_pressure(p)
        if rs is None:
            rs = self.rs(p)
        else:
            rs = self._validate_solution_ratio(rs, pressure_pa=p)
        dp_under = (p - self.bubble_pressure(rs)).clamp(min=0.0)
        return (
            self.reference_viscosity_pa_s
            * (1.0 - self.viscosibility_per_solution_ratio * rs)
            * (1.0 + self.viscosibility_pa_inv * dp_under)
        )

    def density(self, p: torch.Tensor, rs: torch.Tensor | None = None) -> torch.Tensor:
        # Reservoir oil mass = stock-tank oil + dissolved gas, per reservoir volume.
        p = self._validate_pressure(p)
        if rs is None:
            rs = self.rs(p)
        else:
            rs = self._validate_solution_ratio(rs, pressure_pa=p)
        return (self.surface_oil_density_kg_m3 + self.surface_gas_density_kg_m3 * rs) / self.fvf(
            p, rs
        )


# --- tabulated live-oil PVT (saturated PVTO branch) -----------------------


class PVTLiveOilTable(PVT):
    """
    Tabulated **live-oil** PVT from a saturated solution-GOR table (PVTO).

    A live-oil PVT is specified by its *saturated* branch: a set of rows
    ``(R_s, p_b, B_o^sat, μ_o^sat)`` where, at bubble-point pressure ``p_b``, the
    oil holds dissolved gas ``R_s`` and has the saturated FVF / viscosity given.
    Between rows everything is interpolated **linearly in pressure** (the
    industry-standard PVTO reading): the saturated curves

        ``R_s,sat(p)``:   solution GOR of oil that is saturated at ``p``
        ``B_o,sat(p)``:   saturated oil FVF
        ``μ_o,sat(p)``:   saturated oil viscosity

    are piecewise-linear in ``p`` through the table's ``(p_b, ·)`` knots, with the
    table endpoints clamped on extrapolation. This is the exact constitutive form
    the SPE1 (Odeh 1981) live-oil benchmark uses, where ``B_o`` rises and ``μ_o``
    falls with pressure as more gas dissolves; it replaces the analytic
    :class:`PVTLiveOil` (linear ``R_s(p)``) wherever a measured PVTO table is
    available, so the saturated curve shape is matched, not assumed.

    **Undersaturated branch.** Above an oil's own bubble point ``p_b(R_s)`` the
    dissolved gas is fixed and the oil simply compresses. With no per-Rs
    undersaturated sub-tables supplied, we apply a constant-compressibility
    undersaturated law anchored at the saturated state::

        ``B_o(p, R_s) = B_o,sat(p_b)·exp(−c_o·(p − p_b)_+)``
        ``μ_o(p, R_s) = μ_o,sat(p_b)·(1 + c_v·(p − p_b)_+)``

    where ``(·)_+ = max(·, 0)`` and ``p_b = p_b(R_s)`` is the saturation pressure
    of the supplied ``R_s`` (found by inverting ``R_s,sat``). With ``rs=None`` the
    oil is taken saturated at ``p`` (``R_s = R_s,sat(p)``) so the saturated branch
    is returned directly.

    Canonical inputs are separate one-dimensional tensors: solution ratio
    [standard m³ / stock-tank m³], bubble pressure [Pa], dimensionless FVF,
    and viscosity [Pa·s]. Densities are ``(ρ_o,sc + ρ_g,sc·R_s) / B_o`` in
    kg/m³. Any FIELD-unit deck conversion belongs in an input adapter.

    Differentiable in ``p`` (and ``rs``) via piecewise-linear interpolation; all
    table data are buffers, so this is a drop-in for :class:`PVTLiveOil` in the
    black-oil gas balance (it exposes the same ``rs``/``fvf``/``viscosity``/
    ``density``/``bubble_pressure`` surface).
    """

    def __init__(
        self,
        solution_gas_oil_ratio_m3_m3: torch.Tensor,
        bubble_pressure_pa: torch.Tensor,
        formation_volume_factor: torch.Tensor,
        viscosity_pa_s: torch.Tensor,
        surface_oil_density_kg_m3: float,
        surface_gas_density_kg_m3: float,
        oil_compressibility_pa_inv: float = 1.4503773773e-9,
        viscosibility_pa_inv: float = 2.9007547546e-9,
        bounds_policy: Literal["error", "constant"] = "error",
    ) -> None:
        super().__init__()
        tensors = {
            "solution_gas_oil_ratio_m3_m3": solution_gas_oil_ratio_m3_m3,
            "bubble_pressure_pa": bubble_pressure_pa,
            "formation_volume_factor": formation_volume_factor,
            "viscosity_pa_s": viscosity_pa_s,
        }
        if (
            not isinstance(bubble_pressure_pa, torch.Tensor)
            or not bubble_pressure_pa.is_floating_point()
            or bubble_pressure_pa.ndim != 1
            or bubble_pressure_pa.numel() < 2
        ):
            raise FlowContractError(
                "bubble_pressure_pa must be a floating knot vector",
                object_name="PVTLiveOilTable",
                field="bubble_pressure_pa",
                expected="floating [knot] tensor with at least two knots",
                actual=(
                    type(bubble_pressure_pa).__name__,
                    tuple(getattr(bubble_pressure_pa, "shape", ())),
                ),
            )
        for field, value in tensors.items():
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim != 1
                or value.shape != bubble_pressure_pa.shape
            ):
                raise FlowContractError(
                    f"{field} must align with the live-oil pressure table",
                    object_name="PVTLiveOilTable",
                    field=field,
                    expected=f"floating {tuple(bubble_pressure_pa.shape)} tensor",
                    actual=(type(value).__name__, tuple(getattr(value, "shape", ()))),
                )
            if value.dtype != bubble_pressure_pa.dtype or value.device != bubble_pressure_pa.device:
                raise FlowContractError(
                    "live-oil table tensors must share one dtype and device",
                    object_name="PVTLiveOilTable",
                    field=f"{field}.dtype/device",
                    expected=(str(bubble_pressure_pa.dtype), str(bubble_pressure_pa.device)),
                    actual=(str(value.dtype), str(value.device)),
                )
        pressure_table = PropertyTable(
            coordinates=bubble_pressure_pa,
            values=torch.stack(
                (solution_gas_oil_ratio_m3_m3, formation_volume_factor, viscosity_pa_s),
                dim=-1,
            ),
            bounds_policy=bounds_policy,
        )
        rs_col = solution_gas_oil_ratio_m3_m3
        pb_col = pressure_table.coordinates
        bo_col = formation_volume_factor
        mu_col = viscosity_pa_s
        if pressure_table.values.ndim != 2 or pressure_table.values.shape[1] != 3:
            raise FlowContractError(
                "live-oil table columns must align",
                object_name="PVTLiveOilTable",
                field="table tensors",
                expected="four aligned floating [knot] tensors",
                actual="misaligned table tensors",
            )
        if not bool((rs_col[1:] > rs_col[:-1]).all()):
            raise FlowContractError(
                "solution-ratio knots must be strictly increasing",
                object_name="PVTLiveOilTable",
                field="solution_gas_oil_ratio_m3_m3",
                expected="strictly increasing",
                actual=rs_col.detach().cpu().tolist(),
            )
        if bool((rs_col < 0).any()) or bool((bo_col <= 0).any()) or bool((mu_col <= 0).any()):
            raise FlowContractError(
                "live-oil table values are outside their physical domain",
                object_name="PVTLiveOilTable",
                field="solution ratio/FVF/viscosity",
                expected="ratio >= 0, FVF > 0, viscosity > 0 Pa·s",
                actual="contains an invalid value",
            )
        scalars = {
            "surface_oil_density_kg_m3": surface_oil_density_kg_m3,
            "surface_gas_density_kg_m3": surface_gas_density_kg_m3,
            "oil_compressibility_pa_inv": oil_compressibility_pa_inv,
            "viscosibility_pa_inv": viscosibility_pa_inv,
        }
        for field, value in scalars.items():
            finite = isinstance(value, (int, float)) and not isinstance(value, bool)
            finite = finite and math.isfinite(float(value))
            nonnegative = field.endswith("_pa_inv")
            valid = finite and (float(value) >= 0 if nonnegative else float(value) > 0)
            if not valid:
                raise FlowContractError(
                    f"{field} is outside its physical domain",
                    object_name="PVTLiveOilTable",
                    field=field,
                    expected=">= 0" if nonnegative else "> 0 in canonical SI units",
                    actual=value,
                )
        self.bounds_policy = pressure_table.bounds_policy
        self.device = pb_col.device
        self.dtype = pb_col.dtype
        for field, value in {
            "solution_gas_oil_ratio_m3_m3": rs_col,
            "bubble_pressure_pa": pb_col,
            "formation_volume_factor": bo_col,
            "viscosity_pa_s": mu_col,
        }.items():
            self.register_buffer(field, value.contiguous())
        for field, value in scalars.items():
            self.register_buffer(field, pb_col.new_tensor(float(value)))

    def _validate_pressure_query(self, pressure_pa: torch.Tensor) -> torch.Tensor:
        if isinstance(pressure_pa, torch.Tensor) and (
            bool((pressure_pa <= 0).any()) if pressure_pa.is_floating_point() else False
        ):
            raise FlowContractError(
                "pressure_pa must be positive",
                object_name="PVTLiveOilTable",
                field="pressure_pa",
                expected="> 0 Pa",
                actual="contains a non-positive value",
            )
        return pressure_pa

    def _validate_solution_ratio_query(self, ratio_m3_m3: torch.Tensor) -> torch.Tensor:
        if isinstance(ratio_m3_m3, torch.Tensor) and (
            bool((ratio_m3_m3 < 0).any()) if ratio_m3_m3.is_floating_point() else False
        ):
            raise FlowContractError(
                "solution ratio must be non-negative",
                object_name="PVTLiveOilTable",
                field="solution_gas_oil_ratio_m3_m3",
                expected=">= 0 standard m³ / stock-tank m³",
                actual="contains a negative value",
            )
        return ratio_m3_m3

    @property
    def maximum_solution_gas_oil_ratio_m3_m3(self) -> torch.Tensor:
        """Maximum dissolved-gas ratio represented by the table."""
        return self.solution_gas_oil_ratio_m3_m3[-1]

    @property
    def maximum_bubble_pressure_pa(self) -> torch.Tensor:
        """Highest bubble-point pressure represented by the table [Pa]."""
        return self.bubble_pressure_pa[-1]

    def _interp(self, x: torch.Tensor, xtab: torch.Tensor, ytab: torch.Tensor) -> torch.Tensor:
        return PropertyTable(xtab, ytab, self.bounds_policy).interpolate(x)

    def rs(self, p: torch.Tensor) -> torch.Tensor:
        """Saturated solution GOR ``R_s,sat(p)`` from the PVTO bubble-point knots."""
        p = self._validate_pressure_query(p)
        return self._interp(p, self.bubble_pressure_pa, self.solution_gas_oil_ratio_m3_m3)

    def bubble_pressure(self, rs: torch.Tensor) -> torch.Tensor:
        """Saturation pressure ``p_b(R_s)``: inverse of ``R_s,sat`` (interp on Rs)."""
        rs = self._validate_solution_ratio_query(rs)
        return self._interp(rs, self.solution_gas_oil_ratio_m3_m3, self.bubble_pressure_pa)

    def fvf(self, p: torch.Tensor, rs: torch.Tensor | None = None) -> torch.Tensor:
        """Oil FVF. ``rs=None`` ⇒ saturated at ``p``; otherwise the oil is
        undersaturated above ``p_b(rs)`` and compresses there."""
        p = self._validate_pressure_query(p)
        if rs is None:
            return self._interp(p, self.bubble_pressure_pa, self.formation_volume_factor)
        rs = self._validate_solution_ratio_query(rs)
        pb = self.bubble_pressure(rs)
        bo_sat = self._interp(pb, self.bubble_pressure_pa, self.formation_volume_factor)
        dp_under = (p - pb).clamp(min=0.0)
        return bo_sat * torch.exp(-self.oil_compressibility_pa_inv * dp_under)

    def viscosity(self, p: torch.Tensor, rs: torch.Tensor | None = None) -> torch.Tensor:
        p = self._validate_pressure_query(p)
        if rs is None:
            return self._interp(p, self.bubble_pressure_pa, self.viscosity_pa_s)
        rs = self._validate_solution_ratio_query(rs)
        pb = self.bubble_pressure(rs)
        mu_sat = self._interp(pb, self.bubble_pressure_pa, self.viscosity_pa_s)
        dp_under = (p - pb).clamp(min=0.0)
        return mu_sat * (1.0 + self.viscosibility_pa_inv * dp_under)

    def density(self, p: torch.Tensor, rs: torch.Tensor | None = None) -> torch.Tensor:
        p = self._validate_pressure_query(p)
        if rs is None:
            rs = self.rs(p)
        else:
            rs = self._validate_solution_ratio_query(rs)
        return (self.surface_oil_density_kg_m3 + self.surface_gas_density_kg_m3 * rs) / self.fvf(
            p, rs
        )


# --- tabulated PVT (linear interpolation) ---------------------------------


class PVTTable(PVT):
    """
    Tabulated PVT (PVDO / PVDG / PVTW keywords).

    Pressure [Pa], dimensionless formation-volume factor, and viscosity [Pa·s]
    are supplied as separate one-dimensional tensors. Density is
    ``surface_density_kg_m3 / B``. Mixed-unit gas tables must be converted by an
    explicit input adapter before construction.
    """

    def __init__(
        self,
        pressure_pa: torch.Tensor,
        formation_volume_factor: torch.Tensor,
        viscosity_pa_s: torch.Tensor,
        surface_density_kg_m3: float,
        bounds_policy: Literal["error", "constant"] = "error",
    ) -> None:
        super().__init__()
        if (
            not isinstance(pressure_pa, torch.Tensor)
            or not pressure_pa.is_floating_point()
            or pressure_pa.ndim != 1
        ):
            raise FlowContractError(
                "pressure_pa must be a floating knot vector",
                object_name="PVTTable",
                field="pressure_pa",
                expected="floating [knot] tensor",
                actual=(
                    type(pressure_pa).__name__,
                    tuple(getattr(pressure_pa, "shape", ())),
                ),
            )
        tensors = {
            "formation_volume_factor": formation_volume_factor,
            "viscosity_pa_s": viscosity_pa_s,
        }
        for field, value in tensors.items():
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim != 1
                or value.shape != pressure_pa.shape
            ):
                raise FlowContractError(
                    f"{field} must align with the pressure table",
                    object_name="PVTTable",
                    field=field,
                    expected="floating [knot] tensor",
                    actual=(type(value).__name__, tuple(getattr(value, "shape", ()))),
                )
            if value.dtype != pressure_pa.dtype or value.device != pressure_pa.device:
                raise FlowContractError(
                    "PVTTable tensors must share one dtype and device",
                    object_name="PVTTable",
                    field=f"{field}.dtype/device",
                    expected=(str(pressure_pa.dtype), str(pressure_pa.device)),
                    actual=(str(value.dtype), str(value.device)),
                )
        if bool((pressure_pa <= 0).any()):
            raise FlowContractError(
                "pressure_pa must be positive",
                object_name="PVTTable",
                field="pressure_pa",
                expected="> 0 Pa",
                actual="contains a non-positive value",
            )
        table = PropertyTable(
            coordinates=pressure_pa,
            values=torch.stack((formation_volume_factor, viscosity_pa_s), dim=-1),
            bounds_policy=bounds_policy,
        )
        if bool((formation_volume_factor <= 0).any()):
            raise FlowContractError(
                "formation_volume_factor must be positive",
                object_name="PVTTable",
                field="formation_volume_factor",
                expected="> 0",
                actual="contains a non-positive value",
            )
        if bool((viscosity_pa_s <= 0).any()):
            raise FlowContractError(
                "viscosity_pa_s must be positive",
                object_name="PVTTable",
                field="viscosity_pa_s",
                expected="> 0 Pa·s",
                actual="contains a non-positive value",
            )
        if (
            isinstance(surface_density_kg_m3, bool)
            or not isinstance(surface_density_kg_m3, (int, float))
            or not math.isfinite(float(surface_density_kg_m3))
            or float(surface_density_kg_m3) <= 0
        ):
            raise FlowContractError(
                "surface_density_kg_m3 must be positive and finite",
                object_name="PVTTable",
                field="surface_density_kg_m3",
                expected="> 0 kg/m³",
                actual=surface_density_kg_m3,
            )
        self.bounds_policy = table.bounds_policy
        self.device = pressure_pa.device
        self.dtype = pressure_pa.dtype
        self.register_buffer("pressure_pa", pressure_pa)
        self.register_buffer("formation_volume_factor", formation_volume_factor)
        self.register_buffer("viscosity_pa_s", viscosity_pa_s)
        self.register_buffer(
            "surface_density_kg_m3",
            pressure_pa.new_tensor(float(surface_density_kg_m3)),
        )

    def _interp(self, p: torch.Tensor, ytab: torch.Tensor) -> torch.Tensor:
        table = PropertyTable(
            coordinates=self.pressure_pa,
            values=ytab,
            bounds_policy=self.bounds_policy,
        )
        return table.interpolate(p)

    def fvf(self, p: torch.Tensor) -> torch.Tensor:
        return self._interp(p, self.formation_volume_factor)

    def viscosity(self, p: torch.Tensor) -> torch.Tensor:
        return self._interp(p, self.viscosity_pa_s)

    def density(self, p: torch.Tensor) -> torch.Tensor:
        return self.surface_density_kg_m3 / self.fvf(p)


__all__ = [
    "PropertyTable",
    "PVT",
    "PVTAnalytic",
    "PVTLiveOil",
    "PVTLiveOilTable",
    "PVTTable",
]
