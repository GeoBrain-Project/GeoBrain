"""
Capillary-pressure models and Pc hysteresis.

Capillary pressure ``Pc(S_w) = p_o − p_w`` is a distinct constitutive
relation from relative permeability (it enters the phase-potential /
accumulation terms, not the mobility), so the standalone analytic models
live here rather than alongside the ``kr`` curves in :mod:`.relperm`.

- :class:`BrooksCoreyPc`: analytic Brooks-Corey drainage curve, with
  inversion-ready ``nn.Parameter`` endpoints.
- :class:`CapillaryHysteresis`: drainage/imbibition bounding curves with a
  saturation-history scanning curve.

(The tabulated SWOF Pc column stays inside :class:`~geobrain.physics.flow.properties.relperm.RelPermTable`,
since that Pc is intrinsic to the table format.)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....core import GeoBrainError
from .._defaults import DEVICE, DTYPE, S_MAX, S_MIN
from ..errors import FlowContractError


class BrooksCoreyPc(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Brooks-Corey drainage capillary-pressure curve.

    ``Pc(S_w) = P_e · S_e^{−1/λ}`` where ``S_e`` is the normalised
    water saturation between ``S_wc`` and ``1 − S_or``. Returns the
    oil-water capillary pressure ``Pc_ow = p_o − p_w`` in Pa.
    """

    def __init__(
        self,
        entry_pressure_pa: float = 1.0,
        lam: float = 2.0,
        swc: float = 0.2,
        sor: float = 0.2,
        trainable: bool = False,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        scalars = {
            "entry_pressure_pa": entry_pressure_pa,
            "lam": lam,
            "swc": swc,
            "sor": sor,
        }
        for field, value in scalars.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise FlowContractError(
                    f"BrooksCoreyPc {field} must be a finite scalar",
                    object_name="BrooksCoreyPc",
                    field=field,
                    expected="finite scalar",
                    actual=value,
                )
        if not (entry_pressure_pa > 0):
            raise FlowContractError(
                "BrooksCoreyPc entry_pressure_pa must be > 0",
                object_name="BrooksCoreyPc",
                field="entry_pressure_pa",
                expected="> 0 Pa",
                actual=entry_pressure_pa,
            )
        if not (lam > 0):
            raise FlowContractError(
                "BrooksCoreyPc lam must be > 0",
                object_name="BrooksCoreyPc",
                field="lam",
                expected="> 0",
                actual=lam,
            )
        if not (0.0 <= swc < 1.0):
            raise GeoBrainError(
                "BrooksCoreyPc swc must lie in [0, 1)",
                object_name="BrooksCoreyPc",
                field="swc",
                expected="0 <= swc < 1",
                actual=swc,
            )
        if not (0.0 <= sor < 1.0) or not (swc + sor < 1.0):
            raise GeoBrainError(
                "BrooksCoreyPc requires 0 <= sor < 1 and swc + sor < 1",
                object_name="BrooksCoreyPc",
                field="sor",
                expected="0 <= sor < 1 and swc + sor < 1",
                actual=(swc, sor),
            )
        self.device = torch.device(device)
        self.dtype = dtype
        kw = {"device": self.device, "dtype": self.dtype}
        ps = scalars
        if trainable:
            for k, v in ps.items():
                setattr(self, k, nn.Parameter(torch.tensor(v, **kw)))
        else:
            for k, v in ps.items():
                self.register_buffer(k, torch.tensor(v, **kw))

    def __call__(self, sw: torch.Tensor) -> torch.Tensor:
        return self.pc(sw)

    def pc(self, sw: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(sw, torch.Tensor)
            or not sw.is_floating_point()
            or sw.dtype != self.dtype
            or sw.device != self.device
        ):
            raise FlowContractError(
                "water saturation metadata must match BrooksCoreyPc",
                object_name="BrooksCoreyPc",
                field="water_saturation.dtype/device",
                expected=(str(self.dtype), str(self.device)),
                actual=(
                    type(sw).__name__,
                    str(getattr(sw, "dtype", None)),
                    str(getattr(sw, "device", None)),
                ),
            )
        if not bool(torch.isfinite(sw).all()) or bool(((sw < 0) | (sw > 1)).any()):
            raise FlowContractError(
                "water saturation must lie in its physical domain",
                object_name="BrooksCoreyPc",
                field="water_saturation",
                expected="finite values in [0, 1]",
                actual="contains a non-finite value or lies outside [0, 1]",
            )
        denom = 1.0 - self.swc - self.sor
        se = ((sw - self.swc) / denom).clamp(min=S_MIN, max=S_MAX)
        return self.entry_pressure_pa * se.pow(-1.0 / self.lam)


class CapillaryHysteresis(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """Capillary-pressure hysteresis between bounding drainage / imbibition.

    Holds a higher bounding **drainage** curve and a lower bounding
    **imbibition** curve (both :class:`BrooksCoreyPc`-style). A scanning curve
    leaving the drainage bound at the turning point ``Sw_turn`` (the minimum
    water saturation reached on drainage) relaxes toward the imbibition bound as
    the wetting phase imbibes::

        Pc(Sw) = Pc_imb(Sw) + w·(Pc_drain(Sw) − Pc_imb(Sw))
        w = (1 − Sw)/(1 − Sw_turn)              (1 at the turning point → drainage,
                                                 → 0 as Sw → 1 → imbibition)

    so at ``Sw = Sw_turn`` the scanning curve coincides with the drainage bound
    (continuity with the drainage path it left) and relaxes toward imbibition as
    water imbibes. With ``Sw_turn`` defaulting to ``Sw`` the drainage bound is
    returned.
    """

    def __init__(self, drainage: BrooksCoreyPc, imbibition: BrooksCoreyPc) -> None:
        super().__init__()
        if drainage.dtype != imbibition.dtype or drainage.device != imbibition.device:
            raise FlowContractError(
                "capillary hysteresis bounds must share one dtype and device",
                object_name="CapillaryHysteresis",
                field="drainage/imbibition.dtype/device",
                expected=(str(drainage.dtype), str(drainage.device)),
                actual=(str(imbibition.dtype), str(imbibition.device)),
            )
        self.drainage = drainage
        self.imbibition = imbibition

    def __call__(self, sw: torch.Tensor, sw_turn: torch.Tensor | None = None) -> torch.Tensor:
        return self.pc(sw, sw_turn)

    def pc(self, sw: torch.Tensor, sw_turn: torch.Tensor | None = None) -> torch.Tensor:
        pc_d = self.drainage.pc(sw)
        if sw_turn is None:
            return pc_d
        sw_turn = torch.minimum(sw_turn, sw)  # turning point ≤ current (water increasing)
        pc_i = self.imbibition.pc(sw)
        # weight on the drainage bound: 1 at the turning point, → 0 as Sw → 1.
        w = ((1.0 - sw) / (1.0 - sw_turn + 1e-12)).clamp(min=0.0, max=1.0)
        return pc_i + w * (pc_d - pc_i)


__all__ = [
    "BrooksCoreyPc",
    "CapillaryHysteresis",
]
