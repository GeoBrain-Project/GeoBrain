"""
Relative permeability models (kr curves + gas-relperm hysteresis).

All as ``nn.Module`` so endpoint parameters (``Swc``, ``Sor``, ``n_w``,
``n_o``, ``kr_max``) can become ``nn.Parameter`` for endpoint-scaling
inversion.

Two-phase: oil-water Corey, tabulated SWOF (the table carries its own Pc
column). Standalone analytic Pc and Pc hysteresis live in :mod:`.capillary`.
Three-phase: Stone-II (extended Corey) interpolation.

The gas relative-permeability hysteresis models (Land / Killough / Carlson,
:class:`_GasHysteresis` and subclasses) follow the standard formulations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from ....core import GeoBrainError
from .._defaults import DEVICE, DTYPE
from ..errors import FlowContractError
from .pvt import PropertyTable


class RelPerm(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """Abstract relperm. Implements :meth:`kr_oil` and :meth:`kr_water`."""

    def _validate_saturation(self, saturation: torch.Tensor, *, field: str) -> torch.Tensor:
        dtype = getattr(self, "dtype", None)
        device = getattr(self, "device", None)
        if (
            not isinstance(saturation, torch.Tensor)
            or not saturation.is_floating_point()
            or (dtype is not None and saturation.dtype != dtype)
            or (device is not None and saturation.device != device)
        ):
            raise FlowContractError(
                "saturation tensor metadata must match the relative-permeability model",
                object_name=type(self).__name__,
                field=f"{field}.dtype/device",
                expected=(str(dtype), str(device)),
                actual=(
                    type(saturation).__name__,
                    str(getattr(saturation, "dtype", None)),
                    str(getattr(saturation, "device", None)),
                ),
            )
        endpoint_tolerance = max(1.0e-12, 8.0 * torch.finfo(saturation.dtype).eps)
        if not bool(torch.isfinite(saturation).all()) or bool(
            ((saturation < -endpoint_tolerance) | (saturation > 1 + endpoint_tolerance)).any()
        ):
            raise FlowContractError(
                "saturation must lie in its physical domain",
                object_name=type(self).__name__,
                field=field,
                expected="finite values in [0, 1]",
                actual="contains a non-finite value or lies outside [0, 1]",
            )
        # Endpoint states acquire a few ulps of pollution in coupled Newton
        # solves (for example an absent gas phase). Accept only that numerical
        # boundary layer; material excursions still fail above.
        return saturation.clamp(min=0.0, max=1.0)

    def kr_oil(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def kr_water(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# --- Corey-style oil-water relperm ----------------------------------------


class RelPermCorey(RelPerm):
    """
    Corey oil-water relative permeability.

    With normalised water saturation ``S_e = (S_w − S_wc) / (1 − S_wc − S_or)``
    clamped to ``[0, 1]``::

        k_rw = k_rw_max · S_e^{n_w}
        k_ro = k_ro_max · (1 − S_e)^{n_o}

    All parameters can be inverted by passing ``trainable=True``.
    """

    def __init__(
        self,
        swc: float = 0.2,
        sor: float = 0.2,
        n_w: float = 2.0,
        n_o: float = 2.0,
        kr_w_max: float = 1.0,
        kr_o_max: float = 1.0,
        trainable: bool = False,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        scalars = {
            "swc": swc,
            "sor": sor,
            "n_w": n_w,
            "n_o": n_o,
            "kr_w_max": kr_w_max,
            "kr_o_max": kr_o_max,
        }
        for name, value in scalars.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise FlowContractError(
                    f"RelPermCorey {name} must be a finite scalar",
                    object_name="RelPermCorey",
                    field=name,
                    expected="finite scalar",
                    actual=value,
                )
        if not (0.0 <= swc < 1.0):
            raise GeoBrainError(
                "RelPermCorey swc must lie in [0, 1)",
                object_name="RelPermCorey",
                field="swc",
                expected="0 <= swc < 1",
                actual=swc,
            )
        if not (0.0 <= sor < 1.0):
            raise GeoBrainError(
                "RelPermCorey sor must lie in [0, 1)",
                object_name="RelPermCorey",
                field="sor",
                expected="0 <= sor < 1",
                actual=sor,
            )
        if not (swc + sor < 1.0):
            raise GeoBrainError(
                "RelPermCorey requires swc + sor < 1 (movable range)",
                object_name="RelPermCorey",
                field="swc+sor",
                expected="swc + sor < 1",
                actual=swc + sor,
            )
        for name, val in (
            ("n_w", n_w),
            ("n_o", n_o),
            ("kr_w_max", kr_w_max),
            ("kr_o_max", kr_o_max),
        ):
            if not (val > 0):
                raise GeoBrainError(
                    f"RelPermCorey {name} must be strictly positive",
                    object_name="RelPermCorey",
                    field=name,
                    expected="> 0",
                    actual=val,
                )
        self.device = torch.device(device)
        self.dtype = dtype
        kw = {"device": self.device, "dtype": self.dtype}
        params = scalars
        if trainable:
            for k, v in params.items():
                setattr(self, k, nn.Parameter(torch.tensor(v, **kw)))
        else:
            for k, v in params.items():
                self.register_buffer(k, torch.tensor(v, **kw))

    def _se(self, sw: torch.Tensor) -> torch.Tensor:
        sw = self._validate_saturation(sw, field="water_saturation")
        denom = 1.0 - self.swc - self.sor
        return ((sw - self.swc) / denom).clamp(min=0.0, max=1.0)

    def kr_water(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        se = self._se(sw)
        return self.kr_w_max * se.pow(self.n_w)

    def kr_oil(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        se = self._se(sw)
        return self.kr_o_max * (1.0 - se).pow(self.n_o)


# --- Tabulated relperm (SWOF format) --------------------------------------


class RelPermTable(RelPerm):
    """
    Tabulated oil-water relperm + capillary pressure (SWOF format).

    Table format: ``(n_rows, 4)`` columns ``(Sw, kr_w, kr_o, Pc_ow [Pa])``.
    Bounds behavior is explicit: ``"error"`` rejects extrapolation and
    ``"constant"`` clamps to the nearest endpoint. The Pc column is optional;
    if the table has only 3 columns, Pc defaults to zero everywhere.

    Use this when matching SWOF reference data exactly. For
    parameter-fitting workflows where the curve shape is itself
    inverted, use :class:`RelPermCorey` (its endpoints and exponents
    are first-class :class:`nn.Parameter`).
    """

    def __init__(
        self,
        table: torch.Tensor,
        bounds_policy: Literal["error", "constant"] = "error",
    ) -> None:
        super().__init__()
        if (
            not isinstance(table, torch.Tensor)
            or not table.is_floating_point()
            or table.ndim != 2
            or table.shape[1] not in (3, 4)
        ):
            raise FlowContractError(
                "RelPermTable expects (n_rows, >=3) columns (Sw, kr_w, kr_o [, Pc])",
                object_name="RelPermTable",
                field="table",
                expected="floating (n_rows, 3 or 4) tensor",
                actual=(type(table).__name__, tuple(getattr(table, "shape", ()))),
            )
        if int(table.shape[0]) < 2:
            raise FlowContractError(
                "RelPermTable requires at least 2 rows for interpolation",
                object_name="RelPermTable",
                field="table",
                expected=">= 2 rows",
                actual=int(table.shape[0]),
            )
        self.device = table.device
        self.dtype = table.dtype
        sw_col = table[:, 0]
        kr_columns = table[:, 1:3]
        if bool(((sw_col < 0) | (sw_col > 1)).any()) or bool(
            ((kr_columns < 0) | (kr_columns > 1)).any()
        ):
            raise FlowContractError(
                "RelPermTable saturation and relative permeability are outside their domain",
                object_name="RelPermTable",
                field="table[:,0:3]",
                expected="saturation and relative permeability in [0, 1]",
                actual="contains a value outside [0, 1]",
            )
        pc_column = table[:, 3] if table.shape[1] == 4 else torch.zeros_like(sw_col)
        interpolation = PropertyTable(
            coordinates=sw_col,
            values=torch.column_stack((kr_columns, pc_column)),
            bounds_policy=bounds_policy,
        )
        self.bounds_policy = interpolation.bounds_policy
        self.register_buffer("sw_tab", interpolation.coordinates)
        self.register_buffer("krw_tab", interpolation.values[:, 0])
        self.register_buffer("kro_tab", interpolation.values[:, 1])
        self.register_buffer("pc_tab", interpolation.values[:, 2])

    @property
    def swc(self) -> torch.Tensor:
        """Connate (minimum-tabulated) saturation: the first table row.

        Lets a tabulated curve stand in for a Corey curve wherever the connate
        endpoint is needed (e.g. Stone-II oil-relperm normalization in
        :class:`ThreePhaseRelPerm`, which evaluates ``kr_oil`` at connate water)."""
        return self.sw_tab[0]

    def _interp(self, sw: torch.Tensor, ytab: torch.Tensor) -> torch.Tensor:
        sw = self._validate_saturation(sw, field="water_saturation")
        return PropertyTable(self.sw_tab, ytab, self.bounds_policy).interpolate(sw)

    def kr_water(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._interp(sw, self.krw_tab)

    def kr_oil(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._interp(sw, self.kro_tab)

    def pc(self, sw: torch.Tensor) -> torch.Tensor:
        return self._interp(sw, self.pc_tab)


# --- Three-phase Stone-II -------------------------------------------------


class ThreePhaseRelPerm(RelPerm):
    """
    Simple three-phase oil-water-gas relperm (Stone-II).

    Takes two two-phase systems (oil-water + oil-gas) and interpolates
    oil relperm via Stone-II::

        k_ro = (k_row · k_rog) / k_row(S_wc)

    The oil-gas auxiliary system is parameterised so that
    ``og.kr_water`` plays the role of ``kr_gas`` (gas is the non-wetting
    phase in the og pair). Used by :class:`BlackOilModel` for
    three-phase residual assembly.
    """

    def __init__(
        self,
        ow: RelPermCorey,
        og: RelPermCorey,
        gas_hysteresis: "_GasHysteresis | None" = None,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        # ``ow``/``og`` are two-phase curves that must expose kr_oil + kr_water
        # (a hysteresis object is gas-only and belongs in ``gas_hysteresis=``).
        for name, m in (("ow", ow), ("og", og)):
            if not (hasattr(m, "kr_oil") and hasattr(m, "kr_water")):
                raise GeoBrainError(
                    f"ThreePhaseRelPerm {name} must provide kr_oil and kr_water "
                    "(a RelPermCorey / RelPerm). A gas-hysteresis model is gas-only "
                    "and goes in gas_hysteresis=, not as og.",
                    object_name="ThreePhaseRelPerm",
                    field=name,
                    expected="RelPerm with kr_oil and kr_water",
                    actual=type(m).__name__,
                )
        if gas_hysteresis is not None and not hasattr(gas_hysteresis, "kr_gas"):
            raise GeoBrainError(
                "ThreePhaseRelPerm gas_hysteresis must provide kr_gas(sg, sg_turn)",
                object_name="ThreePhaseRelPerm",
                field="gas_hysteresis",
                expected="a gas-hysteresis model (KilloughHysteresis/CarlsonHysteresis)",
                actual=type(gas_hysteresis).__name__,
            )
        self.ow = ow
        self.og = og
        self.gas_hysteresis = gas_hysteresis
        self.device = torch.device(device)
        self.dtype = dtype

    def kr_water(self, sw: torch.Tensor, sg: torch.Tensor | None = None) -> torch.Tensor:
        return self.ow.kr_water(sw)

    def kr_gas(self, sg: torch.Tensor, sg_turn: torch.Tensor | None = None) -> torch.Tensor:
        # With a gas-hysteresis model the gas relperm follows the imbibition
        # scanning curve for the supplied turning point ``Sg_turn``; otherwise
        # the bounding drainage curve ``og.kr_water(sg)`` (gas is non-wetting in
        # the oil-gas pair, so the natural sg argument: NOT 1 − sg).
        if self.gas_hysteresis is not None:
            return self.gas_hysteresis.kr_gas(sg, sg_turn)
        return self.og.kr_water(sg)

    def kr_oil(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sg is None:
            raise FlowContractError(
                "ThreePhaseRelPerm.kr_oil requires gas saturation",
                object_name="ThreePhaseRelPerm.kr_oil",
                field="sg",
                expected="gas-saturation tensor",
                actual=None,
            )
        kr_ow = self.ow.kr_oil(sw)
        kr_og = self.og.kr_oil(sg)
        # Stone-II normalised by k_ro at connate water.
        kr_at_swc = self.ow.kr_oil(self.ow.swc)
        return (kr_ow * kr_og) / (kr_at_swc + 1e-30)


# --- relative-permeability hysteresis (non-wetting / gas) ------------------


class _GasHysteresis(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """Land trapping + drainage/imbibition bounding curves for the non-wetting
    (gas) phase.

    Hysteresis is *stateful*; it depends on the saturation history through the
    turning point ``Sg_turn`` (the largest gas saturation a cell has reached on
    drainage; the transient loop tracks it as a per-cell running max of the
    saturations). Takes a **drainage** and an **imbibition**
    bounding curve (``RelPermCorey``; gas is the non-wetting phase so their
    ``kr_water`` is k_rg). Endpoints come from the curves: drainage critical
    ``Sgc = drainage.swc`` and max ``Sg_max = 1 − drainage.sor``; the imbibition
    critical ``Sgr_max = imbibition.swc`` is the maximum trapped gas. Trapped gas
    at a turning point follows Land (1968) with Killough's regularisation::

        K   = 1/(Sgr_max − Sgc) − 1/(Sg_max − Sgc)        (Land coefficient)
        M   = 1 + tol·(Sg_max − Sg_turn)                  (numerical reg.; tol→0 = pure Land)
        Sgr = Sgc + (Sg_turn − Sgc)/(M + K·(Sg_turn − Sgc))

    so a deeper drainage excursion traps more gas (``Sgr`` rises with
    ``Sg_turn``, → ``Sgr_max`` as ``Sg_turn → Sg_max``). With no history
    (``Sg_turn = Sg``) the bounding drainage curve is returned. Subclasses
    supply the scanning curve between the two bounds.
    """

    def __init__(
        self,
        drainage: RelPermCorey,
        imbibition: RelPermCorey,
        *,
        tol: float = 0.1,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        sgc = float(drainage.swc)
        sg_max = 1.0 - float(drainage.sor)
        sgr_max = float(imbibition.swc)  # imbibition critical = max trapped gas
        imb_s_max = 1.0 - float(imbibition.sor)
        if not (0.0 <= sgc < sgr_max < sg_max <= 1.0):
            raise GeoBrainError(
                "Gas hysteresis needs drainage.swc(Sgc) < imbibition.swc(Sgr_max) "
                "< 1−drainage.sor(Sg_max); pass an imbibition curve whose connate "
                "saturation is the trapped gas.",
                object_name=type(self).__name__,
                field="(Sgc, Sgr_max, Sg_max)",
                expected="0 <= Sgc < Sgr_max < Sg_max <= 1",
                actual=(sgc, sgr_max, sg_max),
            )
        self.drainage = drainage
        self.imbibition = imbibition
        self.device = torch.device(device)
        self.dtype = dtype
        land_c = 1.0 / (sgr_max - sgc) - 1.0 / (sg_max - sgc)
        for k, v in {
            "sgc": sgc,
            "sg_max": sg_max,
            "sgr_max": sgr_max,
            "imb_s_max": imb_s_max,
            "land_c": land_c,
            "tol": float(tol),
        }.items():
            self.register_buffer(k, torch.tensor(float(v), device=self.device, dtype=self.dtype))

    def trapped_gas(self, sg_turn: torch.Tensor) -> torch.Tensor:
        """Land (Killough-regularised) trapped gas saturation for ``Sg_turn``."""
        d = (sg_turn - self.sgc).clamp(min=0.0)
        m = 1.0 + self.tol * (self.sg_max - sg_turn).clamp(min=0.0)
        return self.sgc + d / (m + self.land_c * d)

    def _scanning(
        self,
        sg: torch.Tensor,
        sg_turn: torch.Tensor,
        sgr: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def kr_gas(self, sg: torch.Tensor, sg_turn: torch.Tensor | None = None) -> torch.Tensor:
        """Gas relperm with hysteresis. ``Sg_turn`` defaults to ``Sg`` (drainage)."""
        if sg_turn is None:
            sg_turn = sg
        sg_turn = torch.maximum(sg_turn, sg)  # turning point ≥ current saturation
        sgr = self.trapped_gas(sg_turn)
        kr_scan = self._scanning(sg, sg_turn, sgr)
        eps = 1e-8
        # Dispatch (kr hysteresis): drainage on the drainage path / below
        # critical; the bounding imbibition curve once the turning point reaches
        # imbibition's max; otherwise the scanning curve between the two bounds.
        on_drainage = (sg >= sg_turn - eps) | (sg <= self.sgc)
        on_imb_bound = sg_turn >= self.imb_s_max - eps
        kr = torch.where(
            on_drainage,
            self.drainage.kr_water(sg),
            torch.where(on_imb_bound, self.imbibition.kr_water(sg), kr_scan),
        )
        return torch.where(sg > sgr, kr, torch.zeros_like(kr))  # trapped ⇒ immobile


class KilloughHysteresis(_GasHysteresis):
    """Killough (1976) non-wetting relperm hysteresis.

    The imbibition scanning curve maps onto the bounding **imbibition** curve via
    a normalized saturation and is rescaled by the drainage endpoints::

        Sg_norm = Sgr_max + (Sg − Sgr)·(Sg_max − Sgr_max)/(Sg_turn − Sgr)
        krg(Sg) = krg_imb(Sg_norm) · krg_drain(Sg_turn)/krg_drain(Sg_max)

    which is continuous with drainage at the reversal (``Sg = Sg_turn`` ⇒
    ``Sg_norm = Sg_max`` ⇒ ``krg_imb(Sg_max)·krg_drain(Sg_turn)/krg_drain(Sg_max)
    = krg_drain(Sg_turn)`` when the two bounds share the same k_rg,max) and zero
    at the trapped saturation (``Sg = Sgr`` ⇒ ``Sg_norm = Sgr_max`` ⇒ 0).
    """

    def _scanning(
        self,
        sg: torch.Tensor,
        sg_turn: torch.Tensor,
        sgr: torch.Tensor,
    ) -> torch.Tensor:
        den = (sg_turn - sgr).clamp(min=1e-12)
        s_norm = self.sgr_max + (sg - sgr) * (self.sg_max - self.sgr_max) / den
        kr_imb = self.imbibition.kr_water(s_norm)
        scale = self.drainage.kr_water(sg_turn) / (self.drainage.kr_water(self.sg_max) + 1e-30)
        return kr_imb * scale


class CarlsonHysteresis(_GasHysteresis):
    """Carlson (1981) non-wetting relperm hysteresis.

    The bounding **imbibition** curve is shifted so it passes through the
    drainage value at the turning point: find ``Sg_meet`` where the imbibition
    curve equals ``krg_drain(Sg_turn)``, then evaluate it shifted::

        krg(Sg) = krg_imb(Sg + Sg_meet − Sg_turn),   krg_imb(Sg_meet) = krg_drain(Sg_turn)

    This is continuous at the reversal by construction (``Sg = Sg_turn`` ⇒
    ``krg_imb(Sg_meet) = krg_drain(Sg_turn)``). ``Sg_meet`` is obtained
    analytically by inverting the imbibition Corey curve (no root solve).
    """

    def _scanning(
        self,
        sg: torch.Tensor,
        sg_turn: torch.Tensor,
        sgr: torch.Tensor,
    ) -> torch.Tensor:
        kr_at_max = self.drainage.kr_water(sg_turn)  # drainage krg at the turning point
        imb = self.imbibition
        # invert imbibition Corey  krg = kr_max·Se^n  ⇒  Se = (krg/kr_max)^(1/n)
        se = (kr_at_max / (imb.kr_w_max + 1e-30)).clamp(min=0.0, max=1.0).pow(1.0 / imb.n_w)
        s_meet = imb.swc + se * (1.0 - imb.swc - imb.sor)
        return imb.kr_water(sg + s_meet - sg_turn)


__all__ = [
    "CarlsonHysteresis",
    "KilloughHysteresis",
    "RelPerm",
    "RelPermCorey",
    "RelPermTable",
    "ThreePhaseRelPerm",
]
