"""Explicit Eclipse/SPE deck import into canonical SI Flow objects.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

from geobrain.core import GeoBrainError

from ..errors import FlowCapabilityError, FlowContractError
from ._eclipse_deck import (
    BlackOilCase,
    BlackOilDeck,
    WellSpec,
    _build_blackoil_case,
    _parse_blackoil_deck_source,
)

UnitSystem = Literal["FIELD", "METRIC", "SI"]

_SUPPORTED_UNITS = frozenset({"FIELD", "METRIC", "SI"})
_PA_PER_PSI = 6894.757293168
_PA_PER_BAR = 100_000.0
_M_PER_FT = 0.3048
_M2_PER_MD = 9.869233e-16
_PA_S_PER_CP = 1.0e-3
_S_PER_DAY = 86_400.0
_M3_PER_STB = 0.158987294928
_M3_PER_SCF = 0.028316846592
_KG_M3_PER_LBM_FT3 = 16.01846337396014


def _validate_requested_units(unit_system: object) -> UnitSystem:
    if unit_system not in _SUPPORTED_UNITS:
        raise FlowContractError(
            "unsupported Eclipse unit_system",
            object_name="read_eclipse_case",
            field="unit_system",
            expected=tuple(sorted(_SUPPORTED_UNITS)),
            actual=unit_system,
        )
    return cast(UnitSystem, unit_system)


def _declared_units(text: str) -> UnitSystem:
    found: list[str] = []
    for raw in text.splitlines():
        token = raw.split("--", 1)[0].strip().upper()
        if token in _SUPPORTED_UNITS:
            found.append(token)
    unique = tuple(dict.fromkeys(found))
    if not unique:
        raise FlowContractError(
            "deck is missing an explicit unit declaration",
            object_name="read_eclipse_case",
            field="unit_system",
            expected=tuple(sorted(_SUPPORTED_UNITS)),
            actual=None,
        )
    if len(unique) != 1:
        raise FlowContractError(
            "deck has conflicting unit declarations",
            object_name="read_eclipse_case",
            field="unit_system",
            expected="exactly one declaration",
            actual=unique,
        )
    return cast(UnitSystem, unique[0])


def _source_factors(unit_system: UnitSystem) -> dict[str, float]:
    if unit_system == "FIELD":
        return {
            "length": _M_PER_FT,
            "permeability": _M2_PER_MD,
            "pressure": _PA_PER_PSI,
            "compressibility": 1.0 / _PA_PER_PSI,
            "viscosity": _PA_S_PER_CP,
            "time": _S_PER_DAY,
            "liquid_rate": _M3_PER_STB / _S_PER_DAY,
            "gas_rate": _M3_PER_SCF / _S_PER_DAY,
            "density": _KG_M3_PER_LBM_FT3,
            "solution_ratio": 1000.0 * _M3_PER_SCF / _M3_PER_STB,
            "gas_fvf": _M3_PER_STB / (1000.0 * _M3_PER_SCF),
        }
    if unit_system == "METRIC":
        return {
            "length": 1.0,
            "permeability": _M2_PER_MD,
            "pressure": _PA_PER_BAR,
            "compressibility": 1.0 / _PA_PER_BAR,
            "viscosity": _PA_S_PER_CP,
            "time": _S_PER_DAY,
            "liquid_rate": 1.0 / _S_PER_DAY,
            "gas_rate": 1.0 / _S_PER_DAY,
            "density": 1.0,
            "solution_ratio": 1.0,
            "gas_fvf": 1.0,
        }
    return {
        name: 1.0
        for name in (
            "length",
            "permeability",
            "pressure",
            "compressibility",
            "viscosity",
            "time",
            "liquid_rate",
            "gas_rate",
            "density",
            "solution_ratio",
            "gas_fvf",
        )
    }


def _convert_well(well: WellSpec, *, factors: dict[str, float]) -> WellSpec:
    mode = (well.control_mode or "").upper()
    allowed = (
        {"RATE", "RESV", "BHP"}
        if well.well_type == "INJ"
        else {"ORAT", "WRAT", "GRAT", "LRAT", "RESV", "BHP"}
    )
    if mode not in allowed:
        raise FlowContractError(
            "unsupported Eclipse well control mode",
            object_name="read_eclipse_case",
            field=well.name,
            expected=tuple(sorted(allowed)),
            actual=mode or None,
        )
    if mode == "BHP":
        target_factor = factors["pressure"]
    elif mode == "GRAT" or (
        mode == "RATE" and well.well_type == "INJ" and well.injection_phase == "gas"
    ):
        target_factor = factors["gas_rate"]
    else:
        target_factor = factors["liquid_rate"]
    return WellSpec(
        name=well.name,
        well_type=well.well_type,
        head_i=well.head_i,
        head_j=well.head_j,
        perforations=[
            (i, j, k, radius * factors["length"]) for i, j, k, radius in well.perforations
        ],
        control_mode=well.control_mode,
        target=None if well.target is None else well.target * target_factor,
        bhp_limit=(None if well.bhp_limit is None else well.bhp_limit * factors["pressure"]),
        injection_phase=well.injection_phase,
    )


def _to_si(deck: BlackOilDeck, *, source_units: UnitSystem) -> BlackOilDeck:
    factors = _source_factors(source_units)
    return replace(
        deck,
        dx=[value * factors["length"] for value in deck.dx],
        dy=[value * factors["length"] for value in deck.dy],
        dz=[value * factors["length"] for value in deck.dz],
        tops=[value * factors["length"] for value in deck.tops],
        permx=[value * factors["permeability"] for value in deck.permx],
        permy=[value * factors["permeability"] for value in deck.permy],
        permz=[value * factors["permeability"] for value in deck.permz],
        pvto=[
            (
                rs * factors["solution_ratio"],
                pressure * factors["pressure"],
                fvf,
                viscosity * factors["viscosity"],
            )
            for rs, pressure, fvf, viscosity in deck.pvto
        ],
        pvdg=[
            (
                pressure * factors["pressure"],
                fvf * factors["gas_fvf"],
                viscosity * factors["viscosity"],
            )
            for pressure, fvf, viscosity in deck.pvdg
        ],
        pvtw=(
            deck.pvtw[0] * factors["pressure"],
            deck.pvtw[1],
            deck.pvtw[2] * factors["compressibility"],
            deck.pvtw[3] * factors["viscosity"],
            deck.pvtw[4] * factors["compressibility"],
        ),
        rock=(
            deck.rock[0] * factors["pressure"],
            deck.rock[1] * factors["compressibility"],
        ),
        density=(
            deck.density[0] * factors["density"],
            deck.density[1] * factors["density"],
            deck.density[2] * factors["density"],
        ),
        swof=[
            (saturation, krw, kro, pressure * factors["pressure"])
            for saturation, krw, kro, pressure in deck.swof
        ],
        sgof=[
            (saturation, krg, kro, pressure * factors["pressure"])
            for saturation, krg, kro, pressure in deck.sgof
        ],
        wells=[_convert_well(well, factors=factors) for well in deck.wells],
        tstep=[value * factors["time"] for value in deck.tstep],
        unit_system="SI",
        source_unit_system=source_units,
    )


def read_eclipse_deck_si(
    path: Path, *, unit_system: Literal["FIELD", "METRIC", "SI"]
) -> BlackOilDeck:
    """Read one explicitly declared deck into a canonical SI deck.

    The requested unit system must match the deck's sole explicit declaration.
    This adapter is the single conversion boundary: its returned values must
    never be converted a second time.
    """
    requested = _validate_requested_units(unit_system)
    if not isinstance(path, Path):
        raise FlowContractError(
            "Eclipse adapter path must be pathlib.Path",
            object_name="read_eclipse_case",
            field="path",
            expected="pathlib.Path",
            actual=type(path).__qualname__,
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FlowContractError(
            "Eclipse deck cannot be read",
            object_name="read_eclipse_case",
            field="path",
            expected="readable UTF-8 file",
            actual=str(path),
            hint="verify that the deck path exists and is readable",
        ) from exc
    declared = _declared_units(text)
    if declared != requested:
        raise FlowContractError(
            "requested unit_system does not match the deck declaration",
            object_name="read_eclipse_case",
            field="unit_system",
            expected=declared,
            actual=requested,
        )
    try:
        deck = _parse_blackoil_deck_source(text)
    except GeoBrainError as exc:
        raise FlowContractError(
            exc.message,
            object_name="read_eclipse_case",
            field=exc.field,
            expected=exc.expected,
            actual=exc.actual,
            hint="correct the declared Eclipse deck record before importing it",
        ) from exc
    return _to_si(deck, source_units=declared)


def read_eclipse_case(path: Path, *, unit_system: Literal["FIELD", "METRIC", "SI"]) -> BlackOilCase:
    """Build an executable case only when the selected kernel accepts SI.

    The current :class:`BlackOilModel` is a FIELD-unit TPFA kernel.  The
    Eclipse adapter deliberately returns canonical SI data, so execution is
    rejected until an explicit SI-native kernel or an SI-to-FIELD kernel
    adapter exists.
    """
    deck = read_eclipse_deck_si(path, unit_system=unit_system)
    from ..models.black_oil import BlackOilModel

    schema = BlackOilModel.schema
    if schema.unit_system != "SI":
        raise FlowCapabilityError(
            "Flow execution requires a canonical SI model schema",
            object_name="read_eclipse_case",
            field="unit_system",
            expected={"execution_unit_system": "SI"},
            actual={
                "requested_unit_system": unit_system,
                "deck_unit_system": deck.unit_system,
                "source_unit_system": deck.source_unit_system,
                "model_name": schema.model_name,
                "execution_unit_system": schema.unit_system,
            },
            hint="Use an SI-native kernel or add an explicit SI-to-kernel adapter.",
        )
    return _build_blackoil_case(deck)


__all__ = ["read_eclipse_case", "read_eclipse_deck_si"]
