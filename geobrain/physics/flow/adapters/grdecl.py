"""Explicit GRDECL source-unit conversion into canonical SI records.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal, cast

import torch

from ..errors import FlowContractError
from ..discretization.mpfa3d import MPFAGrid3D
from ._eclipse_deck import DeckRecordStream, parse_deck_records

UnitSystem = Literal["FIELD", "METRIC", "SI"]

_LENGTH_KEYWORDS = frozenset({"COORD", "ZCORN", "DX", "DY", "DZ", "TOPS", "DEPTH"})
_PERMEABILITY_KEYWORDS = frozenset({"PERMX", "PERMY", "PERMZ", "PERMXY", "PERMXZ", "PERMYZ"})
_PRESSURE_KEYWORDS = frozenset({"PRESSURE", "PBUB", "BHP"})
_VISCOSITY_KEYWORDS = frozenset({"VISC", "VISCOSITY"})
_TIME_KEYWORDS = frozenset({"TSTEP", "TIME"})
_DIMENSIONLESS_KEYWORDS = frozenset(
    {
        "ACTNUM",
        "EQLNUM",
        "FIPNUM",
        "KRNUM",
        "MULTX",
        "MULTX-",
        "MULTY",
        "MULTY-",
        "MULTZ",
        "MULTZ-",
        "NTG",
        "OPERNUM",
        "PORO",
        "PVTNUM",
        "SATNUM",
        "SWATINIT",
    }
)
_STRUCTURED_KEYWORDS = frozenset({"NNC", "COMPDAT", "WELSPECS", "WCONPROD", "WCONINJE"})
_CONTROL_KEYWORDS = frozenset(
    {
        "EDIT",
        "END",
        "F",
        "FIELD",
        "GRID",
        "METRIC",
        "PROPS",
        "REGIONS",
        "RUNSPEC",
        "SCHEDULE",
        "SI",
        "SOLUTION",
        "SUMMARY",
    }
)


def _validate_unit_system(unit_system: object) -> UnitSystem:
    if unit_system not in {"FIELD", "METRIC", "SI"}:
        raise FlowContractError(
            "unsupported GRDECL unit_system",
            object_name="parse_grdecl_si",
            field="unit_system",
            expected=("FIELD", "METRIC", "SI"),
            actual=unit_system,
        )
    return cast(UnitSystem, unit_system)


def _validate_declared_units(parsed: DeckRecordStream, requested: UnitSystem) -> None:
    declarations = tuple(
        dict.fromkeys(
            block.keyword for block in parsed.blocks if block.keyword in {"FIELD", "METRIC", "SI"}
        )
    )
    if len(declarations) > 1:
        raise FlowContractError(
            "GRDECL source has conflicting unit declarations",
            object_name="parse_grdecl_si",
            field="unit_system",
            expected="at most one source declaration",
            actual=declarations,
        )
    if declarations and declarations[0] != requested:
        raise FlowContractError(
            "requested unit_system does not match the GRDECL declaration",
            object_name="parse_grdecl_si",
            field="unit_system",
            expected=declarations[0],
            actual=requested,
        )


def _source_factors(unit_system: UnitSystem) -> dict[str, float]:
    if unit_system == "FIELD":
        pressure = 6894.757293168
        length = 0.3048
        liquid_rate = 0.158987294928 / 86_400.0
        return {
            "length": length,
            "permeability": 9.869233e-16,
            "pressure": pressure,
            "viscosity": 1.0e-3,
            "time": 86_400.0,
            "liquid_rate": liquid_rate,
            "gas_rate": 0.028316846592 / 86_400.0,
            "transmissibility": liquid_rate * 1.0e-3 / pressure,
            "permeability_length": 9.869233e-16 * length,
        }
    if unit_system == "METRIC":
        pressure = 100_000.0
        liquid_rate = 1.0 / 86_400.0
        return {
            "length": 1.0,
            "permeability": 9.869233e-16,
            "pressure": pressure,
            "viscosity": 1.0e-3,
            "time": 86_400.0,
            "liquid_rate": liquid_rate,
            "gas_rate": liquid_rate,
            "transmissibility": liquid_rate * 1.0e-3 / pressure,
            "permeability_length": 9.869233e-16,
        }
    return {
        name: 1.0
        for name in (
            "length",
            "permeability",
            "pressure",
            "viscosity",
            "time",
            "liquid_rate",
            "gas_rate",
            "transmissibility",
            "permeability_length",
        )
    }


def _numeric(value: object, *, keyword: str, item: int) -> float:
    if not isinstance(value, (int, float)):
        raise FlowContractError(
            "GRDECL dimensional item must be numeric",
            object_name="parse_grdecl_si",
            field=f"{keyword}[{item}]",
            expected="number",
            actual=value,
        )
    return float(value)


def _scale_item(record: list[object], index: int, factor: float, *, keyword: str) -> None:
    if index >= len(record) or record[index] is None:
        return
    record[index] = _numeric(record[index], keyword=keyword, item=index) * factor


def _flat_records(records: Sequence[Sequence[object]]) -> list[object]:
    return [item for record in records for item in record]


def _scale_flat(records: Sequence[Sequence[object]], factor: float, *, keyword: str) -> list[float]:
    values = _flat_records(records)
    return [
        _numeric(value, keyword=keyword, item=index) * factor for index, value in enumerate(values)
    ]


def _convert_nnc(
    records: Sequence[Sequence[object]], factors: dict[str, float]
) -> list[list[object]]:
    converted: list[list[object]] = []
    for source in records:
        if len(source) < 6:
            raise FlowContractError(
                "NNC record requires six cell indices",
                object_name="parse_grdecl_si",
                field="NNC",
                expected=">= 6 items",
                actual=len(source),
            )
        record = list(source)
        _scale_item(record, 6, factors["transmissibility"], keyword="NNC")
        converted.append(record)
    return converted


def _convert_compdat(
    records: Sequence[Sequence[object]], factors: dict[str, float]
) -> list[list[object]]:
    converted: list[list[object]] = []
    for source in records:
        record = list(source)
        _scale_item(record, 7, factors["transmissibility"], keyword="COMPDAT")
        _scale_item(record, 8, factors["length"], keyword="COMPDAT")
        _scale_item(record, 9, factors["permeability_length"], keyword="COMPDAT")
        if len(record) > 11 and record[11] is not None:
            raise FlowContractError(
                "COMPDAT D-factor conversion is not supported",
                object_name="parse_grdecl_si",
                field="COMPDAT[11]",
                expected="defaulted item",
                actual=record[11],
            )
        converted.append(record)
    return converted


def _convert_welspecs(
    records: Sequence[Sequence[object]], factors: dict[str, float]
) -> list[list[object]]:
    converted = [list(record) for record in records]
    for record in converted:
        _scale_item(record, 4, factors["length"], keyword="WELSPECS")
    return converted


def _convert_wconprod(
    records: Sequence[Sequence[object]], factors: dict[str, float]
) -> list[list[object]]:
    converted: list[list[object]] = []
    for source in records:
        record = list(source)
        for index in (3, 4, 6, 7):
            _scale_item(record, index, factors["liquid_rate"], keyword="WCONPROD")
        _scale_item(record, 5, factors["gas_rate"], keyword="WCONPROD")
        for index in (8, 9):
            _scale_item(record, index, factors["pressure"], keyword="WCONPROD")
        if len(record) > 11 and record[11] is not None:
            raise FlowContractError(
                "WCONPROD artificial-lift quantity conversion is not supported",
                object_name="parse_grdecl_si",
                field="WCONPROD[11]",
                expected="defaulted item",
                actual=record[11],
            )
        converted.append(record)
    return converted


def _convert_wconinje(
    records: Sequence[Sequence[object]], factors: dict[str, float]
) -> list[list[object]]:
    converted: list[list[object]] = []
    for source in records:
        record = list(source)
        phase = str(record[1]).upper() if len(record) > 1 else ""
        rate_factor = factors["gas_rate"] if phase.startswith("GAS") else factors["liquid_rate"]
        _scale_item(record, 4, rate_factor, keyword="WCONINJE")
        _scale_item(record, 5, factors["liquid_rate"], keyword="WCONINJE")
        for index in (6, 7):
            _scale_item(record, index, factors["pressure"], keyword="WCONINJE")
        converted.append(record)
    return converted


def _convert_structured(
    keyword: str,
    records: Sequence[Sequence[object]],
    factors: dict[str, float],
) -> list[list[object]]:
    if keyword == "NNC":
        return _convert_nnc(records, factors)
    if keyword == "COMPDAT":
        return _convert_compdat(records, factors)
    if keyword == "WELSPECS":
        return _convert_welspecs(records, factors)
    if keyword == "WCONPROD":
        return _convert_wconprod(records, factors)
    return _convert_wconinje(records, factors)


def parse_grdecl_si(text: str, *, unit_system: UnitSystem) -> dict[str, object]:
    """Parse supported GRDECL/deck records and convert every dimensional item to SI.

    Bulk arrays stay flattened for the geometry consumers. Mixed NNC and well
    keywords retain record boundaries and defaulted ``N*`` items as ``None`` so
    each column can be converted according to its declared physical meaning.
    Unknown populated keywords fail instead of being silently treated as
    dimensionless.
    """
    source_units = _validate_unit_system(unit_system)
    factors = _source_factors(source_units)
    parsed = parse_deck_records(text)
    _validate_declared_units(parsed, source_units)
    converted: dict[str, object] = {}
    for keyword, records in parsed.items():
        if keyword in {"SPECGRID", "DIMENS"}:
            values = _flat_records(records)
            if len(values) < 3:
                raise FlowContractError(
                    "GRDECL dimensions require NX, NY, and NZ",
                    object_name="parse_grdecl_si",
                    field=keyword,
                    expected=">= 3 items",
                    actual=len(values),
                )
            numeric_dimensions = tuple(
                _numeric(values[i], keyword=keyword, item=i) for i in range(3)
            )
            if any(
                not math.isfinite(value) or value <= 0 or not value.is_integer()
                for value in numeric_dimensions
            ):
                raise FlowContractError(
                    "GRDECL dimensions must be finite positive integers",
                    object_name="parse_grdecl_si",
                    field=keyword,
                    expected="three finite positive integers",
                    actual=numeric_dimensions,
                )
            converted["dims"] = tuple(int(value) for value in numeric_dimensions)
        elif keyword in _STRUCTURED_KEYWORDS:
            converted[keyword] = _convert_structured(keyword, records, factors)
        elif keyword in _LENGTH_KEYWORDS:
            converted[keyword] = _scale_flat(records, factors["length"], keyword=keyword)
        elif keyword in _PERMEABILITY_KEYWORDS:
            converted[keyword] = _scale_flat(records, factors["permeability"], keyword=keyword)
        elif keyword in _PRESSURE_KEYWORDS:
            converted[keyword] = _scale_flat(records, factors["pressure"], keyword=keyword)
        elif keyword in _VISCOSITY_KEYWORDS:
            converted[keyword] = _scale_flat(records, factors["viscosity"], keyword=keyword)
        elif keyword in _TIME_KEYWORDS:
            converted[keyword] = _scale_flat(records, factors["time"], keyword=keyword)
        elif keyword == "PORV":
            converted[keyword] = _scale_flat(records, factors["length"] ** 3, keyword=keyword)
        elif keyword in _DIMENSIONLESS_KEYWORDS:
            converted[keyword] = _scale_flat(records, 1.0, keyword=keyword)
        elif keyword in _CONTROL_KEYWORDS and not _flat_records(records):
            continue
        elif records:
            raise FlowContractError(
                "unsupported dimensional GRDECL keyword",
                object_name="parse_grdecl_si",
                field=keyword,
                expected="a keyword with an explicit SI conversion rule",
                actual=keyword,
            )
    return converted


def read_grdecl_grid_si(
    text: str,
    *,
    unit_system: UnitSystem,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> tuple[MPFAGrid3D, torch.Tensor]:
    """Build an MPFA grid from a declared-unit GRDECL source exactly once.

    External numeric records are converted by :func:`parse_grdecl_si` before
    tensor construction. Returned nodes already use the requested dtype/device,
    canonical ``(x, y, z)`` metre columns, positive-down depth, and x-fastest
    cell order; stencil builders therefore never need to cast or move geometry.
    """

    if dtype not in {torch.float32, torch.float64}:
        raise FlowContractError(
            "GRDECL grid dtype must be a supported real floating dtype",
            object_name="read_grdecl_grid_si",
            field="dtype",
            expected=(str(torch.float32), str(torch.float64)),
            actual=str(dtype),
        )
    target_device = torch.device(device)
    converted = parse_grdecl_si(text, unit_system=unit_system)
    missing = tuple(keyword for keyword in ("dims", "COORD", "ZCORN") if keyword not in converted)
    if missing:
        raise FlowContractError(
            "GRDECL grid requires dimensions, COORD, and ZCORN",
            object_name="read_grdecl_grid_si",
            field="keywords",
            expected=("dims", "COORD", "ZCORN"),
            actual={"missing": missing},
        )
    dimensions = converted["dims"]
    coord = converted["COORD"]
    zcorn = converted["ZCORN"]
    if not isinstance(dimensions, tuple) or len(dimensions) != 3:
        raise FlowContractError(
            "GRDECL dimensions must contain NX, NY, and NZ",
            object_name="read_grdecl_grid_si",
            field="dims",
            expected="tuple[int, int, int]",
            actual=dimensions,
        )
    if not isinstance(coord, list) or not isinstance(zcorn, list):
        raise FlowContractError(
            "GRDECL geometry arrays must be flat numeric lists",
            object_name="read_grdecl_grid_si",
            field="geometry",
            expected="COORD and ZCORN lists",
            actual=(type(coord).__name__, type(zcorn).__name__),
        )

    from ..discretization.mpfa3d import build_mpfa_grid_3d
    from ..grid.corner_point import corner_point_to_hex

    nodes, cell_nodes = corner_point_to_hex(
        dimensions,
        coord,
        zcorn,
        dtype=dtype,
    )
    if target_device.type != "cpu":
        nodes = nodes.to(device=target_device)
    grid = build_mpfa_grid_3d(nodes, cell_nodes)
    nx, ny, nz = dimensions
    active_values = converted.get("ACTNUM", [1.0] * (nx * ny * nz))
    if not isinstance(active_values, list) or len(active_values) != nx * ny * nz:
        raise FlowContractError(
            "ACTNUM length must equal the x-fastest Cartesian cell count",
            object_name="read_grdecl_grid_si",
            field="ACTNUM",
            expected=nx * ny * nz,
            actual=(
                len(active_values)
                if isinstance(active_values, list)
                else type(active_values).__name__
            ),
        )
    active = torch.tensor(active_values, dtype=torch.int64, device=target_device)
    return grid, active


__all__ = ["parse_grdecl_si", "read_grdecl_grid_si"]
