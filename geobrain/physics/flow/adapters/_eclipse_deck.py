"""Private Eclipse record parser and canonical-SI case assembler.

Parses the classic keyword-deck format used by the SPE comparative-solution
benchmarks (Odeh 1981 SPE1 being the canonical three-phase live-oil case) and
assembles a runnable GeoBrain black-oil problem from it: the Cartesian grid, the
per-cell rock (porosity + diagonal permeability), the three-phase PVT (live-oil
``PVTO``, dry-gas ``PVDG``, water ``PVTW``), the relative-permeability tables
(``SWOF`` / ``SGOF``), the wells (``WELSPECS`` / ``COMPDAT`` / ``WCONPROD`` /
``WCONINJE``) and the report-step schedule (``TSTEP``).

Scope is deliberately small: the keyword subset SPE1 needs, but the parsing is
faithful: ``--`` comments, ``/`` record/block terminators, the ``N*value``
repeat-count shorthand, and the per-record ``/`` inside tabular keywords (``PVTO``
saturated rows, ``SWOF`` rows, one ``COMPDAT`` connection per record) are all
honoured. The private record tokenizer is shared by the canonical-SI Eclipse and
GRDECL adapters.

The private source parser preserves the declared deck units. The public Eclipse
adapter performs the single FIELD/METRIC-to-SI conversion boundary before this
module assembles executable Flow objects; `_build_blackoil_case` therefore
accepts only a canonical-SI ``BlackOilDeck``.

Live-oil ``R_s(p)``, ``B_o(p)``, ``μ_o(p)`` come from the saturated ``PVTO``
branch through :class:`~geobrain.physics.flow.properties.PVTLiveOilTable`; dry gas
and water through :class:`~geobrain.physics.flow.properties.PVTTable` /
:class:`~geobrain.physics.flow.properties.PVTAnalytic`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import shlex
from types import MappingProxyType
from typing import Literal, cast, overload

import torch

from ....core import GeoBrainError
from .._defaults import DTYPE
from ..errors import FlowContractError
from ..grid import CartGrid
from ..properties import (
    BlackOilFluid,
    PVTAnalytic,
    PVTLiveOilTable,
    PVTTable,
    RelPermCorey,
    RelPermTable,
    Rock,
    ThreePhaseRelPerm,
)
from ..wells import Perforation, Well, compute_well_index

_SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")

# Bare control keywords have no slash-delimited data record. Their block ends
# when the next known keyword starts at a source-line boundary.
_BARE_CTRL = frozenset(
    {
        "INIT",
        "NOECHO",
        "ECHO",
        "UNIFIN",
        "UNIFOUT",
        "NOSIM",
        "RPTGRID",
        "RPTSOL",
        "RPTSCHED",
        "RPTRST",
    }
)
_BARE_KEYWORDS = frozenset(
    {
        *_SECTIONS,
        *_BARE_CTRL,
        "TITLE",
        "FIELD",
        "METRIC",
        "SI",
        "OIL",
        "GAS",
        "WATER",
        "DISGAS",
        "END",
        "F",
    }
)
# ---------------------------------------------------------------------------
# Record-structured keyword tokenizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeckKeywordBlock:
    """One keyword occurrence with its slash-delimited records."""

    keyword: str
    records: tuple[tuple[object, ...], ...]


class DeckRecordStream(Mapping[str, tuple[tuple[object, ...], ...]]):
    """Immutable record mapping plus the exact source-order keyword blocks."""

    __slots__ = ("_records", "blocks")

    def __init__(self, blocks: Sequence[DeckKeywordBlock]) -> None:
        frozen_blocks = tuple(blocks)
        aggregated: dict[str, list[tuple[object, ...]]] = {}
        for block in frozen_blocks:
            aggregated.setdefault(block.keyword, []).extend(block.records)
        self._records = MappingProxyType(
            {keyword: tuple(records) for keyword, records in aggregated.items()}
        )
        self.blocks = frozen_blocks

    def __getitem__(self, keyword: str) -> tuple[tuple[object, ...], ...]:
        return self._records[keyword]

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True, slots=True)
class _DeckToken:
    text: str
    line_number: int
    keyword_column: bool


def _deck_tokens(text: str) -> tuple[_DeckToken, ...]:
    tokens: list[_DeckToken] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("--", 1)[0]
        lexer = shlex.shlex(line, posix=False, punctuation_chars="/")
        lexer.whitespace_split = True
        lexer.commenters = ""
        begins_in_keyword_column = bool(line) and not line[0].isspace()
        for index, token in enumerate(lexer):
            tokens.append(
                _DeckToken(
                    token,
                    line_number,
                    keyword_column=index == 0 and begins_in_keyword_column,
                )
            )
    return tuple(tokens)


def _keyword_token(token: str) -> str | None:
    if token.startswith(("'", '"')) or not token[:1].isalpha():
        return None
    return token.upper()


def parse_deck_records(text: str) -> DeckRecordStream:
    """Tokenize a keyword deck into ``keyword → list-of-records``.

    A *record* is the run of tokens between two ``/`` terminators; tabular
    keywords (``PVTO``, ``SWOF``, ``COMPDAT``, …) therefore yield one record per
    physical data line/row, while a single-record keyword (``DIMENS``, ``EQUIL``)
    yields exactly one record. The ``N*value`` repeat shorthand is expanded and
    quoted well names keep their quotes stripped. Section headers and pure
    control keywords with no data carry an empty record list.

    The canonical-SI adapters flatten only the dimensionally declared records
    they own; this parser preserves source record boundaries and defaulted
    values for control keywords.
    """
    tokens = _deck_tokens(text)
    blocks: list[DeckKeywordBlock] = []
    i, n = 0, len(tokens)
    while i < n:
        source_keyword = tokens[i]
        kw = _keyword_token(source_keyword.text) if source_keyword.keyword_column else None
        i += 1
        if kw is None:
            continue
        records: list[tuple[object, ...]] = []
        record: list[object] = []
        at_record_boundary = True
        while i < n:
            source_token = tokens[i]
            candidate = _keyword_token(source_token.text)
            keyword_boundary = (
                candidate is not None
                and source_token.keyword_column
                and (at_record_boundary or kw in _BARE_KEYWORDS)
            )
            if keyword_boundary:
                break
            tok = source_token.text
            i += 1
            if tok == "/":
                if not record and at_record_boundary:
                    break
                records.append(tuple(record))
                record = []
                at_record_boundary = True
                continue
            if "*" in tok and tok[0].isdigit():
                cnt, _, val = tok.partition("*")
                expanded = [_coerce(val)] * int(cnt) if val != "" else [None] * int(cnt)
                record.extend(expanded)
            else:
                record.append(_coerce(tok))
            at_record_boundary = False
        if record:
            records.append(tuple(record))
        blocks.append(DeckKeywordBlock(kw, tuple(records)))
    return DeckRecordStream(blocks)


def _coerce(tok: str) -> float | str | None:
    """Token → float if numeric, else the unquoted string (``1*`` default ⇒ None)."""
    if tok == "1*":
        return None
    t = tok.strip("'")
    try:
        return float(t)
    except ValueError:
        return t


# ---------------------------------------------------------------------------
# Parsed deck container
# ---------------------------------------------------------------------------


@dataclass
class WellSpec:
    """One well as read from ``WELSPECS`` + ``COMPDAT`` + a control keyword."""

    name: str
    well_type: str  # 'INJ' or 'PROD'
    head_i: int  # 1-based head location (WELSPECS)
    head_j: int
    perforations: list[tuple[int, int, int, float]] = field(default_factory=list)
    # control:
    control_mode: str | None = None  # 'ORAT' / 'RATE' / 'BHP' …
    target: float | None = None
    bhp_limit: float | None = None
    injection_phase: str | None = None  # 'water' / 'gas'


@dataclass
class BlackOilDeck:
    """Parsed black-oil deck: grid dims, rock arrays, PVT tables, wells, schedule.

    The private source parser preserves the explicitly declared deck units and
    records them in ``unit_system``. The public Eclipse adapter returns a copied
    record with all physical values converted once to canonical SI,
    ``unit_system="SI"``, and the declaration retained in ``source_unit_system``.
    Only the canonical-SI form may be passed to the case assembler.
    """

    dims: tuple[int, int, int]
    dx: list[float]
    dy: list[float]
    dz: list[float]
    tops: list[float]
    poro: list[float]
    permx: list[float]
    permy: list[float]
    permz: list[float]
    pvto: list[tuple[float, float, float, float]]  # (Rs, bubble pressure, Bo, mu_o)
    pvdg: list[tuple[float, float, float]]  # (pressure, Bg, mu_g)
    pvtw: tuple[float, float, float, float, float]  # (pressure, Bw, cw, mu_w, cv)
    rock: tuple[float, float]  # (reference pressure, compressibility)
    density: tuple[float, float, float]  # (surface oil, water, gas density)
    swof: list[tuple[float, float, float, float]]
    sgof: list[tuple[float, float, float, float]]
    wells: list[WellSpec]
    tstep: list[float]
    unit_system: str = "UNDECLARED"
    source_unit_system: str = "UNDECLARED"

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * nz


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _flat(deck: Mapping[str, Sequence[Sequence[object]]], kw: str) -> list[float]:
    """Flatten all records of a keyword into one value list."""
    out: list[float] = []
    for rec in deck.get(kw, []):
        for index, value in enumerate(rec):
            if not isinstance(value, (int, float)):
                raise GeoBrainError(
                    "deck numeric record contains a non-numeric item",
                    object_name="parse_blackoil_deck",
                    field=f"{kw}[{index}]",
                    expected="number",
                    actual=value,
                )
            out.append(float(value))
    return out


def _number(value: object, *, keyword: str, index: int) -> float:
    if not isinstance(value, (int, float)):
        raise GeoBrainError(
            "deck record item must be numeric",
            object_name="parse_blackoil_deck",
            field=f"{keyword}[{index}]",
            expected="number",
            actual=value,
        )
    return float(value)


def _parse_blackoil_deck_source(text: str) -> BlackOilDeck:
    """Parse a black-oil keyword deck (SPE1 keyword set) into a :class:`BlackOilDeck`."""
    rec = parse_deck_records(text)
    flat = {
        keyword: _flat(rec, keyword)
        for keyword in ("DX", "DY", "DZ", "TOPS", "PORO", "PERMX", "PERMY", "PERMZ")
        if keyword in rec
    }

    dims_rec = next((r for r in rec.get("DIMENS", rec.get("SPECGRID", [])) if len(r) >= 3), None)
    if dims_rec is None:
        raise GeoBrainError(
            "deck is missing DIMENS / SPECGRID",
            object_name="parse_blackoil_deck",
            field="DIMENS",
            expected="DIMENS NX NY NZ /",
            actual=sorted(flat)[:8],
        )
    nx, ny, nz = (
        int(_number(dims_rec[0], keyword="DIMENS", index=0)),
        int(_number(dims_rec[1], keyword="DIMENS", index=1)),
        int(_number(dims_rec[2], keyword="DIMENS", index=2)),
    )
    n = nx * ny * nz

    @overload
    def grid_arr(kw: str, optional: Literal[False] = False) -> list[float]: ...

    @overload
    def grid_arr(kw: str, optional: Literal[True]) -> list[float] | None: ...

    def grid_arr(kw: str, optional: bool = False) -> list[float] | None:
        vals = flat.get(kw)
        if vals is None:
            if optional:
                return None
            raise GeoBrainError(
                f"deck is missing grid keyword {kw}",
                object_name="parse_blackoil_deck",
                field=kw,
                expected=f"{n} values",
                actual=None,
            )
        vals = [float(v) for v in vals]
        if len(vals) != n and not (kw == "TOPS" and len(vals) == nx * ny):
            raise GeoBrainError(
                f"{kw} has wrong length",
                object_name="parse_blackoil_deck",
                field=kw,
                expected=n,
                actual=len(vals),
            )
        return vals

    permx = grid_arr("PERMX")
    permy = grid_arr("PERMY", optional=True) or permx
    permz = grid_arr("PERMZ", optional=True) or permx

    # --- PVTO: each record is a saturated row (Rs pb Bo mu) plus optional
    #     undersaturated continuation rows we ignore (only the first triple
    #     after Rs is the saturated point). ---
    pvto: list[tuple[float, float, float, float]] = []
    for r in rec.get("PVTO", []):
        nums = [v for v in r if isinstance(v, (int, float))]
        if len(nums) >= 4:
            pvto.append((nums[0], nums[1], nums[2], nums[3]))
    if not pvto:
        raise GeoBrainError(
            "deck is missing a usable PVTO live-oil table",
            object_name="parse_blackoil_deck",
            field="PVTO",
            expected=">=2 saturated rows",
            actual=len(pvto),
        )

    # PVDG is a single ``/``-terminated record holding all (p, Bg, mu) triples.
    pvdg: list[tuple[float, float, float]] = []
    for r in rec.get("PVDG", []):
        nums = [v for v in r if isinstance(v, (int, float))]
        for k in range(0, len(nums) - 2, 3):
            pvdg.append((nums[k], nums[k + 1], nums[k + 2]))
    pvtw_rec = next((r for r in rec.get("PVTW", []) if len(r) >= 4), None)
    if pvtw_rec is None:
        raise GeoBrainError(
            "deck is missing PVTW",
            object_name="parse_blackoil_deck",
            field="PVTW",
            expected="1 record",
            actual=None,
        )
    pvtw = (
        _number(pvtw_rec[0], keyword="PVTW", index=0),
        _number(pvtw_rec[1], keyword="PVTW", index=1),
        _number(pvtw_rec[2], keyword="PVTW", index=2),
        _number(pvtw_rec[3], keyword="PVTW", index=3),
        _number(pvtw_rec[4], keyword="PVTW", index=4) if len(pvtw_rec) > 4 else 0.0,
    )

    rock_rec = next((r for r in rec.get("ROCK", []) if len(r) >= 2), None)
    if rock_rec is None:
        # Inert TODAY only because cref=0.0 removes pref from phi(p); the
        # 14.7 value is FIELD psia and would be ~14x wrong read as barsa in
        # a METRIC deck, so a defaulted ROCK is never silent.
        warnings.warn(
            "deck has no ROCK record; defaulting to pref=14.7 (FIELD psia), "
            "cref=0.0 (incompressible rock). Supply ROCK explicitly for "
            "METRIC decks or compressible rock.",
            stacklevel=2,
        )
        rock_rec = [14.7, 0.0]
    dens_rec = next((r for r in rec.get("DENSITY", []) if len(r) >= 3), None)
    if dens_rec is None:
        raise GeoBrainError(
            "deck is missing DENSITY",
            object_name="parse_blackoil_deck",
            field="DENSITY",
            expected="rho_o rho_w rho_g",
            actual=None,
        )

    def table4(kw: str) -> list[tuple[float, float, float, float]]:
        rows: list[tuple[float, float, float, float]] = []
        for r in rec.get(kw, []):
            nums = [v for v in r if isinstance(v, (int, float))]
            for k in range(0, len(nums) - 3, 4):
                rows.append((nums[k], nums[k + 1], nums[k + 2], nums[k + 3]))
        return rows

    swof = table4("SWOF")
    sgof = table4("SGOF")

    wells = _parse_wells(rec, nx)
    tstep = _flat(rec, "TSTEP") if "TSTEP" in rec else []
    tstep = [float(t) for t in tstep]

    return BlackOilDeck(
        dims=(nx, ny, nz),
        dx=grid_arr("DX"),
        dy=grid_arr("DY"),
        dz=grid_arr("DZ"),
        tops=grid_arr("TOPS"),
        poro=grid_arr("PORO"),
        permx=permx,
        permy=permy,
        permz=permz,
        pvto=pvto,
        pvdg=pvdg,
        pvtw=pvtw,
        rock=(
            _number(rock_rec[0], keyword="ROCK", index=0),
            _number(rock_rec[1], keyword="ROCK", index=1),
        ),
        density=(
            _number(dens_rec[0], keyword="DENSITY", index=0),
            _number(dens_rec[1], keyword="DENSITY", index=1),
            _number(dens_rec[2], keyword="DENSITY", index=2),
        ),
        swof=swof,
        sgof=sgof,
        wells=wells,
        tstep=tstep,
        unit_system="FIELD" if "FIELD" in rec else "METRIC",
        source_unit_system="FIELD" if "FIELD" in rec else "METRIC",
    )


def _parse_wells(rec: Mapping[str, Sequence[Sequence[object]]], nx: int) -> list[WellSpec]:
    specs: dict[str, WellSpec] = {}
    for r in rec.get("WELSPECS", []):
        if len(r) < 4 or not isinstance(r[0], str):
            continue
        name = r[0]
        head_i = int(_number(r[2], keyword="WELSPECS", index=2))
        head_j = int(_number(r[3], keyword="WELSPECS", index=3))
        phase = str(r[5]).lower() if len(r) > 5 and isinstance(r[5], str) else None
        specs[name] = WellSpec(
            name=name, well_type="PROD", head_i=head_i, head_j=head_j, injection_phase=phase
        )
    for r in rec.get("COMPDAT", []):
        if len(r) < 5 or not isinstance(r[0], str):
            continue
        name = r[0]
        if name not in specs:
            specs[name] = WellSpec(
                name=name,
                well_type="PROD",
                head_i=int(_number(r[1], keyword="COMPDAT", index=1)),
                head_j=int(_number(r[2], keyword="COMPDAT", index=2)),
            )
        i = int(_number(r[1], keyword="COMPDAT", index=1))
        j = int(_number(r[2], keyword="COMPDAT", index=2))
        k1 = int(_number(r[3], keyword="COMPDAT", index=3))
        k2 = int(_number(r[4], keyword="COMPDAT", index=4))
        # COMPDAT item 9 is the wellbore diameter; radius = diameter/2.
        rw = (float(r[8]) / 2.0) if (len(r) > 8 and isinstance(r[8], (int, float))) else 0.25
        for k in range(k1, k2 + 1):
            specs[name].perforations.append((i, j, k, rw))
    for r in rec.get("WCONPROD", []):
        if not r or not isinstance(r[0], str) or r[0] not in specs:
            continue
        w = specs[r[0]]
        w.well_type = "PROD"
        mode = str(r[2]).upper() if len(r) > 2 and isinstance(r[2], str) else ""
        target_index = {
            "ORAT": 3,
            "WRAT": 4,
            "GRAT": 5,
            "LRAT": 6,
            "RESV": 7,
            "BHP": 8,
        }.get(mode)
        w.control_mode = mode or None
        w.target = _record_number(r, target_index)
        w.bhp_limit = None if mode == "BHP" else _record_number(r, 8)
    for r in rec.get("WCONINJE", []):
        if not r or not isinstance(r[0], str) or r[0] not in specs:
            continue
        w = specs[r[0]]
        w.well_type = "INJ"
        phase = str(r[1]).lower() if len(r) > 1 and isinstance(r[1], str) else "gas"
        w.injection_phase = "water" if phase.startswith("wat") else "gas"
        mode = str(r[3]).upper() if len(r) > 3 and isinstance(r[3], str) else "RATE"
        target_index = {"RATE": 4, "RESV": 5, "BHP": 6}.get(mode)
        w.control_mode = mode
        w.target = _record_number(r, target_index)
        w.bhp_limit = None if mode == "BHP" else _record_number(r, 6)
    return list(specs.values())


def _record_number(record: Sequence[object], index: int | None) -> float | None:
    """Return one typed Eclipse record item without shifting across defaults."""
    if index is None or index >= len(record):
        return None
    value = record[index]
    return float(value) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------


@dataclass
class BlackOilCase:
    """A runnable black-oil case assembled from a deck."""

    grid: CartGrid
    rock: Rock
    fluid: BlackOilFluid
    wells: list[Well]
    well_heads: dict[str, int]  # well name → head cell id
    tstep: list[float]
    deck: BlackOilDeck


def _cart_grid_from_deck(deck: BlackOilDeck, dtype: torch.dtype) -> CartGrid:
    nx, ny, nz = deck.dims
    # SPE1 (and the SPE benchmarks generally) use a block-uniform DX/DY per axis
    # and a layer-uniform DZ. Read those off the deck arrays (cell ordering is
    # x-fastest: id = i + j*nx + k*nx*ny).
    dx_axis = [deck.dx[i] for i in range(nx)]
    dy_axis = [deck.dy[j * nx] for j in range(ny)]
    dz_axis = [deck.dz[k * nx * ny] for k in range(nz)]
    # Top of the first layer (datum for cell depths). TOPS may be per top-layer
    # cell (nx*ny) or per cell; take the mean top as the grid origin in z.
    oz = float(sum(deck.tops) / len(deck.tops)) if deck.tops else 0.0
    return CartGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx_m=torch.tensor(dx_axis, dtype=dtype),
        dy_m=torch.tensor(dy_axis, dtype=dtype),
        dz_m=torch.tensor(dz_axis, dtype=dtype),
        origin_m=(0.0, 0.0, oz),
        dtype=dtype,
    )


def _relperm_from_deck(deck: BlackOilDeck, dtype: torch.dtype) -> ThreePhaseRelPerm:
    """Build a 3-phase relperm from SWOF/SGOF if present, else a Corey fallback.

    ``ThreePhaseRelPerm`` (Stone-II) needs two-phase curves exposing
    ``kr_oil``/``kr_water``; the oil-gas pair uses ``kr_water`` as ``kr_gas``
    (gas non-wetting). SWOF columns are ``(Sw, krw, kro, Pc)``, used directly.
    SGOF columns are ``(Sg, krg, kro, Pc)``; mapped to the og pair's table
    ``(Sg, krg→"krw", kro, Pc)`` so ``og.kr_water(Sg)=krg`` and
    ``og.kr_oil(Sg)=kro``. ``RelPermTable`` requires strictly increasing first
    column and ≥2 rows; SPE1's SWOF/SGOF satisfy both."""
    if deck.swof and deck.sgof and len(deck.swof) >= 2 and len(deck.sgof) >= 2:
        ow = RelPermTable(torch.tensor(deck.swof, dtype=dtype), bounds_policy="constant")
        og = RelPermTable(
            torch.tensor([(g, krg, kro, pc) for (g, krg, kro, pc) in deck.sgof], dtype=dtype),
            bounds_policy="constant",
        )
        return ThreePhaseRelPerm(ow=ow, og=og)
    ow = RelPermCorey(swc=0.12, sor=0.0, n_w=2.0, n_o=2.0, kr_w_max=1.0, kr_o_max=1.0)
    og = RelPermCorey(swc=0.0, sor=0.0, n_w=2.0, n_o=2.0, kr_w_max=1.0, kr_o_max=1.0)
    return ThreePhaseRelPerm(ow=ow, og=og)


def _build_blackoil_case(deck: BlackOilDeck, *, dtype: torch.dtype = DTYPE) -> BlackOilCase:
    """Assemble grid + rock + 3-phase fluid + wells + schedule from a parsed deck."""
    if deck.unit_system != "SI":
        raise FlowContractError(
            "case assembly requires a canonical SI deck",
            object_name="_build_blackoil_case",
            field="unit_system",
            expected="SI",
            actual=deck.unit_system,
            hint="Use read_eclipse_deck_si() before assembling a case.",
        )
    nx, ny, nz = deck.dims
    n = deck.n_cells
    grid = _cart_grid_from_deck(deck, dtype)

    perm = torch.tensor(
        [[deck.permx[c], deck.permy[c], deck.permz[c]] for c in range(n)], dtype=dtype
    )
    rock = Rock(
        permeability_m2=perm,
        porosity=torch.tensor(deck.poro, dtype=dtype),
        reference_pressure_pa=float(deck.rock[0]),
        compressibility_pa_inv=float(deck.rock[1]),
    )

    rho_o, rho_w, rho_g = deck.density

    # --- live oil from the canonical-SI saturated PVTO branch ---
    pvto_tab = torch.tensor(deck.pvto, dtype=dtype)
    pvt_o = PVTLiveOilTable(
        solution_gas_oil_ratio_m3_m3=pvto_tab[:, 0],
        bubble_pressure_pa=pvto_tab[:, 1],
        formation_volume_factor=pvto_tab[:, 2],
        viscosity_pa_s=pvto_tab[:, 3],
        surface_oil_density_kg_m3=rho_o,
        surface_gas_density_kg_m3=rho_g,
        bounds_policy="constant",
    )

    # --- dry gas from the canonical-SI PVDG table ---
    pvdg_tab = torch.tensor(deck.pvdg, dtype=dtype)
    pvt_g = PVTTable(
        pressure_pa=pvdg_tab[:, 0],
        formation_volume_factor=pvdg_tab[:, 1],
        viscosity_pa_s=pvdg_tab[:, 2],
        surface_density_kg_m3=rho_g,
        bounds_policy="constant",
    )

    # --- water: PVTW const-compressibility (p_ref, Bw, cw, mu_w) ---
    p_ref_w, bw_ref, cw, mu_w, _ = deck.pvtw
    pvt_w = PVTAnalytic(
        density_ref_kg_m3=rho_w / bw_ref,
        viscosity_ref_pa_s=mu_w,
        formation_volume_factor_ref=bw_ref,
        reference_pressure_pa=p_ref_w,
        compressibility_pa_inv=cw,
        dtype=dtype,
    )

    relperm = _relperm_from_deck(deck, dtype)
    fluid = BlackOilFluid(pvt_o=pvt_o, pvt_w=pvt_w, pvt_g=pvt_g, relperm=relperm)

    # --- wells (Peaceman well index per perforation) ---
    wells: list[Well] = []
    heads: dict[str, int] = {}
    dx_m, dy_m, dz_m = grid._axis_widths_view()
    dx_axis = [dx_m[i].item() for i in range(nx)]
    dy_axis = [dy_m[j].item() for j in range(ny)]
    dz_axis = [dz_m[k].item() for k in range(nz)]
    # Cell-centre depths (z, +down): the per-perforation hydrostatic datum for
    # the implicit-BHP well model. Each perforation's depth_offset_m is its cell-centre
    # depth below the well's top (shallowest) perforation, so a multi-layer well
    # referenced to one BHP sees the correct per-connection hydrostatic head.
    z_depth = grid._cell_centers_view()[:, 2]
    from ..wells import BHPControl, RateControl, WellRateKind, WellStandardConditions

    for ws in deck.wells:
        # First pass: gather (cell_idx, wi, depth) so the datum (top perforation)
        # is known before each Perforation's gdz can be set.
        connections: list[tuple[int, float, float]] = []
        for i, j, k, rw in ws.perforations:
            cid = (i - 1) + (j - 1) * nx + (k - 1) * nx * ny  # 1-based deck → 0-based
            wi = compute_well_index(
                dx_m=dx_axis[i - 1],
                dy_m=dy_axis[j - 1],
                dz_m=dz_axis[k - 1],
                kx_m2=deck.permx[cid],
                ky_m2=deck.permy[cid],
                well_radius_m=max(rw, 1e-3),
            )
            connections.append((cid, wi, float(z_depth[cid])))
        if not connections:
            continue
        z_ref = min(c[2] for c in connections)  # top-perforation datum
        perfs = [
            Perforation(
                cell_idx=cid,
                well_index_m3=wi,
                depth_offset_m=depth - z_ref,
            )
            for (cid, wi, depth) in connections
        ]
        head_cid = (ws.head_i - 1) + (ws.head_j - 1) * nx
        heads[ws.name] = head_cid
        datum_depth = z_ref
        # Surface-rate control for both producers (ORAT/total) and injectors
        # (RATE of injected phase), with the deck's BHP limit kept ALONGSIDE the
        # rate target (WCONPROD BHP floor / WCONINJE BHP ceiling) so the implicit
        # well model can switch a rate well onto its BHP limit between solves.
        # Falls back gracefully if no rate target was parsed.
        mode = (ws.control_mode or "").upper()
        control: BHPControl | RateControl
        if mode == "BHP":
            if ws.target is None:
                raise GeoBrainError(
                    "BHP-controlled well is missing its pressure target",
                    object_name="_build_blackoil_case",
                    field=ws.name,
                    expected="numeric BHP target",
                    actual=ws.target,
                )
            control = BHPControl(pressure_pa=float(ws.target))
        elif ws.target is not None:
            if mode == "RATE":
                rate_kind = (
                    WellRateKind.WRAT if ws.injection_phase == "water" else WellRateKind.GRAT
                )
            else:
                try:
                    rate_kind = WellRateKind(mode)
                except ValueError as error:
                    raise GeoBrainError(
                        "well uses an unsupported rate-control mode",
                        object_name="_build_blackoil_case",
                        field=ws.name,
                        expected=tuple(kind.value for kind in WellRateKind),
                        actual=mode,
                    ) from error
            control = RateControl(kind=rate_kind, target_m3_s=float(ws.target))
        elif ws.bhp_limit is not None:
            control = BHPControl(pressure_pa=float(ws.bhp_limit))
        else:
            raise GeoBrainError(
                "well is missing an explicit control target",
                object_name="_build_blackoil_case",
                field=ws.name,
                expected="rate, reservoir-rate, or BHP target",
                actual=None,
            )
        if ws.well_type == "INJ":
            injection_phase = cast(
                Literal["water", "oil", "gas", "fluid"],
                ws.injection_phase or "gas",
            )
            wells.append(
                Well(
                    name=ws.name,
                    well_type="INJ",
                    control=control,
                    perforations=tuple(perfs),
                    injection_phase=injection_phase,
                    standard_conditions=WellStandardConditions(101_325.0, 288.15),
                    standard_densities_kg_m3={"oil": rho_o, "water": rho_w, "gas": rho_g},
                    bhp_limit_pa=(float(ws.bhp_limit) if ws.bhp_limit is not None else None),
                    datum_depth_m=datum_depth,
                )
            )
        else:
            wells.append(
                Well(
                    name=ws.name,
                    well_type="PROD",
                    control=control,
                    perforations=tuple(perfs),
                    standard_conditions=WellStandardConditions(101_325.0, 288.15),
                    standard_densities_kg_m3={"oil": rho_o, "water": rho_w, "gas": rho_g},
                    bhp_limit_pa=(float(ws.bhp_limit) if ws.bhp_limit is not None else None),
                    datum_depth_m=datum_depth,
                )
            )

    return BlackOilCase(
        grid=grid,
        rock=rock,
        fluid=fluid,
        wells=wells,
        well_heads=heads,
        tstep=list(deck.tstep),
        deck=deck,
    )


__all__ = [
    "BlackOilCase",
    "BlackOilDeck",
    "DeckKeywordBlock",
    "DeckRecordStream",
    "WellSpec",
    "parse_deck_records",
]
