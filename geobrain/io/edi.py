"""EDI (SEG Electrical Data Interchange) MT sounding reader (pure text).

Parses the sections MT practice actually exchanges: ``>FREQ``, the
impedance blocks ``>ZXXR/>ZXXI …``, optional tipper blocks
(``>TXR.EXP`` … or ``>TZXR`` … spellings), optional ``>ZROT``, and the
``>HEAD`` location fields (``LAT``/``LONG`` in decimal degrees or
``DD:MM:SS``, ``ELEV`` in metres).

Units: EDI impedance blocks carry the MT field convention
``mV/km/nT``. :func:`read_edi` converts to SI ohms on ingest,
``Z_si = Z_field * 1e3 * MU_0`` (1 mV/km = 1e-6 V/m; 1 nT of B is
``1e-9 / MU_0`` A/m of H), so downstream code never sees field units.
Derived quantities follow the platform conventions: apparent
resistivity in ohm-m and phase in RADIANS (display adapters convert to
degrees, exactly as for the MT operators).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import os
import re
from typing import NamedTuple

import numpy as np

from geobrain.core.constants import MU_0
from geobrain.core.errors import GeoBrainError

#: Field-unit (mV/km/nT) impedance -> SI ohm.
Z_FIELD_TO_SI = 1.0e3 * MU_0

_COMPONENTS = ("xx", "xy", "yx", "yy")
_TIPPER_SPELLINGS = {
    "x": ("TXR.EXP", "TXI.EXP", "TZXR", "TZXI"),
    "y": ("TYR.EXP", "TYI.EXP", "TZYR", "TZYI"),
}


class EDIData(NamedTuple):
    """Return type of :func:`read_edi` (single MT station).

    Attributes:
        frequencies_hz: ``(nf,)`` sounding frequencies, descending or as
            stored in the file.
        impedance_si_ohm: ``(nf, 2, 2)`` complex impedance tensor in SI
            ohms, rows/cols ordered ``(x, y)`` so ``[k, 0, 1]`` is Zxy.
        tipper: ``(nf, 2)`` complex ``(Tzx, Tzy)`` or ``None``.
        rotation_deg: ``(nf,)`` impedance rotation angles (``>ZROT``),
            zeros when absent.
        latitude_deg / longitude_deg / elevation_m: station location from
            ``>HEAD`` (``nan`` when absent).
        station: ``DATAID`` string, ``""`` when absent.
        header: raw ``>HEAD`` key/value pairs.
    """

    frequencies_hz: np.ndarray
    impedance_si_ohm: np.ndarray
    tipper: np.ndarray | None
    rotation_deg: np.ndarray
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    station: str
    header: dict


def _parse_angle(text: str) -> float:
    """``DD:MM:SS.s`` (sign on the degrees) or plain decimal degrees."""
    text = text.strip()
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    degrees = float(parts[0])
    minutes = float(parts[1]) if len(parts) > 1 else 0.0
    seconds = float(parts[2]) if len(parts) > 2 else 0.0
    sign = -1.0 if text.lstrip().startswith("-") else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def _split_sections(text: str) -> list[tuple[str, str, str]]:
    """Split on ``>`` section markers → (name, header_tail, payload)."""
    sections: list[tuple[str, str, str]] = []
    name = ""
    header_tail = ""
    payload: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if name:
                sections.append((name, header_tail, "\n".join(payload)))
            head = line[1:].strip()
            match = re.match(r"([=A-Za-z0-9_.]+)", head)
            name = match.group(1).upper().lstrip("=") if match else ""
            header_tail = head[len(match.group(1)) :] if match else head
            payload = []
        elif name:
            payload.append(line)
    if name:
        sections.append((name, header_tail, "\n".join(payload)))
    return sections


def _floats(payload: str) -> np.ndarray:
    tokens = payload.replace(",", " ").split()
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            continue
    return np.array(values, dtype=float)


def _keyvals(text: str) -> dict:
    pairs = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(\"[^\"]*\"|\S+)", text):
        pairs[match.group(1).upper()] = match.group(2).strip('"')
    return pairs


def read_edi(path: str | os.PathLike) -> EDIData:
    """Read one MT station from an EDI file (impedances converted to SI)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    sections = _split_sections(text)
    blocks = {name: _floats(payload) for name, _, payload in sections}
    header: dict = {}
    for name, header_tail, payload in sections:
        if name == "HEAD":
            header = _keyvals(header_tail + "\n" + payload)

    if "FREQ" not in blocks or blocks["FREQ"].size == 0:
        raise GeoBrainError(
            "EDI file has no >FREQ block",
            object_name="read_edi",
            field="path",
            actual=str(path),
        )
    freq = blocks["FREQ"]
    nf = freq.size

    impedance = np.zeros((nf, 2, 2), dtype=complex)
    for idx, comp in enumerate(_COMPONENTS):
        real = blocks.get(f"Z{comp.upper()}R")
        imag = blocks.get(f"Z{comp.upper()}I")
        if real is None or imag is None:
            raise GeoBrainError(
                f"EDI file lacks impedance block Z{comp.upper()}R/I",
                object_name="read_edi",
                field=f"z_{comp}",
                actual=str(path),
            )
        if real.size != nf or imag.size != nf:
            raise GeoBrainError(
                f"Z{comp.upper()} length disagrees with >FREQ",
                object_name="read_edi",
                field=f"z_{comp}",
                expected=nf,
                actual=(real.size, imag.size),
            )
        impedance[:, idx // 2, idx % 2] = (real + 1j * imag) * Z_FIELD_TO_SI

    tipper = None
    tipper_parts = {}
    for axis, spellings in _TIPPER_SPELLINGS.items():
        real = blocks.get(spellings[0])
        imag = blocks.get(spellings[1])
        if real is None or imag is None:
            real = blocks.get(spellings[2])
            imag = blocks.get(spellings[3])
        if real is not None and imag is not None and real.size == nf:
            tipper_parts[axis] = real + 1j * imag
    if set(tipper_parts) == {"x", "y"}:
        tipper = np.stack([tipper_parts["x"], tipper_parts["y"]], axis=1)

    rotation = blocks.get("ZROT")
    if rotation is None or rotation.size != nf:
        rotation = np.zeros(nf)

    return EDIData(
        frequencies_hz=freq,
        impedance_si_ohm=impedance,
        tipper=tipper,
        rotation_deg=rotation,
        latitude_deg=_parse_angle(header["LAT"]) if "LAT" in header else math.nan,
        longitude_deg=_parse_angle(header["LONG"]) if "LONG" in header else math.nan,
        elevation_m=float(header["ELEV"]) if "ELEV" in header else math.nan,
        station=header.get("DATAID", ""),
        header=header,
    )


def apparent_resistivity_ohm_m(edi: EDIData) -> np.ndarray:
    """``(nf, 2, 2)`` apparent resistivity ``|Z|^2 / (omega * MU_0)``."""
    omega = 2.0 * math.pi * edi.frequencies_hz
    return np.abs(edi.impedance_si_ohm) ** 2 / (omega * MU_0)[:, None, None]


def impedance_phase_rad(edi: EDIData) -> np.ndarray:
    """``(nf, 2, 2)`` impedance phase in RADIANS (platform convention)."""
    return np.angle(edi.impedance_si_ohm)


__all__ = [
    "EDIData",
    "Z_FIELD_TO_SI",
    "read_edi",
    "apparent_resistivity_ohm_m",
    "impedance_phase_rad",
]
