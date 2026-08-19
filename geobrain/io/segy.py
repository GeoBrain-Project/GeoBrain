"""
SEG-Y reader / writer (segyio backend).

Minimal-and-explicit: we expose two functions, read and write, both keyed
around a 2D ``(n_traces, nt)`` ``numpy.ndarray`` plus a ``dt`` (seconds).
3D volumes load as ``(n_inlines, n_xlines, nt)`` when the file has a regular
inline/xline grid; otherwise the caller gets a 2D layout.

We do not try to be SEG-Y-aware beyond what scientific code typically needs.
For exotic byte locations / coordinate systems, users should call segyio
directly and pass the resulting numpy array back into our framework.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

import os
from typing import Any, NamedTuple

import numpy as np

from ._common import _to_numpy


class SegYData(NamedTuple):
    """Return type of :func:`read_segy`.

    Attributes:
        traces: ``(n_traces, nt)`` or ``(n_inlines, n_xlines, nt)`` float32 array.
        dt: Sample interval in **seconds**.
        meta: Small dict with ``"n_traces"``, ``"nt"``, ``"format"``, and geometry info.
    """

    traces: np.ndarray
    dt: float
    meta: dict

try:
    import segyio
except ImportError:  # pragma: no cover - guarded at import site
    segyio = None  # type: ignore[assignment]


def _require_segyio() -> Any:
    if segyio is None:
        raise ImportError(
            "segyio not installed. Install with: pip install segyio"
        )
    return segyio


def read_segy(path: str | os.PathLike) -> SegYData:
    """
    Read a SEG-Y file.

    Returns a :class:`SegYData` named tuple ``(traces, dt, meta)``:

    - ``traces``: ``(n_traces, nt)`` 2D float32 array, OR
      ``(n_inlines, n_xlines, nt)`` 3D when the file is a regular cube.
    - ``dt``: sample interval in **seconds** (segyio stores microseconds; we convert).
    - ``meta``: small dict with ``"n_traces"``, ``"nt"``, ``"format"``,
      ``"t0"`` (recording delay in **seconds**, from the first trace's
      ``DelayRecordingTime`` header word, SEG-Y stores milliseconds; 0.0 when
      absent; per-trace variation is not resolved here), and (if 3D)
      ``"inlines"``, ``"xlines"``.

    Tuple-unpacking is backward-compatible::

        traces, dt, meta = read_segy("shot.sgy")
    """
    sgy = _require_segyio()
    with sgy.open(str(path), "r", strict=False, ignore_geometry=True) as f:
        nt = int(f.samples.size)
        dt_us = int(f.bin[sgy.BinField.Interval])     # microseconds
        if dt_us <= 0:
            # The binary-header interval is absent/zero: common in industry files
            # that carry the sample interval only per-trace. Fall back to the first
            # trace header; if that is also missing, fail loud rather than silently
            # returning dt = 0 and poisoning every downstream time/frequency axis.
            try:
                dt_us = int(f.header[0][sgy.TraceField.TRACE_SAMPLE_INTERVAL])
            except Exception:
                dt_us = 0
            if dt_us <= 0:
                raise GeoBrainError(
                    "SEG-Y sample interval is 0 in both the binary header and the "
                    "first trace header; cannot infer the time axis.",
                    object_name="read_segy", field="sample_interval",
                    expected="positive microseconds", actual=dt_us,
                )
        traces = f.trace.raw[:].astype(np.float32)     # (n_traces, nt)
        n_traces = traces.shape[0]
        # Recording delay (SEG-Y trace header word 109-110, milliseconds).
        # Best-effort: a malformed header degrades to t0 = 0.0, not a crash.
        try:
            t0_ms = int(f.header[0][sgy.TraceField.DelayRecordingTime])
        except Exception:
            t0_ms = 0
        meta: dict = {
            "n_traces": n_traces, "nt": nt, "format": "segy",
            "t0": t0_ms * 1e-3,
        }

    # Try to reshape into a 3D cube if the inline / xline headers look regular.
    # Geometry extraction is best-effort: on failure we fall back to the flat
    # ``(n_traces, nt)`` array, but record *why* in ``meta`` so the degradation
    # is observable (check ``meta["geometry"]`` / ``meta["cube_error"]``) rather
    # than silent.
    try:
        with sgy.open(str(path), "r") as f3:
            if f3.unstructured:
                meta["geometry"] = "unstructured"
            else:
                cube = sgy.tools.cube(f3).astype(np.float32)
                meta["inlines"] = list(f3.ilines)
                meta["xlines"] = list(f3.xlines)
                meta["geometry"] = "cube"
                return SegYData(cube, dt_us * 1e-6, meta)
    except Exception as exc:  # irregular geometry / missing headers; stay flat
        meta["cube_error"] = f"{type(exc).__name__}: {exc}"

    meta.setdefault("geometry", "traces")
    return SegYData(traces, dt_us * 1e-6, meta)


def write_segy(
    path: str | os.PathLike,
    traces: np.ndarray,
    dt: float,
    *,
    format: int = 5,           # IEEE float32 (segyio constant)
) -> None:
    """
    Write a 2D ``(n_traces, nt)`` array to a SEG-Y file.

    ``dt`` is in **seconds**. ``traces`` may be a tensor; we detach and copy
    to numpy. Trace and binary headers carry only the essentials (sample
    interval and count); full SEG-Y conformance is the user's job.

    Args:
        path: output ``.sgy`` file.
        traces: ``(n_traces, nt)`` array to write.
        dt: sample interval in seconds.
        format: SEG-Y data-sample format code (default 5 = IEEE float32).
    """
    sgy = _require_segyio()

    # Detach tensors at the IO boundary via the shared chokepoint.
    traces = _to_numpy(traces, dtype=np.float32)
    if traces.ndim != 2:
        raise GeoBrainError(
            f"write_segy expects 2D (n_traces, nt), got shape {traces.shape}"
        )
    if dt <= 0:
        raise GeoBrainError(f"dt must be positive seconds, got {dt}")

    n_traces, nt = traces.shape
    dt_us = int(round(dt * 1e6))

    spec = sgy.spec()
    spec.sorting = 1
    spec.format = format
    spec.samples = list(range(nt))
    spec.tracecount = n_traces

    with sgy.create(str(path), spec) as f:
        f.bin = {sgy.BinField.Interval: dt_us, sgy.BinField.Samples: nt}
        for i in range(n_traces):
            f.header[i] = {
                sgy.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                sgy.TraceField.TRACE_SAMPLE_COUNT: nt,
                sgy.TraceField.TRACE_SEQUENCE_LINE: i + 1,
            }
            f.trace[i] = traces[i]
