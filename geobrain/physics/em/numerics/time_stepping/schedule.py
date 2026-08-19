"""
Log-substep BDF time schedule for TEM3D.

Build a variable-step BDF time grid that

1. **Hits every requested gate exactly**, the cumulative sum of the
   returned ``dts`` lands on each ``time_gates[k]`` within float64
   precision.
2. **Is log-dense**, within every (gate_{k-1}, gate_k] window (and a
   pre-roll window from ``t1/10`` up to ``t1``), substeps are placed
   geometrically with ~``n_substeps_per_decade`` substeps per decade of
   ``log10(t)``.

:class:`~geobrain.physics.em.time_domain.tem3d.TEM3D` imports
:func:`build_log_substep_schedule` from this module as the single source of
its BDF time grid (``tem3d.py`` constructs the schedule, then steps it). The
algorithm is:

- **Pre-roll**::

      ramp = np.logspace(log10(t1/10), log10(t1), n_per_dec + 1)
      for tt in ramp[:-1]:        # drops t1; t1 is appended as gate
          t_grid.append(tt)

  ``ramp`` has ``n_per_dec + 1`` points from ``t1/10`` to ``t1``;
  dropping the last point keeps the first ``n_per_dec`` points
  (``t1/10`` first, then the geometric ramp up to but not including
  ``t1``). The grid then **starts at ``t1/10``**, i.e. ``dts[0] =
  t1/10`` (a single "big" step from ``t = 0`` to ``t = t1/10``), and
  the next ``n_per_dec - 1`` steps geometrically climb to ``t1``.
- **Gate-0 append**::

      t_grid.append(t1)                         # gate 0
      gate_indices_in_grid.append(len(t_grid))  # 1-based hist index

  The history index ``len(t_grid)`` is **1-based** because the
  history has ``E^0`` at index 0 (cumulative ``t = 0``). This returns
  a **0-based** ``gate_step_idx`` into ``dts`` such that
  ``dts[:gate_step_idx[k] + 1].sum() == time_gates[k]``, which means
  the index equals the ``len(t_grid) - 1``.
- **Inter-gate**::

      n_extra = max(1, ceil(n_per_dec * log10(t_hi / t_lo)))
      sub = np.logspace(log10(t_lo), log10(t_hi), n_extra + 1)
      for tt in sub[1:]:        # drops t_lo (already in grid)
          t_grid.append(tt)

So for each (t_lo, t_hi) window we append ``n_extra`` new t-grid
points, the last of which is ``t_hi`` (the next gate).

Returned shapes / dtypes:

- ``dts``        ``torch.float64`` ``(n_steps,)``
- ``gate_step_idx`` ``torch.int64`` ``(n_gates,)``

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .....core.errors import GeoBrainError


def build_log_substep_schedule(
    time_gates: tuple[float, ...],
    n_substeps_per_decade: int = 8,
    *,
    dt_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a BDF time schedule that hits each gate exactly.

    This is the single source of TEM3D's BDF time grid (imported by
    ``tem3d.py``). Pre-roll: log-ramp from ``t1/10`` to ``t1`` with the
    grid **starting at** ``t1/10`` and ending at ``t1`` after
    ``n_substeps_per_decade`` substeps. Inter-gate: same log density,
    rounded up to at least one substep.

    Args:
        time_gates: Ascending positive gate times in seconds. Must be
            non-empty.
        n_substeps_per_decade: Substep density in ``log10(t)`` space.
            Must be ``>= 1``. Defaults to ``8`` (gives ~32 steps for a
            4-decade survey such as 1e-5..1e-1 s).
        dt_max: Optional upper bound (seconds) on individual step
            intervals. When ``None`` (default) the pure log-substep
            schedule is returned. When set, any
            substep that would exceed ``dt_max`` is subdivided into
            fixed-size ``dt_max`` chunks (with a possibly shorter final
            piece so the gate is still reached exactly). This caps the
            late-time backward-Euler step size; diffusion modes are
            otherwise over-damped once the log-spaced schedule's dt
            grows past a few percent of the gate spacing.

    Returns:
        ``(dts, gate_step_idx)``:

        - ``dts``: ``(n_steps,)`` float64 step intervals.
        - ``gate_step_idx``: ``(n_gates,)`` int64 indices into ``dts``
          where each gate is reached. The k-th gate satisfies
          ``dts[:gate_step_idx[k] + 1].sum() == time_gates[k]`` (within
          float64 precision).

    Raises:
        ValueError: If ``time_gates`` is empty, contains a non-positive
            entry, is not strictly ascending, ``n_substeps_per_decade
            < 1``, or ``dt_max <= 0``.
    """
    if n_substeps_per_decade < 1:
        raise GeoBrainError(
            "n_substeps_per_decade must be >= 1; "
            f"got {n_substeps_per_decade}",
            object_name="build_log_substep_schedule",
            field="n_substeps_per_decade",
            expected=">= 1",
            actual=n_substeps_per_decade,
        )
    if dt_max is not None and dt_max <= 0:
        raise GeoBrainError(
            f"dt_max must be > 0 when provided; got {dt_max}",
            object_name="build_log_substep_schedule",
            field="dt_max",
            expected="> 0",
            actual=dt_max,
        )
    if not time_gates:
        raise GeoBrainError(
            "time_gates is empty",
            object_name="build_log_substep_schedule",
            field="time_gates",
            expected="non-empty",
            actual=time_gates,
        )
    for k, t in enumerate(time_gates):
        if t <= 0:
            raise GeoBrainError(
                f"time_gates[{k}] = {t} is not positive",
                object_name="build_log_substep_schedule",
                field=f"time_gates[{k}]",
                expected="> 0",
                actual=t,
            )
        if k > 0 and t <= time_gates[k - 1]:
            raise GeoBrainError(
                f"time_gates[{k}] = {t} not strictly greater than "
                f"time_gates[{k - 1}] = {time_gates[k - 1]}",
                object_name="build_log_substep_schedule",
                field=f"time_gates[{k}]",
                expected=f"> time_gates[{k - 1}] ({time_gates[k - 1]})",
                actual=t,
            )

    gate_arr = np.asarray(time_gates, dtype=np.float64).ravel()
    n_per_dec = int(n_substeps_per_decade)

    # Build cumulative t-grid (excluding t = 0).
    t_grid: list[float] = []
    gate_indices_in_grid: list[int] = []  # 1-based history-style index

    # Pre-roll: log-ramp from t1/10 .. t1.
    t1 = float(gate_arr[0])
    ramp = np.logspace(
        math.log10(t1 / 10.0), math.log10(t1), n_per_dec + 1
    )
    # Drop the trailing t1 (it is added as a gate below); keep the
    # leading t1/10 plus the intermediate geometric points.
    for tt in ramp[:-1]:
        t_grid.append(float(tt))

    # Append gate 0 exactly.
    t_grid.append(t1)
    gate_indices_in_grid.append(len(t_grid))  # 1-based hist index

    # Sub-steps between consecutive gates.
    for g in range(1, gate_arr.size):
        t_lo = float(gate_arr[g - 1])
        t_hi = float(gate_arr[g])
        n_dec = math.log10(t_hi / t_lo)
        n_extra = max(1, int(math.ceil(n_per_dec * n_dec)))
        sub = np.logspace(
            math.log10(t_lo), math.log10(t_hi), n_extra + 1
        )
        # Drop the leading t_lo (already in grid); append the rest, the
        # last entry of which is t_hi (lands exactly on the gate).
        for tt in sub[1:]:
            t_grid.append(float(tt))
        gate_indices_in_grid.append(len(t_grid))

    t_grid_np = np.asarray(t_grid, dtype=np.float64)

    # dts[i] = t_grid[i] - t_grid[i - 1], with t_grid[-1] := 0.
    dt_arr = np.empty(t_grid_np.size, dtype=np.float64)
    dt_arr[0] = t_grid_np[0]
    dt_arr[1:] = t_grid_np[1:] - t_grid_np[:-1]
    if np.any(dt_arr <= 0):
        raise GeoBrainError(
            "Internal time schedule produced a non-positive dt; "
            f"check time_gates ordering. Got dt_arr={dt_arr!r}",
            object_name="build_log_substep_schedule",
            field="dt_arr",
            expected="all dt > 0",
            actual=dt_arr,
        )

    # Convert the 1-based history indices to the 0-based dts
    # indices: 0-based dts index = 1-based history index - 1.
    gate_step_idx_np = (
        np.asarray(gate_indices_in_grid, dtype=np.int64) - 1
    )

    # Optional dt cap: split any oversize step into equal-size sub-steps
    # so cumulative time (and therefore gate positions) stay exact while
    # the per-step interval never exceeds ``dt_max``. Gate indices are
    # shifted to point at the LAST sub-step of the split (which still
    # lands on the original gate time).
    if dt_max is not None:
        new_dts: list[float] = []
        # Map: original index -> last new-array index for that step.
        last_new_idx: list[int] = []
        for old_dt in dt_arr.tolist():
            if old_dt > dt_max:
                n_split = int(math.ceil(old_dt / dt_max))
                # Equal-size pieces summing exactly to old_dt (avoids
                # drift from repeated dt_max + small remainder).
                piece = old_dt / n_split
                for _ in range(n_split):
                    new_dts.append(piece)
            else:
                new_dts.append(old_dt)
            last_new_idx.append(len(new_dts) - 1)
        # Re-map gate indices into the new dts array.
        gate_step_idx_np = np.asarray(
            [last_new_idx[i] for i in gate_step_idx_np.tolist()],
            dtype=np.int64,
        )
        dt_arr = np.asarray(new_dts, dtype=np.float64)

    dts = torch.from_numpy(dt_arr)
    gate_step_idx = torch.from_numpy(gate_step_idx_np)
    return dts, gate_step_idx
