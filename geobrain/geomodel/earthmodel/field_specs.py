"""
SI-only field-spec table: ``{name: (unit, plausible_range)}``.

Consumed by :class:`~geobrain.geomodel.earthmodel.EarthModel` for two purposes:

- **Default units.** ``Field(unit=None)`` (the default) resolves its stamped
  unit from this table by field/link NAME; a name absent here has no default
  (``unit=None`` stays ``None`` unless the ``Field``/``Link`` declares one
  explicitly).
- **First-resolve validation.** ``EarthModel.resolve()`` magnitude-checks
  every name present in this table against its ``plausible_range``; see
  the normative semantics in ``EarthModel.resolve``'s docstring.

Since the SI unification, there is exactly one convention per quantity, no
dual mass-density unit clause (kg/m^3 only), no GPa/Pa split.

**Scope guard**: entries are added here ONLY alongside a real consumer
(a ``Field``/``Link`` name an in-repo example or test actually uses), never
speculatively. The six entries below are the platform's existing rock/EM
vocabulary (porosity, water saturation, the two elastic wave speeds, bulk
density, and electrical conductivity), each with day-one consumers in the
P2b rock pure functions and the Marmousi rock-coupled example.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

__all__ = ["FIELD_SPECS"]

# name -> (unit, (lo, hi))
FIELD_SPECS: dict[str, tuple[str, tuple[float, float]]] = {
    "phi": ("1", (0.0, 0.6)),
    "sw": ("1", (0.0, 1.0)),
    "vp": ("m/s", (150.0, 9000.0)),
    "vs": ("m/s", (0.0, 6000.0)),
    "rho": ("kg/m^3", (500.0, 8000.0)),
    "sigma": ("S/m", (1e-6, 1e2)),
}
