"""Channel registry: the single, test-enforced source of truth for the public
**output channel strings** emitted by GeoBrain forward operators.

A *channel* is one of the string keys an operator declares in
``DifferentiabilitySpec.output_keys``, writes into ``ForwardOutput.data`` (for a
:class:`ForwardOperator`) or into the output ``ModelState`` (for a
:class:`PropertyTransform`), and that ``InverseProblem`` matches observed data
against. These strings are the platform's data vocabulary; renaming one is a
cross-cutting breaking change (operator + tests + examples + observed-data dict).

``CHANNEL_REGISTRY`` makes that vocabulary machine-checkable:
The architecture governance suite walks every concrete
``ForwardOperator`` subclass and asserts each declared output key is registered.
A new operator that emits an unregistered channel fails CI. This registry is
the single, authoritative source for output-channel semantics, the **code**
wins over any informal notes elsewhere.

Scope: this registry covers *observable output channels* (what an operator
emits), not *model-parameter / input* fields (``sigma``, ``permeability``,
``phi``, ``Sw``, …) that flow in through ``ModelState``. Where a model parameter
and an output channel share a name (``pressure``, ``rho``, ``vp``), the entry's
``meaning`` notes it.

dtype convention: a single ``dtype`` field of ``"real"`` or ``"complex"`` per
channel. EM field components (``ex``/``bx``/…) are *native complex* in the
frequency domain and *real* in the time domain; the same key carries both; the
registry marks them ``"complex"`` (their primary/declared form) and the
``meaning`` records the FD/TD duality.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from .errors import GeoBrainError

__all__ = [
    "ChannelSpec",
    "CHANNEL_REGISTRY",
    "CHANNEL_ALIASES",
    "resolve_channel",
    "channel_spec",
]


@dataclass(frozen=True)
class ChannelSpec:
    """One public output channel's contract.

    Attributes:
        name: the channel string (key in ``output_keys`` / ``ForwardOutput.data``).
        unit: physical unit (``"dimensionless"`` / ``"relative amplitude"`` where
            no SI unit applies).
        dtype: ``"real"`` or ``"complex"``, the emitted tensor's primary form.
        meaning: one-line human description.
        kind: coarse shape class, ``"per_receiver"`` (one value per survey
            receiver/station/frequency), ``"per_cell"`` (one value per mesh cell,
            i.e. a model-state field), or ``"field"`` (a gridded field). Optional.
    """

    name: str
    unit: str
    dtype: str
    meaning: str
    kind: str | None = None

    def __post_init__(self) -> None:
        if self.dtype not in ("real", "complex"):
            raise GeoBrainError(
                "ChannelSpec.dtype must be 'real' or 'complex'",
                object_name="ChannelSpec", field="dtype",
                expected="'real' or 'complex'", actual=self.dtype,
            )
        if self.kind is not None and self.kind not in (
            "per_receiver",
            "per_cell",
            "field",
        ):
            raise GeoBrainError(
                "ChannelSpec.kind must be 'per_receiver'/'per_cell'/'field' or None",
                object_name="ChannelSpec", field="kind",
                expected="'per_receiver'/'per_cell'/'field' or None",
                actual=self.kind,
            )


def _spec(name: str, unit: str, dtype: str, meaning: str, kind: str | None = None) -> ChannelSpec:
    return ChannelSpec(name=name, unit=unit, dtype=dtype, meaning=meaning, kind=kind)


# =========================================================================
# The registry. Grouped by physics family; verified against each operator's
# _forward emission + DifferentiabilitySpec.output_keys (code wins over glossary).
# =========================================================================
_SPECS: tuple[ChannelSpec, ...] = (
    # ---------------------------------------------------------------- wave
    _spec("seismic", "relative amplitude", "real",
          "time-domain wavefield traces at receivers (acoustic pressure / elastic "
          "particle field); universal time-domain trace channel", "per_receiver"),
    _spec("p", "relative amplitude", "complex",
          "frequency-domain Helmholtz pressure field at receivers (complex: carries "
          "phase)", "per_receiver"),
    _spec("trace", "relative amplitude", "real",
          "post-convolution synthetic seismogram trace", "per_receiver"),
    _spec("reflectivity", "dimensionless", "real",
          "AVO/AVA angle-dependent reflection coefficient", "per_receiver"),
    # ----------------------------------------------------------------- em
    # Field components: complex in the frequency domain, real in the time domain
    # (TEM3D). One key per complex component (no split real/imag).
    _spec("ex", "V/m", "complex", "electric-field E_x component (complex FD / real TD)", "per_receiver"),
    _spec("ey", "V/m", "complex", "electric-field E_y component (complex FD / real TD)", "per_receiver"),
    _spec("ez", "V/m", "complex", "electric-field E_z component (complex FD / real TD)", "per_receiver"),
    _spec("bx", "T", "complex", "magnetic-flux-density B_x component (complex FD / real TD)", "per_receiver"),
    _spec("by", "T", "complex", "magnetic-flux-density B_y component (complex FD / real TD)", "per_receiver"),
    _spec("bz", "T", "complex", "magnetic-flux-density B_z component (complex FD / real TD)", "per_receiver"),
    _spec("hx", "A/m", "complex", "magnetic-field H_x component", "per_receiver"),
    _spec("hy", "A/m", "complex", "magnetic-field H_y component", "per_receiver"),
    _spec("hz", "A/m", "complex", "magnetic-field H_z component", "per_receiver"),
    _spec("dbdt_x", "T/s", "real",
          "x-directed time-derivative of B (TEM transient decay)", "per_receiver"),
    _spec("dbdt_y", "T/s", "real",
          "y-directed time-derivative of B (TEM transient decay)", "per_receiver"),
    _spec("dbdt_z", "T/s", "real",
          "vertical time-derivative of B (TEM transient decay)", "per_receiver"),
    _spec("z_xx", "Ohm (V/m per A/m)", "complex", "MT impedance-tensor component Z_xx", "per_receiver"),
    _spec("z_xy", "Ohm (V/m per A/m)", "complex", "MT impedance-tensor component Z_xy", "per_receiver"),
    _spec("z_yx", "Ohm (V/m per A/m)", "complex", "MT impedance-tensor component Z_yx", "per_receiver"),
    _spec("z_yy", "Ohm (V/m per A/m)", "complex", "MT impedance-tensor component Z_yy", "per_receiver"),
    _spec("t_zx", "dimensionless", "complex", "MT tipper component T_zx (vertical mag transfer fn)", "per_receiver"),
    _spec("t_zy", "dimensionless", "complex", "MT tipper component T_zy (vertical mag transfer fn)", "per_receiver"),
    _spec("impedance", "Ohm", "complex",
          "MT 1-D/2-D scalar surface impedance (per-mode suffixed for 2-D 'both')",
          "per_receiver"),
    _spec("impedance_TE", "Ohm", "complex", "MT2D TE-mode impedance (mode='both')", "per_receiver"),
    _spec("impedance_TM", "Ohm", "complex", "MT2D TM-mode impedance (mode='both')", "per_receiver"),
    _spec("apparent_resistivity", "Ohm·m", "real", "MT-derived apparent resistivity", "per_receiver"),
    _spec("apparent_resistivity_TE", "Ohm·m", "real", "MT2D TE apparent resistivity (mode='both')", "per_receiver"),
    _spec("apparent_resistivity_TM", "Ohm·m", "real", "MT2D TM apparent resistivity (mode='both')", "per_receiver"),
    _spec("phase", "rad", "real", "MT-derived impedance phase", "per_receiver"),
    _spec("phase_TE", "rad", "real", "MT2D TE phase (mode='both')", "per_receiver"),
    _spec("phase_TM", "rad", "real", "MT2D TM phase (mode='both')", "per_receiver"),
    _spec("voltage", "V", "real", "DC-resistivity receiver potential", "per_receiver"),
    _spec("sp", "V", "real", "self-potential receiver potential", "per_receiver"),
    _spec("chargeability_obs", "dimensionless", "real",
          "time-domain IP apparent chargeability M_a (numerically mV/V, not ×1000)",
          "per_receiver"),
    _spec("m_app", "dimensionless", "complex",
          "spectral-IP apparent (complex) chargeability summary", "per_receiver"),
    # ---------------------------------------------------------- potential
    _spec("gx", "m/s^2", "real", "gravity acceleration g_x", "per_receiver"),
    _spec("gy", "m/s^2", "real", "gravity acceleration g_y", "per_receiver"),
    _spec("gz", "m/s^2", "real", "vertical gravity acceleration g_z", "per_receiver"),
    _spec("gxx", "s^-2", "real", "gravity-gradient tensor g_xx", "per_receiver"),
    _spec("gxy", "s^-2", "real", "gravity-gradient tensor g_xy", "per_receiver"),
    _spec("gxz", "s^-2", "real", "gravity-gradient tensor g_xz", "per_receiver"),
    _spec("gyy", "s^-2", "real", "gravity-gradient tensor g_yy", "per_receiver"),
    _spec("gyz", "s^-2", "real", "gravity-gradient tensor g_yz", "per_receiver"),
    _spec("gzz", "s^-2", "real", "gravity-gradient tensor g_zz", "per_receiver"),
    _spec("tmi", "T", "real", "total-magnetic-intensity anomaly", "per_receiver"),
    _spec("bxx", "T/m", "real", "magnetic-gradient tensor B_xx", "per_receiver"),
    _spec("bxy", "T/m", "real", "magnetic-gradient tensor B_xy", "per_receiver"),
    _spec("bxz", "T/m", "real", "magnetic-gradient tensor B_xz", "per_receiver"),
    _spec("byy", "T/m", "real", "magnetic-gradient tensor B_yy", "per_receiver"),
    _spec("byz", "T/m", "real", "magnetic-gradient tensor B_yz", "per_receiver"),
    _spec("bzz", "T/m", "real", "magnetic-gradient tensor B_zz", "per_receiver"),
    # Potential bx/by/bz are real anomaly components in tesla. Their key strings
    # collide with the complex EM fields above, so dtype is family-specific while
    # the global strict-SI unit remains tesla for both families.
    # ------------------------------------------------------------- flow
    _spec("pressure", "Pa", "real",
          "per-cell pore pressure (both ModelState input and operator output)",
          "per_cell"),
    _spec("sw", "dimensionless", "real", "water saturation (oil-water + black-oil)", "per_cell"),
    _spec("sg", "dimensionless", "real", "gas saturation (black-oil only)", "per_cell"),
    _spec("final_pressure", "Pa", "real", "final per-cell pressure (TransientFlowOperator)", "per_cell"),
    _spec("final_sw", "dimensionless", "real", "final water saturation (TransientFlowOperator)", "per_cell"),
    _spec("final_sg", "dimensionless", "real", "final gas saturation (TransientFlowOperator)", "per_cell"),
    _spec("bhp", "Pa", "real", "per-well bottom-hole pressure", "per_receiver"),
    _spec("q_water", "m^3/s", "real", "per-well water rate (signed: + inj, - prod)", "per_receiver"),
    _spec("q_oil", "m^3/s", "real", "per-well oil rate", "per_receiver"),
    _spec("q_gas", "m^3/s", "real", "per-well gas rate", "per_receiver"),
    _spec("bhp_series", "Pa", "real", "per-step per-well BHP time series", "per_receiver"),
    _spec("q_water_series", "m^3/s", "real", "per-step per-well water-rate series", "per_receiver"),
    _spec("q_oil_series", "m^3/s", "real", "per-step per-well oil-rate series", "per_receiver"),
    _spec("q_gas_series", "m^3/s", "real", "per-step per-well gas-rate series", "per_receiver"),
    # ------------------------------------------------------------- rock
    # Canonical cross-model elastic staples (bridge into wave/AVO forward).
    _spec("vp", "m/s", "real", "P-wave velocity (cross-model staple; also output of Gassmann/Han)", "per_cell"),
    _spec("vs", "m/s", "real", "S-wave velocity (cross-model staple; Castagna/Greenberg/Han)", "per_cell"),
    # The platform is single-convention SI. rho is kg/m^3
    # EVERYWHERE (rock emits it, wave and gravity consume it). The historical
    # dual convention (rock/gravity g/cm^3 vs wave kg/m^3) and its (SI-EXEMPT: historical note)
    # construction-time straddle guard case died with the SI unification.
    _spec("rho", "kg/m^3", "real", "bulk density (SI, single platform convention)", "per_cell"),
    # Anisotropy / Thomsen (anisotropy.py)
    _spec("vp0", "m/s", "real", "vertical P-velocity (Thomsen reference)", "per_cell"),
    _spec("vs0", "m/s", "real", "vertical S-velocity (Thomsen reference)", "per_cell"),
    _spec("vp_eff", "m/s", "real", "Backus-averaged effective P-velocity", "per_cell"),
    _spec("vs_eff", "m/s", "real", "Backus-averaged effective S-velocity", "per_cell"),
    _spec("rho_eff", "kg/m^3", "real", "Backus-averaged effective density", "per_cell"),
    _spec("vp_theta", "m/s", "real", "phase P-velocity at angle theta (ThomsenOperator)", "per_cell"),
    _spec("vsv_theta", "m/s", "real", "phase SV-velocity at angle theta (ThomsenOperator)", "per_cell"),
    _spec("vsh_theta", "m/s", "real", "phase SH-velocity at angle theta (ThomsenOperator)", "per_cell"),
    _spec("epsilon", "dimensionless", "real", "Thomsen anisotropy parameter epsilon", "per_cell"),
    _spec("delta", "dimensionless", "real", "Thomsen anisotropy parameter delta", "per_cell"),
    _spec("gamma", "dimensionless", "real", "Thomsen anisotropy parameter gamma", "per_cell"),
    # Effective-medium moduli (effective.py)
    _spec("K_eff", "Pa", "real", "effective bulk modulus (VRH default / DEM)", "per_cell"),
    _spec("mu_eff", "Pa", "real", "effective shear modulus (DEM)", "per_cell"),
    _spec("K_HS", "Pa", "real", "Hashin-Shtrikman bulk modulus", "per_cell"),
    _spec("mu_HS", "Pa", "real", "Hashin-Shtrikman shear modulus", "per_cell"),
    _spec("K_KT", "Pa", "real", "Kuster-Toksoz bulk modulus", "per_cell"),
    _spec("mu_KT", "Pa", "real", "Kuster-Toksoz shear modulus", "per_cell"),
    _spec("K_SC", "Pa", "real", "self-consistent bulk modulus", "per_cell"),
    _spec("mu_SC", "Pa", "real", "self-consistent shear modulus", "per_cell"),
    _spec("K_dry", "Pa", "real", "dry-rock bulk modulus (XuWhite/MacBeth/SoftSand/StiffSand)", "per_cell"),
    _spec("mu_dry", "Pa", "real", "dry-rock shear modulus", "per_cell"),
    # Strict-SI rock operator facades (rock/operators/*): lowercase SI channel set.
    _spec("k_dry", "Pa", "real", "dry-frame bulk modulus (HertzMindlin/frames facades)", "per_cell"),
    _spec("k_eff", "Pa", "real", "effective bulk modulus (SelfConsistent facade)", "per_cell"),
    _spec("k_sat", "Pa", "real", "saturated bulk modulus (Gassmann facade)", "per_cell"),
    _spec("mu_sat", "Pa", "real", "saturated shear modulus (Gassmann facade)", "per_cell"),
    _spec("bulk_modulus", "Pa", "real", "bulk modulus (mixtures/fluids/elastic facades)", "per_cell"),
    _spec("shear_modulus", "Pa", "real", "shear modulus (elastic facade)", "per_cell"),
    _spec("stiffness", "Pa", "real", "6x6 Voigt stiffness (Hudson/SayersKachanov facades)", "per_cell"),
    _spec("resistivity", "Ohm·m", "real", "electrical resistivity (ArchieResistivity facade)", "per_cell"),
    _spec("permeability", "m^2", "real", "permeability (KozenyCarman facade)", "per_cell"),
    _spec("fast_p_velocity", "m/s", "real", "Biot fast P velocity (BiotHighFrequency facade)", "per_cell"),
    _spec("slow_p_velocity", "m/s", "real", "Biot slow P velocity (BiotHighFrequency facade)", "per_cell"),
    _spec("shear_velocity", "m/s", "real", "shear velocity (BiotHighFrequency facade)", "per_cell"),
    _spec("density", "kg/m^3", "real", "bulk density grid (RockPhysicsTemplate facade)", "per_cell"),
    _spec("fluid_bulk_modulus", "Pa", "real", "pore-fluid bulk modulus (RockPhysicsTemplate facade)", "per_cell"),
    _spec("vp_vs_ratio", "dimensionless", "real", "Vp/Vs ratio (RockPhysicsTemplate facade)", "per_cell"),
    _spec("rho_dry", "kg/m^3", "real", "dry-rock density (XuWhite)", "per_cell"),
    # Granular (granular.py)
    _spec("K_HM", "Pa", "real", "Hertz-Mindlin bulk modulus", "per_cell"),
    _spec("mu_HM", "Pa", "real", "Hertz-Mindlin shear modulus", "per_cell"),
    _spec("K_W", "Pa", "real", "Walton bulk modulus", "per_cell"),
    _spec("mu_W", "Pa", "real", "Walton shear modulus", "per_cell"),
    # Fluid (fluid.py)
    _spec("K_fluid", "Pa", "real", "pore-fluid bulk modulus (BatzleWang/Wood/Brie)", "per_cell"),
    _spec("rho_fluid", "kg/m^3", "real", "pore-fluid density (BatzleWang/Wood/Brie)", "per_cell"),
)

CHANNEL_REGISTRY: dict[str, ChannelSpec] = {s.name: s for s in _SPECS}

# Guard against accidental duplicate keys silently overwriting (the tuple is
# the source; the dict must preserve every distinct name). A raise, not an
# assert: this must survive ``python -O``.
if len(CHANNEL_REGISTRY) != len(_SPECS):
    _dupes = sorted(
        {sp.name for sp in _SPECS if sum(x.name == sp.name for x in _SPECS) > 1}
    )
    raise GeoBrainError(
        "duplicate channel name in CHANNEL_REGISTRY _SPECS",
        object_name="CHANNEL_REGISTRY", field="name",
        expected="unique channel names", actual=_dupes,
    )


# --- Channel-renaming seam --------------------------------------------------
#
# When a channel is renamed, record ``old_name -> canonical_name`` here.
# ``resolve_channel`` then accepts the old name, emits a ``DeprecationWarning``,
# and returns the canonical name, so stored datasets and downstream code that
# still speak the old vocabulary keep working across a rename without forking
# the registry. Empty until the first rename: the seam exists so that the first
# rename is a one-line table entry rather than a cross-cutting change.
CHANNEL_ALIASES: dict[str, str] = {}


def resolve_channel(name: str) -> str:
    """Resolve a possibly-aliased channel name to its canonical form.

    Returns ``name`` unchanged when it is already canonical (the common case).
    When ``name`` is a registered alias, emits a ``DeprecationWarning`` and
    returns the canonical name it points to.
    """
    canonical = CHANNEL_ALIASES.get(name)
    if canonical is None:
        return name
    warnings.warn(
        f"channel name {name!r} is deprecated; use {canonical!r} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return canonical


# --- Per-family overrides for family-colliding channel keys -----------------
#
# A registry key maps to ONE ChannelSpec, but a few keys legitimately carry
# different semantics across physics families. The ``bx``/``by``/``bz`` keys are
# EM field components (complex, tesla) in the primary registry, while the
# potential-field (magnetic) family emits real anomaly components in tesla under
# the same keys. ``channel_spec(name, family=...)`` consults this table first so
# a caller that knows its family gets the correct unit/dtype instead of the
# EM default. Callers that pass no family keep the primary-registry behaviour.
_FAMILY_OVERRIDES: dict[tuple[str, str], ChannelSpec] = {
    ("potential", "bx"): _spec(
        "bx", "T", "real",
        "magnetic anomaly B_x component (potential family; real tesla, distinct "
        "from the EM complex-tesla registry entry only by dtype)", "per_receiver"),
    ("potential", "by"): _spec(
        "by", "T", "real",
        "magnetic anomaly B_y component (potential family; real tesla)",
        "per_receiver"),
    ("potential", "bz"): _spec(
        "bz", "T", "real",
        "magnetic anomaly B_z component (potential family; real tesla)",
        "per_receiver"),
}


def channel_spec(name: str, *, family: str | None = None) -> ChannelSpec | None:
    """Look up a channel's :class:`ChannelSpec`, resolving aliases first.

    Returns ``None`` when the (resolved) name is not in ``CHANNEL_REGISTRY``.

    ``family`` (optional) disambiguates a channel key whose semantics differ by
    physics family, currently the magnetic ``bx``/``by``/``bz`` anomaly
    components (``family="potential"``: real, tesla) versus the EM field components
    (the default: complex, tesla). An unknown ``family``, or a key with no
    family-specific override, falls back to the primary registry entry, so
    passing ``family`` is always safe.
    """
    resolved = resolve_channel(name)
    if family is not None:
        override = _FAMILY_OVERRIDES.get((family, resolved))
        if override is not None:
            return override
    return CHANNEL_REGISTRY.get(resolved)
