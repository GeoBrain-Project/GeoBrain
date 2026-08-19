"""SI rock Link-ready authored compositions: the module IS its
compositions (a generic bucket name here would fall to the naming
governance ban on ``functions``/``utils``).

Four plain functions, each returning ONE SI tensor from SI tensor inputs,
built for :class:`geobrain.geomodel.earthmodel.Link`: ``Link(gassmann_vp, args=("phi",
"sw"), params={...})`` wires one of these straight into an
:class:`~geobrain.geomodel.earthmodel.EarthModel`'s dependency DAG, every keyword-only
argument here maps 1:1 onto ``Link.params``, and every function is fully
differentiable end to end (plain ``torch`` composition of the existing SI
rock kernels, no custom autograd).

These are AUTHORED compositions, not aliases: each stitches together
existing single-purpose kernels from ``geobrain.physics.rock.models`` /
``.moduli`` (never reimplementing their math; see each stage's docstring
for the exact kernel it calls) into the one multi-stage pipeline a
porosity/saturation-parameterized earth model actually needs.

Functions:
    gardner_rho:    density from Vp (Gardner 1974): ``rho = a * vp**b``.
    gassmann_vp:    the 4-stage porosity/saturation -> Vp composition:
                    (i) fluid mix, (ii) SoftSand dry frame, (iii) Gassmann
                    fluid substitution, (iv) bulk density + Vp.
    gassmann_rho:   stage (iv)'s bulk density alone, so ``gassmann_vp`` and
                    ``gassmann_rho`` share ONE density parameterization.
    archie_sigma:   Archie's law, conductivity form.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...core import GeoBrainError
from .models._types import EPS
from .models._types import F as _HM_F
from .models._fluid_gassmann import gassmann_k_sat
from .models import Brie, Wood, hm_moduli_v05, soft_sand_math

__all__ = ["gardner_rho", "gassmann_vp", "gassmann_rho", "archie_sigma"]


# ---------------------------------------------------------------------------
# Shared coercion helpers
# ---------------------------------------------------------------------------


def _as_tensor(x: Tensor | float) -> Tensor:
    """Promote a bare Python scalar to a tensor; pass an existing tensor
    through UNCHANGED (a plain ``torch.as_tensor`` cast is safe for scalars,
    but re-casting an existing tensor risks silently detaching/mutating an
    autograd leaf or its dtype, e.g. downcasting a float64 test tensor)."""
    return x if isinstance(x, Tensor) else torch.as_tensor(x)


def _tensor_like(x: Tensor | float, ref: Tensor) -> Tensor:
    """Same promotion as :func:`_as_tensor`, but a non-tensor is cast to
    ``ref``'s dtype/device. Needed before calling kernels that read
    ``.dtype``/``.device`` off a modulus argument (:func:`hm_moduli_v05`);
    a bare Python float has neither."""
    if isinstance(x, Tensor):
        return x
    return torch.tensor(float(x), dtype=ref.dtype, device=ref.device)


# ---------------------------------------------------------------------------
# gardner_rho
# ---------------------------------------------------------------------------


def gardner_rho(vp: Tensor | float, *, a: float = 310.0, b: float = 0.25) -> Tensor:
    """Gardner et al. (1974) empirical density-velocity relation.

    ``rho = a * vp**b``. Trivial (single-stage), kept here rather than
    aliased from :class:`~geobrain.physics.rock.models.empirical.Gardner`
    so it is a bare function with the same 1:1 kwargs-onto-``Link.params``
    shape as the other three (the operator class exists for the
    ``ModelState`` world; this is the Link world).

    Args:
        vp: P-wave velocity (m/s).
        a: Gardner coefficient (SI default 310.0 for vp in m/s and rho in
            kg/m^3, the SI form of the classic g/cm^3 coefficient (SI-EXEMPT:
            historical note) 0.31, ``a_SI = 1000 * a_gcm3``; see the
            ``Gardner`` operator's docstring for the historical unit note).
        b: Gardner exponent (dimensionless), default 0.25.

    Returns:
        Bulk density (kg/m^3), same shape as ``vp``.
    """
    from .empirical import gardner_density

    vp = _as_tensor(vp)
    return gardner_density(vp, a, b)


# ---------------------------------------------------------------------------
# gassmann_vp / gassmann_rho: the 4-stage composition
# ---------------------------------------------------------------------------


def _fluid_mix_k(
    sw: Tensor, k_brine: Tensor | float, k_gas: Tensor | float,
    rho_brine: Tensor | float, rho_gas: Tensor | float,
    mix: str, brie_e: Tensor | float,
) -> Tensor:
    """Stage (i): mixed-fluid bulk modulus ``K_fl(sw)`` over (brine, gas).

    Calls the EXACT existing SI mixer kernel, ``Wood``/``Brie``'s
    ``compute_fluid_mix`` "workflow hook" (``models/_fluid_mixing.py``),
    written precisely for this "the workflow centralises the hydrocarbon/gas
    endmember" composition (its docstring's own words): brine is ``fl1``,
    gas is ``fl2``, both passed in directly as tensors rather than baked
    into the model's constructor, so gradients flow through ``k_gas`` /
    ``rho_gas`` too when they arrive as trainable ``Link.params``.

    ``mix="wood"``: Reuss (harmonic) average, uniform/fine saturation.
    ``mix="brie"``: Brie et al. (1995) empirical patchy-saturation mixing;
    ``brie_e`` controls patchiness (e=1 patchy end, e -> inf uniform).
    """
    if mix == "wood":
        K_fl, _rho_fl = Wood().compute_fluid_mix(k_brine, k_gas, rho_brine, rho_gas, sw)
    elif mix == "brie":
        K_fl, _rho_fl = Brie(exponent=float(brie_e)).compute_fluid_mix(
            k_brine, k_gas, rho_brine, rho_gas, sw
        )
    else:
        raise GeoBrainError(
            "gassmann_vp mix= must be 'wood' or 'brie'",
            object_name="gassmann_vp", field="mix",
            expected="'wood' or 'brie'", actual=mix,
        )
    return K_fl


def _dry_rock_moduli(
    phi: Tensor, k_mineral: Tensor, mu_mineral: Tensor,
    phi_c: Tensor | float, coord: Tensor | float, p_eff: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Stage (ii): SoftSand dry-frame moduli ``K_dry(phi)``, ``mu_dry(phi)``.

    The exact two-call composition
    :meth:`~geobrain.physics.rock.models.granular.SoftSand.compute_dry_rock`
    uses internally: Hertz-Mindlin pack moduli AT the critical porosity
    (:func:`hm_moduli_v05`, rough-contact ``f=1.0``) interpolated down to the
    actual porosity ``phi`` via the modified-HS-lower formula
    (:func:`soft_sand_math`). Mirrored here as a plain function (no
    ``SoftSand`` ``nn.Module`` instantiation needed, ``compute_dry_rock``
    itself never touches ``self``) so this stays a bare composition of the
    two module-level kernels, unit-for-unit identical to the ``SoftSand``
    operator's own dry-rock stage (the composition test suite
    cross-validates the two).

    ``k_mineral``/``mu_mineral`` are expected as tensors already (matching
    ``phi``'s dtype/device), :func:`gassmann_vp` coerces them once via
    :func:`_tensor_like` before calling here.
    """
    K_HM, mu_HM = hm_moduli_v05(k_mineral, mu_mineral, phi_c, coord, p_eff, _HM_F)
    K_dry, mu_dry = soft_sand_math(k_mineral, mu_mineral, K_HM, mu_HM, phi, phi_c)
    return K_dry, mu_dry


def _bulk_density(
    phi: Tensor, sw: Tensor,
    rho_mineral: Tensor | float, rho_brine: Tensor | float, rho_gas: Tensor | float,
) -> Tensor:
    """The shared SI bulk-density parameterization:

    ``rho = (1 - phi) * rho_mineral + phi * (sw * rho_brine + (1 - sw) * rho_gas)``

    Used by BOTH :func:`gassmann_vp` (stage iv) and :func:`gassmann_rho`,
    factored out so the two genuinely share one formula rather than two
    independently-typed copies of the same physics.
    """
    return (1.0 - phi) * rho_mineral + phi * (sw * rho_brine + (1.0 - sw) * rho_gas)


def gassmann_vp(
    phi: Tensor | float,
    sw: Tensor | float,
    *,
    k_mineral: float = 36.6e9,
    mu_mineral: float = 45.0e9,
    rho_mineral: float = 2650.0,
    k_brine: float = 2.25e9,
    rho_brine: float = 1030.0,
    k_gas: float = 0.05e9,
    rho_gas: float = 150.0,
    phi_c: float = 0.4,
    coord: float = 8.6,
    p_eff: float = 20.0e6,
    mix: str = "brie",
    brie_e: float = 3.0,
) -> Tensor:
    """Gassmann-saturated P-wave velocity: porosity + saturation -> Vp (m/s).

    The 4-stage SI composition (each stage reuses an existing rock-physics
    kernel; none of the math below is reimplemented; see each helper's
    docstring for the exact kernel):

    (i)   Fluid mix: ``K_fl(sw)``, Wood (Reuss) or Brie (patchy) mixing of
          brine (``k_brine``, ``rho_brine``) and gas (``k_gas``, ``rho_gas``),
          :func:`_fluid_mix_k`.
    (ii)  Dry rock: ``K_dry(phi)``, ``mu_dry(phi)`` via the SoftSand path,
          Hertz-Mindlin pack moduli at the critical porosity ``phi_c``
          (coordination number ``coord``, effective pressure ``p_eff``)
          interpolated down to ``phi``, :func:`_dry_rock_moduli`.
    (iii) Gassmann fluid substitution: ``K_dry -> K_sat`` via
          :func:`~geobrain.physics.rock.models._fluid_gassmann.gassmann_k_sat`
          (the platform's single source of truth for the Gassmann formula).
    (iv)  Bulk density (:func:`_bulk_density`) + ``vp = sqrt((K_sat +
          4/3 mu_dry) / rho_bulk)`` (``mu_sat = mu_dry``, Gassmann's
          shear-invariance assumption, same as the ``Gassmann`` operator).
          Written out directly rather than through
          :func:`~geobrain.physics.rock.moduli.v_from_moduli`: that helper
          routes every input through ``rock.models._types.as_tensor``, whose
          ``dtype`` parameter defaults to float32 and is applied even to an
          ALREADY-float64 tensor (``x.to(dtype)`` unconditionally), a
          silent double -> single precision downcast for any float64 caller
          (uncaught by the existing ``v_from_moduli`` tests, which only ever
          pass default-dtype float32 tensors). This function promises SI
          float64-safe autograd end to end, so stage (iv) is the same
          two-line formula ``v_from_moduli`` itself computes internally
          (``Vp = sqrt((K + 4G/3) / (rho + EPS))``), just without that
          upstream cast.

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water (brine) saturation (fraction, 0-1); ``1 - sw`` is the gas
            saturation.
        k_mineral: Mineral (grain) bulk modulus, Pa. Default is a
            quartz/clay-ish VRH-style blend (36.6 GPa).
        mu_mineral: Mineral (grain) shear modulus, Pa (45.0 GPa default).
        rho_mineral: Mineral (grain) density, kg/m^3 (2650.0, quartz).
        k_brine: Brine bulk modulus, Pa (2.25 GPa default).
        rho_brine: Brine density, kg/m^3 (1030.0 default).
        k_gas: Gas bulk modulus, Pa (0.05 GPa default, light hydrocarbon
            gas, deliberately soft relative to brine).
        rho_gas: Gas density, kg/m^3 (150.0 default).
        phi_c: Critical (unconsolidated-pack) porosity for the Hertz-Mindlin
            / SoftSand dry-frame stage (0.4 default).
        coord: Granular coordination number (contacts per grain), 8.6
            default.
        p_eff: Effective (overburden minus pore) pressure, Pa (20 MPa
            default).
        mix: ``"wood"`` or ``"brie"``, the stage-(i) fluid-mixing law.
        brie_e: Brie patchiness exponent (only used when ``mix="brie"``). NOT
            differentiable when passed as a trainable ``Link.params`` tensor:
            the underlying ``Brie`` model casts its ``exponent`` constructor
            argument to a plain ``float`` (``models/_fluid_mixing.py``), so
            the gradient graph is severed at that one parameter. Every OTHER
            keyword argument here (including ``k_gas``/``rho_gas`` via the
            ``compute_fluid_mix`` workflow hook, and ``phi_c``/``coord``/
            ``p_eff`` via ``hm_moduli_v05``) stays differentiable end to end.

    Returns:
        P-wave velocity (m/s), broadcast shape of ``phi``/``sw``.
    """
    phi = _as_tensor(phi)
    sw = _as_tensor(sw)
    k_mineral_t = _tensor_like(k_mineral, phi)
    mu_mineral_t = _tensor_like(mu_mineral, phi)

    K_fl = _fluid_mix_k(sw, k_brine, k_gas, rho_brine, rho_gas, mix, brie_e)
    K_dry, mu_dry = _dry_rock_moduli(phi, k_mineral_t, mu_mineral_t, phi_c, coord, p_eff)
    K_sat = gassmann_k_sat(K_dry, k_mineral_t, K_fl, phi)
    rho_bulk = _bulk_density(phi, sw, rho_mineral, rho_brine, rho_gas)

    # See the docstring: v_from_moduli's as_tensor(dtype=float32 default)
    # would silently downcast a float64 chain: the same EPS-guarded
    # formula it uses internally, without the cast.
    vp = torch.sqrt((K_sat + (4.0 / 3.0) * mu_dry) / (rho_bulk + EPS))
    return vp


def gassmann_rho(
    phi: Tensor | float,
    sw: Tensor | float,
    *,
    rho_mineral: float = 2650.0,
    rho_brine: float = 1030.0,
    rho_gas: float = 150.0,
) -> Tensor:
    """The Gassmann bulk density matching :func:`gassmann_vp`'s stage (iv),
    same ``rho_mineral``/``rho_brine``/``rho_gas`` parameterization, kept as
    its own ``Link``-ready function so a model can wire ``vp`` and ``rho``
    from the SAME (phi, sw) unknowns without duplicating the formula (see
    :func:`_bulk_density`, shared by both).

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water (brine) saturation (fraction, 0-1).
        rho_mineral: Mineral (grain) density, kg/m^3 (2650.0 default).
        rho_brine: Brine density, kg/m^3 (1030.0 default).
        rho_gas: Gas density, kg/m^3 (150.0 default).

    Returns:
        Bulk density (kg/m^3), broadcast shape of ``phi``/``sw``.
    """
    phi = _as_tensor(phi)
    sw = _as_tensor(sw)
    return _bulk_density(phi, sw, rho_mineral, rho_brine, rho_gas)


# ---------------------------------------------------------------------------
# archie_sigma
# ---------------------------------------------------------------------------


def archie_sigma(
    phi: Tensor | float,
    sw: Tensor | float,
    *,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    sigma_w: float = 3.0,
) -> Tensor:
    """Archie's law (1942), electrical-conductivity form.

    ``sigma = sigma_w * phi**m * sw**n / a`` (S/m), the reciprocal of the
    resistivity form ``R_t = a * Rw * phi**(-m) * sw**(-n)`` used by
    :class:`~geobrain.physics.rock.models.resistivity.ArchieResistivity`
    (``sigma_w = 1 / Rw``); written directly in conductivity because that is
    :data:`~geobrain.geomodel.earthmodel.field_specs.FIELD_SPECS`'s SI unit for
    ``"sigma"`` (S/m), and because a Link output should not need a
    downstream ``1/x`` just to land in the model's SI vocabulary.

    Args:
        phi: Porosity (fraction, 0-1).
        sw: Water saturation (fraction, 0-1).
        a: Tortuosity factor (dimensionless, default 1.0).
        m: Cementation exponent (dimensionless, default 2.0), a plausible
            ``Link.params`` inversion target (e.g. lithology-dependent);
            differentiable end to end, including through ``sw**n`` w.r.t.
            ``n`` and ``phi**m`` w.r.t. ``m``.
        n: Saturation exponent (dimensionless, default 2.0).
        sigma_w: Formation-water conductivity, S/m (default 3.0, a
            moderately saline brine).

    Returns:
        Bulk electrical conductivity (S/m), broadcast shape of ``phi``/``sw``.
    """
    from .petrophysics import archie_conductivity

    phi = _as_tensor(phi)
    return archie_conductivity(
        phi,
        sw,
        sigma_w,
        tortuosity_factor=a,
        cementation_exponent=m,
        saturation_exponent=n,
    )
