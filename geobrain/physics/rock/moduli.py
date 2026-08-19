"""
Shared rock-physics utility functions.

All functions accept scalars,
Python lists, or torch tensors via :func:`~geobrain.physics.rock.models._types.as_tensor`,
and propagate autograd through their outputs.

Sections:

- **Velocity ↔ moduli**: :func:`v_from_moduli`, :func:`moduli_from_v`.
  Conventions: ``K`` / ``G`` in Pa, ``ρ`` in kg/m³, ``Vp`` / ``Vs`` in
  m/s (SI throughout, no internal unit factors).
- **Elastic relations**: :func:`poisson`, :func:`lame_lambda`,
  :func:`youngs_modulus`, :func:`bulk_modulus`, :func:`shear_modulus`.
- **Stiffness writers**: :func:`write_iso_matrix`, :func:`write_vti_matrix`,
  :func:`write_hti_matrix`, :func:`write_ortho_matrix`.
- **Validators**: :func:`validate_porosity`, :func:`validate_positive`.
- **Granular helpers**: :func:`coordination_number` (Murphy 1982),
  :func:`critical_porosity_default`.
- **QI helpers**: :func:`matrix_modulus`, :func:`den_matrix`,
  :func:`normalize_kde`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ...core import GeoBrainError
from .models._types import EPS, TensorLike, as_tensor, ensure_same_device


# --- Velocity ↔ moduli ------------------------------------------------------


def v_from_moduli(
    K: TensorLike, G: TensorLike, rho: TensorLike,
) -> tuple[Tensor, Tensor]:
    """
    ``Vp = sqrt((K + 4G/3) / ρ)``,  ``Vs = sqrt(G / ρ)``.

    Inputs in Pa / kg·m⁻³, outputs in m/s.
    """
    K, G, rho = as_tensor(K), as_tensor(G), as_tensor(rho)
    K, G, rho = ensure_same_device(K, G, rho)
    M = K + 4.0 / 3.0 * G
    Vp = torch.sqrt(M / (rho + EPS))
    Vs = torch.sqrt(G / (rho + EPS))
    return Vp, Vs


def moduli_from_v(
    Vp: TensorLike, Vs: TensorLike, rho: TensorLike,
) -> tuple[Tensor, Tensor]:
    """Inverse of :func:`v_from_moduli`. m/s, kg·m⁻³ → Pa."""
    Vp, Vs, rho = as_tensor(Vp), as_tensor(Vs), as_tensor(rho)
    Vp, Vs, rho = ensure_same_device(Vp, Vs, rho)
    G = rho * Vs ** 2
    K = rho * Vp ** 2 - 4.0 * G / 3.0
    return K, G


# --- Elastic relations ------------------------------------------------------


def poisson(K: TensorLike, G: TensorLike) -> Tensor:
    """``ν = (3K − 2G) / (6K + 2G)``."""
    K, G = ensure_same_device(as_tensor(K), as_tensor(G))
    return (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G + EPS)


def lame_lambda(K: TensorLike, G: TensorLike) -> Tensor:
    """First Lamé parameter ``λ = K − 2G/3``."""
    K, G = ensure_same_device(as_tensor(K), as_tensor(G))
    return K - 2.0 * G / 3.0


def youngs_modulus(K: TensorLike, G: TensorLike) -> Tensor:
    """Young's modulus ``E = 9KG / (3K + G)``."""
    K, G = ensure_same_device(as_tensor(K), as_tensor(G))
    return 9.0 * K * G / (3.0 * K + G + EPS)


def bulk_modulus(E: TensorLike, nu: TensorLike) -> Tensor:
    """``K = E / (3(1 − 2ν))``."""
    E, nu = ensure_same_device(as_tensor(E), as_tensor(nu))
    return E / (3.0 * (1.0 - 2.0 * nu) + EPS)


def shear_modulus(E: TensorLike, nu: TensorLike) -> Tensor:
    """``G = E / (2(1 + ν))``."""
    E, nu = ensure_same_device(as_tensor(E), as_tensor(nu))
    return E / (2.0 * (1.0 + nu) + EPS)


# --- Validators -------------------------------------------------------------


def validate_porosity(
    phi: TensorLike, name: str = "phi", clamp: bool = False,
) -> Tensor:
    """Validate ``phi ∈ [0, 1]``. Optionally clamp instead of raising."""
    phi = as_tensor(phi)
    if clamp:
        return torch.clamp(phi, min=0.0, max=1.0)
    if (phi < 0.0).any() or (phi > 1.0).any():
        raise ValueError(
            f"{name} must be in [0, 1], got values in "
            f"[{phi.min().item():.4f}, {phi.max().item():.4f}]"
        )
    return phi


def validate_positive(x: TensorLike, name: str) -> Tensor:
    """Raise ``ValueError`` if any entry is non-positive."""
    x = as_tensor(x)
    if (x <= 0.0).any():
        raise ValueError(f"{name} must be positive, got min value {x.min().item():.6f}")
    return x


def check_bounded(
    x: TensorLike,
    name: str,
    *,
    owner: str,
    lower: float = 0.0,
    upper: float = 1.0,
    upper_inclusive: bool = True,
) -> Tensor:
    """Validate every entry of ``x`` is finite and within ``[lower, upper]``.

    Entry-validation helper for operator ``_forward`` methods that must reject
    non-physical saturations / porosities (e.g. ``Sw`` outside ``[0, 1]`` or a
    granular porosity above the critical porosity) instead of silently
    returning finite garbage. Raises a structured :class:`GeoBrainError` on the
    first non-finite or out-of-range entry; returns ``x`` unchanged otherwise.
    """
    x = as_tensor(x)
    if not bool(torch.isfinite(x).all()):
        raise GeoBrainError(
            f"{owner} {name} must be finite",
            object_name=owner, field=name,
            expected="finite", actual="non-finite value(s)",
        )
    below = x < lower
    above = (x > upper) if upper_inclusive else (x >= upper)
    if bool(below.any()) or bool(above.any()):
        rb = "]" if upper_inclusive else ")"
        raise GeoBrainError(
            f"{owner} {name} must lie in [{lower}, {upper}{rb}",
            object_name=owner, field=name,
            expected=f"[{lower}, {upper}{rb}",
            actual=f"[{float(x.min()):.4g}, {float(x.max()):.4g}]",
        )
    return x


# --- Stiffness matrices (6×6 Voigt notation) --------------------------------


def write_iso_matrix(K: TensorLike, G: TensorLike) -> Tensor:
    """Isotropic stiffness ``C(K, G)``.

    Tensor inputs keep their dtype (so float64 stays float64); non-tensor
    inputs follow the library's float32 default.
    """
    dtype = K.dtype if isinstance(K, Tensor) else torch.float32
    K, G = ensure_same_device(
        as_tensor(K, dtype=dtype).squeeze(), as_tensor(G, dtype=dtype).squeeze(),
    )
    lamda = K - 2.0 * G / 3.0
    C11 = lamda + 2.0 * G

    C = torch.zeros(6, 6, dtype=K.dtype, device=K.device)
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = G
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lamda
    return C


def write_vti_matrix(
    C11: TensorLike, C33: TensorLike, C13: TensorLike,
    C44: TensorLike, C66: TensorLike,
) -> Tensor:
    """VTI stiffness (symmetry axis = vertical)."""
    args = [as_tensor(c).squeeze() for c in (C11, C33, C13, C44, C66)]
    args = ensure_same_device(*args)
    C11, C33, C13, C44, C66 = args

    C12 = C11 - 2.0 * C66
    C = torch.zeros(6, 6, dtype=C11.dtype, device=C11.device)
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    C[3, 3] = C[4, 4] = C44
    C[5, 5] = C66
    return C


def write_hti_matrix(
    C11: TensorLike, C33: TensorLike, C13: TensorLike,
    C44: TensorLike, C55: TensorLike,
) -> Tensor:
    """HTI stiffness (symmetry axis = x₁)."""
    args = [as_tensor(c).squeeze() for c in (C11, C33, C13, C44, C55)]
    args = ensure_same_device(*args)
    C11, C33, C13, C44, C55 = args

    C23 = C33 - 2.0 * C44
    C = torch.zeros(6, 6, dtype=C11.dtype, device=C11.device)
    C[0, 0] = C11
    C[1, 1] = C[2, 2] = C33
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C23
    C[3, 3] = C44
    C[4, 4] = C[5, 5] = C55
    return C


def write_ortho_matrix(
    C11: TensorLike, C22: TensorLike, C33: TensorLike,
    C12: TensorLike, C13: TensorLike, C23: TensorLike,
    C44: TensorLike, C55: TensorLike, C66: TensorLike,
) -> Tensor:
    """Orthorhombic stiffness (3 mutually perpendicular symmetry planes)."""
    args = [as_tensor(c).squeeze() for c in (C11, C22, C33, C12, C13, C23, C44, C55, C66)]
    args = ensure_same_device(*args)
    C11, C22, C33, C12, C13, C23, C44, C55, C66 = args

    C = torch.zeros(6, 6, dtype=C11.dtype, device=C11.device)
    C[0, 0] = C11
    C[1, 1] = C22
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C13
    C[1, 2] = C[2, 1] = C23
    C[3, 3] = C44
    C[4, 4] = C55
    C[5, 5] = C66
    return C


# --- Granular helpers -------------------------------------------------------


def coordination_number(phi: TensorLike) -> Tensor:
    """
    Murphy's relation ``Cn = 20 − 34φ + 14φ²``, clamped to ``[4, 12]``.

    The polynomial alone overshoots at the porosity extremes; the clamp
    keeps the value in a physically reasonable range so downstream
    Hertz-Mindlin / Soft-Sand don't see absurd contact counts.
    """
    phi = as_tensor(phi)
    Cn = 20.0 - 34.0 * phi + 14.0 * phi ** 2
    return torch.clamp(Cn, min=4.0, max=12.0)


def critical_porosity_default(lithology: str = "sandstone") -> float:
    """Typical critical porosity by lithology (used as model default)."""
    defaults = {
        "sandstone": 0.40,
        "sand": 0.40,
        "limestone": 0.35,
        "carbonate": 0.35,
        "shale": 0.30,
        "chalk": 0.65,
        "clay": 0.55,
    }
    return defaults.get(lithology.lower(), 0.40)


# --- QI helpers -------------------------------------------------------------


def matrix_modulus(
    vsh: TensorLike, phi_c: TensorLike, phi_0: TensorLike,
    M_sh: TensorLike, M_g: TensorLike, M_c: TensorLike,
) -> Tensor:
    """
    VRH (Hill-average) matrix modulus during cementation.

    Mixes shale (``vsh``), grain (``M_g``), and cement (``M_c``)
    fractions at intermediate porosity ``phi_0 ≤ phi_c``.
    """
    vsh = as_tensor(vsh)
    phi_c = as_tensor(phi_c)
    phi_0 = as_tensor(phi_0)
    M_sh, M_g, M_c = as_tensor(M_sh), as_tensor(M_g), as_tensor(M_c)
    vsh, phi_c, phi_0, M_sh, M_g, M_c = ensure_same_device(
        vsh, phi_c, phi_0, M_sh, M_g, M_c,
    )

    fc = phi_c - phi_0
    denom = 1.0 - phi_0 + EPS
    vsh_s = vsh / denom
    fgs = (1.0 - phi_c - vsh) / denom
    fcs = fc / denom

    M_voigt = vsh_s * M_sh + fgs * M_g + fcs * M_c
    M_reuss = 1.0 / (vsh_s / (M_sh + EPS) + fgs / (M_g + EPS) + fcs / (M_c + EPS) + EPS)
    return 0.5 * (M_voigt + M_reuss)


def den_matrix(
    vsh: TensorLike, phi_c: TensorLike, phi_0: TensorLike,
    den_sh: TensorLike, den_g: TensorLike, den_c: TensorLike,
) -> Tensor:
    """Voigt-average matrix density during cementation (linear mix by fraction)."""
    vsh = as_tensor(vsh)
    phi_c = as_tensor(phi_c)
    phi_0 = as_tensor(phi_0)
    den_sh, den_g, den_c = as_tensor(den_sh), as_tensor(den_g), as_tensor(den_c)
    vsh, phi_c, phi_0, den_sh, den_g, den_c = ensure_same_device(
        vsh, phi_c, phi_0, den_sh, den_g, den_c,
    )

    fc = phi_c - phi_0
    denom = 1.0 - phi_0 + EPS
    vsh_s = vsh / denom
    fgs = (1.0 - phi_c - vsh) / denom
    fcs = fc / denom
    return vsh_s * den_sh + fgs * den_g + fcs * den_c


def normalize_kde(density: TensorLike) -> Tensor:
    """Normalize a KDE so the maximum value maps to 1.0."""
    density = as_tensor(density)
    return density / (density.max() + EPS)


def vrh_average(values, fractions, *, eps: float = 0.0) -> Tensor:
    """Voigt-Reuss-Hill average ``0.5·(Σ fᵢ·vᵢ + 1 / Σ(fᵢ/vᵢ))``.

    Shared by the effective-medium VRH bounds, the 4-mineral velocity mix in
    GreenbergCastagna, and the V-R-H mineral mix in XuWhite. ``eps`` regularises
    the Reuss (harmonic) denominator; pass each caller's historical value
    explicitly so the numerics stay bit-identical.
    """
    voigt = sum(f * v for f, v in zip(fractions, values))
    reuss = 1.0 / (sum(f / v for f, v in zip(fractions, values)) + eps)
    return 0.5 * (voigt + reuss)


__all__ = [
    "bulk_modulus",
    "coordination_number",
    "critical_porosity_default",
    "den_matrix",
    "lame_lambda",
    "matrix_modulus",
    "moduli_from_v",
    "normalize_kde",
    "poisson",
    "shear_modulus",
    "v_from_moduli",
    "validate_porosity",
    "validate_positive",
    "vrh_average",
    "write_hti_matrix",
    "write_iso_matrix",
    "write_ortho_matrix",
    "write_vti_matrix",
    "youngs_modulus",
]
