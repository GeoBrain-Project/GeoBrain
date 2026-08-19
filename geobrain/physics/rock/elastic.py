"""Canonical SI elasticity conversions and stiffness tensor writers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .errors import RockContractError

TensorInput: TypeAlias = torch.Tensor | int | float


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    tensors = cast(
        tuple[torch.Tensor, ...], require_compatible_tensors(object_name, *fields)
    )
    for (name, _), tensor in zip(fields, tensors):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "Rock elastic kernels require strided tensors",
                object_name=object_name,
                field=name,
                expected="torch.strided layout",
                actual=str(tensor.layout),
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "Rock elastic kernels require materialized values",
                object_name=object_name,
                field=name,
                expected="a materialized CPU or accelerator tensor",
                actual=str(tensor.device),
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "Rock elastic input must be finite",
                object_name=object_name,
                field=name,
                expected="finite values",
                actual="non-finite value(s)",
            )
    return tensors


def _require_positive(
    object_name: str,
    field: str,
    value: torch.Tensor,
) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "Rock elastic input must be positive",
            object_name=object_name,
            field=field,
            expected="> 0",
            actual={"minimum": value.amin().item(), "maximum": value.amax().item()},
        )


def _matrix(rows: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    return torch.stack(tuple(torch.stack(tuple(row), dim=-1) for row in rows), dim=-2)


def _require_positive_definite(object_name: str, stiffness: torch.Tensor) -> None:
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    if bool(torch.any(eigenvalues <= 0.0)):
        raise RockContractError(
            "Rock stiffness tensor must be positive definite",
            object_name=object_name,
            field="stiffness",
            expected="all eigenvalues > 0",
            actual={"minimum_eigenvalue": eigenvalues.amin().item()},
        )


def bulk_modulus_from_velocities(
    vp: torch.Tensor,
    vs: TensorInput,
    rho: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bulk and shear moduli in Pa from SI velocities and density.

    Args:
        vp: P-wave velocity [m/s].
        vs: S-wave velocity [m/s].
        rho: bulk density [kg/m^3].
    """

    vp, vs, rho = _validated_inputs(
        "bulk_modulus_from_velocities", ("vp", vp), ("vs", vs), ("rho", rho)
    )
    _require_positive("bulk_modulus_from_velocities", "vp", vp)
    _require_positive("bulk_modulus_from_velocities", "vs", vs)
    _require_positive("bulk_modulus_from_velocities", "rho", rho)
    vp, vs, rho = torch.broadcast_tensors(vp, vs, rho)
    shear_modulus = rho * vs.square()
    bulk_modulus = rho * vp.square() - vp.new_tensor(4.0 / 3.0) * shear_modulus
    _require_positive(
        "bulk_modulus_from_velocities", "bulk_modulus", bulk_modulus
    )
    return bulk_modulus, shear_modulus


def velocities_from_moduli(
    bulk_modulus: torch.Tensor,
    shear_modulus: TensorInput,
    rho: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P- and S-wave velocities in m/s from SI moduli and density.

    Args:
        bulk_modulus: bulk modulus K [Pa].
        shear_modulus: shear modulus mu [Pa].
        rho: bulk density [kg/m^3].
    """

    bulk_modulus, shear_modulus, rho = _validated_inputs(
        "velocities_from_moduli",
        ("bulk_modulus", bulk_modulus),
        ("shear_modulus", shear_modulus),
        ("rho", rho),
    )
    _require_positive("velocities_from_moduli", "bulk_modulus", bulk_modulus)
    _require_positive("velocities_from_moduli", "shear_modulus", shear_modulus)
    _require_positive("velocities_from_moduli", "rho", rho)
    bulk_modulus, shear_modulus, rho = torch.broadcast_tensors(
        bulk_modulus, shear_modulus, rho
    )
    p_wave_modulus = bulk_modulus + bulk_modulus.new_tensor(4.0 / 3.0) * shear_modulus
    return torch.sqrt(p_wave_modulus / rho), torch.sqrt(shear_modulus / rho)


def lame_lambda_from_moduli(
    bulk_modulus: torch.Tensor,
    shear_modulus: TensorInput,
) -> torch.Tensor:
    """Return the first Lamé parameter from bulk and shear modulus."""

    bulk_modulus, shear_modulus = _validated_inputs(
        "lame_lambda_from_moduli",
        ("bulk_modulus", bulk_modulus),
        ("shear_modulus", shear_modulus),
    )
    _require_positive("lame_lambda_from_moduli", "bulk_modulus", bulk_modulus)
    _require_positive("lame_lambda_from_moduli", "shear_modulus", shear_modulus)
    return bulk_modulus - bulk_modulus.new_tensor(2.0 / 3.0) * shear_modulus


def poisson_ratio_from_moduli(
    bulk_modulus: torch.Tensor,
    shear_modulus: TensorInput,
) -> torch.Tensor:
    """Return isotropic Poisson ratio from bulk and shear modulus."""

    bulk_modulus, shear_modulus = _validated_inputs(
        "poisson_ratio_from_moduli",
        ("bulk_modulus", bulk_modulus),
        ("shear_modulus", shear_modulus),
    )
    _require_positive("poisson_ratio_from_moduli", "bulk_modulus", bulk_modulus)
    _require_positive("poisson_ratio_from_moduli", "shear_modulus", shear_modulus)
    numerator = 3.0 * bulk_modulus - 2.0 * shear_modulus
    denominator = 6.0 * bulk_modulus + 2.0 * shear_modulus
    return numerator / denominator


def youngs_modulus_from_moduli(
    bulk_modulus: torch.Tensor,
    shear_modulus: TensorInput,
) -> torch.Tensor:
    """Return Young's modulus from bulk and shear modulus."""

    bulk_modulus, shear_modulus = _validated_inputs(
        "youngs_modulus_from_moduli",
        ("bulk_modulus", bulk_modulus),
        ("shear_modulus", shear_modulus),
    )
    _require_positive("youngs_modulus_from_moduli", "bulk_modulus", bulk_modulus)
    _require_positive("youngs_modulus_from_moduli", "shear_modulus", shear_modulus)
    return 9.0 * bulk_modulus * shear_modulus / (
        3.0 * bulk_modulus + shear_modulus
    )


def moduli_from_youngs_modulus_and_poisson_ratio(
    youngs_modulus: torch.Tensor,
    poisson_ratio: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bulk and shear modulus from Young's modulus and Poisson ratio."""

    youngs_modulus, poisson_ratio = _validated_inputs(
        "moduli_from_youngs_modulus_and_poisson_ratio",
        ("youngs_modulus", youngs_modulus),
        ("poisson_ratio", poisson_ratio),
    )
    _require_positive(
        "moduli_from_youngs_modulus_and_poisson_ratio",
        "youngs_modulus",
        youngs_modulus,
    )
    invalid = (poisson_ratio <= -1.0) | (poisson_ratio >= 0.5)
    if bool(torch.any(invalid)):
        raise RockContractError(
            "poisson_ratio is outside the stable isotropic interval",
            object_name="moduli_from_youngs_modulus_and_poisson_ratio",
            field="poisson_ratio",
            expected="-1 < poisson_ratio < 0.5",
            actual={
                "minimum": poisson_ratio.amin().item(),
                "maximum": poisson_ratio.amax().item(),
            },
        )
    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    return bulk_modulus, shear_modulus


def isotropic_stiffness(
    bulk_modulus: torch.Tensor,
    shear_modulus: TensorInput,
) -> torch.Tensor:
    """Assemble a broadcast-capable isotropic 6×6 Voigt stiffness tensor."""

    bulk_modulus, shear_modulus = _validated_inputs(
        "isotropic_stiffness",
        ("bulk_modulus", bulk_modulus),
        ("shear_modulus", shear_modulus),
    )
    _require_positive("isotropic_stiffness", "bulk_modulus", bulk_modulus)
    _require_positive("isotropic_stiffness", "shear_modulus", shear_modulus)
    bulk_modulus, shear_modulus = torch.broadcast_tensors(
        bulk_modulus, shear_modulus
    )
    lame_lambda = bulk_modulus - bulk_modulus.new_tensor(2.0 / 3.0) * shear_modulus
    axial = lame_lambda + 2.0 * shear_modulus
    zero = torch.zeros_like(axial)
    stiffness = _matrix(
        (
            (axial, lame_lambda, lame_lambda, zero, zero, zero),
            (lame_lambda, axial, lame_lambda, zero, zero, zero),
            (lame_lambda, lame_lambda, axial, zero, zero, zero),
            (zero, zero, zero, shear_modulus, zero, zero),
            (zero, zero, zero, zero, shear_modulus, zero),
            (zero, zero, zero, zero, zero, shear_modulus),
        )
    )
    return stiffness


def vti_stiffness(
    c11: torch.Tensor,
    c33: TensorInput,
    c13: TensorInput,
    c44: TensorInput,
    c66: TensorInput,
) -> torch.Tensor:
    """Assemble a VTI 6×6 Voigt stiffness tensor."""

    values = _validated_inputs(
        "vti_stiffness",
        ("c11", c11),
        ("c33", c33),
        ("c13", c13),
        ("c44", c44),
        ("c66", c66),
    )
    c11, c33, c13, c44, c66 = torch.broadcast_tensors(*values)
    for name, value in (("c11", c11), ("c33", c33), ("c44", c44), ("c66", c66)):
        _require_positive("vti_stiffness", name, value)
    c12 = c11 - 2.0 * c66
    zero = torch.zeros_like(c11)
    stiffness = _matrix(
        (
            (c11, c12, c13, zero, zero, zero),
            (c12, c11, c13, zero, zero, zero),
            (c13, c13, c33, zero, zero, zero),
            (zero, zero, zero, c44, zero, zero),
            (zero, zero, zero, zero, c44, zero),
            (zero, zero, zero, zero, zero, c66),
        )
    )
    _require_positive_definite("vti_stiffness", stiffness)
    return stiffness


def hti_stiffness(
    c11: torch.Tensor,
    c33: TensorInput,
    c13: TensorInput,
    c44: TensorInput,
    c55: TensorInput,
) -> torch.Tensor:
    """Assemble an HTI 6×6 Voigt stiffness tensor."""

    values = _validated_inputs(
        "hti_stiffness",
        ("c11", c11),
        ("c33", c33),
        ("c13", c13),
        ("c44", c44),
        ("c55", c55),
    )
    c11, c33, c13, c44, c55 = torch.broadcast_tensors(*values)
    for name, value in (("c11", c11), ("c33", c33), ("c44", c44), ("c55", c55)):
        _require_positive("hti_stiffness", name, value)
    c23 = c33 - 2.0 * c44
    zero = torch.zeros_like(c11)
    stiffness = _matrix(
        (
            (c11, c13, c13, zero, zero, zero),
            (c13, c33, c23, zero, zero, zero),
            (c13, c23, c33, zero, zero, zero),
            (zero, zero, zero, c44, zero, zero),
            (zero, zero, zero, zero, c55, zero),
            (zero, zero, zero, zero, zero, c55),
        )
    )
    _require_positive_definite("hti_stiffness", stiffness)
    return stiffness


def orthorhombic_stiffness(
    c11: torch.Tensor,
    c22: TensorInput,
    c33: TensorInput,
    c12: TensorInput,
    c13: TensorInput,
    c23: TensorInput,
    c44: TensorInput,
    c55: TensorInput,
    c66: TensorInput,
) -> torch.Tensor:
    """Assemble an orthorhombic 6×6 Voigt stiffness tensor."""

    names = ("c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66")
    raw_values: tuple[TensorInput, ...] = (
        c11, c22, c33, c12, c13, c23, c44, c55, c66
    )
    values = _validated_inputs(
        "orthorhombic_stiffness", *tuple(zip(names, raw_values))
    )
    c11, c22, c33, c12, c13, c23, c44, c55, c66 = torch.broadcast_tensors(
        *values
    )
    for name, value in zip(
        ("c11", "c22", "c33", "c44", "c55", "c66"),
        (c11, c22, c33, c44, c55, c66),
    ):
        _require_positive("orthorhombic_stiffness", name, value)
    zero = torch.zeros_like(c11)
    stiffness = _matrix(
        (
            (c11, c12, c13, zero, zero, zero),
            (c12, c22, c23, zero, zero, zero),
            (c13, c23, c33, zero, zero, zero),
            (zero, zero, zero, c44, zero, zero),
            (zero, zero, zero, zero, c55, zero),
            (zero, zero, zero, zero, zero, c66),
        )
    )
    _require_positive_definite("orthorhombic_stiffness", stiffness)
    return stiffness


__all__ = [
    "bulk_modulus_from_velocities",
    "hti_stiffness",
    "isotropic_stiffness",
    "lame_lambda_from_moduli",
    "moduli_from_youngs_modulus_and_poisson_ratio",
    "orthorhombic_stiffness",
    "poisson_ratio_from_moduli",
    "velocities_from_moduli",
    "vti_stiffness",
    "youngs_modulus_from_moduli",
]
