"""
Hankel / Sincos digital linear filter transforms.

Filters live as verified ``.npz`` resources alongside this module. Immutable
coefficient sources are cached by asset, dtype, and device; public loaders
always return defensive tensor copies. Public surface:

    load_hankel_filter(name="key_201_2012") -> dict[str, torch.Tensor]
    load_sincos_filter(name="sincos_201")   -> dict[str, torch.Tensor]
    dlf_hankel(integrand_fn, r, *, kind, filter_name="key_201_2012")
    dlf_sincos(integrand_fn, t, *, kind, filter_name="sincos_201")

Filter coefficients: the shipped 201-point Hankel (J0/J1) and Fourier
(sin/cos) tables are the digital linear filters of Kerry Key, "Is the fast
Hankel transform faster than quadrature?", Geophysics 77(3), F21-F30, 2012,
DOI 10.1190/geo2011-0237.1, licensed CC BY 4.0 (filter id ``key_201_2012``). Repackaged as ``.npz``, no
numerical changes. Machine-readable family provenance lives in
``provenance.json``; repository-notice integration is tracked separately.

The differentiable tensor J0 evaluator is a coefficient-preserving port of
``scipy/xsf``'s ``include/xsf/cephes/j0.h`` at xsf commit
``0d0a593fd31073af10062d0093144e13ae34f8f3`` (Cephes 2.8,
BSD-3-Clause). Exact source hashes and the complete license text ship in
``cephes_j0_provenance.json`` and ``CEPHES_J0_LICENSE.txt``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.resources
import io
import json
import math
from typing import Any, cast

import numpy as np
import torch

from geobrain.core.errors import ErrorCode
from geobrain.physics.em.errors import (
    EMCapabilityError,
    EMContractError,
    EMNumericsError,
)


_HANKEL_KEYS = ("base", "j0", "j1")
_SINCOS_KEYS = ("base", "sin", "cos")
_PACKAGE = "geobrain.physics.em.numerics.hankel"
_AXIAL_MAX_REFINEMENTS = 12

_J0_DR1 = 5.78318596294678452118
_J0_DR2 = 30.4712623436620863991
_J0_RP = (
    -4.79443220978201773821e9,
    1.95617491946556577543e12,
    -2.49248344360967716210e14,
    9.70862251047306323952e15,
)
_J0_RQ = (
    4.99563147152651017219e2,
    1.73785401676374683123e5,
    4.84409658339962045305e7,
    1.11855537045356834862e10,
    2.11277520115489217587e12,
    3.10518229857422583814e14,
    3.18121955943204943306e16,
    1.71086294081043136091e18,
)
_J0_PP = (
    7.96936729297347051624e-4,
    8.28352392107440799803e-2,
    1.23953371646414299388,
    5.44725003058768775090,
    8.74716500199817011941,
    5.30324038235394892183,
    1.0,
)
_J0_PQ = (
    9.24408810558863637013e-4,
    8.56288474354474431428e-2,
    1.25352743901058953537,
    5.47097740330417105182,
    8.76190883237069594232,
    5.30605288235394617618,
    1.0,
)
_J0_QP = (
    -1.13663838898469149931e-2,
    -1.28252718670509318512,
    -19.5539544257735972385,
    -93.2060152123768231369,
    -177.681167980488050595,
    -147.077505154951170175,
    -51.4105326766599330220,
    -6.05014350600728481186,
)
_J0_QQ = (
    64.3178256118178023184,
    856.430025976980587198,
    3882.40183605401609683,
    7240.46774195652478189,
    5930.72701187316984827,
    2062.09331660327847417,
    242.005740240291393179,
)


def _polevl(value: torch.Tensor, coefficients: tuple[float, ...]) -> torch.Tensor:
    """Evaluate a polynomial in descending coefficient order."""
    result = torch.full_like(value, coefficients[0])
    for coefficient in coefficients[1:]:
        result = result * value + coefficient
    return result


def _p1evl(value: torch.Tensor, coefficients: tuple[float, ...]) -> torch.Tensor:
    """Evaluate a polynomial with an implicit leading coefficient of one."""
    result = value + coefficients[0]
    for coefficient in coefficients[1:]:
        result = result * value + coefficient
    return result


def _torch_bessel_j0(argument: torch.Tensor) -> torch.Tensor:
    """Differentiable float64 Cephes J0 approximation on CPU and CUDA.

    The rational/asymptotic coefficient tables are the double-precision
    Cephes approximation also used by SciPy. Keeping the evaluation in torch
    avoids SciPy/CPU detours and preserves radius autograd on CUDA.
    """
    magnitude = torch.abs(argument)
    squared = magnitude * magnitude
    low = (
        (squared - _J0_DR1)
        * (squared - _J0_DR2)
        * _polevl(squared, _J0_RP)
        / _p1evl(squared, _J0_RQ)
    )
    low = torch.where(magnitude < 1.0e-5, 1.0 - 0.25 * squared, low)

    safe_magnitude = torch.clamp(magnitude, min=5.0)
    inverse_scale = 5.0 / safe_magnitude
    asymptotic_coordinate = inverse_scale * inverse_scale
    cosine_amplitude = _polevl(asymptotic_coordinate, _J0_PP) / _polevl(
        asymptotic_coordinate,
        _J0_PQ,
    )
    sine_amplitude = _polevl(asymptotic_coordinate, _J0_QP) / _p1evl(
        asymptotic_coordinate,
        _J0_QQ,
    )
    phase = safe_magnitude - math.pi / 4.0
    high = (
        (cosine_amplitude * torch.cos(phase) - inverse_scale * sine_amplitude * torch.sin(phase))
        * math.sqrt(2.0 / math.pi)
        / torch.sqrt(safe_magnitude)
    )
    return torch.where(magnitude <= 5.0, low, high)


@dataclass(frozen=True, slots=True)
class HankelAssetRecord:
    """Immutable runtime view of one verified DLF provenance record."""

    filename: str
    sha256: str
    filter_family: str
    coefficient_count: int
    coefficient_keys: tuple[str, ...]
    publication_author: str
    publication_title: str
    publication_journal: str
    publication_year: int
    publication_doi: str
    upstream_project: str
    upstream_release: str
    upstream_commit: str
    upstream_path: str
    license_identifier: str
    attribution_text: str
    acquisition_date: str
    transformation: str
    verification_command: tuple[str, ...]


def _artifact_invalid(
    message: str,
    *,
    asset: str,
    expected: object,
    actual: object,
) -> EMContractError:
    return EMContractError(
        message,
        details={"asset": asset, "expected": expected, "actual": actual},
        object_name="verify_hankel_asset",
        field="payload",
        expected=expected,
        actual=actual,
        code=ErrorCode.ARTIFACT_INVALID,
    )


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _artifact_invalid(
            "Hankel provenance record is malformed",
            asset="provenance.json",
            expected=f"mapping at {field}",
            actual=type(value).__name__,
        )
    return value


@lru_cache(maxsize=1)
def _provenance_records() -> dict[str, HankelAssetRecord]:
    resource = importlib.resources.files(_PACKAGE) / "provenance.json"
    try:
        manifest = json.loads(resource.read_text(encoding="utf-8"))
        root = _require_mapping(manifest, field="root")
        if root.get("schema") != "geobrain.em-hankel-provenance/1.0":
            raise ValueError("unsupported provenance schema")
        assets = _require_mapping(root.get("assets"), field="assets")
        records: dict[str, HankelAssetRecord] = {}
        for filename, raw_value in assets.items():
            raw = _require_mapping(raw_value, field=f"assets.{filename}")
            publication = _require_mapping(
                raw.get("publication"),
                field=f"assets.{filename}.publication",
            )
            upstream = _require_mapping(
                raw.get("upstream"),
                field=f"assets.{filename}.upstream",
            )
            license_record = _require_mapping(
                raw.get("license"),
                field=f"assets.{filename}.license",
            )
            records[filename] = HankelAssetRecord(
                filename=str(raw["filename"]),
                sha256=str(raw["sha256"]),
                filter_family=str(raw["filter_family"]),
                coefficient_count=int(raw["coefficient_count"]),
                coefficient_keys=tuple(str(key) for key in raw["coefficient_keys"]),
                publication_author=str(publication["author"]),
                publication_title=str(publication["title"]),
                publication_journal=str(publication["journal"]),
                publication_year=int(publication["year"]),
                publication_doi=str(publication["doi"]),
                upstream_project=str(upstream["project"]),
                upstream_release=str(upstream["release"]),
                upstream_commit=str(upstream["commit"]),
                upstream_path=str(upstream["path"]),
                license_identifier=str(license_record["identifier"]),
                attribution_text=str(license_record["attribution"]),
                acquisition_date=str(raw["acquisition_date"]),
                transformation=str(raw["transformation"]),
                verification_command=tuple(str(token) for token in raw["verification_command"]),
            )
        return records
    except EMContractError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _artifact_invalid(
            "Hankel provenance manifest is malformed",
            asset="provenance.json",
            expected="complete geobrain.em-hankel-provenance/1.0 manifest",
            actual=type(exc).__name__,
        ) from exc


def verify_hankel_asset(name: str, payload: bytes) -> HankelAssetRecord:
    """Verify one packaged filter payload and return its immutable provenance."""
    record = _provenance_records().get(name)
    if record is None:
        raise _artifact_invalid(
            "Unknown Hankel filter asset",
            asset=name,
            expected=sorted(_provenance_records()),
            actual=name,
        )
    if not isinstance(payload, bytes):
        raise _artifact_invalid(
            "Hankel filter payload must be bytes",
            asset=name,
            expected="bytes",
            actual=type(payload).__name__,
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record.sha256:
        raise _artifact_invalid(
            "Hankel filter checksum does not match provenance",
            asset=name,
            expected=record.sha256,
            actual=digest,
        )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            missing = sorted(set(record.coefficient_keys) - set(archive.files))
            if missing:
                raise ValueError(f"missing coefficient keys: {missing}")
            for key in record.coefficient_keys:
                values = np.asarray(archive[key])
                if values.shape != (record.coefficient_count,):
                    raise ValueError(
                        f"{key} shape {values.shape} is not ({record.coefficient_count},)"
                    )
                if values.dtype != np.dtype(np.float64):
                    raise ValueError(f"{key} dtype {values.dtype} is not float64")
                if not bool(np.isfinite(values).all()):
                    raise ValueError(f"{key} contains non-finite coefficients")
            base = np.asarray(archive[record.coefficient_keys[0]])
            if not bool((base > 0.0).all()) or not bool((np.diff(base) > 0.0).all()):
                raise ValueError("filter base must be finite, positive, and increasing")
    except (OSError, TypeError, ValueError) as exc:
        raise _artifact_invalid(
            "Hankel filter archive does not satisfy its coefficient contract",
            asset=name,
            expected={
                "keys": list(record.coefficient_keys),
                "shape": [record.coefficient_count],
                "dtype": "float64",
            },
            actual=type(exc).__name__,
        ) from exc
    return record


def _load_npz_as_torch(
    filename: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    resource = importlib.resources.files(_PACKAGE) / filename
    payload = resource.read_bytes()
    record = verify_hankel_asset(filename, payload)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {
            key: torch.from_numpy(np.ascontiguousarray(archive[key])).to(
                device=device,
                dtype=dtype,
            )
            for key in record.coefficient_keys
        }


def _filter_device(
    device: torch.device | str | None,
    *,
    object_name: str,
) -> torch.device:
    try:
        resolved = torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError) as exc:
        raise EMContractError(
            "DLF device must be a valid torch device",
            details={"device": str(device)},
            object_name=object_name,
            field="device",
            expected="valid CPU or CUDA torch device",
            actual=str(device),
        ) from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise EMCapabilityError(
            "CUDA filter coefficients requested but CUDA is unavailable",
            details={"device": str(resolved), "available": False},
            object_name=object_name,
            field="device",
            expected="available cpu or cuda device",
            actual=str(resolved),
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    if resolved.type not in {"cpu", "cuda"}:
        raise EMCapabilityError(
            "DLF coefficients support only CPU and CUDA devices",
            details={"device": str(resolved), "supported": ["cpu", "cuda"]},
            object_name=object_name,
            field="device",
            expected=["cpu", "cuda"],
            actual=str(resolved),
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    return resolved


def _filter_dtype(dtype: object, *, object_name: str) -> torch.dtype:
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise EMCapabilityError(
            "DLF coefficients require float32 or float64",
            details={
                "dtype": str(dtype),
                "supported": ["torch.float32", "torch.float64"],
            },
            object_name=object_name,
            field="dtype",
            expected=["torch.float32", "torch.float64"],
            actual=str(dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )
    return cast(torch.dtype, dtype)


def _filter_name(value: object, *, expected: str, object_name: str) -> str:
    """Validate a public filter identifier before it reaches the LRU cache."""
    if not isinstance(value, str) or value != expected:
        raise EMContractError(
            f"Unknown DLF filter {value!r}",
            details={"name": str(value), "supported": [expected]},
            object_name=object_name,
            field="name",
            expected=expected,
            actual=str(value),
        )
    return value


@lru_cache(maxsize=16)
def _load_hankel_filter_cached(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if name != "key_201_2012":
        raise EMContractError(
            f"Unknown Hankel filter {name!r}",
            details={"name": name, "supported": ["key_201_2012"]},
            object_name="_load_hankel_filter_cached",
            field="name",
            expected="key_201_2012",
            actual=name,
        )
    return _load_npz_as_torch(
        "dlf_key_201_2012.npz",
        dtype=dtype,
        device=device,
    )


def load_hankel_filter(
    name: str = "key_201_2012",
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Return the Hankel J0/J1 filter coefficients for ``name``.

    The npz read is cached per ``name``; each call returns a fresh **copy** of
    the coefficient tensors so a caller's in-place op (``mul_``, ``*=``) or
    device move can never corrupt the shared cache for other callers.
    """
    resolved_name = _filter_name(
        name,
        expected="key_201_2012",
        object_name="load_hankel_filter",
    )
    resolved_device = _filter_device(device, object_name="load_hankel_filter")
    resolved_dtype = _filter_dtype(dtype, object_name="load_hankel_filter")
    cached = _load_hankel_filter_cached(
        resolved_name,
        resolved_dtype,
        resolved_device,
    )
    return {key: value.clone() for key, value in cached.items()}


@lru_cache(maxsize=16)
def _load_sincos_filter_cached(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if name != "sincos_201":
        raise EMContractError(
            f"Unknown sin/cos filter {name!r}",
            details={"name": name, "supported": ["sincos_201"]},
            object_name="_load_sincos_filter_cached",
            field="name",
            expected="sincos_201",
            actual=name,
        )
    return _load_npz_as_torch(
        "dlf_sincos_201.npz",
        dtype=dtype,
        device=device,
    )


def load_sincos_filter(
    name: str = "sincos_201",
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Return the Sin/Cos DLF filter coefficients for ``name``.

    The npz read is cached per ``name``; each call returns a fresh **copy** of
    the coefficient tensors so a caller's in-place op (``mul_``, ``*=``) or
    device move can never corrupt the shared cache for other callers.
    """
    resolved_name = _filter_name(
        name,
        expected="sincos_201",
        object_name="load_sincos_filter",
    )
    resolved_device = _filter_device(device, object_name="load_sincos_filter")
    resolved_dtype = _filter_dtype(dtype, object_name="load_sincos_filter")
    cached = _load_sincos_filter_cached(
        resolved_name,
        resolved_dtype,
        resolved_device,
    )
    return {key: value.clone() for key, value in cached.items()}


def _log_trapezoid_hankel_j0(
    integrand: Callable[[torch.Tensor], torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
    relative_tolerance: float,
    max_refinements: int,
    radius: torch.Tensor | None,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate an axial or finite-radius J0 transform on a log grid."""
    label = "Axial" if radius is None else "Direct finite-radius"
    if not callable(integrand):
        raise EMContractError(
            f"{label} Hankel integrand must be callable",
            details={"integrand_type": type(integrand).__name__},
            object_name=object_name,
            field="integrand",
            expected="callable",
            actual=type(integrand).__name__,
        )
    resolved_device = _filter_device(device, object_name=object_name)
    resolved_dtype = _filter_dtype(dtype, object_name=object_name)
    if radius is not None and (
        radius.ndim != 0
        or radius.dtype != resolved_dtype
        or radius.device != resolved_device
        or not bool(torch.isfinite(radius))
        or not bool(radius > 0.0)
    ):
        raise EMContractError(
            "Direct Hankel radius must be one finite positive scalar on the transform device",
            details={
                "shape": list(radius.shape),
                "dtype": str(radius.dtype),
                "device": str(radius.device),
            },
            object_name=object_name,
            field="radius",
            expected={
                "shape": [],
                "dtype": str(resolved_dtype),
                "device": str(resolved_device),
                "domain": "> 0",
            },
            actual={
                "shape": list(radius.shape),
                "dtype": str(radius.dtype),
                "device": str(radius.device),
            },
        )
    if (
        not isinstance(relative_tolerance, float)
        or not math.isfinite(relative_tolerance)
        or relative_tolerance <= 0.0
        or relative_tolerance >= 1.0
    ):
        raise EMContractError(
            f"{label} Hankel relative tolerance must be finite and in (0, 1)",
            details={"relative_tolerance": relative_tolerance},
            object_name=object_name,
            field="relative_tolerance",
            expected="finite float in (0, 1)",
            actual=relative_tolerance,
        )
    if (
        type(max_refinements) is not int
        or max_refinements < 2
        or max_refinements > _AXIAL_MAX_REFINEMENTS
    ):
        raise EMContractError(
            f"{label} Hankel refinement count is outside the supported range",
            details={"max_refinements": max_refinements},
            object_name=object_name,
            field="max_refinements",
            expected=f"integer in [2, {_AXIAL_MAX_REFINEMENTS}]",
            actual=max_refinements,
        )

    previous: torch.Tensor | None = None
    last_relative_change: float | str | None = None
    for refinement in range(max_refinements):
        half_span = 12.0 + 4.0 * refinement
        intervals = 256 * (2**refinement)
        log_lam = torch.linspace(
            -half_span,
            half_span,
            intervals + 1,
            dtype=resolved_dtype,
            device=resolved_device,
        )
        lam = torch.exp(log_lam)
        values = integrand(lam)
        if not isinstance(values, torch.Tensor):
            raise EMContractError(
                f"{label} Hankel integrand must return a tensor",
                details={"value_type": type(values).__name__},
                object_name=object_name,
                field="integrand",
                expected="torch.Tensor",
                actual=type(values).__name__,
            )
        if values.is_nested or values.layout != torch.strided:
            raise EMContractError(
                f"{label} Hankel integrand output must be a dense strided tensor",
                details={"is_nested": values.is_nested, "layout": str(values.layout)},
                object_name=object_name,
                field="integrand",
                expected="dense strided torch.Tensor",
                actual="nested tensor" if values.is_nested else str(values.layout),
            )
        if values.device != resolved_device:
            raise EMContractError(
                f"{label} Hankel integrand output must remain on the lambda device",
                details={
                    "expected_device": str(resolved_device),
                    "actual_device": str(values.device),
                },
                object_name=object_name,
                field="integrand",
                expected=str(resolved_device),
                actual=str(values.device),
            )
        matching_complex_dtype = {
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }[resolved_dtype]
        if values.dtype not in {resolved_dtype, matching_complex_dtype}:
            raise EMContractError(
                f"{label} Hankel integrand output dtype must match the lambda precision",
                details={
                    "lambda_dtype": str(resolved_dtype),
                    "value_dtype": str(values.dtype),
                    "supported": [str(resolved_dtype), str(matching_complex_dtype)],
                },
                object_name=object_name,
                field="integrand",
                expected=[str(resolved_dtype), str(matching_complex_dtype)],
                actual=str(values.dtype),
            )
        if values.numel() == 0 or values.ndim == 0 or values.shape[-1] != lam.numel():
            raise EMContractError(
                f"{label} Hankel integrand must preserve the lambda axis",
                details={"lambda_shape": list(lam.shape), "value_shape": list(values.shape)},
                object_name=object_name,
                field="integrand",
                expected=f"last axis length {lam.numel()}",
                actual=list(values.shape),
            )
        if radius is not None:
            argument = lam * radius
            values = values * _torch_bessel_j0(argument)
        current = torch.trapezoid(values * lam, log_lam, dim=-1)
        if not bool(torch.isfinite(current).all()):
            raise EMNumericsError(
                f"{label} Hankel quadrature produced a non-finite estimate",
                details={
                    "operator": object_name,
                    "stage": "quadrature",
                    "backend": "torch_log_trapezoid",
                    "dtype": str(resolved_dtype),
                    "device": str(resolved_device),
                    "tolerance": relative_tolerance,
                    "iteration": refinement + 1,
                },
                object_name=object_name,
                field="integrand",
            )
        if previous is not None:
            denominator = torch.clamp(
                torch.abs(current),
                min=torch.finfo(resolved_dtype).tiny,
            )
            relative_change = torch.abs(current - previous) / denominator
            relative_change_max = torch.max(relative_change)
            relative_change_value = float(relative_change_max.detach().cpu())
            last_relative_change = (
                relative_change_value
                if math.isfinite(relative_change_value)
                else str(relative_change_value)
            )
            if (
                refinement == max_refinements - 1
                and math.isfinite(relative_change_value)
                and relative_change_value <= relative_tolerance
            ):
                return current, relative_change
        previous = current

    raise EMNumericsError(
        f"{label} Hankel quadrature did not converge",
        details={
            "operator": object_name,
            "stage": "convergence",
            "backend": "torch_log_trapezoid",
            "dtype": str(resolved_dtype),
            "device": str(resolved_device),
            "tolerance": relative_tolerance,
            "iteration_count": max_refinements,
            "last_relative_change": last_relative_change,
        },
        object_name=object_name,
        field="max_refinements",
    )


def axial_hankel_j0(
    integrand: Callable[[torch.Tensor], torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
    relative_tolerance: float,
    max_refinements: int,
) -> torch.Tensor:
    """Evaluate ``integral_0^inf integrand(lambda) dlambda`` on a log grid."""
    value, _ = _log_trapezoid_hankel_j0(
        integrand,
        dtype=dtype,
        device=device,
        relative_tolerance=relative_tolerance,
        max_refinements=max_refinements,
        radius=None,
        object_name="axial_hankel_j0",
    )
    return value


def _direct_hankel_j0_with_error(
    integrand: Callable[[torch.Tensor], torch.Tensor],
    radius: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    relative_tolerance: float,
    max_refinements: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite-radius J0 quadrature and its final relative-change estimate."""
    return _log_trapezoid_hankel_j0(
        integrand,
        dtype=dtype,
        device=device,
        relative_tolerance=relative_tolerance,
        max_refinements=max_refinements,
        radius=radius,
        object_name="direct_hankel_j0",
    )


def _validate_dlf_argument(value: object, *, object_name: str, field: str) -> torch.Tensor:
    """Return a dense, non-empty, finite positive real DLF coordinate tensor."""
    if not isinstance(value, torch.Tensor):
        raise EMContractError(
            "DLF transform coordinate must be a tensor",
            details={"value_type": type(value).__name__},
            object_name=object_name,
            field=field,
            expected="torch.Tensor",
            actual=type(value).__name__,
        )
    if value.is_nested or value.layout != torch.strided:
        raise EMContractError(
            "DLF transform coordinate must be a dense strided tensor",
            details={"is_nested": value.is_nested, "layout": str(value.layout)},
            object_name=object_name,
            field=field,
            expected="dense strided torch.Tensor",
            actual="nested tensor" if value.is_nested else str(value.layout),
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise EMContractError(
            "DLF transform coordinate must use float32 or float64",
            details={"dtype": str(value.dtype)},
            object_name=object_name,
            field=field,
            expected=["torch.float32", "torch.float64"],
            actual=str(value.dtype),
        )
    if value.device.type not in {"cpu", "cuda"}:
        raise EMCapabilityError(
            "DLF transforms support only CPU and CUDA tensors",
            details={"device": str(value.device)},
            object_name=object_name,
            field=field,
            expected=["cpu", "cuda"],
            actual=str(value.device),
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    if value.numel() == 0 or not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise EMContractError(
            "DLF transform coordinate must contain finite positive values",
            details={"shape": list(value.shape)},
            object_name=object_name,
            field=field,
            expected="non-empty finite values > 0",
            actual="invalid tensor values",
        )
    return value


def _validate_dlf_output(
    values: object,
    *,
    samples: torch.Tensor,
    object_name: str,
) -> torch.Tensor:
    """Validate the public integrand protocol before tensor operations."""
    if not isinstance(values, torch.Tensor):
        raise EMContractError(
            "DLF integrand must return a tensor",
            details={"value_type": type(values).__name__},
            object_name=object_name,
            field="integrand_fn",
            expected="torch.Tensor",
            actual=type(values).__name__,
        )
    if values.is_nested or values.layout != torch.strided:
        raise EMContractError(
            "DLF integrand output must be a dense strided tensor",
            details={"is_nested": values.is_nested, "layout": str(values.layout)},
            object_name=object_name,
            field="integrand_fn",
            expected="dense strided torch.Tensor",
            actual="nested tensor" if values.is_nested else str(values.layout),
        )
    matching_complex_dtype = {
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[samples.dtype]
    if values.device != samples.device or values.dtype not in {
        samples.dtype,
        matching_complex_dtype,
    }:
        raise EMContractError(
            "DLF integrand output must preserve sample precision and device",
            details={
                "sample_dtype": str(samples.dtype),
                "value_dtype": str(values.dtype),
                "sample_device": str(samples.device),
                "value_device": str(values.device),
            },
            object_name=object_name,
            field="integrand_fn",
            expected={
                "dtype": [str(samples.dtype), str(matching_complex_dtype)],
                "device": str(samples.device),
            },
            actual={"dtype": str(values.dtype), "device": str(values.device)},
        )
    if values.numel() == 0 or values.ndim == 0 or values.shape[-1] != samples.shape[-1]:
        raise EMContractError(
            "DLF integrand must preserve the filter sample axis",
            details={
                "sample_shape": list(samples.shape),
                "value_shape": list(values.shape),
            },
            object_name=object_name,
            field="integrand_fn",
            expected=f"last axis length {samples.shape[-1]}",
            actual=list(values.shape),
        )
    if not bool(torch.isfinite(values).all()):
        raise EMNumericsError(
            "DLF integrand produced non-finite samples",
            details={
                "operator": object_name,
                "stage": "integrand",
                "value_shape": list(values.shape),
            },
            object_name=object_name,
            field="integrand_fn",
        )
    value_axes = values.shape[:-1]
    sample_axes = samples.shape[:-1]
    broadcastable = all(
        left == right or left == 1 or right == 1
        for left, right in zip(reversed(value_axes), reversed(sample_axes))
    )
    if not broadcastable:
        raise EMContractError(
            "DLF integrand leading axes must broadcast with transform coordinates",
            details={
                "sample_shape": list(samples.shape),
                "value_shape": list(values.shape),
            },
            object_name=object_name,
            field="integrand_fn",
            expected=f"leading axes broadcastable with {list(samples.shape[:-1])}",
            actual=list(values.shape[:-1]),
        )
    return values


def _validate_dlf_samples(samples: torch.Tensor, *, object_name: str) -> None:
    """Reject an unrepresentable transformed sample grid before integration."""
    if not bool(torch.isfinite(samples).all()):
        raise EMNumericsError(
            "DLF filter samples are not representable at this coordinate scale",
            details={
                "operator": object_name,
                "stage": "filter_samples",
                "sample_shape": list(samples.shape),
                "dtype": str(samples.dtype),
            },
            object_name=object_name,
            field="r" if object_name == "dlf_hankel" else "t",
        )


def _validate_dlf_result(result: torch.Tensor, *, object_name: str) -> torch.Tensor:
    """Return a finite DLF result or raise the EM numerical boundary error."""
    if not bool(torch.isfinite(result).all()):
        raise EMNumericsError(
            "DLF transform produced a non-finite result",
            details={"operator": object_name, "stage": "transform_output"},
            object_name=object_name,
            field="output",
        )
    return result


def dlf_hankel(
    integrand_fn: Callable[[torch.Tensor], torch.Tensor],
    r: torch.Tensor,
    *,
    kind: str,
    filter_name: str = "key_201_2012",
) -> torch.Tensor:
    """
    Compute ∫₀^∞ integrand(λ) Jₙ(λ r) dλ via digital linear filter.

    Args:
        integrand_fn: callable mapping a λ-tensor of shape ``(..., n_filter)``
            to a tensor with the same shape (the integrand sampled at every λ).
            Must be autograd-traceable (torch ops only) for gradients to flow.
        r: positive offsets, shape ``(...,)``. ``float64`` recommended;
            lower precision will cap accuracy.
        kind: ``"j0"`` or ``"j1"``.
        filter_name: filter set; only ``"key_201_2012"`` ships in E2.

    Returns:
        Tensor of shape ``r.shape`` (or broadcast result if the integrand
        function broadcasts additional leading axes).
    """
    if not callable(integrand_fn):
        raise EMContractError(
            "dlf_hankel integrand_fn must be callable",
            details={"integrand_type": type(integrand_fn).__name__},
            object_name="dlf_hankel",
            field="integrand_fn",
            expected="callable",
            actual=type(integrand_fn).__name__,
        )
    r = _validate_dlf_argument(r, object_name="dlf_hankel", field="r")
    if kind not in ("j0", "j1"):
        raise EMContractError(
            f"dlf_hankel kind must be 'j0' or 'j1', got {kind!r}",
            details={"kind": kind, "supported": ["j0", "j1"]},
            object_name="dlf_hankel",
            field="kind",
            expected="'j0' or 'j1'",
            actual=kind,
        )
    f = load_hankel_filter(filter_name, dtype=r.dtype, device=r.device)
    base = f["base"]
    weights = f[kind]

    r_ = r.unsqueeze(-1)  # (..., 1)
    lam = base / r_  # broadcast → (..., n_filter)
    _validate_dlf_samples(lam, object_name="dlf_hankel")

    values = _validate_dlf_output(
        integrand_fn(lam),
        samples=lam,
        object_name="dlf_hankel",
    )

    return _validate_dlf_result(
        (values * weights).sum(dim=-1) / r,
        object_name="dlf_hankel",
    )


def dlf_sincos(
    integrand_fn: Callable[[torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    *,
    kind: str,
    filter_name: str = "sincos_201",
) -> torch.Tensor:
    """
    Compute ∫₀^∞ integrand(ω) cos(ωt) dω (or sin) via digital linear filter.

    Args:
        integrand_fn: callable mapping a 1-D ω tensor (shape ``(..., n_filter)``)
            to its sampled integrand. Must be autograd-traceable.
        t: positive time samples (or general kernel argument), shape ``(...,)``.
            ``float64`` recommended.
        kind: ``"sin"`` or ``"cos"``.
        filter_name: filter set; only ``"sincos_201"`` ships in E2.

    Returns:
        Tensor of shape ``t.shape`` (or broadcast result).
    """
    if not callable(integrand_fn):
        raise EMContractError(
            "dlf_sincos integrand_fn must be callable",
            details={"integrand_type": type(integrand_fn).__name__},
            object_name="dlf_sincos",
            field="integrand_fn",
            expected="callable",
            actual=type(integrand_fn).__name__,
        )
    t = _validate_dlf_argument(t, object_name="dlf_sincos", field="t")
    if kind not in ("sin", "cos"):
        raise EMContractError(
            f"dlf_sincos kind must be 'sin' or 'cos', got {kind!r}",
            details={"kind": kind, "supported": ["sin", "cos"]},
            object_name="dlf_sincos",
            field="kind",
            expected="'sin' or 'cos'",
            actual=kind,
        )
    f = load_sincos_filter(filter_name, dtype=t.dtype, device=t.device)
    base = f["base"]
    weights = f[kind]

    t_ = t.unsqueeze(-1)
    om = base / t_
    _validate_dlf_samples(om, object_name="dlf_sincos")
    values = _validate_dlf_output(
        integrand_fn(om),
        samples=om,
        object_name="dlf_sincos",
    )

    return _validate_dlf_result(
        (values * weights).sum(dim=-1) / t,
        object_name="dlf_sincos",
    )
