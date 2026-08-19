"""Immutable discovery contracts for reflectivity and convolution operators.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ..capabilities import WaveCapabilityReport, WaveUnsupportedCombination

MAX_ABS_CONTRAST = 0.15
MAX_ANGLE_DEG = 30.0
MAX_ABS_ERROR = 0.029
CONTRAST_MATRIX = (-0.15, 0.0, 0.15)
ANGLE_MATRIX_DEG = tuple(float(value) for value in range(0, 31, 5))


class _FrozenJSONList(list[object]):
    """JSON-encoder-compatible list that rejects mutation."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("GeoBrain schema is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __iadd__ = _immutable  # type: ignore[assignment]
    __imul__ = _immutable  # type: ignore[assignment]


class _FrozenJSONDict(dict[str, object]):
    """JSON-encoder-compatible dict that rejects mutation."""

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("GeoBrain schema is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable  # type: ignore[assignment]


def _freeze_json(value: object) -> object:
    """Recursively freeze a JSON tree without breaking ``json.dumps``."""
    if isinstance(value, Mapping):
        return _FrozenJSONDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _FrozenJSONList([_freeze_json(item) for item in value])
    return value


def capability_report(name: str) -> WaveCapabilityReport:
    """Return the standard Wave report for one validated 1-D operator."""
    is_convolution = name in {"Convolutional1D", "ConvolutionalAVO"}
    is_approximation = name in {"AkiRichards", "Shuey", "ConvolutionalAVO"}
    model_fields: tuple[tuple[str, str], ...] = (("vp", "m/s"), ("rho", "kg/m^3"))
    differentiable_fields: tuple[str, ...] = ("vp", "rho")
    if name != "Convolutional1D":
        model_fields = (("vp", "m/s"), ("vs", "m/s"), ("rho", "kg/m^3"))
        differentiable_fields = ("vp", "vs", "rho")

    unsupported: list[WaveUnsupportedCombination] = []
    if is_approximation:
        unsupported.extend(
            (
                WaveUnsupportedCombination(
                    selection=(("model.max_abs_contrast", ">0.15"),),
                    reason="the measured 0.029 maximum absolute error is not validated beyond contrast 0.15",
                    remediation="use Zoeppritz or report the result as outside the validated approximation domain",
                ),
                WaveUnsupportedCombination(
                    selection=(("angles_deg", ">30"),),
                    reason="the measured 0.029 maximum absolute error is validated only through 30 degrees",
                    remediation="use Zoeppritz or report the result as outside the validated approximation domain",
                ),
            )
        )

    equation = {
        "AkiRichards": "Aki-Richards three-term PP reflectivity",
        "Shuey": "Shuey PP reflectivity",
        "Zoeppritz": "exact isotropic Zoeppritz PP reflectivity",
        "Convolutional1D": "normal-incidence impedance reflectivity convolution",
        "ConvolutionalAVO": "Aki-Richards PP reflectivity convolution",
    }[name]
    return WaveCapabilityReport(
        physics="seismic-reflectivity" if not is_convolution else "convolutional-seismic",
        equation=equation,
        dimension=1,
        maturity="production",
        required_model_fields=model_fields,
        components=("trace",) if is_convolution else ("PP-reflectivity",),
        dtypes=("float32", "float64", "complex64", "complex128")
        if name == "Convolutional1D"
        else ("float32", "float64"),
        devices=("cpu", "cuda"),
        backends=("torch",),
        boundaries=("zero-same",) if is_convolution else (),
        memory_strategies=(),
        differentiable_model_fields=differentiable_fields,
        differentiable_wavelets=name == "Convolutional1D",
        mesh_capabilities=(),
        resource_estimate_supported=False,
        unsupported=tuple(unsupported),
    )


def input_schema(name: str) -> Mapping[str, object]:
    """Return an immutable, unit-aware Agent/UI schema for one operator."""
    report = capability_report(name)
    is_convolution = name in {"Convolutional1D", "ConvolutionalAVO"}
    is_approximation = name in {"AkiRichards", "Shuey", "ConvolutionalAVO"}
    real_dtypes = ["float32", "float64"]
    complex_capable_dtypes = ["float32", "float64", "complex64", "complex128"]
    model_dtypes = complex_capable_dtypes if name == "Convolutional1D" else real_dtypes
    output_dtypes = (
        complex_capable_dtypes
        if name in {"Zoeppritz", "Convolutional1D"}
        else real_dtypes
    )
    model: dict[str, object] = {
        "vp": {
            "unit": "m/s",
            "dtypes": model_dtypes,
            "axes": ["sample"],
            "exclusiveMinimum": 0.0,
        },
        "rho": {
            "unit": "kg/m^3",
            "dtypes": model_dtypes,
            "axes": ["sample"],
            "exclusiveMinimum": 0.0,
        },
    }
    if name != "Convolutional1D":
        model["vs"] = {
            "unit": "m/s",
            "dtypes": real_dtypes,
            "axes": ["sample"],
            "exclusiveMinimum": 0.0,
        }

    schema: dict[str, object] = {
        "title": f"GeoBrain {name}",
        "version": "0.2.0",
        "maturity": "production",
        "differentiable_wavelets": report.differentiable_wavelets,
        "model": model,
        "output": {
            "trace" if is_convolution else "reflectivity": {
                "unit": "1",
                "axes": ["sample", "angle"]
                if name == "ConvolutionalAVO"
                else (["sample"] if is_convolution else ["interface", "angle"]),
                "dtypes": output_dtypes,
            }
        },
        "unsupported": report.to_dict()["unsupported"],
    }
    if name != "Convolutional1D":
        schema["angles_deg"] = {
            "unit": "degree",
            "minimum": 0.0,
            "exclusiveMaximum": 90.0,
            "axes": ["angle"],
        }
    if name == "Zoeppritz":
        output = cast(dict[str, object], cast(dict[str, object], schema["output"])["reflectivity"])
        output["complex_precision"] = {
            "float32": "complex64",
            "float64": "complex128",
        }
    if is_convolution:
        schema["wavelet"] = {
            "unit": "relative-amplitude",
            "axes": ["sample"],
            "boundary": "zero-same",
            "dtypes": complex_capable_dtypes
            if name == "Convolutional1D"
            else real_dtypes,
        }
    if is_approximation:
        schema["approximation_validation"] = {
            "contrast_definition": "abs((lower-upper)/((lower+upper)/2))",
            "contrast_matrix": list(CONTRAST_MATRIX),
            "angle_matrix_deg": list(ANGLE_MATRIX_DEG),
            "max_abs_contrast": MAX_ABS_CONTRAST,
            "max_angle_deg": MAX_ANGLE_DEG,
            "metric": "maximum absolute PP reflectivity error",
            "max_abs_error": MAX_ABS_ERROR,
            "reference": "independent displacement/traction-continuity Zoeppritz solve",
        }
    return cast(Mapping[str, object], _freeze_json(schema))


__all__ = [
    "ANGLE_MATRIX_DEG",
    "CONTRAST_MATRIX",
    "MAX_ABS_CONTRAST",
    "MAX_ABS_ERROR",
    "MAX_ANGLE_DEG",
    "capability_report",
    "input_schema",
]
