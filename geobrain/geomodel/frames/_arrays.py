"""
Internal numpy typing & coercion helpers for the geomodel package.

Kept private (``_arrays``) because the geomodel
sub-package is the only place that uses numpy directly,
the rest of the project must stay pure-torch.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union, cast

import numpy as np

AnyArray = np.ndarray[tuple[Any, ...], np.dtype[Any]]
BoolArray = np.ndarray[tuple[Any, ...], np.dtype[np.bool_]]
ComplexArray = np.ndarray[tuple[Any, ...], np.dtype[np.complex128]]
FloatArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[Any, ...], np.dtype[np.int64]]
Shape = Union[int, tuple[int, ...]]


def as_array(values: object) -> AnyArray:
    array: AnyArray = np.asarray(values)
    return array


def as_bool_array(values: object) -> BoolArray:
    return cast(BoolArray, np.asarray(values, dtype=np.bool_))


def as_complex_array(values: object) -> ComplexArray:
    return cast(ComplexArray, np.asarray(values, dtype=np.complex128))


def as_float_array(values: object) -> FloatArray:
    return cast(FloatArray, np.asarray(values, dtype=np.float64))


def as_int_array(values: object) -> IntArray:
    return cast(IntArray, np.asarray(values, dtype=np.int64))


def column_stack_float(values: Sequence[object]) -> FloatArray:
    arrays = tuple(np.asarray(value) for value in values)
    return cast(FloatArray, np.column_stack(arrays))


def empty_float(shape: Shape) -> FloatArray:
    return cast(FloatArray, np.empty(shape, dtype=np.float64))


def full_float(shape: Shape, fill_value: float) -> FloatArray:
    return cast(FloatArray, np.full(shape, fill_value, dtype=np.float64))


def ones_bool(shape: Shape) -> BoolArray:
    return cast(BoolArray, np.ones(shape, dtype=np.bool_))


def ones_float(shape: Shape) -> FloatArray:
    return cast(FloatArray, np.ones(shape, dtype=np.float64))


def ravel_float(values: object) -> FloatArray:
    return cast(FloatArray, np.asarray(values, dtype=np.float64).ravel())


def zeros_float(shape: Shape) -> FloatArray:
    return cast(FloatArray, np.zeros(shape, dtype=np.float64))


__all__ = [
    "AnyArray",
    "BoolArray",
    "ComplexArray",
    "FloatArray",
    "IntArray",
    "Shape",
    "as_array",
    "as_bool_array",
    "as_complex_array",
    "as_float_array",
    "as_int_array",
    "column_stack_float",
    "empty_float",
    "full_float",
    "ones_bool",
    "ones_float",
    "ravel_float",
    "zeros_float",
]
