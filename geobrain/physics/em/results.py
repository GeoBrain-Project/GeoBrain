"""Canonical EM result component names and complex result containers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class FieldComponent(str, Enum):
    """Physical EM field component used by receiver declarations."""

    EX = "ex"
    EY = "ey"
    EZ = "ez"
    HX = "hx"
    HY = "hy"
    HZ = "hz"
    BX = "bx"
    BY = "by"
    BZ = "bz"
    JX = "Jx"
    JY = "Jy"
    JZ = "Jz"
    V = "V"
    APPARENT_RESISTIVITY = "apparent_resistivity"
    PHASE = "phase"
    IMPEDANCE = "impedance"


@dataclass(frozen=True, slots=True)
class ComplexData:
    """Real/imaginary tensor storage with an explicit native-complex view.

    Attributes:
        real / imag: the split real and imaginary parts.
    """

    real: torch.Tensor
    imag: torch.Tensor

    def to_complex(self) -> torch.Tensor:
        """Materialize the native-complex tensor without changing placement."""
        return torch.complex(self.real, self.imag)

    @classmethod
    def from_complex(cls, value: torch.Tensor) -> "ComplexData":
        """Split a native-complex tensor into contiguous real-valued parts."""
        return cls(real=value.real.contiguous(), imag=value.imag.contiguous())


__all__ = ["ComplexData", "FieldComponent"]
