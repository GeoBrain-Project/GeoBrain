"""
Type aliases, constants, and tensor helpers for rock-physics models.

The module is import-
cheap on purpose: every rock model imports these helpers, so we don't
pull torch.nn / abc / typing machinery in here.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import List, Union

import torch
from torch import Tensor

TensorLike = Union[float, int, List[float], Tensor]

PI = 3.141592653589793
EPS = 1e-10

# ---------------------------------------------------------------------------
# Rock-physics defaults (used by config presets + granular models)
# ---------------------------------------------------------------------------

PHI_C = 0.36           # critical porosity (typical sandstones)
CN = 8.5               # coordination number (typical random close pack)
P = 20.0e6             # effective pressure (Pa), SI; the v05 seam
                        # (hm_moduli_v05 + SoftSand/StiffSand/HertzMindlin
                        # compute_dry_rock hooks) is Pa-native, and this is
                        # their default-pressure fallback.
F = 1.0                # Hertz-Mindlin shear-reduction (1=rough, 0=smooth)
BRIE_E = 3.0           # Brie exponent for patchy saturation
PHI_C_CEMENTED = 0.40  # critical porosity for cemented sands


def as_tensor(x: TensorLike, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Convert scalar / list / tensor to a torch tensor of at least 1-D.

    Scalars are promoted to 1-D so downstream broadcasting works without
    rank-mismatch surprises. ``dtype`` defaults to float32; pass an
    explicit dtype when you need double-precision math.
    """
    if isinstance(x, Tensor):
        return x.to(dtype)
    return torch.atleast_1d(torch.as_tensor(x, dtype=dtype))


def ensure_same_device(*tensors: Tensor):
    """
    Co-locate tensors on the first non-CPU device found, else leave alone.

    Useful when mixing scalar config params (CPU) with autograd-tracked
    inputs (potentially on CUDA): without this, ``a + b`` raises a
    device-mismatch error mid-pipeline.
    """
    device = None
    for t in tensors:
        if isinstance(t, Tensor) and t.device.type != "cpu":
            device = t.device
            break
    if device is None:
        return tensors
    return tuple(t.to(device) if isinstance(t, Tensor) else t for t in tensors)


__all__ = [
    "BRIE_E",
    "CN",
    "EPS",
    "F",
    "P",
    "PHI_C",
    "PHI_C_CEMENTED",
    "PI",
    "Tensor",
    "TensorLike",
    "as_tensor",
    "ensure_same_device",
]
