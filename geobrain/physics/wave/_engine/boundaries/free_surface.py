"""Shared free-surface field constraints.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor


def zero_free_surface_row(field: Tensor, row: int) -> Tensor:
    """Return ``field`` with its free-surface z row set to zero out-of-place."""
    az = -(field.ndim - 2)  # z axis (−2 in 2-D)
    n = field.shape[az]
    mask = torch.ones(n, dtype=field.dtype, device=field.device)
    mask[row] = 0.0
    shape = [1] * field.ndim
    shape[az] = n
    return field * mask.reshape(shape)
