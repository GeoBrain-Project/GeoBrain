"""
Tiled matrix-free potential-field forward.

The prism forward is linear in the physical property: ``d = G(geometry) · m`` with a
kernel matrix ``G`` that depends only on geometry. The dense implementation materializes
``G`` (transiently) and lets autograd retain it plus the eight-corner temporaries for the
backward; the measured peak is ≈10× the size of ``G`` (21 GB at 100k cells × 2.5k obs,
fp64), which makes survey-scale problems (1M × 10k → 80 GB for ``G`` alone) impossible.

This module evaluates ``G`` in (obs × cell) TILES with an ANALYTIC transpose:

    forward:   d[i-tile]  += K(i-tile, j-tile) @ m[j-tile]
    backward:  m̄[j-tile] += K(i-tile, j-tile)ᵀ @ d̄[i-tile]     (blocks recomputed)

wrapped in a ``torch.autograd.Function`` so nothing is retained between tiles, peak
memory is O(tile) regardless of problem size, and gradients are exact (same block math;
only the floating-point summation order differs from the dense path).

The tile evaluator reuses the EXACT dense kernel (`prism._gz_kernel`, Nagy 2000 with its
``_NAGY_EPS`` guards) on slices, so the physics is byte-identical per block.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .helpers import G_SI, M_TO_MGAL
from .prism import _gz_kernel

BlockFn = Callable[[int, int, int, int], Tensor]
"""``block_fn(i0, i1, j0, j1) -> (i1-i0, j1-j0)`` kernel block (unit property)."""


def _auto_tiles(n_obs: int, n_cells: int, itemsize: int,
                budget_bytes: int) -> tuple[int, int]:
    """Pick (tile_obs, tile_cells) so one block's working set fits the budget.

    The eight-corner Nagy evaluation holds ≈8 live (m×n) temporaries per block;
    we size m×n ≤ budget/(8·itemsize), preferring wide cell tiles (contiguous
    property reads) and clamping to the problem size."""
    max_elems = max(budget_bytes // (8 * itemsize), 1024)
    tile_cells = min(n_cells, 65536)
    tile_obs = max(1, min(n_obs, max_elems // tile_cells))
    return tile_obs, tile_cells


class _TiledLinearFn(torch.autograd.Function):
    """Differentiable ``scale · (G @ m)`` for a tile-evaluable kernel ``G``.

    ``block_fn`` is stashed on the ctx (geometry tensors live in its closure and
    require no grad); only ``m`` is differentiable. First-order only."""

    @staticmethod
    def forward(ctx, m: Tensor, block_fn: BlockFn, n_obs: int,
                tile_obs: int, tile_cells: int, scale: float) -> Tensor:
        n_cells = m.shape[0]
        out = torch.zeros(n_obs, dtype=m.dtype, device=m.device)
        with torch.no_grad():
            for i0 in range(0, n_obs, tile_obs):
                i1 = min(i0 + tile_obs, n_obs)
                acc = torch.zeros(i1 - i0, dtype=m.dtype, device=m.device)
                for j0 in range(0, n_cells, tile_cells):
                    j1 = min(j0 + tile_cells, n_cells)
                    acc += block_fn(i0, i1, j0, j1) @ m[j0:j1]
                out[i0:i1] = acc
        ctx.block_fn = block_fn
        ctx.dims = (n_obs, n_cells, tile_obs, tile_cells, scale)
        return out * scale

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        block_fn = ctx.block_fn
        n_obs, n_cells, tile_obs, tile_cells, scale = ctx.dims
        g = (grad_out * scale).contiguous()
        grad_m = torch.zeros(n_cells, dtype=g.dtype, device=g.device)
        with torch.no_grad():
            for j0 in range(0, n_cells, tile_cells):
                j1 = min(j0 + tile_cells, n_cells)
                acc = torch.zeros(j1 - j0, dtype=g.dtype, device=g.device)
                for i0 in range(0, n_obs, tile_obs):
                    i1 = min(i0 + tile_obs, n_obs)
                    acc += block_fn(i0, i1, j0, j1).transpose(0, 1) @ g[i0:i1]
                grad_m[j0:j1] = acc
        return grad_m, None, None, None, None, None


def tiled_prism_linear(
    m: Tensor, block_fn: BlockFn, n_obs: int, *,
    tile_obs: int, tile_cells: int, scale: float = 1.0,
) -> Tensor:
    """Differentiable tiled ``scale · (G @ m)`` (see :class:`_TiledLinearFn`)."""
    return _TiledLinearFn.apply(m, block_fn, n_obs, tile_obs, tile_cells, scale)


def gravity_prism_gz_tiled(
    obs_x: Tensor, obs_y: Tensor, obs_z: Tensor,
    x1: Tensor, x2: Tensor, y1: Tensor, y2: Tensor, z1: Tensor, z2: Tensor,
    density: Tensor,
    *,
    tile_obs: Optional[int] = None,
    tile_cells: Optional[int] = None,
    memory_budget_bytes: int = 1 << 30,
) -> Tensor:
    """Matrix-free twin of :func:`prism.gravity_prism_gz` (same args, same units).

    Never materializes the (n_obs × n_cells) kernel; peak memory is O(tile),
    sized from ``memory_budget_bytes`` (default 1 GiB) unless explicit tile
    sizes are given. Differentiable w.r.t. ``density`` via the analytic
    transpose. Results match the dense kernel to fp roundoff (identical block
    math; tiled summation order).
    """
    n_obs = obs_x.shape[0]
    n_cells = density.shape[0]
    if tile_obs is None or tile_cells is None:
        to, tc = _auto_tiles(n_obs, n_cells, density.element_size(),
                             memory_budget_bytes)
        tile_obs = to if tile_obs is None else tile_obs
        tile_cells = tile_cells or tc

    def block_fn(i0: int, i1: int, j0: int, j1: int) -> Tensor:
        return _gz_kernel(
            obs_x[i0:i1].unsqueeze(1), obs_y[i0:i1].unsqueeze(1),
            obs_z[i0:i1].unsqueeze(1),
            x1[j0:j1].unsqueeze(0), x2[j0:j1].unsqueeze(0),
            y1[j0:j1].unsqueeze(0), y2[j0:j1].unsqueeze(0),
            z1[j0:j1].unsqueeze(0), z2[j0:j1].unsqueeze(0),
        )

    return tiled_prism_linear(
        density, block_fn, n_obs,
        tile_obs=tile_obs, tile_cells=tile_cells, scale=G_SI * M_TO_MGAL,
    )


def dense_gz_bytes_estimate(n_obs: int, n_cells: int, itemsize: int) -> int:
    """Estimated peak bytes of the DENSE gz path with autograd (measured ≈10×|G|)."""
    return 10 * n_obs * n_cells * itemsize


DENSE_GZ_MEMORY_LIMIT_BYTES = 4 << 30
"""Shared dense-gz peak-memory ceiling used by auto/store preflight guards."""
