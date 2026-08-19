"""Elastic CUDA execution wrappers for the Wave native backend.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

# The numerical ABI below is mechanically retained from the dynamic
# ``torch.autograd.Function`` CUDA wrapper. Local type suppressions identify
# only those PyTorch positional ABI boundaries; request/result APIs stay strict.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ....errors import WaveContractError
from ...contracts import PropagationRequest, PropagationResult
from .execution import (
    assemble_native_result,
    native_receiver_coordinates,
    prepare_native_execution,
)

def _require_wavelet_like(wavelet: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Validate native wavelet placement without moving or casting the live tensor."""
    if wavelet.dtype is not reference.dtype or wavelet.device != reference.device:
        raise WaveContractError(
            "native wavelet must match the prepared equation tensors",
            object_name="native",
            field="wavelet",
            expected=f"dtype={reference.dtype}, device={reference.device}",
            actual=f"dtype={wavelet.dtype}, device={wavelet.device}",
        )
    return wavelet

_NATIVE_HIST_BYTES_LIMIT = 2 << 30  # 2 GiB of p/v histories -> checkpointed
_FORCE_CKPT_SEGMENTS: int | None = None  # test hook: force a segment length


def _ckpt_segments(nt: int, n_state: int, n_hist: int) -> int:
    """Segment length minimizing checkpoint+segment-history memory:
    K* = sqrt(nt * n_state / n_hist), clamped to [8, nt]."""
    import math as _math

    return min(nt, max(8, int(_math.sqrt(nt * n_state / max(n_hist, 1)))))


class _NativeElastic2dCkptFn(torch.autograd.Function):  # type: ignore[misc]
    """Checkpointed twin of :class:`_NativeElastic2dFn` (bitwise gradients).
    vx/vz histories carry ``seg+1`` slots (slot 0 = segment-start state);
    the strain histories e1/e2/gsum carry ``seg`` slots (written at step it,
    consumed at step it, no segment-boundary slot)."""

    @staticmethod
    def forward(ctx_fn, c11, c13, c33, c44, buoyancy, wav, fd, profiles,  # type: ignore[no-untyped-def]
                src_z, src_x, rcv_z, rcv_x, B, nz, nx, dt, dx, dz, fs,
                n_comp, seg, extension):
        dev, dtp = c11.device, c11.dtype
        nt = wav.shape[1]
        n_rcv = rcv_z.shape[0]
        fields = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev)
                  for _ in range(13)]
        shape = (B, nt, n_rcv) if n_comp == 1 else (B, nt, n_rcv, n_comp)
        records = torch.empty(*shape, dtype=dtp, device=dev)
        cs = [c.detach().contiguous() for c in (c11, c13, c33, c44, buoyancy)]
        wv = wav.detach().contiguous()
        empty = torch.empty(0, dtype=dtp, device=dev)

        ckpts = []
        for i0 in range(0, nt, seg):
            i1 = min(i0 + seg, nt)
            ckpts.append([f.clone() for f in fields])
            k = i1 - i0
            rshape = (B, k, n_rcv) if n_comp == 1 else (B, k, n_rcv, n_comp)
            rec_seg = torch.empty(*rshape, dtype=dtp, device=dev)
            extension.forward(*fields, *cs, fd, *profiles,
                        src_z, src_x, wv[:, i0:i1].contiguous(),
                        rcv_z, rcv_x, rec_seg,
                        float(dt), float(dx), float(dz), bool(fs),
                        empty, empty, empty, empty, empty, n_comp)
            records[:, i0:i1] = rec_seg
        ctx_fn.save_for_backward(*cs, wv, fd, *profiles,
                                 src_z, src_x, rcv_z, rcv_x)
        ctx_fn.ckpts = ckpts
        ctx_fn.dims = (B, nz, nx, dt, dx, dz, fs, n_comp, seg, nt)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:5])
        return (records, *fields[:5])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        saved = ctx_fn.saved_tensors
        cs = saved[0:5]
        wv, fd = saved[5], saved[6]
        profiles = saved[7:19]
        src_z, src_x, rcv_z, rcv_x = saved[19:23]
        B, nz, nx, dt, dx, dz, fs, n_comp, seg, nt = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = cs[0].device, cs[0].dtype

        adj = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev)
               for _ in range(13)]
        scratch = [torch.empty(B, 1, nz, nx, dtype=dtp, device=dev)
                   for _ in range(4)]
        grads = [torch.zeros(nz, nx, dtype=dtp, device=dev) for _ in range(5)]
        gwav = torch.zeros_like(wv)
        v_hists = [torch.zeros(seg + 1, B, nz, nx, dtype=dtp, device=dev)
                   for _ in range(2)]
        e_hists = [torch.zeros(seg, B, nz, nx, dtype=dtp, device=dev)
                   for _ in range(3)]
        go = grad_records.contiguous()

        starts = list(range(0, nt, seg))
        for s in reversed(range(len(starts))):
            i0 = starts[s]
            i1 = min(i0 + seg, nt)
            k = i1 - i0
            fields = [c.clone() for c in ctx_fn.ckpts[s]]
            for h, f in zip(v_hists, fields[:2]):
                h[0].copy_(f[:, 0])
            k_hists = [h[: k + 1] for h in v_hists] + [h[:k] for h in e_hists]
            rshape = ((B, k, rcv_z.shape[0]) if n_comp == 1
                      else (B, k, rcv_z.shape[0], n_comp))
            rec_scratch = torch.empty(*rshape, dtype=dtp, device=dev)
            wav_seg = wv[:, i0:i1].contiguous()
            extension.forward(*fields, *cs, fd, *profiles,
                        src_z, src_x, wav_seg, rcv_z, rcv_x, rec_scratch,
                        float(dt), float(dx), float(dz), bool(fs),
                        *k_hists, n_comp)
            gwav_seg = torch.zeros(B, k, dtype=dtp, device=dev)
            extension.backward(go[:, i0:i1].contiguous(), *k_hists, *cs, fd,
                         *profiles, src_z, src_x, wav_seg, rcv_z, rcv_x,
                         *adj, *scratch, *grads, gwav_seg,
                         float(dt), float(dx), float(dz), bool(fs), n_comp)
            gwav[:, i0:i1] = gwav_seg
        return (*grads, gwav) + (None,) * 16

_EL_COEFFS = ("c11", "c13", "c33", "c44", "buoyancy")


def _cpml2d_profiles(
    cpml: Any,
    dtp: torch.dtype,
    dev: torch.device,
) -> tuple[torch.Tensor, ...]:
    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1).to(dtype=dtp, device=dev).contiguous()

    return (
        flat(cpml.bx_int), flat(cpml.ax_int), flat(cpml.kx_int),
        flat(cpml.bz_int), flat(cpml.az_int), flat(cpml.kz_int),
        flat(cpml.bx_half), flat(cpml.ax_half), flat(cpml.kx_half),
        flat(cpml.bz_half), flat(cpml.az_half), flat(cpml.kz_half),
    )


def run_elastic2d_forward(
    extension: Any,
    ctx: Any,
    state: Sequence[torch.Tensor],
    wav: torch.Tensor,
    n_comp: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Plain (no-grad) native elastic-2D run; mirrors ``_run_segment`` semantics."""
    with torch.no_grad():
        fields = [s.contiguous() for s in state]
        n_src, nt = wav.shape
        n_rcv = ctx.rcv_z.shape[0]
        p0 = fields[0]
        shape = (n_src, nt, n_rcv) if n_comp == 1 else (n_src, nt, n_rcv, n_comp)
        records = torch.empty(*shape, dtype=p0.dtype, device=p0.device)
        fd = torch.tensor(ctx.eq._coeffs, dtype=p0.dtype, device=p0.device)
        empty = torch.empty(0, dtype=p0.dtype, device=p0.device)
        extension.forward(
            *fields,
            *[ctx.coeffs[k].to(p0.dtype).contiguous() for k in _EL_COEFFS],
            fd, *_cpml2d_profiles(ctx.cpml, p0.dtype, p0.device),
            ctx.src_z.contiguous(), ctx.src_x.contiguous(),
            _require_wavelet_like(wav, p0).contiguous(),
            ctx.rcv_z.contiguous(), ctx.rcv_x.contiguous(), records,
            float(ctx.dt), float(ctx.dx), float(ctx.dz),
            bool(getattr(ctx.eq, "_free_surface", False)),
            empty, empty, empty, empty, empty, n_comp,
        )
    return fields, records


class _NativeElastic2dFn(torch.autograd.Function):  # type: ignore[misc]
    @staticmethod
    def forward(ctx_fn, c11, c13, c33, c44, buoyancy, wav, fd, profiles,  # type: ignore[no-untyped-def]
                src_z, src_x, rcv_z, rcv_x, B, nz, nx, dt, dx, dz, fs,
                n_comp, extension):
        dev, dtp = c11.device, c11.dtype
        nt = wav.shape[1]
        n_rcv = rcv_z.shape[0]
        fields = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev)
                  for _ in range(13)]
        shape = (B, nt, n_rcv) if n_comp == 1 else (B, nt, n_rcv, n_comp)
        records = torch.empty(*shape, dtype=dtp, device=dev)
        vx_h = torch.zeros(nt + 1, B, nz, nx, dtype=dtp, device=dev)
        vz_h = torch.zeros(nt + 1, B, nz, nx, dtype=dtp, device=dev)
        e1_h = torch.zeros(nt, B, nz, nx, dtype=dtp, device=dev)
        e2_h = torch.zeros(nt, B, nz, nx, dtype=dtp, device=dev)
        gs_h = torch.zeros(nt, B, nz, nx, dtype=dtp, device=dev)
        cs = [c.detach().contiguous() for c in (c11, c13, c33, c44, buoyancy)]
        wv = wav.detach().contiguous()
        extension.forward(*fields, *cs, fd, *profiles,
                    src_z, src_x, wv, rcv_z, rcv_x, records,
                    float(dt), float(dx), float(dz), bool(fs),
                    vx_h, vz_h, e1_h, e2_h, gs_h, n_comp)
        ctx_fn.save_for_backward(*cs, wv, fd, *profiles,
                                 src_z, src_x, rcv_z, rcv_x,
                                 vx_h, vz_h, e1_h, e2_h, gs_h)
        ctx_fn.dims = (B, nz, nx, dt, dx, dz, fs, n_comp)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:5])
        return (records, *fields[:5])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        saved = ctx_fn.saved_tensors
        cs = saved[0:5]
        wv, fd = saved[5], saved[6]
        profiles = saved[7:19]
        src_z, src_x, rcv_z, rcv_x = saved[19:23]
        hists = saved[23:28]
        B, nz, nx, dt, dx, dz, fs, n_comp = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = cs[0].device, cs[0].dtype

        adj = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev)
               for _ in range(13)]
        scratch = [torch.empty(B, 1, nz, nx, dtype=dtp, device=dev)
                   for _ in range(4)]
        grads = [torch.zeros(nz, nx, dtype=dtp, device=dev) for _ in range(5)]
        gwav = torch.zeros_like(wv)
        extension.backward(grad_records.contiguous(), *hists, *cs, fd, *profiles,
                     src_z, src_x, wv, rcv_z, rcv_x, *adj, *scratch,
                     *grads, gwav, float(dt), float(dx), float(dz), bool(fs),
                     n_comp)
        return (*grads, gwav) + (None,) * 15


def run_elastic2d_autograd(
    extension: Any,
    ctx: Any,
    state: Sequence[torch.Tensor],
    wav: torch.Tensor,
    n_comp: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Differentiable native elastic-2D run (hand-written adjoint, fs-capable)."""
    coeff_t = [ctx.coeffs[k] for k in _EL_COEFFS]
    dtp, dev = coeff_t[0].dtype, coeff_t[0].device
    fd = torch.tensor(ctx.eq._coeffs, dtype=dtp, device=dev)
    profiles = _cpml2d_profiles(ctx.cpml, dtp, dev)
    B = wav.shape[0]
    nz, nx = state[0].shape[-2], state[0].shape[-1]
    fs = bool(getattr(ctx.eq, "_free_surface", False))
    nt = wav.shape[1]
    seg = _FORCE_CKPT_SEGMENTS
    if seg is None:
        hist_bytes = (5 * nt + 2) * B * nz * nx * dtp.itemsize
        if hist_bytes > _NATIVE_HIST_BYTES_LIMIT:
            seg = _ckpt_segments(nt, n_state=13, n_hist=5)
    args = (
        *coeff_t, _require_wavelet_like(wav, coeff_t[0]), fd, profiles,
        ctx.src_z.contiguous(), ctx.src_x.contiguous(),
        ctx.rcv_z.contiguous(), ctx.rcv_x.contiguous(),
        B, nz, nx, ctx.dt, ctx.dx, ctx.dz, fs, n_comp,
    )
    if seg is not None:
        records, vx, vz, sxx, szz, sxz = _NativeElastic2dCkptFn.apply(
            *args, seg, extension
        )
    else:
        records, vx, vz, sxx, szz, sxz = _NativeElastic2dFn.apply(
            *args, extension
        )
    zeros = torch.zeros_like(vx)
    return [vx, vz, sxx, szz, sxz] + [zeros] * 8, records

def _cpml3d_profiles(
    cpml: Any,
    dtp: torch.dtype,
    dev: torch.device,
) -> tuple[torch.Tensor, ...]:
    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1).to(dtype=dtp, device=dev).contiguous()

    return (
        flat(cpml.bx_int), flat(cpml.ax_int), flat(cpml.kx_int),
        flat(cpml.by_int), flat(cpml.ay_int), flat(cpml.ky_int),
        flat(cpml.bz_int), flat(cpml.az_int), flat(cpml.kz_int),
        flat(cpml.bx_half), flat(cpml.ax_half), flat(cpml.kx_half),
        flat(cpml.by_half), flat(cpml.ay_half), flat(cpml.ky_half),
        flat(cpml.bz_half), flat(cpml.az_half), flat(cpml.kz_half),
    )

_EL3_COEFFS = ("c11", "c12", "c13", "c33", "c44", "c66", "buoyancy")

def run_elastic3d_forward(
    extension: Any,
    eq: Any,
    cpml: Any,
    coeffs: Mapping[str, torch.Tensor],
    state: Sequence[torch.Tensor],
    wav: torch.Tensor,
    pos: tuple[torch.Tensor, ...],
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    sz, sy, sx, rz, ry, rx = pos
    with torch.no_grad():
        fields = [s.contiguous() for s in state]
        f0 = fields[0]
        n_src, nt = wav.shape
        n_rcv = rz.shape[0]
        records = torch.empty(n_src, nt, n_rcv, dtype=f0.dtype, device=f0.device)
        fd = torch.tensor(eq._coeffs, dtype=f0.dtype, device=f0.device)
        extension.forward(
            fields,
            [coeffs[k].to(f0.dtype).contiguous() for k in _EL3_COEFFS],
            fd, list(_cpml3d_profiles(cpml, f0.dtype, f0.device)),
            sz.contiguous(), sy.contiguous(), sx.contiguous(),
            _require_wavelet_like(wav, f0).contiguous(),
            rz.contiguous(), ry.contiguous(), rx.contiguous(),
            records, float(dt), float(dx), float(dy), float(dz), [],
        )
    return fields, records


class _NativeElastic3dFn(torch.autograd.Function):  # type: ignore[misc]
    @staticmethod
    def forward(ctx_fn, c11, c12, c13, c33, c44, c66, buoyancy, wav, fd,  # type: ignore[no-untyped-def]
                profiles, sz, sy, sx, rz, ry, rx, B, nz, ny, nx,
                dt, dx, dy, dz, extension):
        dev, dtp = c11.device, c11.dtype
        nt = wav.shape[1]
        n_rcv = rz.shape[0]
        fields = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                  for _ in range(27)]
        records = torch.empty(B, nt, n_rcv, dtype=dtp, device=dev)
        vhists = [torch.zeros(nt + 1, B, nz, ny, nx, dtype=dtp, device=dev)
                  for _ in range(3)]
        ehists = [torch.zeros(nt, B, nz, ny, nx, dtype=dtp, device=dev)
                  for _ in range(6)]
        hists = vhists + ehists
        cs = [c.detach().contiguous()
              for c in (c11, c12, c13, c33, c44, c66, buoyancy)]
        wv = wav.detach().contiguous()
        extension.forward(fields, cs, fd, list(profiles), sz, sy, sx, wv,
                    rz, ry, rx, records, float(dt), float(dx), float(dy),
                    float(dz), hists)
        ctx_fn.save_for_backward(*cs, wv, fd, *profiles, sz, sy, sx,
                                 rz, ry, rx, *hists)
        ctx_fn.dims = (B, nz, ny, nx, dt, dx, dy, dz)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:9])
        return (records, *fields[:9])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        saved = ctx_fn.saved_tensors
        cs = list(saved[0:7])
        wv, fd = saved[7], saved[8]
        profiles = list(saved[9:27])
        sz, sy, sx, rz, ry, rx = saved[27:33]
        hists = list(saved[33:42])
        B, nz, ny, nx, dt, dx, dy, dz = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = cs[0].device, cs[0].dtype

        adj = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
               for _ in range(27)]
        scratch = [torch.empty(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                   for _ in range(9)]
        grads = [torch.zeros(nz, ny, nx, dtype=dtp, device=dev)
                 for _ in range(7)]
        gwav = torch.zeros_like(wv)
        extension.backward(grad_records.contiguous(), hists, cs, fd, profiles,
                     sz, sy, sx, wv, rz, ry, rx, adj, scratch, grads, gwav,
                     float(dt), float(dx), float(dy), float(dz))
        return (*grads, gwav) + (None,) * 17


class _NativeElastic3dCkptFn(torch.autograd.Function):  # type: ignore[misc]
    """Checkpointed twin of :class:`_NativeElastic3dFn` (bitwise gradients).
    v-histories carry ``seg+1`` slots (slot 0 = segment-start state); the six
    strain histories carry ``seg`` slots."""

    @staticmethod
    def forward(ctx_fn, c11, c12, c13, c33, c44, c66, buoyancy, wav, fd,  # type: ignore[no-untyped-def]
                profiles, sz, sy, sx, rz, ry, rx, B, nz, ny, nx,
                dt, dx, dy, dz, seg, extension):
        dev, dtp = c11.device, c11.dtype
        nt = wav.shape[1]
        n_rcv = rz.shape[0]
        fields = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                  for _ in range(27)]
        records = torch.empty(B, nt, n_rcv, dtype=dtp, device=dev)
        cs = [c.detach().contiguous()
              for c in (c11, c12, c13, c33, c44, c66, buoyancy)]
        wv = wav.detach().contiguous()

        ckpts = []
        for i0 in range(0, nt, seg):
            i1 = min(i0 + seg, nt)
            ckpts.append([f.clone() for f in fields])
            rec_seg = torch.empty(B, i1 - i0, n_rcv, dtype=dtp, device=dev)
            extension.forward(fields, cs, fd, list(profiles), sz, sy, sx,
                        wv[:, i0:i1].contiguous(), rz, ry, rx, rec_seg,
                        float(dt), float(dx), float(dy), float(dz), [])
            records[:, i0:i1] = rec_seg
        ctx_fn.save_for_backward(*cs, wv, fd, *profiles, sz, sy, sx,
                                 rz, ry, rx)
        ctx_fn.ckpts = ckpts
        ctx_fn.dims = (B, nz, ny, nx, dt, dx, dy, dz, seg, nt)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:9])
        return (records, *fields[:9])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        saved = ctx_fn.saved_tensors
        cs = list(saved[0:7])
        wv, fd = saved[7], saved[8]
        profiles = list(saved[9:27])
        sz, sy, sx, rz, ry, rx = saved[27:33]
        B, nz, ny, nx, dt, dx, dy, dz, seg, nt = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = cs[0].device, cs[0].dtype

        adj = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
               for _ in range(27)]
        scratch = [torch.empty(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                   for _ in range(9)]
        grads = [torch.zeros(nz, ny, nx, dtype=dtp, device=dev)
                 for _ in range(7)]
        gwav = torch.zeros_like(wv)
        v_hists = [torch.zeros(seg + 1, B, nz, ny, nx, dtype=dtp, device=dev)
                   for _ in range(3)]
        e_hists = [torch.zeros(seg, B, nz, ny, nx, dtype=dtp, device=dev)
                   for _ in range(6)]
        go = grad_records.contiguous()

        starts = list(range(0, nt, seg))
        for s in reversed(range(len(starts))):
            i0 = starts[s]
            i1 = min(i0 + seg, nt)
            k = i1 - i0
            fields = [c.clone() for c in ctx_fn.ckpts[s]]
            for h, f in zip(v_hists, fields[:3]):
                h[0].copy_(f[:, 0])
            k_hists = ([h[: k + 1] for h in v_hists]
                       + [h[:k] for h in e_hists])
            rec_scratch = torch.empty(B, k, rz.shape[0], dtype=dtp, device=dev)
            wav_seg = wv[:, i0:i1].contiguous()
            extension.forward(fields, cs, fd, list(profiles), sz, sy, sx, wav_seg,
                        rz, ry, rx, rec_scratch, float(dt), float(dx),
                        float(dy), float(dz), k_hists)
            gwav_seg = torch.zeros(B, k, dtype=dtp, device=dev)
            extension.backward(go[:, i0:i1].contiguous(), k_hists, cs, fd, profiles,
                         sz, sy, sx, wav_seg, rz, ry, rx, adj, scratch,
                         grads, gwav_seg,
                         float(dt), float(dx), float(dy), float(dz))
            gwav[:, i0:i1] = gwav_seg
        return (*grads, gwav) + (None,) * 18


def run_elastic3d_autograd(
    extension: Any,
    eq: Any,
    cpml: Any,
    coeffs: Mapping[str, torch.Tensor],
    state: Sequence[torch.Tensor],
    wav: torch.Tensor,
    pos: tuple[torch.Tensor, ...],
    dt: float,
    dx: float,
    dy: float,
    dz: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    sz, sy, sx, rz, ry, rx = pos
    coeff_t = [coeffs[k] for k in _EL3_COEFFS]
    dtp, dev = coeff_t[0].dtype, coeff_t[0].device
    fd = torch.tensor(eq._coeffs, dtype=dtp, device=dev)
    profiles = _cpml3d_profiles(cpml, dtp, dev)
    B = wav.shape[0]
    nz, ny, nx = state[0].shape[-3], state[0].shape[-2], state[0].shape[-1]
    nt = wav.shape[1]
    seg = _FORCE_CKPT_SEGMENTS
    if seg is None:
        hist_bytes = (9 * nt + 3) * B * nz * ny * nx * dtp.itemsize
        if hist_bytes > _NATIVE_HIST_BYTES_LIMIT:
            seg = _ckpt_segments(nt, n_state=27, n_hist=9)
    args = (
        *coeff_t, _require_wavelet_like(wav, coeff_t[0]), fd, profiles,
        sz.contiguous(), sy.contiguous(), sx.contiguous(),
        rz.contiguous(), ry.contiguous(), rx.contiguous(),
        B, nz, ny, nx, dt, dx, dy, dz,
    )
    if seg is not None:
        out = _NativeElastic3dCkptFn.apply(*args, seg, extension)
    else:
        out = _NativeElastic3dFn.apply(*args, extension)
    records = out[0]
    wavefields = list(out[1:10])
    zeros = torch.zeros_like(wavefields[0])
    return wavefields + [zeros] * 18, records

class _Native2DContext:
    """Attribute view consumed by the mechanically retained 2-D wrappers."""

    def __init__(self, context: Any) -> None:
        self.eq = context.equation
        self.coeffs = context.coefficients
        self.cpml = context.boundary
        self.dt = context.dt
        self.dx, self.dz = tuple(reversed(context.spacing))
        self.src_z, self.src_x = (
            context.source_indices[:, 0],
            context.source_indices[:, 1],
        )
        self.rcv_z, self.rcv_x = native_receiver_coordinates(context)


def execute_elastic2d(request: PropagationRequest, extension: Any) -> PropagationResult:
    """Execute one complete supported elastic 2-D native request."""
    context, state, telemetry = prepare_native_execution(request)
    view = _Native2DContext(context)
    needs_grad = torch.is_grad_enabled() and (
        request.wavelets.requires_grad
        or any(value.requires_grad for value in context.coefficients.values())
    )
    runner = run_elastic2d_autograd if needs_grad else run_elastic2d_forward
    final_state, records = runner(
        extension,
        view,
        list(state),
        request.wavelets,
        len(request.components),
    )
    return assemble_native_result(
        request,
        context,
        final_state,
        records,
        telemetry,
        native_extension="elastic2d",
    )


def execute_elastic3d(request: PropagationRequest, extension: Any) -> PropagationResult:
    """Execute one complete supported elastic 3-D native request."""
    context, state, telemetry = prepare_native_execution(request)
    source = tuple(context.source_indices[:, axis] for axis in range(3))
    receivers = native_receiver_coordinates(context)
    position = (*source, *receivers)
    dz, dy, dx = context.spacing
    needs_grad = torch.is_grad_enabled() and (
        request.wavelets.requires_grad
        or any(value.requires_grad for value in context.coefficients.values())
    )
    runner = run_elastic3d_autograd if needs_grad else run_elastic3d_forward
    final_state, records = runner(
        extension, request.equation, context.boundary, context.coefficients, list(state),
        request.wavelets, position, context.dt, dx, dy, dz,
    )
    return assemble_native_result(
        request,
        context,
        final_state,
        records,
        telemetry,
        native_extension="elastic3d",
    )


__all__ = [
    "execute_elastic2d", "execute_elastic3d",
    "_NativeElastic2dCkptFn", "_ckpt_segments",
]
