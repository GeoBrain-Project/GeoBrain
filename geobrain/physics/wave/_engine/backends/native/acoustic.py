"""Acoustic CUDA execution wrappers for the Wave native backend.

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

def run_acoustic2d_forward(
    extension: Any,
    ctx: Any,
    state: Sequence[torch.Tensor],
    wav: torch.Tensor,
    n_comp: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run the whole acoustic-2D time loop in the native CUDA extension.

    Mirrors ``_run_segment`` semantics exactly, per step: pressure update, velocity
    update, source injection, receiver record, but with zero Python dispatch per step
    (the nt loop lives in C++; 4 kernel launches per step). Forward-only: runs under
    ``no_grad`` and updates the state tensors in place.

    Returns ``(state, records)`` like ``_run_segment``.
    """
    cpml = ctx.cpml
    with torch.no_grad():
        p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz = [s.contiguous() for s in state]
        n_src, nt = wav.shape
        n_rcv = ctx.rcv_z.shape[0]
        shape = (n_src, nt, n_rcv) if n_comp == 1 else (n_src, nt, n_rcv, n_comp)
        records = torch.empty(*shape, dtype=p.dtype, device=p.device)
        fd = torch.tensor(ctx.eq._coeffs, dtype=p.dtype, device=p.device)
        empty = torch.empty(0, dtype=p.dtype, device=p.device)

        def flat(t: torch.Tensor) -> torch.Tensor:
            # (1,1,1,n) / (1,1,n,1) CPML profile -> contiguous (n,)
            return t.reshape(-1).to(dtype=p.dtype, device=p.device).contiguous()

        extension.forward(
            p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz,
            ctx.coeffs["kappa"].to(p.dtype).contiguous(),
            ctx.coeffs["buoyancy"].to(p.dtype).contiguous(),
            fd,
            flat(cpml.bx_int), flat(cpml.ax_int), flat(cpml.kx_int),
            flat(cpml.bz_int), flat(cpml.az_int), flat(cpml.kz_int),
            flat(cpml.bx_half), flat(cpml.ax_half), flat(cpml.kx_half),
            flat(cpml.bz_half), flat(cpml.az_half), flat(cpml.kz_half),
            ctx.src_z.contiguous(), ctx.src_x.contiguous(),
            _require_wavelet_like(wav, p).contiguous(),
            ctx.rcv_z.contiguous(), ctx.rcv_x.contiguous(),
            records,
            float(ctx.dt), float(ctx.dx), float(ctx.dz),
            empty, empty, empty,  # no history: plain forward
            n_comp,
        )
    return [p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz], records


class _NativeAcoustic2dFn(torch.autograd.Function):  # type: ignore[misc]
    """Differentiable native acoustic-2D run: hand-written CUDA forward (with p/vx/vz
    history) + hand-written adjoint-state backward. Gradients flow to kappa, buoyancy,
    and the wavelet; autograd chains them onward to vp/rho through ``eq.prepare``.
    The returned final-state fields are marked non-differentiable (snapshot only)."""

    @staticmethod
    def forward(ctx_fn, kappa, buoyancy, wav, fd, profiles, src_z, src_x,  # type: ignore[no-untyped-def]
                rcv_z, rcv_x, B, nz, nx, dt, dx, dz, n_comp, extension):
        dev, dtp = kappa.device, kappa.dtype
        nt = wav.shape[1]
        n_rcv = rcv_z.shape[0]
        fields = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev) for _ in range(7)]
        p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz = fields
        shape = (B, nt, n_rcv) if n_comp == 1 else (B, nt, n_rcv, n_comp)
        records = torch.empty(*shape, dtype=dtp, device=dev)
        p_hist = torch.zeros(nt + 1, B, nz, nx, dtype=dtp, device=dev)
        vx_hist = torch.zeros(nt + 1, B, nz, nx, dtype=dtp, device=dev)
        vz_hist = torch.zeros(nt + 1, B, nz, nx, dtype=dtp, device=dev)
        kap = kappa.detach().contiguous()
        buo = buoyancy.detach().contiguous()
        wv = wav.detach().contiguous()
        extension.forward(
            p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz,
            kap, buo, fd, *profiles,
            src_z, src_x, wv, rcv_z, rcv_x, records,
            float(dt), float(dx), float(dz),
            p_hist, vx_hist, vz_hist, n_comp,
        )
        ctx_fn.save_for_backward(kap, buo, wv, fd, *profiles,
                                 src_z, src_x, rcv_z, rcv_x,
                                 p_hist, vx_hist, vz_hist)
        ctx_fn.dims = (B, nz, nx, dt, dx, dz, n_comp)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(p, vx, vz)
        return records, p, vx, vz

    @staticmethod
    def backward(ctx_fn, grad_records, _gp, _gvx, _gvz):  # type: ignore[no-untyped-def]
        (kap, buo, wv, fd, *rest) = ctx_fn.saved_tensors
        profiles = rest[:12]
        src_z, src_x, rcv_z, rcv_x, p_hist, vx_hist, vz_hist = rest[12:]
        B, nz, nx, dt, dx, dz, n_comp = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = kap.device, kap.dtype

        adj = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev) for _ in range(7)]
        scratch = [torch.empty(B, 1, nz, nx, dtype=dtp, device=dev)
                   for _ in range(4)]
        gK = torch.zeros(nz, nx, dtype=dtp, device=dev)
        gB = torch.zeros(nz, nx, dtype=dtp, device=dev)
        gwav = torch.zeros_like(wv)

        extension.backward(
            grad_records.contiguous(),
            p_hist, vx_hist, vz_hist,
            kap, buo, fd, *profiles,
            src_z, src_x, wv, rcv_z, rcv_x,
            *adj, *scratch,
            gK, gB, gwav,
            float(dt), float(dx), float(dz), n_comp,
        )
        return (gK, gB, gwav) + (None,) * 14


def run_acoustic2d_autograd(
    extension: Any,
    ctx: Any,
    state: Sequence[torch.Tensor],
    wav: torch.Tensor,
    n_comp: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Differentiable native run (stage C): returns ``(state, records)`` where
    ``records`` carries gradients to kappa/buoyancy/wavelet via the hand-written
    adjoint kernels; the returned state (snapshot) is non-differentiable."""
    cpml = ctx.cpml
    kappa = ctx.coeffs["kappa"]
    buoyancy = ctx.coeffs["buoyancy"]
    dtp = kappa.dtype
    dev = kappa.device
    fd = torch.tensor(ctx.eq._coeffs, dtype=dtp, device=dev)

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1).to(dtype=dtp, device=dev).contiguous()

    profiles = (
        flat(cpml.bx_int), flat(cpml.ax_int), flat(cpml.kx_int),
        flat(cpml.bz_int), flat(cpml.az_int), flat(cpml.kz_int),
        flat(cpml.bx_half), flat(cpml.ax_half), flat(cpml.kx_half),
        flat(cpml.bz_half), flat(cpml.az_half), flat(cpml.kz_half),
    )
    B = wav.shape[0]
    nz, nx = state[0].shape[-2], state[0].shape[-1]
    nt = wav.shape[1]
    seg = _FORCE_CKPT_SEGMENTS
    if seg is None:
        hist_bytes = 3 * (nt + 1) * B * nz * nx * dtp.itemsize
        if hist_bytes > _NATIVE_HIST_BYTES_LIMIT:
            seg = _ckpt_segments(nt, n_state=7, n_hist=3)
    args = (
        kappa, buoyancy, _require_wavelet_like(wav, kappa), fd, profiles,
        ctx.src_z.contiguous(), ctx.src_x.contiguous(),
        ctx.rcv_z.contiguous(), ctx.rcv_x.contiguous(),
        B, nz, nx, ctx.dt, ctx.dx, ctx.dz, n_comp,
    )
    if seg is not None:
        records, p, vx, vz = _NativeAcoustic2dCkptFn.apply(
            *args, seg, extension
        )
    else:
        records, p, vx, vz = _NativeAcoustic2dFn.apply(*args, extension)
    zeros = torch.zeros_like(p)
    return [p, vx, vz, zeros, zeros, zeros, zeros], records


# --------------------------------------------------------------------------
# Segment-checkpointed native adjoint (memory: full-storage O(nt) histories ->
# O(nt/K checkpoints + K segment histories), gradients BITWISE-identical
# because the backward replays each segment with the same kernels from the
# same checkpointed state). Pure python drivers over the existing extension
# entry points: no CUDA changes. Auto-selected when the full-storage
# histories would exceed _NATIVE_HIST_BYTES_LIMIT.
# --------------------------------------------------------------------------

_NATIVE_HIST_BYTES_LIMIT = 2 << 30  # 2 GiB of p/v histories -> checkpointed
_FORCE_CKPT_SEGMENTS: int | None = None  # test hook: force a segment length


def _ckpt_segments(nt: int, n_state: int, n_hist: int) -> int:
    """Segment length minimizing checkpoint+segment-history memory:
    K* = sqrt(nt * n_state / n_hist), clamped to [8, nt]."""
    import math as _math

    return min(nt, max(8, int(_math.sqrt(nt * n_state / max(n_hist, 1)))))


class _NativeAcoustic2dCkptFn(torch.autograd.Function):  # type: ignore[misc]
    """Checkpointed twin of :class:`_NativeAcoustic2dFn` (bitwise gradients)."""

    @staticmethod
    def forward(ctx_fn, kappa, buoyancy, wav, fd, profiles, src_z, src_x,  # type: ignore[no-untyped-def]
                rcv_z, rcv_x, B, nz, nx, dt, dx, dz, n_comp, seg,
                extension):
        dev, dtp = kappa.device, kappa.dtype
        nt = wav.shape[1]
        n_rcv = rcv_z.shape[0]
        fields = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev)
                  for _ in range(7)]
        shape = (B, nt, n_rcv) if n_comp == 1 else (B, nt, n_rcv, n_comp)
        records = torch.empty(*shape, dtype=dtp, device=dev)
        kap = kappa.detach().contiguous()
        buo = buoyancy.detach().contiguous()
        wv = wav.detach().contiguous()
        empty = torch.empty(0, dtype=dtp, device=dev)

        ckpts = []
        for i0 in range(0, nt, seg):
            i1 = min(i0 + seg, nt)
            ckpts.append([f.clone() for f in fields])
            k = i1 - i0
            rshape = (B, k, n_rcv) if n_comp == 1 else (B, k, n_rcv, n_comp)
            rec_seg = torch.empty(*rshape, dtype=dtp, device=dev)
            extension.forward(*fields, kap, buo, fd, *profiles,
                        src_z, src_x, wv[:, i0:i1].contiguous(),
                        rcv_z, rcv_x, rec_seg,
                        float(dt), float(dx), float(dz),
                        empty, empty, empty, n_comp)
            records[:, i0:i1] = rec_seg
        ctx_fn.save_for_backward(kap, buo, wv, fd, *profiles,
                                 src_z, src_x, rcv_z, rcv_x)
        ctx_fn.ckpts = ckpts
        ctx_fn.dims = (B, nz, nx, dt, dx, dz, n_comp, seg, nt)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:3])
        return (records, *fields[:3])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        (kap, buo, wv, fd, *rest) = ctx_fn.saved_tensors
        profiles = rest[:12]
        src_z, src_x, rcv_z, rcv_x = rest[12:]
        B, nz, nx, dt, dx, dz, n_comp, seg, nt = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = kap.device, kap.dtype

        adj = [torch.zeros(B, 1, nz, nx, dtype=dtp, device=dev)
               for _ in range(7)]
        scratch = [torch.empty(B, 1, nz, nx, dtype=dtp, device=dev)
                   for _ in range(4)]
        gK = torch.zeros(nz, nx, dtype=dtp, device=dev)
        gB = torch.zeros(nz, nx, dtype=dtp, device=dev)
        gwav = torch.zeros_like(wv)
        hists = [torch.zeros(seg + 1, B, nz, nx, dtype=dtp, device=dev)
                 for _ in range(3)]
        go = grad_records.contiguous()

        starts = list(range(0, nt, seg))
        for s in reversed(range(len(starts))):
            i0 = starts[s]
            i1 = min(i0 + seg, nt)
            k = i1 - i0
            fields = [c.clone() for c in ctx_fn.ckpts[s]]
            for h, f in zip(hists, fields[:3]):
                h[0].copy_(f[:, 0])
            k_hists = [h[: k + 1] for h in hists]
            rshape = ((B, k, rcv_z.shape[0]) if n_comp == 1
                      else (B, k, rcv_z.shape[0], n_comp))
            rec_scratch = torch.empty(*rshape, dtype=dtp, device=dev)
            wav_seg = wv[:, i0:i1].contiguous()
            extension.forward(*fields, kap, buo, fd, *profiles,
                        src_z, src_x, wav_seg, rcv_z, rcv_x, rec_scratch,
                        float(dt), float(dx), float(dz),
                        *k_hists, n_comp)
            gwav_seg = torch.zeros(B, k, dtype=dtp, device=dev)
            extension.backward(go[:, i0:i1].contiguous(), *k_hists, kap, buo, fd,
                         *profiles, src_z, src_x, wav_seg, rcv_z, rcv_x,
                         *adj, *scratch, gK, gB, gwav_seg,
                         float(dt), float(dx), float(dz), n_comp)
            gwav[:, i0:i1] = gwav_seg
        return (gK, gB, gwav) + (None,) * 15


class _NativeAcoustic3dCkptFn(torch.autograd.Function):  # type: ignore[misc]
    """Checkpointed twin of :class:`_NativeAcoustic3dFn` (bitwise gradients)."""

    @staticmethod
    def forward(ctx_fn, kappa, buoyancy, wav, fd, profiles, sz, sy, sx,  # type: ignore[no-untyped-def]
                rz, ry, rx, B, nz, ny, nx, dt, dx, dy, dz, seg,
                extension):
        dev, dtp = kappa.device, kappa.dtype
        nt = wav.shape[1]
        n_rcv = rz.shape[0]
        fields = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                  for _ in range(10)]
        records = torch.empty(B, nt, n_rcv, dtype=dtp, device=dev)
        kap = kappa.detach().contiguous()
        buo = buoyancy.detach().contiguous()
        wv = wav.detach().contiguous()
        empty = torch.empty(0, dtype=dtp, device=dev)

        ckpts = []
        for i0 in range(0, nt, seg):
            i1 = min(i0 + seg, nt)
            ckpts.append([f.clone() for f in fields])
            rec_seg = torch.empty(B, i1 - i0, n_rcv, dtype=dtp, device=dev)
            extension.forward(*fields, kap, buo, fd, *profiles,
                        sz, sy, sx, wv[:, i0:i1].contiguous(),
                        rz, ry, rx, rec_seg,
                        float(dt), float(dx), float(dy), float(dz),
                        empty, empty, empty, empty)
            records[:, i0:i1] = rec_seg
        ctx_fn.save_for_backward(kap, buo, wv, fd, *profiles,
                                 sz, sy, sx, rz, ry, rx)
        ctx_fn.ckpts = ckpts
        ctx_fn.dims = (B, nz, ny, nx, dt, dx, dy, dz, seg, nt)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:4])
        return (records, *fields[:4])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        (kap, buo, wv, fd, *rest) = ctx_fn.saved_tensors
        profiles = rest[:18]
        sz, sy, sx, rz, ry, rx = rest[18:24]
        B, nz, ny, nx, dt, dx, dy, dz, seg, nt = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = kap.device, kap.dtype

        adj = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
               for _ in range(10)]
        scratch = [torch.empty(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                   for _ in range(6)]
        gK = torch.zeros(nz, ny, nx, dtype=dtp, device=dev)
        gB = torch.zeros(nz, ny, nx, dtype=dtp, device=dev)
        gwav = torch.zeros_like(wv)
        hists = [torch.zeros(seg + 1, B, nz, ny, nx, dtype=dtp, device=dev)
                 for _ in range(4)]
        go = grad_records.contiguous()

        starts = list(range(0, nt, seg))
        for s in reversed(range(len(starts))):
            i0 = starts[s]
            i1 = min(i0 + seg, nt)
            k = i1 - i0
            fields = [c.clone() for c in ctx_fn.ckpts[s]]
            for h, f in zip(hists, fields[:4]):
                h[0].copy_(f[:, 0])
            k_hists = [h[: k + 1] for h in hists]
            rec_scratch = torch.empty(B, k, rz.shape[0], dtype=dtp, device=dev)
            wav_seg = wv[:, i0:i1].contiguous()
            extension.forward(*fields, kap, buo, fd, *profiles,
                        sz, sy, sx, wav_seg, rz, ry, rx, rec_scratch,
                        float(dt), float(dx), float(dy), float(dz), *k_hists)
            gwav_seg = torch.zeros(B, k, dtype=dtp, device=dev)
            extension.backward(go[:, i0:i1].contiguous(), *k_hists, kap, buo, fd,
                         *profiles, sz, sy, sx, wav_seg, rz, ry, rx,
                         *adj, *scratch, gK, gB, gwav_seg,
                         float(dt), float(dx), float(dy), float(dz))
            gwav[:, i0:i1] = gwav_seg
        return (gK, gB, gwav) + (None,) * 18



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


def run_acoustic3d_forward(
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
    """Plain native 3-D forward matching the shared eager traversal semantics."""
    sz, sy, sx, rz, ry, rx = pos
    with torch.no_grad():
        fields = [s.contiguous() for s in state]
        p = fields[0]
        n_src, nt = wav.shape
        n_rcv = rz.shape[0]
        records = torch.empty(n_src, nt, n_rcv, dtype=p.dtype, device=p.device)
        fd = torch.tensor(eq._coeffs, dtype=p.dtype, device=p.device)
        empty = torch.empty(0, dtype=p.dtype, device=p.device)
        extension.forward(
            *fields,
            coeffs["kappa"].to(p.dtype).contiguous(),
            coeffs["buoyancy"].to(p.dtype).contiguous(),
            fd, *_cpml3d_profiles(cpml, p.dtype, p.device),
            sz.contiguous(), sy.contiguous(), sx.contiguous(),
            _require_wavelet_like(wav, p).contiguous(),
            rz.contiguous(), ry.contiguous(), rx.contiguous(),
            records, float(dt), float(dx), float(dy), float(dz),
            empty, empty, empty, empty,
        )
    return fields, records


class _NativeAcoustic3dFn(torch.autograd.Function):  # type: ignore[misc]
    @staticmethod
    def forward(ctx_fn, kappa, buoyancy, wav, fd, profiles, sz, sy, sx,  # type: ignore[no-untyped-def]
                rz, ry, rx, B, nz, ny, nx, dt, dx, dy, dz, extension):
        dev, dtp = kappa.device, kappa.dtype
        nt = wav.shape[1]
        n_rcv = rz.shape[0]
        fields = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                  for _ in range(10)]
        records = torch.empty(B, nt, n_rcv, dtype=dtp, device=dev)
        hists = [torch.zeros(nt + 1, B, nz, ny, nx, dtype=dtp, device=dev)
                 for _ in range(4)]
        kap = kappa.detach().contiguous()
        buo = buoyancy.detach().contiguous()
        wv = wav.detach().contiguous()
        extension.forward(*fields, kap, buo, fd, *profiles,
                    sz, sy, sx, wv, rz, ry, rx, records,
                    float(dt), float(dx), float(dy), float(dz), *hists)
        ctx_fn.save_for_backward(kap, buo, wv, fd, *profiles,
                                 sz, sy, sx, rz, ry, rx, *hists)
        ctx_fn.dims = (B, nz, ny, nx, dt, dx, dy, dz)
        ctx_fn.extension = extension
        ctx_fn.mark_non_differentiable(*fields[:4])
        return (records, *fields[:4])

    @staticmethod
    def backward(ctx_fn, grad_records, *_):  # type: ignore[no-untyped-def]
        (kap, buo, wv, fd, *rest) = ctx_fn.saved_tensors
        profiles = rest[:18]
        sz, sy, sx, rz, ry, rx = rest[18:24]
        hists = rest[24:28]
        B, nz, ny, nx, dt, dx, dy, dz = ctx_fn.dims
        extension = ctx_fn.extension
        dev, dtp = kap.device, kap.dtype

        adj = [torch.zeros(B, 1, nz, ny, nx, dtype=dtp, device=dev)
               for _ in range(10)]
        scratch = [torch.empty(B, 1, nz, ny, nx, dtype=dtp, device=dev)
                   for _ in range(6)]
        gK = torch.zeros(nz, ny, nx, dtype=dtp, device=dev)
        gB = torch.zeros(nz, ny, nx, dtype=dtp, device=dev)
        gwav = torch.zeros_like(wv)
        extension.backward(grad_records.contiguous(), *hists, kap, buo, fd, *profiles,
                     sz, sy, sx, wv, rz, ry, rx, *adj, *scratch,
                     gK, gB, gwav, float(dt), float(dx), float(dy), float(dz))
        return (gK, gB, gwav) + (None,) * 17


def run_acoustic3d_autograd(
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
    """Differentiable native 3-D run (hand-written adjoint)."""
    sz, sy, sx, rz, ry, rx = pos
    kappa = coeffs["kappa"]
    buoyancy = coeffs["buoyancy"]
    dtp, dev = kappa.dtype, kappa.device
    fd = torch.tensor(eq._coeffs, dtype=dtp, device=dev)
    profiles = _cpml3d_profiles(cpml, dtp, dev)
    B = wav.shape[0]
    nz, ny, nx = state[0].shape[-3], state[0].shape[-2], state[0].shape[-1]
    nt = wav.shape[1]
    seg = _FORCE_CKPT_SEGMENTS
    if seg is None:
        hist_bytes = 4 * (nt + 1) * B * nz * ny * nx * dtp.itemsize
        if hist_bytes > _NATIVE_HIST_BYTES_LIMIT:
            seg = _ckpt_segments(nt, n_state=10, n_hist=4)
    args = (
        kappa, buoyancy, _require_wavelet_like(wav, kappa), fd, profiles,
        sz.contiguous(), sy.contiguous(), sx.contiguous(),
        rz.contiguous(), ry.contiguous(), rx.contiguous(),
        B, nz, ny, nx, dt, dx, dy, dz,
    )
    if seg is not None:
        records, p, vx, vy, vz = _NativeAcoustic3dCkptFn.apply(
            *args, seg, extension
        )
    else:
        records, p, vx, vy, vz = _NativeAcoustic3dFn.apply(
            *args, extension
        )
    zeros = torch.zeros_like(p)
    return [p, vx, vy, vz] + [zeros] * 6, records

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


def execute_acoustic2d(request: PropagationRequest, extension: Any) -> PropagationResult:
    """Execute one complete supported acoustic 2-D native request."""
    context, state, telemetry = prepare_native_execution(request)
    view = _Native2DContext(context)
    needs_grad = torch.is_grad_enabled() and (
        request.wavelets.requires_grad
        or any(value.requires_grad for value in context.coefficients.values())
    )
    runner = run_acoustic2d_autograd if needs_grad else run_acoustic2d_forward
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
        native_extension="acoustic2d",
    )


def execute_acoustic3d(request: PropagationRequest, extension: Any) -> PropagationResult:
    """Execute one complete supported acoustic 3-D native request."""
    context, state, telemetry = prepare_native_execution(request)
    source = tuple(context.source_indices[:, axis] for axis in range(3))
    receivers = native_receiver_coordinates(context)
    position = (*source, *receivers)
    dz, dy, dx = context.spacing
    needs_grad = torch.is_grad_enabled() and (
        request.wavelets.requires_grad
        or any(value.requires_grad for value in context.coefficients.values())
    )
    runner = run_acoustic3d_autograd if needs_grad else run_acoustic3d_forward
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
        native_extension="acoustic3d",
    )


__all__ = [
    "execute_acoustic2d", "execute_acoustic3d",
    "_NativeAcoustic2dCkptFn", "_ckpt_segments",
]
