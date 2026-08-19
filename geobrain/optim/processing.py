"""
The Inverter hook contract: gradient processors and step projections.

Two small protocols formalize the two places a user legitimately wants to
intervene in the optimization loop:

- :class:`GradientProcessor`: mutates a parameter's gradient AFTER
  ``backward()`` and BEFORE ``optimizer.step()`` (masking, smoothing,
  clipping, scaling, anything that reshapes the search direction).
- :class:`StepProjection`: mutates the parameter values themselves AFTER
  ``optimizer.step()`` (bound clamps, freezing a sub-region back to a fixed
  value, anything that projects the iterate back onto a feasible set).

Both are plain structural protocols (``Callable``-shaped; no base class, no
registration) so a user's own function or object slots in without importing
anything from this module. The built-ins below cover the common cases so most
scripts never need to write one.

Targeting: every built-in accepts an exact-name ``field=``, ``None``
(the default) applies the processor to every parameter whose gradient (for
:class:`GradientProcessor`) has ``ndim >= 2`` (mesh-shaped fields), skipping
0-d/1-d scalars so a stray scalar hyperparameter never gets smoothed / masked
/ clipped by an apply-to-everything processor. Passing an explicit ``field``
name always applies regardless of ndim, the caller opted in by name. Glob /
prefix targeting is deliberately not implemented (this module only ever does an exact
``name == field`` comparison).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F

from geobrain.core.errors import GeoBrainError
from geobrain.core.validation import clamp_params_in_place

__all__ = [
    "GradientProcessor",
    "StepProjection",
    "NaNGuard",
    "Mask",
    "GaussianSmooth",
    "NormClip",
    "Weight",
    "DepthWeight",
    "BoundsClamp",
    "Freeze",
    "log_every",
]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class GradientProcessor(Protocol):
    """Post-backward, pre-step gradient hook.

    ``__call__(name, grad, params) -> Tensor``: receives the parameter's
    NAME, its current (post-backward) gradient, and the live parameter
    mapping (read-only by convention; conforming processors return a new
    tensor rather than mutating ``params`` in place). The Inverter assigns
    the returned tensor back to ``param.grad``.
    """

    def __call__(
        self,
        name: str,
        grad: torch.Tensor,
        params: Mapping[str, torch.Tensor],
    ) -> torch.Tensor: ...


@runtime_checkable
class StepProjection(Protocol):
    """Post-step parameter projection.

    ``__call__(params) -> None``: mutates the live parameter mapping
    IN PLACE (under ``torch.no_grad()``; the Inverter is responsible for
    the no-grad context). Used for bound clamps, freezing a sub-region, or
    any other feasible-set projection applied after the optimizer step.
    """

    def __call__(self, params: MutableMapping[str, torch.Tensor]) -> None: ...


@runtime_checkable
class _ObservationCadence(Protocol):
    """Private callback predicate checked before materializing a record."""

    def should_observe(self, iteration: int) -> bool: ...


class _DepthWeightMesh(Protocol):
    """Structural mesh surface consumed by :func:`DepthWeight`."""

    @property
    def dz(self) -> float: ...

    @property
    def shape(self) -> Sequence[int]: ...

    def center_lines(self) -> tuple[torch.Tensor, ...]: ...


# ---------------------------------------------------------------------------
# Shared targeting helper
# ---------------------------------------------------------------------------


def _targets(name: str, grad: torch.Tensor, field: str | None) -> bool:
    """Exact-name targeting shared by every built-in :class:`GradientProcessor`.

    ``field is None`` → apply to every mesh-shaped parameter (``grad.ndim >=
    2``); an explicit ``field`` name always applies, regardless of ndim.
    """
    if field is not None:
        return name == field
    return len(grad.shape) >= 2


# ---------------------------------------------------------------------------
# GradientProcessor built-ins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NaNGuard:
    """Replace non-finite gradient entries with zero.

    ``nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)`` on the targeted
    entries. This is the built-in ``nan_policy="guard"`` auto-prepends on
    :class:`~geobrain.optim.Inverter` problems whose likelihood declares
    ``gradient_singular = True`` (a removable 0/0 at a perfectly-fit
    prediction), but it is also usable standalone.
    """

    field: str | None = None

    def __call__(
        self, name: str, grad: torch.Tensor, params: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if not _targets(name, grad, self.field):
            return grad
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class Mask:
    """Elementwise-multiply the gradient by a fixed mask (e.g. freeze a
    water layer by zeroing its gradient entries).

    Attributes:
        mask: multiplicative 0/1 (or float) gradient mask tensor.
        field: exact parameter name to target (``None`` = every param).
    """

    mask: torch.Tensor
    field: str | None = None

    def __call__(
        self, name: str, grad: torch.Tensor, params: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if not _targets(name, grad, self.field):
            return grad
        return grad * self.mask


def _gaussian_kernel1d(k: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Normalized 1-D Gaussian kernel of length ``k`` (odd), std ``sigma``."""
    half = k // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    w = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return w / w.sum()


def _separable_gaussian_blur(x: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the LAST TWO dims of ``x``, replicate-padded.

    Any leading dims are treated as an independent batch. Equivalent (up to
    the truncation radius) to ``scipy.ndimage.gaussian_filter(x, sigma,
    mode="nearest")`` restricted to a ``(2*[k//2]+1)``-tap kernel, replicate
    padding is scipy's "nearest" boundary mode.
    """
    if x.dim() < 2:
        return x
    *lead, H, W = x.shape
    x4 = x.reshape(-1, 1, H, W)
    half = k // 2
    kernel1d = _gaussian_kernel1d(k, sigma, x.dtype, x.device)

    # Width pass.
    xw = F.pad(x4, (half, half, 0, 0), mode="replicate")
    xw = F.conv2d(xw, kernel1d.view(1, 1, 1, k))
    # Height pass.
    xh = F.pad(xw, (0, 0, half, half), mode="replicate")
    out = F.conv2d(xh, kernel1d.view(1, 1, k, 1))

    return out.reshape(*lead, H, W)


@dataclass(frozen=True)
class GaussianSmooth:
    """Separable Gaussian blur of the gradient (replicate-padded boundary).

    Smooths the search direction over the last two (spatial) dims of the
    targeted gradient(s), the platform-wide mesh convention ``(nz, nx[,
    ny])`` (any leading dims, e.g. a batch/channel axis, pass through
    untouched). Adapted from the Marmousi FWI examples' use of
    ``scipy.ndimage.gaussian_filter`` to smooth a starting model, ported to a
    torch ``conv2d``-based separable blur so it composes as a gradient-space
    operator: replicate (edge) padding matches scipy's ``mode="nearest"``.

    Args:
        k: Kernel size (odd, ``>= 1``) per axis; the effective half-width is
            ``k // 2``.
        sigma: Gaussian standard deviation, in cells.
        field: Exact parameter name, or ``None`` for every mesh-shaped
            (``ndim >= 2``) gradient.
    """

    k: int = 3
    sigma: float = 1.0
    field: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or self.k < 1 or self.k % 2 == 0:
            raise GeoBrainError(
                "GaussianSmooth.k must be a positive odd integer",
                object_name="GaussianSmooth",
                field="k",
                expected="positive odd int",
                actual=self.k,
            )
        if not isinstance(self.sigma, (int, float)) or self.sigma <= 0.0:
            raise GeoBrainError(
                "GaussianSmooth.sigma must be a positive float",
                object_name="GaussianSmooth",
                field="sigma",
                expected="positive float",
                actual=self.sigma,
            )

    def __call__(
        self, name: str, grad: torch.Tensor, params: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if not _targets(name, grad, self.field):
            return grad
        return _separable_gaussian_blur(grad, self.k, self.sigma)


@dataclass(frozen=True)
class NormClip:
    """Clip the gradient's L2 norm to ``max_norm`` (per targeted parameter,
    NOT a global norm across parameters, mirrors
    ``torch.nn.utils.clip_grad_norm_`` applied one tensor at a time).

    Attributes:
        max_norm: global-norm ceiling applied to the gradient.
        field: exact parameter name to target (``None`` = every param).
    """

    max_norm: float
    field: str | None = None

    def __call__(
        self, name: str, grad: torch.Tensor, params: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if not _targets(name, grad, self.field):
            return grad
        norm = grad.norm()
        if norm > self.max_norm and norm > 0:
            return grad * (self.max_norm / norm)
        return grad


@dataclass(frozen=True)
class Weight:
    """Elementwise-multiply the gradient by a fixed weight tensor ``w``
    (broadcastable against the targeted gradient's shape). The one generic
    scaling processor; :func:`DepthWeight` is a closed-form factory over
    it.

    Note:
        A static gradient scale reshapes the search direction but CANCELS in
        Adam's per-element (RMS) normalizer; it does not change the recovered
        model under Adam. Weighting that must change the answer belongs in the
        OBJECTIVE (``smoothness(cell_weights=...)`` / ``smallness(weight=...)``;
        see :func:`geobrain.optim.depth_weighting`).

    Attributes:
        w: multiplicative gradient weight (scalar or broadcastable tensor).
        field: exact parameter name to target (``None`` = every param).
        """

    w: torch.Tensor
    field: str | None = None

    def __call__(
        self, name: str, grad: torch.Tensor, params: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if not _targets(name, grad, self.field):
            return grad
        return grad * self.w


def DepthWeight(
    mesh: _DepthWeightMesh,
    exponent: float = 3.0,
    field: str | None = None,
) -> Weight:
    """Closed-form Li & Oldenburg-style depth-weighting factory over :class:`Weight`.

    Builds ``w_z = (z + z0) ** (-exponent / 2)``, normalized so the shallowest
    cell carries unit weight (``w /= w.max()``), where ``z`` is the mesh's
    axis-0 (depth) cell-center coordinate (``mesh.center_lines()[0]``,
    platform convention: depth is the slow axis) and ``z0 = mesh.dz`` (one
    z-cell width) is the standard Li & Oldenburg regularizing offset that
    keeps the top cell's weight finite. The convention is
    ``w(z) ~ (z + z0)^{-exponent/2}``, used e.g. with ``exponent=2`` for
    gravity and ``exponent=3`` for magnetics, the default here, ``3.0``,
    is the more conservative/common choice.

    The returned vector is reshaped to ``(nz, 1, ..., 1)`` so it broadcasts
    against a gradient shaped like ``mesh.shape`` (``(nz, nx)`` in 2-D,
    ``(nz, nx, ny)`` in 3-D), down-weighting along axis 0 only.

    Note:
        As a gradient processor this reshapes the L-BFGS metric / search
        direction, but a STATIC per-cell scale **cancels** in Adam's
        per-element (RMS) normalizer, so it does NOT change the recovered model
        under Adam (numerically verified: model diff ~1.5e-4 with/without). To
        make depth weighting that actually MOVES the model, put it in the
        OBJECTIVE instead: ``smoothness(cell_weights=depth_weighting(...))`` and
        ``smallness(weight=...)`` (see :func:`geobrain.optim.depth_weighting`).

    Args:
        mesh: A :class:`~geobrain.mesh.TensorMesh` (or any object exposing
            ``center_lines()``, ``shape``, and ``dz`` with the same
            semantics).
        exponent: Depth-decay exponent (default ``3.0``).
        field: Exact parameter name the returned :class:`Weight` targets, or
            ``None`` for every mesh-shaped gradient.

    Returns:
        A :class:`Weight` wrapping the closed-form depth-weight vector.
    """
    z = mesh.center_lines()[0]
    z0 = mesh.dz
    w = (z + z0) ** (-exponent / 2.0)
    w = w / w.max()
    ndim = len(mesh.shape)
    w = w.reshape((w.shape[0],) + (1,) * (ndim - 1))
    return Weight(w=w, field=field)


# ---------------------------------------------------------------------------
# StepProjection built-ins
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundsClamp:
    """Post-step ``(lo, hi)`` clamp: the mechanism ``Inverter(bounds=...)``
    sugar compiles to internally (placed FIRST in the effective
    ``step_projections`` list). ``bounds=None`` is a no-op (the Inverter
    fills it from its own normalized ``bounds=`` mapping when compiling the
    sugar; constructed bare with no ``bounds=``, it does nothing)."""

    bounds: Mapping[str, tuple[float | None, float | None]] | None = None

    def __call__(self, params: MutableMapping[str, torch.Tensor]) -> None:
        if not self.bounds:
            return
        clamp_params_in_place(params, self.bounds)


@dataclass(frozen=True)
class Freeze:
    """Force a sub-region of one parameter back to fixed ``values`` after
    every step (e.g. pin a known water-layer velocity in FWI).

    Args:
        field: The exact parameter name to freeze a sub-region of.
        where: A boolean mask or a slice selecting the frozen entries.
        values: The FULL reference tensor, same shape as ``params[field]``
            (e.g. the known/true model), only the ``where``-selected
            entries (``values[where]``) are written into
            ``params[field][where]`` every step; the rest of ``values`` is
            never read. Matches the natural call site
            ``Freeze("vp", where=water_mask, values=vp_true)``, where
            ``vp_true`` is the full model, not a hand-extracted subset.
    """

    field: str
    where: torch.Tensor | slice
    values: torch.Tensor

    def __call__(self, params: MutableMapping[str, torch.Tensor]) -> None:
        if self.field not in params:
            raise GeoBrainError(
                "Freeze.field not in params",
                object_name="Freeze",
                field="field",
                expected=f"key in {sorted(params)}",
                actual=self.field,
            )
        p = params[self.field]
        src = self.values.to(dtype=p.dtype, device=p.device)
        p[self.where] = src[self.where]


# ---------------------------------------------------------------------------
# log_every: the lazy-clone run(callback=) observer
# ---------------------------------------------------------------------------


class _LogEvery:
    """``run(callback=log_every(k))`` observer: prints ``it``/``loss`` every
    ``k`` iterations; returns ``None`` (never triggers early stop).

    ``should_observe(iteration)`` is the narrow typed cadence seam used before
    materializing a detached callback record. ``.every`` remains descriptive
    state for display and introspection.
    """

    __slots__ = ("every",)

    def __init__(self, every: int) -> None:
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise GeoBrainError(
                "log_every(k): k must be a positive integer",
                object_name="log_every",
                field="k",
                expected="positive int",
                actual=every,
            )
        self.every = every

    def __call__(
        self, it: int, loss: float, params: Mapping[str, torch.Tensor]
    ) -> None:
        print(f"iter {it:5d}: loss={loss:.6e}")
        return None

    def should_observe(self, iteration: int) -> bool:
        """Return whether this callback should observe ``iteration``."""
        return iteration % self.every == 0

    def __repr__(self) -> str:
        return f"log_every({self.every})"


def log_every(k: int) -> _LogEvery:
    """Return a ``run(callback=...)`` observer that prints ``it``/``loss``
    every ``k`` iterations.

    The returned typed predicate is checked before a callback record is
    created. Every successful iteration still owns one committed parameter
    snapshot for truthful partial results; ``log_every(30)`` adds a callback
    record clone only once every 30 iterations, avoiding hidden per-iteration
    tensor traffic on large 3-D volumes.
    """
    return _LogEvery(k)
