"""Transforms: change-of-variables bijectors AND the property-prep pipeline.

Two deliberately separate contracts share this module (see the section
divider mid-file): differentiable ``InvertibleTransform`` bijectors
(forward/inverse/``log_abs_det_jacobian``, for sampling and bounded
fields), and the composable per-channel ``Transform`` data-prep pipeline
(forward/optional inverse, no Jacobian).

**Layering note:** distinct from ``geobrain.bayes.transforms`` (the
semantic sampling wrappers built on top), this module is pure ``torch`` math with a single
dependency on :mod:`geobrain.core.errors`, no Bayesian-inference machinery
of any kind, so it was moved down to :mod:`geobrain.core` (L0) where two
independent consumers now share it without an upward import:

- :mod:`geobrain.bayes` (L4): the Bayesian sampling side cannot clamp a
  constrained parameter (a hard projection would bias the posterior), so a
  constrained variable (``sigma > 0``, ``phi in (0, 1)``) is instead sampled
  through a change of variables: the sampler explores an unconstrained real
  space ``u``, and a transform ``f`` maps it to the constrained space
  ``x = f(u)``. The target density in ``u``-space picks up the Jacobian term

      log p_u(u) = log p_x(f(u)) + log|det J_f(u)|,

  so that ``x = f(u)`` is distributed exactly as the constrained posterior.
  :meth:`~geobrain.bayes.Posterior.sample` applies this automatically when a
  ``transforms`` mapping is supplied; the returned samples are already in the
  constrained (``x``) space. ``geobrain.bayes.transforms`` layers thin, named
  subclasses over these bijectors (``PositiveTransform`` over :class:`Exp`,
  ``IntervalTransform`` over :class:`AffineSigmoid`, …) so the sampling API
  speaks prior-space vocabulary while the math stays single-sourced here.
- :mod:`geobrain.geomodel.earthmodel` (L3): ``Field(bounds=(lo, hi))`` sugar constructs an
  :class:`AffineSigmoid` internally: the trainable leaf lives in unconstrained
  space and the physical field is ``transform.forward(leaf)``. Earthmodel (L3)
  importing this module (L0, core) keeps the platform's layer contract
  (``earthmodel`` must not import ``geobrain.bayes``, L4) honest instead of
  reaching upward for a mechanism that was never Bayes-specific.

Each :class:`InvertibleTransform` implements:

- ``forward(u)``:                  unconstrained ``u`` → constrained ``x``
- ``inverse(x)``:                  constrained ``x`` → unconstrained ``u``
- ``log_abs_det_jacobian(u)``:     scalar ``Σ log|d f / d u|`` evaluated at ``u``

All three are pure-torch and differentiable so autograd-based samplers
(HMC / NUTS) get correct gradients of the transformed log-density.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import math

import torch
import torch.nn.functional as F

from .errors import GeoBrainError

__all__ = [
    # Change-of-variables bijectors (sampling contract: forward/inverse/log|det J|)
    "InvertibleTransform",
    "Identity",
    "Exp",
    "AffineSigmoid",
    # Property-pipeline transforms (data-prep contract: forward/inverse, no Jacobian)
    "Transform",
    "Compose",
    "Normalize",
    "Denormalize",
    "Clamp",
    "Sigmoid",
    "Logit",
    "ExpTransform",
    "LogTransform",
    "PropertyBounds",
    "DEFAULT_BOUNDS",
    "clamp_properties",
    "sigmoid_transform",
    "logit_transform",
    "normalize_properties",
    "denormalize_properties",
    "exp_transform",
    "log_transform",
    "compute_property_stats",
]


class InvertibleTransform(ABC):
    """Differentiable, invertible elementwise map between unconstrained ``u``
    and constrained ``x`` parameter space."""

    @abstractmethod
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Map unconstrained ``u`` to constrained ``x``."""

    @abstractmethod
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map constrained ``x`` back to unconstrained ``u`` (inverse of forward)."""

    @abstractmethod
    def log_abs_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        """``Σ log|d forward / d u|`` over all elements (a 0-d tensor)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(InvertibleTransform):
    """No transform: sample the parameter directly in its raw space."""

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return u

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def log_abs_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=u.dtype, device=u.device)


class Exp(InvertibleTransform):
    """Positivity constraint: ``x = exp(u) ∈ (0, ∞)``.

    ``d/du exp(u) = exp(u)`` so ``log|det J| = Σ u``.
    """

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return torch.exp(u)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(x <= 0):
            raise GeoBrainError(
                "Exp transform inverse (log) requires strictly positive values; "
                "pass an initial constrained value > 0",
                object_name="Exp", field="x",
                expected="x > 0", actual=float(x.detach().min()),
            )
        return torch.log(x)

    def log_abs_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        return u.sum()


class AffineSigmoid(InvertibleTransform):
    """Bounded constraint: ``x = lo + (hi - lo)·sigmoid(u) ∈ (lo, hi)``.

    Args:
        lo: open lower bound of the constrained space.
        hi: open upper bound; must satisfy ``hi > lo``.

    ``d/du = (hi - lo)·σ(u)·(1 - σ(u))``; using the numerically stable
    log-sigmoid identities ``log σ(u) = -softplus(-u)`` and
    ``log(1 - σ(u)) = -softplus(u)``,

        log|det J| = Σ [ log(hi - lo) - softplus(-u) - softplus(u) ].

    Round-trip note: ``forward∘inverse`` is the identity only on the **open**
    interval ``(lo, hi)``. For large ``|u|`` (e.g. ``u >= 40`` in float64)
    ``sigmoid(u)`` rounds to exactly ``1.0`` so ``forward(u) == hi``; the true
    preimage of ``hi`` is ``u = +∞`` (no finite preimage exists), so
    :meth:`inverse` deliberately raises on the closed bounds rather than
    fabricate a finite ``u``. The sampling path only feeds the user's initial
    constrained value through :meth:`inverse` (never a saturated ``forward``
    output), so this asymmetry with :class:`Exp` (whose ``inverse(∞)`` returns
    ``∞``) does not affect sampling.
    """

    def __init__(self, lo: float, hi: float) -> None:
        if not hi > lo:
            raise GeoBrainError(
                "AffineSigmoid requires hi > lo",
                object_name="AffineSigmoid", field="(lo, hi)",
                expected="hi > lo", actual=(lo, hi),
            )
        self.lo = float(lo)
        self.hi = float(hi)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(u)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if torch.any(x <= self.lo) or torch.any(x >= self.hi):
            raise GeoBrainError(
                "AffineSigmoid inverse requires values strictly inside (lo, hi); "
                "pass an initial constrained value in the open interval",
                object_name="AffineSigmoid", field="x",
                expected=f"{self.lo} < x < {self.hi}",
                actual=(float(x.detach().min()), float(x.detach().max())),
            )
        z = (x - self.lo) / (self.hi - self.lo)
        return torch.log(z) - torch.log1p(-z)   # logit(z)

    def log_abs_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        span = math.log(self.hi - self.lo)
        return (span - F.softplus(-u) - F.softplus(u)).sum()

    def __repr__(self) -> str:
        return f"AffineSigmoid(lo={self.lo}, hi={self.hi})"


# =============================================================================
# Property-pipeline transforms: per-channel property data prep (a
# preprocessing contract, deliberately not an IO one).
#
# TWO CONTRACTS LIVE IN THIS MODULE; do not mix them up:
#
# * the BIJECTORS above (``InvertibleTransform``: forward / inverse /
#   ``log_abs_det_jacobian``): change-of-variables for sampling and
#   bounded-field construction; scalar-field semantics;
# * the PIPELINE below (``Transform``: forward / optional inverse, NO
#   Jacobian): composable per-channel data preparation (normalise, clamp,
#   sigmoid-squash, log-scale) for property tensors of shape
#   ``[n_channels, ...]``; ``Compose`` chains them, and every class mirrors
#   a functional twin (``Normalize`` <-> ``normalize_properties``, ...).
#
# The pipeline ``Exp``/``Log`` classes were renamed ``ExpTransform`` /
# ``LogTransform`` at the fold (their old names collided with the bijector
# ``Exp``); the functional spellings are unchanged.
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PropertyBounds:
    """
    Configuration for geological property bounds.

    Centralizes default ranges for common geological properties
    to avoid repetition across transform functions.

    Args:
        porosity: (min, max) bounds for porosity.
        shale_volume: (min, max) bounds for shale volume.
        saturation: (min, max) bounds for water saturation.
        permeability_log: (min, max) bounds for log permeability.

    Example:
        >>> bounds = PropertyBounds()
        >>> print(bounds.as_list())  # [[0.0, 0.4], [0.0, 1.0], [0.0, 1.0]]
        >>> bounds = PropertyBounds(porosity=(0.05, 0.35))
    """

    porosity: Tuple[float, float] = (0.0, 0.4)
    shale_volume: Tuple[float, float] = (0.0, 1.0)
    saturation: Tuple[float, float] = (0.0, 1.0)
    permeability_log: Tuple[float, float] = (-3.0, 3.0)

    def as_list(self, properties: Optional[List[str]] = None) -> List[List[float]]:
        """
        Get bounds as list of [min, max] pairs.

        Args:
            properties: List of property names to include.
                       Defaults to ['porosity', 'shale_volume', 'saturation'].

        Returns:
            List of [min, max] bounds for each property.

        Raises:
            GeoBrainError: on a property name this record does not define;
                a typo must fail loud, never map to fake bounds.
        """
        if properties is None:
            properties = ['porosity', 'shale_volume', 'saturation']

        bounds_map = {
            'porosity': self.porosity,
            'shale_volume': self.shale_volume,
            'saturation': self.saturation,
            'permeability_log': self.permeability_log,
        }

        unknown = [p for p in properties if p not in bounds_map]
        if unknown:
            raise GeoBrainError(
                "PropertyBounds does not define the requested properties",
                object_name="PropertyBounds", field="properties",
                expected=f"subset of {sorted(bounds_map)}", actual=unknown,
            )
        return [list(bounds_map[p]) for p in properties]


# Default bounds instance
DEFAULT_BOUNDS = PropertyBounds()

# Default per-channel statistics for the 3-channel (porosity, shale,
# saturation) convention: the single source shared by the functional and
# class APIs below so the two never drift apart.
_DEFAULT_MEANS = [0.2, 0.3, 0.5]
_DEFAULT_STDS = [0.1, 0.2, 0.2]
_DEFAULT_SHIFT = [0.0, 0.0, 0.0]
_DEFAULT_SCALE = [1.0, 1.0, 1.0]


# =============================================================================
# Transform Classes (Object-Oriented API)
# =============================================================================

class Transform(ABC):
    """
    Abstract base class for transforms.

    Transforms are callable objects that can be composed into pipelines.
    Each transform should implement forward() and optionally inverse().

    Example:
        >>> class MyTransform(Transform):
        ...     def forward(self, x):
        ...         return x * 2
        ...     def inverse(self, x):
        ...         return x / 2
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward transform."""
        pass

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transform (optional)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse transform"
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward transform."""
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    Compose multiple transforms into a pipeline.

    Args:
        transforms: List of transforms to apply in sequence.

    Example:
        >>> pipeline = Compose([Normalize(), Clamp()])
        >>> output = pipeline(input_data)
        >>> recovered = pipeline.inverse(output)
    """

    def __init__(self, transforms: List[Transform]):
        for i, t in enumerate(transforms):
            if not isinstance(t, Transform):
                raise GeoBrainError(
                    "Compose expects a list of Transform instances",
                    object_name="Compose", field=f"transforms[{i}]",
                    expected="Transform", actual=type(t).__name__,
                )
        self.transforms = list(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            x = t(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transforms in reverse order."""
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x

    def __repr__(self) -> str:
        transforms_str = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{transforms_str}])"


class Normalize(Transform):
    """
    Normalize by subtracting mean and dividing by std.

    Args:
        means: Mean values per channel.
        stds: Standard deviation values per channel.
        channel_dim: axis holding the channels (default ``0``).

    Example:
        >>> norm = Normalize(means=[0.2, 0.3], stds=[0.1, 0.2])
        >>> normalized = norm(data)
        >>> original = norm.inverse(normalized)
    """

    def __init__(
        self,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
        channel_dim: int = 0,
    ):
        self.means = means or _DEFAULT_MEANS
        self.stds = stds or _DEFAULT_STDS
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_properties(x, self.means, self.stds, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return denormalize_properties(x, self.means, self.stds, self.channel_dim)

    def __repr__(self) -> str:
        return f"Normalize(means={self.means}, stds={self.stds})"


class Denormalize(Transform):
    """
    Denormalize by multiplying by std and adding mean.

    Inverse of :class:`Normalize`.

    Args:
        means: per-channel means added back (defaults to the module's
            3-channel porosity/shale/saturation convention).
        stds: per-channel standard deviations multiplied back.
        channel_dim: axis holding the channels (default ``0``).
    """

    def __init__(
        self,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
        channel_dim: int = 0,
    ):
        self.means = means or _DEFAULT_MEANS
        self.stds = stds or _DEFAULT_STDS
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return denormalize_properties(x, self.means, self.stds, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_properties(x, self.means, self.stds, self.channel_dim)


class Clamp(Transform):
    """
    Clamp values to specified bounds.

    Args:
        bounds: List of [min, max] bounds per channel.
        channel_dim: axis holding the channels (default ``0``).

    Example:
        >>> clamp = Clamp(bounds=[[0, 0.4], [0, 1]])
        >>> clamped = clamp(data)
    """

    def __init__(self, bounds: Optional[List[List[float]]] = None, channel_dim: int = 0):
        self.bounds = bounds or DEFAULT_BOUNDS.as_list()
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return clamp_properties(x, self.bounds, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: clamping is a projection with no true inverse.

        Returning ``x`` unchanged lets ``Clamp`` sit harmlessly in a
        :class:`Compose` pipeline's reverse pass (a projection contributes
        nothing on the way back). It does **not** recover values that
        ``forward`` clipped to the bounds.
        """
        return x

    def __repr__(self) -> str:
        return f"Clamp(bounds={self.bounds})"


class Sigmoid(Transform):
    """
    Apply sigmoid transform to map unconstrained values to bounded range.

    Args:
        ranges: List of [min, max] ranges per channel.
        eps: clamp margin keeping the inverse's ``logit`` away from 0/1.
        channel_dim: axis holding the channels (default ``0``).

    Example:
        >>> sigmoid = Sigmoid(ranges=[[0, 0.4], [0, 1]])
        >>> bounded = sigmoid(unbounded_data)
        >>> unbounded = sigmoid.inverse(bounded)
    """

    def __init__(
        self,
        ranges: Optional[List[List[float]]] = None,
        eps: float = 1e-6,
        channel_dim: int = 0,
    ):
        self.ranges = ranges or DEFAULT_BOUNDS.as_list()
        self.eps = eps
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid_transform(x, self.ranges, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return logit_transform(x, self.ranges, self.eps, self.channel_dim)

    def __repr__(self) -> str:
        return f"Sigmoid(ranges={self.ranges})"


class Logit(Transform):
    """
    Apply logit (inverse sigmoid) transform.

    Maps bounded values to an unconstrained range; inverse of
    :class:`Sigmoid`.

    Args:
        ranges: per-channel ``[min, max]`` bounds of the input values.
        eps: clamp margin protecting ``log(0)`` at the range edges.
        channel_dim: axis holding the channels (default ``0``).
    """

    def __init__(
        self,
        ranges: Optional[List[List[float]]] = None,
        eps: float = 1e-6,
        channel_dim: int = 0,
    ):
        self.ranges = ranges or DEFAULT_BOUNDS.as_list()
        self.eps = eps
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return logit_transform(x, self.ranges, self.eps, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid_transform(x, self.ranges, self.channel_dim)


class ExpTransform(Transform):
    """
    Apply exponential transform for log-normal properties.

    Args:
        shift: Shift values per channel.
        scale: Scale values per channel.
        eps: floor for the inverse's ``log`` argument (keeps
            ``log_transform`` an exact inverse for every ``y >= eps``).
        channel_dim: axis holding the channels (default ``0``).
    """

    def __init__(
        self,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        eps: float = 1e-6,
        channel_dim: int = 0,
    ):
        self.shift = shift or _DEFAULT_SHIFT
        self.scale = scale or _DEFAULT_SCALE
        self.eps = eps
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exp_transform(x, self.shift, self.scale, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return log_transform(x, self.shift, self.scale, self.eps, self.channel_dim)


class LogTransform(Transform):
    """
    Apply logarithmic transform for log-normal properties.

    Inverse of :class:`ExpTransform`: ``(log(clamp(y, eps)) - shift) / scale``
    per channel.

    Args:
        shift: per-channel shift subtracted after the log.
        scale: per-channel scale divided out after the shift.
        eps: lower clamp on the argument guarding ``log(0)`` (clamping;
            not adding; keeps the map an exact inverse for ``y >= eps``).
        channel_dim: axis holding the channels (default ``0``).
    """

    def __init__(
        self,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        eps: float = 1e-6,
        channel_dim: int = 0,
    ):
        self.shift = shift or _DEFAULT_SHIFT
        self.scale = scale or _DEFAULT_SCALE
        self.eps = eps
        self.channel_dim = channel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_transform(x, self.shift, self.scale, self.eps, self.channel_dim)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return exp_transform(x, self.shift, self.scale, self.channel_dim)


# =============================================================================
# Functional API
# =============================================================================


def _require_channel_coverage(
    properties: torch.Tensor, n_spec: int, fn: str, field: str, channel_dim: int,
) -> None:
    """Fail fast if the per-channel spec does not cover every channel.

    The per-channel helpers below iterate the spec list and touch only
    channels ``0 .. n_spec-1`` along ``channel_dim``; a tensor with *more*
    channels than the spec covers would have its trailing channels silently
    passed through unprocessed. Raise instead so the mismatch surfaces.
    """
    n = int(properties.shape[channel_dim])
    if n > n_spec:
        raise GeoBrainError(
            f"{fn}: {field} covers {n_spec} channels but the tensor has {n} "
            f"along channel_dim={channel_dim}, "
            "trailing channels would be silently left unprocessed",
            object_name=fn, field=field,
            expected=f"len({field}) >= {n}", actual=n_spec,
        )


def _apply_per_channel(
    properties: torch.Tensor,
    spec: Iterable,
    per_channel: Callable[[torch.Tensor, object], torch.Tensor],
    *,
    fn: str,
    field: str,
    channel_dim: int = 0,
) -> torch.Tensor:
    """Run ``per_channel`` over each channel slice, with shared guards.

    Every functional transform in this module shares the same scaffold:
    validate that the spec covers all channels, clone the input, then loop
    over channels applying a per-channel rule. Only the per-channel math
    differs, so it is passed in as ``per_channel(slice, spec_item) -> slice``;
    the guard + clone + loop live here once.

    The channel axis is ``channel_dim`` (default ``0``, the documented
    ``[n_channels, ...]`` layout). Supplying ``channel_dim`` makes the
    layout explicit for batched ``(N, C, ...)`` inputs, so a per-sample
    mis-transform (treating the batch axis as the channel axis) becomes a
    caller choice rather than a silent accident.

    The whole operation is autograd-safe: ``clone`` is differentiable and
    each slice write is an in-place update on the clone, not on ``properties``.
    """
    spec = list(spec)
    _require_channel_coverage(properties, len(spec), fn, field, channel_dim)

    # Work in the canonical "channel axis at dim 0" frame so the loop indexes
    # dim 0. ``movedim`` returns a view; for channel_dim == 0 it is a no-op,
    # keeping the default path behaviour-identical to the hand-written loops.
    moved = properties.movedim(channel_dim, 0)
    result = moved.clone()
    n_channels = result.shape[0]
    for i, spec_item in enumerate(spec):
        if i < n_channels:
            result[i] = per_channel(moved[i], spec_item)

    return result.movedim(0, channel_dim)


def clamp_properties(
        properties: torch.Tensor,
        bounds: Optional[List[List[float]]] = None,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Clamp geological properties to physically plausible ranges.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        bounds (list, optional): List of [min, max] bounds for each channel
                               Defaults to standard geological property bounds
        channel_dim (int): Axis holding the channels. Defaults to 0, the
                           documented ``[n_channels, ...]`` layout. Set to 1
                           for batched ``(N, C, ...)`` inputs.

    Returns:
        torch.Tensor: Clamped properties tensor
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS.as_list()

    return _apply_per_channel(
        properties, bounds,
        lambda chan, b: torch.clamp(chan, min=b[0], max=b[1]),
        fn="clamp_properties", field="bounds", channel_dim=channel_dim,
    )


def _sigmoid_channel(chan: torch.Tensor, rng) -> torch.Tensor:
    min_val, max_val = rng
    return min_val + (max_val - min_val) * torch.sigmoid(chan)


def sigmoid_transform(
        properties: torch.Tensor,
        ranges: Optional[List[List[float]]] = None,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Transform unconstrained values to specified ranges using sigmoid function.

    This is useful when generating properties from unconstrained outputs
    (e.g., from neural networks).

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        ranges (list, optional): List of [min, max] ranges for each channel
                               Defaults to standard geological property ranges
        channel_dim (int): Axis holding the channels (default 0).

    Returns:
        torch.Tensor: Transformed properties tensor
    """
    if ranges is None:
        ranges = DEFAULT_BOUNDS.as_list()

    return _apply_per_channel(
        properties, ranges, _sigmoid_channel,
        fn="sigmoid_transform", field="ranges", channel_dim=channel_dim,
    )


def logit_transform(
        properties: torch.Tensor,
        ranges: Optional[List[List[float]]] = None,
        eps: float = 1e-6,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Apply inverse sigmoid (logit) transform to convert bounded values
    to unconstrained values.

    This is the inverse of sigmoid_transform and useful for preparing
    bounded data for neural network training.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        ranges (list, optional): List of [min, max] ranges for each channel
        eps (float): Small value to prevent numerical instability
        channel_dim (int): Axis holding the channels (default 0).

    Returns:
        torch.Tensor: Logit-transformed properties tensor
    """
    if ranges is None:
        ranges = DEFAULT_BOUNDS.as_list()

    def _logit_channel(chan: torch.Tensor, rng) -> torch.Tensor:
        min_val, max_val = rng
        # Normalize to [0, 1], clip to avoid log(0)/log(inf), then logit.
        normalized = (chan - min_val) / (max_val - min_val)
        normalized = torch.clamp(normalized, eps, 1.0 - eps)
        return torch.log(normalized / (1.0 - normalized))

    return _apply_per_channel(
        properties, ranges, _logit_channel,
        fn="logit_transform", field="ranges", channel_dim=channel_dim,
    )


def normalize_properties(
        properties: torch.Tensor,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Normalize properties by subtracting mean and dividing by standard deviation.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        means (list, optional): List of mean values for each channel
        stds (list, optional): List of standard deviation values for each channel
        channel_dim (int): Axis holding the channels (default 0).

    Returns:
        torch.Tensor: Normalized properties tensor
    """
    if means is None:
        means = _DEFAULT_MEANS

    if stds is None:
        stds = _DEFAULT_STDS

    zero = [i for i, sd in enumerate(stds) if float(sd) == 0.0]
    if zero:
        raise GeoBrainError(
            "normalize_properties received zero standard deviations, the "
            "division would silently produce inf/nan",
            object_name="normalize_properties", field="stds",
            expected="non-zero per-channel standard deviations",
            actual=f"zero at channel indices {zero}",
        )
    return _apply_per_channel(
        properties, zip(means, stds),
        lambda chan, ms: (chan - ms[0]) / ms[1],
        fn="normalize_properties", field="means", channel_dim=channel_dim,
    )


def denormalize_properties(
        properties: torch.Tensor,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Denormalize properties by multiplying by standard deviation and adding mean.

    This is the inverse of normalize_properties.

    Args:
        properties (torch.Tensor): Normalized properties tensor of shape [n_channels, ...]
        means (list, optional): List of mean values for each channel
        stds (list, optional): List of standard deviation values for each channel
        channel_dim (int): Axis holding the channels (default 0).

    Returns:
        torch.Tensor: Denormalized properties tensor
    """
    if means is None:
        means = _DEFAULT_MEANS

    if stds is None:
        stds = _DEFAULT_STDS

    return _apply_per_channel(
        properties, zip(means, stds),
        lambda chan, ms: chan * ms[1] + ms[0],
        fn="denormalize_properties", field="means", channel_dim=channel_dim,
    )


def exp_transform(
        properties: torch.Tensor,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Apply exponential transform to handle properties with log-normal distributions
    (like permeability).

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        shift (list, optional): List of shift values for each channel
        scale (list, optional): List of scale values for each channel
        channel_dim (int): Axis holding the channels (default 0).

    Returns:
        torch.Tensor: Exponentially transformed properties tensor
    """
    if shift is None:
        shift = _DEFAULT_SHIFT

    if scale is None:
        scale = _DEFAULT_SCALE

    return _apply_per_channel(
        properties, zip(shift, scale),
        lambda chan, ss: torch.exp(chan * ss[1] + ss[0]),
        fn="exp_transform", field="shift", channel_dim=channel_dim,
    )


def log_transform(
        properties: torch.Tensor,
        shift: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        eps: float = 1e-6,
        channel_dim: int = 0,
) -> torch.Tensor:
    """
    Apply logarithmic transform for properties with log-normal distributions.

    This is the inverse of exp_transform.

    Args:
        properties (torch.Tensor): Properties tensor of shape [n_channels, ...]
        shift (list, optional): List of shift values for each channel
        scale (list, optional): List of scale values for each channel
        eps (float): Lower clamp on the argument to guard log(0)/negatives
            without shifting valid positive values
        channel_dim (int): Axis holding the channels (default 0).

    Returns:
        torch.Tensor: Logarithmically transformed properties tensor
    """
    if shift is None:
        shift = _DEFAULT_SHIFT

    if scale is None:
        scale = _DEFAULT_SCALE

    def _log_channel(chan: torch.Tensor, ss) -> torch.Tensor:
        s, sc = ss
        # Clamp the argument to ``eps`` (rather than adding it) so the
        # transform stays the EXACT inverse of exp_transform for every
        # ``y >= eps``: an additive ``+eps`` corrupts the inverse for
        # small positive ``y`` (e.g. log(3.6e-6 + 1e-6) != log(3.6e-6)).
        return (torch.log(torch.clamp(chan, min=eps)) - s) / sc

    return _apply_per_channel(
        properties, zip(shift, scale), _log_channel,
        fn="log_transform", field="shift", channel_dim=channel_dim,
    )


def compute_property_stats(
        properties: torch.Tensor,
        channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation for each property channel.

    Args:
        properties (torch.Tensor): Properties tensor with a channel axis.
        channel_dim (int): Axis holding the channels (default 0), the same
            convention as the other functional-API members.

    Returns:
        tuple: (means, stds) - Mean and standard deviation tensors.

    Note:
        ``torch.std`` is unbiased (``n - 1`` denominator): a channel with a
        single element yields ``nan``, feed at least two samples per channel.
    """
    # Flatten all dimensions except channel dimension
    props = properties.movedim(channel_dim, 0)
    flat_shape = (props.shape[0], -1)
    flat_props = props.reshape(flat_shape)

    # Compute statistics
    means = torch.mean(flat_props, dim=1)
    stds = torch.std(flat_props, dim=1)

    return means, stds
