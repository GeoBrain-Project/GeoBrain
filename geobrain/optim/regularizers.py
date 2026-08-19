"""
Functional regularizers for inversion.

Pure callables (no ``nn.Module`` subclassing) that return scalar ``torch.Tensor``
values and preserve autograd through their input.

All regularizers accept a ``weight`` keyword that scales the output linearly, so
they can be summed directly in a loss function::

    loss = misfit + smoothness(m, weight=1e-3) + l1(m, weight=1e-5)

Single-tensor regularizers (penalize one parameter field):

- :func:`smoothness`: first-difference L2 norm (1D/2D/3D), optionally with
  per-cell (depth / sensitivity) weights averaged onto faces.
- :func:`smallness`: element-wise L2 deviation from a reference model.
- :func:`depth_weighting`: Li & Oldenburg per-cell depth weight to feed into
  ``smoothness(cell_weights=...)`` / ``smallness(weight=...)``.
- :func:`tikhonov`: L2 deviation from a reference model.
- :func:`total_variation`: anisotropic or isotropic TV (1D/2D/3D).
- :func:`l1`: L1 norm.
- :func:`l2`: squared L2 norm.

Inter-parameter coupling regularizers (couple TWO parameter fields, used as a
whole-dict ``Inverter`` regularizer for joint inversion, the parameter-coupling
counterpart to :class:`~geobrain.core.composition.OperatorBundle`'s forward-model
coupling):

- :func:`cross_gradient`: Gallardo–Meju structural similarity ``|∇a × ∇b|²``.
- :func:`gmm_prior`: petrophysically-guided (PGI) Gaussian-mixture negative
  log-prior over per-cell parameter vectors.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

import torch

from geobrain.core.errors import GeoBrainError

__all__ = [
    "smoothness_second_order", "total_variation_second_order",
    "smoothness",
    "smallness",
    "depth_weighting",
    "tikhonov",
    "total_variation",
    "l1",
    "l2",
    "cross_gradient",
    "gmm_prior",
]


def smoothness(
    x: torch.Tensor,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    weight: float = 1.0,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_z: float = 1.0,
    cell_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    First-difference L2 (gradient) regularizer.

    Supports 1D, 2D, and 3D tensors. For 3D input ``x`` the penalty is::

        weight * sum_i alpha_i * sum( f_i * ((Delta_i x) / d_i) ** 2 )

    with differences taken along each axis independently
    (``x[..., 1:] - x[..., :-1]``) and ``f_i`` the per-face weight (``1`` when
    ``cell_weights is None``).

    Args:
        x: 1D, 2D, or 3D real or complex tensor. Autograd is preserved through it.
        dx: Grid spacing along the x axis.
        dy: Grid spacing along the y axis (3-D only; rejected for 2-D input).
        dz: Grid spacing along the z axis (the first axis in both 2-D and 3-D).
        weight: Overall scalar weight; the result scales linearly with it.
        alpha_x: Per-axis weight on the x axis (on top of ``weight``).
        alpha_y: Per-axis weight on the y axis (3-D only; rejected for 2-D input).
        alpha_z: Per-axis weight on the z axis.
        cell_weights: Optional per-CELL weight tensor (same shape as ``x``, or
            broadcastable to it, e.g. a ``(nz, 1, 1)`` depth column). ``None``
            (default) is a no-op and is **byte-identical** to the unweighted
            call (the golden-safety / back-compat contract). When given, each
            axis's squared first difference between cells ``i`` and ``i+1`` is
            multiplied by the FACE weight = arithmetic mean
            ``0.5 * (w_i + w_{i+1})`` of the two adjacent cell weights, the
            standard cell→face averaging used by depth-weighted
            inversions. This is the seam that puts depth / sensitivity weighting into
            the OBJECTIVE, alongside :func:`smallness`'s per-element ``weight``;
            see :func:`depth_weighting`. Uniform (all-ones) weights are an exact
            no-op (the face mean is ``1.0`` and multiplying by ``1.0`` is
            bit-exact).

    Returns:
        Zero-dimensional scalar tensor.

    Note:
        Axis→weight mapping under the platform ``(nz, nx, ny)`` convention:
        1D ``(nx,)`` uses ``dx``/``alpha_x``; 2D ``(nz, nx)`` uses
        ``dz``/``alpha_z`` on axis-0 (z) and ``dx``/``alpha_x`` on axis-1 (x);
        3D ``(nz, nx, ny)`` uses ``dx``/``alpha_x`` on axis-1 (x),
        ``dy``/``alpha_y`` on axis-2 (y), and ``dz``/``alpha_z`` on axis-0 (z).
        A 2-D field has no y axis, so a non-default ``dy``/``alpha_y`` with 2-D
        input is rejected loudly (a legacy signature bound the 2-D vertical
        axis to ``dy``; that binding is retired; use ``dz``/``alpha_z``).
    """
    if x.dim() not in (1, 2, 3):
        raise GeoBrainError(
            "smoothness expects a 1D/2D/3D tensor",
            object_name="smoothness",
            field="x",
            expected="dim in {1, 2, 3}",
            actual=tuple(x.shape),
        )
    if x.dim() == 2 and (dy != 1.0 or alpha_y != 1.0):
        raise GeoBrainError(
            "smoothness: a 2-D field is (nz, nx); it has no y axis. The "
            "vertical (first) axis binds to dz/alpha_z; use those instead of "
            "the retired dy/alpha_y 2-D binding.",
            object_name="smoothness",
            field="dy/alpha_y",
            expected="left at their defaults for 2-D input",
            actual={"dy": dy, "alpha_y": alpha_y},
        )

    total = x.new_zeros(())

    if cell_weights is None:
        # Unweighted path: kept byte-for-byte identical to the historical
        # implementation so that ``cell_weights=None`` never perturbs a golden.
        if x.dim() == 1:
            diff_x = (x[1:] - x[:-1]) / dx
            total = total + alpha_x * (diff_x.abs() ** 2).sum()

        elif x.dim() == 2:
            # x has shape (nz, nx): axis-0 is z (dz/alpha_z), axis-1 is x.
            diff_x = (x[:, 1:] - x[:, :-1]) / dx
            diff_z = (x[1:, :] - x[:-1, :]) / dz
            total = (
                total
                + alpha_x * (diff_x.abs() ** 2).sum()
                + alpha_z * (diff_z.abs() ** 2).sum()
            )

        else:  # 3D
            # x has shape (nz, nx, ny): first is z, middle is x, last is y: the
            # platform (nz, nx, ny) convention. x/alpha_x weights the x (axis-1)
            # differences, y/alpha_y the y (axis-2) differences.
            diff_x = (x[:, 1:, :] - x[:, :-1, :]) / dx
            diff_y = (x[:, :, 1:] - x[:, :, :-1]) / dy
            diff_z = (x[1:, :, :] - x[:-1, :, :]) / dz
            total = (
                total
                + alpha_x * (diff_x.abs() ** 2).sum()
                + alpha_y * (diff_y.abs() ** 2).sum()
                + alpha_z * (diff_z.abs() ** 2).sum()
            )

        return weight * total

    # Cell-weighted path. Broadcast the per-cell weights up to x's shape (a no-op
    # view when they already match) so a (nz, 1, 1) depth column and a full
    # (nz, nx, ny) sensitivity field are both accepted, then average adjacent
    # cell weights onto the shared face before scaling the squared difference.
    if cell_weights.dim() != x.dim():
        raise GeoBrainError(
            "smoothness cell_weights must be broadcastable to x's shape",
            object_name="smoothness",
            field="cell_weights",
            expected=f"dim == {x.dim()} (shape {tuple(x.shape)})",
            actual=tuple(cell_weights.shape),
        )
    w = cell_weights.expand_as(x)

    if x.dim() == 1:
        diff_x = (x[1:] - x[:-1]) / dx
        face_x = 0.5 * (w[1:] + w[:-1])
        total = total + alpha_x * (face_x * diff_x.abs() ** 2).sum()

    elif x.dim() == 2:
        # (nz, nx): axis-0 is z (dz/alpha_z), axis-1 is x: same binding as
        # the unweighted branch above.
        diff_x = (x[:, 1:] - x[:, :-1]) / dx
        diff_z = (x[1:, :] - x[:-1, :]) / dz
        face_x = 0.5 * (w[:, 1:] + w[:, :-1])
        face_z = 0.5 * (w[1:, :] + w[:-1, :])
        total = (
            total
            + alpha_x * (face_x * diff_x.abs() ** 2).sum()
            + alpha_z * (face_z * diff_z.abs() ** 2).sum()
        )

    else:  # 3D
        diff_x = (x[:, 1:, :] - x[:, :-1, :]) / dx
        diff_y = (x[:, :, 1:] - x[:, :, :-1]) / dy
        diff_z = (x[1:, :, :] - x[:-1, :, :]) / dz
        face_x = 0.5 * (w[:, 1:, :] + w[:, :-1, :])
        face_y = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
        face_z = 0.5 * (w[1:, :, :] + w[:-1, :, :])
        total = (
            total
            + alpha_x * (face_x * diff_x.abs() ** 2).sum()
            + alpha_y * (face_y * diff_y.abs() ** 2).sum()
            + alpha_z * (face_z * diff_z.abs() ** 2).sum()
        )

    return weight * total


def depth_weighting(
    depths: torch.Tensor,
    *,
    reference_depth: float,
    exponent: float = 2.0,
) -> torch.Tensor:
    """
    Li & Oldenburg-style depth weighting for potential-field inversion.

    Returns the normalized per-cell weight ``w = (depth + z0)^(-exponent/2)``,
    scaled so ``max(w) == 1``. Potential-field kernels
    (gravity, magnetics) decay rapidly with distance from the receivers, so an
    unweighted least-squares inversion plates all recovered mass at the surface.
    Feeding this weight into the OBJECTIVE, as the per-cell weight of the
    regularization, cheapens structure where the data barely see it and moves
    the recovery down to depth.

    This is the tensor to pass to :func:`smoothness` (``cell_weights=``) and
    :func:`smallness` (``weight=``) to put depth weighting in the objective,
    where it actually changes the recovered model. (Passing it only as the
    :class:`~geobrain.optim.DepthWeight` gradient-processor hook does NOT: a
    static per-cell gradient scale cancels in Adam's per-element normalizer and
    merely rescales the search direction under a quasi-Newton solver; see the
    :class:`~geobrain.optim.Weight` / :class:`~geobrain.optim.DepthWeight`
    docstrings.) ``w`` is the diagonal of the Li & Oldenburg model-space
    weighting matrix ``W_z``; the classic quadratic form ``||W_z (m - m_ref)||^2``
    uses ``w ** 2`` as the per-cell multiplier; pass
    ``depth_weighting(...) ** 2`` as ``cell_weights=`` / ``weight=`` to
    reproduce it.

    Args:
        depths: Per-cell centre depths, of the model-grid shape (any shape; the
            returned weight has the same shape). For pure depth weighting a
            ``(nz,)`` depth line reshaped to a ``(nz, 1, 1)`` column broadcasts
            against a ``(nz, nx, ny)`` field.
        reference_depth: The Li & Oldenburg offset ``z0`` (keeps the shallowest
            cell's weight finite); commonly one top cell width.
        exponent: Depth-decay exponent ``beta`` (default ``2.0`` for gravity;
            ``3.0`` for magnetics, TMI kernels decay ~1/r^3).

    Returns:
        A tensor the same shape as ``depths`` with ``max == 1``, decreasing with
        depth.
    """
    w = (depths + reference_depth) ** (-exponent / 2.0)
    return w / w.max()


def smallness(
    p: torch.Tensor,
    *,
    m_ref: torch.Tensor | float = 0.0,
    weight: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Smallness regularization: penalize deviation from a reference model.

    L2 penalty ``weight * sum (p - m_ref)**2``. Acts as a soft prior pulling
    parameters toward ``m_ref`` where data is uninformative, common for regions
    that should default to a known background value.

    Args:
        p: Parameter tensor (any shape). Autograd is preserved through it.
        m_ref: Reference model, same shape as ``p`` (or scalar/broadcastable).
            Treated as a constant: gradients flow through ``p`` only.
        weight: Per-element weight, a float for a uniform scalar weight, or a
            tensor (broadcastable to ``p``) for depth/spatial weighting.

    Returns:
        Zero-dimensional non-negative scalar tensor.

    Note:
        With ``m_ref=0.0`` and ``weight=1.0`` this equals :func:`l2` and
        :func:`tikhonov` with ``m_ref=None``. Kept separate to express the
        classic smallness + smoothness pairing and to support element-wise
        weighting.
    """# ``.abs()`` so the penalty is the squared MODULUS: real and non-negative
    # for complex params too (e.g. complex conductivity / impedance), matching
    # the sibling L2-deviation terms (tikhonov / l2 / smoothness). For real
    # params this is bit-identical (|x|² == x², and the gradient agrees).
    diff = (p - m_ref).abs()
    if isinstance(weight, (int, float)) and weight == 1.0:
        return (diff ** 2).sum()
    return (weight * diff ** 2).sum()


def tikhonov(
    x: torch.Tensor,
    *,
    m_ref: torch.Tensor | None = None,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    L2 deviation from a reference model.

    Computes ``weight * ||x - m_ref||_2^2``; with ``m_ref=None`` this is
    ``weight * ||x||_2^2``. The reference-model keyword ``m_ref`` matches
    :func:`smallness`.

    Args:
        x: Real or complex tensor of any shape. Autograd is preserved through it.
        m_ref: Reference model broadcastable to ``x``, treated as a constant.
        weight: Overall scalar weight.

    Returns:
        Zero-dimensional scalar tensor.
    """
    if m_ref is None:
        diff = x
    else:
        diff = x - m_ref
    return weight * (diff.abs() ** 2).sum()


def total_variation(
    x: torch.Tensor,
    *,
    isotropic: bool = False,
    weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Total-variation regularizer.

    Two flavours:

    - **Anisotropic** (default): sum of absolute first differences along each axis
      independently, ``sum_i sum |partial_i x|``.
    - **Isotropic**: per-location L2 of the gradient vector,
      ``sum sqrt(sum_i (partial_i x)^2 + eps)``. A small ``eps`` keeps the square
      root differentiable at zero.

    Supports 1D, 2D, and 3D inputs with forward differences
    (``x[..., 1:] - x[..., :-1]``).

    Args:
        x: 1D/2D/3D real or complex tensor.
        isotropic: If ``True``, use the isotropic formulation.
        weight: Overall scalar weight.
        eps: Smoothing constant inside the square root (isotropic only).

    Returns:
        Zero-dimensional non-negative scalar tensor.
    """
    if x.dim() not in (1, 2, 3):
        raise GeoBrainError(
            "total_variation expects a 1D/2D/3D tensor",
            object_name="total_variation",
            field="x",
            expected="dim in {1, 2, 3}",
            actual=tuple(x.shape),
        )

    # Collect axis-wise forward differences.
    if x.dim() == 1:
        diffs = [x[1:] - x[:-1]]
    elif x.dim() == 2:
        diffs = [x[:, 1:] - x[:, :-1], x[1:, :] - x[:-1, :]]
    else:
        diffs = [
            x[:, :, 1:] - x[:, :, :-1],
            x[:, 1:, :] - x[:, :-1, :],
            x[1:, :, :] - x[:-1, :, :],
        ]

    if not isotropic:
        total = x.new_zeros(())
        for d in diffs:
            total = total + d.abs().sum()
        return weight * total

    # Isotropic: align differences to a common shape by cropping each
    # diff to the minimum extent along every axis, then sum the squared
    # magnitudes at matching positions.
    min_shape = [
        min(d.shape[ax] for d in diffs) for ax in range(x.dim())
    ]
    cropped = []
    for d in diffs:
        slicer = tuple(slice(0, min_shape[ax]) for ax in range(x.dim()))
        cropped.append(d[slicer])

    sq_sum = x.new_zeros(cropped[0].shape, dtype=x.real.dtype if x.is_complex() else x.dtype)
    for d in cropped:
        sq_sum = sq_sum + (d.abs() ** 2)
    return weight * torch.sqrt(sq_sum + eps).sum()


def _second_diffs(x: torch.Tensor) -> list[tuple[torch.Tensor, str]]:
    """Per-axis interior second differences ``x[i+1] - 2x[i] + x[i-1]`` (the
    ``[1, -2, 1]`` discrete-Laplacian stencil) for a 1D/2D/3D tensor.

    Returns a list of ``(diff, spacing_attr)`` where ``spacing_attr`` names the
    grid spacing that axis uses under the platform ``(nz, nx, ny)`` convention
    (last=x, first=z, middle=y in 3D). Endpoints carry no second difference
    (matching the zeroed-endpoint stencil of the reference).
    """
    if x.dim() == 1:
        return [(x[2:] - 2 * x[1:-1] + x[:-2], "x")]
    if x.dim() == 2:
        return [
            (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2], "x"),
            (x[2:, :] - 2 * x[1:-1, :] + x[:-2, :], "y"),
        ]
    return [
        (x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :], "x"),
        (x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2], "y"),
        (x[2:, :, :] - 2 * x[1:-1, :, :] + x[:-2, :, :], "z"),
    ]


def smoothness_second_order(
    x: torch.Tensor,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    weight: float = 1.0,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_z: float = 1.0,
) -> torch.Tensor:
    """Second-order (curvature / Laplacian) L2 regularizer.

    The second-derivative sibling of :func:`smoothness`: penalizes the squared
    discrete Laplacian per axis::

        weight * sum_i alpha_i * ((Delta^2_i x) / d_i^2) ** 2

    with ``Delta^2_i x = x[i+1] - 2 x[i] + x[i-1]``. Penalizing curvature
    (rather than the gradient) favours smoothly-VARYING models, piecewise-
    linear trends survive, only bends are penalized. This is classic
    2nd-order Tikhonov regularization. Axis→spacing mapping matches
    :func:`smoothness`.

    Args:
        x: 2-D ``(nz, nx)`` or 3-D ``(nz, nx, ny)`` field.
        dx / dy / dz: grid spacings used in the second differences.
        weight: overall multiplier on the returned scalar.
        alpha_x / alpha_y / alpha_z: per-axis weights.
    """
    if x.dim() not in (1, 2, 3):
        raise GeoBrainError(
            "smoothness_second_order expects a 1D/2D/3D tensor",
            object_name="smoothness_second_order", field="x",
            expected="dim in {1, 2, 3}", actual=tuple(x.shape),
        )
    spacing = {"x": dx, "y": dy, "z": dz}
    alpha = {"x": alpha_x, "y": alpha_y, "z": alpha_z}
    total = x.new_zeros(())
    for diff, axis in _second_diffs(x):
        scaled = diff / (spacing[axis] ** 2)
        total = total + alpha[axis] * (scaled.abs() ** 2).sum()
    return weight * total


def total_variation_second_order(
    x: torch.Tensor,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    weight: float = 1.0,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_z: float = 1.0,
) -> torch.Tensor:
    """Second-order (curvature) total-variation regularizer.

    The 2nd-order sibling of :func:`total_variation`: the L1 norm of the
    discrete Laplacian per axis::

        weight * sum_i alpha_i * |Delta^2_i x| / d_i^2

    Anisotropic-style (axis-summed). Where 1st-order TV promotes piecewise-
    CONSTANT models (blocky), 2nd-order TV promotes piecewise-LINEAR ones
    (ramps with sparse kinks). Axis→spacing mapping matches
    :func:`smoothness`.

    Args:
        x: 2-D ``(nz, nx)`` or 3-D ``(nz, nx, ny)`` field.
        dx / dy / dz: grid spacings used in the second differences.
        weight: overall multiplier on the returned scalar.
        alpha_x / alpha_y / alpha_z: per-axis weights.
    """
    if x.dim() not in (1, 2, 3):
        raise GeoBrainError(
            "total_variation_second_order expects a 1D/2D/3D tensor",
            object_name="total_variation_second_order", field="x",
            expected="dim in {1, 2, 3}", actual=tuple(x.shape),
        )
    spacing = {"x": dx, "y": dy, "z": dz}
    alpha = {"x": alpha_x, "y": alpha_y, "z": alpha_z}
    total = x.new_zeros(())
    for diff, axis in _second_diffs(x):
        total = total + alpha[axis] * (diff / (spacing[axis] ** 2)).abs().sum()
    return weight * total


def l1(x: torch.Tensor, *, weight: float = 1.0) -> torch.Tensor:
    """
    L1 norm: ``weight * sum |x|``.

    Args:
        x: Tensor of any shape; autograd preserved through it.
        weight: Overall scalar weight.

    Returns:
        Zero-dimensional non-negative scalar tensor.
    """
    return weight * x.abs().sum()


def l2(x: torch.Tensor, *, weight: float = 1.0) -> torch.Tensor:
    """
    Squared L2 norm: ``weight * sum |x|^2``.

    Args:
        x: Tensor of any shape; autograd preserved through it.
        weight: Overall scalar weight.

    Returns:
        Zero-dimensional non-negative scalar tensor.
    """
    return weight * (x.abs() ** 2).sum()


def cross_gradient(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    weight: float = 1.0,
    normalized: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Gallardo–Meju cross-gradient structural-similarity coupling of two models.

    Penalizes structural *dissimilarity* between two co-located parameter fields
    ``a`` and ``b`` (e.g. density and susceptibility) by ``|∇a × ∇b|²``, zero
    where the two gradients are parallel or anti-parallel (a common interface),
    positive where they cross. This is the regularization-coupling counterpart to
    :class:`~geobrain.core.composition.OperatorBundle` (which couples *forward
    models*); it is meant to be summed inside a whole-dict ``Inverter`` regularizer
    closure, e.g. ``regularizer=lambda p: cross_gradient(p["rho"], p["chi"], ...)``.

    Gradients are forward differences cropped to a common interior grid, so all
    components live on the same cells before the cross product. Autograd is
    preserved through both inputs.

    Args:
        a, b: real tensors of the SAME shape; 2D ``(nz, nx)`` or 3D
            ``(nz, nx, ny)``: the platform mesh-axis order. (The penalty
            itself is axis-label-invariant: differences use unit spacing, and
            ``|∇a × ∇b|²`` is unchanged under any consistent axis permutation
            applied to BOTH fields. Cross-gradient is trivially zero in 1D;
            gradients are always colinear, so 1D is rejected.)
        weight: overall scalar weight; the result scales linearly with it.
        normalized: when ``True`` (default) returns the ``sin²(θ)`` form
            ``Σ |∇a × ∇b|² / (|∇a|² |∇b|² + eps)``, bounded per-cell in ``[0, 1]``
            and NOT nullable by flattening either model toward zero. When
            ``False`` returns the bare ``Σ |∇a × ∇b|²`` (magnitude-dependent).
        eps: stabiliser for the normalized denominator.

    Returns:
        Zero-dimensional non-negative scalar tensor.

    Raises:
        GeoBrainError: if ``a`` and ``b`` differ in shape, or are not 2D/3D.
    """
    if a.shape != b.shape:
        raise GeoBrainError(
            "cross_gradient requires a and b of identical shape",
            object_name="cross_gradient",
            field="b.shape",
            expected=tuple(a.shape),
            actual=tuple(b.shape),
        )
    if a.dim() == 2:
        g1x = (a[:, 1:] - a[:, :-1])[:-1, :]
        g1y = (a[1:, :] - a[:-1, :])[:, :-1]
        g2x = (b[:, 1:] - b[:, :-1])[:-1, :]
        g2y = (b[1:, :] - b[:-1, :])[:, :-1]
        cross_sq = (g1x * g2y - g1y * g2x) ** 2
        norm1_sq = g1x ** 2 + g1y ** 2
        norm2_sq = g2x ** 2 + g2y ** 2
    elif a.dim() == 3:
        # Three gradient components per model, cropped to the shared interior
        # (nz-1, ny-1, nx-1) so the per-cell cross product is well defined.
        g1z = (a[1:, :, :] - a[:-1, :, :])[:, :-1, :-1]
        g1x = (a[:, 1:, :] - a[:, :-1, :])[:-1, :, :-1]
        g1y = (a[:, :, 1:] - a[:, :, :-1])[:-1, :-1, :]
        g2z = (b[1:, :, :] - b[:-1, :, :])[:, :-1, :-1]
        g2x = (b[:, 1:, :] - b[:, :-1, :])[:-1, :, :-1]
        g2y = (b[:, :, 1:] - b[:, :, :-1])[:-1, :-1, :]
        cx = g1y * g2z - g1z * g2y
        cy = g1z * g2x - g1x * g2z
        cz = g1x * g2y - g1y * g2x
        cross_sq = cx ** 2 + cy ** 2 + cz ** 2
        norm1_sq = g1x ** 2 + g1y ** 2 + g1z ** 2
        norm2_sq = g2x ** 2 + g2y ** 2 + g2z ** 2
    else:
        raise GeoBrainError(
            "cross_gradient is defined for 2D/3D models only "
            "(it is trivially zero in 1D; gradients are always colinear)",
            object_name="cross_gradient",
            field="a",
            expected="dim in {2, 3}",
            actual=tuple(a.shape),
        )

    if normalized:
        return weight * (cross_sq / (norm1_sq * norm2_sq + eps)).sum()
    return weight * cross_sq.sum()


def gmm_prior(
    pairs: torch.Tensor,
    gmm: Any,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Petrophysically-guided (PGI) Gaussian-mixture negative log-prior.

    Pulls per-cell parameter vectors toward a learned/assumed petrophysical
    Gaussian-mixture distribution (the deterministic, regularization-side analogue
    of a Bayesian GMM prior). Returns ``weight * (−Σ log p_GMM(pairs))`` so that
    minimizing it is maximizing the mixture log-likelihood of the model cells.
    Used as a term in a whole-dict ``Inverter`` regularizer, typically alongside
    :func:`cross_gradient` / :func:`smoothness` for joint inversion.

    Args:
        pairs: ``(n_cells, n_props)`` stacked per-cell parameter vectors (e.g.
            ``torch.stack([rho.flatten(), chi.flatten()], dim=1)``). Autograd is
            preserved through it.
        gmm: any object exposing ``log_prob(x: (n, n_props)) -> (n,)``, in
            particular :class:`geobrain.bayes.distributions.GaussianMixture`, so
            the deterministic prior and the Bayesian one share one implementation.
            Passed in (not imported) to keep the optim/bayes views uncoupled.
        weight: overall scalar weight.

    Returns:
        Zero-dimensional scalar tensor.
    """
    return weight * (-gmm.log_prob(pairs).sum())
