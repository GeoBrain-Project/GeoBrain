"""
geomodel -> bayes bridge: GeoGrid<->TensorMesh conversion and SGSIM-ensemble
particle adapters.

Two pieces:

- :func:`geogrid_to_mesh_array`: the explicit converter between a
  :class:`~geobrain.geomodel.GeoGrid` (GSLIB flat F-order, x-fastest) and a
  :class:`~geobrain.mesh.TensorMesh` (C-order, ``(nz, nx[, ny])``, depth
  slowest). The two containers disagree about what "2-D" means (a 2-D
  GeoGrid is a HORIZONTAL ``(nx, ny)`` slice; a 2-D TensorMesh is a VERTICAL
  ``(nz, nx)`` section) so ``axis_map`` is REQUIRED in that case; see the
  function docstring.
- :func:`ensemble_to_particles`: turns a batch of SGSIM realisations
  (normal-score space, per :mod:`geobrain.geomodel.geostats.simulation.sgsim`)
  into the ``{field: Tensor(n_realisations, *mesh.shape)}`` particle format
  :class:`~geobrain.bayes.SVGD` consumes via ``init_particles=``. It applies
  the caller-supplied :class:`~geobrain.geomodel.NormalScore` back-transform,
  validates the realisations (finite; within the target
  :class:`~geobrain.geomodel.earthmodel.Field`'s bounds/plausible range), and; this is the
  subtle correctness point, maps PHYSICAL values into the Field's LEAF
  (unconstrained) space when the field carries a ``bounds=``/``transform=``,
  because :class:`~geobrain.bayes.SVGD` always operates on unconstrained
  leaves (``EarthModel.trainables()`` are raw leaves; ``resolve()`` is what
  applies the transform forward). Particles handed to SVGD must therefore
  already be leaf-space, or every particle's implied physical value would be
  silently wrong.

Layering: ``geobrain.geomodel`` (L3) importing ``geobrain.geomodel.earthmodel`` (a
sibling subpackage) is an allowed edge, no layer contract restricts
geomodel's imports (only earthmodel's own purity rule restricts what
``earthmodel`` itself may import, i.e. earthmodel -> core only); this module lives
outside the ``geostats`` numpy-island (the family-isolation rule only
forbids ``torch`` under ``geomodel/geostats/``), so importing torch and earthmodel
here is unrestricted by every existing architecture test.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

from ..core import GeoBrainError
from .frames import GeoFrame, GeoGrid, gslib_grid_layout
from .frames._arrays import FloatArray, as_float_array

if TYPE_CHECKING:
    from .earthmodel import EarthModel
    from .geostats.transform.normal_score import NormalScore

__all__ = ["geogrid_to_mesh_array", "ensemble_to_particles"]


# ---------------------------------------------------------------------------
# GeoGrid <-> TensorMesh axis converter
# ---------------------------------------------------------------------------


def _as_flat_float64(
    values: Any,
    *,
    n_expected: int,
    object_name: str,
    field: str,
) -> FloatArray:
    """Coerce ``values`` (numpy / list / torch.Tensor) to a flat float64 array."""
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        arr = np.asarray(values, dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != n_expected:
        raise GeoBrainError(
            "value count does not match the GeoGrid's cell count",
            object_name=object_name,
            field=field,
            expected=n_expected,
            actual=int(arr.size),
        )
    return as_float_array(arr)


def _validate_axis_map(axis_map: Sequence[int], *, length: int) -> tuple[int, ...]:
    try:
        entries = tuple(int(a) for a in axis_map)
    except (TypeError, ValueError) as exc:
        raise GeoBrainError(
            "axis_map must be a sequence of GeoGrid axis indices",
            object_name="geogrid_to_mesh_array",
            field="axis_map",
            expected=f"length-{length} sequence of ints in {{0, 1, 2}}",
            actual=axis_map,
        ) from exc
    if (
        len(entries) != length
        or len(set(entries)) != length
        or any(a not in (0, 1, 2) for a in entries)
    ):
        raise GeoBrainError(
            "axis_map must hold distinct GeoGrid axis indices (0=x, 1=y, 2=z)",
            object_name="geogrid_to_mesh_array",
            field="axis_map",
            expected=f"length-{length} permutation drawn from {{0, 1, 2}}",
            actual=entries,
        )
    return entries


def geogrid_to_mesh_array(
    values: Any,
    geogrid: GeoGrid,
    mesh: Any,
    *,
    axis_map: Sequence[int] | None = None,
) -> torch.Tensor:
    """Convert a GSLIB-flat GeoGrid property into a mesh-shaped tensor.

    ``geogrid`` stores properties GSLIB flat and x-fastest. This named adapter
    creates a transient singleton third axis only when an external three-axis
    layout requires it; the source 2-D grid remains strictly two-dimensional.
    A :class:`~geobrain.mesh.TensorMesh` is C-order, ``(nz, nx[, ny])``, depth
    slowest, then x, with y (in 3-D) fastest. This function performs the
    explicit ``reshape(order="F") -> transpose -> ascontiguousarray`` mapping.

    Args:
        values: A flat, length-``geogrid.ncells`` array-like (numpy / list /
            ``torch.Tensor``) in GSLIB order, e.g. a GeoFrame column.
        geogrid: The source :class:`~geobrain.geomodel.GeoGrid`.
        mesh: The target mesh (duck-typed: reads ``.shape``, and, best
            effort, warning-only, ``.spacing``/``.origin``/``.is_uniform``
            for a geometry-compatibility sanity check).
        axis_map: For each mesh axis (in mesh-axis order), which GeoGrid
            axis (``0=x``, ``1=y``, ``2=z``) supplies it.

            - **3-D mesh** (``shape=(nz, nx, ny)``): defaults to ``(2, 0,
              1)``, the literal semantic mapping ``nz<-z, nx<-x, ny<-y``.
              Override only for a deliberately reordered mesh.
            - **2-D mesh** (``shape=(nz, nx)``): REQUIRED, no default. A 2-D
                GeoGrid is a HORIZONTAL ``(nx, ny)`` domain; a 2-D TensorMesh
                is a VERTICAL ``(nz, nx)`` section. These are different
                physical planes; there is no safe default. Pass a 2-tuple of
                adapter-layout axis indices for
              ``(mesh's nz, mesh's nx)``, e.g. ``axis_map=(1, 0)`` to read
              the grid's y-axis as depth. The GeoGrid axis NOT named in
              ``axis_map`` must have size 1 (checked), otherwise data
              would be silently dropped.

    Returns:
        A C-contiguous ``torch.float64`` CPU tensor of shape ``mesh.shape``.

    Raises:
        GeoBrainError: ``values`` count != ``geogrid.ncells``; ``axis_map``
            missing/malformed for a 2-D mesh; the mesh shape (after
            ``axis_map``) does not match the GeoGrid's cell counts.
    """
    if not isinstance(geogrid, GeoGrid):
        raise GeoBrainError(
            "geogrid must be a GeoGrid",
            object_name="geogrid_to_mesh_array",
            field="geogrid",
            expected="GeoGrid",
            actual=type(geogrid).__name__,
        )
    mesh_shape = tuple(int(s) for s in getattr(mesh, "shape", ()))
    mesh_ndim = len(mesh_shape)
    if mesh_ndim not in (2, 3):
        raise GeoBrainError(
            "mesh.shape must be 2-D or 3-D",
            object_name="geogrid_to_mesh_array",
            field="mesh",
            expected="len(mesh.shape) in (2, 3)",
            actual=mesh_shape,
        )

    grid_shape = gslib_grid_layout(geogrid).shape

    if mesh_ndim == 3:
        if axis_map is None:
            axis_map = (2, 0, 1)
        full_order = _validate_axis_map(axis_map, length=3)
    else:
        if axis_map is None:
            raise GeoBrainError(
                "axis_map is REQUIRED to convert into a 2-D TensorMesh: a "
                "2-D GeoGrid is a HORIZONTAL (nx, ny) slice, while a 2-D "
                "TensorMesh is a VERTICAL (nz, nx) section; these are "
                "different physical planes and there is no safe default "
                "mapping between them. Pass axis_map=(grid_axis_for_nz, "
                "grid_axis_for_nx) using GeoGrid axis indices 0=x, 1=y, 2=z "
                "(e.g. axis_map=(1, 0) to read the grid's y-axis as depth; "
                "the unmapped GeoGrid axis must have size 1).",
                object_name="geogrid_to_mesh_array",
                field="axis_map",
                expected="explicit 2-tuple for a 2-D mesh",
                actual=None,
            )
        two_axes = _validate_axis_map(axis_map, length=2)
        omitted = ({0, 1, 2} - set(two_axes)).pop()
        if grid_shape[omitted] != 1:
            raise GeoBrainError(
                f"axis_map {two_axes} omits GeoGrid axis {omitted} "
                f"(0=x, 1=y, 2=z), which has size {grid_shape[omitted]} != "
                "1, converting into a 2-D mesh would silently drop that "
                "axis' data; pass a 3-D mesh, or an axis_map that accounts "
                "for every non-trivial GeoGrid axis.",
                object_name="geogrid_to_mesh_array",
                field="axis_map",
                expected=f"GeoGrid axis {omitted} size 1",
                actual=grid_shape[omitted],
            )
        full_order = two_axes + (omitted,)

    expected_mesh_shape = tuple(grid_shape[a] for a in full_order[:mesh_ndim])
    if expected_mesh_shape != mesh_shape:
        raise GeoBrainError(
            "mesh.shape is not compatible with geogrid.shape under axis_map",
            object_name="geogrid_to_mesh_array",
            field="mesh",
            expected=expected_mesh_shape,
            actual=mesh_shape,
        )

    arr = _as_flat_float64(
        values,
        n_expected=geogrid.ncells,
        object_name="geogrid_to_mesh_array",
        field="values",
    )

    grid_arr = arr.reshape(grid_shape, order="F")  # (nx, ny, nz)
    transposed = np.transpose(
        grid_arr, full_order
    )  # mesh-axis order (+ trailing trivial axis in 2-D)
    if mesh_ndim == 2:
        transposed = transposed[..., 0]
    out = np.ascontiguousarray(transposed, dtype=np.float64)

    _warn_on_geometry_mismatch(geogrid, mesh, full_order, mesh_ndim)

    return torch.from_numpy(out).clone()


def _warn_on_geometry_mismatch(
    geogrid: GeoGrid, mesh: Any, full_order: tuple[int, ...], mesh_ndim: int
) -> None:
    """Best-effort spacing/origin sanity check (warn, never raise/block).

    Skipped whenever the mesh does not advertise a simple uniform
    ``spacing``/``origin`` (reading ``TensorMesh.spacing`` on a graded mesh
    itself emits a UserWarning we do not want to trigger here).
    """
    if not getattr(mesh, "is_uniform", False):
        return
    mesh_spacing = getattr(mesh, "spacing", None)
    mesh_origin = getattr(mesh, "origin", None)
    if mesh_spacing is None or mesh_origin is None:
        return
    layout = gslib_grid_layout(geogrid)
    grid_origin = layout.origin_m
    grid_spacing = layout.spacing_m
    expected_spacing = tuple(grid_spacing[a] for a in full_order[:mesh_ndim])
    expected_origin = tuple(grid_origin[a] for a in full_order[:mesh_ndim])
    if not np.allclose(expected_spacing, mesh_spacing, rtol=1e-6, atol=1e-9):
        warnings.warn(
            f"geogrid_to_mesh_array: mesh.spacing {tuple(mesh_spacing)} does "
            f"not match geogrid.spacing permuted by axis_map "
            f"({expected_spacing}); the converted VALUES are still correct "
            "(this only checks physical cell size agreement).",
            UserWarning,
            stacklevel=3,
        )
    elif not np.allclose(expected_origin, mesh_origin, rtol=1e-6, atol=1e-9):
        warnings.warn(
            f"geogrid_to_mesh_array: mesh.origin {tuple(mesh_origin)} does "
            f"not match geogrid.origin permuted by axis_map "
            f"({expected_origin}); the converted VALUES are still correct "
            "(this only checks physical placement agreement).",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# SGSIM ensemble -> SVGD init_particles
# ---------------------------------------------------------------------------


def _physical_to_leaf(transforms: Sequence[Any], physical: torch.Tensor) -> torch.Tensor:
    """Physical -> unconstrained leaf space, mirroring
    ``geobrain.geomodel.earthmodel.model._inverse_chain`` (apply each transform's
    ``.inverse`` right-to-left)."""
    value = physical
    for t in reversed(tuple(transforms)):
        value = t.inverse(value)
    return value


def _realisation_ns_values(
    realisation: Any, normal_score: "NormalScore", *, index: int
) -> FloatArray:
    """Flat GSLIB-order normal-score values for one realisation."""
    if isinstance(realisation, GeoFrame):
        if "simulation" in realisation.columns:
            col = "simulation"
        elif normal_score.output_column in realisation.columns:
            col = normal_score.output_column
        else:
            raise GeoBrainError(
                f"realisation {index} GeoFrame has neither a 'simulation' "
                f"column nor normal_score.output_column "
                f"({normal_score.output_column!r})",
                object_name="ensemble_to_particles",
                field=f"realisations[{index}]",
                expected="'simulation' or normal_score.output_column",
                actual=realisation.columns,
            )
        return as_float_array(realisation[col])
    if isinstance(realisation, torch.Tensor):
        return as_float_array(
            realisation.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        )
    return as_float_array(np.asarray(realisation, dtype=np.float64).reshape(-1))


def ensemble_to_particles(
    realisations: Sequence[Any],
    field: str,
    model: "EarthModel",
    *,
    normal_score: "NormalScore",
    geogrid: GeoGrid,
    axis_map: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """SGSIM realisations (normal-score space) -> ``SVGD(init_particles=)``.

    Each realisation is (1) back-transformed to physical space via
    ``normal_score.inverse_transform`` (READ: :mod:`.geostats.transform
    .normal_score`, the realisations from
    :class:`~geobrain.geomodel.SGSIM` are normal-score-space by contract;
    this function does the back-transform the caller would otherwise have to
    do by hand), (2) reshaped from GeoGrid GSLIB order into ``model.mesh``'s
    layout via :func:`geogrid_to_mesh_array`, (3) validated, and (4) mapped
    from PHYSICAL space into the target :class:`~geobrain.geomodel.earthmodel.Field`'s
    LEAF (unconstrained) space when that field carries a
    ``bounds=``/``transform=``.

    **Why leaf space (the subtle correctness point):** ``EarthModel``
    stores/optimises its base fields as UNCONSTRAINED leaves,
    ``EarthModel.trainables()`` hands back the raw leaf, and ``resolve()``
    is what applies ``Field.transforms`` forward to get the physical value
    (see ``geobrain/geomodel/earthmodel/model.py``). :class:`~geobrain.bayes.SVGD` (via
    ``params=``/``init_particles=``) always operates on that same leaf
    space; it never sees a ``Field``, only tensors. If this adapter handed
    back PHYSICAL particles for a bounded field, SVGD would silently sample
    the wrong distribution (feeding physical porosity values, say, into a
    space where SVGD is actually meant to explore
    ``logit((phi - lo) / (hi - lo))``). So every returned particle is in
    LEAF space, keyed by the field's bare trainable name (matching
    ``EarthModel.trainables()``'s naming for base fields); pass the result
    straight to ``SVGD(init_particles=...)``.

    Args:
        realisations: SGSIM output, a list of ``GeoFrame`` (with a
            ``"simulation"`` or ``normal_score.output_column`` column), raw
            flat arrays, or ``torch.Tensor`` s, one per realisation, all in
            NORMAL-SCORE space and GSLIB flat order.
        field: The target base-field name on ``model`` (must be a
            non-generator :class:`~geobrain.geomodel.earthmodel.Field`, a generator field
            has no per-cell leaf to seed this way).
        model: The :class:`~geobrain.geomodel.earthmodel.EarthModel` whose ``mesh`` and
            ``field``'s bounds/transform this call targets.
        normal_score: A FITTED :class:`~geobrain.geomodel.NormalScore` (the
            same one used to transform the conditioning data before SGSIM).
        geogrid: The :class:`~geobrain.geomodel.GeoGrid` domain the
            realisations were simulated on.
        axis_map: Forwarded to :func:`geogrid_to_mesh_array` (required for a
            2-D ``model.mesh``).

    Returns:
        ``{field: Tensor(n_realisations, *model.mesh.shape)}`` in LEAF space,
        dtype/device matching ``model.trainables()[field]``.

    Raises:
        GeoBrainError: Unknown/generator-backed ``field``; any realisation
            with the wrong cell count or a non-finite value (raw or
            back-transformed), naming the offending realisation's index (one
            bad particle must abort the whole batch rather than poison an
            SVGD run); a back-transformed value outside the Field's
            ``bounds=``/``transform=`` domain (also index-named).

    Warns:
        UserWarning: A realisation's physical value falls outside the
            field's :data:`~geobrain.geomodel.earthmodel.field_specs.FIELD_SPECS`
            plausible SI range (soft, physically implausible but not a hard
            contract violation, unlike an explicit ``Field(bounds=)``).
    """
    from .earthmodel.field_specs import FIELD_SPECS

    # ``EarthModel.field`` is the public per-field accessor;
    # ``model.trainables()`` gives leaf TENSORS, not the ``Field`` descriptor
    # (bounds/transform live there). The getattr-with-None keeps the old
    # duck-typing tolerance: a non-EarthModel input (no accessor) gets the
    # same clean GeoBrainError, never an AttributeError.
    accessor = getattr(model, "field", None)
    if not callable(accessor):
        raise GeoBrainError(
            f"unknown field name {field!r} on model",
            object_name="ensemble_to_particles",
            field="field",
            expected="one of []",
            actual=field,
        )
    try:
        field_obj = accessor(field)
    except KeyError as exc:
        raise GeoBrainError(
            f"unknown field name {field!r} on model",
            object_name="ensemble_to_particles",
            field="field",
            expected=exc.args[0],
            actual=field,
        ) from exc
    if field_obj.generator is not None:
        raise GeoBrainError(
            f"ensemble_to_particles does not support generator-backed "
            f"field {field!r}; there is no per-cell physical/leaf value "
            "to seed particles from (the generator's own weights are the "
            "trainables)",
            object_name="ensemble_to_particles",
            field="field",
            expected="a Field(init=...) (non-generator) field",
            actual=field,
        )

    realisations = list(realisations)
    if not realisations:
        raise GeoBrainError(
            "realisations must be non-empty",
            object_name="ensemble_to_particles",
            field="realisations",
            expected="len(realisations) >= 1",
            actual=0,
        )

    spec = FIELD_SPECS.get(field)
    leaves: list[torch.Tensor] = []
    for i, r in enumerate(realisations):
        ns_flat = _realisation_ns_values(r, normal_score, index=i)
        if ns_flat.size != geogrid.ncells:
            raise GeoBrainError(
                f"realisation {i} value count does not match geogrid.ncells",
                object_name="ensemble_to_particles",
                field=f"realisations[{i}]",
                expected=geogrid.ncells,
                actual=int(ns_flat.size),
            )
        if not np.isfinite(ns_flat).all():
            raise GeoBrainError(
                f"realisation {i} contains NaN/Inf in normal-score space, "
                "one bad realisation would silently poison the whole SVGD "
                "particle ensemble",
                object_name="ensemble_to_particles",
                field=f"realisations[{i}]",
                expected="all-finite",
                actual="contains NaN/Inf",
            )

        temp = GeoFrame(geogrid, properties={normal_score.output_column: ns_flat})
        phys_frame = normal_score.inverse_transform(temp)
        phys_flat = np.asarray(phys_frame[normal_score.column], dtype=np.float64)
        if not np.isfinite(phys_flat).all():
            raise GeoBrainError(
                f"realisation {i} back-transformed to NaN/Inf physical "
                "values (normal_score tail extrapolation), one bad "
                "realisation would silently poison the whole SVGD particle "
                "ensemble",
                object_name="ensemble_to_particles",
                field=f"realisations[{i}]",
                expected="all-finite physical values",
                actual="contains NaN/Inf",
            )

        mesh_tensor = geogrid_to_mesh_array(
            phys_flat,
            geogrid,
            model.mesh,
            axis_map=axis_map,
        )

        if spec is not None:
            unit, (lo, hi) = spec
            if not bool(((mesh_tensor >= lo) & (mesh_tensor <= hi)).all()):
                vmin = float(mesh_tensor.min())
                vmax = float(mesh_tensor.max())
                warnings.warn(
                    f"ensemble_to_particles: realisation {i} field {field!r} "
                    f"is outside its plausible SI range [{lo}, {hi}] {unit} "
                    f"(observed [{vmin}, {vmax}])",
                    UserWarning,
                    stacklevel=2,
                )

        if field_obj.transforms:
            try:
                leaf = _physical_to_leaf(field_obj.transforms, mesh_tensor)
            except GeoBrainError as exc:
                raise GeoBrainError(
                    f"realisation {i} field {field!r} back-transformed to a "
                    "physical value outside the Field's bounds/transform "
                    f"domain: {exc}",
                    object_name="ensemble_to_particles",
                    field=f"realisations[{i}]",
                    expected="within the Field's bounds/transform domain",
                    actual=f"{type(exc).__name__}: {exc}",
                ) from exc
        else:
            leaf = mesh_tensor
        leaves.append(leaf)

    particles = torch.stack(leaves, dim=0)
    ref = model.trainables()[field]
    particles = particles.to(dtype=ref.dtype, device=ref.device)
    return {field: particles}
