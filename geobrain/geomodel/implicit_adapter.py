"""``ImplicitFieldAdapter``: the implicit-geology -> earth-model bridge (P4b,
the earth-model layer).

Turns a fitted :class:`~geobrain.geomodel.implicit.ImplicitModel` (surface
points / orientations / faults) into an ``nn.Module`` that plugs straight
into :class:`~geobrain.geomodel.earthmodel.Field`'s ``generator=`` seam::

    adapter = ImplicitFieldAdapter(
        implicit_model, property_values=rho_per_unit, mesh=mesh,
    )
    model = EarthModel(mesh, fields={"rho": Field(generator=adapter)})

Three steps, chained inside :meth:`ImplicitFieldAdapter.forward`:

1. ``implicit_model.forward(soft, temperature)``, the Cokriging scalar
   field + its per-surface iso-values (NOT the ``block``; see "Scope" below).
2. ``interpolator.scalar_field_to_property`` (the documented physics bridge,
   ``geobrain/geomodel/implicit/interpolator.py``), turns the scalar field
   into a per-cell PHYSICAL property using ``property_values`` (one value per
   lithology unit, ordered bottom -> top).
3. A reshape + ``permute`` from the implicit model's flat, config-native grid
   order into the target :class:`~geobrain.mesh.TensorMesh`'s ``mesh.shape``
   layout (see "Grid ordering" below; this is NOT the GSLIB order
   :func:`~geobrain.geomodel.bridge.geogrid_to_mesh_array` converts, so that
   function is not reused here; the permute below is the implicit-model
   analogue).

Trainables (spec §8's "invert unit properties + fault displacement"
parameterization): ``implicit_model.fault_displacements`` (if the model has
faults, an ``nn.ParameterList``, already registered by ``ImplicitModel``
itself) and ``property_values`` (only when the tensor passed in has
``requires_grad=True``: otherwise it is registered as a frozen buffer, so it
is still moved/cast correctly by ``EarthModel``'s ``net.to(dtype, device)``
generator-seam step but does NOT become a trainable leaf). Both compose
through ordinary ``nn.Module`` registration, so
``torch.func.functional_call`` (the mechanism ``EarthModel._compute_base_field``
uses to run a generator field; see that module's docstring) reparametrizes
BOTH by their dotted path with no special-casing here.

Scope (per spec §8): fault-DISPLACEMENT and per-unit PROPERTY inversion only.
Surface-point / orientation geometry inversion is out of scope (that would
require ``surface_points``/``orientations`` coordinate tensors to be
trainable, which ``ImplicitModel`` does not wire up; see
``examples/03_geomodel/30_implicit_inversion.py`` for that different,
existing "invert surface-point depths directly" pattern, which does NOT go
through this adapter or ``Field(generator=)`` at all).

Also out of scope: **multi-series** models. ``scalar_field_to_property``
operates on ONE series' scalar field / iso-values; combining several series'
property fields would need the same ERODE/ONLAP masking
``ImplicitModel._stack_erode``/``_stack_onlap`` apply to lithology BLOCK ids,
generalized to property VALUES, a real feature, just not one either P4b
example needs (both are single-series + a fault), so it is left as a
documented follow-up rather than implemented speculatively.
:class:`ImplicitFieldAdapter` raises at construction if
``len(implicit_model.series) != 1``.

Grid ordering (the ImplicitModel side of the "reuse P4a's semantics if
GSLIB-ordered, else handle + document" instruction): ``ImplicitModelConfig
.make_grid()`` builds its evaluation grid via ``torch.meshgrid(*per_axis_
linspace, indexing="ij")`` then a plain ``.reshape(-1)``, i.e. a **C-order**
flatten of a ``config.resolution``-shaped array (config axis 0 slowest, the
LAST config axis fastest), NOT GSLIB's F-order/x-fastest convention a
:class:`~geobrain.geomodel.GeoGrid` uses. So the flat ``(Q,)`` outputs of
``ImplicitModel.forward()`` reshape directly (no ``order="F"`` reshape, no
GeoGrid involved) via ``flat.reshape(config.resolution)``, that alone
recovers the ``(nx, ny[, nz])``-shaped array (see
``examples/03_geomodel/29_implicit_modeling.py``'s ``to_image`` helper and
``examples/03_geomodel/30_implicit_inversion.py``'s ``block.reshape(NX,
NZ)``, which rely on exactly this). ``axis_map`` (for each MESH axis, which
CONFIG axis feeds it: the same "target-axis-order" convention
:func:`~geobrain.geomodel.bridge.geogrid_to_mesh_array` uses) then permutes
that into ``mesh.shape``.

Unlike :func:`~geobrain.geomodel.bridge.geogrid_to_mesh_array`'s 2-D case
(which REQUIRES an explicit ``axis_map`` because a 2-D ``GeoGrid`` is a
HORIZONTAL slice while a 2-D ``TensorMesh`` is a VERTICAL section, two
genuinely different planes with no safe default), an ``ImplicitModelConfig``
has no such structural ambiguity: it only ever has ``config.ndim`` axes,
period, and this codebase's own 2-D implicit examples (29, 30 above) already
establish the convention "the LAST ``resolution`` axis is depth" (they build
``resolution=(NX, NZ)`` and transpose for display). ``axis_map`` therefore
DOES have a default here, ``(ndim - 1, 0, 1, ..., ndim - 2)``, i.e. the
config's last axis feeds the mesh's depth axis (``nz``) and the rest fill the
remaining mesh axes in order (for ``ndim=3`` that is ``(2, 0, 1)``, the exact
literal default :func:`geogrid_to_mesh_array` uses for its 3-D case), while
still accepting an explicit override for a deliberately different layout.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from ..core.errors import GeoBrainError
from .implicit.model import ImplicitModel

__all__ = ["ImplicitFieldAdapter"]


def _default_axis_map(ndim: int) -> tuple[int, ...]:
    """The config's last axis is depth (mesh's ``nz``); earlier axes fill the
    remaining mesh axes in order; see the module docstring's "Grid
    ordering" section."""
    return (ndim - 1,) + tuple(range(ndim - 1))


def _validate_axis_map(axis_map: Sequence[int], *, ndim: int) -> tuple[int, ...]:
    try:
        entries = tuple(int(a) for a in axis_map)
    except (TypeError, ValueError) as exc:
        raise GeoBrainError(
            "axis_map must be a sequence of config-grid axis indices",
            object_name="ImplicitFieldAdapter", field="axis_map",
            expected=f"length-{ndim} permutation of range({ndim})",
            actual=axis_map,
        ) from exc
    if len(entries) != ndim or set(entries) != set(range(ndim)):
        raise GeoBrainError(
            "axis_map must be a permutation of the config's axis indices "
            f"(0..{ndim - 1})",
            object_name="ImplicitFieldAdapter", field="axis_map",
            expected=f"length-{ndim} permutation of range({ndim})",
            actual=entries,
        )
    return entries


class ImplicitFieldAdapter(nn.Module):
    """Bridges an :class:`~geobrain.geomodel.implicit.ImplicitModel` into a
    :class:`~geobrain.geomodel.earthmodel.Field` ``generator=``; see the module docstring
    for the full contract, scope, and grid-ordering notes.

    Args:
        implicit_model: A single-series :class:`~geobrain.geomodel.implicit
            .ImplicitModel` (multi-series is out of scope; see the module
            docstring). Its ``fault_displacements`` (if any faults were
            given) become trainable leaves via ordinary ``nn.Module``
            submodule registration.
        property_values: ``(n_surfaces + 1,)`` tensor of per-lithology-unit
            physical values, ordered bottom -> top (the same convention as
            ``CokrigingInterpolator.scalar_field_to_property``). Must already
            sit on ``implicit_model.config``'s dtype/device. Registered as a
            trainable ``nn.Parameter`` when ``property_values.requires_grad``
            is ``True``, otherwise as a frozen buffer.
        mesh: The target mesh (duck-typed: reads ``.shape``). Its
            ``len(mesh.shape)`` must equal ``implicit_model.config.ndim``.
        axis_map: For each MESH axis (in mesh-axis order), which CONFIG axis
            (``0..ndim-1``) supplies it. ``None`` (default) uses
            ``(ndim-1, 0, ..., ndim-2)``: see the module docstring.
        soft: Forwarded to ``implicit_model.forward`` every call.
        temperature: Forwarded to BOTH ``implicit_model.forward`` and
            ``scalar_field_to_property`` every call.

    Raises:
        GeoBrainError: ``implicit_model`` is not an ``ImplicitModel``; it
            does not have exactly one series; ``property_values`` is not a
            correctly-shaped/typed tensor; ``mesh``/``axis_map`` are
            incompatible with ``implicit_model.config``; or, checked LAST,
            after every other validation; the resulting module has ZERO
            trainable parameters (no faults and a non-trainable
            ``property_values``), which would silently make this
            ``Field(generator=...)`` a constant with nothing for an
            inversion to update.
    """

    def __init__(
        self,
        implicit_model: ImplicitModel,
        *,
        property_values: torch.Tensor,
        mesh: Any,
        axis_map: Sequence[int] | None = None,
        soft: bool = True,
        temperature: float = 50.0,
    ) -> None:
        super().__init__()
        if not isinstance(implicit_model, ImplicitModel):
            raise GeoBrainError(
                "implicit_model must be an ImplicitModel",
                object_name="ImplicitFieldAdapter", field="implicit_model",
                expected=ImplicitModel, actual=type(implicit_model),
            )
        if len(implicit_model.series) != 1:
            raise GeoBrainError(
                "ImplicitFieldAdapter requires an ImplicitModel with exactly "
                "one geological series, multi-series ERODE/ONLAP property "
                "combination is out of scope (see the module docstring)",
                object_name="ImplicitFieldAdapter", field="implicit_model.series",
                expected="len(series) == 1", actual=len(implicit_model.series),
            )

        cfg = implicit_model.config
        n_surfaces = int(
            implicit_model.series[0].surface_points.surface_id.max().item()
        ) + 1
        expected_prop_shape = (n_surfaces + 1,)

        if not isinstance(property_values, torch.Tensor):
            raise GeoBrainError(
                "property_values must be a torch.Tensor",
                object_name="ImplicitFieldAdapter", field="property_values",
                expected=torch.Tensor, actual=type(property_values),
            )
        if tuple(property_values.shape) != expected_prop_shape:
            raise GeoBrainError(
                "property_values shape does not match the series' lithology "
                "unit count (n_surfaces + 1)",
                object_name="ImplicitFieldAdapter", field="property_values",
                expected=expected_prop_shape, actual=tuple(property_values.shape),
            )
        if property_values.dtype != cfg.torch_dtype or property_values.device != cfg.torch_device:
            raise GeoBrainError(
                "property_values dtype/device must match implicit_model"
                ".config's (torch_dtype/torch_device), cast it yourself "
                "before construction so trainable leaves stay real leaves "
                "(a .to() on a requires_grad tensor returns a non-leaf)",
                object_name="ImplicitFieldAdapter", field="property_values",
                expected=(cfg.torch_dtype, cfg.torch_device),
                actual=(property_values.dtype, property_values.device),
            )

        # implicit_model is registered as a submodule: its own
        # ``fault_displacements`` ParameterList (possibly empty) becomes part
        # of this adapter's parameter tree with no special-casing.
        self.implicit_model = implicit_model

        if property_values.requires_grad:
            self.property_values = nn.Parameter(property_values)
        else:
            self.register_buffer("property_values", property_values)

        mesh_shape = tuple(int(s) for s in getattr(mesh, "shape", ()))
        ndim = cfg.ndim
        if len(mesh_shape) != ndim:
            raise GeoBrainError(
                "mesh.shape rank must match implicit_model.config.ndim",
                object_name="ImplicitFieldAdapter", field="mesh",
                expected=f"len(mesh.shape) == {ndim}", actual=mesh_shape,
            )
        if axis_map is None:
            axis_map = _default_axis_map(ndim)
        axis_map = _validate_axis_map(axis_map, ndim=ndim)
        resolution = tuple(int(r) for r in cfg.resolution)
        expected_mesh_shape = tuple(resolution[a] for a in axis_map)
        if expected_mesh_shape != mesh_shape:
            raise GeoBrainError(
                "mesh.shape is not compatible with implicit_model.config"
                ".resolution under axis_map",
                object_name="ImplicitFieldAdapter", field="mesh",
                expected=expected_mesh_shape, actual=mesh_shape,
            )

        self._axis_map = axis_map
        self._resolution = resolution
        self._mesh_shape = mesh_shape
        self.soft = bool(soft)
        self.temperature = float(temperature)

        if next(self.parameters(), None) is None:
            raise GeoBrainError(
                "ImplicitFieldAdapter has zero trainable parameters; pass "
                "an ImplicitModel with at least one FaultDefinition (so "
                "fault_displacements is non-empty) and/or property_values "
                "with requires_grad=True; otherwise there is nothing for an "
                "inversion to update through this Field(generator=...) seam",
                object_name="ImplicitFieldAdapter",
                field="implicit_model/property_values",
                expected="faults and/or trainable property_values",
                actual="zero parameters",
            )

    def forward(self) -> torch.Tensor:
        """Run the implicit model, map its scalar field to a property field,
        and reshape into ``mesh.shape``. No input; this is the ``generator=
        net`` (no ``fixed_input``) form of the ``Field`` contract, matching
        ``ImplicitModel.forward`` itself taking no tensor input."""
        out = self.implicit_model(soft=self.soft, temperature=self.temperature)
        Z = out["scalar_fields"][0]
        iso_vals = out["iso_vals"][0]
        flat = self.implicit_model.interpolator.scalar_field_to_property(
            Z, iso_vals, self.property_values, temperature=self.temperature,
        )
        grid_arr = flat.reshape(self._resolution)
        permuted = grid_arr.permute(*self._axis_map)
        return permuted.contiguous()
