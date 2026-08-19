"""Wave-equation abstraction: the *math* layer.

A :class:`WaveEquation` owns everything about a single time step: which wavefield
tensors exist and where they live on the staggered grid (``FieldSpec``), which
model parameters it consumes (``ModelSpec``), how to turn raw model parameters
into per-step coefficients (:meth:`prepare`), and how to advance the state by one
step (:meth:`step`). It knows nothing about the number of time steps, sources,
receivers, or memory strategy; those belong to the propagator (the *engineering*
layer). This uses an Equation/Propagator split, so adding a new physics
(elastic, anisotropic) means writing a new ``step`` rather than a new time loop.

The state is carried as an ordered ``dict[str, Tensor]`` (insertion order =
``state_names``), so the propagator can flatten it to a positional tuple for
``torch.utils.checkpoint`` without knowing the physics.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, ClassVar, Literal, Mapping, Protocol, Sequence, cast

import torch
from torch import Tensor

from ..boundaries.cpml import CPML
from ..contracts import WaveEquationDeclaration

Stagger = Literal["00", "x", "y", "z"]


@dataclass(frozen=True)
class FieldSpec:
    """A wavefield tensor declared by an equation.

    Args:
        name: state key (e.g. ``"p"``, ``"vx"``).
        stagger: node location, ``"00"`` integer, ``"x"`` half in x,
            ``"z"`` half in z.
        is_memory: True for CPML memory variables (not sourced/recorded, reset
            to zero each shot, and reconstructible rather than stored).
    """

    name: str
    stagger: Stagger = "00"
    is_memory: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """A model parameter consumed by an equation (e.g. ``"vp"``, ``"rho"``)."""

    name: str


class _DimensionSpecificEquation(Protocol):
    """Narrow concrete hooks used by the dimension-neutral recording substrate."""

    source_field: str
    snapshot_field: str
    default_receiver_field: str
    fd_order: int

    def max_velocity(self, models: Mapping[str, Tensor]) -> float: ...

    def cfl_dt_max(self, maximum_velocity: float, *spacing: float) -> float: ...

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, *spacing: float
    ) -> Mapping[str, Tensor]: ...

    def init_state(self, n_shot: int, *shape_device_dtype: object) -> Sequence[Tensor]: ...

    def step(
        self,
        state: Sequence[Tensor],
        coefficients: Mapping[str, Tensor],
        boundary: object,
        dt: float,
        *spacing: float,
    ) -> Sequence[Tensor]: ...

    def add_source(
        self, state: Sequence[Tensor], *coordinates_amplitudes_dt: object
    ) -> Sequence[Tensor]: ...


class ReceiverRecording:
    """Dimension-agnostic equation substrate, shared by the 2-D
    :class:`WaveEquation` and the standalone 3-D equation classes.

    Owns the two pieces of an equation that do **not** depend on the spatial
    dimension:

    - **Field/model introspection** declared by ``FIELD_SPECS`` / ``MODEL_SPECS``
      (``state_names``, ``wavefield_names``, ``memory_names``, ``model_names``,
      ``field_index``). CPML memory variables are flagged declaratively with
      ``FieldSpec(is_memory=True)``, so both the 2-D and 3-D equations declare
      their fields the same way, and neither needs a ``name.startswith("psi")``
      heuristic to tell memory variables apart.
    - **Receiver recording**: the primary trace, optional extra raw components
      on a trailing axis (multi-component gathers), and the illumination hook.
      Receiver coordinates are passed positionally (``(z, x)`` in 2-D,
      ``(z, y, x)`` in 3-D) so one implementation serves both.

    The dimension-specific physics (``prepare`` / ``step`` / ``add_source`` /
    ``init_state`` / ``cfl_dt_max``) stays on the concrete classes; those
    signatures are intrinsically 2-D vs 3-D (``dx, dz`` vs ``dx, dy, dz``;
    ``iz, ix`` vs ``iz, iy, ix``), so there is nothing to share there.
    """

    FIELD_SPECS: ClassVar[Sequence[FieldSpec]] = ()
    MODEL_SPECS: ClassVar[Sequence[ModelSpec]] = ()

    # Extra raw wavefields appended to the receiver gather (e.g. ``("vx","vz")``
    # for multi-component output). Class-level default → no extra components.
    _extra_components: tuple[str, ...] = ()
    DIMENSION: ClassVar[int] = 2
    SOURCE_FIELDS: ClassVar[tuple[str, ...]] = ()

    @property
    def declaration(self) -> WaveEquationDeclaration:
        """Return the exact immutable contract consumed by propagation backends."""
        identifiers = {
            "AcousticVelocityStress": "acoustic-2d",
            "AcousticVelocityStress3D": "acoustic-3d",
            "ElasticVelocityStress": "elastic-2d",
            "ElasticVelocityStress3D": "elastic-3d",
            "ElasticVTI": "elastic-vti-2d",
            "ElasticTTI": "elastic-tti-2d",
            "ElasticVTI3D": "elastic-vti-3d",
            "ElasticTTI3D": "elastic-tti-3d",
            "ViscoAcousticVelocityStress": "viscoacoustic-2d",
            "ViscoElasticVelocityStress": "viscoelastic-2d",
        }
        units = {
            "vp": "m/s",
            "vs": "m/s",
            "rho": "kg/m^3",
            "Q": "1",
            "Qp": "1",
            "Qs": "1",
            "epsilon": "1",
            "delta": "1",
            "gamma": "1",
            "theta": "rad",
        }
        wavefields = self.wavefield_names
        raw_components = tuple(name for name in wavefields if name != "p")
        return WaveEquationDeclaration(
            identifier=identifiers.get(
                type(self).__name__, type(self).__name__.lower()
            ),
            dimension=self.DIMENSION,
            required_model_fields=self.model_names,
            model_units=MappingProxyType(
                {name: units.get(name, "1") for name in self.model_names}
            ),
            state_fields=wavefields,
            cpml_fields=self.memory_names,
            declared_components=("pressure", *raw_components),
            source_component="pressure",
            source_injection="additive",
        )

    # --- declarative field / model introspection (dimension-agnostic) -----
    @property
    def state_names(self) -> tuple[str, ...]:
        """All field names in canonical order (wavefields then memory vars)."""
        return tuple(f.name for f in self.FIELD_SPECS)

    @property
    def wavefield_names(self) -> tuple[str, ...]:
        """Sourced/recorded fields: the non-memory ``FieldSpec``s."""
        return tuple(f.name for f in self.FIELD_SPECS if not f.is_memory)

    @property
    def memory_names(self) -> tuple[str, ...]:
        """CPML memory variables: the ``FieldSpec(is_memory=True)`` fields."""
        return tuple(f.name for f in self.FIELD_SPECS if f.is_memory)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.MODEL_SPECS)

    def field_index(self, name: str) -> int:
        """Position of ``name`` within ``state_names``."""
        return self.state_names.index(name)

    @property
    def source_fields(self) -> tuple[str, ...]:
        """State fields driven by one physical source amplitude."""
        equation = cast(_DimensionSpecificEquation, self)
        return self.SOURCE_FIELDS or (equation.source_field,)

    def source_scale(self, dt: float) -> float:
        """Return the equation's multiplicative source scale."""
        return dt

    def cfl_limit(
        self, model: Mapping[str, Tensor], spacing: tuple[float, ...]
    ) -> float:
        """Evaluate the concrete equation CFL formula in platform axis order."""
        equation = cast(_DimensionSpecificEquation, self)
        return float(
            equation.cfl_dt_max(
                equation.max_velocity(model), *tuple(reversed(spacing))
            )
        )

    def prepare_model(
        self,
        model: Mapping[str, Tensor],
        *,
        dt: float,
        spacing: tuple[float, ...],
    ) -> Mapping[str, Tensor]:
        """Prepare coefficients without exposing dimension-specific arguments."""
        equation = cast(_DimensionSpecificEquation, self)
        return equation.prepare(model, dt, *tuple(reversed(spacing)))

    def initialize_state(
        self,
        n_shot: int,
        mesh_shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, ...]:
        """Allocate the equation state through one dimension-neutral method."""
        equation = cast(_DimensionSpecificEquation, self)
        return tuple(equation.init_state(n_shot, *mesh_shape, device, dtype))

    def advance(
        self,
        state: Sequence[Tensor],
        coefficients: Mapping[str, Tensor],
        *,
        boundary: object,
        dt: float,
        spacing: tuple[float, ...],
    ) -> tuple[Tensor, ...]:
        """Advance one step while retaining the concrete stencil operation order."""
        equation = cast(_DimensionSpecificEquation, self)
        return tuple(
            equation.step(
                state,
                coefficients,
                boundary,
                dt,
                *tuple(reversed(spacing)),
            )
        )

    def inject_sources(
        self,
        state: Sequence[Tensor],
        source_indices: Tensor,
        source_shot_index: Tensor,
        amplitudes: Tensor,
        *,
        dt: float,
    ) -> tuple[Tensor, ...]:
        """Add all packed sources with one scatter per driven state field."""
        equation = cast(_DimensionSpecificEquation, self)
        field_device = state[0].device
        shot_index = source_shot_index.to(device=field_device)
        coordinates = tuple(
            source_indices[:, axis] for axis in range(source_indices.shape[1])
        )
        n_shot = int(state[0].shape[0])
        if source_indices.shape[0] == n_shot and torch.equal(
            shot_index,
            torch.arange(n_shot, device=field_device, dtype=torch.int64),
        ):
            return tuple(equation.add_source(state, *coordinates, amplitudes, dt))
        channel = torch.zeros_like(shot_index, device=field_device)
        values = self.source_scale(dt) * amplitudes
        updated = list(state)
        for name in self.source_fields:
            index = self.field_index(name)
            updated[index] = torch.index_put(
                updated[index],
                (shot_index, channel, *coordinates),
                values,
                accumulate=True,
            )
        return tuple(updated)

    def sample_receivers(
        self,
        state: Sequence[Tensor],
        receiver_indices: Tensor,
        receiver_shot_index: Tensor,
        components: tuple[str, ...],
    ) -> Mapping[str, Tensor]:
        """Sample packed receiver traces using authoritative receiver shot IDs."""
        coordinates = tuple(
            receiver_indices[:, axis] for axis in range(receiver_indices.shape[1])
        )
        shot_index = receiver_shot_index.to(device=receiver_indices.device)
        n_shot = int(state[0].shape[0])
        n_trace = int(receiver_indices.shape[0])
        if (
            n_trace % n_shot == 0
            and components == ("pressure", *self._extra_components)
        ):
            receivers_per_shot = n_trace // n_shot
            expected_shots = torch.arange(
                n_shot, dtype=torch.int64
            ).repeat_interleave(receivers_per_shot)
            base = receiver_indices[:receivers_per_shot]
            expected_indices = base.repeat((n_shot, 1))
            if torch.equal(receiver_shot_index, expected_shots) and torch.equal(
                receiver_indices, expected_indices
            ):
                dense = self.record_receivers(
                    state,
                    *(
                        coordinate[:receivers_per_shot]
                        for coordinate in coordinates
                    ),
                )
                if not self._extra_components:
                    return {"pressure": dense.reshape(-1)}
                return {
                    "pressure": dense[..., 0].reshape(-1),
                    **{
                        name: dense[..., index + 1].reshape(-1)
                        for index, name in enumerate(self._extra_components)
                    },
                }
        channel = torch.zeros_like(shot_index, device=receiver_indices.device)
        result: dict[str, Tensor] = {}
        for component in components:
            if component == "pressure":
                field = self.primary_wavefield(state)
            else:
                field = state[self.field_index(component)]
            result[component] = field[(shot_index, channel, *coordinates)]
        return result

    def snapshot_fields(
        self, state: Sequence[Tensor]
    ) -> Mapping[str, Tensor]:
        """Expose the final diagnostic field without altering its live tensor."""
        equation = cast(_DimensionSpecificEquation, self)
        name = getattr(self, "snapshot_field", equation.source_field)
        return {"wavefield": state[self.field_index(name)]}

    @property
    def halo_width(self) -> int:
        """Stencil reach (cells) of one interior update: sets the boundary-saving
        rim width. Defaults to the FD half-order; equations whose update reaches
        farther (e.g. TTI's averaged cross terms, +1) must widen it. Lives on the
        dimension-agnostic substrate so the 2-D and 3-D boundary-saving adjoints
        size their rims from the same declaration."""
        equation = cast(_DimensionSpecificEquation, self)
        return equation.fd_order // 2

    def _record_primary(self, state: Sequence[Tensor], *coords: Tensor) -> Tensor:
        """Sample the primary receiver trace → ``(batch, n_rcv)``.

        Defaults to ``default_receiver_field``; override to record a derived
        quantity (e.g. pressure ``-(sxx+szz)/2``). This is the single-component
        gather; multi-component output is assembled by :meth:`record_receivers`.
        """
        return self.primary_wavefield(state)[(slice(None), 0, *coords)]

    def primary_wavefield(self, state: Sequence[Tensor]) -> Tensor:
        """Return the full field whose packed samples define pressure traces."""
        equation = cast(_DimensionSpecificEquation, self)
        idx = self.field_index(equation.default_receiver_field)
        return state[idx]

    def record_field(
        self, state: Sequence[Tensor], field: str, *coords: Tensor
    ) -> Tensor:
        """Sample raw state field ``field`` at the receivers → ``(batch, n_rcv)``.

        Unlike :meth:`_record_primary` (which may return a derived quantity),
        this samples the named wavefield directly, used for multi-component
        gathers such as particle-velocity ``vx``/``vz`` traces.
        """
        idx = self.field_index(field)
        return state[idx][(slice(None), 0, *coords)]

    def set_receiver_components(self, extra: Sequence[str] = ()) -> None:
        """Append raw wavefields ``extra`` to the receiver gather.

        With ``extra=("vx", "vz")`` the gather carries a trailing component axis
        ``[primary, vx, vz]``; with ``()`` it is the single primary trace. The
        extra axis flows through the propagator's ``stack``/``cat`` recording
        plumbing unchanged, so multi-component recording works for the ``full``
        and ``checkpoint`` memory strategies (standard autograd). The
        ``boundary`` strategy's custom-VJP records only the primary trace.
        """
        for f in extra:
            if f not in self.wavefield_names:
                raise GeoBrainError(
                    f"receiver component {f!r} is not a wavefield of "
                    f"{type(self).__name__}: {self.wavefield_names}"
                )
        self._extra_components = tuple(extra)

    def record_receivers(self, state: Sequence[Tensor], *coords: Tensor) -> Tensor:
        """Sample the receiver gather → ``(batch, n_rcv)`` or, when extra
        components are configured via :meth:`set_receiver_components`,
        ``(batch, n_rcv, 1 + n_extra)`` stacked as ``[primary, *extra]``.
        """
        primary = self._record_primary(state, *coords)
        if not self._extra_components:
            return primary
        extras = [self.record_field(state, f, *coords)
                  for f in self._extra_components]
        return torch.stack([primary, *extras], dim=-1)

    def min_velocity(self, models: Mapping[str, Tensor]) -> float:
        """Slowest characteristic speed (shortest wavelength) from padded models,
        the binding constraint for the dispersion / points-per-wavelength check.

        Prefers ``vs`` (shear waves resolve worst), then ``vp``; falls back to the
        smallest model field. Dimension-agnostic (2-D and 3-D); it only reads the
        model dict. Anisotropic equations may override for a direction-aware speed."""
        for name in ("vs", "vp"):
            if name in models:
                return float(models[name].detach().abs().min())
        return min(float(m.detach().abs().min()) for m in models.values())

    def illumination_fields(self, state: Sequence[Tensor]) -> dict[str, Tensor]:
        """Full-domain wavefields whose time/shot-summed square gives the
        illumination maps, keyed by the public ``forward_wavefield_<name>``
        suffix. Defaults to every wavefield; override to expose a derived
        quantity (e.g. elastic pressure ``-(sxx+szz)/2`` as ``"p"``)."""
        return {n: state[self.field_index(n)] for n in self.wavefield_names}


class WaveEquation(ReceiverRecording, ABC):
    """Base class for one-step 2-D wave-equation updates.

    Subclasses declare their fields/params via ``FIELD_SPECS`` / ``MODEL_SPECS``
    (the declarative introspection lives on :class:`ReceiverRecording`, shared
    with the 3-D equations) and implement the 2-D physics, :meth:`prepare`,
    :meth:`step`, :meth:`cfl_dt_max`.
    """

    def __init__(
        self, fd_order: int = 8, *, normalize_source_by_cell_volume: bool = False
    ) -> None:
        if fd_order < 2 or fd_order % 2 != 0:
            raise GeoBrainError(f"fd_order must be a positive even integer: {fd_order}")
        self.fd_order = fd_order
        # Top-boundary condition, set by the propagator via
        # :meth:`configure_free_surface`. Default = no free surface, so the update
        # is byte-identical to the plain (CPML/pad) top-edge path.
        self._free_surface = False
        self._surface_row = 0
        # Opt-in cell-volume source normalization (see :meth:`add_source`). Default
        # False keeps the injection byte-identical to the historical ``dt*amp``.
        self._normalize_source_by_cell_volume = bool(normalize_source_by_cell_volume)
        self._source_cell_volume = 1.0  # set by the propagator (dx*dz in 2-D)

    def configure_free_surface(self, free_surface: bool, surface_row: int = 0) -> None:
        """Select the top-face boundary condition for subsequent runs.

        Called by the propagator at construction. ``free_surface=True`` switches
        the surface-row update to the pressure-release / traction-free *image*
        (``R = -1``); ``False`` (default) leaves the top edge to the CPML/pad and
        keeps the step byte-identical to the no-free-surface path. Equations that
        do not model a free surface simply carry the flag unused. ``surface_row``
        is the padded z-index of the surface (0 under a free surface, whose top
        face is not padded)."""
        self._free_surface = bool(free_surface)
        self._surface_row = int(surface_row)

    def configure_source_normalization(self, cell_volume: float) -> None:
        """Record the physical cell volume (``dx*dz`` in 2-D) the propagator uses.

        Only consulted when ``normalize_source_by_cell_volume=True`` (see
        :meth:`add_source`); harmless otherwise. Called by the propagator at
        construction so :meth:`add_source` can divide the injected amplitude by
        the cell volume for grid-independent absolute amplitude."""
        self._source_cell_volume = float(cell_volume)

    def source_scale(self, dt: float) -> float:
        """Preserve the opt-in 2-D cell-volume source normalization."""
        return (
            dt / self._source_cell_volume
            if self._normalize_source_by_cell_volume
            else dt
        )

    @property
    @abstractmethod
    def source_field(self) -> str:
        """State key the source term is injected into (used by the defaults)."""

    @property
    @abstractmethod
    def default_receiver_field(self) -> str:
        """State key sampled at receivers by default."""

    @property
    def snapshot_field(self) -> str:
        """State key returned as the diagnostic wavefield snapshot."""
        return self.source_field

    # --- source / receiver (override to customise per equation) -----------
    def add_source(
        self,
        state: Sequence[Tensor],
        src_z: Tensor,
        src_x: Tensor,
        amp: Tensor,
        dt: float,
    ) -> list[Tensor]:
        """Inject ``dt·amp`` (one value per shot) into ``source_field``.

        Out-of-place so the loop stays checkpoint-/autograd-safe. Override for
        multi-component sources (e.g. an explosive source drives ``sxx`` and
        ``szz`` together).

        .. note::
           **The injected amplitude is grid-dependent.** A point source deposits
           ``dt·amp`` into a single cell, so the resulting field amplitude scales
           with the cell size: halving ``dx``/``dz`` (a 4× finer 2-D cell)
           quarters the recorded amplitude for the *same* physical setup (the
           relative waveform / kinematics are unchanged; only the absolute scale
           moves). For multiscale or absolute-amplitude work either normalize the
           gathers externally, or construct the equation with
           ``normalize_source_by_cell_volume=True`` to divide the injected
           amplitude by the cell volume (``dx·dz`` in 2-D) so the absolute
           amplitude is grid-independent. The flag defaults to ``False`` so every
           existing (pinned) amplitude is byte-identical.
        """
        idx = self.field_index(self.source_field)
        new = list(state)
        factor = (dt / self._source_cell_volume
                  if self._normalize_source_by_cell_volume else dt)
        new[idx] = _index_add(new[idx], src_z, src_x, factor * amp)
        return new

    def add_source_multi(
        self,
        state: Sequence[Tensor],
        iz: Tensor,
        ix: Tensor,
        amp: Tensor,
        dt: float,
    ) -> list[Tensor]:
        """Inject many sources into every batch element (source encoding).

        Args:
            iz, ix: ``(n_pos,)`` padded source coordinates.
            amp: ``(batch, n_pos)`` per-position amplitude this step.

        Injects ``dt·amp[b, k]`` at ``(iz[k], ix[k])`` of batch ``b`` of
        ``source_field``. Override for multi-component sources. Honours the same
        opt-in ``normalize_source_by_cell_volume`` cell-volume normalization as
        :meth:`add_source` (default off → byte-identical ``dt·amp``).
        """
        idx = self.field_index(self.source_field)
        new = list(state)
        factor = (dt / self._source_cell_volume
                  if self._normalize_source_by_cell_volume else dt)
        new[idx] = _index_add_multi(new[idx], iz, ix, factor * amp)
        return new

    # Receiver recording (primary trace, multi-component gather, illumination)
    # is provided by the :class:`ReceiverRecording` mixin.

    # --- physics ----------------------------------------------------------
    @abstractmethod
    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        """Turn padded model tensors into per-step coefficient tensors."""

    @abstractmethod
    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML,
        dt: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        """Advance ``state`` (in ``state_names`` order) by one step.

        Returns the new state as a tuple in the same order. Must be a pure
        function (no in-place mutation of inputs) so it is checkpoint-safe.
        """

    @abstractmethod
    def cfl_dt_max(self, vmax: float, dx: float, dz: float) -> float:
        """Largest stable ``dt`` for this scheme at the given ``vmax``/spacing."""

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        """Maximum propagation speed (for the CFL check) from padded models.

        Defaults to ``max(vp)`` if present, else the max over all model fields.
        Anisotropic equations override this (e.g. horizontal P velocity).
        """
        if "vp" in models:
            return float(models["vp"].detach().max())
        return max(float(m.detach().max()) for m in models.values())

    def inverse_step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        dt: float,
        dx: float,
        dz: float,
        set_rim: Callable[[Tensor, str], Tensor],
    ) -> list[Tensor]:
        """Exact inverse of one *plain* (CPML-identity) step, for boundary-saving.

        Reconstructs the previous wavefields from the current ones. ``set_rim(f,
        name)`` returns ``f`` with its rim cells overwritten by the saved truth;
        the equation calls it on each reconstructed wavefield immediately, before
        that field is read downstream. Memory variables are returned as zeros
        (CPML is the identity in the physical interior). Equations that support
        boundary-saving override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement inverse_step "
            f"(boundary-saving unavailable for this equation)"
        )

    def init_state(
        self,
        batch: int,
        nz: int,
        nx: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        """Allocate zeroed state tensors (collocated storage; staggering is
        encoded by the ``D+`` / ``D−`` operators, not by array size)."""
        return [
            torch.zeros((batch, 1, nz, nx), device=device, dtype=dtype)
            for _ in self.FIELD_SPECS
        ]


def _index_add(field: Tensor, iz: Tensor, ix: Tensor, values: Tensor) -> Tensor:
    """Out-of-place ``field[b, 0, iz[b], ix[b]] += values[b]`` (one src per shot)."""
    n = field.shape[0]
    batch = torch.arange(n, device=field.device)
    chan = torch.zeros(n, dtype=torch.long, device=field.device)
    return torch.index_put(field, (batch, chan, iz, ix), values, accumulate=True)


def _index_add_multi(field: Tensor, iz: Tensor, ix: Tensor, values: Tensor) -> Tensor:
    """Out-of-place ``field[b, 0, iz[k], ix[k]] += values[b, k]`` for all ``b, k``.

    ``iz``/``ix`` are ``(n_pos,)``; ``values`` is ``(batch, n_pos)``. Used for
    source encoding, where every batch element is a super-shot containing all
    source positions.
    """
    b, n_pos = values.shape
    batch = torch.arange(b, device=field.device).repeat_interleave(n_pos)
    zidx = iz.repeat(b)
    xidx = ix.repeat(b)
    chan = torch.zeros(b * n_pos, dtype=torch.long, device=field.device)
    return torch.index_put(
        field, (batch, chan, zidx, xidx), values.reshape(-1), accumulate=True
    )
