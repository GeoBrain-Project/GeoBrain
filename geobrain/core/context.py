"""
Typed per-call runtime configuration threaded through a forward call.

- :class:`ForwardContext`: the physics-agnostic top-level context: a typed
  namespace of optional sub-context layers (mesh/time/source/flow) plus an
  experimental-only ``extras`` overflow.
- :class:`TimeContext`: time-stepping control (wave/EM time-domain, flow
  transient).
- :class:`SourceContext`: source-side excitation (wave). Family-scoped.
- :class:`FlowContext`: flow controls (wells, observers). Family-scoped.

Split out of :mod:`geobrain.core.containers` so that file holds the frozen
*data* containers (:class:`ModelState` / :class:`ForwardOutput`) while the
runtime *context* layers, which have grown to four classes, a terse ``.of()``
builder, override/flatten machinery and reserved-key governance, live here.
The two modules are peers: both depend only on :mod:`geobrain.core.validation`
(shared immutability/validation helpers) and :mod:`geobrain.core.errors`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import torch

from .validation import freeze_mapping, validate_mapping_key
from .errors import GeoBrainError, MissingFieldError

if TYPE_CHECKING:
    from geobrain.mesh import Mesh


_RESERVED_CONTEXT_KEYS = frozenset(
    {
        "mesh",
        "time",
        "source",
        "flow",
        "extras",
        "dt",
        "t_end",
        "scheduler",
        "t0",
        "wavelet",
        "wells",
        "well_observer",
        "return_states",
    }
)


@dataclass(frozen=True, kw_only=True)
class TimeContext:
    """Per-call time-stepping control (wave/EM time-domain, flow transient).

    The time axis runs ``t0 + i*dt`` for ``i in 0..nt-1``, so ``t_end`` is the
    *last* sample (``t0 + (nt-1)*dt``), not ``t0 + nt*dt``.
    """
    dt: float | None = None
    t_end: float | None = None
    scheduler: Any | None = None
    t0: float = 0.0


@dataclass(frozen=True, kw_only=True)
class SourceContext:
    """Per-call source-side excitation (wave). Family-scoped."""
    wavelet: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class FlowContext:
    """Per-call flow controls. Family-scoped."""
    wells: Any | None = None
    well_observer: Any | None = None
    return_states: bool = False


@dataclass(frozen=True, kw_only=True)
class ForwardContext:
    """
    Per-call configuration. Physics-agnostic.

    A typed namespace of per-call runtime configuration the kernel needs. Each
    layer is an optional sub-context (``None`` when that family is not in play):

    - :attr:`mesh`: the :class:`~geobrain.mesh.Mesh` the kernel
      discretises on (top-level; read it with :meth:`require_mesh`).
    - :attr:`time`: :class:`TimeContext` (``dt`` / ``t_end`` / ``scheduler``);
      the cross-cutting time-stepping layer (wave/EM time-domain, flow
      transient).
    - :attr:`source`: :class:`SourceContext` (``wavelet``); family-scoped (wave).
    - :attr:`flow`: :class:`FlowContext` (``wells`` / ``well_observer`` /
      ``return_states``); family-scoped (flow).
    - :attr:`extras`: free-form, **experimental-only** overflow for keys that do
      not belong to a typed layer. Stable operators must NOT read it (a dedicated governance test
      enforces this).

    Construct with the terse keyword builder :meth:`of`, which routes leaves into
    the right layers::

        ctx = ForwardContext.of(mesh=m, dt=0.1, t_end=2.0, wells=w)

    Attributes:
        mesh: the mesh the kernel discretises on (top-level slot).
        time: optional :class:`TimeContext` layer.
        source: optional :class:`SourceContext` layer.
        flow: optional :class:`FlowContext` layer.
        extras: frozen experimental-only overflow mapping.
    """

    mesh: "Mesh | None" = None
    time: TimeContext | None = None
    source: SourceContext | None = None
    flow: FlowContext | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze ``extras`` into a read-only mapping."""
        for name in self.extras:
            validate_mapping_key("ForwardContext", "extras", name)
        reserved = sorted(set(self.extras) & _RESERVED_CONTEXT_KEYS)
        if reserved:
            raise GeoBrainError(
                "ForwardContext extras contain reserved typed keys",
                object_name="ForwardContext",
                field="extras",
                expected="non-reserved experimental keys",
                actual=reserved,
            )
        object.__setattr__(self, "extras", freeze_mapping(self.extras))

    @classmethod
    def of(
        cls,
        *,
        mesh: Any = None,
        dt: float | None = None,
        t_end: float | None = None,
        scheduler: Any = None,
        t0: float | None = None,
        wavelet: Any = None,
        wells: Any = None,
        well_observer: Any = None,
        return_states: bool | None = None,
        **extras: Any,
    ) -> "ForwardContext":
        """Terse keyword builder routing leaves into the typed layers."""
        if return_states is not None and not isinstance(return_states, bool):
            raise GeoBrainError(
                "ForwardContext return_states must be a bool",
                object_name="ForwardContext",
                field="return_states",
                expected=bool,
                actual=type(return_states),
            )
        time = (
            TimeContext(dt=dt, t_end=t_end, scheduler=scheduler,
                        t0=t0 if t0 is not None else 0.0)
            if (dt is not None or t_end is not None or scheduler is not None
                or t0 is not None)
            else None
        )
        source = SourceContext(wavelet=wavelet) if wavelet is not None else None
        flow = (
            FlowContext(
                wells=wells,
                well_observer=well_observer,
                return_states=return_states if return_states is not None else False,
            )
            if (wells is not None or well_observer is not None or return_states is not None)
            else None
        )
        return cls(mesh=mesh, time=time, source=source, flow=flow, extras=extras)

    def with_overrides(self, overlay: "Mapping[str, Any]") -> "ForwardContext":
        """Return a new context with ``overlay`` (canonical kwargs and/or extras)
        applied on top of this one. Used by OperatorBundle per-channel configs.

        Each key in the overlay resolves to one of three outcomes::

            key omitted from overlay   → inherited from this context, unchanged
            key present  with a value  → overridden with that value
            key present  with ``None`` → DELETED from the context (not set None)

        Note:
            The ``None``-deletes rule is deliberate: this context is flattened to
            ``of()``-style kwargs, the overlay is applied key-by-key, and the
            result is rebuilt via :meth:`of`. So
            ``with_overrides({"wavelet": None})`` drops the inherited wavelet
            entirely. The same rule applies to experimental ``extras`` keys. To
            leave a shared value unchanged, simply omit its key from the overlay
            rather than passing ``None``.
        """
        base = self._to_kwargs()
        for key, value in overlay.items():
            if value is None:
                base.pop(key, None)
            else:
                base[key] = value
        return type(self).of(**base)

    def _to_kwargs(self) -> dict:
        """Flatten back to of()-style kwargs (inverse of of()).

        The two defaulted leaves: ``t0 == 0.0`` and ``return_states is False``;
        are OMITTED rather than emitted unconditionally, so the flatten is
        idempotent and never materialises a phantom layer. Without this,
        ``with_overrides({"dt": None})`` on a dt-only context would leave a
        ``t0=0.0``-only TimeContext behind (and a return_states-only FlowContext
        likewise), because ``of()`` builds a layer whenever any of its leaves is
        non-``None``. Non-default values are always kept.
        """
        kw: dict[str, Any] = {}
        if self.mesh is not None:
            kw["mesh"] = self.mesh
        if self.time is not None:
            if self.time.dt is not None:
                kw["dt"] = self.time.dt
            if self.time.t_end is not None:
                kw["t_end"] = self.time.t_end
            if self.time.scheduler is not None:
                kw["scheduler"] = self.time.scheduler
            if self.time.t0 != 0.0:  # omit the default; keeps flatten idempotent
                kw["t0"] = self.time.t0
        if self.source is not None and self.source.wavelet is not None:
            kw["wavelet"] = self.source.wavelet
        if self.flow is not None:
            if self.flow.wells is not None:
                kw["wells"] = self.flow.wells
            if self.flow.well_observer is not None:
                kw["well_observer"] = self.flow.well_observer
            if self.flow.return_states:  # omit the default (False)
                kw["return_states"] = self.flow.return_states
        kw.update(dict(self.extras))
        return kw

    # ---- Strict leaf accessors (required reads) ----
    def require_mesh(self) -> "Mesh":
        if self.mesh is None:
            raise MissingFieldError("ForwardContext is missing a required mesh",
                                    object_name="ForwardContext", field="mesh",
                                    expected="a Mesh in ctx")
        return self.mesh

    def require_dt(self) -> float:
        if self.time is None or self.time.dt is None:
            raise MissingFieldError("ForwardContext is missing a required time.dt",
                                    object_name="ForwardContext", field="time.dt", expected="a dt")
        return self.time.dt

    def require_t_end(self) -> float:
        if self.time is None or self.time.t_end is None:
            raise MissingFieldError("ForwardContext is missing a required time.t_end",
                                    object_name="ForwardContext", field="time.t_end", expected="a t_end")
        return self.time.t_end

    def require_scheduler(self) -> Any:
        if self.time is None or self.time.scheduler is None:
            raise MissingFieldError("ForwardContext is missing a required time.scheduler",
                                    object_name="ForwardContext", field="time.scheduler", expected="a scheduler")
        return self.time.scheduler

    def require_wavelet(self) -> torch.Tensor:
        if self.source is None or self.source.wavelet is None:
            raise MissingFieldError("ForwardContext is missing a required source.wavelet",
                                    object_name="ForwardContext", field="source.wavelet", expected="a wavelet")
        return self.source.wavelet

    def require_wells(self) -> Any:
        if self.flow is None or self.flow.wells is None:
            raise MissingFieldError("ForwardContext is missing a required flow.wells",
                                    object_name="ForwardContext", field="flow.wells", expected="wells")
        return self.flow.wells

    # ---- Layer accessors (raise if the whole layer is absent) ----
    def require_time(self) -> "TimeContext":
        if self.time is None:
            raise MissingFieldError("ForwardContext is missing the time layer",
                                    object_name="ForwardContext", field="time", expected="time config")
        return self.time

    def require_source(self) -> "SourceContext":
        if self.source is None:
            raise MissingFieldError("ForwardContext is missing the source layer",
                                    object_name="ForwardContext", field="source", expected="source config")
        return self.source

    def require_flow(self) -> "FlowContext":
        if self.flow is None:
            raise MissingFieldError("ForwardContext is missing the flow layer",
                                    object_name="ForwardContext", field="flow", expected="flow config")
        return self.flow
