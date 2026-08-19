"""
Differentiability contract for operators.

An operator declares (a) what gradient path it supports and (b) which inputs are
trainable. This is a *contract*, not a dispatch switch: the backward mechanism is
baked into each operator's own ``_forward`` (plain ``torch`` ops, an implicit-VJP
linear solve, or a custom ``autograd.Function``). Solvers never branch on ``level``
to choose a backward; they read it only to validate the contract:
``supports_gradients`` gates whether trainable inputs are allowed, and a composed
chain takes the weakest ``level`` of its links.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .errors import GeoBrainError


class DifferentiabilityLevel(Enum):
    """
    What gradient mechanism the operator supports.

    - ``NON_DIFFERENTIABLE``: no gradients (e.g. MCMC samplers).
    - ``FORWARD_ONLY``: gradients explicitly disabled for this instance.
    - ``IMPLICIT_VJP``: gradient via the implicit function theorem (linear
      solves: EM, Helmholtz).
    - ``CUSTOM_VJP``: operator provides its own hand-written backward that is
      *not* the implicit-solve adjoint, e.g. the boundary-saving reverse-time
      wavefield reconstruction (``physics/wave/_engine/memory/boundary.py``).
      (Note: ``torch.utils.checkpoint`` is *not* a custom VJP; it is a
      transparent autograd recompute, so checkpointed FDTD, whether uniform
      (``"checkpoint"``) or recursive treeverse-style nested bisection
      (``"recursive"``); stays ``FULL_AUTOGRAD``.)
    - ``FULL_AUTOGRAD``: gradients flow through ``torch.autograd`` (default for
      wave / rock).
    """

    NON_DIFFERENTIABLE = "non_differentiable"
    FORWARD_ONLY = "forward_only"
    IMPLICIT_VJP = "implicit_vjp"
    CUSTOM_VJP = "custom_vjp"
    FULL_AUTOGRAD = "full_autograd"


@dataclass(frozen=True)
class DifferentiabilitySpec:
    """
    Operator's differentiability declaration.

    Attributes:
        level: Which gradient mechanism the operator supports.
        trainable_inputs: ModelState tensor names this operator is differentiable
            w.r.t. AND requires present at forward, capability metadata, NOT a
            request to "train these" (parameter selection is explicit at the
            ``Inverter(params=...)`` seam; an ``OperatorChain`` does not merge or
            act on non-first members' declarations). Presence and float/complex
            dtype are enforced per call by ``Operator._check_trainable_inputs``.
        output_keys: :class:`ForwardOutput` keys (for a :class:`ForwardOperator`) or
            output ModelState tensor names (for a :class:`PropertyTransform`) that
            this operator writes. The subset direction is **by operator kind**: a
            ForwardOperator's emitted data keys must be ⊆ ``output_keys`` (a
            declared menu, conditional channels may under-emit); a
            PropertyTransform's ``output_keys`` must be ⊆ the tensors it emits (a
            guaranteed floor, extra passthrough tensors are allowed).
        structural_params: names of model-side trainable parameter tensors that
            live **outside** the ModelState, the introspectable counterpart of
            ``trainable_inputs`` for domains (like flow) whose real inversion
            variables are constructor-time tensors rather than ModelState fields.
            For example :class:`~geobrain.physics.flow.FlowEvolutionOperator`
            differentiates w.r.t. ``rock.perm`` / ``rock.poro_ref`` fed via
            ``param_tensors``; declaring their names here lets tooling discover the
            domain's full trainable surface (``trainable_inputs`` alone cannot,
            since those tensors are not ModelState entries). Metadata only, like
            ``trainable_inputs`` it is not a "train these" request.
        input_units: physical units the operator EXPECTS for the ModelState INPUT
            fields it consumes, a ``{field_name: unit_string}`` mapping (e.g.
            ``{"rho": "kg/m^3", "vp": "m/s"}``). Opaque strings, compared for
            equality (no dimensional analysis), mirroring the output-side
            ``ForwardOutput.metadata["units"]`` contract. Optional and opt-in:
            the default empty mapping means "undeclared", so every existing
            operator keeps working unchanged and nothing is checked. Its purpose
            is joint multi-physics safety: an :class:`~geobrain.core.composition.OperatorBundle`
            shares ONE ``ModelState`` across members, so a field two members read
            under DIFFERENT unit conventions (e.g. one operator declaring ``x`` in
            ``foo`` while another declares ``x`` in ``bar``) is silently wrong. The
            bundle reconciles these at construction time by comparing members'
            declared ``input_units`` for shared fields; undeclared fields are
            skipped.
    """

    level: DifferentiabilityLevel
    trainable_inputs: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()
    structural_params: tuple[str, ...] = ()
    # hash=False: dicts are unhashable, so exclude this metadata field from the
    # frozen dataclass's generated __hash__ (restoring the spec's hashability,
    # which the tuple-only fields had). Kept in __eq__ so specs that declare
    # different input units are still correctly unequal.
    input_units: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Validate the level type and that input/output names are non-empty strings."""
        if not isinstance(self.level, DifferentiabilityLevel):
            raise GeoBrainError(
                "DifferentiabilitySpec.level must be a DifferentiabilityLevel",
                field="level",
                expected=DifferentiabilityLevel,
                actual=type(self.level),
            )
        for field_name in ("trainable_inputs", "output_keys", "structural_params"):
            for name in getattr(self, field_name):
                if not isinstance(name, str) or not name:
                    raise GeoBrainError(
                        f"{field_name} entries must be non-empty strings",
                        field=field_name,
                        actual=name,
                    )
        for key, unit in self.input_units.items():
            if not isinstance(key, str) or not key:
                raise GeoBrainError(
                    "input_units keys must be non-empty strings",
                    field="input_units",
                    actual=key,
                )
            if not isinstance(unit, str) or not unit:
                raise GeoBrainError(
                    "input_units values must be non-empty strings",
                    field="input_units",
                    expected="non-empty unit string",
                    actual=unit,
                )

    @property
    def supports_gradients(self) -> bool:
        """True unless the level is ``NON_DIFFERENTIABLE`` or ``FORWARD_ONLY``."""
        return self.level not in (
            DifferentiabilityLevel.NON_DIFFERENTIABLE,
            DifferentiabilityLevel.FORWARD_ONLY,
        )
