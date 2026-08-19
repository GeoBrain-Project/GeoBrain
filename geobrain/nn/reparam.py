"""
Neural reparameterization of the model space, as PropertyTransforms.

The one abstraction behind Deep Image Prior, latent-space inversion and
generative-geomodel priors: a network ``g`` maps trainable leaves (weights
θ or a latent code z) to physical ModelState fields, and any physics
operator consumes those fields downstream::

    problem = InverseProblem(
        forward=physics_op @ LatentReparameterization(decoder, outputs="vp"),
        observed=..., likelihood=..., prior=...,
    )

Because the seam is a :class:`~geobrain.core.operator.PropertyTransform`
inside the problem's forward chain (chain ``trainable_inputs`` = FIRST
operator's, the entry-point contract of
:class:`~geobrain.core.composition.OperatorChain`), every solver works
unchanged: :class:`~geobrain.optim.Inverter` (Tier 1/2), HMC / NUTS /
LangevinDynamics,
and SVGD (whose single-field arity the latent mode satisfies naturally).

Device contract: the caller moves the network (both classes are
``nn.Module``s via ``Operator``, so ``reparam.to(device)`` moves the wrapped
network too); a device mismatch between the network parameters and the
reference tensor raises a structured error rather than a bare torch
``RuntimeError``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
import torch.nn as nn

from geobrain.core.containers import ModelState
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.core.errors import GeoBrainError
from geobrain.core.operator import PropertyTransform

__all__ = ["LatentReparameterization", "WeightReparameterization"]

_FieldTransforms = Mapping[str, Callable[[torch.Tensor], torch.Tensor]]


class _NeuralReparamBase(PropertyTransform):
    """Shared output-head machinery for the neural reparameterizations."""

    # differentiability depends on constructor arguments (latent_field / the
    # network's parameter names), so each concrete subclass assigns
    # ``self.differentiability = DifferentiabilitySpec(...)`` once in its own
    # __init__: a plain instance attribute, no
    # property, no class-level placeholder.

    def _init_head(
        self,
        network: nn.Module,
        outputs: str | Mapping[str, int],
        transforms: _FieldTransforms | None,
        owner: str,
    ) -> None:
        if not isinstance(network, nn.Module):
            raise GeoBrainError(
                f"{owner} network must be a torch.nn.Module",
                object_name=owner,
                field="network",
                expected=nn.Module,
                actual=type(network),
            )
        if isinstance(outputs, str):
            if not outputs:
                raise GeoBrainError(
                    f"{owner} outputs must be a non-empty field name or mapping",
                    object_name=owner,
                    field="outputs",
                    expected="non-empty str or {field: channel}",
                    actual=outputs,
                )
            field_names: tuple[str, ...] = (outputs,)
            channel_map: dict[str, int] | None = None
        else:
            channel_map = dict(outputs)
            if not channel_map:
                raise GeoBrainError(
                    f"{owner} outputs mapping must be non-empty",
                    object_name=owner,
                    field="outputs",
                    expected="{field: channel_index}",
                    actual=channel_map,
                )
            for name, idx in channel_map.items():
                if not isinstance(name, str) or not name:
                    raise GeoBrainError(
                        f"{owner} outputs keys must be non-empty strings",
                        object_name=owner,
                        field="outputs",
                        expected="non-empty str keys",
                        actual=name,
                    )
                if not isinstance(idx, int) or idx < 0:
                    raise GeoBrainError(
                        f"{owner} outputs channels must be non-negative ints",
                        object_name=owner,
                        field=f"outputs[{name!r}]",
                        expected=">= 0 int",
                        actual=idx,
                    )
            field_names = tuple(channel_map)
        tf = dict(transforms or {})
        unknown = set(tf) - set(field_names)
        if unknown:
            raise GeoBrainError(
                f"{owner} transforms contain unknown output fields",
                object_name=owner,
                field="transforms",
                expected=f"keys in {sorted(field_names)}",
                actual=sorted(unknown),
            )
        for name, fn in tf.items():
            if not callable(fn):
                raise GeoBrainError(
                    f"{owner} transforms values must be callable",
                    object_name=owner,
                    field=f"transforms[{name!r}]",
                    expected="callable",
                    actual=type(fn),
                )
        self.network = network
        self._owner = owner
        self._field_names = field_names
        self._channel_map = channel_map
        self._transforms = tf

    def _check_device(self, ref: torch.Tensor) -> None:
        p = next(self.network.parameters(), None)
        if p is not None and p.device != ref.device:
            raise GeoBrainError(
                f"{self._owner} network and its input live on different devices "
                ", move the reparameterization (reparam.to(device)) to match",
                object_name=self._owner,
                field="network",
                expected=f"parameters on {ref.device}",
                actual=str(p.device),
            )

    def _extract_outputs(self, decoded: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(decoded, torch.Tensor):
            raise GeoBrainError(
                f"{self._owner} network must return a torch.Tensor",
                object_name=self._owner,
                field="network",
                expected=torch.Tensor,
                actual=type(decoded),
            )
        fields: dict[str, torch.Tensor] = {}
        if self._channel_map is None:
            fields[self._field_names[0]] = decoded
        else:
            n_channels = decoded.shape[0] if decoded.ndim >= 1 else 0
            for name, idx in self._channel_map.items():
                if idx >= n_channels:
                    raise GeoBrainError(
                        f"{self._owner} output channel out of range",
                        object_name=self._owner,
                        field=f"outputs[{name!r}]",
                        expected=f"channel < {n_channels} "
                        f"(decoded shape {tuple(decoded.shape)})",
                        actual=idx,
                    )
                fields[name] = decoded[idx]
        for name, fn in self._transforms.items():
            value = fn(fields[name])
            if not isinstance(value, torch.Tensor):
                raise GeoBrainError(
                    f"{self._owner} transform must return a torch.Tensor",
                    object_name=self._owner,
                    field=f"transforms[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(value),
                )
            fields[name] = value
        return fields


class LatentReparameterization(_NeuralReparamBase):
    """Latent-space reparameterization: the latent code is the unknown.

    Wraps a FROZEN network ``z -> decoded`` as a
    :class:`~geobrain.core.operator.PropertyTransform` whose single trainable
    input is ``latent_field`` and whose outputs are physical fields. Compose
    with any physics operator, ``physics @ LatentReparameterization(...)``,
    and the resulting problem drives Inverter and every bayes sampler
    unchanged (SVGD's single-field arity is satisfied by construction).

    Args:
        network: Decoder module. Frozen at construction
            (``requires_grad_(False)`` on every parameter), latent inversion
            optimises ``z``, never the generator.
        outputs: Either a single field name (the whole decoded tensor becomes
            that field) or a mapping ``{field: channel}`` splitting the
            leading dim of the (batch-squeezed) decoded tensor.
        latent_field: Name of the trainable latent field in the ModelState.
        latent_shape: When given, the stored latent may be FLAT (e.g. an SVGD
            particle); it is reshaped to ``(1, *latent_shape)`` (batch dim
            added) before decoding and the decoded batch dim is squeezed
            away. ``None`` passes the latent through unchanged (no squeeze).
        transforms: Optional per-field post-maps (e.g. affine to a physical
            range), applied after channel extraction.
    """

    def __init__(
        self,
        network: nn.Module,
        outputs: str | Mapping[str, int],
        *,
        latent_field: str = "latent",
        latent_shape: tuple[int, ...] | None = None,
        transforms: _FieldTransforms | None = None,
    ) -> None:
        super().__init__()
        self._init_head(network, outputs, transforms, "LatentReparameterization")
        if not isinstance(latent_field, str) or not latent_field:
            raise GeoBrainError(
                "LatentReparameterization latent_field must be a non-empty string",
                object_name="LatentReparameterization",
                field="latent_field",
                expected="non-empty str",
                actual=latent_field,
            )
        if latent_shape is not None:
            latent_shape = tuple(latent_shape)
            for dim in latent_shape:
                if not isinstance(dim, int) or dim <= 0:
                    raise GeoBrainError(
                        "LatentReparameterization latent_shape dims must be "
                        "positive ints",
                        object_name="LatentReparameterization",
                        field="latent_shape",
                        expected="positive int dims",
                        actual=latent_shape,
                    )
        self._latent_field = latent_field
        self._latent_shape = latent_shape
        for p in self.network.parameters():
            p.requires_grad_(False)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=(latent_field,),
            output_keys=self._field_names,
        )

    def initial_params(
        self,
        value: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """``params=`` mapping for a solver: ``{latent_field: z0}``.

        ``value`` is used as-is when given; otherwise a standard-normal FLAT
        draw of ``latent_shape`` numel (which must then be set) seeds the
        latent; the flat form is what SVGD particles and the
        ``latent_shape`` reshape path expect.
        """
        if value is None:
            if self._latent_shape is None:
                raise GeoBrainError(
                    "LatentReparameterization.initial_params needs value= when "
                    "latent_shape was not set",
                    object_name="LatentReparameterization",
                    field="value",
                    expected="explicit initial latent tensor",
                    actual=None,
                )
            value = torch.randn(self._latent_shape, generator=generator).reshape(-1)
        return {self._latent_field: value}

    def _forward(self, state: ModelState, ctx) -> ModelState:
        z = state.tensors[self._latent_field]
        self._check_device(z)
        squeeze = False
        if self._latent_shape is not None:
            numel = 1
            for d in self._latent_shape:
                numel *= d
            if z.numel() != numel:
                raise GeoBrainError(
                    "latent field does not match latent_shape",
                    object_name="LatentReparameterization",
                    field=self._latent_field,
                    expected=f"numel == {numel} (latent_shape {self._latent_shape})",
                    actual=tuple(z.shape),
                )
            z = z.reshape(1, *self._latent_shape)
            squeeze = True
        decoded = self.network(z)
        if squeeze and isinstance(decoded, torch.Tensor) and decoded.ndim >= 1:
            decoded = decoded.squeeze(0)
        fields = self._extract_outputs(decoded)
        return ModelState(tensors={**state.tensors, **fields}, metadata=state.metadata)


class WeightReparameterization(_NeuralReparamBase):
    """Weight-space reparameterization (Deep Image Prior): θ is the unknown.

    The network's ``named_parameters()`` become the trainable ModelState
    fields; every forward runs ``torch.func.functional_call(network, θ,
    fixed_input)`` with the CURRENT θ read from the state. The decoded batch
    dim is squeezed when it is 1 (DIP inputs are conventionally batched).

    ``initial_params()`` returns the module's parameter tensors as INITIAL
    VALUES: solvers own their optimization state (the Inverter clones its
    ``params``), so the module itself does not track the optimization. Every
    consumer of the current θ must read it from the ModelState / params dict
    (as ``_forward`` here and :func:`geobrain.nn.kl_regularizer` both do,
    via ``torch.func.functional_call``).

    Args:
        network: The DIP decoder; its named parameters are the unknowns.
        fixed_input: Frozen input fed to the network every step (detached
            copy; typically a noise tensor WITH a leading batch dim of 1).
        outputs: Field name (whole tensor) or ``{field: channel}`` split of
            the batch-squeezed decoded tensor.
        transforms: Optional per-field post-maps.
        squeeze_batch: Squeeze the decoded leading dim when it is 1
            (default True).
    """

    def __init__(
        self,
        network: nn.Module,
        *,
        fixed_input: torch.Tensor,
        outputs: str | Mapping[str, int],
        transforms: _FieldTransforms | None = None,
        squeeze_batch: bool = True,
    ) -> None:
        super().__init__()
        self._init_head(network, outputs, transforms, "WeightReparameterization")
        if not isinstance(fixed_input, torch.Tensor):
            raise GeoBrainError(
                "WeightReparameterization fixed_input must be a torch.Tensor",
                object_name="WeightReparameterization",
                field="fixed_input",
                expected=torch.Tensor,
                actual=type(fixed_input),
            )
        names = tuple(name for name, _ in network.named_parameters())
        if not names:
            raise GeoBrainError(
                "WeightReparameterization network has no trainable parameters",
                object_name="WeightReparameterization",
                field="network",
                expected="non-empty network.named_parameters()",
                actual=(),
            )
        self._param_names = names
        # Registered as a buffer so ``reparam.to(device)`` carries it along
        # with the network; detached copy so the caller's tensor stays free.
        self.register_buffer(
            "fixed_input", fixed_input.detach().clone(), persistent=False
        )
        self._squeeze_batch = bool(squeeze_batch)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=names,
            output_keys=self._field_names,
        )

    def initial_params(self) -> dict[str, torch.Tensor]:
        """``params=`` mapping for a solver: the module's live parameter tensors."""
        return {name: p for name, p in self.network.named_parameters()}

    def _forward(self, state: ModelState, ctx) -> ModelState:
        weights = {name: state.tensors[name] for name in self._param_names}
        self._check_device(self.fixed_input)
        decoded = torch.func.functional_call(
            self.network, weights, (self.fixed_input,)
        )
        if (
            self._squeeze_batch
            and isinstance(decoded, torch.Tensor)
            and decoded.ndim >= 1
            and decoded.shape[0] == 1
        ):
            decoded = decoded.squeeze(0)
        fields = self._extract_outputs(decoded)
        return ModelState(tensors={**state.tensors, **fields}, metadata=state.metadata)
