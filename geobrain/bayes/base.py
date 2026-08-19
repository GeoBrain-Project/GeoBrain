"""
Sampler ABC and shared log-posterior target protocol.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Mapping,
    Protocol,
    cast,
    runtime_checkable,
)

import torch

from ..core import ForwardContext, ModelState
from ..core.errors import GeoBrainError, RegistryError
from ._callable_problem import build_callable_problem

if TYPE_CHECKING:
    from typing import Self

    from .results import InferenceResult, _PartialInferenceResult


@runtime_checkable
class LogPosteriorTarget(Protocol):
    """Structural type a :class:`Sampler` samples from.

    Every sampler in this package consumes its ``target`` purely through a
    single method, ``log_posterior(state, ctx)`` returning the (unnormalised)
    scalar log-density at ``state``. Several unrelated classes satisfy this
    contract with no shared base: :class:`~geobrain.InverseProblem`,
    :class:`~geobrain.bayes.Posterior`, the constrained-space
    ``_TransformedTarget``, and the callable-mode ``_CallableProblem`` adapter.
    This Protocol names that duck type so sampler signatures can say what they
    mean (and ``isinstance`` checks are possible via ``@runtime_checkable``).

    Exported from ``geobrain.bayes`` (reversing an earlier internal-only
    decision): a third-party sampler or
    target adapter can type/assert against it without reaching into
    ``bayes.base``, mirroring :class:`~geobrain.inverse.InverseProblemLike` on
    the deterministic-optimisation side.
    """

    def log_posterior(
        self, state: ModelState, ctx: ForwardContext | None = ...
    ) -> torch.Tensor:
        ...


def _check_initial_log_post(lp: torch.Tensor, sampler_name: str) -> None:
    """Raise if the very first log-posterior evaluation is NaN.

    A NaN at the *initial* point (bad starting model, corrupt likelihood
    inputs) would otherwise produce a silently frozen (HMC / MALA / NUTS)
    or all-NaN (ULA / SVGD) chain. A ``-inf`` start is deliberately NOT rejected:
    HMC and LangevinDynamics carry explicit escape logic for a start inside a hard prior
    wall (a ``+inf`` log-accept is a guaranteed accept), and that behaviour
    is pinned by the bayes regression suite.

    ``lp`` may be a scalar (MCMC samplers) or a per-particle vector (SVGD).
    """
    if torch.isnan(lp).any():
        raise GeoBrainError(
            "initial log-posterior is NaN; check the starting model / "
            "likelihood inputs",
            object_name=sampler_name,
            field="params",
        )


def _check_initial_gradients(
    theta: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    sampler_name: str,
) -> None:
    """Validate an initial gradient bundle before any sampler cache is written."""
    if gradients.keys() != theta.keys():
        raise GeoBrainError(
            "initial target gradients must cover every parameter exactly",
            object_name=sampler_name,
            field="target",
            expected=f"gradient keys {sorted(theta)}",
            actual=sorted(gradients),
        )
    for name, parameter in theta.items():
        gradient = gradients[name]
        valid_tensor = isinstance(gradient, torch.Tensor)
        valid_layout = valid_tensor and gradient.layout is torch.strided
        valid_shape = valid_tensor and gradient.shape == parameter.shape
        valid_device = valid_tensor and gradient.device == parameter.device
        valid_dtype = valid_tensor and gradient.dtype == parameter.dtype
        real = valid_tensor and not gradient.is_complex()
        finite = (
            valid_tensor
            and valid_layout
            and real
            and bool(torch.isfinite(gradient).all())
        )
        if (
            valid_tensor
            and valid_layout
            and valid_shape
            and valid_device
            and valid_dtype
            and real
            and finite
        ):
            continue
        raise GeoBrainError(
            "initial target gradient must be a finite real strided tensor "
            "matching its parameter before sampler state can initialize",
            object_name=sampler_name,
            field="target",
            expected={
                "field": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "layout": str(torch.strided),
                "real": True,
                "finite": True,
            },
            actual={
                "field": name,
                "type": type(gradient).__name__,
                "shape": tuple(gradient.shape) if valid_tensor else None,
                "dtype": str(gradient.dtype) if valid_tensor else None,
                "device": str(gradient.device) if valid_tensor else None,
                "layout": str(gradient.layout) if valid_tensor else None,
                "real": real,
                "finite": finite,
            },
        )


def _partial_run_error(
    sampler_name: str,
    completed: int,
    total: int,
    partial: "_PartialInferenceResult | None",
    cause: BaseException,
    *,
    field: str = "target",
) -> GeoBrainError:
    """Build a structured execution failure with committed work when present.

    A target that raises at iteration ``k`` of ``total`` would otherwise abort
    ``run()`` and lose every completed draw. The samplers wrap their iteration
    loop and re-raise through this helper. The returned :class:`GeoBrainError`
    is chained to ``cause`` at the call site. If this call committed work, the
    error carries an owned private partial as ``err.partial_result``; the
    partial has no public returned-result stop reason.
    """
    suffix = (
        "; committed work attached as private .partial_result"
        if partial is not None
        else "; no transition committed in this call"
    )
    err = GeoBrainError(
        f"{sampler_name} {field} raised {type(cause).__name__} "
        f"mid-chain: {completed} of {total} iterations completed{suffix}",
        object_name=sampler_name,
        field=field,
        actual=f"{type(cause).__name__}: {cause}",
    )
    if partial is not None:
        err.partial_result = partial
    return err


class _TargetEvaluationFailure(Exception):
    """Internal marker separating target failures from sampler-kernel failures."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _evaluate_log_posterior(
    target: LogPosteriorTarget,
    state: ModelState,
    ctx: ForwardContext,
) -> torch.Tensor:
    """Evaluate a target while preserving an explicit execution-phase marker."""
    try:
        return target.log_posterior(state, ctx)
    except Exception as exc:
        raise _TargetEvaluationFailure(exc) from exc


def _requires_grad_leaves(
    theta: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Detached, grad-tracking leaf copies of a parameter dict.

    Shared by the gradient-based samplers (HMC / NUTS / LangevinDynamics):
    each step builds
    fresh leaves so a ``log_posterior`` backward fills ``.grad`` without
    touching the caller's tensors or accumulating gradients across steps.
    """
    return {
        name: t.detach().clone().requires_grad_(True) for name, t in theta.items()
    }


def _validate_constructor_convention(subclass: type, name: str) -> None:
    """Enforce the load-bearing ``Sampler`` constructor convention at
    registration time: ``__init__(self, target, *, params, **kwargs)``, a
    leading positional ``target`` and a keyword-only ``params`` (see
    :class:`Sampler`'s "Constructor convention" docstring section). The
    dispatch sites (``from_callable`` / ``Posterior.sample``) call every
    sampler this way, so a subclass with a mismatched signature would fail
    confusingly deep inside dispatch instead of loudly at
    ``@Sampler.register`` time.
    """
    try:
        init = inspect.getattr_static(subclass, "__init__")
        sig = inspect.signature(init)
    except (TypeError, ValueError):  # pragma: no cover - no introspectable signature
        return
    params = list(sig.parameters.values())[1:]  # drop 'self'
    positional = [
        p for p in params
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    # "keyword-capable": reachable via ``params=...`` at the call sites, which
    # pass ``target`` positionally and ``params`` by keyword. A parameter
    # satisfies this whether it is declared keyword-only (``*, params=...``,
    # the shipped samplers' style) or merely positional-or-keyword (``target,
    # params, *, ...``: still callable as ``params=...``); only POSITIONAL_ONLY
    # (rare, ``/``-marked) would fail a keyword call.
    keyword_capable = {
        p.name for p in params
        if p.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    }
    if not positional:
        raise RegistryError(
            "sampler __init__ must accept a leading positional 'target' argument",
            object_name="Sampler.register",
            field="name",
            expected="def __init__(self, target, *, params, **kwargs)",
            actual=f"{subclass.__name__}{sig}",
        )
    if "params" not in keyword_capable:
        raise RegistryError(
            "sampler __init__ must accept 'params' as a keyword argument "
            "(the dispatch sites always pass it as params=...)",
            object_name="Sampler.register",
            field="name",
            expected="def __init__(self, target, *, params, **kwargs)",
            actual=f"{subclass.__name__}{sig}",
        )


class Sampler(ABC):
    """
    Sampler ABC.

    Constructor convention:
        ``__init__(self, target, *, params: Mapping[str, Tensor], **kwargs)``;
        ``target`` is any :class:`LogPosteriorTarget` (an object with a
        ``log_posterior(state, ctx)`` method, typically
        :class:`~geobrain.InverseProblem` or
        :class:`~geobrain.bayes.Posterior`). ``params`` is the *initial state*
        for sampling (one tensor per field; ``requires_grad`` is managed
        internally on temporary copies). The dict is unchanged after :meth:`run`.

        This signature is load-bearing, not merely illustrative: the dispatch
        sites (:meth:`from_callable` and :meth:`Posterior.sample`) pass
        ``target`` **positionally** and ``params`` **by keyword**, so a
        third-party sampler must accept a leading positional ``target`` and a
        ``params`` keyword. Enforced at ``@Sampler.register`` time via an
        ``inspect.signature`` check, a
        mismatched constructor fails loudly at registration with a
        :class:`~geobrain.core.errors.RegistryError`, not confusingly deep
        inside dispatch.

    Note:
        Decorate each concrete sampler with ``@Sampler.register("name")`` to make
        it dispatchable from :meth:`Posterior.sample` (e.g.
        ``posterior.sample("mala", ...)``). Registering the same name twice raises
        :class:`~geobrain.core.errors.RegistryError`. Third-party samplers register
        the same way; importing the module is enough to make the name reachable.

        Contract alignment with :class:`~geobrain.core.registry.Registry`: this
        registry raises the same :class:`RegistryError` and exposes :meth:`names`
        (with :meth:`registered_names` as a deliberate dual-name convenience). It diverges
        from ``Registry`` in one documented way; **lookup is case-insensitive**:
        names are normalised via ``str.lower()`` on both ``register`` and ``get``,
        so ``Sampler.get("HMC") is Sampler.get("hmc")``. This is deliberate and
        relied upon (``Posterior.sample`` advertises case-insensitive ``method``).
    """

    # Class-level registry of name → concrete sampler class. Populated by the
    # ``@Sampler.register(name)`` decorator at module import time.
    _registry: ClassVar[dict[str, type["Sampler"]]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type["Sampler"]], type["Sampler"]]:
        """Decorator to register a Sampler subclass under ``name``."""
        if not isinstance(name, str) or not name:
            raise RegistryError(
                "sampler name must be a non-empty string",
                object_name="Sampler.register",
                field="name",
                expected="non-empty string",
                actual=name,
            )

        def decorator(subclass: type["Sampler"]) -> type["Sampler"]:
            key = name.lower()
            if key in cls._registry:
                raise RegistryError(
                    "sampler name already registered",
                    object_name="Sampler.register",
                    field="name",
                    expected=f"unique (existing: {sorted(cls._registry)})",
                    actual=name,
                )
            _validate_constructor_convention(subclass, name)
            cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> type["Sampler"]:
        """Look up a registered sampler class. Raises on unknown name."""
        if not isinstance(name, str) or not name:
            raise RegistryError(
                "sampler name must be a non-empty string",
                object_name="Sampler.get",
                field="name",
                expected="non-empty string",
                actual=name,
            )
        key = name.lower()
        if key not in cls._registry:
            raise RegistryError(
                "unknown sampler",
                object_name="Sampler.get",
                field="name",
                expected=f"one of {sorted(cls._registry)}",
                actual=name,
            )
        return cls._registry[key]

    @classmethod
    def names(cls) -> list[str]:
        """Sorted list of registered sampler names (matches ``Registry.names``)."""
        return sorted(cls._registry)

    @classmethod
    def registered_names(cls) -> list[str]:
        """Dual-name convenience for :meth:`names` (same result)."""
        return cls.names()

    @classmethod
    def from_callable(
        cls,
        log_prob_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        init: Mapping[str, torch.Tensor],
        **kwargs: Any,
    ) -> "Self":
        """
        Build a sampler from a log-probability callable.

        Adapts callable-mode usage (custom log-posterior functions) to the
        class-based, ``InverseProblem``-driven samplers: the callable is wrapped
        in a minimal :class:`LogPosteriorTarget` adapter (``_CallableProblem``)
        and handed to the concrete sampler's constructor.

        Args:
            log_prob_fn: callable ``θ_dict → scalar log p(θ)``. Receives a dict
                keyed by ``init.keys()`` and returns an autograd-aware scalar
                Tensor.
            init: initial parameter values; one Tensor per field (samplers with a
                stricter arity, e.g. SVGD's single field, enforce it in their
                constructor).
            **kwargs: forwarded to the concrete sampler's constructor
                (``step_size``, ``n_leapfrog``, ``max_depth``, ``adjusted``,
                ``n_particles``, ``generator``, ...).
        """
        problem = build_callable_problem(log_prob_fn, init)
        sampler_factory = cast(Any, cls)
        return cast("Self", sampler_factory(problem, params=init, **kwargs))

    def _generator_for(self, device: torch.device) -> torch.Generator | None:
        """Generator matching ``device`` (or ``None`` for the global RNG).

        Adopts the ``geobrain.nn`` convention
        (:meth:`BaseVariationalLayer._generator_for`): PyTorch requires a
        random op's generator to live on the op's device, so a CPU-seeded
        sampler driving CUDA params (or vice versa) would otherwise crash on
        the first momentum / noise draw with a bare ``RuntimeError``. When
        ``self.generator`` already matches ``device`` it is returned
        **unchanged**, so seeded same-device chains stay bit-identical to
        before. On a mismatch, a device-native generator is lazily created,
        and cached per device, so repeated ``run()`` calls keep advancing one
        stream, seeded deterministically from ``self.generator``'s seed.
        With no injected generator the global RNG is used (``None``).
        """
        gen = getattr(self, "generator", None)
        if gen is None or gen.device == device:
            return gen
        cache = self.__dict__.setdefault("_device_generators", {})
        derived = cache.get(device)
        if derived is None:
            derived = torch.Generator(device=device)
            derived.manual_seed(gen.initial_seed())
            cache[device] = derived
        return derived

    @staticmethod
    def _stack_chains(
        chains: Mapping[str, list[torch.Tensor]],
        reference: Mapping[str, torch.Tensor],
        store_dtype: "torch.dtype | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Finalise per-field sample chains into stacked ``(n_draws, *shape)`` tensors.

        Shared by the MCMC samplers (HMC / NUTS / LangevinDynamics) whose
        ``run`` accumulates
        a list of recorded draws per field. An empty chain (e.g. ``n_iters == 0``
        or an immediate callback stop) yields a correctly-shaped ``(0, *shape)``
        empty tensor rather than failing in ``torch.stack``; ``reference`` supplies
        the field shape / dtype / device for that empty case, with ``store_dtype``
        (when set) overriding the dtype so an empty chain matches what recorded
        draws would have been stored as.
        """
        return {
            name: (
                torch.stack(chunks, dim=0)
                if chunks
                else torch.empty(
                    (0, *reference[name].shape),
                    dtype=store_dtype
                    if store_dtype is not None
                    else reference[name].dtype,
                    device=reference[name].device,
                )
            )
            for name, chunks in chains.items()
        }

    @abstractmethod
    def run(
        self,
        n_iters: int,
        ctx: ForwardContext | None = None,
        callback: Callable[[int, float, Mapping[str, torch.Tensor]], bool | None] | None = None,
    ) -> InferenceResult:
        """Draw ``n_iters`` samples and return an :class:`InferenceResult`."""
        ...
