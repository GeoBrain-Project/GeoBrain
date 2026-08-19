"""Bounded progressive-multinomial trajectory construction for NUTS.

The tree retains only its left endpoint, right endpoint, one representative
proposal, and the active recursion stack. ``TreeWorkspaceStats`` instruments
those live algorithmic state slots while the real tree is being built; it is
not inferred later from the requested depth.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Mapping, Protocol

import torch

from ...core.validation import validate_finite_float, validate_int
from ._hamiltonian import (
    MassInfo,
    _dot_dict,
    _drift,
    _kinetic,
    _sample_momentum,
)


GradientEvaluator = Callable[
    [Mapping[str, torch.Tensor]],
    tuple[torch.Tensor, dict[str, torch.Tensor]],
]


def validate_tree_options(
    step_size: float,
    max_depth: int,
    delta_max: float,
) -> tuple[float, int, float]:
    """Validate the public trajectory controls consumed by this module."""
    return (
        validate_finite_float(
            step_size,
            owner="NUTS",
            field="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        validate_int(
            max_depth,
            owner="NUTS",
            field="max_depth",
            minimum=0,
            minimum_inclusive=False,
        ),
        validate_finite_float(
            delta_max,
            owner="NUTS",
            field="delta_max",
            minimum=0.0,
            minimum_inclusive=False,
        ),
    )


@dataclass(frozen=True)
class HamiltonianState:
    """One position, momentum, gradient, and log-density state."""

    position: Mapping[str, torch.Tensor]
    momentum: Mapping[str, torch.Tensor]
    gradient: Mapping[str, torch.Tensor]
    log_prob: torch.Tensor


StateStepper = Callable[
    [HamiltonianState, float, MassInfo | None],
    HamiltonianState,
]


@dataclass(frozen=True)
class TreeResult:
    """Endpoints, proposal, weight, and stopping diagnostics for one subtree."""

    left: HamiltonianState
    right: HamiltonianState
    proposal: HamiltonianState
    log_weight: float
    accept_sum: float
    accept_count: int
    divergent: bool
    turning: bool

    @property
    def keep(self) -> bool:
        """Whether another subtree may be joined to this result."""
        return not self.divergent and not self.turning


@dataclass
class TreeWorkspaceStats:
    """Unique live states owned by the active production tree execution.

    States are registered when the integrator actually creates them. Identity
    aliases are counted once, and weak references ensure instrumentation never
    extends a state's tensor lifetime. ``live_states`` is scoped to builder
    ownership: the outermost execution clears its registrations on return even
    though the returned :class:`TreeResult` continues to own its endpoints.
    """

    peak_live_states: int = 0
    _references: dict[
        int,
        weakref.ReferenceType[HamiltonianState],
    ] = field(default_factory=dict, init=False, repr=False)
    _execution_depth: int = field(default=0, init=False, repr=False)

    @property
    def live_states(self) -> int:
        """Number of unique registered states still alive in the active scope."""
        dead = [
            identity
            for identity, reference in self._references.items()
            if reference() is None
        ]
        for identity in dead:
            self._references.pop(identity, None)
        return len(self._references)

    def observe(self, state: HamiltonianState) -> None:
        """Register one actually created tensor-bearing state by identity."""
        identity = id(state)
        current = self._references.get(identity)
        if current is not None and current() is state:
            return

        def release(
            reference: weakref.ReferenceType[HamiltonianState],
            *,
            state_identity: int = identity,
        ) -> None:
            registered = self._references.get(state_identity)
            if registered is reference:
                self._references.pop(state_identity, None)

        self._references[identity] = weakref.ref(state, release)
        self.peak_live_states = max(
            self.peak_live_states,
            self.live_states,
        )

    @contextmanager
    def execution(self) -> Iterator[None]:
        """Scope registrations to the outermost builder invocation."""
        outermost = self._execution_depth == 0
        self._execution_depth += 1
        try:
            yield
        finally:
            self._execution_depth -= 1
            if outermost:
                self._references.clear()


@dataclass(frozen=True)
class TrajectoryResult:
    """One completed progressive-multinomial NUTS trajectory."""

    proposal: HamiltonianState
    depth: int
    accept_sum: float
    accept_count: int
    divergent: bool


class TreeBuilder(Protocol):
    """Callable boundary used by the sampler facade for subtree construction."""

    def __call__(
        self,
        state: HamiltonianState,
        *,
        log_joint_initial: float,
        log_reference: float,
        direction: int,
        depth: int,
        step_size: float,
        mass_info: MassInfo | None,
        delta_max: float,
        generator: torch.Generator | None,
        device: torch.device,
        evaluate_gradient: GradientEvaluator,
        workspace: TreeWorkspaceStats,
    ) -> TreeResult:
        ...


def _logaddexp(a: float, b: float) -> float:
    """Compute ``log(exp(a) + exp(b))`` without losing ``-inf`` weights."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    maximum = max(a, b)
    return maximum + math.log1p(math.exp(min(a, b) - maximum))


def leapfrog(
    state: HamiltonianState,
    step_size: float,
    mass_info: MassInfo | None,
    evaluate_gradient: GradientEvaluator,
) -> HamiltonianState:
    """Advance one symmetric leapfrog step, reusing the cached leading force."""
    p_half = {
        name: state.momentum[name]
        + 0.5 * step_size * state.gradient[name]
        for name in state.position
    }
    drift = _drift(p_half, mass_info)
    position_new = {
        name: state.position[name] + step_size * drift[name]
        for name in state.position
    }
    log_prob_new, gradient_new = evaluate_gradient(position_new)
    momentum_new = {
        name: p_half[name] + 0.5 * step_size * gradient_new[name]
        for name in state.position
    }
    return HamiltonianState(
        position=position_new,
        momentum=momentum_new,
        gradient=gradient_new,
        log_prob=log_prob_new,
    )


def build_tree(
    state: HamiltonianState,
    *,
    log_joint_initial: float,
    log_reference: float,
    direction: int,
    depth: int,
    step_size: float,
    mass_info: MassInfo | None,
    delta_max: float,
    generator: torch.Generator | None,
    device: torch.device,
    evaluate_gradient: GradientEvaluator,
    workspace: TreeWorkspaceStats,
    take_step: StateStepper | None = None,
) -> TreeResult:
    """Recursively build one bounded progressive-multinomial subtree."""
    with workspace.execution():
        if depth == 0:
            next_state = (
                leapfrog(
                    state,
                    direction * step_size,
                    mass_info,
                    evaluate_gradient,
                )
                if take_step is None
                else take_step(
                    state,
                    direction * step_size,
                    mass_info,
                )
            )
            workspace.observe(next_state)
            kinetic = _kinetic(next_state.momentum, mass_info)
            log_joint = float((next_state.log_prob - kinetic).item())
            if not math.isfinite(log_joint):
                return TreeResult(
                    left=next_state,
                    right=next_state,
                    proposal=next_state,
                    log_weight=float("-inf"),
                    accept_sum=0.0,
                    accept_count=1,
                    divergent=True,
                    turning=False,
                )
            delta_log_joint = log_joint - log_joint_initial
            divergent = delta_log_joint < -delta_max
            acceptance = min(1.0, math.exp(min(0.0, delta_log_joint)))
            return TreeResult(
                left=next_state,
                right=next_state,
                proposal=next_state,
                log_weight=log_joint - log_reference,
                accept_sum=acceptance,
                accept_count=1,
                divergent=divergent,
                turning=False,
            )

        first = build_tree(
            state,
            log_joint_initial=log_joint_initial,
            log_reference=log_reference,
            direction=direction,
            depth=depth - 1,
            step_size=step_size,
            mass_info=mass_info,
            delta_max=delta_max,
            generator=generator,
            device=device,
            evaluate_gradient=evaluate_gradient,
            workspace=workspace,
            take_step=take_step,
        )
        if not first.keep:
            return first

        sibling_root = first.left if direction == -1 else first.right
        second = build_tree(
            sibling_root,
            log_joint_initial=log_joint_initial,
            log_reference=log_reference,
            direction=direction,
            depth=depth - 1,
            step_size=step_size,
            mass_info=mass_info,
            delta_max=delta_max,
            generator=generator,
            device=device,
            evaluate_gradient=evaluate_gradient,
            workspace=workspace,
            take_step=take_step,
        )
        if direction == -1:
            left, right = second.left, first.right
        else:
            left, right = first.left, second.right

        accept_sum = first.accept_sum + second.accept_sum
        accept_count = first.accept_count + second.accept_count
        divergent = first.divergent or second.divergent
        if not second.keep:
            return TreeResult(
                left=left,
                right=right,
                proposal=first.proposal,
                log_weight=first.log_weight,
                accept_sum=accept_sum,
                accept_count=accept_count,
                divergent=divergent,
                turning=second.turning,
            )

        log_weight = _logaddexp(
            first.log_weight,
            second.log_weight,
        )
        take_second = float(
            torch.rand(
                (),
                generator=generator,
                device=device,
            ).item()
        ) < math.exp(second.log_weight - log_weight)
        proposal = second.proposal if take_second else first.proposal

        delta = {
            name: right.position[name] - left.position[name]
            for name in state.position
        }
        left_velocity = _drift(left.momentum, mass_info)
        right_velocity = _drift(right.momentum, mass_info)
        keeps_direction = (
            _dot_dict(delta, left_velocity) >= 0
            and _dot_dict(delta, right_velocity) >= 0
        )
        return TreeResult(
            left=left,
            right=right,
            proposal=proposal,
            log_weight=log_weight,
            accept_sum=accept_sum,
            accept_count=accept_count,
            divergent=divergent,
            turning=not keeps_direction,
        )


def build_trajectory(
    position: Mapping[str, torch.Tensor],
    log_prob: float,
    gradient: Mapping[str, torch.Tensor],
    *,
    step_size: float,
    max_depth: int,
    delta_max: float,
    mass_info: MassInfo | None,
    generator: torch.Generator | None,
    device: torch.device,
    evaluate_gradient: GradientEvaluator,
    workspace: TreeWorkspaceStats,
    tree_builder: TreeBuilder | None = None,
) -> TrajectoryResult:
    """Build one full NUTS trajectory with the historical RNG call ordering."""
    momentum = _sample_momentum(position, mass_info, generator)
    kinetic_initial = _kinetic(momentum, mass_info)
    log_joint = log_prob - float(kinetic_initial.item())
    log_reference = log_joint if math.isfinite(log_joint) else 0.0
    reference = next(iter(position.values()))
    root = HamiltonianState(
        position=position,
        momentum=momentum,
        gradient=gradient,
        log_prob=torch.as_tensor(
            log_prob,
            dtype=reference.dtype,
            device=reference.device,
        ),
    )
    left = root
    right = root
    proposal = root
    trajectory_log_weight = log_joint - log_reference
    divergent = False
    depth = 0
    accept_sum = 0.0
    accept_count = 0

    with workspace.execution():
        workspace.observe(root)
        builder = build_tree if tree_builder is None else tree_builder
        while depth < max_depth:
            direction = (
                1
                if torch.rand(
                    (),
                    generator=generator,
                    device=device,
                ).item()
                > 0.5
                else -1
            )
            subtree = builder(
                left if direction == -1 else right,
                log_joint_initial=log_joint,
                log_reference=log_reference,
                direction=direction,
                depth=depth,
                step_size=step_size,
                mass_info=mass_info,
                delta_max=delta_max,
                generator=generator,
                device=device,
                evaluate_gradient=evaluate_gradient,
                workspace=workspace,
            )
            if direction == -1:
                left = subtree.left
            else:
                right = subtree.right
            accept_sum += subtree.accept_sum
            accept_count += subtree.accept_count
            divergent = divergent or subtree.divergent
            if not subtree.keep:
                break
            log_ratio = subtree.log_weight - trajectory_log_weight
            if log_ratio >= 0.0 or float(
                torch.rand(
                    (),
                    generator=generator,
                    device=device,
                ).item()
            ) < math.exp(log_ratio):
                proposal = subtree.proposal
            trajectory_log_weight = _logaddexp(
                trajectory_log_weight,
                subtree.log_weight,
            )
            delta = {
                name: right.position[name] - left.position[name]
                for name in position
            }
            left_velocity = _drift(left.momentum, mass_info)
            right_velocity = _drift(right.momentum, mass_info)
            if (
                _dot_dict(delta, left_velocity) < 0
                or _dot_dict(delta, right_velocity) < 0
            ):
                break
            depth += 1

    return TrajectoryResult(
        proposal=proposal,
        depth=depth,
        accept_sum=accept_sum,
        accept_count=accept_count,
        divergent=divergent,
    )
