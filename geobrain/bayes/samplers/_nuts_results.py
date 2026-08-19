"""Call-local NUTS buffers, metadata, and result presentation.

This dependency-leaf module owns committed per-call output accounting and
turns it into completed or partial inference results. It does not orchestrate
sampling, callbacks, failure translation, trajectory construction, or warmup.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch

from ..execution import RunAccounting, SamplerStopReason, stored_draw
from ..results import InferenceResult, _PartialInferenceResult


class _DualAveragingResultState(Protocol):
    """Step-size field required for result metadata."""

    @property
    def log_epsilon_bar(self) -> float:
        ...


class _WarmupScheduleResultState(Protocol):
    """Schedule field required for result metadata."""

    @property
    def windowed(self) -> bool:
        ...


class WarmupResultState(Protocol):
    """Structural warmup snapshot consumed only for result presentation."""

    @property
    def dual_averaging(self) -> _DualAveragingResultState:
        ...

    @property
    def mass_adapted(self) -> bool:
        ...

    @property
    def schedule(self) -> _WarmupScheduleResultState:
        ...

    @property
    def n_mass_updates(self) -> int:
        ...


ResultMassInfo = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _ResultPayload:
    """Typed common fields shared by completed and partial NUTS results."""

    samples: Mapping[str, torch.Tensor]
    log_post_history: torch.Tensor
    acceptance_rate: float
    requested_iters: int
    completed_iters: int
    metadata: Mapping[str, Any]


@dataclass
class NUTSRunOutput:
    """Own output buffers and exact result metadata for one run call."""

    accounting: RunAccounting
    requested_iters: int
    warmup: int
    thin: int
    store_dtype: torch.dtype | None
    generator_initial_seed: int
    step_size_initial: float
    max_depth: int
    mass_type_requested: str
    mass_injected: bool
    chains: dict[str, list[torch.Tensor]]
    history: list[float] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    acceptance_sum: float = 0.0
    callback_phase: str | None = None
    callback_iteration: int | None = None

    @classmethod
    def initialize(
        cls,
        *,
        accounting: RunAccounting,
        reference: Mapping[str, torch.Tensor],
        warmup: int,
        thin: int,
        store_dtype: torch.dtype | None,
        generator_initial_seed: int,
        step_size_initial: float,
        max_depth: int,
        mass_type_requested: str,
        mass_injected: bool,
    ) -> NUTSRunOutput:
        """Create empty owned buffers for every parameter field."""
        return cls(
            accounting=accounting,
            requested_iters=accounting.requested_iters,
            warmup=warmup,
            thin=thin,
            store_dtype=store_dtype,
            generator_initial_seed=generator_initial_seed,
            step_size_initial=step_size_initial,
            max_depth=max_depth,
            mass_type_requested=mass_type_requested,
            mass_injected=mass_injected,
            chains={name: [] for name in reference},
        )

    def record_sampling(
        self,
        position: Mapping[str, torch.Tensor],
        *,
        cumulative_iteration: int,
        log_prob: float,
        acceptance: float,
        depth: int,
    ) -> None:
        """Record dense diagnostics and storage-thinned parameter draws."""
        self.acceptance_sum += acceptance
        self.depths.append(depth)
        self.history.append(log_prob)
        if cumulative_iteration % self.thin != 0:
            return
        for name in self.chains:
            self.chains[name].append(
                stored_draw(position[name], self.store_dtype)
            )

    def record_callback_stop(self, *, phase: str, iteration: int) -> None:
        """Record the callback location exposed in result metadata."""
        self.callback_phase = phase
        self.callback_iteration = iteration

    def committed_since(self, warmup_at_entry: int) -> bool:
        """Whether this call committed sampling or new warmup state."""
        return bool(
            self.accounting.completed_sampling > 0
            or self.accounting.completed_warmup > warmup_at_entry
        )

    def result(
        self,
        stop_reason: SamplerStopReason,
        *,
        reference: Mapping[str, torch.Tensor],
        warmup_state: WarmupResultState | None,
        mass_info: ResultMassInfo | None,
        n_divergent_total: int,
    ) -> InferenceResult:
        """Build the public immutable result for a successful run boundary."""
        payload = self._payload(
            reference=reference,
            warmup_state=warmup_state,
            mass_info=mass_info,
            n_divergent_total=n_divergent_total,
        )
        return InferenceResult(
            samples=payload.samples,
            log_post_history=payload.log_post_history,
            acceptance_rate=payload.acceptance_rate,
            requested_iters=payload.requested_iters,
            completed_iters=payload.completed_iters,
            stop_reason=stop_reason,
            metadata=payload.metadata,
        )

    def partial(
        self,
        *,
        reference: Mapping[str, torch.Tensor],
        warmup_state: WarmupResultState | None,
        mass_info: ResultMassInfo | None,
        n_divergent_total: int,
    ) -> _PartialInferenceResult:
        """Build the private immutable result attached to a structured error."""
        payload = self._payload(
            reference=reference,
            warmup_state=warmup_state,
            mass_info=mass_info,
            n_divergent_total=n_divergent_total,
        )
        return _PartialInferenceResult(
            samples=payload.samples,
            log_post_history=payload.log_post_history,
            acceptance_rate=payload.acceptance_rate,
            requested_iters=payload.requested_iters,
            completed_iters=payload.completed_iters,
            metadata=payload.metadata,
        )

    def _payload(
        self,
        *,
        reference: Mapping[str, torch.Tensor],
        warmup_state: WarmupResultState | None,
        mass_info: ResultMassInfo | None,
        n_divergent_total: int,
    ) -> _ResultPayload:
        """Assemble common result fields without compressed dict literals."""
        return _ResultPayload(
            samples=self._stack_chains(reference),
            log_post_history=torch.as_tensor(
                self.history,
                dtype=torch.float64,
            ),
            acceptance_rate=(
                self.acceptance_sum / self.accounting.completed_sampling
                if self.accounting.completed_sampling
                else 0.0
            ),
            requested_iters=self.requested_iters,
            completed_iters=self.accounting.completed_sampling,
            metadata=self._metadata(
                reference=reference,
                warmup_state=warmup_state,
                mass_info=mass_info,
                n_divergent_total=n_divergent_total,
            ),
        )

    def _stack_chains(
        self,
        reference: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Stack stored draws while preserving empty-chain shape and dtype."""
        return {
            name: (
                torch.stack(chunks, dim=0)
                if chunks
                else torch.empty(
                    (0, *reference[name].shape),
                    dtype=(
                        self.store_dtype
                        if self.store_dtype is not None
                        else reference[name].dtype
                    ),
                    device=reference[name].device,
                )
            )
            for name, chunks in self.chains.items()
        }

    def _metadata(
        self,
        *,
        reference: Mapping[str, torch.Tensor],
        warmup_state: WarmupResultState | None,
        mass_info: ResultMassInfo | None,
        n_divergent_total: int,
    ) -> dict[str, Any]:
        """Return the complete NUTS diagnostics namespace."""
        reference_tensor = next(iter(reference.values()))
        stored_draws = (
            len(next(iter(self.chains.values())))
            if self.chains
            else 0
        )
        mean_depth = (
            float(sum(self.depths) / len(self.depths))
            if self.depths
            else 0.0
        )
        step_size_final = (
            math.exp(warmup_state.dual_averaging.log_epsilon_bar)
            if warmup_state is not None
            else self.step_size_initial
        )
        mass_kinds = (
            {
                name: entry["kind"]
                for name, entry in mass_info.items()
            }
            if mass_info
            else {}
        )
        return {
            "sampler": "NUTS",
            "sample_layout": "draws",
            "generator_initial_seed": self.generator_initial_seed,
            "requested_warmup": self.warmup,
            "completed_warmup": self.accounting.completed_warmup,
            "thin": self.thin,
            "stored_draws": stored_draws,
            "accepted_sampling": self.accounting.accepted_sampling,
            "divergent_sampling": self.accounting.divergent_sampling,
            "continued_from_iteration": (
                self.accounting.continued_from_iteration
            ),
            "dtype": str(reference_tensor.dtype),
            "device": str(reference_tensor.device),
            "callback_phase": self.callback_phase,
            "callback_iteration": self.callback_iteration,
            "cumulative_completed_sampling": (
                self.accounting.cumulative_completed_sampling
            ),
            "cumulative_accepted_sampling": (
                self.accounting.cumulative_accepted_sampling
            ),
            "cumulative_divergent_sampling": (
                self.accounting.cumulative_divergent_sampling
            ),
            "step_size_initial": self.step_size_initial,
            "step_size_final": step_size_final,
            "max_depth": self.max_depth,
            "mean_depth": mean_depth,
            "warmup": self.warmup,
            "n_completed": (
                self.accounting.completed_warmup
                + self.accounting.cumulative_completed_sampling
            ),
            "n_divergent": self.accounting.divergent_sampling,
            "n_divergent_total": n_divergent_total,
            "mass_type_requested": self.mass_type_requested,
            "mass_adapted": (
                warmup_state.mass_adapted
                if warmup_state is not None
                else False
            ),
            "mass_injected": self.mass_injected,
            "mass_kinds": mass_kinds,
            "warmup_windowed": (
                warmup_state.schedule.windowed
                if warmup_state is not None
                else False
            ),
            "n_mass_updates": (
                warmup_state.n_mass_updates
                if warmup_state is not None
                else 0
            ),
        }
