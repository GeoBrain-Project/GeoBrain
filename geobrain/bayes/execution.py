"""Shared Bayesian chain configuration, snapshots, and run accounting.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import torch

from ..core.errors import GeoBrainError
from ..core.validation import validate_bool, validate_int


# A hard-wall start may legitimately require several trajectories before it
# enters finite support.  The budget is deliberately generous, deterministic,
# and shared by every sampler that supports this escape path.  It counts only
# consecutive uncommitted attempts whose selected state is still non-finite.
_MAX_CONSECUTIVE_NONFINITE_ATTEMPTS = 128


class _NonfiniteEscapeExhausted(Exception):
    """Internal marker carrying the structured hard-wall exhaustion error."""

    def __init__(self, error: GeoBrainError) -> None:
        super().__init__(str(error))
        self.error = error


class _NonfiniteEscapeBudget:
    """Track consecutive uncommitted attempts to enter finite target support."""

    __slots__ = ("_attempts", "_owner")

    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._attempts = 0

    def observe(self, *, finite: bool) -> None:
        """Reset on finite support or fail after the shared deterministic limit."""
        if finite:
            self._attempts = 0
            return
        self._attempts += 1
        if self._attempts < _MAX_CONSECUTIVE_NONFINITE_ATTEMPTS:
            return
        maximum = _MAX_CONSECUTIVE_NONFINITE_ATTEMPTS
        error = GeoBrainError(
            f"{self._owner} could not enter finite target support after "
            f"{maximum} consecutive uncommitted attempts",
            object_name=self._owner,
            field="target",
            expected=(
                "finite target support within "
                f"{maximum} consecutive uncommitted attempts"
            ),
            actual={
                "consecutive_nonfinite_attempts": self._attempts,
                "maximum_consecutive_nonfinite_attempts": maximum,
            },
        )
        raise _NonfiniteEscapeExhausted(error)


class _CallbackTransformFailure(Exception):
    """Internal marker separating callback remapping from user callback code."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


@dataclass(frozen=True)
class ChainConfig:
    """Immutable storage and warmup configuration for an MCMC chain.

    Attributes:
        warmup: total number of once-only warmup transitions owned by a
            sampler instance.
        thin: sample-STORAGE stride only; transition accounting and the
            log-posterior history remain dense.
        store_dtype: cast applied to owned stored draws, without changing
            sampler arithmetic (``None`` keeps the sampling dtype).
    """

    warmup: int = 0
    thin: int = 1
    store_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Validate all constructor boundaries with structured errors."""
        warmup = validate_int(
            self.warmup,
            owner="ChainConfig",
            field="warmup",
            minimum=0,
        )
        thin = validate_int(
            self.thin,
            owner="ChainConfig",
            field="thin",
            minimum=1,
        )
        store_dtype = self.store_dtype
        if store_dtype is not None:
            if not isinstance(store_dtype, torch.dtype):
                raise GeoBrainError(
                    "ChainConfig.store_dtype must be a torch.dtype or None",
                    object_name="ChainConfig",
                    field="store_dtype",
                    expected="floating-point / complex torch.dtype | None",
                    actual=type(store_dtype),
                )
            if not (store_dtype.is_floating_point or store_dtype.is_complex):
                raise GeoBrainError(
                    "ChainConfig.store_dtype must not truncate stored draws",
                    object_name="ChainConfig",
                    field="store_dtype",
                    expected="floating-point / complex torch.dtype | None",
                    actual=store_dtype,
                )
        object.__setattr__(self, "warmup", warmup)
        object.__setattr__(self, "thin", thin)


class SamplerStopReason(str, Enum):
    """Why a Bayesian ``run`` call returned before control reached its caller."""

    COMPLETED = "completed"
    CALLBACK = "callback"


class RunAccounting:
    """Validated call-local counters with read-only public observations.

    ``requested_iters`` and every non-``continued_*`` sampling counter describe
    one ``run`` call. ``completed_warmup`` is cumulative because warmup belongs
    to the sampler instance and is performed only once. The ``continued_*``
    values snapshot cumulative sampler state at call entry; cumulative
    properties combine those offsets with committed call-local work.

    Public counters are properties without setters. Transitions can change
    accounting only through :meth:`commit_warmup` and
    :meth:`commit_sampling`, which preserve all counter relations.

    Args:
        requested_iters: sampling transitions requested for THIS call.
        warmup_iters: warmup transitions the sampler instance owns.
        completed_warmup: cumulative warmup already performed at entry.
        completed_sampling: call-local committed sampling transitions.
        accepted_sampling: call-local accepted transitions.
        divergent_sampling: call-local divergent transitions.
        continued_from_iteration: cumulative iteration count at call entry.
        continued_accepted_sampling: cumulative accepted count at entry.
        continued_divergent_sampling: cumulative divergent count at entry.
    """

    _requested_iters: int
    _warmup_iters: int
    _completed_warmup: int
    _completed_sampling: int
    _accepted_sampling: int
    _divergent_sampling: int
    _continued_from_iteration: int
    _continued_accepted_sampling: int
    _continued_divergent_sampling: int

    __slots__ = (
        "_requested_iters",
        "_warmup_iters",
        "_completed_warmup",
        "_completed_sampling",
        "_accepted_sampling",
        "_divergent_sampling",
        "_continued_from_iteration",
        "_continued_accepted_sampling",
        "_continued_divergent_sampling",
    )

    def __init__(
        self,
        requested_iters: int,
        warmup_iters: int,
        completed_warmup: int = 0,
        completed_sampling: int = 0,
        accepted_sampling: int = 0,
        divergent_sampling: int = 0,
        continued_from_iteration: int = 0,
        continued_accepted_sampling: int = 0,
        continued_divergent_sampling: int = 0,
    ) -> None:
        """Reject booleans, negative counts, and impossible counter relations."""
        values = {
            "requested_iters": requested_iters,
            "warmup_iters": warmup_iters,
            "completed_warmup": completed_warmup,
            "completed_sampling": completed_sampling,
            "accepted_sampling": accepted_sampling,
            "divergent_sampling": divergent_sampling,
            "continued_from_iteration": continued_from_iteration,
            "continued_accepted_sampling": continued_accepted_sampling,
            "continued_divergent_sampling": continued_divergent_sampling,
        }
        for field_name in (
            "requested_iters",
            "warmup_iters",
            "completed_warmup",
            "completed_sampling",
            "accepted_sampling",
            "divergent_sampling",
            "continued_from_iteration",
            "continued_accepted_sampling",
            "continued_divergent_sampling",
        ):
            value = validate_int(
                values[field_name],
                owner="RunAccounting",
                field=field_name,
                minimum=0,
            )
            setattr(self, f"_{field_name}", value)
        if self.completed_warmup > self.warmup_iters:
            self._invalid(
                "completed_warmup",
                f"<= warmup_iters ({self.warmup_iters})",
                self.completed_warmup,
            )
        if self.completed_sampling > self.requested_iters:
            self._invalid(
                "completed_sampling",
                f"<= requested_iters ({self.requested_iters})",
                self.completed_sampling,
            )
        if self.accepted_sampling > self.completed_sampling:
            self._invalid(
                "accepted_sampling",
                f"<= completed_sampling ({self.completed_sampling})",
                self.accepted_sampling,
            )
        if self.divergent_sampling > self.completed_sampling:
            self._invalid(
                "divergent_sampling",
                f"<= completed_sampling ({self.completed_sampling})",
                self.divergent_sampling,
            )
        if self.continued_accepted_sampling > self.continued_from_iteration:
            self._invalid(
                "continued_accepted_sampling",
                f"<= continued_from_iteration ({self.continued_from_iteration})",
                self.continued_accepted_sampling,
            )
        if self.continued_divergent_sampling > self.continued_from_iteration:
            self._invalid(
                "continued_divergent_sampling",
                f"<= continued_from_iteration ({self.continued_from_iteration})",
                self.continued_divergent_sampling,
            )

    @property
    def requested_iters(self) -> int:
        """Requested post-warmup transitions for this call."""
        return self._requested_iters

    @property
    def warmup_iters(self) -> int:
        """Total once-only warmup transitions for the sampler instance."""
        return self._warmup_iters

    @property
    def completed_warmup(self) -> int:
        """Cumulative committed warmup transitions."""
        return self._completed_warmup

    @property
    def completed_sampling(self) -> int:
        """Committed post-warmup transitions in this call."""
        return self._completed_sampling

    @property
    def accepted_sampling(self) -> int:
        """Accepted post-warmup transitions in this call."""
        return self._accepted_sampling

    @property
    def divergent_sampling(self) -> int:
        """Divergent post-warmup transitions in this call."""
        return self._divergent_sampling

    @property
    def continued_from_iteration(self) -> int:
        """Cumulative committed sampling count at call entry."""
        return self._continued_from_iteration

    @property
    def continued_accepted_sampling(self) -> int:
        """Cumulative accepted sampling count at call entry."""
        return self._continued_accepted_sampling

    @property
    def continued_divergent_sampling(self) -> int:
        """Cumulative divergent sampling count at call entry."""
        return self._continued_divergent_sampling

    @staticmethod
    def _invalid(field: str, expected: object, actual: object) -> None:
        raise GeoBrainError(
            f"RunAccounting.{field} is inconsistent with committed work",
            object_name="RunAccounting",
            field=field,
            expected=expected,
            actual=actual,
        )

    @property
    def cumulative_completed_sampling(self) -> int:
        """Committed sampling transitions across this instance, including this call."""
        return self.continued_from_iteration + self.completed_sampling

    @property
    def cumulative_accepted_sampling(self) -> int:
        """Accepted sampling transitions across this instance, including this call."""
        return self.continued_accepted_sampling + self.accepted_sampling

    @property
    def cumulative_divergent_sampling(self) -> int:
        """Divergent sampling transitions across this instance, including this call."""
        return self.continued_divergent_sampling + self.divergent_sampling

    @property
    def acceptance_rate(self) -> float:
        """Call-local accepted fraction; zero when no sampling transition committed."""
        if self.completed_sampling == 0:
            return 0.0
        return self.accepted_sampling / self.completed_sampling

    def commit_warmup(self) -> None:
        """Commit one successful warmup transition."""
        if self.completed_warmup >= self.warmup_iters:
            self._invalid(
                "completed_warmup",
                f"< warmup_iters ({self.warmup_iters}) before commit",
                self.completed_warmup,
            )
        self._completed_warmup += 1

    def commit_sampling(
        self,
        *,
        accepted: bool,
        divergent: bool = False,
    ) -> None:
        """Commit one successful sampling transition and its diagnostics."""
        accepted = validate_bool(
            accepted,
            owner="RunAccounting.commit_sampling",
            field="accepted",
        )
        divergent = validate_bool(
            divergent,
            owner="RunAccounting.commit_sampling",
            field="divergent",
        )
        if self.completed_sampling >= self.requested_iters:
            self._invalid(
                "completed_sampling",
                f"< requested_iters ({self.requested_iters}) before commit",
                self.completed_sampling,
            )
        self._completed_sampling += 1
        self._accepted_sampling += int(accepted)
        self._divergent_sampling += int(divergent)


def callback_snapshot(
    values: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Return a read-only mapping over detached tensor clones for callbacks."""
    return MappingProxyType(
        {name: tensor.detach().clone() for name, tensor in values.items()}
    )


def stored_draw(
    tensor: torch.Tensor,
    store_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Return one detached owned draw, optionally cast for storage."""
    detached = tensor.detach()
    if store_dtype is None or detached.dtype == store_dtype:
        return detached.clone()
    return detached.to(store_dtype)


def validate_chain_storage(
    owner: str,
    thin: int,
    store_dtype: torch.dtype | None,
) -> tuple[int, torch.dtype | None]:
    """Validate legacy sampler constructor arguments through :class:`ChainConfig`."""
    try:
        config = ChainConfig(thin=thin, store_dtype=store_dtype)
    except GeoBrainError as exc:
        raise GeoBrainError(
            str(exc).replace("ChainConfig", owner),
            object_name=owner,
            field=exc.field,
            expected=exc.expected,
            actual=exc.actual,
        ) from exc
    return config.thin, config.store_dtype
