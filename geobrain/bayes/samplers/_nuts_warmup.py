"""Step-size, mass-matrix, and window scheduling support for NUTS warmup.

The adaptation objects are immutable snapshots so the sampler can stage a
whole warmup transition before committing it. A slow-window close finalizes
the new metric, clears its Welford accumulator, reruns the reasonable-epsilon
search under that metric, and restarts dual averaging as one atomic update.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from ...core import GeoBrainError
from ...core.validation import (
    validate_bool,
    validate_finite_float,
    validate_int,
)
from ._hamiltonian import MassEntry, MassInfo, _kinetic, _sample_momentum
from ._nuts_tree import GradientEvaluator, HamiltonianState, leapfrog


MIN_WINDOWED_WARMUP = 20
EPSILON_SEARCH_MAX_STEPS = 50
FieldMassKinds = Mapping[str, str]
MutableMassInfo = dict[str, MassEntry]
EpsilonRestart = Callable[[float, MassInfo], float]


def validate_warmup_options(
    warmup: int,
    target_accept: float,
    adapt_mass: bool,
    mass_type: str,
    max_dense_dim: int,
) -> tuple[int, float, bool, str, int]:
    """Validate public NUTS adaptation controls at the owning boundary."""
    warmup = validate_int(warmup, owner="NUTS", field="warmup", minimum=0)
    target_accept = validate_finite_float(
        target_accept,
        owner="NUTS",
        field="target_accept",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if not 0.0 < target_accept < 1.0:
        raise GeoBrainError(
            "NUTS target_accept must be in (0, 1)",
            object_name="NUTS",
            field="target_accept",
            expected="(0, 1)",
            actual=target_accept,
        )
    if mass_type not in ("identity", "diagonal", "dense"):
        raise GeoBrainError(
            "NUTS mass_type must be 'identity', 'diagonal', or 'dense'",
            object_name="NUTS",
            field="mass_type",
            expected="identity|diagonal|dense",
            actual=mass_type,
        )
    return (
        warmup,
        target_accept,
        validate_bool(adapt_mass, owner="NUTS", field="adapt_mass"),
        mass_type,
        validate_int(
            max_dense_dim,
            owner="NUTS",
            field="max_dense_dim",
            minimum=1,
        ),
    )


def detached_mass_copy(
    mass_info: MassInfo | None,
) -> MutableMassInfo | None:
    """Return an owned exposure copy of an internal mass structure."""
    if mass_info is None:
        return None
    return {
        name: {
            key: (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in entry.items()
        }
        for name, entry in mass_info.items()
    }


@dataclass(frozen=True)
class DualAveraging:
    """One immutable Hoffman-Gelman dual-averaging state."""

    log_epsilon: float
    log_epsilon_bar: float
    h_bar: float
    mu: float
    gamma: float
    t0: float
    kappa: float
    iteration: int = 0

    @classmethod
    def initialize(
        cls,
        epsilon: float,
        *,
        windowed: bool,
    ) -> DualAveraging:
        """Create the historical plain or windowed NUTS adaptation anchor."""
        log_epsilon = math.log(epsilon)
        return cls(
            log_epsilon=log_epsilon,
            log_epsilon_bar=log_epsilon,
            h_bar=0.0,
            mu=log_epsilon if windowed else math.log(10.0 * epsilon),
            gamma=0.4 if windowed else 0.05,
            t0=10.0,
            kappa=1.0 if windowed else 0.75,
        )

    def update(
        self,
        acceptance: float,
        target_accept: float,
    ) -> DualAveraging:
        """Stage one dual-averaging update in the historical operation order."""
        iteration = self.iteration + 1
        h_bar = (
            1.0 - 1.0 / (iteration + self.t0)
        ) * self.h_bar + (
            1.0 / (iteration + self.t0)
        ) * (
            target_accept - acceptance
        )
        log_epsilon = self.mu - (
            math.sqrt(iteration) / self.gamma
        ) * h_bar
        eta = iteration ** (-self.kappa)
        log_epsilon_bar = (
            eta * log_epsilon
            + (1.0 - eta) * self.log_epsilon_bar
        )
        return DualAveraging(
            log_epsilon=log_epsilon,
            log_epsilon_bar=log_epsilon_bar,
            h_bar=h_bar,
            mu=self.mu,
            gamma=self.gamma,
            t0=self.t0,
            kappa=self.kappa,
            iteration=iteration,
        )

    def restart(self, epsilon: float) -> DualAveraging:
        """Restart under a new metric without changing schedule constants."""
        log_epsilon = math.log(epsilon)
        return DualAveraging(
            log_epsilon=log_epsilon,
            log_epsilon_bar=log_epsilon,
            h_bar=0.0,
            mu=log_epsilon,
            gamma=self.gamma,
            t0=self.t0,
            kappa=self.kappa,
            iteration=0,
        )


@dataclass(frozen=True)
class WarmupSchedule:
    """Scaled Stan warmup buffers and zero-based slow-window close indices."""

    init_buffer: int
    terminal_buffer: int
    window_ends: tuple[int, ...]
    slow_start: int
    slow_end: int
    windowed: bool

    @classmethod
    def for_warmup(
        cls,
        warmup: int,
        *,
        windowed: bool = True,
    ) -> WarmupSchedule:
        """Build the same scaled schedule previously embedded in ``NUTS``."""
        if not windowed:
            return cls(0, 0, (), 0, 0, False)
        if warmup >= 150:
            init_buffer, terminal_buffer, base = 75, 50, 25
        else:
            init_buffer = max(1, int(0.15 * warmup))
            terminal_buffer = max(1, int(0.10 * warmup))
            base = warmup - init_buffer - terminal_buffer
        slow_end = warmup - terminal_buffer
        ends: list[int] = []
        position, size = init_buffer, base
        while position < slow_end:
            end = position + size
            if end + 2 * size > slow_end:
                end = slow_end
            ends.append(end - 1)
            position, size = end, size * 2
        return cls(
            init_buffer=init_buffer,
            terminal_buffer=terminal_buffer,
            window_ends=tuple(ends),
            slow_start=init_buffer,
            slow_end=slow_end,
            windowed=True,
        )


def field_mass_kinds(
    params: Mapping[str, torch.Tensor],
    mass_type: str,
    max_dense_dim: int,
) -> dict[str, str]:
    """Resolve the per-field identity/diagonal/dense adaptation kinds."""
    kinds: dict[str, str] = {}
    for name, tensor in params.items():
        if mass_type == "identity":
            kinds[name] = "identity"
        elif mass_type == "diagonal":
            kinds[name] = "diagonal"
        else:
            kinds[name] = (
                "dense" if tensor.numel() <= max_dense_dim else "diagonal"
            )
    return kinds


def _clone_accumulators(
    accumulators: Mapping[str, MassEntry],
) -> MutableMassInfo:
    return {
        name: {
            key: (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in entry.items()
        }
        for name, entry in accumulators.items()
    }


@dataclass(frozen=True)
class WelfordMass:
    """Per-field online diagonal/dense covariance accumulators."""

    accumulators: MutableMassInfo
    n_observations: int = 0

    @classmethod
    def initialize(
        cls,
        params: Mapping[str, torch.Tensor],
        kinds: FieldMassKinds,
    ) -> WelfordMass:
        """Allocate accumulators for the requested field metric kinds."""
        accumulators: MutableMassInfo = {}
        for name, tensor in params.items():
            kind = kinds[name]
            if kind == "identity":
                accumulators[name] = {"kind": "identity"}
            elif kind == "diagonal":
                accumulators[name] = {
                    "kind": "diagonal",
                    "mean": torch.zeros_like(tensor),
                    "M2": torch.zeros_like(tensor),
                }
            else:
                size = tensor.numel()
                accumulators[name] = {
                    "kind": "dense",
                    "mean_flat": torch.zeros(
                        size,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                    "M2_flat": torch.zeros(
                        size,
                        size,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    ),
                    "field_shape": tuple(tensor.shape),
                }
        return cls(accumulators)

    def update(
        self,
        position: Mapping[str, torch.Tensor],
    ) -> WelfordMass:
        """Return an owned accumulator snapshot after observing a position."""
        accumulators = _clone_accumulators(self.accumulators)
        count = self.n_observations + 1
        for name, tensor in position.items():
            entry = accumulators[name]
            if entry["kind"] == "identity":
                continue
            if entry["kind"] == "diagonal":
                delta = tensor - entry["mean"]
                entry["mean"] = entry["mean"] + delta / count
                delta2 = tensor - entry["mean"]
                entry["M2"] = entry["M2"] + delta * delta2
            else:
                flat = tensor.reshape(-1)
                delta = flat - entry["mean_flat"]
                entry["mean_flat"] = entry["mean_flat"] + delta / count
                delta2 = flat - entry["mean_flat"]
                entry["M2_flat"] = (
                    entry["M2_flat"]
                    + delta.unsqueeze(-1) * delta2.unsqueeze(-2)
                )
        return WelfordMass(accumulators, count)

    def finalize(self) -> MutableMassInfo | None:
        """Convert accumulated covariance estimates into mass entries."""
        if self.n_observations < 2:
            return None
        mass_info: MutableMassInfo = {}
        for name, entry in self.accumulators.items():
            if entry["kind"] == "identity":
                continue
            if entry["kind"] == "diagonal":
                variance = entry["M2"] / (self.n_observations - 1)
                variance = variance.clamp(min=1e-10)
                mass_info[name] = {
                    "kind": "diagonal",
                    "diag": 1.0 / variance,
                }
                continue

            size = entry["M2_flat"].shape[0]
            covariance = entry["M2_flat"] / (self.n_observations - 1)
            diagonal_fallback: MassEntry = {
                "kind": "diagonal",
                "diag": (
                    1.0 / covariance.diagonal().clamp(min=1e-10)
                ).reshape(entry["field_shape"]),
            }
            if self.n_observations <= size + 1:
                mass_info[name] = diagonal_fallback
                continue
            jitter = 1e-6 * covariance.diagonal().mean().clamp(min=1e-12)
            covariance = covariance + jitter * torch.eye(
                size,
                dtype=covariance.dtype,
                device=covariance.device,
            )
            try:
                mass = torch.linalg.inv(covariance)
                cholesky = torch.linalg.cholesky(mass)
            except RuntimeError:
                mass_info[name] = diagonal_fallback
                continue
            mass_info[name] = {
                "kind": "dense",
                "M_inv": covariance,
                "chol_M": cholesky,
                "field_shape": entry["field_shape"],
            }
        return mass_info if mass_info else None


@dataclass(frozen=True)
class WarmupState:
    """Committed NUTS adaptation state, including metric/restart evidence."""

    dual_averaging: DualAveraging
    mass_info: MutableMassInfo | None
    mass_adapted: bool
    welford: WelfordMass | None
    schedule: WarmupSchedule
    next_window: int = 0
    n_mass_updates: int = 0
    metric_update_iters: tuple[int, ...] = ()
    epsilon_restart_iters: tuple[int, ...] = ()

    def epsilon(self, *, in_warmup: bool) -> float:
        """Return the live warmup epsilon or smoothed sampling epsilon."""
        log_epsilon = (
            self.dual_averaging.log_epsilon
            if in_warmup
            else self.dual_averaging.log_epsilon_bar
        )
        return math.exp(log_epsilon)

    def advance(
        self,
        *,
        warmup_index: int,
        total_warmup: int,
        acceptance: float,
        target_accept: float,
        position: Mapping[str, torch.Tensor],
        adapt_mass: bool,
        params: Mapping[str, torch.Tensor],
        kinds: FieldMassKinds,
        restart_epsilon: EpsilonRestart,
    ) -> WarmupState:
        """Stage one atomic warmup update, including any window close."""
        dual = self.dual_averaging.update(acceptance, target_accept)
        welford = self.welford
        if (
            adapt_mass
            and welford is not None
            and (
                not self.schedule.windowed
                or self.schedule.slow_start
                <= warmup_index
                < self.schedule.slow_end
            )
        ):
            welford = welford.update(position)

        next_window = self.next_window
        mass_info = self.mass_info
        mass_adapted = self.mass_adapted
        n_mass_updates = self.n_mass_updates
        metric_update_iters = self.metric_update_iters
        epsilon_restart_iters = self.epsilon_restart_iters
        if (
            self.schedule.windowed
            and next_window < len(self.schedule.window_ends)
            and warmup_index == self.schedule.window_ends[next_window]
        ):
            next_window += 1
            new_mass = welford.finalize() if welford is not None else None
            if new_mass is not None:
                mass_info = new_mass
                mass_adapted = True
                n_mass_updates += 1
                metric_update_iters += (warmup_index,)
                welford = WelfordMass.initialize(params, kinds)
                epsilon = restart_epsilon(
                    math.exp(dual.log_epsilon_bar),
                    new_mass,
                )
                dual = dual.restart(epsilon)
                epsilon_restart_iters += (warmup_index,)
        elif (
            not self.schedule.windowed
            and warmup_index == total_warmup - 1
            and adapt_mass
            and welford is not None
            and welford.n_observations > 1
        ):
            new_mass = welford.finalize()
            mass_info = new_mass
            mass_adapted = new_mass is not None
            if new_mass is not None:
                n_mass_updates += 1
                metric_update_iters += (warmup_index,)

        return WarmupState(
            dual_averaging=dual,
            mass_info=mass_info,
            mass_adapted=mass_adapted,
            welford=welford,
            schedule=self.schedule,
            next_window=next_window,
            n_mass_updates=n_mass_updates,
            metric_update_iters=metric_update_iters,
            epsilon_restart_iters=epsilon_restart_iters,
        )


def uses_windowed_warmup(
    warmup: int,
    adapt_mass: bool,
    kinds: FieldMassKinds,
) -> bool:
    """Whether this configuration has a non-degenerate slow adaptation window."""
    return (
        adapt_mass
        and warmup >= MIN_WINDOWED_WARMUP
        and any(kind != "identity" for kind in kinds.values())
    )


def initialize_warmup_state(
    epsilon: float,
    *,
    warmup: int,
    adapt_mass: bool,
    params: Mapping[str, torch.Tensor],
    kinds: FieldMassKinds,
    mass_info: MutableMassInfo | None,
    windowed: bool,
) -> WarmupState:
    """Create the owned adaptation snapshot after any initial epsilon search."""
    return WarmupState(
        dual_averaging=DualAveraging.initialize(
            epsilon,
            windowed=windowed,
        ),
        mass_info=mass_info,
        mass_adapted=False,
        welford=(
            WelfordMass.initialize(params, kinds)
            if adapt_mass
            else None
        ),
        schedule=WarmupSchedule.for_warmup(
            warmup,
            windowed=windowed,
        ),
    )


def find_reasonable_epsilon(
    position: Mapping[str, torch.Tensor],
    log_prob: float,
    gradient: Mapping[str, torch.Tensor],
    epsilon_initial: float,
    mass_info: MassInfo | None,
    generator: torch.Generator | None,
    evaluate_gradient: GradientEvaluator,
) -> float:
    """Run Hoffman-Gelman Algorithm 4 under the supplied current metric."""
    if not math.isfinite(log_prob):
        return epsilon_initial
    momentum = _sample_momentum(position, mass_info, generator)
    reference = next(iter(position.values()))
    state = HamiltonianState(
        position=position,
        momentum=momentum,
        gradient=gradient,
        log_prob=torch.as_tensor(
            log_prob,
            dtype=reference.dtype,
            device=reference.device,
        ),
    )
    log_joint_initial = log_prob - float(
        _kinetic(momentum, mass_info).item()
    )

    def log_ratio(epsilon: float) -> float:
        next_state = leapfrog(
            state,
            epsilon,
            mass_info,
            evaluate_gradient,
        )
        log_joint = float(next_state.log_prob.item()) - float(
            _kinetic(next_state.momentum, mass_info).item()
        )
        ratio = log_joint - log_joint_initial
        return ratio if math.isfinite(ratio) else float("-inf")

    log_half = -math.log(2.0)
    epsilon = epsilon_initial
    ratio = log_ratio(epsilon)
    direction = 1.0 if ratio > log_half else -1.0
    for _ in range(EPSILON_SEARCH_MAX_STEPS):
        if not direction * ratio > direction * log_half:
            break
        epsilon *= 2.0 ** direction
        if not 1e-10 < epsilon < 1e7:
            break
        ratio = log_ratio(epsilon)
    return epsilon


def build_injected_mass_info(
    mass_matrix: Mapping[str, torch.Tensor],
    *,
    params: Mapping[str, torch.Tensor],
    adapt_mass: bool,
) -> MutableMassInfo:
    """Validate and normalize user-provided per-field NUTS mass matrices."""
    if adapt_mass:
        raise GeoBrainError(
            "NUTS mass_matrix injection requires adapt_mass=False "
            "(injection and warmup adaptation are mutually exclusive)",
            object_name="NUTS",
            field="mass_matrix",
            expected="adapt_mass=False when mass_matrix is given",
            actual="adapt_mass=True",
        )
    if not isinstance(mass_matrix, Mapping):
        raise GeoBrainError(
            "NUTS mass_matrix must be a mapping of field name -> tensor",
            object_name="NUTS",
            field="mass_matrix",
            expected="Mapping[str, torch.Tensor]",
            actual=type(mass_matrix),
        )
    unknown = sorted(set(mass_matrix) - set(params))
    if unknown:
        raise GeoBrainError(
            "NUTS mass_matrix names fields absent from params",
            object_name="NUTS",
            field="mass_matrix",
            expected=f"subset of {sorted(params)}",
            actual=unknown,
        )
    info: MutableMassInfo = {}
    for name, tensor in params.items():
        matrix = mass_matrix.get(name)
        if matrix is None:
            info[name] = {
                "kind": "diagonal",
                "diag": torch.ones_like(tensor),
            }
            continue
        if not isinstance(matrix, torch.Tensor):
            raise GeoBrainError(
                "NUTS mass_matrix values must be torch.Tensor",
                object_name="NUTS",
                field=f"mass_matrix[{name!r}]",
                expected=torch.Tensor,
                actual=type(matrix),
            )
        size = tensor.numel()
        if matrix.ndim == 2:
            if matrix.shape != (size, size):
                raise GeoBrainError(
                    "NUTS dense mass_matrix entry has wrong shape",
                    object_name="NUTS",
                    field=f"mass_matrix[{name!r}]",
                    expected=f"({size}, {size}) for field of numel {size}",
                    actual=tuple(matrix.shape),
                )
            matrix = matrix.detach().to(
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if not torch.isfinite(matrix).all():
                raise GeoBrainError(
                    "NUTS dense mass_matrix entry has non-finite values",
                    object_name="NUTS",
                    field=f"mass_matrix[{name!r}]",
                    expected="finite SPD matrix",
                    actual="non-finite entries",
                )
            if not torch.allclose(
                matrix,
                matrix.mT,
                rtol=1e-6,
                atol=1e-10,
            ):
                raise GeoBrainError(
                    "NUTS dense mass_matrix entry is not symmetric",
                    object_name="NUTS",
                    field=f"mass_matrix[{name!r}]",
                    expected="symmetric positive-definite matrix",
                    actual="asymmetric matrix",
                )
            try:
                cholesky = torch.linalg.cholesky(matrix)
            except RuntimeError as exc:
                raise GeoBrainError(
                    "NUTS dense mass_matrix entry is not positive-definite",
                    object_name="NUTS",
                    field=f"mass_matrix[{name!r}]",
                    expected="symmetric positive-definite matrix",
                    actual=f"cholesky failed: {exc}",
                ) from exc
            info[name] = {
                "kind": "dense",
                "M_inv": torch.cholesky_inverse(cholesky),
                "chol_M": cholesky,
                "field_shape": tuple(tensor.shape),
            }
            continue

        if matrix.numel() != size:
            raise GeoBrainError(
                "NUTS diagonal mass_matrix entry has wrong size",
                object_name="NUTS",
                field=f"mass_matrix[{name!r}]",
                expected=(
                    f"numel {size} (diagonal) or shape ({size}, {size}) "
                    "(dense)"
                ),
                actual=(
                    f"numel {matrix.numel()}, shape {tuple(matrix.shape)}"
                ),
            )
        matrix = matrix.detach().to(
            dtype=tensor.dtype,
            device=tensor.device,
        ).reshape(tensor.shape)
        if not torch.isfinite(matrix).all() or not bool((matrix > 0).all()):
            raise GeoBrainError(
                "NUTS diagonal mass_matrix entry must be finite and "
                "strictly positive",
                object_name="NUTS",
                field=f"mass_matrix[{name!r}]",
                expected="finite, > 0 (positive-definite diagonal)",
                actual=matrix,
            )
        info[name] = {"kind": "diagonal", "diag": matrix.clone()}
    return info
