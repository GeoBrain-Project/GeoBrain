"""Differentiable batched QR/SVD Euler deconvolution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Final, NoReturn, cast

import torch

from geobrain.core import ErrorCode

from .errors import PotentialContractError
from .processing import horizontal_gradient, vertical_derivative


STRUCTURAL_INDEX: Final[dict[str, float]] = {
    "mag_sphere": 3.0,
    "mag_dipole": 3.0,
    "mag_pipe": 2.0,
    "mag_dyke": 1.0,
    "mag_contact": 0.0,
    "grav_sphere": 2.0,
    "grav_pipe": 1.0,
    "grav_cylinder": 1.0,
    "grav_sheet": 0.0,
    "grav_dyke": 0.0,
    "grav_contact": -1.0,
}


@dataclass(frozen=True, slots=True)
class EulerConfig:
    """Numerical controls for windowed Euler deconvolution.

    Attributes:
        window: square moving-window size [samples].
        stride: window stride [samples].
        rank_rtol: relative tolerance of the rank test.
        max_condition_number: reject windows above this conditioning.
        batch_windows: windows solved per batch.
    """

    window: int = 11
    stride: int = 1
    rank_rtol: float = 1.0e-10
    max_condition_number: float = 1.0e10
    batch_windows: int = 256


@dataclass(frozen=True, slots=True)
class EulerResult:
    """Per-window QR/SVD solutions and numerical diagnostics.

    Attributes:
        solutions: ``(n, 4)`` estimated ``(x, y, z, background)`` rows.
        uncertainty: per-solution uncertainty estimates.
        residual_norm: per-window residual norms.
        rank / condition_number: per-window solve diagnostics.
        accepted: mask of windows passing the quality gates.
        solver: which least-squares path solved the windows.
    """

    solutions: torch.Tensor
    uncertainty: torch.Tensor
    residual_norm: torch.Tensor
    rank: torch.Tensor
    condition_number: torch.Tensor
    accepted: torch.Tensor
    solver: tuple[str, ...]


def _contract_error(
    message: str,
    *,
    field: str,
    expected: object,
    actual: object,
    code: ErrorCode = ErrorCode.CONFIG_INVALID,
) -> NoReturn:
    raise PotentialContractError(
        message,
        object_name="euler_deconvolution",
        field=field,
        expected=expected,
        actual=actual,
        code=code,
        hint="Provide a value satisfying the documented Euler QR/SVD contract.",
    )


def _finite_real(
    value: object,
    *,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _contract_error(
            "Euler scalar parameters must be real numbers.",
            field=field,
            expected="finite real scalar",
            actual=value,
        )
    try:
        resolved = float(value)
    except (OverflowError, ValueError):
        _contract_error(
            "Euler scalar parameter is not representable as a finite float.",
            field=field,
            expected="finite float-representable real scalar",
            actual=value,
        )
    if not math.isfinite(resolved):
        _contract_error(
            "Euler scalar parameters must be finite.",
            field=field,
            expected="finite real scalar",
            actual=value,
        )
    if positive and resolved <= 0.0:
        _contract_error(
            "Euler parameter must be strictly positive.",
            field=field,
            expected="> 0",
            actual=resolved,
        )
    if nonnegative and resolved < 0.0:
        _contract_error(
            "Euler parameter must be non-negative.",
            field=field,
            expected=">= 0",
            actual=resolved,
        )
    return resolved


def _validate_data(data: object) -> torch.Tensor:
    """Return one finite, materialized, real two-dimensional grid."""
    if not isinstance(data, torch.Tensor):
        _contract_error(
            "Euler deconvolution requires a torch.Tensor grid.",
            field="data",
            expected="finite real float32/float64 torch.Tensor [ny, nx]",
            actual=data,
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if data.is_nested or data.layout != torch.strided:
        _contract_error(
            "Euler deconvolution requires a regular strided tensor grid.",
            field="data.layout",
            expected=str(torch.strided),
            actual=data.layout,
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
        )
    if data.dtype not in {torch.float32, torch.float64}:
        _contract_error(
            "Euler deconvolution supports only real float32/float64 grids.",
            field="data.dtype",
            expected=[str(torch.float32), str(torch.float64)],
            actual=data.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )
    if data.device.type == "meta":
        _contract_error(
            "Euler deconvolution requires a materialized device.",
            field="data.device",
            expected="materialized CPU or accelerator tensor",
            actual=data.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    if data.ndim != 2 or data.numel() == 0 or 0 in data.shape:
        _contract_error(
            "Euler deconvolution requires a non-empty two-dimensional grid.",
            field="data",
            expected="non-empty [ny, nx]",
            actual=tuple(data.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if not bool(torch.isfinite(data).all()):
        _contract_error(
            "Euler deconvolution requires finite grid values.",
            field="data",
            expected="all finite values",
            actual="contains non-finite values",
        )
    return data


def _validate_config(config: object, *, ny: int, nx: int) -> EulerConfig:
    """Validate the immutable Euler configuration before making derivatives."""
    if not isinstance(config, EulerConfig):
        _contract_error(
            "Euler deconvolution requires an EulerConfig instance.",
            field="config",
            expected="EulerConfig",
            actual=config,
        )
    if isinstance(config.window, bool) or not isinstance(config.window, int):
        _contract_error(
            "Euler window must be an integer.",
            field="config.window",
            expected="odd integer >= 3",
            actual=config.window,
        )
    if config.window < 3 or config.window % 2 == 0 or config.window > min(ny, nx):
        _contract_error(
            "Euler window must be odd, at least three, and fit the grid.",
            field="config.window",
            expected=f"odd integer in [3, {min(ny, nx)}]",
            actual=config.window,
        )
    for field, value in (("config.stride", config.stride), ("config.batch_windows", config.batch_windows)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _contract_error(
                "Euler stride and batch_windows must be positive integers.",
                field=field,
                expected="positive integer",
                actual=value,
            )
    _finite_real(config.rank_rtol, field="config.rank_rtol", positive=True)
    _finite_real(
        config.max_condition_number,
        field="config.max_condition_number",
        positive=True,
    )
    return config


def _window_rows(data: torch.Tensor, *, window: int, stride: int) -> torch.Tensor:
    """Return overlapping two-axis ``Tensor.unfold`` windows as a view."""
    return data.unfold(0, window, stride).unfold(1, window, stride)


def _materialize_window_rows(
    windows: torch.Tensor,
    *,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Materialize only one row-major linear batch of overlapping windows."""
    windows_per_row = windows.shape[1]
    linear_indices = torch.arange(start, stop, device=windows.device)
    row_indices = torch.div(linear_indices, windows_per_row, rounding_mode="floor")
    column_indices = torch.remainder(linear_indices, windows_per_row)
    return windows[row_indices, column_indices].flatten(start_dim=1)


def _svd_solution(
    u: torch.Tensor,
    singular_values: torch.Tensor,
    vh: torch.Tensor,
    rhs: torch.Tensor,
    *,
    rank_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pseudoinverse solution, rank, and SVD covariance diagonal."""
    threshold = singular_values[:, :1] * rank_rtol
    retained = singular_values > threshold
    rank = retained.sum(dim=-1)
    safe_singular_values = singular_values.masked_fill(~retained, 1.0)
    inverse_singular_values = retained.to(dtype=rhs.dtype) / safe_singular_values
    projected_rhs = torch.matmul(u.transpose(-2, -1), rhs.unsqueeze(-1)).squeeze(-1)
    solution = torch.matmul(
        vh.transpose(-2, -1),
        (projected_rhs * inverse_singular_values).unsqueeze(-1),
    ).squeeze(-1)
    covariance_diagonal = (
        vh.transpose(-2, -1).square() * inverse_singular_values.square().unsqueeze(-2)
    ).sum(dim=-1)
    return solution, rank, covariance_diagonal


def _qr_solution_and_covariance(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve full-rank systems and covariance factors without normal equations."""
    q_factor, r_factor = torch.linalg.qr(matrix, mode="reduced")
    solution = torch.linalg.solve_triangular(
        r_factor,
        torch.matmul(q_factor.transpose(-2, -1), rhs.unsqueeze(-1)),
        upper=True,
    ).squeeze(-1)
    identity = torch.eye(4, dtype=matrix.dtype, device=matrix.device).expand(
        matrix.shape[0], -1, -1
    )
    inverse_r = torch.linalg.solve_triangular(r_factor, identity, upper=True)
    return solution, inverse_r.square().sum(dim=-1)


_EulerTensorResults = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_EulerWindowViews = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _prepare_window_views(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    config: EulerConfig,
    x_origin_m: float,
    y_origin_m: float,
) -> tuple[_EulerWindowViews, int]:
    """Build derivative grids and shared-storage overlapping window views."""
    ny, nx = data.shape
    df_dx, df_dy, _ = horizontal_gradient(data, dx_m=dx_m, dy_m=dy_m)
    df_dz = vertical_derivative(data, dx_m=dx_m, dy_m=dy_m, order=1)
    x_coordinates = x_origin_m + (torch.arange(
        nx, dtype=data.dtype, device=data.device
    ) + 0.5) * dx_m
    y_coordinates = y_origin_m + (torch.arange(
        ny, dtype=data.dtype, device=data.device
    ) + 0.5) * dy_m
    if not bool(torch.isfinite(x_coordinates).all()) or not bool(
        torch.isfinite(y_coordinates).all()
    ):
        _contract_error(
            "Euler grid coordinates are not representable in the input dtype.",
            field="dx_m/dy_m/x_origin_m/y_origin_m",
            expected="finite representable grid coordinates",
            actual={
                "dx_m": dx_m,
                "dy_m": dy_m,
                "x_origin_m": x_origin_m,
                "y_origin_m": y_origin_m,
                "dtype": data.dtype,
            },
        )
    x_grid, y_grid = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

    data_windows = _window_rows(data, window=config.window, stride=config.stride)
    fx_windows = _window_rows(df_dx, window=config.window, stride=config.stride)
    fy_windows = _window_rows(df_dy, window=config.window, stride=config.stride)
    fz_windows = _window_rows(df_dz, window=config.window, stride=config.stride)
    x_windows = _window_rows(x_grid, window=config.window, stride=config.stride)
    y_windows = _window_rows(y_grid, window=config.window, stride=config.stride)
    windows = (
        data_windows,
        fx_windows,
        fy_windows,
        fz_windows,
        x_windows,
        y_windows,
    )
    return windows, data_windows.shape[0] * data_windows.shape[1]


def _solve_window_batch(
    windows: _EulerWindowViews,
    *,
    start: int,
    stop: int,
    structural_index: float,
    config: EulerConfig,
) -> _EulerTensorResults:
    """Solve one materialized batch and return values plus diagnostics."""
    data_windows, fx_windows, fy_windows, fz_windows, x_windows, y_windows = windows
    f = _materialize_window_rows(data_windows, start=start, stop=stop)
    fx = _materialize_window_rows(fx_windows, start=start, stop=stop)
    fy = _materialize_window_rows(fy_windows, start=start, stop=stop)
    fz = _materialize_window_rows(fz_windows, start=start, stop=stop)
    x = _materialize_window_rows(x_windows, start=start, stop=stop)
    y = _materialize_window_rows(y_windows, start=start, stop=stop)
    matrix = torch.stack(
        (fx, fy, fz, torch.full_like(fx, structural_index)),
        dim=-1,
    )
    rhs = x * fx + y * fy + structural_index * f
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    svd_solution, rank, svd_covariance = _svd_solution(
        u,
        singular_values,
        vh,
        rhs,
        rank_rtol=config.rank_rtol,
    )
    smallest = singular_values[:, -1]
    safe_smallest = smallest.masked_fill(smallest <= 0.0, 1.0)
    condition_number = torch.where(
        smallest > 0.0,
        singular_values[:, 0] / safe_smallest,
        torch.full_like(smallest, math.inf),
    )
    accepted = (
        (rank == 4)
        & torch.isfinite(condition_number)
        & (condition_number <= config.max_condition_number)
    )
    solutions = svd_solution
    covariance_diagonal = svd_covariance
    accepted_indices = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
    if accepted_indices.numel() > 0:
        qr_solution, qr_covariance = _qr_solution_and_covariance(
            matrix[accepted_indices], rhs[accepted_indices]
        )
        solutions = torch.index_copy(solutions, 0, accepted_indices, qr_solution)
        covariance_diagonal = torch.index_copy(
            covariance_diagonal,
            0,
            accepted_indices,
            qr_covariance,
        )
    residual_norm = torch.linalg.vector_norm(
        torch.matmul(matrix, solutions.unsqueeze(-1)).squeeze(-1) - rhs,
        dim=-1,
    )
    degrees_of_freedom = (config.window * config.window - rank).clamp_min(1).to(
        dtype=matrix.dtype
    )
    residual_variance = residual_norm.square() / degrees_of_freedom
    uncertainty = torch.sqrt(residual_variance.unsqueeze(-1) * covariance_diagonal)
    uncertainty = torch.where(
        (rank == 4).unsqueeze(-1),
        uncertainty,
        torch.full_like(uncertainty, math.inf),
    )
    return (
        solutions,
        uncertainty,
        residual_norm,
        rank,
        condition_number,
        accepted,
    )


def _stream_euler_tensors(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    structural_index: float,
    config: EulerConfig,
    x_origin_m: float,
    y_origin_m: float,
) -> _EulerTensorResults:
    """Compute all result tensors while materializing one window batch at a time."""
    windows, window_count = _prepare_window_views(
        data,
        dx_m=dx_m,
        dy_m=dy_m,
        config=config,
        x_origin_m=x_origin_m,
        y_origin_m=y_origin_m,
    )

    solution_batches: list[torch.Tensor] = []
    uncertainty_batches: list[torch.Tensor] = []
    residual_batches: list[torch.Tensor] = []
    rank_batches: list[torch.Tensor] = []
    condition_batches: list[torch.Tensor] = []
    accepted_batches: list[torch.Tensor] = []

    for start in range(0, window_count, config.batch_windows):
        stop = min(start + config.batch_windows, window_count)
        (
            solutions,
            uncertainty,
            residual_norm,
            rank,
            condition_number,
            accepted,
        ) = _solve_window_batch(
            windows,
            start=start,
            stop=stop,
            structural_index=structural_index,
            config=config,
        )
        solution_batches.append(solutions)
        uncertainty_batches.append(uncertainty)
        residual_batches.append(residual_norm)
        rank_batches.append(rank)
        condition_batches.append(condition_number)
        accepted_batches.append(accepted)

    return (
        torch.cat(solution_batches, dim=0),
        torch.cat(uncertainty_batches, dim=0),
        torch.cat(residual_batches, dim=0),
        torch.cat(rank_batches, dim=0),
        torch.cat(condition_batches, dim=0),
        torch.cat(accepted_batches, dim=0),
    )


class _EulerAutogradContext:
    """Static typing view of the attributes supplied by autograd contexts."""

    dx_m: float
    dy_m: float
    structural_index: float
    euler_config: EulerConfig
    x_origin_m: float
    y_origin_m: float
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: torch.Tensor) -> None: ...

    def set_materialize_grads(self, value: bool) -> None: ...


class _BoundedEulerFunction(torch.autograd.Function):  # type: ignore[misc, unused-ignore]
    """Recompute streamed QR/SVD batches during VJP to bound saved state."""

    @staticmethod
    def forward(
        data: torch.Tensor,
        dx_m: float,
        dy_m: float,
        structural_index: float,
        config: EulerConfig,
        x_origin_m: float,
        y_origin_m: float,
    ) -> _EulerTensorResults:
        return _stream_euler_tensors(
            data,
            dx_m=dx_m,
            dy_m=dy_m,
            structural_index=structural_index,
            config=config,
            x_origin_m=x_origin_m,
            y_origin_m=y_origin_m,
        )

    @staticmethod
    def setup_context(
        ctx: object,
        inputs: tuple[
            torch.Tensor,
            float,
            float,
            float,
            EulerConfig,
            float,
            float,
        ],
        output: _EulerTensorResults,
    ) -> None:
        context = cast(_EulerAutogradContext, ctx)
        data, dx_m, dy_m, structural_index, config, x_origin_m, y_origin_m = inputs
        _, _, _, rank, _, accepted = output
        context.save_for_backward(data)
        context.mark_non_differentiable(rank, accepted)
        context.set_materialize_grads(False)
        context.dx_m = dx_m
        context.dy_m = dy_m
        context.structural_index = structural_index
        context.euler_config = config
        context.x_origin_m = x_origin_m
        context.y_origin_m = y_origin_m

    @staticmethod
    def backward(
        ctx: object,
        grad_solutions: torch.Tensor | None,
        grad_uncertainty: torch.Tensor | None,
        grad_residual_norm: torch.Tensor | None,
        grad_rank: torch.Tensor | None,
        grad_condition_number: torch.Tensor | None,
        grad_accepted: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, None, None, None, None, None, None]:
        del grad_rank, grad_accepted
        context = cast(_EulerAutogradContext, ctx)
        (data,) = context.saved_tensors
        cotangents = (
            grad_solutions,
            grad_uncertainty,
            grad_residual_norm,
            grad_condition_number,
        )
        if all(cotangent is None for cotangent in cotangents):
            return None, None, None, None, None, None, None

        gradient: torch.Tensor | None = None
        higher_order = torch.is_grad_enabled()
        row_count = (data.shape[0] - context.euler_config.window) // context.euler_config.stride + 1
        column_count = (data.shape[1] - context.euler_config.window) // context.euler_config.stride + 1
        window_count = row_count * column_count
        with torch.enable_grad():
            for start in range(0, window_count, context.euler_config.batch_windows):
                stop = min(start + context.euler_config.batch_windows, window_count)
                windows, _ = _prepare_window_views(
                    data,
                    dx_m=context.dx_m,
                    dy_m=context.dy_m,
                    config=context.euler_config,
                    x_origin_m=context.x_origin_m,
                    y_origin_m=context.y_origin_m,
                )
                batch = _solve_window_batch(
                    windows,
                    start=start,
                    stop=stop,
                    structural_index=context.structural_index,
                    config=context.euler_config,
                )
                differentiable_batch = (batch[0], batch[1], batch[2], batch[4])
                batch_cotangents = tuple(
                    None if cotangent is None else cotangent[start:stop]
                    for cotangent in cotangents
                )
                active = tuple(
                    (value, cotangent)
                    for value, cotangent in zip(
                        differentiable_batch,
                        batch_cotangents,
                        strict=True,
                    )
                    if cotangent is not None
                )
                contribution = torch.autograd.grad(
                    tuple(value for value, _ in active),
                    data,
                    grad_outputs=tuple(cotangent for _, cotangent in active),
                    retain_graph=higher_order,
                    create_graph=higher_order,
                )[0]
                gradient = contribution if gradient is None else gradient + contribution
                del active, batch, batch_cotangents, contribution, differentiable_batch, windows
        return gradient, None, None, None, None, None, None


def euler_deconvolution(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    structural_index: float,
    config: EulerConfig = EulerConfig(),
    x_origin_m: float = 0.0,
    y_origin_m: float = 0.0,
) -> EulerResult:
    """Solve windowed Euler systems with the validated QR/SVD contract.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
        structural_index: source-type structural index.
        config: window/quality-gate settings (:class:`EulerConfig`).
        x_origin_m / y_origin_m: grid origin [m].
    """
    data = _validate_data(data)
    ny, nx = data.shape
    config = _validate_config(config, ny=ny, nx=nx)
    dx_m = _finite_real(dx_m, field="dx_m", positive=True)
    dy_m = _finite_real(dy_m, field="dy_m", positive=True)
    structural_index = _finite_real(
        structural_index,
        field="structural_index",
        nonnegative=True,
    )
    x_origin_m = _finite_real(x_origin_m, field="x_origin_m")
    y_origin_m = _finite_real(y_origin_m, field="y_origin_m")
    tensors = cast(
        _EulerTensorResults,
        _BoundedEulerFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
            data,
            dx_m,
            dy_m,
            structural_index,
            config,
            x_origin_m,
            y_origin_m,
        ),
    )
    solutions, uncertainty, residual_norm, rank, condition_number, accepted = tensors
    return EulerResult(
        solutions=solutions,
        uncertainty=uncertainty,
        residual_norm=residual_norm,
        rank=rank,
        condition_number=condition_number,
        accepted=accepted,
        solver=tuple("qr" if bool(item) else "svd" for item in accepted),
    )


__all__ = ["EulerConfig", "EulerResult", "STRUCTURAL_INDEX", "euler_deconvolution"]
