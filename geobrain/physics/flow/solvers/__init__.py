"""Curated nonlinear, linear, and Jacobian solver facade for Flow.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .config import KrylovConfig, NewtonConfig
from .convergence import CNVMBCriterion, TOL_CNV, TOL_MB, cnv_mb_errors, reservoir_pore_volume
from .diagnostics import (
    FlowConvergenceDiagnostics,
    convergence_diagnostics,
    normalize_convergence_diagnostics,
)
from .jacobian import (
    JacobianSparsitySpec,
    compute_coloring,
    compute_sparse_jacobian,
    compute_sparsity_pattern,
    make_sparsity_spec,
)
from .linear_solvers import (
    BILU0Preconditioner,
    BiCGSTABSolver,
    CPRSolver,
    DirectSolver,
    GMRESSolver,
    ILU0Preconditioner,
    IdentityPreconditioner,
    JacobiPreconditioner,
    LinearSolveStats,
    MultigridSolver,
    SparseDirectSolver,
)
from .multigrid import GeometricMultigrid
from .newton import NewtonResult, NewtonSolver, solve_newton

__all__ = (
    "BILU0Preconditioner",
    "BiCGSTABSolver",
    "CNVMBCriterion",
    "CPRSolver",
    "DirectSolver",
    "FlowConvergenceDiagnostics",
    "GMRESSolver",
    "GeometricMultigrid",
    "ILU0Preconditioner",
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "JacobianSparsitySpec",
    "KrylovConfig",
    "LinearSolveStats",
    "MultigridSolver",
    "NewtonConfig",
    "NewtonResult",
    "NewtonSolver",
    "SparseDirectSolver",
    "TOL_CNV",
    "TOL_MB",
    "cnv_mb_errors",
    "compute_coloring",
    "compute_sparse_jacobian",
    "compute_sparsity_pattern",
    "convergence_diagnostics",
    "make_sparsity_spec",
    "normalize_convergence_diagnostics",
    "reservoir_pore_volume",
    "solve_newton",
)
