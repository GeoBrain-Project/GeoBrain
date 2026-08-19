"""Family-internal inductive EM orchestration surface.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .cache import EMExecutionCache
from .contracts import (
    AssemblyCacheKey,
    EMExecutionDiagnostics,
    FactorCacheKey,
    RecordingDiagnostics,
    exact_float_token,
)
from .dispatch import (
    build_assembly_cache_key,
    material_fingerprint,
    mesh_fingerprint,
    resolve_inductive_mesh_path,
)
from .frequency import solve_compatible_rhs
from .recording import (
    RecordingPlan,
    RecordingPolicy,
    execute_recorded_recurrence,
    prepare_recording,
)
from .time import (
    SparseSolverSettings,
    recording_execution_metadata,
    resolve_engine_solver,
    solver_execution_metadata,
    solver_factor_key,
    solver_settings,
)

__all__ = [
    "AssemblyCacheKey",
    "EMExecutionCache",
    "EMExecutionDiagnostics",
    "FactorCacheKey",
    "RecordingDiagnostics",
    "RecordingPlan",
    "RecordingPolicy",
    "SparseSolverSettings",
    "build_assembly_cache_key",
    "exact_float_token",
    "material_fingerprint",
    "mesh_fingerprint",
    "resolve_engine_solver",
    "recording_execution_metadata",
    "resolve_inductive_mesh_path",
    "solver_execution_metadata",
    "solver_factor_key",
    "solver_settings",
    "solve_compatible_rhs",
    "execute_recorded_recurrence",
    "prepare_recording",
]
