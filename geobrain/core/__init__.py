"""GeoBrain Core: physics-agnostic primitives for geophysical inversion.

This package contains the abstract base classes and state containers shared by
every physics family. It defines the ``ForwardOperator`` contract and
supporting infrastructure for adjoint solves, sparse linear algebra, and
differentiable composition. Core must not import from ``geobrain.physics.*``.

Note:
    Problem definition (``InverseProblem``, ``Likelihood``, ``Prior``) lives in
    the sibling top-level package :mod:`geobrain.inverse`, not here; core must
    not import from ``geobrain.physics.*``, and the inverse-problem layer sits
    above core, binding it to the solvers in :mod:`geobrain.optim` /
    :mod:`geobrain.bayes`.

Architecture::

    core/
    ├── Operator Pattern
    │   ├── operator.py         # Operator, PropertyTransform, ForwardOperator
    │   ├── containers.py       # ModelState, ForwardOutput
    │   ├── context.py          # ForwardContext + typed sub-contexts
    │   ├── composition.py      # OperatorChain, OperatorBundle
    │   ├── differentiability.py  # DifferentiabilityLevel, DifferentiabilitySpec
    │   └── channels.py         # CHANNEL_REGISTRY (output-channel vocabulary)
    └── Infrastructure
        ├── adjoint/            # (sparse) linear_solve_with_adjoint, checkpoint, vjp, gradient_check
        ├── linalg/             # sparse solver seams, preconditioners
        ├── runtime.py          # device/dtype/AMP/compile policy (imports nothing from geobrain)
        ├── registry.py         # generic name→type Registry
        ├── errors.py           # GeoBrainError hierarchy
        ├── survey.py           # survey Protocols + coords_to_cell_indices
        ├── transforms.py       # invertible bijectors (bayes/earthmodel share)
        │                       #   + the per-channel property-prep pipeline (ex-io)
        ├── constants.py        # cross-family physical constants (MU_0, STANDARD_GRAVITY)
        ├── validation.py       # platform-wide argument validators / immutability helpers
        └── _require.py         # capability-based mesh requirement checks (duck-typed)
    (meshes themselves live in the top-level ``geobrain.mesh`` package)

Quick Start (implement ``_forward``; the base ``forward`` runs the contract
checks, then delegates; never override ``forward`` directly or you bypass input
validation, the mesh-capability gate, and the output-key check)::

    >>> from geobrain.core import (
    ...     ForwardOperator, ForwardOutput, ModelState, ForwardContext,
    ...     DifferentiabilitySpec, DifferentiabilityLevel,
    ... )
    >>> class MyOp(ForwardOperator):
    ...     differentiability = DifferentiabilitySpec(
    ...         level=DifferentiabilityLevel.FULL_AUTOGRAD,
    ...         trainable_inputs=("m",), output_keys=("d",),
    ...     )
    ...     def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
    ...         (m,) = state.fetch("m")
    ...         return ForwardOutput(data={"d": m})

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

# =========================================================================
# Operator Pattern
# =========================================================================
from .operator import ForwardOperator, Operator, PropertyTransform
from .containers import ForwardOutput, ModelState
from .context import ForwardContext
# Sub-contexts: reachable off the root for explicit construction
# (``ForwardContext(time=TimeContext(...))``) but NOT advertised in __all__;
# ``ForwardContext.of()`` is the construction path. Canonical home is
# ``geobrain.core.context``.
from .context import FlowContext, SourceContext, TimeContext  # noqa: F401
from .composition import OperatorBundle, OperatorChain
from .differentiability import DifferentiabilityLevel, DifferentiabilitySpec
from .channels import ChannelSpec, CHANNEL_REGISTRY

# =========================================================================
# Infrastructure
# =========================================================================
# Meshes live in the top-level ``geobrain.mesh`` package: a first-class
# user-facing noun does not hide behind the "internal machinery" name. Import ``Mesh``/``TensorMesh``/``OctreeMesh``/
# ``UnstructuredMesh``/``MeshProjection``/``mesh_from_dict`` from
# ``geobrain.mesh`` (or ``Mesh``/``TensorMesh`` from the root facade).
from .adjoint import (
    checkpoint,
    gradient_check,
    linear_solve_with_adjoint,
    sparse_linear_solve_with_adjoint,
    vjp,
)
from .registry import Registry
from .errors import (
    DifferentiabilityError,
    ErrorCode,
    FieldShapeError,
    GeoBrainError,
    MissingFieldError,
    RegistryError,
)
from .survey import (
    FrequencyDomainSurveyLike,
    TimeDomainSurveyLike,
    list_frequencies,
    list_times,
)

__all__ = [
    # Operator pattern
    "ForwardOperator",
    "Operator",
    "PropertyTransform",
    "ForwardContext",
    "ModelState",
    "ForwardOutput",
    "OperatorBundle",
    "OperatorChain",
    "DifferentiabilityLevel",
    "DifferentiabilitySpec",
    # Channel registry (single source of truth for output channel strings)
    "ChannelSpec",
    "CHANNEL_REGISTRY",
    # Adjoint / AD
    "checkpoint",
    "gradient_check",
    "linear_solve_with_adjoint",
    "sparse_linear_solve_with_adjoint",
    "vjp",
    # Registry
    "Registry",
    # Error hierarchy
    "DifferentiabilityError",
    "ErrorCode",
    "FieldShapeError",
    "GeoBrainError",
    "MissingFieldError",
    "RegistryError",
    # Survey protocols
    "FrequencyDomainSurveyLike",
    "TimeDomainSurveyLike",
    "list_frequencies",
    "list_times",
]
