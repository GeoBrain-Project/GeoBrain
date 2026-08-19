"""GeoBrain: differentiable multi-physics geophysical inversion platform.

GeoBrain provides a unified framework for forward modelling, deterministic
inversion, and Bayesian inference across seismic, electromagnetic, gravity,
and rock-physics domains. All physics operators share a common
``InverseProblem`` contract, enabling mix-and-match workflows.

The user-visible inversion API has three tiers, all consuming the same
:class:`InverseProblem`:

- **Tier 1 (canonical):**
  ``problem.create_inverter(...).run(...)`` for deterministic inversion;
  ``problem.as_posterior().sample("hmc", ...)`` for Bayesian sampling.
- **Tier 2 (explicit class):** :class:`geobrain.optim.Inverter`,
  :class:`geobrain.bayes.HMC` / ``NUTS`` / ``LangevinDynamics`` / ``SVGD``.
- **Tier 3 (bare optimizer):** :class:`geobrain.optim.Adam`,
  :class:`geobrain.optim.LBFGS`: no regularizers, no bounds,
  no best-snapshot. For reference loops and tests.

Architecture: the packages are flat on disk but strictly layered; import
directions are enforced by the architecture layer-contract tests, and the
full map + rationale lives in the architecture documentation::

    geobrain/
    ├── core/       # L0+L1, physics-agnostic primitives (Operator, ModelState,
    │               #   ForwardContext, channels, errors), internal numerics
    │               #   (core.linalg, core.adjoint), and the dependency-free
    │               #   device/accel policy (core.runtime)
    ├── mesh/       # L1, user-facing discretizations: TensorMesh, OctreeMesh,
    │               #   UnstructuredMesh, MeshProjection; imports core only
    ├── physics/    # L2, forward-operator families, isolated from each other
    │   ├── wave/       # seismic: acoustic/elastic/aniso/visco, AVO
    │   ├── em/         # DC/IP, MT 1-D/2-D/3-D, CSEM/FDEM, TEM, SIP
    │   ├── potential/  # gravity, magnetics, Euler deconvolution
    │   ├── rock/       # rock physics: Gassmann, Gardner, Archie, DEM, ...
    │   └── flow/       # reservoir flow: single/two-phase, black-oil, wells
    ├── nn/         # L3, neural parameterizations (deep-image-prior, reparam seams);
    │               #   narrow leaf, imports core only
    ├── geomodel/   # L3, geological modelling: geostats, implicit structural
    │               #   and generative models, spatial containers, and the
    │               #   shared earth model it produces
    │               #   (geomodel.earthmodel: typed Field/Link DAG, a narrow
    │               #   leaf importing core only; physics may not import it)
    ├── inverse/    # L4, problem definition: InverseProblem / JointProblem,
    │               #   likelihoods + waveform misfits, priors
    ├── optim/      # L4, deterministic inversion: Inverter (+hook contract),
    │               #   Adam, L-BFGS
    ├── bayes/      # L4, Bayesian inference: Posterior, HMC/NUTS/LangevinDynamics/SVGD
    ├── decision/   # L4, value of information, closed-loop control
    ├── datasets/   # cross-cutting, built-in benchmark tables (Walker Lake,
    │               #   Jura, Meuse, mining_3d) as GeoFrames; no I/O, no
    │               #   downloads, deterministic
    ├── io/         # cross-cutting, file formats (SEG-Y/HDF5/LAS/VTK) plus
    │               #   in-memory tensor datasets/transforms; autograd stops
    │               #   only at the file-format boundary
    └── vis/        # cross-cutting, matplotlib/PyVista/Plotly plotting
                    # (every cross-cutting package depends on core + mesh only;
                    #  device/accel policy lives in core.runtime, which
                    #  imports nothing from geobrain at all)

Quick Start:
    >>> from geobrain import InverseProblem
    >>> from geobrain.inverse import GaussianLikelihood
    >>> from geobrain.optim import AdamConfig
    >>> from geobrain.physics.wave import Acoustic2D
    >>> problem = InverseProblem(forward, observed, GaussianLikelihood(std=0.05))
    >>> result = problem.create_inverter(
    ...     optimizer=AdamConfig(lr=1e-3)
    ... ).run(n_iters=200)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._version import __version__ as __version__

if TYPE_CHECKING:
    from .core import (
        DifferentiabilitySpec as DifferentiabilitySpec,
        ForwardContext as ForwardContext,
        ForwardOperator as ForwardOperator,
        ForwardOutput as ForwardOutput,
        GeoBrainError as GeoBrainError,
        ModelState as ModelState,
        Operator as Operator,
        OperatorBundle as OperatorBundle,
        OperatorChain as OperatorChain,
        PropertyTransform as PropertyTransform,
    )
    from .mesh import Mesh as Mesh, TensorMesh as TensorMesh
    from .inverse import InverseProblem as InverseProblem, JointProblem as JointProblem
    from .io.artifacts import ArtifactRef as ArtifactRef
    from .optim import InversionResult as InversionResult, Inverter as Inverter

__all__ = [
    "__version__",
    "ArtifactRef",
    "DifferentiabilitySpec",
    "ForwardContext",
    "ForwardOperator",
    "ForwardOutput",
    "GeoBrainError",
    "InverseProblem",
    "InversionResult",
    "Inverter",
    "JointProblem",
    "Mesh",
    "ModelState",
    "Operator",
    "OperatorBundle",
    "OperatorChain",
    "PropertyTransform",
    "TensorMesh",
]


# Stable root export -> (owning module, attribute). The package root is a
# convenience facade, not an internal dependency hub: importing ``geobrain``
# itself loads only version metadata, while each public object keeps its
# historical import path and resolves on first access.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactRef": (".io.artifacts", "ArtifactRef"),
    "DifferentiabilitySpec": (".core", "DifferentiabilitySpec"),
    "ForwardContext": (".core", "ForwardContext"),
    "ForwardOperator": (".core", "ForwardOperator"),
    "ForwardOutput": (".core", "ForwardOutput"),
    "GeoBrainError": (".core", "GeoBrainError"),
    "InverseProblem": (".inverse", "InverseProblem"),
    "InversionResult": (".optim", "InversionResult"),
    "Inverter": (".optim", "Inverter"),
    "JointProblem": (".inverse", "JointProblem"),
    "Mesh": (".mesh", "Mesh"),
    "ModelState": (".core", "ModelState"),
    "Operator": (".core", "Operator"),
    "OperatorBundle": (".core", "OperatorBundle"),
    "OperatorChain": (".core", "OperatorChain"),
    "PropertyTransform": (".core", "PropertyTransform"),
    "TensorMesh": (".mesh", "TensorMesh"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
