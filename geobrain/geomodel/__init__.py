"""GeoBrain Geomodel: spatial estimation, simulation, and generative modelling.

Provides classical geostatistics (kriging, sequential simulation, variogram
modelling, transforms, cross-validation), implicit geological modelling
(UCK kernel regression), and neural generative models (VAE, GAN, Diffusion).
The geostats sub-package is written in pure numpy/scipy (no ``import torch`` in
its own source); ``implicit`` and ``generative`` are torch-native and are loaded
lazily (deferred until first use). Note: importing any sub-package still pulls
torch transitively via ``geobrain.core``; there is no torch-free import path.

Architecture:
    geomodel/
    ├── data/           # Spatial containers (GeoFrame, GeoPoints, GeoGrid)
    ├── geostats/       # Classical geostatistics (numpy island, no torch)
    │   ├── estimation/ #   SK / OK / UK / IK / Block / Collocated Cokriging
    │   ├── model/      #   Variogram kernels, calculators, covariance
    │   ├── simulation/ #   SGSIM, LUSIM, FFTMA, DSSIM, SISIM, SNESIM, ...
    │   ├── transform/  #   NormalScore, BoxCox, Pipeline, Detrend, Decluster
    │   └── validate/   #   Cross-validation (LOO, K-fold, MSDR)
    ├── implicit/       # Implicit modelling (UCK, faults, series) [PyTorch]
    └── generative/         # Neural generative models (VAE, GAN, Diffusion) [PyTorch]

Quick Start:
    >>> from geobrain.geomodel import SimpleKriging, CovarianceModel
    >>> cov = CovarianceModel.spherical(sill=1.0, range_=50.0)
    >>> sk = SimpleKriging(cov, mean=0.0)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# =========================================================================
# Data Containers
# =========================================================================
from .capabilities import (
    GeomodelCapabilityReport,
    GeomodelUnsupportedCombination,
    discover_geomodel_capabilities,
)
from .conditioning import ConditioningPolicy, ConditioningSet, normalize_conditioning
from .cache_keys import (
    CheckpointCacheKey,
    CovarianceCacheKey,
    DomainCacheKey,
    NeighbourhoodCacheKey,
    TrainingImageCacheKey,
)
from .frames import (
    Category,
    ColumnRole,
    GeoFrame,
    Geometry,
    GeoGrid,
    GeoPoints,
    GslibGridLayout,
    PropertyMetadata,
    gslib_grid_layout,
)
from .domain_contract import DomainContract, domain_contract
from .errors import (
    GeomodelCapabilityError,
    GeomodelContractError,
    GeomodelNumericsError,
    GeomodelResourceError,
)
from .resources import (
    GeomodelResourceEstimate,
    GeomodelResourceRequest,
    enforce_budget,
)
from .neighbourhood import (
    DynamicKDTreeNeighbourhood,
    ExhaustiveNeighbourhood,
    NeighbourhoodBackend,
    NeighbourhoodSelection,
    NeighbourhoodSpec,
    RegularGridNeighbourhood,
    StaticKDTreeNeighbourhood,
)

# =========================================================================
# Geostatistics: Estimation
# =========================================================================
from .geostats.estimation.kriging import (
    KrigingSolvePolicy,
    OrdinaryKriging,
    SimpleKriging,
    UniversalKriging,
)

# =========================================================================
# Geostatistics: Variogram & Covariance Modelling
# =========================================================================
from .geostats.models.covariance import CovarianceModel
from .geostats.models.variogram_calculator import VariogramCalculator
from .geostats.models.variogram_kernel import VariogramKernel

# =========================================================================
# Geostatistics: Simulation
# =========================================================================
from .geostats.simulation.sgsim import SGSIM

# =========================================================================
# Geostatistics: Transforms & Validation
# =========================================================================
from .geostats.transform.base import Pipeline
from .geostats.transform.normal_score import NormalScore
from .geostats.validate.cross_validation import k_fold, leave_one_out

# Geostatistics: sibling to implicit/ and generative/. Eager-imported because it is
# the common-case entry point. (Its own source is torch-free, though torch is
# still pulled transitively via ``geobrain.core``.)
from . import geostats

# Implicit modelling (UCK) and generative AI siblings: torch-native. Kept off
# the eager-import path so the (heavy) neural / implicit machinery is only
# materialised on first use; the names below stay reachable through
# ``geomodel.<name>`` via the lazy ``__getattr__`` resolver further down.

__all__ = [
    "Category",
    "CheckpointCacheKey",
    "ColumnRole",
    "ConditioningPolicy",
    "ConditioningSet",
    "CovarianceCacheKey",
    "CovarianceModel",
    "DomainContract",
    "DomainCacheKey",
    "DynamicKDTreeNeighbourhood",
    "ExhaustiveNeighbourhood",
    "GeoFrame",
    "GeomodelCapabilityError",
    "GeomodelCapabilityReport",
    "GeomodelContractError",
    "GeomodelNumericsError",
    "GeomodelResourceError",
    "GeomodelResourceEstimate",
    "GeomodelResourceRequest",
    "GeomodelUnsupportedCombination",
    "Geometry",
    "NormalScore",
    "NeighbourhoodBackend",
    "NeighbourhoodCacheKey",
    "NeighbourhoodSelection",
    "NeighbourhoodSpec",
    "KrigingSolvePolicy",
    "OrdinaryKriging",
    "Pipeline",
    "GeoPoints",
    "GeoGrid",
    "GslibGridLayout",
    "PropertyMetadata",
    "RegularGridNeighbourhood",
    "SGSIM",
    "SimpleKriging",
    "StaticKDTreeNeighbourhood",
    "TrainingImageCacheKey",
    "UniversalKriging",
    "VariogramCalculator",
    "VariogramKernel",
    "discover_geomodel_capabilities",
    "domain_contract",
    "enforce_budget",
    "gslib_grid_layout",
    "k_fold",
    "leave_one_out",
    "normalize_conditioning",
    # sub-packages
    "geostats",
    # implicit modelling
    "implicit",
    "FaultDefinition",
    "ImplicitModel",
    "ImplicitModelConfig",
    "OrientationData",
    "SeriesDefinition",
    "StackRelation",
    "SurfacePointData",
    # generative AI
    "generative",
    "DiffusionSimulator",
    "GANSimulator",
    "Generator3D",
    "VAESimulator",
    # shared earth model (typed Field/Link DAG)
    "earthmodel",
    "EarthModel",
    "Field",
    "Link",
]

# =========================================================================
# Public façade exports (eagerly resolved and advertised)
# =========================================================================
_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    # Two reachable-but-lazy tiers: torch subpackages / their classes kept lazy
    # so a geostats-only import stays torch-free (these ARE advertised in
    # __all__), plus the extended geostats catalog (BlockKriging/FFTMA/SISIM/…)
    # which is intentionally reachable-but-unadvertised. ``(module, None)`` means
    # the entry IS the module itself (e.g. ``geomodel.generative``).
    # PyTorch subpackages: lazy so geostats-only users don't pay the torch import.
    "generative": (".generative", None),
    "earthmodel": (".earthmodel", None),
    "EarthModel": (".earthmodel", "EarthModel"),
    "Field": (".earthmodel", "Field"),
    "Link": (".earthmodel", "Link"),
    "implicit": (".implicit", None),
    "DiffusionSimulator": (".generative", "DiffusionSimulator"),
    "GenerativeConfig": (".generative", "GenerativeConfig"),
    "GANSimulator": (".generative", "GANSimulator"),
    "Generator3D": (".generative", "Generator3D"),
    "LDMC_PROVIDER": (".generative", "LDMC_PROVIDER"),
    "ModelCard": (".generative", "ModelCard"),
    "OptionalProvider": (".generative", "OptionalProvider"),
    "VAESimulator": (".generative", "VAESimulator"),
    "FaultDefinition": (".implicit", "FaultDefinition"),
    "ImplicitModel": (".implicit", "ImplicitModel"),
    "ImplicitModelConfig": (".implicit", "ImplicitModelConfig"),
    "OrientationData": (".implicit", "OrientationData"),
    "SeriesDefinition": (".implicit", "SeriesDefinition"),
    "StackRelation": (".implicit", "StackRelation"),
    "SurfacePointData": (".implicit", "SurfacePointData"),
    "load_verified_state_dict": (".generative", "load_verified_state_dict"),
    "require_provider": (".generative", "require_provider"),
    "BlockKriging": (".geostats.estimation", "BlockKriging"),
    "BoxCox": (".geostats.transform", "BoxCox"),
    "CoSGSIM": (".geostats.simulation", "CoSGSIM"),
    "CollocatedCokriging": (".geostats.estimation", "CollocatedCokriging"),
    "CrossValidationResult": (".geostats.validate", "CrossValidationResult"),
    "CrossVariogramCalculator": (".geostats.models", "CrossVariogramCalculator"),
    "CrossVariogramResult": (".geostats.models", "CrossVariogramResult"),
    "DSSIM": (".geostats.simulation", "DSSIM"),
    "DenseFactorPolicy": (".geostats.simulation", "DenseFactorPolicy"),
    "Decluster": (".geostats.transform", "Decluster"),
    "Detrend": (".geostats.transform", "Detrend"),
    "DirectSampling": (".geostats.simulation", "DirectSampling"),
    "ExperimentalVariogram": (".geostats.models", "ExperimentalVariogram"),
    "FFTMA": (".geostats.simulation", "FFTMA"),
    "FFTEmbeddingPolicy": (".geostats.simulation", "FFTEmbeddingPolicy"),
    "FILTERSIM": (".geostats.simulation", "FILTERSIM"),
    "HistogramSmooth": (".geostats.transform", "HistogramSmooth"),
    "ImageQuilting": (".geostats.simulation", "ImageQuilting"),
    "IndicatorEncode": (".geostats.transform", "IndicatorEncode"),
    "IndicatorKriging": (".geostats.estimation", "IndicatorKriging"),
    "KFold": (".geostats.validate", "KFold"),
    "LUSIM": (".geostats.simulation", "LUSIM"),
    "LeaveOneOut": (".geostats.validate", "LeaveOneOut"),
    "LinearModelOfCoregionalization": (".geostats.models", "LinearModelOfCoregionalization"),
    "MAPSCleaning": (".geostats.postprocessing", "MAPSCleaning"),
    "PlurigaussianSim": (".geostats.simulation", "PlurigaussianSim"),
    "PostIndicatorKriging": (".geostats.postprocessing", "PostIndicatorKriging"),
    "PostSimulation": (".geostats.postprocessing", "PostSimulation"),
    "SISIM": (".geostats.simulation", "SISIM"),
    "SequentialSolvePolicy": (".geostats.simulation", "SequentialSolvePolicy"),
    "SimulationEnsemble": (".geostats.simulation", "SimulationEnsemble"),
    "SimulationExecutionConfig": (".geostats.simulation", "SimulationExecutionConfig"),
    "SimulationRealization": (".geostats.simulation", "SimulationRealization"),
    "SNESIM": (".geostats.simulation", "SNESIM"),
    "TrainingImageIndex": (".geostats.simulation", "TrainingImageIndex"),
    "TrainingImageSelection": (".geostats.simulation", "TrainingImageSelection"),
    "TrainingImageSpec": (".geostats.simulation", "TrainingImageSpec"),
    "ScatterSmooth": (".geostats.transform", "ScatterSmooth"),
    "SoftIndicatorKriging": (".geostats.estimation", "SoftIndicatorKriging"),
    "Transform": (".geostats.transform", "Transform"),
    "VariogramFitAttempt": (".geostats.models", "VariogramFitAttempt"),
    "VariogramFitDiagnostics": (".geostats.models", "VariogramFitDiagnostics"),
    "VariogramFitResult": (".geostats.models", "VariogramFitResult"),
    "YeoJohnson": (".geostats.transform", "YeoJohnson"),
    "anisotropic_distance": (".geostats.models", "anisotropic_distance"),
    "anisotropic_squared_distance": (".geostats.models", "anisotropic_squared_distance"),
    "covariance_matrix": (".geostats.estimation", "covariance_matrix"),
    "covariance_vector": (".geostats.estimation", "covariance_vector"),
    "cressie_hawkins": (".geostats.models", "cressie_hawkins"),
    "cross_validation_report": (".geostats.validate", "cross_validation_report"),
    "dowd": (".geostats.models", "dowd"),
    "drift_basis": (".geostats.estimation", "drift_basis"),
    "genton": (".geostats.models", "genton"),
    "matheron": (".geostats.models", "matheron"),
    "msdr": (".geostats.validate", "msdr"),
    "normalize_role": (".frames", "normalize_role"),
    "setup_rotation_matrix": (".geostats.models", "setup_rotation_matrix"),
    "simulate_realization": (".geostats.simulation", "simulate_realization"),
}


__all__.extend(name for name in sorted(_PUBLIC_EXPORTS) if name not in __all__)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
