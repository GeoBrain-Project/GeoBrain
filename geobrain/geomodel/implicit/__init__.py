"""
Differentiable implicit geological modeling via Universal Cokriging.

Implements the Lajaunie et al. (1997) / GemPy algorithm in pure PyTorch,
supporting 2D + 3D, multi-series stacking (ERODE / ONLAP), and simple
fault displacement, all end-to-end differentiable.

Quick Start:
    >>> from geobrain.geomodel.implicit import (
    ...     ImplicitModel, ImplicitModelConfig,
    ...     SeriesDefinition, SurfacePointData, OrientationData,
    ... )
    >>> import torch
    >>>
    >>> sp = SurfacePointData(
    ...     coords=torch.tensor([[0.3, 0.5], [0.7, 0.5]]),
    ...     surface_id=torch.tensor([0, 1]),
    ... )
    >>> ori = OrientationData(
    ...     coords=torch.tensor([[0.5, 0.5]]),
    ...     gradients=torch.tensor([[0.0, 1.0]]),
    ... )
    >>> series = SeriesDefinition("strata", sp, ori)
    >>> config = ImplicitModelConfig(extent=(0,1,0,1), resolution=(50,50))
    >>> model = ImplicitModel(config, series=[series])
    >>> result = model(soft=True)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .data import (
    StackRelation,
    SurfacePointData,
    OrientationData,
    SeriesDefinition,
    FaultDefinition,
    InterpolationInput,
    ImplicitModelConfig,
)
from .kernel import (
    CovarianceKernel,
    CubicKernel,
    GaussianKernel,
    anisotropy_matrix_2d,
)
from .interpolator import CokrigingInterpolator
from .model import ImplicitModel
from .simulator import ImplicitSimulator

__all__ = [
    # Data structures
    "StackRelation",
    "SurfacePointData",
    "OrientationData",
    "SeriesDefinition",
    "FaultDefinition",
    "InterpolationInput",
    "ImplicitModelConfig",
    # Kernels
    "CovarianceKernel",
    "CubicKernel",
    "GaussianKernel",
    "anisotropy_matrix_2d",
    # Interpolator
    "CokrigingInterpolator",
    # Model
    "ImplicitModel",
    # Simulator (factory integration)
    "ImplicitSimulator",
]
