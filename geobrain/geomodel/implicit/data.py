"""
Data structures for differentiable implicit geological modeling.

Defines input data types for Universal Cokriging with gradient constraints
(Lajaunie et al. 1997 / GemPy algorithm).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import torch
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

from ..errors import GeomodelContractError


def _invalid(object_name: str, field: str, expected: object, actual: object) -> None:
    raise GeomodelContractError(
        "invalid implicit-model tensor or configuration contract",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


class StackRelation(Enum):
    """Relation between geological series for multi-series stacking."""
    ERODE = "erode"
    ONLAP = "onlap"
    FAULT = "fault"
    BASEMENT = "basement"


@dataclass(frozen=True, slots=True)
class SurfacePointData:
    """
    Surface contact points defining geological interfaces.

    Args:
        coords: Point coordinates, shape (N, D) where D=2 or 3.
        surface_id: Integer surface index for each point, shape (N,).
        nugget: Nugget effect for regularization.
    """
    coords: torch.Tensor
    surface_id: torch.Tensor
    nugget: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.coords, torch.Tensor) or self.coords.ndim != 2 or self.coords.shape[1] not in (2, 3):
            _invalid(type(self).__name__, "coords", "tensor with shape (N, 2|3)", getattr(self.coords, "shape", None))
        if not isinstance(self.surface_id, torch.Tensor) or self.surface_id.shape != self.coords.shape[:1] or self.surface_id.dtype != torch.int64:
            _invalid(type(self).__name__, "surface_id", "int64 tensor with shape (N,)", getattr(self.surface_id, "dtype", None))
        if self.coords.dtype not in (torch.float32, torch.float64):
            _invalid(type(self).__name__, "coords.dtype", "float32 or float64", self.coords.dtype)


@dataclass(frozen=True, slots=True)
class OrientationData:
    """
    Orientation measurements (gradient constraints).

    Args:
        coords: Measurement locations, shape (M, D).
        gradients: Gradient directions, shape (M, D). Internally normalized.
        nugget: Nugget effect for regularization.
    """
    coords: torch.Tensor
    gradients: torch.Tensor
    nugget: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.coords, torch.Tensor) or not isinstance(self.gradients, torch.Tensor):
            _invalid(type(self).__name__, "coords/gradients", "torch.Tensor", (type(self.coords).__name__, type(self.gradients).__name__))
        if self.coords.ndim != 2 or self.coords.shape != self.gradients.shape or self.coords.shape[1] not in (2, 3):
            _invalid(type(self).__name__, "coords/gradients.shape", "matching (N, 2|3)", (self.coords.shape, self.gradients.shape))
        if self.coords.dtype not in (torch.float32, torch.float64) or self.gradients.dtype != self.coords.dtype:
            _invalid(type(self).__name__, "coords/gradients.dtype", "matching float32 or float64", (self.coords.dtype, self.gradients.dtype))
        if self.coords.device != self.gradients.device:
            _invalid(type(self).__name__, "coords/gradients.device", "matching devices", (self.coords.device, self.gradients.device))


@dataclass(frozen=True, slots=True)
class SeriesDefinition:
    """
    Definition of a geological series (a group of related surfaces).

    Args:
        name: Human-readable name for the series.
        surface_points: Surface contact points.
        orientations: Orientation / gradient measurements.
        relation: How this series interacts with older series.
        surface_names: Optional names for each surface in this series.
    """
    name: str
    surface_points: SurfacePointData
    orientations: OrientationData
    relation: StackRelation = StackRelation.ERODE
    surface_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            _invalid(type(self).__name__, "name", "non-empty string", self.name)
        if self.surface_names is not None:
            object.__setattr__(self, "surface_names", tuple(self.surface_names))


@dataclass(frozen=True, slots=True)
class FaultDefinition:
    """
    Definition of a fault surface.

    Args:
        name: Fault name.
        surface_points: Contact points on the fault surface.
        orientations: Orientation measurements on the fault.
        affected_series_indices: Which series indices are offset by this fault.
            None means all series are affected.
        displacement: Displacement magnitude (can be learned via gradient descent).
    """
    name: str
    surface_points: SurfacePointData
    orientations: OrientationData
    affected_series_indices: Optional[Tuple[int, ...]] = None
    displacement: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            _invalid(type(self).__name__, "name", "non-empty string", self.name)
        if self.affected_series_indices is not None:
            object.__setattr__(self, "affected_series_indices", tuple(self.affected_series_indices))


@dataclass(frozen=True, slots=True)
class InterpolationInput:
    """
    Packed input for a single Cokriging interpolation.

    Created from a SeriesDefinition with gradients normalized.
    """
    sp_coords: torch.Tensor       # (N, D)
    sp_surface_id: torch.Tensor   # (N,) int
    ori_coords: torch.Tensor      # (M, D)
    ori_gradients: torch.Tensor   # (M, D) unit vectors
    sp_nugget: float
    ori_nugget: float
    ndim: int
    n_surfaces: int

    @staticmethod
    def from_series(series: SeriesDefinition) -> 'InterpolationInput':
        """
        Create InterpolationInput from a SeriesDefinition.

        Normalizes gradient vectors to unit length.
        """
        sp = series.surface_points
        ori = series.orientations

        # Normalize gradients
        g = ori.gradients.clone()
        norms = g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        g = g / norms

        ndim = sp.coords.shape[-1]
        n_surfaces = int(sp.surface_id.max().item()) + 1

        return InterpolationInput(
            sp_coords=sp.coords,
            sp_surface_id=sp.surface_id,
            ori_coords=ori.coords,
            ori_gradients=g,
            sp_nugget=sp.nugget,
            ori_nugget=ori.nugget,
            ndim=ndim,
            n_surfaces=n_surfaces,
        )


@dataclass(frozen=True, slots=True)
class ImplicitModelConfig:
    """
    Configuration for implicit geological model.

    Args:
        extent: Spatial extent as (x0, x1, y0, y1) for 2D
            or (x0, x1, y0, y1, z0, z1) for 3D.
        resolution: Grid resolution as (nx, ny) or (nx, ny, nz).
        kernel: Covariance kernel type: "cubic" or "gaussian".
        range: Kernel range parameter. Default sqrt(3), GemPy convention.
        c_o: Sill / variance. If None, auto-computed as range^2 / (14/3).
        drift_degree: Polynomial drift degree. 0=constant, 1=linear.
        device: Compute device. Default ``"cpu"`` for reproducibility across
            machines; pass ``"cuda"`` or ``"auto"`` to opt into GPU.
        dtype: Data type string. "float64" recommended for Cokriging stability.
        anisotropy: Optional ``(D, D)`` anisotropy transform (rotation +
            per-axis rescaling). ``None`` ⇒ isotropic. Lets the prior encode
            the directional correlation of real geology (layering / dip)
            instead of an isotropic blob. Build it from ranges + angle via
            :func:`anisotropy_matrix_2d`, or pass any ``(D, D)`` array/tensor.
    """
    extent: Tuple[float, ...] = (0.0, 1.0, 0.0, 1.0)
    resolution: Tuple[int, ...] = (50, 50)
    kernel: str = "cubic"
    range: float = 1.7320508075688772  # sqrt(3)
    c_o: Optional[float] = None
    drift_degree: int = 1
    device: str = "cpu"
    dtype: str = "float64"
    anisotropy: Optional[Any] = None
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        extent = tuple(float(item) for item in self.extent)
        resolution = tuple(int(item) for item in self.resolution)
        if len(resolution) not in (2, 3) or len(extent) != 2 * len(resolution):
            _invalid(type(self).__name__, "extent/resolution", "2-D or 3-D matching ranks", (extent, resolution))
        if any(item < 1 for item in resolution):
            _invalid(type(self).__name__, "resolution", "positive integers", resolution)
        if self.dtype not in ("float32", "float64"):
            _invalid(type(self).__name__, "dtype", "float32 or float64", self.dtype)
        if self.device not in ("cpu", "cuda", "mps"):
            _invalid(type(self).__name__, "device", "available cpu, cuda, or mps", self.device)
        if self.device == "cuda" and not torch.cuda.is_available():
            _invalid(type(self).__name__, "device", "available cuda", self.device)
        if self.device == "mps" and not torch.backends.mps.is_available():
            _invalid(type(self).__name__, "device", "available mps", self.device)
        if self.budget_bytes is not None and (isinstance(self.budget_bytes, bool) or self.budget_bytes < 1):
            _invalid(type(self).__name__, "budget_bytes", "positive integer or None", self.budget_bytes)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "resolution", resolution)

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (2 or 3)."""
        return len(self.resolution)

    @property
    def is_3d(self) -> bool:
        return self.ndim == 3

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    @property
    def computed_c_o(self) -> float:
        """Sill value, auto-computed if not set."""
        if self.c_o is not None:
            return self.c_o
        return self.range ** 2 / (14.0 / 3.0)

    def make_grid(self) -> torch.Tensor:
        """
        Create a regular evaluation grid from extent and resolution.

        Returns:
            Tensor of shape (n_points, ndim) with grid coordinates.
        """
        ndim = self.ndim
        slices = []
        for i in range(ndim):
            lo = self.extent[2 * i]
            hi = self.extent[2 * i + 1]
            slices.append(torch.linspace(
                lo, hi, self.resolution[i],
                dtype=self.torch_dtype, device=self.torch_device,
            ))

        grids = torch.meshgrid(*slices, indexing='ij')
        grid = torch.stack([g.reshape(-1) for g in grids], dim=-1)
        return grid
