"""GeoBrain Potential Fields: gravity and magnetic forward modelling and processing.

Provides 3D voxel-based and analytic prism-kernel forward operators for
gravity and magnetics, frequency-domain processing (RTP, derivatives, analytic
signal, tilt, upward continuation, regional/residual separation), gravity
reductions (free-air, Bouguer, terrain), and Euler deconvolution for
depth-to-source estimation.

Architecture:
    potential/
    ├── corrections.py  # Gravity reductions (free-air, Bouguer, terrain)
    ├── euler.py        # Euler deconvolution depth-to-source estimation
    ├── gravity.py      # Gravity forward operators (2D, 3D voxel)
    ├── helpers.py      # Constants (G_SI, MU_0), direction cosines, results
    ├── magnetics.py    # Magnetic forward operators (3D voxel, EarthField)
    ├── prism.py        # Analytic prism kernels (gravity; magnetics uses _engine)
    ├── processing.py   # RTP, derivatives, analytic signal, tilt, upward cont.
    ├── surveys.py      # PotentialSurvey2D / PotentialSurvey3D station geometry
    ├── units.py        # Declared-unit conversions (family template)
    ├── capabilities.py # Capability report + input schema (family template)
    ├── resources.py    # Checked resource estimation (family template)
    ├── config.py       # Immutable operator configuration records
    ├── errors.py       # PotentialError hierarchy (family template)
    ├── _tiled.py       # Private bounded-tile evaluation behind the voxel ops
    └── _engine/        # Private plan/cache/execute pipeline shared by the
                        #   gravity and magnetics voxel operators

Quick Start:
    >>> import torch
    >>> from geobrain.physics.potential import Gravity3D, PotentialSurvey3D
    >>> survey = PotentialSurvey3D(
    ...     torch.tensor([[0., 0., 10.], [100., 0., 10.]], dtype=torch.float64)
    ... )
    >>> op = Gravity3D(survey)   # mesh is supplied at call time via ForwardContext

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

# =========================================================================
# Strict SI Contracts
# =========================================================================
from ._engine import PrismKernelPlan
from .capabilities import PotentialCapabilityReport
from .resources import PotentialResourceEstimate
from .config import EarthField, PotentialExecutionConfig
from .errors import (
    PotentialCapabilityError,
    PotentialContractError,
    PotentialNumericsError,
    PotentialResourceError,
)
from .surveys import PotentialSurvey2D, PotentialSurvey3D
from .units import gravity_gradient_to_eotvos, gravity_to_mgal, magnetic_to_nt

# =========================================================================
# Prism Kernels (sub-module)
# =========================================================================
from . import prism

# =========================================================================
# Gravity Corrections & Reductions
# =========================================================================
from .corrections import (
    FREE_AIR_GRADIENT_M_PER_S2_PER_M,
    bouguer_slab,
    bouguer_spherical_cap,
    free_air_correction,
    terrain_correction_prism,
)

# =========================================================================
# Euler Deconvolution
# =========================================================================
from .euler import EulerConfig, EulerResult, STRUCTURAL_INDEX, euler_deconvolution

# =========================================================================
# Gravity Forward Operators
# =========================================================================
from .gravity import Gravity2D, Gravity3D

# =========================================================================
# Constants & Helpers
# =========================================================================
from .helpers import (
    EPS,
    G_SI,
    M_TO_EOTVOS,
    M_TO_MGAL,
    MU_0,
    T_TO_NT,
    field_direction_cosines,
)

# =========================================================================
# Magnetic Forward Operators
# =========================================================================
from .magnetics import Magnetic3D, MagneticComponent, MagneticVector3D

# =========================================================================
# Prism Forward Functions
# =========================================================================
from .prism import (
    gravity_prism_gradient_tensor,
    gravity_prism_gz,
)

# =========================================================================
# Frequency-Domain Processing
# =========================================================================
from .processing import (
    analytic_signal_amplitude,
    field_gradient_tensor,
    horizontal_gradient,
    reduce_to_pole,
    regional_residual_lowpass,
    tilt_derivative,
    upward_continue,
    vertical_derivative,
)

__all__ = [
    "EPS",
    "EarthField",
    "EulerConfig",
    "FREE_AIR_GRADIENT_M_PER_S2_PER_M",
    "EulerResult",
    "G_SI",
    "Gravity2D",
    "Gravity3D",
    "M_TO_EOTVOS",
    "M_TO_MGAL",
    "MU_0",
    "Magnetic3D",
    "MagneticComponent",
    "MagneticVector3D",
    "PotentialCapabilityError",
    "PotentialCapabilityReport",
    "PotentialContractError",
    "PotentialExecutionConfig",
    "PotentialNumericsError",
    "PotentialResourceError",
    "PotentialResourceEstimate",
    "PotentialSurvey2D",
    "PotentialSurvey3D",
    "PrismKernelPlan",
    "STRUCTURAL_INDEX",
    "T_TO_NT",
    "analytic_signal_amplitude",
    "bouguer_slab",
    "bouguer_spherical_cap",
    "euler_deconvolution",
    "field_direction_cosines",
    "field_gradient_tensor",
    "free_air_correction",
    "gravity_prism_gradient_tensor",
    "gravity_prism_gz",
    "gravity_gradient_to_eotvos",
    "gravity_to_mgal",
    "horizontal_gradient",
    "magnetic_to_nt",
    "prism",
    "reduce_to_pole",
    "regional_residual_lowpass",
    "terrain_correction_prism",
    "tilt_derivative",
    "upward_continue",
    "vertical_derivative",
]
