"""GeoBrain Wave Physics: finite-difference and convolutional seismic forward operators.

This package provides acoustic and elastic wave operators for 2-D and 3-D
domains backed by a single configurable-order FDTD engine, a frequency-domain
Helmholtz solver, plus convolutional-model synthetics and AVO reflectivity
operators. All operators conform to the GeoBrain ``ForwardOperator`` contract and
integrate with ``InverseProblem`` for deterministic inversion or Bayesian sampling.

Architecture: public facades over one private time-domain engine, plus the
independent frequency-domain and convolutional families::

    wave/
    ├── acoustic.py       # Immutable packed Acoustic2D/Acoustic3D facades
    ├── elastic.py        # Immutable packed Elastic2D/Elastic3D facades
    ├── anisotropic.py    # Production tilted-TI elastic facades
    ├── attenuation.py    # Production constant-Q facades
    ├── experimental/     # Reserved for facades without production evidence
    ├── _engine/          # Private equations, backends, memory, boundaries, and results
    ├── frequency_domain/ # Frequency-domain (time-harmonic) solver, Helmholtz2D
    │                     #   (sparse solve + implicit adjoint; NOT a propagator)
    ├── acquisition.py    # Packed physical-coordinate surveys
    ├── wavelets/         # Source time functions (Ricker, Gaussian, Ormsby, Klauder)
    ├── reflectivity/     # AVO reflectivity operators
    ├── convolutional.py  # 1-D convolutional synthetics
    ├── _facade.py        # Private shared facade base behind the four
    │                     #   time-domain operator modules above
    ├── capabilities.py   # Capability report (family template)
    ├── resources.py      # Resource-estimation door (family template;
    │                     #   re-exports the engine's public pieces)
    ├── config.py         # Immutable solver/backend configuration records
    └── errors.py         # WaveError hierarchy (family template)

FWI is a composition, not a wave-physics primitive: this family's
``Acoustic2D`` + ``geobrain.physics.rock.Gardner`` chained into
``geobrain.inverse.InverseProblem`` and driven by ``geobrain.optim.Inverter``.

Acquisition quick start (facade execution also requires a live model and mesh):
    >>> from geobrain.physics.wave import Seismic2DSurvey, shared_wavelet, ricker
    >>> # Packed positions are physical metres in public (x, z) order.
    >>> survey = Seismic2DSurvey.from_positions(
    ...     source_positions=[[300.0, 20.0]], source_shot_index=[0],
    ...     receiver_positions=[[float(x), 20.0] for x in range(50, 450, 20)],
    ...     receiver_shot_index=[0] * 20,
    ...     nt=500, dt=0.001,
    ... )
    >>> source_wavelets = shared_wavelet(ricker(500, 0.001, 25.0), n_source=survey.n_source)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

# =========================================================================
# Public pure-data contracts. These carry no acquisition or engine wiring.
# =========================================================================
from .capabilities import WaveCapabilityReport, WaveUnsupportedCombination
from .config import (
    WaveBackendConfig,
    WaveBoundaryConfig,
    WaveDiscretizationConfig,
    WaveMemoryConfig,
    WaveOutputConfig,
    WaveSimulationConfig,
)
from .errors import (
    WaveCapabilityError,
    WaveContractError,
    WaveNumericsError,
    WaveResourceError,
)

# =========================================================================
# Convolutional Synthetics
# =========================================================================
from .convolutional import Convolutional1D, ricker

# =========================================================================
# Time-domain immutable packed facades
# =========================================================================
from .acoustic import Acoustic2D, Acoustic3D
from .anisotropic import AnisotropicElastic2D, AnisotropicElastic3D
from .attenuation import ViscoAcoustic2D, ViscoElastic2D
from .elastic import Elastic2D, Elastic3D

# =========================================================================
# Frequency-domain (time-harmonic) solvers: Helmholtz (sparse solve +
# implicit adjoint). A boundary-value solver, not a time-domain propagator.
# =========================================================================
from .frequency_domain import (
    Helmholtz2D,
    Helmholtz2DReceiver,
    Helmholtz2DSource,
    Helmholtz2DSurvey,
)

# =========================================================================
# AVO Reflectivity Operators
# =========================================================================
from .reflectivity.operators import (
    AkiRichards,
    ConvolutionalAVO,
    Shuey,
    Zoeppritz,
)

# =========================================================================
# Survey Geometry
# =========================================================================
from .acquisition import Seismic2DSurvey, Seismic3DSurvey, shared_wavelet

# =========================================================================
# Wavelets
# =========================================================================
from .wavelets import (
    GaussianWavelet,
    KlauderWavelet,
    OrmsbyWavelet,
    RickerWavelet,
    WaveletGenerator,
    create_wavelet,
)

_PRODUCTION_CAPABILITY_FACTORIES = (
    Acoustic2D.capabilities,
    Acoustic3D.capabilities,
    AkiRichards.capabilities,
    Convolutional1D.capabilities,
    ConvolutionalAVO.capabilities,
    Elastic2D.capabilities,
    Elastic3D.capabilities,
    Helmholtz2D.capabilities,
    Shuey.capabilities,
    Zoeppritz.capabilities,
    AnisotropicElastic2D.capabilities,
    AnisotropicElastic3D.capabilities,
    ViscoAcoustic2D.capabilities,
    ViscoElastic2D.capabilities,
)
_EXPERIMENTAL_CAPABILITY_FACTORIES: tuple[()] = ()


def discover_wave_capabilities(
    *, include_experimental: bool = False
) -> tuple[dict[str, object], ...]:
    """Return fresh JSON-ready reports in stable façade-name order.

    The default registry contains only accepted production facades. Passing
    ``include_experimental=True`` appends the separately governed experimental
    registry without changing the production prefix. All currently implemented
    facades are promoted, so that registry is empty in version 0.2.0.
    """
    factories = _PRODUCTION_CAPABILITY_FACTORIES
    if include_experimental:
        factories = (
            *_PRODUCTION_CAPABILITY_FACTORIES,
            *_EXPERIMENTAL_CAPABILITY_FACTORIES,
        )
    reports = tuple(factory() for factory in factories)
    production_count = len(_PRODUCTION_CAPABILITY_FACTORIES)
    if any(report.maturity != "production" for report in reports[:production_count]):
        raise RuntimeError("production Wave capability registry contains an unaccepted facade")
    return tuple(report.to_dict() for report in reports)


__all__ = [
    "Acoustic2D", "Acoustic3D", "AkiRichards", "Convolutional1D",
    "ConvolutionalAVO", "Elastic2D", "Elastic3D", "GaussianWavelet",
    "Helmholtz2D", "Helmholtz2DReceiver", "Helmholtz2DSource",
    "Helmholtz2DSurvey", "KlauderWavelet", "OrmsbyWavelet",
    "RickerWavelet", "Seismic2DSurvey", "Seismic3DSurvey", "Shuey",
    "WaveBackendConfig", "WaveBoundaryConfig", "WaveCapabilityError",
    "WaveCapabilityReport", "WaveContractError", "WaveDiscretizationConfig",
    "WaveMemoryConfig", "WaveNumericsError", "WaveOutputConfig",
    "WaveResourceError", "WaveSimulationConfig", "WaveUnsupportedCombination",
    "WaveletGenerator", "Zoeppritz", "create_wavelet",
    "discover_wave_capabilities", "ricker", "shared_wavelet",
    "AnisotropicElastic2D", "AnisotropicElastic3D", "ViscoAcoustic2D",
    "ViscoElastic2D",
]
