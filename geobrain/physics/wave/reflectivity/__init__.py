"""AVO: Amplitude Variation with Offset (pre-stack seismic).

AVO sits inside the wave physics family because it describes seismic reflection
coefficients at an interface (a wave phenomenon) even though its INPUT is a
rock-physics contrast (vp/vs/rho). There is no sibling ``physics/avo/`` package.

Two layers:

- **Math nn.Module reflectivity classes** (in ``approximations.py`` and
  ``zoeppritz.py``), suffix-tagged ``*Reflectivity`` to disambiguate from the
  operator wrappers below. They return ``R(θ)`` per interface.
- **Operator classes** (in ``operators.py``), short-named :class:`AkiRichards` /
  :class:`Shuey` / :class:`Zoeppritz` / :class:`ConvolutionalAVO`,
  ``ForwardOperator`` wrappers consuming a :class:`ModelState` with
  ``vp / vs / rho`` and emitting a reflectivity (or convolved trace)
  :class:`ForwardOutput`.

User-facing canonical entry points::

    from geobrain.physics.wave import AkiRichards, Shuey, Zoeppritz, ConvolutionalAVO
    from geobrain.physics.wave.reflectivity import (
        AkiRichardsReflectivity, ShueyReflectivity, ZoeppritzReflectivity,
    )

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from typing import cast

from geobrain.core import GeoBrainError

from torch import Tensor

from .approximations import AkiRichardsReflectivity, ShueyReflectivity
from .base import (
    AVOModel,
    EPS,
    as_tensor,
    deg2rad,
    normal_incidence_rc,
    vectorize_angles,
)
from .operators import (
    AkiRichards,
    ConvolutionalAVO,
    Shuey,
    Zoeppritz,
)
from .registry import (
    create_avo,
    get_avo,
    list_avo,
    list_avo_aliases,
    register_avo,
)
from .zoeppritz import ZoeppritzReflectivity


def compute_reflectivity(
    vp1: object,
    vs1: object,
    rho1: object,
    vp2: object,
    vs2: object,
    rho2: object,
    theta: object,
    method: str = "shuey",
) -> Tensor:
    """
    Compute P-P reflection coefficients using the named method.

    Args:
        vp1, vs1, rho1: Upper-layer P-vel, S-vel, density.
        vp2, vs2, rho2: Lower-layer counterparts.
        theta: Incidence angles in degrees (scalar, list, or tensor).
        method: Canonical name (``"AkiRichardsReflectivity"``,
            ``"ShueyReflectivity"``, ``"ZoeppritzReflectivity"``) or
            historical alias (``"AkiRichards"``, ``"Shuey"``,
            ``"Zoeppritz"``, ``"aki"``, ``"shuey"``, ``"ar"``,
            ``"zoeppritz"``, ``"exact"``).

    Returns:
        Reflection coefficients with leading dimension ``n_angles``.
    """
    try:
        model = create_avo(method)
    except Exception as exc:
        raise GeoBrainError(
            f"Unknown AVO method: '{method}'. "
            f"Available: {list_avo()}. "
            f"Aliases: {sorted(list_avo_aliases())}"
        ) from exc
    return cast(Tensor, model(vp1, vs1, rho1, vp2, vs2, rho2, theta))


__all__ = [
    # Math (nn.Module reflectivity, suffix-tagged)
    "AkiRichardsReflectivity",
    "ShueyReflectivity",
    "ZoeppritzReflectivity",
    # Operators (short-named, ForwardOperator)
    "AkiRichards",
    "ConvolutionalAVO",
    "Shuey",
    "Zoeppritz",
    # Math primitives + registry
    "AVOModel",
    "EPS",
    "as_tensor",
    "compute_reflectivity",
    "create_avo",
    "deg2rad",
    "get_avo",
    "list_avo",
    "list_avo_aliases",
    "normal_incidence_rc",
    "register_avo",
    "vectorize_angles",
]
