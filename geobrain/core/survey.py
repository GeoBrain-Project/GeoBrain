"""
Survey type-contract Protocols.

Surveys describe the source / receiver / sampling geometry a physics operator
consumes. Each physics family owns its own survey shape,
:class:`~geobrain.physics.wave.Seismic2DSurvey`,
:class:`~geobrain.physics.potential.PotentialSurvey2D`,
:class:`~geobrain.physics.em.MT1DSurvey`, ..., because the natural
field layouts are genuinely domain-specific (a magnetotellurics survey has
frequencies + layer thicknesses; a seismic survey has source / receiver
coordinates; a DC survey has electrode triples). Unifying them under a single
ABC would either lose type safety or balloon into a ``**kwargs`` bag.

The compromise is a small set of *capability* Protocols. Each declares **one**
common slice, frequency samples, time samples, position samples, so generic
code can demand only what it needs. Surveys with matching attributes satisfy the
Protocol via structural typing (no inheritance required).

Metre → cell convention (repo-wide)
-----------------------------------
Surveys carry source / receiver positions as **physical coordinates in metres**.
There is exactly **one** rule for snapping a metre coordinate to a structured-grid
cell index, shared by every physics family (EM finite-volume, wave time-domain,
wave frequency-domain):

    The physical position of cell ``i`` along an axis is its **centre**,
    ``(i + 0.5)·spacing``. The metre → cell map is therefore
    ``floor(coord / spacing)`` (clamped to ``[0, n - 1]``), which inverts the
    cell-centre map exactly: ``floor((i + 0.5)·spacing / spacing) == i``.

Use ``floor``, not ``round``, ``round`` implies a node-centred grid (cell ``i``
at ``i·spacing``) and would land the same coordinate in a different cell than the
finite-volume families. Any ``from_grid_indices`` helper must place index ``i`` at
the cell centre ``(i + 0.5)·spacing`` so it round-trips through ``floor``.

Example::

    # A generic frequency-sweep utility, typed without depending on any one
    # EM family:
    from geobrain.core import FrequencyDomainSurveyLike

    def list_frequencies(survey: FrequencyDomainSurveyLike) -> list[float]:
        return list(survey.frequencies)

These survey shapes satisfy :class:`FrequencyDomainSurveyLike`:

- :class:`~geobrain.physics.em.surveys.FrequencyDomainSurvey`
  and its subclasses (CSEM, FDEM, HEM, MT 1D/2D/3D variants).
- :class:`~geobrain.physics.wave.Helmholtz2DSurvey`.

These survey shapes satisfy :class:`TimeDomainSurveyLike`:

- :class:`~geobrain.physics.em.surveys.TimeDomainSurvey`
  and its subclasses (TEM 1D/3D, VTEM, ...).

This module is **not** an attempt to retro-unify the existing survey namespaces.
Concrete naming (``station_x`` vs ``rcv_x`` vs ``source_x/sink_x``) is left
alone; the Protocol-as-contract approach lets future generic helpers gain type
safety incrementally without a global renaming campaign.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

import torch

from .errors import GeoBrainError


@runtime_checkable
class FrequencyDomainSurveyLike(Protocol):
    """A survey carrying explicit frequency samples (Hz).

    Any object with a ``frequencies`` attribute that is a sequence of
    floats structurally satisfies this Protocol; inheritance is not
    required.
    """

    frequencies: Sequence[float]


@runtime_checkable
class TimeDomainSurveyLike(Protocol):
    """A survey carrying explicit time samples (seconds).

    Any object with a ``times`` attribute that is a sequence of floats
    structurally satisfies this Protocol.
    """

    times: Sequence[float]


def list_frequencies(survey: FrequencyDomainSurveyLike) -> list[float]:
    """Frequency samples (Hz) of any frequency-domain survey.

    The generic accessor this module's Protocols exist to enable, typed
    against :class:`FrequencyDomainSurveyLike`, so it accepts every survey that
    carries a ``frequencies`` attribute (CSEM / FDEM / HEM / MT 1D-3D,
    Helmholtz2D, ...) by structural typing, without depending on any one
    EM / wave family.
    """
    return [float(f) for f in survey.frequencies]


def list_times(survey: TimeDomainSurveyLike) -> list[float]:
    """Time-gate samples (seconds) of any time-domain survey.

    The :class:`TimeDomainSurveyLike` counterpart of :func:`list_frequencies`
    (TEM 1D/3D, VTEM, ...).
    """
    return [float(t) for t in survey.times]


def coords_to_cell_indices(
    coords: "torch.Tensor | float", spacing: float, n: int, *, origin: float = 0.0
) -> "torch.Tensor | int":
    """Snap physical coordinate(s) in metres along one axis to cell indices.

    The single repo-wide metre→cell rule (see the module docstring): cell ``i``
    is centred at ``origin + (i + 0.5)·spacing``, so the inverse map is
    ``floor((coord - origin) / spacing)`` clamped to ``[0, n - 1]``. Use
    ``floor`` (not ``round``), a ``from_grid_indices`` helper places index
    ``i`` at the cell centre so it round-trips through ``floor``.

    ``origin`` is the axis coordinate of the mesh's near (index-0) corner,
    the per-axis entry of ``TensorMesh.origin``. It defaults to ``0.0``, which
    reproduces the pre-origin behaviour byte-for-byte; a structured-grid survey
    consumer on an offset mesh must pass ``mesh.origin[axis]`` so a world-metre
    coordinate lands in the correct cell (a non-zero origin was previously
    ignored, silently mis-binding sources/receivers).

    Accepts a scalar (returns ``int``) or a 1-D tensor (returns a ``long`` tensor
    of the same length). Every structured-grid survey's ``_forward`` should snap
    through this one function so the convention has exactly one implementation.

    Non-finite coordinates (``NaN`` / ``±inf``) are a malformed survey and raise
    :class:`GeoBrainError` on **both** the scalar and tensor paths, rather than
    the tensor path silently flooring-then-clamping ``NaN → 0`` / ``+inf → n-1``
    while the scalar path errors. (The wave survey family validates finiteness at
    construction; this guard extends the same contract to every other consumer.)
    """
    origin = float(origin)
    if isinstance(coords, torch.Tensor):
        if not bool(torch.isfinite(coords).all()):
            raise GeoBrainError(
                "coords_to_cell_indices: coordinates must be finite",
                object_name="coords_to_cell_indices", field="coords",
                expected="finite (no NaN/inf)", actual="non-finite coordinate(s)",
            )
        idx = torch.floor(
            (coords.to(torch.float64) - origin) / float(spacing)
        ).long()
        return idx.clamp_(0, int(n) - 1)
    if not math.isfinite(float(coords)):
        raise GeoBrainError(
            "coords_to_cell_indices: coordinate must be finite",
            object_name="coords_to_cell_indices", field="coords",
            expected="finite (no NaN/inf)", actual=coords,
        )
    i = int(math.floor((float(coords) - origin) / float(spacing)))
    return max(0, min(int(n) - 1, i))
