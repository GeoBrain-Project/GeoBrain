"""Optional, per-channel unit-consistency check for the InverseProblem contract.

Compares the units a forward operator declares on its
:class:`~geobrain.core.containers.ForwardOutput` (``metadata["units"]``, a
``{channel: unit_string}`` mapping) against user-declared observed units.
Validation is **opt-in and validate-when-present**: only channels that declare a
unit on *both* sides are compared; anything missing a unit is skipped. Units are
opaque strings compared for equality, no dimensional analysis.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

from ..core.errors import GeoBrainError


def check_units(
    predicted_units: Mapping[str, str],
    observed_units: Mapping[str, str],
    *,
    object_name: str = "InverseProblem",
) -> None:
    """Raise :class:`GeoBrainError` if a channel declared on both sides disagrees.

    Args:
        predicted_units: forward-output units, ``{channel: unit}`` (e.g. from
            ``ForwardOutput.metadata["units"]``).
        observed_units: user-declared observed units, ``{channel: unit}``.
        object_name: name reported in the raised error.

    Channels absent from either mapping are skipped (validate-when-present), so
    an empty mapping on either side is always a no-op.
    """
    for channel, pred_unit in predicted_units.items():
        obs_unit = observed_units.get(channel)
        if obs_unit is None:
            continue
        if pred_unit != obs_unit:
            raise GeoBrainError(
                f"unit mismatch on channel {channel!r}: forward output is in "
                f"{pred_unit!r} but observed data is declared in {obs_unit!r}",
                object_name=object_name,
                field=channel,
                expected=obs_unit,
                actual=pred_unit,
            )
