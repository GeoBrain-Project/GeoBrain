"""Supported explicit comparison, channel, and unit adapters for the EM family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .channels import (
    EM_CHANNEL_SPECS,
    EM_SOURCE_SPECS,
    EMChannelSpec,
    build_em_channel_table,
    get_em_channel_spec,
)
from .conventions import (
    from_legacy_mt2d_native,
    phase_degrees_to_radians,
    phase_radians_to_degrees,
    to_minus_iwt_complex,
)
from .units import (
    chargeability_to_mv_per_v,
    mv_per_v_to_chargeability,
    nanotesla_to_tesla,
    tesla_to_nanotesla,
)


__all__ = [
    "EM_CHANNEL_SPECS",
    "EM_SOURCE_SPECS",
    "EMChannelSpec",
    "build_em_channel_table",
    "chargeability_to_mv_per_v",
    "from_legacy_mt2d_native",
    "get_em_channel_spec",
    "mv_per_v_to_chargeability",
    "nanotesla_to_tesla",
    "phase_degrees_to_radians",
    "phase_radians_to_degrees",
    "tesla_to_nanotesla",
    "to_minus_iwt_complex",
]
