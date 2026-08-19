"""
Yee receiver-anchor resolution shared by the 3-D inductive operators.

The strict projection builder lives in :mod:`geobrain.physics.em.receivers`.
This module applies its immutable multilinear plans to Yee edge/face vectors
and retains ``yee_component_anchor`` only as a strict legacy single-index view.
No receiver coordinate is clamped, snapped from outside, or extrapolated.

The Yee anchor table: each component lives on either the cell-centre or the
node along each axis:

====  ============  ============  ============
Comp  x-anchor      y-anchor      z-anchor
====  ============  ============  ============
Ex    cell-centre   node          node
Ey    node          cell-centre   node
Ez    node          node          cell-centre
Bx    node          cell-centre   cell-centre
By    cell-centre   node          cell-centre
Bz    cell-centre   cell-centre   node
====  ============  ============  ============

``HX/HY/HZ`` share the ``BX/BY/BZ`` face anchors (the caller post-divides by μ₀).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.mesh import TensorMesh
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.receivers import (
    YeeReceiverProjection,
    build_yee_receiver_projection,
)


def apply_yee_receiver_projection(
    projection: YeeReceiverProjection,
    *,
    E: torch.Tensor,
    B: torch.Tensor,
    receiver_index: int,
) -> torch.Tensor:
    """Apply one immutable Yee stencil without breaking field autograd."""
    indices = torch.tensor(
        projection.dof_indices[receiver_index],
        dtype=torch.long,
        device=E.device,
    )
    channel = projection.channel
    field = E if channel.startswith("e") else B
    weights = field.new_tensor(projection.interpolation_weights[receiver_index])
    return (field[indices] * weights).sum()


def yee_component_anchor(
    mesh: TensorMesh,
    position: tuple[float, float, float],
    component: FieldComponent,
) -> tuple[str, int]:
    """Return the dominant legal anchor of one strict multilinear stencil."""
    projection = build_yee_receiver_projection(
        mesh,
        torch.tensor([position], dtype=torch.float64),
        channel=component.value,
        layout="cartesian",
        n_sources=1,
    )
    weights = projection.interpolation_weights[0]
    maximum = max(range(len(weights)), key=weights.__getitem__)
    family = "E" if projection.channel.startswith("e") else "B"
    return family, projection.dof_indices[0][maximum]


__all__ = ["apply_yee_receiver_projection", "yee_component_anchor"]
