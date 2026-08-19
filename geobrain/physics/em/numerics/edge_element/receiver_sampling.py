"""
Edge-element receiver sampling shared by the 3-D inductive operators.

The unstructured (Whitney) counterpart of
:mod:`geobrain.physics.em.numerics.finite_volume.receiver_sampling`, which holds
the structured Yee twin ``yee_component_anchor``. Two σ-independent helpers the
FDEM3D / MT3D / TEM3D edge-FEM receiver paths all share, the edge analogue of
the already-shared Yee anchor, closing the asymmetry where only the structured
path had been factored out:

- :func:`apply_edge_electric_projection`: apply the exact containing-cell
  Whitney basis stored by an :class:`EdgeReceiverProjection`.
- :func:`bind_receivers_to_cells`: retained for narrow legacy tests only;
  production FDEM/TEM/MT receiver paths use strict containing-cell plans.
- :func:`pick_field_component`: select one :class:`FieldComponent` from total
  ``(3,)`` E / B vectors (``HX/HY/HZ`` = ``B / μ₀``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.errors import GeoBrainError
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.receivers import EdgeReceiverProjection


def apply_edge_electric_projection(
    projection: EdgeReceiverProjection,
    edge_dofs: torch.Tensor,
    receiver_index: int,
) -> torch.Tensor:
    """Evaluate a Whitney electric vector at one resolved receiver."""
    indices = torch.tensor(
        projection.local_edge_dof_indices[receiver_index],
        dtype=torch.long,
        device=edge_dofs.device,
    )
    signs = edge_dofs.new_tensor(projection.orientation_signs[receiver_index])
    basis = edge_dofs.new_tensor(projection.basis_weights[receiver_index]).reshape(6, 3)
    return torch.einsum("e,ek->k", signs * edge_dofs[indices], basis)


def bind_receivers_to_cells(
    cell_centers: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Nearest-cell index for each receiver point (argmin over cell centres).

    The edge-FEM receiver-binding rule shared by FDEM3D / MT3D / TEM3D: snap each
    point to the tetrahedral cell whose barycentre is closest (squared Euclidean
    distance), the unstructured analogue of the structured Yee anchor snap. The
    caller then evaluates fields at that cell (the Whitney interpolant at its
    barycentre for E, the per-cell constant curl for B).

    Args:
        cell_centers: ``(n_cells, 3)`` float64 cell barycentres.
        points: ``(n_points, 3)`` float64 receiver positions in metres.

    Returns:
        ``(n_points,)`` long cell indices.
    """
    d2 = ((cell_centers.unsqueeze(0) - points.unsqueeze(1)) ** 2).sum(-1)
    return d2.argmin(dim=1)


def pick_field_component(
    e_vec: torch.Tensor,
    b_vec: torch.Tensor,
    component: FieldComponent,
) -> torch.Tensor:
    """Select one :class:`FieldComponent` from total ``(3,)`` E / B vectors.

    ``EX/EY/EZ`` read ``e_vec``; ``BX/BY/BZ`` read ``b_vec``; ``HX/HY/HZ`` read
    ``b_vec / μ₀``. Returns a 0-d tensor in the inputs' dtype (complex128 for the
    frequency-domain E/B of FDEM3D, float64 for the TEM3D step-off dB/dt passed
    as ``b_vec``); autograd flows through the inputs (pure indexing + constant
    scaling). The TEM3D receiver contract scopes observables to E/B and rejects
    H upstream, so its float64 path never reaches the ``HX/HY/HZ`` branch.

    Args:
        e_vec: ``(3,)`` total E vector at the receiver.
        b_vec: ``(3,)`` total B (or dB/dt) vector at the receiver.
        component: the :class:`FieldComponent` to read out.

    Returns:
        The selected component as a 0-d tensor in the inputs' dtype.

    Raises:
        GeoBrainError: for a component outside EX/EY/EZ/BX/BY/BZ/HX/HY/HZ.
    """
    c = FieldComponent
    if component is c.EX:
        return e_vec[0]
    if component is c.EY:
        return e_vec[1]
    if component is c.EZ:
        return e_vec[2]
    if component is c.BX:
        return b_vec[0]
    if component is c.BY:
        return b_vec[1]
    if component is c.BZ:
        return b_vec[2]
    if component is c.HX:
        return b_vec[0] / MU_0
    if component is c.HY:
        return b_vec[1] / MU_0
    if component is c.HZ:
        return b_vec[2] / MU_0
    raise GeoBrainError(
        f"pick_field_component: unsupported FieldComponent {component!r}; "
        f"expected one of EX/EY/EZ/BX/BY/BZ/HX/HY/HZ",
        object_name="pick_field_component",
        field="component",
        expected="EX/EY/EZ/BX/BY/BZ/HX/HY/HZ",
        actual=component,
    )


__all__ = [
    "apply_edge_electric_projection",
    "bind_receivers_to_cells",
    "pick_field_component",
]
