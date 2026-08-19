"""
Lowest-order Nédélec (Whitney) edge elements on tetrahedral meshes.

The numerical substrate of the em 3-D curl-curl family on unstructured
(tetrahedral) meshes, the FDEM3D/TEM3D/MT3D unstructured branches build on it
today (each operator's ``_forward_edge_fem`` path drives this machinery). Three
layers:

- :mod:`.whitney`: batched per-cell geometry and closed-form local 6×6
  matrices (barycentric gradients, unit-σ Whitney mass, unit-coefficient
  curl-curl stiffness).
- :mod:`.assembly`: the :class:`EdgeAssemblyPlan` precomputation over an
  :class:`~geobrain.mesh.capabilities.EdgeConnectivityMesh`, the
  differentiable COO assembly ``A = stiffness_coeff·K + mass_coeff·M``, and a
  matrix-free gather/``scatter_add`` matvec.
- :mod:`.topology`: derived boundary topology (which global edges lie on the
  outer surface), the input of the operators' PEC pinning.
- :mod:`.pec_solve`: the assemble → PEC-pin → solve tail the FDEM3D / MT3D /
  TEM3D secondary solves share (the editorial layer above the leaf primitives).
- :mod:`.receiver_sampling`: the edge analogue of the structured Yee anchor:
  nearest-cell receiver binding and the E/B/H component selector.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .assembly import (
    EdgeAssemblyPlan,
    assemble_edge_operator,
    build_edge_assembly_plan,
    edge_operator_matvec,
)
from .assets import EdgeFemAssetsBase
from ..mesh_path import resolve_inductive_mesh_path  # re-export door: consumers import it from this package
from .pec_solve import (
    assemble_pec_pinned_operator,
    solve_pec_pinned_edge_system,
)
from .receiver_sampling import bind_receivers_to_cells, pick_field_component
from .topology import (
    PecPinPlan,
    apply_pec_pin,
    boundary_edge_mask,
    build_pec_pin_plan,
    pec_zero_rhs,
)
from .whitney import (
    TET_LOCAL_EDGES,
    barycentric_gradients,
    local_curl_curl,
    local_edge_mass,
    whitney_centroid_factors,
)

__all__ = [
    "EdgeAssemblyPlan",
    "EdgeFemAssetsBase",
    "PecPinPlan",
    "TET_LOCAL_EDGES",
    "apply_pec_pin",
    "assemble_edge_operator",
    "assemble_pec_pinned_operator",
    "barycentric_gradients",
    "bind_receivers_to_cells",
    "boundary_edge_mask",
    "build_edge_assembly_plan",
    "build_pec_pin_plan",
    "edge_operator_matvec",
    "local_curl_curl",
    "local_edge_mass",
    "pec_zero_rhs",
    "pick_field_component",
    "resolve_inductive_mesh_path",
    "solve_pec_pinned_edge_system",
    "whitney_centroid_factors",
]
