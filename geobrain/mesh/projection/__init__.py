"""MeshProjection subpackage: differentiable field projection between meshes.

Public surface: :class:`MeshProjection` (and ``Mesh.projection_to``, which
constructs it). The internal modules:

- ``_operator``: the operator/adjoint classes, mode selection, method
  validation, padding policy;
- ``_kernels``: the per-mode projection kernel strategies the operator
  dispatches to;
- ``_builders``: pure geometry/weight builders the kernels precompute with;
- ``_points``: mesh→scattered-points interpolation
  (:func:`interpolation_matrix` / :func:`interpolate_to_points`).

The historical flat-module import path ``geobrain.mesh.projection``
keeps working for every name (including the private builders the projection
test battery pins).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from ._operator import (
    MeshProjection,
    _MeshProjectionAdjoint as _MeshProjectionAdjoint,
)
from ._points import interpolate_to_points, interpolation_matrix
from ._builders import (
    _COARSER_REL_TOL as _COARSER_REL_TOL,
    _GRID_EPS as _GRID_EPS,
    _OM_TO_TM_TARGET_BLOCK as _OM_TO_TM_TARGET_BLOCK,
    _OVERLAP_REL_TOL as _OVERLAP_REL_TOL,
    _RANGE_EPS as _RANGE_EPS,
    _build_general_weights as _build_general_weights,
    _grid_uncovered as _grid_uncovered,
    _general_uncovered as _general_uncovered,
    _source_domain_bounds as _source_domain_bounds,
    _build_tm_linear_weights as _build_tm_linear_weights,
    _tm_axis_edges as _tm_axis_edges,
    _tm_cell_boxes as _tm_cell_boxes,
    _uniform_target_coarser as _uniform_target_coarser,
    _rect_target_coarser as _rect_target_coarser,
    _build_rect_source_overlap as _build_rect_source_overlap,
    _build_om_to_tm_conservative as _build_om_to_tm_conservative,
    _build_om_to_om_overlap as _build_om_to_om_overlap,
    _nearest_center_rows as _nearest_center_rows,
    _build_barycentric_weights as _build_barycentric_weights,
    _flat_strides as _flat_strides,
    _kron_expand as _kron_expand,
    _replace_rows as _replace_rows,
    _drop_rows as _drop_rows,
    _grid_sample_weight_matrix as _grid_sample_weight_matrix,
    _interpolate_weight_matrix as _interpolate_weight_matrix,
    _validate_field_names as _validate_field_names,
    _validate_k_neighbors as _validate_k_neighbors,
    _build_om_to_tm_index_map as _build_om_to_tm_index_map,
    _tm_extents_match as _tm_extents_match,
    _tm_origins_match as _tm_origins_match,
    _build_tm_to_tm_grid as _build_tm_to_tm_grid,
    _build_tm_to_om_grid as _build_tm_to_om_grid,
    _build_tm_to_om_conservative as _build_tm_to_om_conservative,
)

__all__ = ["MeshProjection", "interpolation_matrix", "interpolate_to_points"]
