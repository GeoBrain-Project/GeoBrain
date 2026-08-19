"""
Finite-volume numerical primitives for EM (E6a).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .averaging import assemble_cell_to_edge_averaging
from .helmholtz_2d import assemble_helmholtz_2d, assemble_helmholtz_2d_tm
from .helmholtz_fv import (
    HelmholtzBoundaryPartition,
    assemble_helmholtz_fv,
    assemble_helmholtz_fv_tm,
    partition_helmholtz_boundary,
)
from .mixed_boundary import mixed_boundary_alpha
from .poisson import assemble_poisson_2d, assemble_poisson_3d
from .poisson_fv import assemble_poisson_fv
from .poisson_octree import (
    _find_octree_face_neighbors,
    assemble_poisson_octree_3d,
    solve_poisson_octree_3d,
)
from .poisson_octree_mpfa import assemble_poisson_octree_3d_mpfa
from .singularity_removal import effective_cell_radius, primary_potential
from .curl_curl import (
    assemble_yee_curl_curl_stiffness,
    yee_curl_curl_assemble,
    yee_curl_e_to_f,
    yee_edge_count,
    yee_face_count,
)

__all__ = [
    "HelmholtzBoundaryPartition",
    "_find_octree_face_neighbors",
    "assemble_cell_to_edge_averaging",
    "assemble_helmholtz_2d",
    "assemble_helmholtz_2d_tm",
    "assemble_helmholtz_fv",
    "assemble_helmholtz_fv_tm",
    "assemble_poisson_2d",
    "assemble_poisson_3d",
    "assemble_poisson_fv",
    "assemble_poisson_octree_3d",
    "assemble_poisson_octree_3d_mpfa",
    "assemble_yee_curl_curl_stiffness",
    "effective_cell_radius",
    "mixed_boundary_alpha",
    "partition_helmholtz_boundary",
    "primary_potential",
    "solve_poisson_octree_3d",
    "yee_curl_curl_assemble",
    "yee_curl_e_to_f",
    "yee_edge_count",
    "yee_face_count",
]
