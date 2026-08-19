"""
Numerical primitives for the EM physics subpackage.

- :mod:`.edge_element`: lowest-order Nédélec (Whitney) edge elements on
  tetrahedral meshes: local mass / curl-curl matrices + differentiable global
  COO assembly and matrix-free matvec (the 3-D unstructured curl-curl seam).
- :mod:`.finite_volume`: cell-centred FV operators (Poisson, Helmholtz 2-D,
  Yee curl-curl). Octree variants ship in
  :mod:`.finite_volume.poisson_octree` (TPFA) and
  :mod:`.finite_volume.poisson_octree_mpfa` (MPFA-O placeholder).
- :mod:`.hankel`: DLF filters for layered-Earth EM responses.
- :mod:`.layered`: 1-D layered MT recursion.
- :mod:`.sparse`: closed-form σ-Jacobian sparse adjoint bridges.
- :mod:`.time_stepping`: BDF1 / BDF2 time integrators for transient EM.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from . import edge_element, finite_volume, sparse  # noqa: F401

__all__ = ["edge_element", "finite_volume", "sparse"]
