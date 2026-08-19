"""
DC3D survey-infrastructure dataclasses (ElectrodeArray + DipoleDipoleSurvey).

Private submodule of :mod:`geobrain.physics.em.static.dc3d`.
Public symbols are re-exported from ``dc3d.py``.

These two dataclasses own the electrode positions and the
ABMN-quadripole geometry. :class:`DipoleDipoleSurvey` is pure geometry
(positions + ABMN quadripoles + binding mode); the mesh-specific
electrode→cell binding (with trilinear or nearest-cell weights) is
computed on demand against a mesh via :meth:`DipoleDipoleSurvey.bind`,
which returns a :class:`BoundDipoleDipoleSurvey`. The bulk DC3D
operator, sparse-Poisson assembly, and closed-form σ-Jacobian live in
``dc3d.py``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from ....core import GeoBrainError
from ....mesh import TensorMesh


@dataclass(frozen=True)
class ElectrodeArray:
    """
    Frozen container for ``(x, y, z)`` electrode positions.

    Stores
    positions as ``torch.float64`` 1-D tensors plus an optional
    name → index map for human-readable quadripole construction. The
    operator does not read ``ids`` directly; it is a convenience for
    callers authoring surveys by hand or from CSV.

    Args:
        x, y, z: electrode coordinates. Accepts torch.Tensor or anything
            ``torch.as_tensor`` accepts (numpy array, Python list).
            ``y`` and ``z`` may be omitted (zero-filled, useful for 1D
            or strictly-surface layouts).
        ids: optional ``{label: index}`` map.
    """

    x: torch.Tensor
    y: torch.Tensor | None = None
    z: torch.Tensor | None = None
    ids: Optional[dict] = None

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass __setattr__.
        x_t = torch.as_tensor(self.x, dtype=torch.float64).contiguous().reshape(-1)
        n = int(x_t.numel())
        if self.y is None:
            y_t = torch.zeros(n, dtype=torch.float64)
        else:
            y_t = torch.as_tensor(self.y, dtype=torch.float64).contiguous().reshape(-1)
        if self.z is None:
            z_t = torch.zeros(n, dtype=torch.float64)
        else:
            z_t = torch.as_tensor(self.z, dtype=torch.float64).contiguous().reshape(-1)
        if y_t.numel() != n or z_t.numel() != n:
            raise GeoBrainError(
                "ElectrodeArray: x, y, z must have the same length",
                object_name="ElectrodeArray",
                field="x/y/z",
                expected=n,
                actual=(int(y_t.numel()), int(z_t.numel())),
            )
        object.__setattr__(self, "x", x_t)
        object.__setattr__(self, "y", y_t)
        object.__setattr__(self, "z", z_t)
        if self.ids is not None:
            object.__setattr__(self, "ids", dict(self.ids))

    @property
    def n_electrodes(self) -> int:
        return int(self.x.numel())

    @property
    def positions(self) -> torch.Tensor:
        """``(n_electrodes, 3)`` stacked positions."""
        return torch.stack([self.x, self.y, self.z], dim=-1)

    def __len__(self) -> int:
        return self.n_electrodes


@dataclass
class DipoleDipoleSurvey:
    """
    ABMN-quadripole DC survey geometry (mesh-free).

    Pure geometry: electrode positions, ABMN quadripoles, and the binding
    mode. It carries **no** mesh and computes **no** electrode→cell
    binding at construction. The mesh-specific binding is computed on
    demand by :meth:`bind`, which returns a
    :class:`BoundDipoleDipoleSurvey` carrying the cell indices/weights and
    the mesh-aware sampling/RHS helpers (``ab_currents``, ``sample_phi``,
    ``electrode_cells``, ``trilinear_cells``, ``trilinear_weights``,
    ``n_cells``).

    Two binding modes are supported (resolved at :meth:`bind` time):

    - ``"nearest"``: each electrode is assigned to the single nearest
      cell. Simple and fast, but on a coarse mesh adjacent electrodes
      can collide into the same cell and produce zero ``V_MN``.
    - ``"trilinear"`` (default): each electrode is assigned 8
      surrounding cells with trilinear weights summing to 1, both for
      current injection (RHS) and potential sampling. Removes the
      coarse-mesh collision problem at the cost of a small per-electrode
      pre-computation.

    The geometry-only quantities (``n_obs``, ``mn_pairs``,
    ``geometric_factor``/``geometric_factors``) are available directly
    here without a mesh.

    Args:
        electrodes: :class:`ElectrodeArray`.
        quadripoles: integer array shape ``(n_obs, 4)``, columns
            ``(A, B, M, N)``, each value an index into ``electrodes``.
        binding: ``"trilinear"`` (default) or ``"nearest"``.
    """

    electrodes: ElectrodeArray
    quadripoles: object  # numpy ndarray int64 (n_obs, 4)
    binding: Literal["trilinear", "nearest"] = "trilinear"

    def __post_init__(self) -> None:
        import numpy as np

        q = np.ascontiguousarray(np.asarray(self.quadripoles, dtype=np.int64))
        if q.ndim != 2 or q.shape[1] != 4:
            raise GeoBrainError(
                "quadripoles must have shape (n_obs, 4)",
                object_name="DipoleDipoleSurvey",
                field="quadripoles.shape",
                expected="(n_obs, 4)",
                actual=q.shape,
            )
        if q.size and (q.min() < 0 or q.max() >= self.electrodes.n_electrodes):
            raise GeoBrainError(
                "quadripole indices must lie in [0, n_electrodes)",
                object_name="DipoleDipoleSurvey",
                field="quadripoles",
                expected=f"[0, {self.electrodes.n_electrodes})",
                actual=(int(q.min()), int(q.max())),
            )
        if self.binding not in ("trilinear", "nearest"):
            raise GeoBrainError(
                "binding must be 'trilinear' or 'nearest'",
                object_name="DipoleDipoleSurvey",
                field="binding",
                expected="'trilinear' | 'nearest'",
                actual=self.binding,
            )
        self.quadripoles = q

    # ------------------------------------------------------------------
    # Geometry-only counts and indices (no mesh required)
    # ------------------------------------------------------------------

    @property
    def n_obs(self) -> int:
        return int(self.quadripoles.shape[0])

    def mn_pairs(self):
        """Return the ``(M, N)`` columns, shape ``(n_obs, 2)``."""
        import numpy as np
        return np.ascontiguousarray(self.quadripoles[:, 2:4])

    def geometric_factor(self, i: int) -> float:
        """
        Surface dipole-dipole geometric factor for observation ``i``.

        ``K = 2π / (1/AM - 1/AN - 1/BM + 1/BN)``: the standard
        surface-array form.
        """
        import numpy as np

        a, b, m, n = (int(x) for x in self.quadripoles[i])
        pos = self.electrodes.positions.detach().cpu().numpy()
        am = float(np.linalg.norm(pos[a] - pos[m]))
        an = float(np.linalg.norm(pos[a] - pos[n]))
        bm = float(np.linalg.norm(pos[b] - pos[m]))
        bn = float(np.linalg.norm(pos[b] - pos[n]))
        denom = 1.0 / am - 1.0 / an - 1.0 / bm + 1.0 / bn
        if denom == 0.0:
            raise GeoBrainError(
                f"observation {i}: degenerate geometry, geometric factor is infinite",
                object_name="DipoleDipoleSurvey",
                field=f"geometric_factor({i})",
                expected="non-zero (1/AM - 1/AN - 1/BM + 1/BN)",
                actual=0.0,
            )
        import math as _m
        return 2.0 * _m.pi / denom

    def geometric_factors(self):
        """All geometric factors as a numpy ``(n_obs,)`` array."""
        import numpy as np
        return np.array(
            [self.geometric_factor(i) for i in range(self.n_obs)],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Mesh-specific binding (on demand)
    # ------------------------------------------------------------------

    def bind(self, mesh) -> "BoundDipoleDipoleSurvey":
        """
        Compute the electrode→cell binding against ``mesh``.

        Returns a :class:`BoundDipoleDipoleSurvey` carrying the trilinear /
        nearest cell indices and weights plus the mesh-aware RHS / sampling
        helpers. On a structured ``TensorMesh`` the binding (trilinear or
        nearest) is bit-for-bit reproducible from the original per-axis
        algorithm; on a ``ConnectivityMesh``-only mesh (``UnstructuredMesh`` /
        octree) only ``binding="nearest"`` is defined (nearest cell-centre by
        squared distance) and ``"trilinear"`` raises. Consumers, chiefly
        :class:`IP3D`: can rebind the same geometry to any
        such mesh on demand.
        """
        return _bind_dipole_dipole(self.electrodes, self.quadripoles, self.binding, mesh)


@dataclass
class BoundDipoleDipoleSurvey:
    """
    A :class:`DipoleDipoleSurvey` bound to a specific mesh.

    Coordinate frame: electrode coordinates live on the bound
    :class:`ElectrodeArray` as per-axis ``x``/``y``/``z`` tensors in metres
    (z depth, positive down); ``electrode_cells`` / ``trilinear_cells`` are
    FLAT cell indices in the platform ``(nz, nx, ny)`` C-order.

    Produced by :meth:`DipoleDipoleSurvey.bind`. Carries the geometry
    (electrodes + quadripoles + binding) together with the mesh-specific
    electrode→cell binding (``electrode_cells`` / ``trilinear_cells`` /
    ``trilinear_weights``) and the cell count, and exposes the RHS and
    potential-sampling helpers (``ab_currents`` / ``sample_phi``) that the
    operator reads. On a structured ``TensorMesh`` ``shape`` is ``(nz, nx, ny)``
    and the binding may be trilinear or nearest; on a ``ConnectivityMesh``-only
    mesh ``shape`` is ``None``, ``n_cells`` is stored explicitly, and only the
    nearest binding is populated (trilinear has no 8-corner cell box there).

    Constructed via :func:`_bind_dipole_dipole`; not intended for direct
    user construction.

    Attributes:
        electrodes: electrode coordinate table.
        quadripoles: ABMN electrode-index rows.
        binding: electrode-to-mesh binding policy.
        shape: bound mesh shape.
    """

    electrodes: ElectrodeArray
    quadripoles: object  # numpy ndarray int64 (n_obs, 4)
    binding: Literal["trilinear", "nearest"]
    shape: tuple | None  # (nz, nx, ny) on a StructuredMesh; None on a ConnectivityMesh
    _electrode_cells: object
    _trilinear_cells: object
    _trilinear_weights: object
    # Cell count on a ConnectivityMesh-only mesh (no structured ``shape`` to
    # derive it from). ``None`` on a StructuredMesh, where ``n_cells`` is the
    # ``shape`` product.
    _n_cells: int | None = None

    # ------------------------------------------------------------------
    # Counts and indices
    # ------------------------------------------------------------------

    @property
    def n_obs(self) -> int:
        return int(self.quadripoles.shape[0])

    @property
    def n_cells(self) -> int:
        if self._n_cells is not None:
            return int(self._n_cells)
        nz, nx, ny = self.shape
        return int(nz * nx * ny)

    @property
    def electrode_cells(self):
        """Cell flat index for each electrode, shape ``(n_electrodes,)``."""
        return self._electrode_cells

    @property
    def trilinear_cells(self):
        """``(n_electrodes, 8)`` cell flat indices for trilinear binding."""
        return self._trilinear_cells

    @property
    def trilinear_weights(self):
        """``(n_electrodes, 8)`` weights summing to 1 per row."""
        return self._trilinear_weights

    def ab_currents(self, current: float = 1.0):
        """
        Dense ``(n_cells, n_obs)`` numpy ``float64`` RHS array.

        Trilinear binding spreads each electrode's current across 8
        surrounding cells; nearest binding deposits the full current on
        the snapped cell.
        """
        import numpy as np

        n_cells = self.n_cells
        n_obs = self.n_obs
        b = np.zeros((n_cells, n_obs), dtype=np.float64)
        if self.binding == "nearest":
            cells = self._electrode_cells
            a_cells = cells[self.quadripoles[:, 0]]
            b_cells = cells[self.quadripoles[:, 1]]
            np.add.at(b, (a_cells, np.arange(n_obs)), +float(current))
            np.add.at(b, (b_cells, np.arange(n_obs)), -float(current))
            return b
        a_idx = self.quadripoles[:, 0]
        b_idx = self.quadripoles[:, 1]
        tri_c = self._trilinear_cells
        tri_w = self._trilinear_weights
        for col, e_a, e_b in zip(np.arange(n_obs), a_idx, b_idx):
            np.add.at(b, (tri_c[e_a], col), +float(current) * tri_w[e_a])
            np.add.at(b, (tri_c[e_b], col), -float(current) * tri_w[e_b])
        return b

    def sample_phi(self, phi):
        """
        Sample ``(phi_M, phi_N)`` at M and N electrodes per observation.

        ``phi`` is a numpy ``(n_cells, n_obs)`` array. Returns a tuple
        of two ``(n_obs,)`` arrays.
        """
        import numpy as np

        n_obs = self.n_obs
        obs_idx = np.arange(n_obs)
        m_e = self.quadripoles[:, 2]
        n_e = self.quadripoles[:, 3]
        if self.binding == "nearest":
            m_cells = self._electrode_cells[m_e]
            n_cells = self._electrode_cells[n_e]
            return phi[m_cells, obs_idx], phi[n_cells, obs_idx]
        tri_c = self._trilinear_cells
        tri_w = self._trilinear_weights
        phi_m = np.zeros(n_obs, dtype=phi.dtype)
        phi_n = np.zeros(n_obs, dtype=phi.dtype)
        for c in range(8):
            phi_m += tri_w[m_e, c] * phi[tri_c[m_e, c], obs_idx]
            phi_n += tri_w[n_e, c] * phi[tri_c[n_e, c], obs_idx]
        return phi_m, phi_n

    def mn_pairs(self):
        """Return the ``(M, N)`` columns, shape ``(n_obs, 2)``."""
        import numpy as np
        return np.ascontiguousarray(self.quadripoles[:, 2:4])


def _bind_dipole_dipole(
    electrodes: ElectrodeArray,
    quadripoles,
    binding: Literal["trilinear", "nearest"],
    mesh,
) -> BoundDipoleDipoleSurvey:
    """
    Compute the electrode→cell binding for ``electrodes`` against ``mesh``.

    Module-level helper backing :meth:`DipoleDipoleSurvey.bind`. Two paths:

    - **StructuredMesh** (``TensorMesh``): the historical per-axis nearest
      snapping + trilinear corner weights, byte-for-byte unchanged from the
      original ``DipoleDipoleSurvey.__post_init__``.
    - **ConnectivityMesh-only** (``UnstructuredMesh`` / octree, no structured
      ``shape``): a connectivity-generic ``"nearest"`` binding, each electrode
      snaps to the cell whose centre (from ``mesh.cell_centers()``) is the
      nearest in 3-D (argmin squared distance), matching DC3D's unstructured
      electrode resolution. ``"trilinear"`` is NOT defined without a structured
      grid (no 8-corner cell box), so it RAISES rather than silently degrading;
      pass ``binding="nearest"`` for an unstructured mesh.
    """
    import numpy as np

    from ....mesh.capabilities import StructuredMesh

    if mesh.n_dim != 3:
        raise GeoBrainError(
            "DipoleDipoleSurvey.bind requires a 3-D mesh",
            object_name="DipoleDipoleSurvey",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )

    # ConnectivityMesh-only path (no structured shape): nearest-centre binding.
    if not mesh.declares(StructuredMesh):
        if binding != "nearest":
            raise GeoBrainError(
                "DipoleDipoleSurvey.bind: binding='trilinear' needs a structured "
                "mesh (8-corner cell box). On a ConnectivityMesh-only mesh "
                "(UnstructuredMesh / octree) use binding='nearest'.",
                object_name="DipoleDipoleSurvey",
                field="binding",
                expected="'nearest' on a non-structured mesh",
                actual=binding,
            )
        return _bind_dipole_dipole_connectivity(electrodes, quadripoles, mesh)

    if not isinstance(mesh, TensorMesh):
        raise GeoBrainError(
            "DipoleDipoleSurvey.bind structured path requires a TensorMesh",
            object_name="DipoleDipoleSurvey",
            field="mesh",
            expected="TensorMesh",
            actual=type(mesh).__name__,
        )

    q = np.ascontiguousarray(np.asarray(quadripoles, dtype=np.int64))

    # Build per-axis cell-centre coordinate arrays (numpy float64)
    # from mesh.cell_widths so non-uniform meshes work. Mesh shape is
    # ``(nz, nx, ny)``; spacing tuple is ``(dz, dx, dy)``; cell_widths
    # tuple is ``(wz, wx, wy)`` (axis-1 = x, axis-2 = y).
    nz, nx, ny = mesh.shape
    # Platform frame ``(nz, nx, ny)`` -> centre lines unpack as (z, x, y).
    # Origin-aware (the hand-built cumsum arrays silently assumed origin 0).
    # NOTE the arithmetic family changed on uniform meshes (multiplicative
    # ``(arange+0.5)*d`` vs historical cumsum): equal bitwise for exactly-
    # representable spacings (all pinned fixtures), otherwise within 1 ulp;
    # an electrode-binding decision can only differ for an electrode within
    # 1 ulp of a cell-centre plane.
    cz_t, cx_t, cy_t = mesh.center_lines()
    cx = cx_t.numpy()
    cy = cy_t.numpy()
    cz = cz_t.numpy()

    n_el = electrodes.n_electrodes
    ex = electrodes.x.detach().cpu().numpy().astype(np.float64, copy=False)
    ey = electrodes.y.detach().cpu().numpy().astype(np.float64, copy=False)
    ez = electrodes.z.detach().cpu().numpy().astype(np.float64, copy=False)

    # Nearest-cell binding (always cached for diagnostics + the
    # 'nearest' branch). Snap each electrode to the cell whose centre
    # is closest along each axis.
    def _nearest_idx(centres, p, n):
        i = int(np.argmin(np.abs(centres - p)))
        return int(np.clip(i, 0, n - 1))

    cells = np.empty(n_el, dtype=np.int64)
    for e in range(n_el):
        ix = _nearest_idx(cx, ex[e], nx)
        iy = _nearest_idx(cy, ey[e], ny)
        iz = _nearest_idx(cz, ez[e], nz)
        # Flat index follows the assembler convention
        # ``flat = iy + ix*ny + iz*nx*ny`` (y-fastest, ``(nz, nx, ny)``).
        cells[e] = iy + ix * ny + iz * nx * ny

    # Trilinear weights: 8 surrounding cells per electrode (clipped
    # to mesh extent), weights summing to 1.
    tri_cells = np.empty((n_el, 8), dtype=np.int64)
    tri_weights = np.zeros((n_el, 8), dtype=np.float64)
    for e in range(n_el):
        px, py, pz = float(ex[e]), float(ey[e]), float(ez[e])
        ix1 = int(np.clip(np.searchsorted(cx, px) - 1, 0, nx - 2))
        iy1 = int(np.clip(np.searchsorted(cy, py) - 1, 0, ny - 2))
        iz1 = int(np.clip(np.searchsorted(cz, pz) - 1, 0, nz - 2))
        x0, x1_ = float(cx[ix1]), float(cx[ix1 + 1])
        y0, y1_ = float(cy[iy1]), float(cy[iy1 + 1])
        z0, z1_ = float(cz[iz1]), float(cz[iz1 + 1])
        u = float(np.clip((px - x0) / max(x1_ - x0, 1e-30), 0.0, 1.0))
        v = float(np.clip((py - y0) / max(y1_ - y0, 1e-30), 0.0, 1.0))
        w = float(np.clip((pz - z0) / max(z1_ - z0, 1e-30), 0.0, 1.0))
        for corner, (du, dv, dw) in enumerate([
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]):
            ix = ix1 + du
            iy = iy1 + dv
            iz = iz1 + dw
            wx_c = (1.0 - u) if du == 0 else u
            wy_c = (1.0 - v) if dv == 0 else v
            wz_c = (1.0 - w) if dw == 0 else w
            tri_cells[e, corner] = iy + ix * ny + iz * nx * ny
            tri_weights[e, corner] = wx_c * wy_c * wz_c

    return BoundDipoleDipoleSurvey(
        electrodes=electrodes,
        quadripoles=q,
        binding=binding,
        shape=(nz, nx, ny),
        _electrode_cells=cells,
        _trilinear_cells=tri_cells,
        _trilinear_weights=tri_weights,
    )


def _bind_dipole_dipole_connectivity(
    electrodes: ElectrodeArray,
    quadripoles,
    mesh,
) -> BoundDipoleDipoleSurvey:
    """Nearest-centre electrode→cell binding on a ConnectivityMesh-only mesh.

    Each electrode snaps to the cell whose centre (``mesh.cell_centers()``,
    platform ``(z, x, y)`` columns, metres) is the nearest in 3-D by squared
    distance, the same rule DC3D's unstructured path uses
    (:meth:`DC3D._nearest_cell`). No structured ``shape`` is needed; the bound
    survey carries ``shape=None`` and an explicit ``_n_cells``. The trilinear
    arrays are left empty because trilinear binding is undefined here (the caller
    has already rejected ``binding != "nearest"``).
    """
    import numpy as np

    q = np.ascontiguousarray(np.asarray(quadripoles, dtype=np.int64))
    centres = mesh.cell_centers().detach().cpu().numpy().astype(np.float64)  # (n_cells, 3)
    n_cells = int(centres.shape[0])

    ex = electrodes.x.detach().cpu().numpy().astype(np.float64, copy=False)
    ey = electrodes.y.detach().cpu().numpy().astype(np.float64, copy=False)
    ez = electrodes.z.detach().cpu().numpy().astype(np.float64, copy=False)
    n_el = electrodes.n_electrodes

    # Electrode coords in the mesh's platform ``(z, x, y)`` column order so the
    # nearest-centre comparison is axis-consistent with ``cell_centers()``.
    targets = np.stack([ez, ex, ey], axis=1)  # (n_el, 3)
    cells = np.empty(n_el, dtype=np.int64)
    for e in range(n_el):
        d2 = ((centres - targets[e]) ** 2).sum(axis=1)
        cells[e] = int(np.argmin(d2))

    empty_cells = np.empty((n_el, 8), dtype=np.int64)
    empty_weights = np.zeros((n_el, 8), dtype=np.float64)
    return BoundDipoleDipoleSurvey(
        electrodes=electrodes,
        quadripoles=q,
        binding="nearest",
        shape=None,
        _electrode_cells=cells,
        _trilinear_cells=empty_cells,
        _trilinear_weights=empty_weights,
        _n_cells=n_cells,
    )


__all__ = ["ElectrodeArray", "DipoleDipoleSurvey", "BoundDipoleDipoleSurvey"]
