"""Interactive 3-D visualization: dual backend (PyVista / Plotly).

GeoBrain's outward-facing 3-D viewer. Compose a scene once and render it either
way:

- **PyVista** (VTK): full desktop / Jupyter rendering with volume rendering,
  movable slice & threshold widgets, and self-contained interactive HTML export.
- **Plotly**: pure-browser / notebook 3-D (no VTK runtime needed); every figure
  is natively interactive and trivially embeddable in a web report.

Both backends share one fluent API::

    from geobrain.vis import Scene3D
    scene = (
        Scene3D(backend="pyvista")           # or "plotly"
        .add_volume(vp, mesh=mesh, cmap="geo_velocity")
        .add_slices(vp, mesh=mesh)
        .add_points(receivers, label="receivers")
        .add_wells(well_path)
    )
    scene.show()                              # interactive window / notebook
    scene.to_html("model.html")              # shareable interactive 3-D

A scalar field is given either on a 3-D :class:`~geobrain.mesh.TensorMesh`
(``mesh=``) or as a bare 3-D array with ``spacing`` / ``origin``. The platform
axis order is ``(nz, nx, ny)`` (axis 1 = x, axis 2 = y; matching ``mesh.shape``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from geobrain.core.errors import GeoBrainError
from geobrain.vis._utils import to_numpy

_BACKENDS = ("pyvista", "plotly")


def _in_jupyter() -> bool:
    """True when running inside a Jupyter / IPython notebook kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:                      # pragma: no cover - IPython absent
        return False


def set_jupyter_backend(name: str = "trame") -> None:
    """Set PyVista's Jupyter rendering backend (``'trame'`` for live inline 3-D,
    ``'static'`` for a screenshot, ``'html'`` for a self-contained widget).

    Call once near the top of a notebook; :meth:`Scene3D.show` then renders inline.
    """
    import pyvista as pv
    pv.set_jupyter_backend(name)


# =========================================================================
# Geometry bridge (TensorMesh / raw array -> node + centre coordinates)
# =========================================================================

def _node_arrays(mesh, spacing, origin, shape):
    """Per-axis node coordinates ``(x_nodes, y_nodes, z_nodes)``.

    From a 3-D TensorMesh (cumulative ``cell_widths``) or a regular grid defined
    by ``spacing`` / ``origin``. ``shape`` is ``(nz, nx, ny)``.

    Bare-array ``spacing`` / ``origin`` are **z-first**, ``(dz, dx, dy)`` /
    ``(oz, ox, oy)``: matching the ``(nz, nx, ny)`` field order, the
    ``TensorMesh.spacing`` convention, and the :class:`Slicer` viewer, so a tuple
    built for one viewer is not silently transposed by the other.
    """
    oz, ox, oy = (float(origin[0]), float(origin[1]), float(origin[2]))
    if mesh is not None:
        cw = mesh.cell_widths                      # (wz, wx, wy) by convention
        wz = to_numpy(cw[0]).astype(float)
        wx = to_numpy(cw[1]).astype(float)
        wy = to_numpy(cw[2]).astype(float)
        # Honour the mesh's own coordinate origin (defaults to zeros, so a
        # legacy origin-0 mesh is unchanged). It is ADDED to any explicit
        # ``origin`` argument, so a shifted mesh places its nodes at the correct
        # physical position without silently overriding a caller-supplied offset.
        mesh_origin = getattr(mesh, "origin", None)
        if mesh_origin is not None and len(mesh_origin) == 3:
            oz += float(mesh_origin[0])
            ox += float(mesh_origin[1])
            oy += float(mesh_origin[2])
    else:
        nz, nx, ny = shape
        sz, sx, sy = float(spacing[0]), float(spacing[1]), float(spacing[2])
        wx = np.full(nx, sx)
        wy = np.full(ny, sy)
        wz = np.full(nz, sz)
    x_nodes = ox + np.concatenate([[0.0], np.cumsum(wx)])
    y_nodes = oy + np.concatenate([[0.0], np.cumsum(wy)])
    z_nodes = oz + np.concatenate([[0.0], np.cumsum(wz)])
    return x_nodes, y_nodes, z_nodes


def _resolve_field(field, mesh, spacing, origin):
    """Return ``(field_3d [nz,nx,ny], x_nodes, y_nodes, z_nodes)``."""
    arr = np.asarray(to_numpy(field), dtype=np.float64)
    if mesh is not None:
        shape = tuple(int(s) for s in mesh.shape)
        if len(shape) != 3:
            raise GeoBrainError(
                "Scene3D needs a 3-D mesh / field",
                object_name="Scene3D", field="mesh.shape",
                expected="(nz, nx, ny)", actual=shape,
            )
        arr = arr.reshape(shape)
    else:
        if arr.ndim != 3:
            raise GeoBrainError(
                "field must be a 3-D array when no mesh is given",
                object_name="Scene3D", field="field",
                expected="(nz, nx, ny)", actual=tuple(arr.shape),
            )
        shape = arr.shape
    x_nodes, y_nodes, z_nodes = _node_arrays(mesh, spacing, origin, shape)
    return arr, x_nodes, y_nodes, z_nodes


def _centres(nodes: np.ndarray) -> np.ndarray:
    return 0.5 * (nodes[:-1] + nodes[1:])


# VTK hexahedron corner offsets (half-width units) and its 12 triangular faces.
_HEX_OFFSETS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
_HEX_FACES = np.array(
    [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]], dtype=np.int64)


def _octree_corners(mesh):
    """``(points (8*n_cells, 3), n_cells)``: the 8 hex corners of every octree cell.

    Corner ``k`` of cell ``c`` is row ``8*c + k`` (VTK hexahedron order), built from
    the per-cell ``cell_centers`` ± ``half_widths`` (variable cell sizes).

    ``cell_centers`` / ``half_widths`` are in **mesh-axis order ``(z, x, y)``**
    (column 0 = z/depth, column 1 = x, column 2 = y, the platform
    ``(nz, nx, ny)`` convention), but the plotters read column 0 as scene-X and
    column 2 as scene-Z. Reorder the columns to ``(x, y, z)`` (the ``[1, 2, 0]``
    permutation below) so add_octree puts mesh-x on scene-X, matching every
    other scene path (``_node_arrays`` / ``_mesh_xyzv`` / ``_pv_rectilinear``).
    The hex is axis-aligned, so the column reorder is a pure axis relabel and the
    ``_HEX_FACES`` winding is preserved.
    """
    centers = np.asarray(to_numpy(mesh.cell_centers()), dtype=np.float64)
    half = np.asarray(to_numpy(mesh.half_widths), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise GeoBrainError(
            "add_octree needs a 3-D OctreeMesh",
            object_name="Scene3D", field="mesh",
            expected="(n_cells, 3) centers", actual=tuple(centers.shape),
        )
    centers = centers[:, [1, 2, 0]]                # (z, x, y) -> (x, y, z) scene order
    half = half[:, [1, 2, 0]]
    pts = (centers[:, None, :] + _HEX_OFFSETS[None, :, :] * half[:, None, :]).reshape(-1, 3)
    return pts, centers.shape[0]


def _pv_rectilinear(pv, f3d, xn, yn, zn, name):
    """A ``pv.RectilinearGrid`` carrying ``f3d`` (nz,nx,ny) as x-fastest cell data."""
    grid = pv.RectilinearGrid(xn, yn, zn)
    # Grid cells are (nx, ny, nz) x-fastest; f3d is (nz, nx, ny). transpose(1,2,0)
    # -> (nx, ny, nz) so cell (ix,iy,iz) reads f3d[iz,ix,iy].
    grid.cell_data[name] = f3d.transpose(1, 2, 0).ravel(order="F")
    return grid


def _plotly_field_trace(go, f3d, xn, yn, zn, *, style, levels, cmap, clim, name):
    """A single Plotly volume / slices / iso-surface trace for one 3-D frame."""
    cx, cy, cz = _centres(xn), _centres(yn), _centres(zn)
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")   # (nx, ny, nz)
    v = f3d.transpose(1, 2, 0).ravel()                    # (nz,nx,ny) -> (nx,ny,nz)
    cmin = None if clim is None else clim[0]
    cmax = None if clim is None else clim[1]
    common = dict(x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=v,
                  colorscale=_plotly_cmap(cmap), cmin=cmin, cmax=cmax,
                  colorbar=dict(title=name))
    if style == "isosurface":
        if levels:
            lv = list(levels)
        elif clim is not None:                              # fixed default: clim midpoint
            lv = [0.5 * (clim[0] + clim[1])]                # (matches PyVista _pv_add)
        else:
            lv = [float(np.mean(v))]
        return go.Isosurface(**common, isomin=min(lv), isomax=max(lv), surface_count=len(lv),
                             caps=dict(x_show=False, y_show=False, z_show=False))
    if style == "volume":
        return go.Volume(**common, opacity=0.1, surface_count=15)
    # Default slice plane = domain mid-plane 0.5*(node_lo+node_hi), matching
    # PyVista's slice_orthogonal() (dataset-bounds center), not the median
    # cell-CENTER (which is half a cell off on even-dimensioned axes).
    mx, my, mz = (0.5 * (xn[0] + xn[-1]), 0.5 * (yn[0] + yn[-1]), 0.5 * (zn[0] + zn[-1]))
    return go.Volume(**common, opacity=1.0, surface_count=1,                 # orthogonal slices
                     slices=dict(x=dict(show=True, locations=[mx]),
                                 y=dict(show=True, locations=[my]),
                                 z=dict(show=True, locations=[mz])),
                     caps=dict(x_show=False, y_show=False, z_show=False))


# =========================================================================
# PyVista backend
# =========================================================================

class _PyVistaBackend:
    def __init__(self, *, title, background, window_size, off_screen):
        try:
            import pyvista as pv
        except ImportError as exc:  # pragma: no cover - optional dep
            raise GeoBrainError(
                "Scene3D(backend='pyvista') requires pyvista. "
                "Install with: pip install 'geobrain[vis]'  (or pip install pyvista)",
                object_name="Scene3D", field="backend",
            ) from exc
        self._pv = pv
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size),
                                  title=title or "GeoBrain 3-D")
        self.plotter.set_background(background)
        self._has_scalar_bar = False

    def _grid(self, field, mesh, spacing, origin, name):
        f3d, xn, yn, zn = _resolve_field(field, mesh, spacing, origin)
        return _pv_rectilinear(self._pv, f3d, xn, yn, zn, name)

    def add_volume(self, field, *, mesh, spacing, origin, cmap, opacity, clim, name):
        grid = self._grid(field, mesh, spacing, origin, name).cell_data_to_point_data()
        self.plotter.add_volume(grid, scalars=name, cmap=cmap, opacity=opacity,
                                clim=clim, scalar_bar_args={"title": name})

    def add_slices(self, field, *, mesh, spacing, origin, cmap, clim, name, x, y, z):
        grid = self._grid(field, mesh, spacing, origin, name)
        slices = grid.slice_orthogonal(x=x, y=y, z=z)
        self.plotter.add_mesh(slices, scalars=name, cmap=cmap, clim=clim,
                              scalar_bar_args={"title": name})

    def add_isosurface(self, field, levels, *, mesh, spacing, origin, cmap, clim, name, opacity):
        grid = self._grid(field, mesh, spacing, origin, name).cell_data_to_point_data()
        contour = grid.contour(isosurfaces=list(levels), scalars=name)
        self.plotter.add_mesh(contour, scalars=name, cmap=cmap, clim=clim,
                              opacity=opacity, scalar_bar_args={"title": name})

    def add_threshold(self, field, *, mesh, spacing, origin, cmap, clim, name, value, invert):
        grid = self._grid(field, mesh, spacing, origin, name)
        if value is None:                      # interactive widget
            self.plotter.add_mesh_threshold(grid, scalars=name, cmap=cmap, invert=invert)
        else:
            self.plotter.add_mesh(grid.threshold(value, scalars=name, invert=invert),
                                  scalars=name, cmap=cmap, clim=clim)

    def add_points(self, coords, *, scalars, size, cmap, color, label, render_spheres):
        cloud = self._pv.PolyData(np.asarray(coords, dtype=np.float64))
        kw = dict(point_size=size, render_points_as_spheres=render_spheres, label=label)
        if scalars is not None:
            cloud["scalars"] = np.asarray(to_numpy(scalars)).ravel()
            self.plotter.add_mesh(cloud, scalars="scalars", cmap=cmap, **kw)
        else:
            self.plotter.add_mesh(cloud, color=color or "red", **kw)

    def add_wells(self, paths, *, radius, color, labels):
        for k, path in enumerate(paths):
            p = np.asarray(path, dtype=np.float64)
            tube = self._pv.Spline(p, max(2, p.shape[0])).tube(radius=radius)
            self.plotter.add_mesh(tube, color=color,
                                  label=None if labels is None else labels[k])

    def add_octree(self, mesh, field, *, cmap, clim, name, opacity, show_edges, threshold):
        pv = self._pv
        pts, n = _octree_corners(mesh)
        conn = np.empty((n, 9), dtype=np.int64)
        conn[:, 0] = 8
        conn[:, 1:] = np.arange(8 * n).reshape(n, 8)
        ctypes = np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        grid = pv.UnstructuredGrid(conn.ravel(), ctypes, pts)
        grid.cell_data[name] = np.asarray(to_numpy(field), dtype=np.float64).ravel()
        if threshold is not None:
            grid = grid.threshold(threshold, scalars=name)
        self.plotter.add_mesh(grid, scalars=name, cmap=cmap, clim=clim, opacity=opacity,
                              show_edges=show_edges, scalar_bar_args={"title": name})

    def show(self, **kw):
        return self.plotter.show(**kw)

    def to_html(self, path):
        try:
            self.plotter.export_html(path)
        except ImportError as exc:  # pragma: no cover - trame/server quirk
            raise GeoBrainError(
                "PyVista HTML export needs a trame/Jupyter context. For headless, "
                "scriptable interactive HTML use Scene3D(backend='plotly').",
                object_name="Scene3D", field="to_html",
            ) from exc
        return path

    def screenshot(self, path):
        self.plotter.screenshot(path)
        return path


# =========================================================================
# Plotly backend
# =========================================================================

class _PlotlyBackend:
    def __init__(self, *, title, background, window_size, off_screen):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover - optional dep
            raise GeoBrainError(
                "Scene3D(backend='plotly') requires plotly. "
                "Install with: pip install 'geobrain[vis]'  (or pip install plotly)",
                object_name="Scene3D", field="backend",
            ) from exc
        self._go = go
        self.fig = go.Figure()
        self.fig.update_layout(
            title=title or "GeoBrain 3-D", template="plotly_white",
            width=window_size[0], height=window_size[1],
            scene=dict(aspectmode="data",
                       xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        )

    def _mesh_xyzv(self, field, mesh, spacing, origin):
        f3d, xn, yn, zn = _resolve_field(field, mesh, spacing, origin)
        cx, cy, cz = _centres(xn), _centres(yn), _centres(zn)
        # meshgrid in (nx, ny, nz) then flatten consistently with field (nz,nx,ny).
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        value = f3d.transpose(1, 2, 0)         # (nz,nx,ny) -> (nx,ny,nz)
        return gx.ravel(), gy.ravel(), gz.ravel(), value.ravel(), (xn, yn, zn, cx, cy, cz), f3d

    def add_volume(self, field, *, mesh, spacing, origin, cmap, opacity, clim, name):
        x, y, z, v, _, _ = self._mesh_xyzv(field, mesh, spacing, origin)
        op = 0.1 if isinstance(opacity, str) else float(opacity)
        self.fig.add_trace(self._go.Volume(
            x=x, y=y, z=z, value=v, opacity=op, surface_count=18,
            colorscale=_plotly_cmap(cmap), colorbar=dict(title=name),
            cmin=None if clim is None else clim[0],
            cmax=None if clim is None else clim[1], name=name))

    def add_slices(self, field, *, mesh, spacing, origin, cmap, clim, name, x, y, z):
        gx, gy, gz, v, (xn, yn, zn, cx, cy, cz), f3d = self._mesh_xyzv(field, mesh, spacing, origin)
        # default: one slice per axis at the domain MID-PLANE 0.5*(node_lo+node_hi),
        # matching PyVista's slice_orthogonal() (dataset-bounds center), not the
        # median cell-CENTER, which is half a cell off on even-dimensioned axes.
        sx = [0.5 * (xn[0] + xn[-1])] if x is None else np.atleast_1d(x)
        sy = [0.5 * (yn[0] + yn[-1])] if y is None else np.atleast_1d(y)
        sz = [0.5 * (zn[0] + zn[-1])] if z is None else np.atleast_1d(z)
        self.fig.add_trace(self._go.Volume(
            x=gx, y=gy, z=gz, value=v, opacity=1.0, surface_count=1,
            colorscale=_plotly_cmap(cmap), colorbar=dict(title=name),
            cmin=None if clim is None else clim[0],
            cmax=None if clim is None else clim[1],
            slices=dict(x=dict(show=True, locations=list(sx)),
                        y=dict(show=True, locations=list(sy)),
                        z=dict(show=True, locations=list(sz))),
            caps=dict(x_show=False, y_show=False, z_show=False), name=name))

    def add_isosurface(self, field, levels, *, mesh, spacing, origin, cmap, clim, name, opacity):
        x, y, z, v, _, _ = self._mesh_xyzv(field, mesh, spacing, origin)
        lv = list(levels)
        self.fig.add_trace(self._go.Isosurface(
            x=x, y=y, z=z, value=v, isomin=min(lv), isomax=max(lv),
            surface_count=len(lv), opacity=opacity, colorscale=_plotly_cmap(cmap),
            cmin=None if clim is None else clim[0],
            cmax=None if clim is None else clim[1],
            colorbar=dict(title=name), caps=dict(x_show=False, y_show=False, z_show=False),
            name=name))

    def add_threshold(self, field, *, mesh, spacing, origin, cmap, clim, name, value, invert):
        x, y, z, v, _, _ = self._mesh_xyzv(field, mesh, spacing, origin)
        thr = float(np.median(v)) if value is None else float(value)
        keep = (v <= thr) if invert else (v >= thr)
        self.fig.add_trace(self._go.Scatter3d(
            x=x[keep], y=y[keep], z=z[keep], mode="markers",
            marker=dict(size=3, color=v[keep], colorscale=_plotly_cmap(cmap),
                        cmin=None if clim is None else clim[0],
                        cmax=None if clim is None else clim[1],
                        colorbar=dict(title=name)), name=name))

    def add_points(self, coords, *, scalars, size, cmap, color, label, render_spheres):
        c = np.asarray(coords, dtype=np.float64)
        marker = dict(size=max(2, size // 2))
        if scalars is not None:
            marker.update(color=np.asarray(to_numpy(scalars)).ravel(),
                          colorscale=_plotly_cmap(cmap), colorbar=dict(title=label or ""))
        else:
            marker.update(color=color or "red")
        self.fig.add_trace(self._go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers",
            marker=marker, name=label or "points"))

    def add_wells(self, paths, *, radius, color, labels):
        for k, path in enumerate(paths):
            p = np.asarray(path, dtype=np.float64)
            self.fig.add_trace(self._go.Scatter3d(
                x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                line=dict(color=color, width=6),
                name="well" if labels is None else labels[k]))

    def add_octree(self, mesh, field, *, cmap, clim, name, opacity, show_edges, threshold):
        pts, n = _octree_corners(mesh)
        vals = np.asarray(to_numpy(field), dtype=np.float64).ravel()
        idx = np.arange(n) if threshold is None else np.where(vals >= float(threshold))[0]
        # 12 triangles per kept cell, indexing the global corner array (8*c + face).
        faces = (idx[:, None, None] * 8 + _HEX_FACES[None, :, :]).reshape(-1, 3)
        self.fig.add_trace(self._go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=np.repeat(vals[idx], _HEX_FACES.shape[0]), intensitymode="cell",
            colorscale=_plotly_cmap(cmap), opacity=opacity, flatshading=True,
            cmin=None if clim is None else clim[0], cmax=None if clim is None else clim[1],
            colorbar=dict(title=name), name=name))

    def show(self, **kw):
        return self.fig.show(**kw)

    def to_html(self, path):
        self.fig.write_html(path, include_plotlyjs="cdn")
        return path

    def screenshot(self, path):
        try:
            self.fig.write_image(path)          # requires kaleido
        except Exception as exc:
            raise GeoBrainError(
                "Plotly static-image export needs `kaleido` (pip install kaleido). "
                "Use to_html() for an interactive figure, or "
                "Scene3D(backend='pyvista').screenshot(...) for a raster.",
                object_name="Scene3D", field="screenshot",
            ) from exc
        return path


def _plotly_cmap(cmap: str) -> str:
    """Map common matplotlib / GeoBrain colormap names to a Plotly colorscale."""
    table = {"viridis": "Viridis", "plasma": "Plasma", "magma": "Magma",
             "cividis": "Cividis", "jet": "Jet", "seismic": "RdBu",
             "geo_velocity": "Turbo", "geo_resistivity": "Jet",
             "geo_seismic": "RdBu", "geo_density": "Cividis", "geo_porosity": "YlGnBu"}
    return table.get(cmap, "Viridis")


# =========================================================================
# Unified facade
# =========================================================================

class Scene3D:
    """Backend-agnostic interactive 3-D scene (``backend='pyvista'`` or ``'plotly'``).

    Every ``add_*`` method returns ``self`` for fluent chaining. A scalar field is
    given on a 3-D :class:`~geobrain.mesh.TensorMesh` (``mesh=``) or as a bare
    ``(nz, nx, ny)`` array with ``spacing`` / ``origin``.

    Args:
        backend: ``'pyvista'`` (interactive) or ``'plotly'``.
        title: window / figure title.
        background: background colour.
        window_size: ``(width, height)`` pixels.
        off_screen: render without opening a window (for saving).
    """

    def __init__(
        self,
        *,
        backend: str = "pyvista",
        title: Optional[str] = None,
        background: str = "white",
        window_size: tuple = (1024, 768),
        off_screen: bool = False,
    ):
        if backend not in _BACKENDS:
            raise GeoBrainError(
                "backend must be 'pyvista' or 'plotly'",
                object_name="Scene3D", field="backend",
                expected=str(_BACKENDS), actual=backend,
            )
        self.backend = backend
        impl = _PyVistaBackend if backend == "pyvista" else _PlotlyBackend
        self._b = impl(title=title, background=background,
                       window_size=window_size, off_screen=off_screen)

    def add_volume(self, field, *, mesh=None, spacing=(1, 1, 1), origin=(0, 0, 0),
                   cmap="viridis", opacity="linear", clim=None, name="field") -> "Scene3D":
        """Volume-render a scalar field."""
        self._b.add_volume(field, mesh=mesh, spacing=spacing, origin=origin,
                           cmap=cmap, opacity=opacity, clim=clim, name=name)
        return self

    def add_slices(self, field, *, mesh=None, spacing=(1, 1, 1), origin=(0, 0, 0),
                   x=None, y=None, z=None, cmap="viridis", clim=None, name="field") -> "Scene3D":
        """Orthogonal slices through the volume (default: one mid-plane per axis)."""
        self._b.add_slices(field, mesh=mesh, spacing=spacing, origin=origin,
                          cmap=cmap, clim=clim, name=name, x=x, y=y, z=z)
        return self

    def add_isosurface(self, field, levels, *, mesh=None, spacing=(1, 1, 1),
                       origin=(0, 0, 0), cmap="viridis", clim=None, name="field",
                       opacity=1.0) -> "Scene3D":
        """Iso-contour surfaces at the given ``levels``."""
        self._b.add_isosurface(field, levels, mesh=mesh, spacing=spacing, origin=origin,
                              cmap=cmap, clim=clim, name=name, opacity=opacity)
        return self

    def add_threshold(self, field, *, mesh=None, spacing=(1, 1, 1), origin=(0, 0, 0),
                      value=None, invert=False, cmap="viridis", clim=None, name="field") -> "Scene3D":
        """Keep cells past a threshold. PyVista ``value=None`` gives an interactive
        widget; Plotly renders a static threshold cloud."""
        self._b.add_threshold(field, mesh=mesh, spacing=spacing, origin=origin,
                             cmap=cmap, clim=clim, name=name, value=value, invert=invert)
        return self

    def add_points(self, coords, *, scalars=None, size=8, cmap="viridis", color=None,
                   label=None, render_spheres=True) -> "Scene3D":
        """Add a point cloud (stations / receivers / sources)."""
        self._b.add_points(coords, scalars=scalars, size=size, cmap=cmap, color=color,
                          label=label, render_spheres=render_spheres)
        return self

    def add_wells(self, paths, *, radius=None, color="black", labels=None) -> "Scene3D":
        """Add one or more well / borehole paths (``(n, 3)`` array or a list of them)."""
        paths = _as_path_list(paths)
        self._b.add_wells(paths, radius=radius, color=color, labels=labels)
        return self

    def add_octree(self, mesh, field, *, cmap="viridis", clim=None, name="field",
                   opacity=1.0, show_edges=True, threshold=None) -> "Scene3D":
        """Render an :class:`~geobrain.mesh.OctreeMesh` as colored hexahedra.

        ``field`` is one value per octree cell (``n_cells``). ``show_edges``
        outlines the variable-resolution cell structure (PyVista). ``threshold``
        keeps only cells whose value is ``>=`` it, handy for isolating the
        refined region.
        """
        self._b.add_octree(mesh, field, cmap=cmap, clim=clim, name=name,
                           opacity=opacity, show_edges=show_edges, threshold=threshold)
        return self

    def show(self, *, jupyter_backend=None, **kw):
        """Open the interactive viewer.

        In a Jupyter notebook the PyVista backend renders **inline via trame**
        (override with ``jupyter_backend`` = ``'trame'`` | ``'client'`` |
        ``'static'`` | ``'html'``); on the desktop it opens a window. The Plotly
        backend renders inline in the notebook natively.
        """
        if self.backend == "pyvista":
            jb = jupyter_backend or ("trame" if _in_jupyter() else None)
            if jb is not None:
                kw.setdefault("jupyter_backend", jb)
        return self._b.show(**kw)

    def _repr_html_(self):
        """Inline notebook display (Plotly only; PyVista uses ``.show()`` via trame)."""
        if self.backend == "plotly":
            return self._b.fig.to_html(include_plotlyjs="cdn", full_html=False)
        return None

    def to_html(self, path: str) -> str:
        """Export a self-contained interactive 3-D HTML file (shareable in a browser)."""
        return self._b.to_html(path)

    def screenshot(self, path: str) -> str:
        """Save a static image (PyVista off-screen render / Plotly kaleido)."""
        return self._b.screenshot(path)


def _as_path_list(paths) -> list:
    """Normalise to a list of ``(n, 3)`` arrays (a single path or several)."""
    if isinstance(paths, np.ndarray):
        return ([np.asarray(paths, dtype=np.float64)] if paths.ndim == 2
                else [np.asarray(p, dtype=np.float64) for p in paths])
    if len(paths) and np.ndim(paths[0]) == 2:           # list of (n, 3) paths
        return [np.asarray(p, dtype=np.float64) for p in paths]
    return [np.asarray(paths, dtype=np.float64)]         # one path as list of points


# =========================================================================
# One-line convenience views
# =========================================================================

def view_volume(field, *, backend="pyvista", mesh=None, **kw) -> Scene3D:
    """One-liner: volume-render ``field``. Returns the :class:`Scene3D`.

    Args:
        field: 3-D scalar cube.
        backend: see :class:`Scene3D`.
        mesh: optional mesh supplying geometry.
        **kw: forwarded to the backend add call.
    """
    return Scene3D(backend=backend).add_volume(field, mesh=mesh, **kw)


def view_slices(field, *, backend="pyvista", mesh=None, **kw) -> Scene3D:
    """One-liner: orthogonal slices through ``field``.

    Args:
        field: 3-D scalar cube.
        backend: see :class:`Scene3D`.
        mesh: optional mesh supplying geometry.
        **kw: forwarded to the backend add call.
    """
    return Scene3D(backend=backend).add_slices(field, mesh=mesh, **kw)


def view_isosurface(field, levels, *, backend="pyvista", mesh=None, **kw) -> Scene3D:
    """One-liner: iso-surfaces of ``field`` at ``levels``.

    Args:
        field: 3-D scalar cube.
        levels: isosurface value(s) to extract.
        backend: see :class:`Scene3D`.
        mesh: optional mesh supplying geometry.
        **kw: forwarded to the backend add call.
    """
    return Scene3D(backend=backend).add_isosurface(field, levels, mesh=mesh, **kw)


def view_points(coords, *, backend="pyvista", **kw) -> Scene3D:
    """One-liner: a 3-D point cloud.

    Args:
        coords: ``(n, 3)`` point coordinates.
        backend: see :class:`Scene3D`.
        **kw: forwarded to the backend add call.
    """
    return Scene3D(backend=backend).add_points(coords, **kw)


def view_octree(mesh, field, *, backend="pyvista", **kw) -> Scene3D:
    """One-liner: render an OctreeMesh's cells as colored hexahedra.

    Args:
        mesh: the :class:`~geobrain.mesh.OctreeMesh`.
        field: per-leaf values.
        backend: see :class:`Scene3D`.
        **kw: forwarded to the backend add call.
    """
    return Scene3D(backend=backend).add_octree(mesh, field, **kw)


# =========================================================================
# Domain-specific one-liners
# =========================================================================

def _scalar(v) -> float:
    return float(np.mean(np.asarray(to_numpy(v), dtype=np.float64)))


def _flow_grid_geometry(grid):
    """``(shape (nz,nx,ny), spacing (dz,dx,dy), origin (oz,ox,oy))`` from a flow
    CartGrid: z-first (axis 1 = x, axis 2 = y), matching the field order and
    :func:`_node_arrays`."""
    shape = (int(grid.nz), int(grid.nx), int(grid.ny))
    spacing = (_scalar(grid.dz_m), _scalar(grid.dx_m), _scalar(grid.dy_m))
    origin = (float(getattr(grid, "oz", 0.0)), float(getattr(grid, "ox", 0.0)),
              float(getattr(grid, "oy", 0.0)))
    return shape, spacing, origin


def _flow_flat_to_zxy(flat, shape):
    """Reshape an x-fastest flow CartGrid per-cell vector to ``(nz, nx, ny)``.

    A flow flat vector is x-fastest (``ix + iy*nx + iz*nx*ny``), i.e. the C-order
    flatten of ``(nz, ny, nx)``. Read it back that way, then swap axes 1<->2 so
    physical x lands on axis-1 as the platform convention requires."""
    nz, nx, ny = shape
    return np.asarray(to_numpy(flat), dtype=np.float64).reshape(
        (nz, ny, nx)).transpose(0, 2, 1)


def view_reservoir(field, *, grid=None, mesh=None, spacing=(1, 1, 1), origin=(0, 0, 0),
                   wells=None, style="slices", cmap="viridis", clim=None,
                   name="property", backend="pyvista", off_screen=False, **kw) -> Scene3D:
    """3-D reservoir state (pressure / saturation / ...) on a flow grid, with wells.

    ``field`` is a 3-D ``(nz, nx, ny)`` array, or a flat per-cell vector together
    with a flow ``grid`` (``CartGrid``, x-fastest) or a ``mesh`` (``TensorMesh``).
    A bare ``spacing`` / ``origin`` is **z-first**, ``(dz, dx, dy)`` /
    ``(oz, ox, oy)``, matching the field order. ``style`` is ``"slices"`` |
    ``"volume"`` | ``"threshold"``; ``wells`` is an ``(n, 3)`` path or a list.

    Args:
        field: 3-D property cube (or per-cell values with a grid/mesh).
        grid / mesh: geometry source (flow ``CartGrid`` or platform mesh).
        spacing / origin: geometry fallback when neither is given.
        wells: optional well trajectories to overlay.
        style: ``'slices'`` / ``'volume'`` / ``'isosurface'``.
        cmap / clim / name: appearance controls.
        backend / off_screen: see :class:`Scene3D`.
        **kw: forwarded to the backend add call.
    """
    if style not in ("slices", "volume", "threshold"):
        raise GeoBrainError("style must be 'slices', 'volume' or 'threshold'",
                            object_name="view_reservoir", field="style", actual=style)
    sc = Scene3D(backend=backend, off_screen=off_screen, title=name)
    add = getattr(sc, "add_" + style)
    if mesh is not None:
        add(field, mesh=mesh, cmap=cmap, clim=clim, name=name, **kw)
    elif grid is not None:
        shape, sp, org = _flow_grid_geometry(grid)
        fld = np.asarray(to_numpy(field), dtype=np.float64)
        if fld.ndim == 1:
            fld = _flow_flat_to_zxy(fld, shape)     # x-fastest -> (nz, nx, ny)
        add(fld, spacing=sp, origin=org, cmap=cmap, clim=clim, name=name, **kw)
    else:
        add(field, spacing=spacing, origin=origin, cmap=cmap, clim=clim, name=name, **kw)
    if wells is not None:
        sc.add_wells(wells)
    return sc


def view_survey(*, sources=None, receivers=None, model=None, mesh=None,
                spacing=(1, 1, 1), origin=(0, 0, 0), wells=None, model_style="slices",
                cmap="viridis", clim=None, name="model", backend="pyvista",
                off_screen=False, **kw) -> Scene3D:
    """Acquisition geometry: source + receiver clouds over an optional model volume.

    ``sources`` / ``receivers`` are ``(n, 3)`` coordinate arrays; ``model`` (with a
    ``mesh`` or ``spacing`` / ``origin``) is drawn faintly underneath via
    ``model_style`` (``"slices"`` | ``"volume"``). ``wells`` adds borehole paths.

    Args:
        sources / receivers: station coordinate tables to draw.
        model: optional background model cube.
        mesh: optional mesh supplying geometry.
        spacing / origin: geometry fallback.
        wells: optional well trajectories.
        model_style: how the background model is rendered.
        cmap / clim / name: appearance controls.
        backend / off_screen: see :class:`Scene3D`.
        **kw: forwarded to the backend add call.
    """
    sc = Scene3D(backend=backend, off_screen=off_screen, title="survey geometry")
    if model is not None:
        getattr(sc, "add_" + model_style)(model, mesh=mesh, spacing=spacing, origin=origin,
                                          cmap=cmap, clim=clim, name=name, **kw)
    if receivers is not None:
        sc.add_points(receivers, color="dodgerblue", size=8, label="receivers")
    if sources is not None:
        sc.add_points(sources, color="red", size=14, label="sources")
    if wells is not None:
        sc.add_wells(wells)
    return sc


def view_geomodel(field, *, mesh=None, spacing=(1, 1, 1), origin=(0, 0, 0),
                  categorical=False, levels=None, cmap="viridis", name="geomodel",
                  opacity=0.6, backend="pyvista", off_screen=False, **kw) -> Scene3D:
    """3-D geological model: category bodies as iso-surfaces, or a continuous volume.

    ``categorical`` (or explicit ``levels``) renders iso-surfaces at the category
    boundaries (the geological-body interfaces); otherwise the continuous property
    is volume-rendered. ``field`` is ``(nz, nx, ny)``; a bare ``spacing`` /
    ``origin`` is z-first ``(dz, dx, dy)`` / ``(oz, ox, oy)``.

    Args:
        field: 3-D property cube (or per-cell values with ``mesh``).
        mesh: optional mesh supplying geometry.
        spacing / origin: cell sizes and origin when no mesh is given.
        categorical: render integer categories with a discrete map.
        levels: isosurface levels when style calls for them.
        cmap / opacity / name: appearance controls.
        backend / off_screen: see :class:`Scene3D`.
        **kw: forwarded to the backend add call.
    """
    arr = np.asarray(to_numpy(field), dtype=np.float64)
    sc = Scene3D(backend=backend, off_screen=off_screen, title=name)
    if categorical or levels is not None:
        if levels is None:
            cats = np.unique(np.round(arr))
            levels = [0.5 * (a + b) for a, b in zip(cats[:-1], cats[1:])] or [float(arr.mean())]
        sc.add_isosurface(field, levels, mesh=mesh, spacing=spacing, origin=origin,
                          cmap=cmap, name=name, opacity=opacity, **kw)
    else:
        sc.add_volume(field, mesh=mesh, spacing=spacing, origin=origin,
                      cmap=cmap, name=name, **kw)
    return sc


# =========================================================================
# 4-D time-lapse animation (flow time series / inversion iterations)
# =========================================================================

class _TimeLapse:
    """A sequence of 3-D frames (constant geometry) rendered as an animation.

    Returned by :func:`view_timelapse`. Export a shareable GIF/MP4 (PyVista,
    off-screen) or an interactive HTML with a play button + time slider (Plotly).
    """

    def __init__(self, frames, xn, yn, zn, *, style, levels, cmap, clim, name, wells, times):
        self.frames, self.xn, self.yn, self.zn = frames, xn, yn, zn
        self.style, self.levels, self.cmap, self.clim = style, levels, cmap, clim
        self.name, self.wells, self.times = name, wells, times
        # One fixed default iso-level from the global clim, shared by BOTH backends
        # so .to_gif() (PyVista) and .to_html() (Plotly) contour the same surface
        # every frame (a per-frame np.mean would drift). None when levels are given.
        self.iso_level = (
            None if levels else 0.5 * (clim[0] + clim[1]) if clim is not None else None
        )

    def _iso_levels(self, v=None):
        """Iso-surface contour levels for one frame: explicit ``levels`` if given,
        else the single fixed clim-midpoint default (identical across frames /
        backends), falling back to this frame's mean only when no clim is known."""
        if self.levels:
            return list(self.levels)
        if self.iso_level is not None:
            return [self.iso_level]
        return [float(np.mean(v))] if v is not None else [0.0]

    # -- PyVista GIF / MP4 -------------------------------------------------
    def _pv_add(self, pl, pv, f3d):
        grid = _pv_rectilinear(pv, f3d, self.xn, self.yn, self.zn, self.name)
        if self.style == "volume":
            pl.add_volume(grid.cell_data_to_point_data(), scalars=self.name,
                          cmap=self.cmap, clim=self.clim)
        elif self.style == "isosurface":
            lv = self._iso_levels()
            c = grid.cell_data_to_point_data().contour(isosurfaces=lv, scalars=self.name)
            if c.n_points > 0:                          # a frame may not cross the level
                pl.add_mesh(c, scalars=self.name, cmap=self.cmap, clim=self.clim)
        else:
            pl.add_mesh(grid.slice_orthogonal(), scalars=self.name, cmap=self.cmap, clim=self.clim)
        for w in (self.wells or []):
            pl.add_mesh(pv.Spline(w, max(2, w.shape[0])).tube(), color="black")

    def _movie(self, path, *, fps, movie):
        try:
            import pyvista as pv
        except ImportError as exc:  # pragma: no cover
            raise GeoBrainError("time-lapse GIF/MP4 needs pyvista (pip install pyvista).",
                                object_name="view_timelapse", field="to_gif") from exc
        pl = pv.Plotter(off_screen=True, window_size=(1024, 768))
        pl.set_background("white")
        pl.open_movie(path, framerate=fps) if movie else pl.open_gif(path, fps=fps)
        for t, f3d in enumerate(self.frames):
            pl.clear()
            self._pv_add(pl, pv, f3d)
            pl.add_text(f"{self.name}   t = {self.times[t]}", font_size=11, color="black")
            pl.camera_position = "iso"
            pl.write_frame()
        pl.close()
        return path

    def to_gif(self, path, *, fps: int = 5) -> str:
        """Write an animated GIF (PyVista, off-screen)."""
        return self._movie(path, fps=fps, movie=False)

    def to_mp4(self, path, *, fps: int = 10) -> str:
        """Write an MP4 movie (PyVista; needs ``imageio[ffmpeg]``)."""
        return self._movie(path, fps=fps, movie=True)

    # -- Plotly slider HTML ------------------------------------------------
    def _plotly_fig(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:  # pragma: no cover
            raise GeoBrainError("time-lapse HTML needs plotly (pip install plotly).",
                                object_name="view_timelapse", field="to_html") from exc

        def tr(f3d):
            # For isosurface frames, pass the single shared fixed level so Plotly
            # contours the SAME surface as PyVista's _pv_add (no per-frame drift).
            lv = self._iso_levels() if self.style == "isosurface" else self.levels
            return _plotly_field_trace(go, f3d, self.xn, self.yn, self.zn, style=self.style,
                                       levels=lv, cmap=self.cmap, clim=self.clim, name=self.name)
        labels = [str(self.times[t]) for t in range(len(self.frames))]
        fig = go.Figure(data=[tr(self.frames[0])])
        fig.frames = [go.Frame(data=[tr(self.frames[t])], name=labels[t])
                      for t in range(len(self.frames))]
        steps = [dict(method="animate", label=labels[t],
                      args=[[labels[t]], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                 for t in range(len(self.frames))]
        fig.update_layout(
            title=self.name, template="plotly_white", scene=dict(aspectmode="data"),
            sliders=[dict(active=0, currentvalue=dict(prefix="t = "), steps=steps)],
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.02, buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])])
        return fig

    def to_html(self, path) -> str:
        """Write an interactive HTML with a play button + time slider (Plotly)."""
        self._plotly_fig().write_html(path, include_plotlyjs="cdn")
        return path

    def show(self):
        """Open the interactive (Plotly) time-slider figure (inline in Jupyter)."""
        return self._plotly_fig().show()

    def _repr_html_(self):
        """Inline notebook display: the Plotly play/slider animation."""
        return self._plotly_fig().to_html(include_plotlyjs="cdn", full_html=False)


def view_timelapse(fields, *, mesh=None, grid=None, spacing=(1, 1, 1), origin=(0, 0, 0),
                   style="slices", levels=None, cmap="viridis", clim=None,
                   name="field", wells=None, times=None) -> _TimeLapse:
    """4-D time-lapse of a 3-D field sequence (flow time series / inversion iterations).

    ``fields`` is ``(T, nz, nx, ny)``, a list of 3-D arrays, or ``(T, n_cells)`` with a
    flow ``grid`` / ``mesh``. A bare ``spacing`` / ``origin`` is z-first
    ``(dz, dx, dy)`` / ``(oz, ox, oy)``, matching the field order. ``style`` is
    ``"slices"`` | ``"volume"`` | ``"isosurface"`` (an iso-surface vividly tracks an
    advancing front). The colour scale is fixed across frames (global ``clim``
    unless given). Returns a :class:`_TimeLapse`; call ``.to_gif(...)`` /
    ``.to_mp4(...)`` / ``.to_html(...)`` / ``.show()``.

    Args:
        fields: sequence of 3-D cubes, one per vintage.
        mesh / grid: geometry source.
        spacing / origin: geometry fallback.
        style / levels / cmap / clim / name: appearance controls.
        wells: optional well trajectories.
        times: per-vintage labels for the slider.
    """
    seq = list(fields)
    if grid is not None:
        shape, sp, org = _flow_grid_geometry(grid)
        xn, yn, zn = _node_arrays(None, sp, org, shape)
        frames = [_flow_flat_to_zxy(f, shape) for f in seq]   # x-fastest -> (nz,nx,ny)
    else:
        f0, xn, yn, zn = _resolve_field(seq[0], mesh, spacing, origin)
        frames = [np.asarray(to_numpy(f), dtype=np.float64).reshape(f0.shape) for f in seq]
    if clim is None:
        allv = np.concatenate([f.ravel() for f in frames])
        clim = (float(allv.min()), float(allv.max()))
    times = list(range(len(frames))) if times is None else list(times)
    wells = None if wells is None else _as_path_list(wells)
    return _TimeLapse(frames, xn, yn, zn, style=style, levels=levels, cmap=cmap,
                      clim=clim, name=name, wells=wells, times=times)
