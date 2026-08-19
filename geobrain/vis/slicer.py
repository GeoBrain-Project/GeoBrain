"""
Three-pane orthogonal slice viewer for 3D scalar volumes.

GeoBrain-native three-plane slice viewer for tensor meshes.
Tailored to the package conventions:

- Volumes are indexed ``(nz, nx, ny)``: Z fastest as the first axis,
  then X (axis 1), then Y (axis 2); matches what :class:`MT3D`,
  :class:`FDEM3D`, :class:`Gravity3D`, etc. consume on the ``sigma`` /
  ``rho`` parameter.
- Accepts ``torch.Tensor`` *or* ``numpy.ndarray`` (the usual
  ``to_numpy`` round-trip).
- Reads non-uniform cell widths straight off a :class:`TensorMesh`
  (``mesh.cell_widths``); also accepts a raw
  ``(wz, wx, wy)`` tuple, or a uniform ``(dz, dx, dy)`` spacing.
- Layout: 2×2 with XY plan view + YZ section + XZ section + colorbar.
  Green crosshairs on each panel mark where the *other* two slices cut
  it, so you can see how the three planes meet.
- The static figure works under ``MPLBACKEND=Agg`` (headless figure
  export); pass ``interactive=True`` to attach
  ``matplotlib.widgets.Slider`` controls when running with a GUI
  backend.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from geobrain.vis._utils import symmetric_clim, to_numpy

try:
    from matplotlib.widgets import Slider as _MplSlider
except ImportError:  # pragma: no cover, matplotlib always ships widgets
    _MplSlider = None


def _resolve_edges(
    data_shape: tuple[int, int, int],
    *,
    mesh,
    cell_widths,
    spacing,
    origin: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(z_edges, y_edges, x_edges)`` (each length n+1) for the
    three axes, given exactly one of ``mesh`` / ``cell_widths`` /
    ``spacing`` (or none → unit spacing)."""
    nz, nx, ny = data_shape
    n_given = sum(x is not None for x in (mesh, cell_widths, spacing))
    if n_given > 1:
        raise GeoBrainError(
            "pass at most one of mesh / cell_widths / spacing "
            f"(received {n_given})"
        )

    mesh_origin = None
    if mesh is not None:
        cw = getattr(mesh, "cell_widths", None)
        if cw is None or len(cw) != 3:
            raise GeoBrainError(
                "mesh must be a 3-D TensorMesh exposing cell_widths "
                f"(got {type(mesh).__name__} with "
                f"{0 if cw is None else len(cw)} axes)"
            )
        wz, wx, wy = (to_numpy(w) for w in cw)
        # Honour the mesh's own coordinate origin (defaults to zeros, so an
        # origin-0 mesh is byte-identical). Added to any explicit ``origin`` arg
        # below so a shifted mesh's slices land at the right physical position.
        mo = getattr(mesh, "origin", None)
        if mo is not None and len(mo) == 3:
            mesh_origin = (float(mo[0]), float(mo[1]), float(mo[2]))
    elif cell_widths is not None:
        if len(cell_widths) != 3:
            raise GeoBrainError(
                "cell_widths must be a 3-tuple (wz, wx, wy); "
                f"got length {len(cell_widths)}"
            )
        wz, wx, wy = (to_numpy(w) for w in cell_widths)
    elif spacing is not None:
        if len(spacing) != 3:
            raise GeoBrainError(
                "spacing must be (dz, dx, dy); "
                f"got length {len(spacing)}"
            )
        dz, dx, dy = (float(s) for s in spacing)
        wz = np.full(nz, dz)
        wx = np.full(nx, dx)
        wy = np.full(ny, dy)
    else:
        wz = np.ones(nz)
        wx = np.ones(nx)
        wy = np.ones(ny)

    if wz.size != nz or wx.size != nx or wy.size != ny:
        raise GeoBrainError(
            f"cell_widths size mismatch: data shape ({nz}, {nx}, {ny}) "
            f"vs widths sizes ({wz.size}, {wx.size}, {wy.size})"
        )

    if len(origin) != 3:
        raise GeoBrainError(
            f"origin must be (oz, ox, oy); got length {len(origin)}"
        )
    oz, ox, oy = (float(o) for o in origin)
    if mesh_origin is not None:
        oz += mesh_origin[0]
        ox += mesh_origin[1]
        oy += mesh_origin[2]
    z_edges = np.concatenate([[oz], oz + np.cumsum(wz)])
    y_edges = np.concatenate([[oy], oy + np.cumsum(wy)])
    x_edges = np.concatenate([[ox], ox + np.cumsum(wx)])
    return z_edges, y_edges, x_edges


def _sign_changing(data: np.ndarray) -> bool:
    """True when the finite data straddles zero (min < 0 < max)."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    return bool(finite.min() < 0.0 < finite.max())


def _auto_cmap(data: np.ndarray, user_cmap: str | None) -> str:
    """
    Default colormap: ``seismic`` for sign-changing data,
    ``viridis`` otherwise. Honours the caller's explicit choice."""
    if user_cmap is not None:
        return user_cmap
    return "seismic" if _sign_changing(data) else "viridis"


class Slicer:
    """
    Three-pane orthogonal slice view of a 3-D scalar volume.

    Created by :func:`plot_3d_slicer`. Holds references to the figure,
    the three image artists, and the crosshair line artists, so the
    caller can later call :meth:`update` to redraw at different slice
    positions without rebuilding the figure.

    Attributes:
        fig: The figure containing all four panels.
        ax_xy, ax_yz, ax_xz, ax_cb: The four axes, XY plan view, YZ section, XZ section, colorbar.
        ix, iy, iz: Current slice indices into ``data`` along the X, Y, Z axes.
        data: The 3-D volume being displayed (shape ``(nz, nx, ny)``).
    """

    def __init__(
        self,
        data_3d: np.ndarray,
        z_edges: np.ndarray,
        y_edges: np.ndarray,
        x_edges: np.ndarray,
        *,
        ix: int,
        iy: int,
        iz: int,
        cmap: str,
        clim: tuple[float, float] | None,
        label: str | None,
        figsize: tuple[float, float],
    ) -> None:
        self.data = data_3d
        self.z_edges = z_edges
        self.y_edges = y_edges
        self.x_edges = x_edges
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.cmap = cmap
        if clim is None:
            finite = data_3d[np.isfinite(data_3d)]
            self.clim = (float(finite.min()), float(finite.max())) \
                if finite.size else (0.0, 1.0)
        else:
            self.clim = (float(clim[0]), float(clim[1]))
        self.label = label

        self.fig, axes = plt.subplots(
            2, 2, figsize=figsize, constrained_layout=True,
        )
        self.ax_xy = axes[0, 0]
        self.ax_yz = axes[0, 1]
        self.ax_xz = axes[1, 0]
        self.ax_cb = axes[1, 1]
        self.ax_cb.set_axis_off()
        self._images: dict = {}
        self._lines: dict[str, dict[str, Line2D]] = {}
        # Slider handles attached by ``_attach_sliders`` when
        # ``interactive=True``; ``Any`` so the same slot can hold
        # ``None`` (no sliders) or a ``matplotlib.widgets.Slider``.
        self.slider_ix: object = None
        self.slider_iy: object = None
        self.slider_iz: object = None
        self._draw_initial()

    # ------------------------------------------------------------------ #
    # cell-center helpers
    # ------------------------------------------------------------------ #
    def _zc(self, iz: int) -> float:
        return 0.5 * (self.z_edges[iz] + self.z_edges[iz + 1])

    def _yc(self, iy: int) -> float:
        return 0.5 * (self.y_edges[iy] + self.y_edges[iy + 1])

    def _xc(self, ix: int) -> float:
        return 0.5 * (self.x_edges[ix] + self.x_edges[ix + 1])

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
    def _draw_initial(self) -> None:
        vmin, vmax = self.clim

        # XY plan view at depth iz: data[iz, :, :] has shape (nx, ny); the
        # pcolormesh C-array must be (len(y_edges)-1, len(x_edges)-1) = (ny, nx),
        # so transpose axis-1 (x) and axis-2 (y) to put X on the horizontal axis.
        ax = self.ax_xy
        im_xy = ax.pcolormesh(
            self.x_edges, self.y_edges, self.data[self.iz, :, :].T,
            cmap=self.cmap, vmin=vmin, vmax=vmax, shading="flat",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.set_title(f"XY @ z = {self._zc(self.iz):g}")
        v = ax.axvline(self._xc(self.ix), color="lime", lw=1.0)
        h = ax.axhline(self._yc(self.iy), color="lime", lw=1.0)
        self._images["xy"] = im_xy
        self._lines["xy"] = {"v": v, "h": h}

        # YZ section holds X constant. X is axis-1, so data[:, ix, :] is (nz, ny).
        ax = self.ax_yz
        im_yz = ax.pcolormesh(
            self.y_edges, self.z_edges, self.data[:, self.ix, :],
            cmap=self.cmap, vmin=vmin, vmax=vmax, shading="flat",
        )
        ax.set_xlabel("Y")
        ax.set_ylabel("Z (depth)")
        ax.set_aspect("auto")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(f"YZ @ x = {self._xc(self.ix):g}")
        v = ax.axvline(self._yc(self.iy), color="lime", lw=1.0)
        h = ax.axhline(self._zc(self.iz), color="lime", lw=1.0)
        self._images["yz"] = im_yz
        self._lines["yz"] = {"v": v, "h": h}

        # XZ section holds Y constant. Y is axis-2, so data[:, :, iy] is (nz, nx).
        ax = self.ax_xz
        im_xz = ax.pcolormesh(
            self.x_edges, self.z_edges, self.data[:, :, self.iy],
            cmap=self.cmap, vmin=vmin, vmax=vmax, shading="flat",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Z (depth)")
        ax.set_aspect("auto")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(f"XZ @ y = {self._yc(self.iy):g}")
        v = ax.axvline(self._xc(self.ix), color="lime", lw=1.0)
        h = ax.axhline(self._zc(self.iz), color="lime", lw=1.0)
        self._images["xz"] = im_xz
        self._lines["xz"] = {"v": v, "h": h}

        # Colorbar fills the 4th panel.
        self.fig.colorbar(
            im_xy, ax=self.ax_cb, fraction=0.5, shrink=0.85,
            label=self.label or "",
        )

    # ------------------------------------------------------------------ #
    # public update
    # ------------------------------------------------------------------ #
    def update(
        self,
        *,
        ix: int | None = None,
        iy: int | None = None,
        iz: int | None = None,
    ) -> None:
        """
        Redraw the slices at new indices.

        Out-of-range indices are clipped to the valid range with no
        error. Pass any subset of ``ix``, ``iy``, ``iz``; unspecified
        indices stay at their current value.
        """
        nz, nx, ny = self.data.shape
        if ix is not None:
            self.ix = int(np.clip(ix, 0, nx - 1))
        if iy is not None:
            self.iy = int(np.clip(iy, 0, ny - 1))
        if iz is not None:
            self.iz = int(np.clip(iz, 0, nz - 1))

        # Update image arrays (ravel for QuadMesh.set_array on shading='flat').
        # Mirror _draw_initial exactly: XY transposes (nx, ny) -> (ny, nx);
        # YZ holds X=axis-1; XZ holds Y=axis-2.
        self._images["xy"].set_array(self.data[self.iz, :, :].T.ravel())
        self._images["yz"].set_array(self.data[:, self.ix, :].ravel())
        self._images["xz"].set_array(self.data[:, :, self.iy].ravel())

        self.ax_xy.set_title(f"XY @ z = {self._zc(self.iz):g}")
        self.ax_yz.set_title(f"YZ @ x = {self._xc(self.ix):g}")
        self.ax_xz.set_title(f"XZ @ y = {self._yc(self.iy):g}")

        self._lines["xy"]["v"].set_xdata([self._xc(self.ix)])
        self._lines["xy"]["h"].set_ydata([self._yc(self.iy)])
        self._lines["yz"]["v"].set_xdata([self._yc(self.iy)])
        self._lines["yz"]["h"].set_ydata([self._zc(self.iz)])
        self._lines["xz"]["v"].set_xdata([self._xc(self.ix)])
        self._lines["xz"]["h"].set_ydata([self._zc(self.iz)])

        self.fig.canvas.draw_idle()

    @property
    def indices(self) -> tuple[int, int, int]:
        """Current ``(ix, iy, iz)`` slice indices."""
        return self.ix, self.iy, self.iz


def _attach_sliders(sl: Slicer) -> None:
    """Wire matplotlib sliders to a Slicer for interactive dragging."""
    if _MplSlider is None:  # pragma: no cover
        warnings.warn(
            "matplotlib.widgets.Slider not available, "
            "interactive=True ignored.",
            UserWarning,
            stacklevel=3,
        )
        return
    nz, nx, ny = sl.data.shape
    # constrained_layout is incompatible with manually-placed slider axes;
    # turn it off before adjusting the bottom margin.
    sl.fig.set_layout_engine(None)
    sl.fig.subplots_adjust(bottom=0.22)
    ax_ix = sl.fig.add_axes((0.15, 0.10, 0.7, 0.02))
    ax_iy = sl.fig.add_axes((0.15, 0.06, 0.7, 0.02))
    ax_iz = sl.fig.add_axes((0.15, 0.02, 0.7, 0.02))
    sl.slider_ix = _MplSlider(
        ax_ix, "ix", 0, nx - 1, valinit=sl.ix, valstep=1,
    )
    sl.slider_iy = _MplSlider(
        ax_iy, "iy", 0, ny - 1, valinit=sl.iy, valstep=1,
    )
    sl.slider_iz = _MplSlider(
        ax_iz, "iz", 0, nz - 1, valinit=sl.iz, valstep=1,
    )
    sl.slider_ix.on_changed(lambda v: sl.update(ix=int(v)))
    sl.slider_iy.on_changed(lambda v: sl.update(iy=int(v)))
    sl.slider_iz.on_changed(lambda v: sl.update(iz=int(v)))


def plot_3d_slicer(
    data_3d,
    *,
    mesh=None,
    cell_widths: Sequence | None = None,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    ix: int | None = None,
    iy: int | None = None,
    iz: int | None = None,
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    label: str | None = None,
    figsize: tuple[float, float] = (11, 9),
    interactive: bool = False,
) -> Slicer:
    """
    Three-pane orthogonal slice view of a 3-D scalar volume.

    Args:
        data_3d: Volume of shape ``(nz, nx, ny)``, Z-first (axis 1 = x,
            axis 2 = y), matching the ``sigma`` / ``rho`` model conventions.
        mesh: Geobrain mesh; ``mesh.cell_widths`` supplies the three axis
            spacings (supports non-uniform meshes). Mutually exclusive with
            ``cell_widths`` and ``spacing``.
        cell_widths: Per-axis 1-D arrays / tensors of cell widths. Mutually exclusive
            with ``mesh`` and ``spacing``.
        spacing: Uniform spacing. Mutually exclusive with ``mesh`` and
            ``cell_widths``.
        origin: Real-world origin of the volume.
        ix, iy, iz: Initial slice indices. Default = middle of each axis.
        cmap: Matplotlib colormap. Default: ``"seismic"`` for sign-changing
            data, ``"viridis"`` otherwise.
        clim: Color limits. Default = data min / max, except for
            sign-changing data (diverging ``seismic`` cmap), where the
            default is symmetric ``(-vmax, vmax)`` so value 0 maps to the
            colormap's white pivot.
        label: Colorbar label.
        figsize: Figure size in inches.
        interactive: When ``True``, add ``matplotlib.widgets.Slider`` controls for
            ix / iy / iz. Requires a GUI matplotlib backend; under
            ``Agg`` the sliders are added but inert.

    Returns:
        Live handle: ``slicer.fig`` is the figure, ``slicer.update(...)``
        redraws at new indices, ``slicer.indices`` gives current state.

    Examples:
    Static figure for save / Agg backend::

        from geobrain.vis import plot_3d_slicer
        sl = plot_3d_slicer(
            sigma_recovered,
            mesh=mesh,
            label="σ (S/m)",
            ix=10, iy=12, iz=4,
        )
        sl.fig.savefig("recovered_sigma.png")

    Interactive (GUI backend)::

        sl = plot_3d_slicer(rho, mesh=mesh, interactive=True)
        # ... drag the sliders ...
    """
    arr = to_numpy(data_3d)
    if arr.ndim != 3:
        raise GeoBrainError(
            f"data_3d must be 3-D, got shape {tuple(arr.shape)} "
            f"(ndim={arr.ndim})"
        )
    nz, nx, ny = arr.shape
    z_edges, y_edges, x_edges = _resolve_edges(
        (nz, nx, ny), mesh=mesh, cell_widths=cell_widths,
        spacing=spacing, origin=origin,
    )

    if ix is None:
        ix = nx // 2
    if iy is None:
        iy = ny // 2
    if iz is None:
        iz = nz // 2
    ix = int(np.clip(ix, 0, nx - 1))
    iy = int(np.clip(iy, 0, ny - 1))
    iz = int(np.clip(iz, 0, nz - 1))

    cmap_final = _auto_cmap(arr, cmap)

    # When the volume straddles zero we default to the diverging ``seismic``
    # cmap, whose white pivot sits at the midpoint of the clim. Couple the
    # default clim to that SAME sign-changing condition so the pivot lands on
    # value 0: symmetric (-vmax, vmax). The linear (viridis) non-sign-changing
    # path keeps the raw data min/max. An explicit ``clim`` always wins.
    if clim is None and _sign_changing(arr):
        clim = symmetric_clim(arr, percentile=100)

    sl = Slicer(
        arr, z_edges, y_edges, x_edges,
        ix=ix, iy=iy, iz=iz,
        cmap=cmap_final, clim=clim, label=label, figsize=figsize,
    )
    if interactive:
        _attach_sliders(sl)
    return sl


__all__ = ["plot_3d_slicer", "Slicer"]
