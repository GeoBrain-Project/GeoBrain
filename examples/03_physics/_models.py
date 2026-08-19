"""Shared earth models for the physics gallery.

Every script in this section runs on a REAL model rather than a rectangle:
either the Marmousi II benchmark, read from ``examples/data/marmousi`` and
resampled onto the working grid, or a geostatistical field drawn with the
platform's own FFT-MA simulator. Keeping both behind one small module means
the physics scripts stay about physics, and every one of them inherits the
same earth when it should.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from pathlib import Path

import torch

from geobrain.geomodel.frames import GeoGrid, PropertyMetadata
from geobrain.geomodel.geostats import (
    FFTMA,
    CovarianceModel,
    SimulationExecutionConfig,
    VariogramKernel,
)
from geobrain.io import read_segy

DATA = Path(__file__).resolve().parents[1] / "data" / "marmousi"
_SEGY = {"vp": "vp_marmousi-ii.segy", "vs": "vs_marmousi-ii.segy",
         "rho": "density_marmousi-ii.segy"}
_NATIVE_SPACING = 1.25          # Marmousi II sample interval [m], both axes

__all__ = ["marmousi", "correlated_fields", "smooth_background", "smooth"]


def smooth(field: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Separable Gaussian blur, the usual way to build an FWI starting model."""
    radius = max(1, int(round(3.0 * sigma_cells)))
    offsets = torch.arange(-radius, radius + 1, dtype=field.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel = kernel / kernel.sum()
    out = field[None, None]
    for dim, size in ((2, (2 * radius + 1, 1)), (3, (1, 2 * radius + 1))):
        pad = [0, 0, radius, radius] if dim == 2 else [radius, radius, 0, 0]
        out = torch.nn.functional.conv2d(
            torch.nn.functional.pad(out, pad, mode="replicate"),
            kernel.reshape(1, 1, *size))
    return out[0, 0].contiguous()


def _read_section(field: str) -> torch.Tensor:
    """Marmousi II as a (n_depth, n_distance) float64 tensor at 1.25 m."""
    path = DATA / _SEGY[field]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The Marmousi II sections are 148 MB each, "
            "past what a git host accepts in one file, so they are fetched "
            "rather than committed. Run:\n\n"
            "    python examples/data/fetch_marmousi.py\n")
    traces = torch.as_tensor(read_segy(str(path)).traces, dtype=torch.float64)
    # SEG-Y stores one trace per column of the section: (n_traces, n_samples)
    return traces.T.contiguous() if traces.shape[0] > traces.shape[1] else traces


def marmousi(
    nz: int,
    nx: int,
    spacing: float,
    *,
    x_start: float = 9000.0,
    z_start: float = 500.0,
    fields: tuple[str, ...] = ("vp", "vs", "rho"),
) -> dict[str, torch.Tensor]:
    """Resample a window of Marmousi II onto an ``(nz, nx)`` grid.

    Args:
        nz, nx: target grid size.
        spacing: target cell size [m], applied to both axes.
        x_start: left edge of the extracted window [m] in Marmousi
            coordinates (the full model spans 0-17 km).
        z_start: top of the window [m]; the default clips the water layer
            so the section is all sediment.
        fields: which of ``vp`` / ``vs`` / ``rho`` to return.

    Returns:
        ``{field: (nz, nx) float64 tensor}``, vp and vs in m/s, rho in
        kg/m³.
    """
    out: dict[str, torch.Tensor] = {}
    for field in fields:
        section = _read_section(field)
        z0 = int(round(z_start / _NATIVE_SPACING))
        x0 = int(round(x_start / _NATIVE_SPACING))
        z1 = z0 + int(round(nz * spacing / _NATIVE_SPACING))
        x1 = x0 + int(round(nx * spacing / _NATIVE_SPACING))
        window = section[z0:z1, x0:x1]
        if window.shape[0] < 2 or window.shape[1] < 2:
            raise ValueError(
                f"requested Marmousi window falls outside the model: "
                f"z {z_start}-{z_start + nz * spacing} m, "
                f"x {x_start}-{x_start + nx * spacing} m")
        resampled = torch.nn.functional.interpolate(
            window[None, None], size=(nz, nx), mode="bilinear",
            align_corners=False)[0, 0]
        # The benchmark ships in field units (km/s and g/cm³), and every
        # operator on this platform is strict SI, so the boundary is here
        # and it is explicit. Raw ranges: vp 1.03-4.70, vs 0-2.80, rho
        # 1.01-2.63.
        out[field] = (resampled * 1000.0).contiguous()
    return out


def correlated_fields(
    shape: tuple[int, int],
    spacing: float,
    *,
    seed: int,
    ranges: tuple[float, float] = (900.0, 220.0),
    azimuth: float = 20.0,
    correlation: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two correlated standard-normal fields on an ``(nz, nx)`` grid.

    Drawn with the platform's FFT-MA simulator on one anisotropic
    spherical variogram, then mixed to the requested correlation with
    ``z2c = r·z1 + sqrt(1 - r²)·z2``, the usual co-simulation recipe for
    a porosity/lithology pair. Map them into physical ranges yourself; a
    logistic squash keeps both bounded without clipping artefacts.
    """
    nz, nx = shape
    # The variogram ranges bind to the GeoGrid axes POSITIONALLY, and the
    # result is read back with ``reshape(nz, nx)``, so the grid must be
    # declared in that same (depth, distance) order, and ``ranges`` given as
    # (vertical, lateral). Declaring (nx, nz) here and reshaping to (nz, nx)
    # silently transposes the anisotropy, which is a wrong earth that still
    # looks plausible.
    domain = GeoGrid(shape=(nz, nx, 1), origin=(0.0, 0.0, 0.0),
                     spacing=(spacing, spacing, 1.0))
    model = CovarianceModel(nugget=0.0, structures=[
        VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=1.0,
                        ranges=(ranges[1], ranges[0], 1.0e6),
                        angles=(azimuth, 0.0, 0.0))])
    prop = PropertyMetadata(name="z", kind="continuous", unit="1")
    draws = []
    for offset in (0, 1):
        frame = FFTMA(model, property=prop,
                      execution=SimulationExecutionConfig(
                          n_realizations=1, seed=seed + offset)
                      )(None, domain).realizations[0].frame
        values = torch.as_tensor(frame.to_numpy("simulation"),
                                 dtype=torch.float64).reshape(-1)
        draws.append(values.reshape(nz, nx))
    first = draws[0]
    second = correlation * draws[0] + (1.0 - correlation ** 2) ** 0.5 * draws[1]
    return first, second


def smooth_background(nz: int, nx: int, spacing: float, *,
                      surface: float, gradient: float) -> torch.Tensor:
    """A depth-increasing background, ``surface + gradient·z``."""
    depth = torch.arange(nz, dtype=torch.float64)[:, None] * spacing
    return (surface + gradient * depth).expand(nz, nx).contiguous()
