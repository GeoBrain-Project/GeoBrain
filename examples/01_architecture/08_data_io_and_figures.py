"""Data in, pictures out: the I/O and visualization doors.

A platform is only useful if your data can reach it and your results can
leave. This script closes the loop that the rest of Part 1 assumes:

    IN    read_segy / read_las / read_ubc_model / read_edi: field formats
          land as tensors, with their sample interval and headers intact
    OUT   write_segy for synthetics, write_tensormesh_vtk for models a
          reviewer will open in ParaView, write_hdf5 for anything else
    LOOK  geobrain.vis wraps the plots you make constantly:
          plot_field_2d, plot_station_map, plot_convergence, so a
          figure is one call, not fifteen matplotlib lines

The demonstration is a full small workflow: model a 2-D section, write
the synthetic gather to SEG-Y, read it back and prove the round trip is
bit-faithful, export the model to VTK and HDF5, then draw the result with
the shipped helpers.

APIs featured:
    - geobrain.io.write_segy / read_segy (SegYData: traces, dt, meta)
    - geobrain.io.write_tensormesh_vtk (ParaView-ready models)
    - geobrain.io.write_hdf5 / read_hdf5 (arrays with attributes)
    - geobrain.vis.plot_field_2d / plot_station_map / plot_convergence

Expected runtime: < 30 s.

Outputs:
    out/08_data_io_and_figures.png: the model, the round-tripped gather,
    the station map and a convergence curve, all drawn with vis helpers.
    out/io_demo/: the SEG-Y, VTK and HDF5 files this script wrote.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _style import (
    CMAP_ANOMALY,
    CMAP_MODEL,
    C_RECOVERED,
    apply_style,
    field,
    figure,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.io import (
    read_hdf5,
    read_segy,
    write_hdf5,
    write_segy,
    write_tensormesh_vtk,
)
from geobrain.mesh import TensorMesh
from geobrain.physics.rock.models import GardnerOperator
from geobrain.physics.wave import (
    Acoustic2D,
    Seismic2DSurvey,
    ricker,
    shared_wavelet,
)
from geobrain.vis import plot_convergence, plot_field_2d, plot_station_map

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
IO_DIR = OUT / "io_demo"
IO_DIR.mkdir(parents=True, exist_ok=True)

# %% 1. Model something worth exporting ------------------------------------
NZ, NX, D = 24, 48, 25.0
mesh = TensorMesh(shape=(NZ, NX), spacing=(D, D))
zz = torch.arange(NZ, dtype=torch.float64)[:, None] * D
vp = (1900.0 + 0.8 * zz).expand(NZ, NX).clone()
vp[14:, :] += 350.0                              # a flat reflector
vp[6:12, 18:30] += 550.0                         # a fast body

NT, DT, F0 = 500, 0.002, 10.0
rcv_x = torch.linspace(25.0, 1175.0, 48, dtype=torch.float64)
survey = Seismic2DSurvey.from_positions(
    source_positions=[[600.0, 25.0]], source_shot_index=[0],
    receiver_positions=[[float(x), 25.0] for x in rcv_x],
    receiver_shot_index=[0] * len(rcv_x), nt=NT, dt=DT,
)
wavelets = shared_wavelet(ricker(NT, DT, F0, causal=True, dtype=torch.float64),
                          n_source=survey.n_source)
forward = Acoustic2D(survey, wavelets) @ GardnerOperator()
with torch.no_grad():
    packed = forward(ModelState(tensors={"vp": vp}),
                     ForwardContext.of(mesh=mesh)).data["seismic"]
gather = survey.to_dense(packed)[0][0, :, :, 0]          # (nt, n_receiver)
print(f"[1] modelled a {tuple(gather.shape)} gather (time, receiver)")

# %% 2. OUT: SEG-Y, and back again -----------------------------------------
#
# write_segy takes trace-major float arrays; read_segy returns a SegYData
# record carrying the traces, the sample interval and the raw headers.
segy_path = IO_DIR / "synthetic_shot.sgy"
write_segy(segy_path,
           np.ascontiguousarray(gather.T.numpy(), dtype=np.float32), dt=DT)
back = read_segy(segy_path)
traces = np.asarray(back.traces)
round_trip = torch.as_tensor(traces, dtype=torch.float64).T
err = float((round_trip - gather).abs().max())
print(f"[2] SEG-Y round trip: {traces.shape[0]} traces, "
      f"dt={back.dt * 1e3:.1f} ms, max |error| = {err:.2e} "
      f"({segy_path.stat().st_size / 1024:.0f} kB on disk)")

# %% 3. OUT: a model a reviewer can open -----------------------------------
# VTK is a volume format, so it wants a 3-D TensorMesh, and the exporter
# says so rather than guessing. Extrude the section into a small cube.
mesh3 = TensorMesh(shape=(NZ, NX, 12), spacing=(D, D, D))
vp3 = vp[:, :, None].expand(NZ, NX, 12).contiguous()
vtk_path = IO_DIR / "velocity_model.vtk"
write_tensormesh_vtk(str(vtk_path), mesh3,
                     cell_data={"vp": vp3.numpy(),
                                "rho": (310.0 * vp3.pow(0.25)).numpy()})
h5_path = IO_DIR / "gather.h5"
write_hdf5(h5_path, "shot_0", gather.numpy(),
           attrs={"dt_s": DT, "f0_hz": F0, "units": "Pa"})
restored = read_hdf5(h5_path, "shot_0")
print(f"[3] wrote {vtk_path.name} ({vtk_path.stat().st_size / 1024:.0f} kB, "
      f"ParaView-ready) and {h5_path.name}; HDF5 read back "
      f"{np.asarray(restored.data if hasattr(restored, 'data') else restored).shape}")

# %% 4. LOOK: the vis helpers ----------------------------------------------
#
# Each of these is the plot you would otherwise hand-write every time.
fake_losses = [float(v) for v in (torch.logspace(2, 0.2, 40)
                                  * (1 + 0.04 * torch.randn(40)))]
station_x = rcv_x.numpy()
station_y = np.zeros_like(station_x)

# This script's subject IS geobrain.vis, so it calls those helpers
# directly and lets each draw its own colour bar. Everywhere else the
# gallery uses the shared toolkit in _style.py, which can put one bar
# across a whole row.
fig, axes = figure(2, 2, scale=1.15)

plot_field_2d(vp.numpy(), dx=D, dz=D, cmap=CMAP_MODEL, label="vp [m/s]",
              title="plot_field_2d: the model", xlabel="Distance [m]",
              ylabel="Depth [m]", ax=axes[0, 0])
# plot_field_2d works in physical coordinates and locks an equal aspect;
# inside a panel grid that leaves the section floating, so hand the aspect
# back to the layout engine.
axes[0, 0].set_aspect("auto")
axes[0, 0].grid(False)

t_axis = (torch.arange(NT, dtype=torch.float64) * DT).numpy()
disp = (round_trip * torch.as_tensor(t_axis)[:, None] ** 2).numpy()
clip = float(abs(disp).max()) * 0.15
field(axes[0, 1], disp, cmap=CMAP_ANOMALY, vmin=-clip, vmax=clip,
      extent=(float(rcv_x[0]), float(rcv_x[-1]), NT * DT, 0.0),
      interpolation="bilinear", title="Read back from SEG-Y (bit-faithful)",
      xlabel="Receiver x [m]", ylabel="Time [s]")

plot_station_map(station_x, station_y, ax=axes[1, 0])
axes[1, 0].plot([600.0], [0.0], "*", ms=16, color=C_RECOVERED, mec="k")
axes[1, 0].annotate("Source", xy=(600.0, 0.0), xytext=(660.0, 22.0),
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9))
axes[1, 0].set_aspect("auto")           # the helper asks for equal aspect
axes[1, 0].set(title=f"plot_station_map: {len(station_x)} receivers",
               xlabel="Distance [m]", ylabel="Offset from line [m]",
               ylim=(-45.0, 45.0))

plot_convergence(fake_losses, ax=axes[1, 1])
axes[1, 1].set(title="plot_convergence: any loss history")

fig.savefig(OUT / "08_data_io_and_figures.png")
print(f"saved {OUT / '08_data_io_and_figures.png'}")
plt.show()
