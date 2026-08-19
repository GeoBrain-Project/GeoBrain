"""Potential fields: mass has no direction, and magnetisation has two.

One buried block, seen by two instruments. Gravity reports a symmetric
high centred over the body and it always will: mass has no direction.
Magnetics reports something that depends on where you are standing AND on
what happened to the rock, because a magnetic anomaly is a dipole field
projected back onto the measurement direction, and the dipole can point
anywhere.

Three magnetisations of the SAME body are modelled here:

    induced, mid-latitude (I = 60 deg)   magnetisation along the present
        field. The familiar asymmetric couplet: a high with a low on its
        northern flank.

    induced, equatorial (I = 0 deg)      the anomaly INVERTS to a central
        low flanked by highs. Nothing about the geology changed; only the
        latitude did. Reduction-to-the-pole exists to undo exactly this.

    remanent                             magnetisation frozen in when the
        rock cooled, pointing 90 degrees away from today's field. The
        anomaly is different again, and this is the case that breaks
        susceptibility inversion, because such an inversion assumes the
        magnetisation direction is known and equal to the inducing field.
        Recovering direction as well as amplitude is what a magnetic
        VECTOR inversion is for, and :class:`MagneticVector3D` is the
        operator that makes it possible: its trainable input is the
        magnetisation vector itself, three numbers per cell.

The measured consequence is in the print-out: over the body centre the
induced anomaly and the remanent one have opposite signs. Interpret the
second with an induced-only assumption and you will put a negative
susceptibility in the ground.

Sign convention: Gravity3D returns ELEVATION-UP gz, so a dense body gives
a negative number; it is negated for display and the axis says
"downward attraction".

APIs featured:
    - geobrain.physics.potential.Gravity3D (grid-shaped rho)
    - geobrain.physics.potential.Magnetic3D + EarthField (flat chi)
    - geobrain.physics.potential.MagneticVector3D (magnetisation vector,
      CUSTOM_VJP) for the remanent case

Expected runtime: < 30 s.

Outputs:
    out/06_potential_fields.png: the gravity map and the three magnetic
    maps of the same body, with its plan outline on each.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_ANOMALY,
    CMAP_EXCESS,
    apply_style,
    field,
    figure,
    outer_labels,
    shared_colorbar,
    symmetric_limits,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.physics.potential import (
    EarthField,
    Gravity3D,
    Magnetic3D,
    MagneticVector3D,
    PotentialSurvey3D,
)

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NZ, NX, NY, DH = 16, 24, 24, 25.0                 # 400 m deep, 600 x 600 m
mesh = TensorMesh(shape=(NZ, NX, NY), spacing=(DH, DH, DH))
ctx = ForwardContext.of(mesh=mesh)

# %% 1. One block, dense and magnetic --------------------------------------
BODY = (slice(4, 10), slice(9, 15), slice(9, 15))     # 100-250 m deep
rho = torch.zeros(mesh.shape, dtype=D)
rho[BODY] = 600.0                                     # kg/m^3
chi = torch.zeros(mesh.shape, dtype=D)
chi[BODY] = 0.06                                      # SI susceptibility
body_mask = torch.zeros(mesh.shape, dtype=torch.bool)
body_mask[BODY] = True
print(f"[1] {mesh.n_cells} cells; body 100-250 m deep, "
      f"rho contrast 600 kg/m3, susceptibility 0.06 SI")

line = torch.linspace(12.5, 587.5, 24, dtype=D)
gx, gy = torch.meshgrid(line, line, indexing="ij")
stations = torch.stack([gx.reshape(-1), gy.reshape(-1),
                        torch.ones(gx.numel(), dtype=D)], dim=1)
survey = PotentialSurvey3D(stations)
print(f"[1] {stations.shape[0]} stations on a {len(line)} x {len(line)} grid")

# %% 2. Gravity: one answer, wherever you stand ----------------------------
with torch.no_grad():
    (gz,) = Gravity3D(survey)(ModelState({"rho": rho}), ctx).fetch("gz")
grav = (-gz * 1e5).reshape(len(line), len(line))      # down-positive mGal
print(f"[2] gravity peaks at {float(grav.max()):.3f} mGal, and would look "
      "the same at any latitude")

# %% 3. Induced magnetisation, two latitudes -------------------------------
FIELD_T = 50e-6
maps: dict[str, torch.Tensor] = {}
for label, inc in (("Induced, I = 60°", 60.0), ("Induced, I = 0°", 0.0)):
    earth_field = EarthField(intensity_tesla=FIELD_T, inclination_deg=inc,
                             declination_deg=0.0)
    with torch.no_grad():
        (tmi,) = Magnetic3D(survey, earth_field)(
            ModelState({"chi": chi.reshape(-1)}), ctx).fetch("tmi")
    maps[label] = (tmi * 1e9).reshape(len(line), len(line))
    m = maps[label]
    print(f"[3] {label}: TMI {float(m.min()):+8.1f} to {float(m.max()):+7.1f} nT")

# %% 4. Remanent magnetisation ---------------------------------------------
#
# MagneticVector3D takes the magnetisation itself, (n_cells, 3) in A/m, so
# the direction is a free choice rather than the field's. Here it is
# rotated 90 degrees in inclination from today's field, a rock that
# cooled through its Curie point when the pole was somewhere else.
MU0 = 4.0e-7 * math.pi
survey_field = EarthField(intensity_tesla=FIELD_T, inclination_deg=60.0,
                          declination_deg=0.0)
amplitude = 0.06 * FIELD_T / MU0                   # same |M| as the induced case
rem_inc = math.radians(-30.0)                      # 90 deg away from I = 60
direction = torch.tensor([math.sin(rem_inc), math.cos(rem_inc), 0.0], dtype=D)
magnetisation = torch.zeros((mesh.n_cells, 3), dtype=D)
magnetisation[body_mask.reshape(-1)] = amplitude * direction
vector_op = MagneticVector3D(survey, survey_field)
with torch.no_grad():
    (tmi_rem,) = vector_op(ModelState({"magnetization": magnetisation}),
                           ctx).fetch("tmi")
maps["Remanent, 90° off"] = (tmi_rem * 1e9).reshape(len(line), len(line))
m = maps["Remanent, 90° off"]
print(f"[4] Remanent: TMI {float(m.min()):+8.1f} to {float(m.max()):+7.1f} nT "
      f"(declared {vector_op.differentiability.level.name} on "
      f"{vector_op.differentiability.trainable_inputs[0]})")

induced = maps["Induced, I = 60°"]
centre = len(line) // 2
print(f"[4] over the body centre the induced anomaly reads "
      f"{float(induced[centre, centre]):+.1f} nT and the remanent one "
      f"{float(m[centre, centre]):+.1f} nT: same rock, opposite sign; an "
      "induced-only inversion would answer with negative susceptibility")

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(2, 2, panel_w=4.3, panel_h=4.3)
MAP_EXTENT = (0.0, NX * DH, 0.0, NY * DH)
MAP = dict(extent=MAP_EXTENT, origin="lower")

# The body's plan footprint, drawn on every panel. Neither the density slice
# nor the normalized profiles survive: the slice said where the body is,
# which this outline says on the maps themselves, and the profiles were the
# three magnetic maps restated as curves.
BODY_X = (BODY[1].start * DH, BODY[1].stop * DH)
BODY_Y = (BODY[2].start * DH, BODY[2].stop * DH)


def footprint(ax):
    ax.add_patch(plt.Rectangle((BODY_X[0], BODY_Y[0]),
                               BODY_X[1] - BODY_X[0], BODY_Y[1] - BODY_Y[0],
                               fill=False, ec="black", ls="--", lw=1.3))


# Gravity takes the RED HALF of the diverging ramp the magnetic panels use,
# not a separate sequential one: across this figure red then means "more" and
# blue "less" in every panel. On a magnitude ramp the gravity high would be
# dark blue, which is what negative TMI is in the panels beside it.
image = field(axes[0, 0], grav.T.numpy(), cmap=CMAP_EXCESS, vmin=0.0,
              vmax=float(grav.max()), title="Gravity - symmetric, always",
              **MAP)
footprint(axes[0, 0])
shared_colorbar(fig, image, axes[0, 0], "Downward attraction [mGal]",
                location="bottom")

# The three magnetic maps share ONE symmetric scale, so the panels report the
# amplitude difference between latitudes as well as the shape change. The
# group is an L across the grid, so its bar goes to the figure's edge.
mag_lim = symmetric_limits(*[m.numpy() for m in maps.values()])
mag_axes = (axes[0, 1], axes[1, 0], axes[1, 1])
for ax, label in zip(mag_axes, maps):
    image = field(ax, maps[label].T.numpy(), cmap=CMAP_ANOMALY,
                  vmin=mag_lim[0], vmax=mag_lim[1], title=label, **MAP)
    footprint(ax)
shared_colorbar(fig, image, mag_axes, "TMI [nT]", location="right")
outer_labels(axes, "Easting [m]", "Northing [m]")

fig.savefig(OUT / "06_potential_fields.png")
print(f"saved {OUT / '06_potential_fields.png'}")
plt.show()
