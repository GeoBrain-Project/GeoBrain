"""Implicit geology: build the structure from contacts, and differentiate it.

Everything earlier in this section models a PROPERTY on a fixed grid.
This script models the GEOMETRY: where the contacts are, which unit sits
on which, and what a fault did to them. That is the other half of a
geological model, and it is not a kriging problem in the usual sense:
the data are contact points and dip measurements, and the answer is a
surface.

The implicit method makes it one interpolation. Instead of triangulating
surfaces by hand, it fits a scalar field whose level sets ARE the
contacts: every point observed on the same horizon gets the same scalar
value, and every dip measurement constrains the field's gradient there.
Universal cokriging does the fit, contacts fall out as iso-surfaces, and
stacking rules (erode, onlap) decide which unit wins where two series
overlap.

Why it matters here that this is differentiable
-----------------------------------------------
The block model this produces is a tensor computed from the contact
coordinates by operations autograd understands. The classification into
units is a sigmoid rather than a step, that is what ``soft=True`` means,
so the whole model is differentiable end to end with respect to the
data that built it.

That is not a curiosity. It means the geometry can be a variable in an
inversion: adjust where a contact sits until the model reproduces
something measured, with gradients instead of a search. This script does
not run that inversion; it establishes the thing that makes it possible,
by taking the derivative of a rock volume with respect to the depth of a
contact point and checking it against a central finite difference.

What is measured
----------------
1. The block model, from six contact points and three dip measurements
   per horizon, with a fault displacing both.

2. Where the soft classification stops being differentiable. The sigmoid
   temperature is usually presented as a cosmetic knob; it is not. At
   temperature 1 the contact is blurred over about nine metres and the
   adjoint matches a central difference to within about one percent. At
   temperature 5 the contact is a metre and a half wide and the two
   disagree by a factor of two; at 50 they disagree in SIGN. A near-step
   classification does not change continuously with the data, so the
   difference quotient sees nothing while the adjoint sees a spike.

   Crisp pictures and usable gradients are different settings. This
   script ships the one that keeps the gradient, and prints the whole
   trade so the choice is visible rather than inherited.

3. The gradient of the mean unit code with respect to one contact point's
   depth, adjoint against central difference, at the shipped temperature.

APIs featured:
    - geobrain.geomodel.ImplicitModel / ImplicitModelConfig
    - SurfacePointData + OrientationData as the two kinds of structural data
    - SeriesDefinition with StackRelation, and FaultDefinition
    - autograd straight through the block model

Expected runtime: < 2 min.

Outputs:
    out/06_implicit_modelling.png: the input data, the scalar field, the
    soft and hard block models, the temperature study, and the gradient
    check.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _style import (
    CMAP_MODEL,
    PALETTE,
    C_LIMIT,
    C_TRUTH,
    apply_style,
    category_colorbar,
    category_norm,
    figure,
    shared_colorbar,
)
from geobrain.geomodel import (
    FaultDefinition,
    ImplicitModel,
    ImplicitModelConfig,
    OrientationData,
    SeriesDefinition,
    StackRelation,
    SurfacePointData,
)

apply_style()
torch.set_default_dtype(torch.float64)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
EXTENT = (0.0, 1000.0, 0.0, 600.0)          # x0, x1, z0, z1 in metres
RES = (100, 60)
TEMPERATURE = 1.0

# %% 1. The data a geologist actually collects ----------------------------
# Two horizons, each seen at three outcrops/intersections, each with a dip
# measurement. The horizons are gently folded and the lower one is deeper.
def horizon(depth: float, amplitude: float) -> torch.Tensor:
    x = torch.tensor([120.0, 500.0, 880.0], dtype=D)
    z = depth + amplitude * torch.sin(2.0 * torch.pi * x / 1400.0)
    return torch.stack([x, z], dim=1)


upper_points = horizon(430.0, 55.0)
lower_points = horizon(260.0, 40.0)
contacts = torch.cat([upper_points, lower_points], dim=0)
surface_id = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

# Orientations are gradients of the scalar field: normal to bedding, so for a
# gently dipping layer they point mostly "up" in z with a small x component.
dip = torch.tensor([[-0.22, 1.0], [0.0, 1.0], [0.22, 1.0]] * 2, dtype=D)

config = ImplicitModelConfig(extent=EXTENT, resolution=RES, kernel="cubic",
                             range=900.0, drift_degree=1)
FAULT_X = 640.0
fault_points = torch.tensor([[FAULT_X, 150.0], [FAULT_X + 40.0, 520.0]],
                            dtype=D)
fault = FaultDefinition(
    name="normal fault",
    surface_points=SurfacePointData(coords=fault_points,
                                    surface_id=torch.zeros(2,
                                                           dtype=torch.long)),
    orientations=OrientationData(coords=fault_points,
                                 gradients=torch.tensor([[1.0, -0.1],
                                                         [1.0, -0.1]],
                                                        dtype=D)),
    affected_series_indices=(0,), displacement=70.0)
print(f"[1] {RES[0]}x{RES[1]} section over {EXTENT[1]:.0f} m by "
      f"{EXTENT[3]:.0f} m; two horizons from three contact points and three "
      f"dip measurements each, cut by a fault at x = {FAULT_X:.0f} m with "
      "70 m of displacement")


def build(contact_coords: torch.Tensor, temperature: float = TEMPERATURE,
          soft: bool = True):
    series = SeriesDefinition(
        name="stratigraphy",
        surface_points=SurfacePointData(coords=contact_coords,
                                        surface_id=surface_id),
        # Same tensor as the contacts: a dip is measured AT an observation,
        # and tying them keeps the perturbation physical.
        orientations=OrientationData(coords=contact_coords, gradients=dip),
        relation=StackRelation.ERODE,
        surface_names=("upper contact", "lower contact"))
    return ImplicitModel(config, series=[series],
                         faults=[fault])(soft=soft, temperature=temperature)


t0 = time.time()
result = build(contacts)
block = result["block"]
scalar = result["scalar_fields"][0]
print(f"[2] model built in {time.time() - t0:.1f} s: block "
      f"{tuple(block.shape)}, values {float(block.min()):.2f} to "
      f"{float(block.max()):.2f}; the scalar field it came from spans "
      f"{float(scalar.min()):.2f} to {float(scalar.max()):.2f} with "
      f"iso-values {[round(float(v), 2) for v in result['iso_vals'][0]]}")

# %% 2. Soft or hard, and where soft stops being differentiable ----------
hard = build(contacts, soft=False)["block"]
cell_z = (EXTENT[3] - EXTENT[2]) / RES[1]


def gradient_check(temperature: float, step: float = 1.0):
    """Adjoint against a central difference, at one contact coordinate."""
    tgt = contacts.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(build(tgt, temperature)["block"].mean(), tgt)
    shifted_values = []
    for sign in (1.0, -1.0):
        shifted = contacts.clone()
        shifted[0, 1] += sign * step
        with torch.no_grad():
            shifted_values.append(float(build(shifted,
                                              temperature)["block"].mean()))
    finite = (shifted_values[0] - shifted_values[1]) / (2.0 * step)
    return float(grad[0, 1]), finite, grad


TEMPERATURES = (0.05, 0.5, 1.0, 5.0, 50.0)
study = {}
print("[3] the sigmoid temperature trades crispness against differentiability, "
      "and the trade is measurable:")
for temperature in TEMPERATURES:
    soft_block = build(contacts, temperature=temperature)["block"]
    grid_block = soft_block.reshape(RES[0], RES[1])
    blur = float(((grid_block > 0.15) & (grid_block < 0.85)).double()
                 .sum(dim=1).mean()) * cell_z
    adjoint, finite, _ = gradient_check(temperature)
    ratio = adjoint / finite if finite != 0.0 else float("nan")
    study[temperature] = (blur, ratio)
    verdict = "gradient exact" if abs(ratio - 1.0) < 0.05 else "GRADIENT WRONG"
    print(f"      temperature {temperature:6.2f}: contact blurred over "
          f"{blur:5.1f} m, adjoint/central-difference {ratio:+7.3f}  "
          f"{verdict}")
print("[3] above about 5 the classification is a step in all but name: the "
      "block stops changing continuously with the data, so the central "
      "difference sees nothing while the adjoint sees a spike. Crisp pictures "
      "and usable gradients are not the same setting, and this script ships "
      f"{TEMPERATURE:.1f} because it wants the gradient")

# %% 3. The point of all this: the geometry carries gradients -------------
adjoint, central, grad = gradient_check(TEMPERATURE, step=1.0)
print("[4] d(mean unit code) / d(depth of contact point 0), at temperature "
      f"{TEMPERATURE:.1f}:")
print(f"      adjoint            {adjoint:+.8e} per metre")
print(f"      central difference {central:+.8e} per metre")
print(f"      ratio {adjoint / central:.6f}")
print(f"[4] every one of the {int(grad.numel())} contact coordinates carries a "
      f"gradient (norm {float(grad.norm()):.3e}), so the STRUCTURE can be the "
      "unknown in an inversion, not just the property painted on it")

# %% 4. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
extent_plot = (EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3])
contacts_np = contacts.detach().numpy()


def show(ax, field, title, style, label=None):
    im = ax.imshow(np.asarray(field).reshape(RES[0], RES[1]).T, origin="lower",
                   extent=extent_plot, aspect="auto", **style)
    ax.set(title=title, xlabel="x [m]", ylabel="z [m]")
    ax.grid(False)
    return im


ax = axes[0, 0]
ax.plot(contacts_np[:3, 0], contacts_np[:3, 1], "o", ms=10, color=PALETTE[0],
        mec="black", ls="none", label="Upper contact")
ax.plot(contacts_np[3:, 0], contacts_np[3:, 1], "s", ms=10, color=PALETTE[2],
        mec="black", ls="none", label="Lower contact")
orient_np = contacts_np
dip_np = dip.detach().numpy()
ax.quiver(orient_np[:, 0], orient_np[:, 1], dip_np[:, 0], dip_np[:, 1],
          color=C_TRUTH, scale=12, width=0.006, label="Dip (gradient)")
ax.plot(fault_points[:, 0].numpy(), fault_points[:, 1].numpy(), "--",
        color=PALETTE[1], lw=2.2, label="Fault")
ax.set(title="All the data there is", xlabel="x [m]", ylabel="z [m]",
       xlim=EXTENT[:2], ylim=EXTENT[2:])
ax.legend(fontsize=8, loc="lower left")

im = show(axes[0, 1], scalar.detach().numpy(),
          "The scalar field\ncontacts are its level sets",
          dict(cmap=CMAP_MODEL))
axes[0, 1].plot(contacts_np[:, 0], contacts_np[:, 1], "o", ms=6,
                color="white", mec="black", ls="none")
shared_colorbar(fig, im, axes[0, 1], "scalar [-]")

# The block model names units; the soft one is a membership between them,
# so both get the discrete unit colours rather than a continuous ramp.
_unit_cmap, _unit_norm = category_norm(2)
im = show(axes[1, 0], block.detach().numpy(),
          f"Soft block model (temperature {TEMPERATURE:.0f})",
          dict(cmap=_unit_cmap, norm=_unit_norm))
show(axes[1, 1], hard.detach().numpy(), "Hard block model",
     dict(cmap=_unit_cmap, norm=_unit_norm))
category_colorbar(fig, im, axes[1, :2], "Unit", ["lower", "upper"])

ax = axes[0, 2]
temps = sorted(study)
ax.semilogx(temps, [study[t][0] for t in temps], "o-", color=PALETTE[0],
            lw=2.0, label="Contact blur")
ax.set(title="Contact blur and gradient error by temperature",
       xlabel="Sigmoid temperature", ylabel="Contact blurred over [m]")
twin = ax.twinx()
twin.semilogx(temps, [abs(study[t][1] - 1.0) for t in temps], "s--",
              color=PALETTE[1], lw=1.8, label="Gradient error")
twin.set_yscale("log")
twin.set_ylabel("|adjoint / finite difference - 1|", color=PALETTE[1])
twin.tick_params(axis="y", colors=PALETTE[1])
twin.axhline(0.05, color=C_LIMIT, ls=":", lw=1.2)
twin.grid(False)
ax.axvline(TEMPERATURE, color=C_TRUTH, ls="--", lw=1.4)
ax.annotate("shipped", xy=(TEMPERATURE, max(study[t][0] for t in temps)),
            xytext=(5, -10), textcoords="offset points", fontsize=8,
            color=C_TRUTH)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1, 2]
grad_np = grad.detach().numpy()
labels = [f"pt {i}\n{'upper' if i < 3 else 'lower'}" for i in range(6)]
positions = np.arange(6)
ax.bar(positions - 0.2, grad_np[:, 0], 0.4, color=PALETTE[0], label="d/dx")
ax.bar(positions + 0.2, grad_np[:, 1], 0.4, color=PALETTE[2], label="d/dz")
ax.axhline(0.0, color="black", lw=0.8)
ax.set(title="Upper-unit volume gradient (adjoint / finite "
             f"difference {adjoint / central:.4f})",
       xlabel="Contact point", ylabel="d(volume fraction) / d(position) [1/m]",
       xticks=positions)
ax.set_xticklabels(labels, fontsize=7)
ax.legend(fontsize=8)

fig.savefig(OUT / "06_implicit_modelling.png")
print(f"saved {OUT / '06_implicit_modelling.png'}")
plt.show()
