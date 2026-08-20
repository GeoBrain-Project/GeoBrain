"""Reservoir flow: a five-spot waterflood, and what the adjoint prices.

Inject water in the middle, produce oil at four corners, and the
reservoir answers with the curves every production engineer reads first:
oil rate per well, and water cut against time. This runs that waterflood
through GeoBrain's two-phase :class:`OilWaterModel` on a geostatistical
permeability field, drawn by the platform's FFT-MA simulator rather than
painted by hand, so the flow paths are the ones the geology gives rather
than the ones the author wanted.

The result is the working rule of waterflooding: SWEEP IS DECIDED BY THE
PERMEABILITY FIELD, NOT BY THE WELL PATTERN. Four producers arranged
symmetrically around one injector produce four different histories,
because the high-permeability streaks between them are not symmetric.
One of them breaks through at 300 days and finishes at 39% water cut;
the other three are still dry when the run ends, and their final oil
rates differ from each other by a factor of 1.6. Nothing distinguishes
those wells except which beds the water found on the way to them.

The last panel is why the physics being differentiable matters here. The
march is an implicit Newton solve at every timestep, and it carries an
implicit-function adjoint, so ONE backward pass returns
``d(cumulative oil)/d(log k)`` for every cell in the model, at the price of
one extra simulation, not one per cell. Read it as an economic map:
permeability is worth having near the injector, where it buys
injectivity, and worth NOT having along a streak that runs straight to a
producer, because that spends the water on one well. That map is the
first step of a history match or a well-placement study.

Sign convention: well rates follow the canonical source convention,
injection positive, production negative. Production is negated for
display and the axes say so.

APIs featured:
    - geobrain.physics.flow.CartGrid, Rock, OilWaterModel
    - OilWaterFluid + PVTAnalytic + RelPermCorey
    - WellGroup / Well / Perforation / BHPControl with the Peaceman well
      index from compute_well_index
    - TransientFlowOperator with FlowHistoryConfig(mode="all") and
      WellObservationOperator for per-step rates
    - _models.correlated_fields for the permeability

Expected runtime: < 4 min.

Outputs:
    out/07_reservoir_flow.png: permeability, the swept saturation at
    the end of the run, the water cut per well, and the adjoint
    sensitivity.
    out/07_flood.gif: the flood front advancing through the streaks.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import torch

from _models import correlated_fields
from _style import (
    CMAP_ANOMALY,
    CMAP_MODEL,
    PALETTE,
    apply_style,
    animation,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.physics.flow import (
    BHPControl,
    CartGrid,
    FlowEvolutionOperator,
    FlowExecutionConfig,
    FlowHistoryConfig,
    Perforation,
    Rock,
    TimeStepScheduler,
    TransientFlowOperator,
    Well,
    WellGroup,
    WellObservationOperator,
)
from geobrain.physics.flow.models import OilWaterModel
from geobrain.physics.flow.properties.fluid import OilWaterFluid
from geobrain.physics.flow.properties.pvt import PVTAnalytic
from geobrain.physics.flow.properties.relperm import RelPermCorey
from geobrain.physics.flow.wells import WellStandardConditions, compute_well_index

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NX, NY, DX, DZ = 20, 12, 45.0, 10.0           # 900 x 540 m, 10 m thick
N_CELLS = NX * NY
P_INIT = 25.0e6
DAY = 86400.0
NSTEP, DT = 15, 30.0 * DAY                    # 450 days in monthly steps
PORO, SWC, SOR = 0.2, 0.2, 0.2
K_BG = 120e-15

# %% 1. A geostatistical permeability field --------------------------------
# Ranges in CELLS matter more than ranges in metres: 800 m along the
# fabric is 18 cells, 200 m across it is under 5. Below about 10 cells of
# continuity a geostatistical field stops reading as bedding and starts
# reading as salt and pepper, whatever its variogram says.
#
# The fabric is axis-aligned because it has to be: at these ranges the
# FFT-MA embedding of a ROTATED anisotropic covariance leaves ~1e-5
# relative imaginary leakage and the simulator refuses it (threshold
# 1e-12). That refusal is the right call, and 0 degrees is a bedding
# direction, not a workaround.
perm_field, _ = correlated_fields((NY, NX), DX, seed=21,
                                  ranges=(800.0, 200.0), azimuth=0.0,
                                  correlation=0.0)
perm_grid = K_BG * torch.exp(1.5 * perm_field)
print(f"[1] {NX}x{NY} cells at {DX:.0f} m; FFT-MA permeability "
      f"{float(perm_grid.min()) / 1e-15:.0f} - "
      f"{float(perm_grid.max()) / 1e-15:.0f} mD, streaks {800 / DX:.0f} "
      f"cells long and {200 / DX:.0f} across")

# %% 2. Rock, fluids, and the two-phase model ------------------------------
#
# Corey relative permeability with a 4:1 viscosity contrast gives an
# end-point mobility ratio of 4, which is unfavourable, so water fingers
# whatever high-permeability path it finds.
grid = CartGrid(nx=NX, ny=NY, nz=1, dx_m=DX, dy_m=DX, dz_m=DZ)
pvt_w = PVTAnalytic(density_ref_kg_m3=1000.0, viscosity_ref_pa_s=5.0e-4,
                    formation_volume_factor_ref=1.0,
                    reference_pressure_pa=P_INIT,
                    compressibility_pa_inv=4.4e-10)
pvt_o = PVTAnalytic(density_ref_kg_m3=800.0, viscosity_ref_pa_s=2.0e-3,
                    formation_volume_factor_ref=1.0,
                    reference_pressure_pa=P_INIT,
                    compressibility_pa_inv=1.0e-9)
fluid = OilWaterFluid(pvt_o=pvt_o, pvt_w=pvt_w,
                      relperm=RelPermCorey(swc=SWC, sor=SOR, n_w=2.0, n_o=2.0))
poro = torch.full((N_CELLS,), PORO, dtype=D)


def build_model(perm: torch.Tensor) -> OilWaterModel:
    """Rebuild the model around a live permeability tensor.

    The transient march declares the DYNAMIC state (pressure, sw) as its
    input surface; rock properties are constructor-bound. Closing over the
    tensor keeps them on the autograd graph so the implicit adjoint
    reaches them. ParametricFlowOperator is the packaged form of this
    pattern when the properties must be wired by name into an
    InverseProblem.
    """
    rock = Rock(permeability_m2=perm.reshape(-1), porosity=poro,
                compressibility_pa_inv=torch.full((N_CELLS,), 4.5e-10, dtype=D),
                reference_pressure_pa=torch.full((N_CELLS,), P_INIT, dtype=D))
    return OilWaterModel(grid, rock, fluid)


# %% 3. The five-spot ------------------------------------------------------
def cell(i: int, j: int) -> int:
    return j * NX + i


STD_RHO = {"oil": 800.0, "water": 1000.0}
STD_COND = WellStandardConditions(pressure_pa=101325.0, temperature_k=288.71)
WI = compute_well_index(dx_m=DX, dy_m=DX, dz_m=DZ, kx_m2=K_BG)
INJ_IJ = (NX // 2, NY // 2)
PROD_IJ = [(2, 2), (NX - 3, 2), (2, NY - 3), (NX - 3, NY - 3)]
PROD_NAMES = ["P1 south-west", "P2 south-east", "P3 north-west",
              "P4 north-east"]
wells = WellGroup([
    Well(name="INJ", well_type="INJ", control=BHPControl(pressure_pa=27.5e6),
         injection_phase="water", standard_conditions=STD_COND,
         standard_densities_kg_m3=STD_RHO,
         perforations=(Perforation(cell_idx=cell(*INJ_IJ), well_index_m3=WI),)),
    *[Well(name=f"P{k}", well_type="PROD",
           control=BHPControl(pressure_pa=22.5e6), standard_conditions=STD_COND,
           standard_densities_kg_m3=STD_RHO,
           perforations=(Perforation(cell_idx=cell(i, j), well_index_m3=WI),))
      for k, (i, j) in enumerate(PROD_IJ, start=1)],
], n_cells=N_CELLS)
print(f"[3] five-spot: injector at 27.5 MPa, four producers at 22.5 MPa; "
      f"Peaceman well index {WI:.2e} m3")

# %% 4. March 450 days, keeping every step ---------------------------------
log_k = perm_grid.reshape(-1).log().clone().requires_grad_(True)
model = build_model(log_k.exp())
march = TransientFlowOperator(FlowEvolutionOperator(
    model, config=FlowExecutionConfig(autograd_mode="implicit",
                                      history=FlowHistoryConfig(mode="all"))))
ctx = ForwardContext.of(
    t_end=NSTEP * DT, scheduler=TimeStepScheduler(dt_list=[DT] * NSTEP),
    wells=wells, well_observer=WellObservationOperator(model))
initial = {"pressure": torch.full((N_CELLS,), P_INIT, dtype=D),
           "sw": torch.full((N_CELLS,), SWC, dtype=D)}
t0 = time.time()
out = march(ModelState(dict(initial)), ctx)
print(f"[4] {NSTEP} steps of {DT / DAY:.0f} days in {time.time() - t0:.0f} s")

q_oil = out.data["oil_surface_m3_s_series"]
q_wat = out.data["water_surface_m3_s_series"]
sw_series = out.fields["sw_series"]
days = torch.arange(1, NSTEP + 1, dtype=D) * DT / DAY
oil_rate = -q_oil[:, 1:] * DAY
water_rate = q_wat[:, 1:].abs() * DAY
inj_rate = q_wat[:, 0] * DAY
water_cut = water_rate / (water_rate + oil_rate).clamp_min(1e-30)

PV = N_CELLS * DX * DX * DZ * PORO
OOIP = PV * (1.0 - SWC)
cum_oil = (oil_rate.sum(dim=1) * DT / DAY).cumsum(0)
for k, name in enumerate(PROD_NAMES):
    crossed = [float(days[i]) for i in range(NSTEP)
               if float(water_cut[i, k]) > 0.01]
    when = f"breakthrough at {crossed[0]:.0f} days" if crossed else "still dry"
    print(f"[4] {name}: final oil {float(oil_rate[-1, k]):5.1f} m3/day, water "
          f"cut {float(water_cut[-1, k]) * 100:4.1f}%, {when}")
spread = float(oil_rate[-1].max() / oil_rate[-1].min())
print(f"[4] the four producers differ by a factor {spread:.1f} in final oil "
      f"rate; recovery {float(cum_oil[-1]) / OOIP * 100:.1f}% of OOIP")

# %% 5. One adjoint pass, one sensitivity map ------------------------------
t0 = time.time()
(sens,) = torch.autograd.grad(cum_oil[-1], log_k)
sens_grid = sens.detach().reshape(NY, NX)
print(f"[5] d(cumulative oil)/d(log k) for all {N_CELLS} cells in "
      f"{time.time() - t0:.2f} s: one adjoint, not {N_CELLS} simulations")
print(f"    at the injector {float(sens_grid[INJ_IJ[1], INJ_IJ[0]]):+8.0f} m3 "
      "per unit log k")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(2, 2)
MAP = dict(extent=(0.0, NX * DX, 0.0, NY * DX), origin="lower",
           xlabel="Easting [m]", ylabel="Northing [m]")
well_xy = [((i + 0.5) * DX, (j + 0.5) * DX) for i, j in PROD_IJ]
inj_xy = ((INJ_IJ[0] + 0.5) * DX, (INJ_IJ[1] + 0.5) * DX)


def draw_wells(ax) -> None:
    ax.plot(*inj_xy, "o", ms=9, color=PALETTE[0], mec="white", mew=1.4,
            ls="none", label="Injector")
    ax.plot([p[0] for p in well_xy], [p[1] for p in well_xy], "v", ms=9,
            color="black", mec="white", mew=1.4, ls="none", label="Producers")
    for k, (x, y) in enumerate(well_xy, start=1):
        ax.annotate(f"P{k}", xy=(x, y), xytext=(8, 5),
                    textcoords="offset points", fontsize=8, color="white",
                    path_effects=[pe.withStroke(linewidth=2.2,
                                                foreground="black")])


image = field(axes[0, 0], (perm_grid / 1e-15).log10().numpy(),
              cmap=CMAP_MODEL, title="Permeability - drawn by FFT-MA", **MAP)
shared_colorbar(fig, image, axes[0, 0], r"$\log_{10} k$ [mD]",
                location="bottom")
draw_wells(axes[0, 0])
axes[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.9,
                  frameon=True)

field_step = sw_series[NSTEP].detach().reshape(NY, NX)
image = field(axes[0, 1], field_step.numpy(), cmap=CMAP_MODEL, vmin=SWC,
              vmax=1.0 - SOR,
              title=f"Water saturation at {NSTEP * DT / DAY:.0f} days", **MAP)
draw_wells(axes[0, 1])
shared_colorbar(fig, image, axes[0, 1], r"$S_w$ [-]", location="bottom")

ax = axes[1, 0]
for k, name in enumerate(PROD_NAMES):
    ax.plot(days.numpy(), (water_cut[:, k] * 100.0).detach().numpy(),
            color=PALETTE[k], lw=2.0, label=name)
ax.set(title="Water cut",
       xlabel="Time [days]", ylabel="Water cut [%]")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1, 1]
off_well = torch.ones(N_CELLS, dtype=torch.bool)
off_well[[cell(*INJ_IJ)] + [cell(i, j) for i, j in PROD_IJ]] = False
lim = float(sens.detach()[off_well].abs().max())
image = field(ax, sens_grid.numpy(), cmap=CMAP_ANOMALY, vmin=-lim,
              vmax=lim,
              title=r"$\partial$ oil / $\partial \log k$",
              **MAP)
shared_colorbar(fig, image, ax,
                r"$\partial$ oil / $\partial \log k$ [m3]",
                location="bottom")
draw_wells(ax)
ax.annotate("well cells run off this scale", xy=(0.03, 0.04),
            xycoords="axes fraction", fontsize=8, color="dimgray")

# %% 7. The flood, as an animation ----------------------------------------
#
# Time is the subject here: which producer the water reaches first is the
# whole result, and no still frame carries it.
# The canvas matches the FWI animation's so the two sit side by side in the
# README at one size. The map keeps its own proportions inside it: a 900 by
# 540 m field stretched to fill a 2.4:1 canvas reads half again as wide as it
# is, and a map that lies about shape is worse than one with space beside it.
anim_fig, anim_ax = figure(1, 1, panel_w=8.6, panel_h=3.6)
anim_frames = list(range(0, NSTEP + 1, max(1, NSTEP // 28)))
anim_image = field(anim_ax, sw_series[0].detach().reshape(NY, NX).numpy(),
                   cmap=CMAP_MODEL, vmin=SWC, vmax=1.0 - SOR,
                   aspect="equal", **MAP)
anim_fig.colorbar(anim_image, ax=anim_ax, label=r"$S_w$ [-]")
draw_wells(anim_ax)


def draw_flood(index: int) -> None:
    step = anim_frames[index]
    anim_image.set_data(sw_series[step].detach().reshape(NY, NX).numpy())
    anim_ax.set_title(f"Water saturation, day {step * DT / DAY:.0f}")


animation(anim_fig, draw_flood, len(anim_frames), OUT / "07_flood.gif")

fig.savefig(OUT / "07_reservoir_flow.png")
print(f"saved {OUT / '07_reservoir_flow.png'}")
plt.show()
