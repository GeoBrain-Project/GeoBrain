"""Operator algebra: physics composes in series and in parallel.

GeoBrain builds every forward model out of two composition primitives,
and nothing else:

- SERIAL, with ``@``:  ``Gravity2D(survey) @ GardnerOperator()``
  where rock physics turns vp into density and gravity observes it.
  The chain is itself an Operator with a DERIVED contract: trainable
  inputs from the entry link, output keys from the terminal link, and
  the weakest differentiability level of the two.

- PARALLEL, with an :class:`OperatorBundle`:
  ``OperatorBundle({"seismic": acoustic_chain, "gravity": gravity_chain})``
  where named channels thread ONE shared :class:`ModelState` through every
  member in a single call, and merge into one :class:`ForwardOutput`.
  Its contract is derived too: trainable inputs are the union, the
  output keys are the channel names.

The two compose with each other. Both channels here are themselves
chains, and both reuse the SAME ``GardnerOperator`` link: the acoustic
FDTD engine needs (vp, rho) and so does the gravity kernel, so one rock
physics transform feeds both. And because every channel reads the same
state, one ``backward()`` accumulates the seismic sensitivity (through
the time-domain adjoint) and the gravity sensitivity (through Gardner's
chain rule) into the SAME ``vp.grad``, which is all "joint inversion"
ever meant.

APIs featured:
    - Operator.__matmul__ -> OperatorChain (serial), derived contract
    - geobrain.core.OperatorBundle (parallel), merged output + units
    - two chains nested inside one bundle, sharing a rock-physics link
    - geobrain.physics.wave.Acoustic2D (time-domain FDTD) + Seismic2DSurvey
    - one torch.autograd backward across both channels

Expected runtime: < 60 s.

Outputs:
    out/02_operator_composition.png: what the serial chain takes in
    and what it puts out, then one parallel channel's data and the single
    gradient both channels land in.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_ANOMALY,
    CMAP_SEISMIC,
    CMAP_VELOCITY,
    C_RECOVERED,
    C_SERIES,
    C_TRUTH,
    apply_style,
    field,
    figure,
    shared_colorbar,
    symmetric_limits,
)
from geobrain.core import ForwardContext, ModelState, OperatorBundle
from geobrain.mesh import TensorMesh
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D
from geobrain.physics.rock.models import GardnerOperator
from geobrain.physics.wave import (
    Acoustic2D,
    Seismic2DSurvey,
    ricker,
    shared_wavelet,
)

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. One earth ----------------------------------------------------------
NZ, NX = 24, 48
DZ = DX = 25.0                                   # 600 m deep, 1.2 km wide
mesh = TensorMesh(shape=(NZ, NX), spacing=(DZ, DX))
zz = torch.arange(NZ, dtype=torch.float64)[:, None] * DZ

BODY = (slice(6, 12), slice(18, 30))             # 150-300 m deep block
vp_bg = (1900.0 + 0.8 * zz).expand(NZ, NX).clone()
vp_bg[14:, :] += 350.0                           # a flat reflector at 350 m
vp_true = vp_bg.clone()
vp_true[BODY] += 550.0                           # fast AND dense (via Gardner)
ctx = ForwardContext.of(mesh=mesh)

# %% 2. SERIAL composition: chain with @ -----------------------------------
xs = torch.linspace(15.0, 1185.0, 40, dtype=torch.float64)
gardner = GardnerOperator()                      # vp -> rho
gravity = Gravity2D(PotentialSurvey2D(torch.stack([xs, torch.ones_like(xs)], 1)))
grav_chain = gravity @ gardner                   # vp -> rho -> gz

spec = grav_chain.differentiability
print(f"SERIAL   {grav_chain}")
print(f"         contract: {spec.trainable_inputs} -> {spec.output_keys}  "
      f"level={spec.level.value}   (entry inputs, terminal outputs, "
      f"weakest level)")

# the intermediate state is one call away whenever you want to see it
(rho_true,) = gardner(ModelState(tensors={"vp": vp_true})).fetch("rho")

# %% 3. PARALLEL composition: a bundle of named channels -------------------
#
# The acoustic engine needs (vp, rho) exactly like gravity does, so the
# SAME Gardner link feeds a second chain, and both chains become
# channels of one bundle.
NT, DT, F0 = 500, 0.002, 10.0                    # 1.0 s record, 10 Hz Ricker
SRC_X, N_RCV = [100.0, 600.0, 1100.0], 48        # 3 shots, full receiver line
rcv_x = torch.linspace(25.0, 1175.0, N_RCV, dtype=torch.float64)
survey = Seismic2DSurvey.from_positions(
    source_positions=[[x, 25.0] for x in SRC_X],
    source_shot_index=list(range(len(SRC_X))),
    receiver_positions=[[float(x), 25.0] for _ in SRC_X for x in rcv_x],
    receiver_shot_index=[s for s in range(len(SRC_X)) for _ in range(N_RCV)],
    nt=NT, dt=DT,
)
wavelets = shared_wavelet(ricker(NT, DT, F0, causal=True, dtype=torch.float64),
                          n_source=survey.n_source)
seis_chain = Acoustic2D(survey, wavelets) @ gardner   # vp -> rho -> seismic

bundle = OperatorBundle({"seismic": seis_chain, "gravity": grav_chain})
bspec = bundle.differentiability
print("PARALLEL OperatorBundle{'seismic': Acoustic2D @ Gardner, "
      "'gravity': Gravity2D @ Gardner}")
print(f"         contract: {bspec.trainable_inputs} -> {bspec.output_keys}  "
      f"level={bspec.level.value}   (union of inputs, channels as outputs)")

# ONE call, every channel
with torch.no_grad():
    obs = bundle(ModelState(tensors={"vp": vp_true}), ctx)
    bg = bundle(ModelState(tensors={"vp": vp_bg}), ctx)
d_obs, gz_obs = obs.fetch("seismic", "gravity")
d_bg, gz_bg = bg.fetch("seismic", "gravity")
print(f"one call -> channels {sorted(obs.data)}: seismic {tuple(d_obs.shape)} "
      f"(trace, time, component), gravity {tuple(gz_obs.shape)}")

# %% 4. One backward, both channels ----------------------------------------
#
# A joint misfit is a sum of channel terms, so backward() sends the
# seismic residual through the time-domain adjoint and the gravity
# residual through Gardner's chain rule, into the SAME vp.grad.
def channel_terms(vp_model: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = bundle(ModelState(tensors={"vp": vp_model}), ctx)
    d_pred, gz_pred = pred.fetch("seismic", "gravity")
    return ((d_pred - d_obs).pow(2).sum() / d_obs.pow(2).sum(),
            (gz_pred - gz_obs).pow(2).sum() / gz_obs.pow(2).sum())

# balance the channels by their gradient norms, so neither drowns the other
def grad_of(term_index: int) -> torch.Tensor:
    leaf = vp_bg.clone().requires_grad_(True)
    channel_terms(leaf)[term_index].backward()
    return leaf.grad.clone()

g_seis, g_grav = grad_of(0), grad_of(1)
w_grav = float(g_seis.norm() / g_grav.norm())

vp_leaf = vp_bg.clone().requires_grad_(True)
seis_term, grav_term = channel_terms(vp_leaf)
(seis_term + w_grav * grav_term).backward()      # ONE backward, both channels
joint_grad = vp_leaf.grad.clone()
print(f"channel balance: gravity weighted x{w_grav:.1f} "
      "so the two gradient norms match")
print(f"one backward -> both channels land in the same tensor: "
      f"vp.grad {tuple(joint_grad.shape)}, "
      f"max |g| = {float(joint_grad.abs().max()):.3e}")

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(2, 2)
extent = (0.0, NX * DX, NZ * DZ, 0.0)


def mode_label(ax, text):
    ax.text(-0.32, 0.5, text, transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=12, fontweight="semibold")


# --- row 1: SERIAL, stage by stage ---
image = field(axes[0, 0], vp_true.numpy(), extent=extent, cmap=CMAP_VELOCITY,
              title="Input state: vp + the two acquisitions",
              ylabel="Depth [m]")
axes[0, 0].plot(rcv_x.numpy(), [25.0] * N_RCV, "v", ms=4, color="white",
                mec="k", mew=0.4, ls="none", label="Geophones")
axes[0, 0].plot(SRC_X, [25.0] * len(SRC_X), "*", ms=13, color=C_RECOVERED,
                mec="k", ls="none", label="Shots")
axes[0, 0].plot(xs.numpy(), torch.zeros_like(xs).numpy(), "^", ms=4,
                color="white", mec="k", mew=0.4, ls="none",
                label="Gravity stations")
axes[0, 0].legend(loc="lower left", framealpha=0.85)
shared_colorbar(fig, image, axes[0, 0], "vp [m/s]")
mode_label(axes[0, 0], "SERIAL  @")

# Gravity2D returns elevation-up gz (attraction points down => negative);
# geophysical displays are down-positive, so negate for the eye
axes[0, 1].plot(xs.numpy(), (-gz_obs * 1e5).numpy(), color=C_SERIES, lw=2.4)
axes[0, 1].set(title="-> Gardner -> rho -> Gravity2D -> $g_z$",
               xlabel="Distance [m]",
               ylabel="Downward attraction [mGal]")

# --- row 2: PARALLEL, channels and the shared gradient ---
# a real shot gather, with the usual t^2 spherical-divergence gain applied
# for display only. Signed data gets the gallery's diverging ramp, on limits
# symmetric about zero, like every other signed quantity here.
t_axis = torch.arange(NT, dtype=torch.float64) * DT
SHOW = 1                                          # the middle shot
dense = survey.to_dense(d_obs)[0]                 # (n_shot, nt, n_rcv, n_comp)
gather = (dense[SHOW, :, :, 0] * t_axis[:, None] ** 2).numpy()
clip = float(abs(gather).max()) * 0.15
field(axes[1, 0], gather, cmap=CMAP_SEISMIC, vmin=-clip, vmax=clip,
      extent=(float(rcv_x[0]), float(rcv_x[-1]), NT * DT, 0.0),
      interpolation="bilinear",
      title=f'Channel "seismic" - shot {SHOW + 1} of {len(SRC_X)}',
      xlabel="Receiver x [m]", ylabel="Time [s]")
axes[1, 0].plot([SRC_X[SHOW]], [0.03], "*", ms=13, color=C_RECOVERED, mec="k")
mode_label(axes[1, 0], "PARALLEL  Bundle")

lim = symmetric_limits(joint_grad.numpy(), quantile=0.90)
image = field(axes[1, 1], joint_grad.numpy(), extent=extent,
              cmap=CMAP_ANOMALY, vmin=lim[0], vmax=lim[1],
              title="Joint gradient, both channels",
              xlabel="Distance [m]", ylabel="Depth [m]")
axes[1, 1].add_patch(plt.Rectangle((BODY[1].start * DX, BODY[0].start * DZ),
                                   (BODY[1].stop - BODY[1].start) * DX,
                                   (BODY[0].stop - BODY[0].start) * DZ,
                                   fill=False, ec=C_TRUTH, ls="--", lw=1.3))
shared_colorbar(fig, image, axes[1, 1],
                r"$\partial$ misfit / $\partial v_p$")

fig.savefig(OUT / "02_operator_composition.png")
print(f"saved {OUT / '02_operator_composition.png'}")
plt.show()
