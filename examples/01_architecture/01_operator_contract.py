"""The operator contract: what every GeoBrain computation promises.

GeoBrain has exactly two kinds of physics node, and everything in the
platform is built by wiring them up:

- a ``PropertyTransform`` maps ``ModelState -> ModelState``
  (rock physics, parameterizations, change of variables);
- a ``ForwardOperator`` maps ``ModelState -> ForwardOutput``
  (a simulation producing named data channels).

Both DECLARE what they compute before they run, in a
``DifferentiabilitySpec``: which input fields they differentiate through,
which output keys they produce, and what gradient mechanism backs them.
The base class checks the declaration on every call, so a wrong input is
a loud, machine-readable error, never a silently wrong number.

This script walks the whole contract on a seismic well tie, and spends
its second half on the parts you only meet when something goes wrong:
the immutability of ``ModelState``, the three compartments of a
``ForwardOutput``, and how to read a structured ``GeoBrainError``.

APIs featured:
    - geobrain.core.ModelState (immutable; .fetch, .with_tensors)
    - geobrain.core.DifferentiabilitySpec (the declared contract)
    - geobrain.core.ForwardOutput (data / fields / metadata)
    - geobrain.core.ForwardContext.of (typed per-call configuration)
    - structured errors: object_name / field / expected / actual / hint

Expected runtime: < 10 s.

Outputs:
    out/01_operator_contract.png: the well log, Gardner density, and the
    synthetic trace the contract produced.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import C_SERIES, C_TRUTH, PALETTE, apply_style, figure
from geobrain.core import ForwardContext, GeoBrainError, MissingFieldError, ModelState
from geobrain.physics.rock.models import GardnerOperator
from geobrain.physics.wave import Convolutional1D, ricker

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. A model lives in a ModelState --------------------------------------
#
# The state is an immutable name -> tensor mapping. Operators never mutate
# it; transforms return a NEW state. Here the model is a blocked well log.
nz, dz = 180, 5.0
z = torch.arange(nz, dtype=torch.float64) * dz

vp = torch.full((nz,), 2350.0, dtype=torch.float64)
vp[60:] = 2650.0                      # a deeper, faster unit
vp[130:] = 3050.0                     # basement trend
vp[92:110] = 2280.0                   # slow gas-charged sand

state = ModelState(tensors={"vp": vp})
(vp_view,) = state.fetch("vp")        # ordered, checked access
print(f"[1] state fields {sorted(state.tensors)}   vp in "
      f"[{float(vp_view.min()):.0f}, {float(vp_view.max()):.0f}] m/s")

# %% 2. A PropertyTransform: ModelState -> ModelState ----------------------
#
# Gardner's relation rho = a * vp^b supplies the missing half of acoustic
# impedance. The contract is declared BEFORE any call runs:
gardner = GardnerOperator()
spec = gardner.differentiability
print(f"[2] GardnerOperator: level={spec.level.value}  "
      f"trainable={spec.trainable_inputs} -> outputs={spec.output_keys}")

state_rho = gardner(state)            # returns a NEW state
(rho,) = state_rho.fetch("rho")
print(f"    after the call: input state still {sorted(state.tensors)}, "
      f"result state {sorted(state_rho.tensors)}  (immutable, vp passed through)")

# %% 3. A ForwardOperator: ModelState -> ForwardOutput ---------------------
#
# Convolutional1D computes impedance Z = vp * rho, interface reflectivity
# R = dZ / sumZ, and convolves R with a source wavelet. The wavelet is
# ACQUISITION, not model, so it travels in the typed ForwardContext, the
# same door that carries meshes and time stepping for the big operators.
conv = Convolutional1D()
cspec = conv.differentiability
print(f"[3] Convolutional1D: level={cspec.level.value}  "
      f"trainable={cspec.trainable_inputs} -> outputs={cspec.output_keys}")

wavelet = ricker(41, dz / 2500.0, 40.0, dtype=torch.float64)   # ~40 Hz
ctx = ForwardContext.of(wavelet=wavelet)
pred = conv(state_rho, ctx)

# A ForwardOutput has three compartments, and the difference matters:
#   data:      the observable channels the contract promised
#   fields:    auxiliary tensors an operator chooses to expose (wavefields,
#              illumination, sensitivities); never part of the contract
#   metadata: units, provenance, per-channel notes
(trace,) = pred.fetch("trace")
print(f"    data     {sorted(pred.data)}  <- the declared channels")
print(f"    fields   {sorted(pred.fields) or 'empty here'}  <- optional extras "
      "(FDTD facades put wavefields and illumination here)")
print(f"    metadata {sorted(pred.metadata) or 'empty here'}  <- optional units "
      "and provenance (the potential-field operators fill this in)")

# %% 4. The contract is enforced, loudly -----------------------------------
#
# Violate it from either side and a structured error fires BEFORE any
# physics runs. Every GeoBrainError carries machine-readable fields, so a
# failure is data your code (or an agent) can act on, not a string.
def report(err: GeoBrainError, what: str) -> None:
    print(f"    {what}")
    print(f"      object={err.object_name!r}  field={err.field!r}  "
          f"expected={err.expected!r}")

print("[4] the three ways to break the contract:")
try:
    conv(ModelState(tensors={"vp": vp}), ctx)          # state missing rho
except MissingFieldError as err:
    report(err, "state is missing a declared input:")

try:
    conv(state_rho, ForwardContext())                   # context missing wavelet
except MissingFieldError as err:
    report(err, "context layer was never filled:")

try:
    conv(ModelState(tensors={"vp": vp, "rho": rho[:20]}), ctx)   # wrong shape
except GeoBrainError as err:
    report(err, "input violates the operator's own precondition:")
    # every GeoBrainError is also a dict, which is what makes a failure
    # something your code (or an agent) can branch on, not a string
    print(f"      as data: {sorted(err.to_dict())}")

# %% 5. One picture of the whole arrow chain -------------------------------
fig, axes = figure(1, 3, panel_w=3.3, panel_h=6.4, sharey=True)

axes[0].plot(vp.numpy(), z.numpy(), color=C_SERIES, lw=1.8)
axes[0].invert_yaxis()
axes[0].set(title="ModelState input: vp", xlabel="vp [m/s]",
            ylabel="Depth [m]")

axes[1].plot(rho.numpy(), z.numpy(), color=PALETTE[1], lw=1.8)
axes[1].set(title="PropertyTransform: Gardner ρ",
            xlabel=r"$\rho$ [kg/m$^3$]")

axes[2].plot(trace.numpy(), z.numpy(), color=C_TRUTH, lw=1.1)
axes[2].fill_betweenx(z.numpy(), 0.0, trace.numpy(),
                      where=(trace > 0).numpy(), color=C_TRUTH, alpha=0.75)
axes[2].axhspan(92 * dz, 110 * dz, color="orange", alpha=0.15,
                label="Gas sand")
axes[2].legend(loc="lower right")
axes[2].set(title="ForwardOutput: synthetic trace", xlabel="Amplitude")

fig.savefig(OUT / "01_operator_contract.png")
print(f"saved {OUT / '01_operator_contract.png'}")
plt.show()
