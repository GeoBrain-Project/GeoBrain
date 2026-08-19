"""Differentiability levels: how each operator earns its gradient.

Not every physics gets its gradient the same way, and GeoBrain makes the
difference part of the contract instead of a footnote. Every operator
declares one of five levels, ordered weakest to strongest:

    NON_DIFFERENTIABLE  no gradient at all (data loaders, pickers)
    FORWARD_ONLY        runs forward, refuses to backpropagate
    IMPLICIT_VJP        gradient via the implicit function theorem: the
                        backward pass is one adjoint SOLVE, not a replay
    CUSTOM_VJP          gradient from a hand-derived analytic adjoint
    FULL_AUTOGRAD       plain torch autograd through every operation

The level is not a label: a chain inherits the WEAKEST level of its
members (see 03), and a solver can query it before deciding whether
second-order methods are available.

The heart of the script is the IMPLICIT_VJP seam. Most geophysics ends in
``A(m) u = q``: assemble a system from the model, solve it, sample the
solution. Unrolling the solver would store every iterate; the implicit
function theorem replaces all of that with ONE adjoint solve against the
same operator, ``A^H λ = ∂L/∂u``. We build a 1-D attenuative Helmholtz
problem, differentiate through the solve, and check the result against
central finite differences, then run the shipped audit tool,
``gradient_check``, which localises WHICH cells disagree.

APIs featured:
    - geobrain.core.DifferentiabilityLevel (the five levels)
    - geobrain.core.adjoint.linear_solve_with_adjoint (implicit VJP)
    - geobrain.core.gradient_check (per-entry FD audit)

Expected runtime: < 10 s.

Outputs:
    out/04_differentiability_levels.png: the wavefield, the adjoint
    gradient against finite differences, and the audit scatter.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    C_OBSERVED,
    C_RECOVERED,
    C_SERIES,
    C_TRUTH,
    PALETTE,
    apply_style,
    figure,
)
from geobrain.core import (
    DifferentiabilityLevel,
    ForwardContext,
    ModelState,
    gradient_check,
)
from geobrain.core.adjoint import linear_solve_with_adjoint
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D
from geobrain.physics.rock.models import GardnerOperator
from geobrain.physics.wave import Acoustic2D, AkiRichards, Helmholtz2D

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. What the shipped operators declare ---------------------------------
#
# Rock physics and convolutional kernels are plain torch, so they are
# FULL_AUTOGRAD. The frequency-domain solver ends in a sparse solve, so it
# earns IMPLICIT_VJP. The potential-field operators carry a hand-derived
# adjoint and report CUSTOM_VJP once their execution policy is bound.
survey = PotentialSurvey2D(torch.tensor([[100.0, 1.0], [300.0, 1.0]],
                                        dtype=torch.float64))
declared = {
    "GardnerOperator": GardnerOperator.differentiability.level,
    "AkiRichards": AkiRichards.differentiability.level,
    "Acoustic2D": Acoustic2D.differentiability.level,
    "Helmholtz2D": Helmholtz2D.differentiability.level,
    "Gravity2D": Gravity2D(survey).differentiability.level,
}
print("[1] declared levels")
for name, level in declared.items():
    print(f"    {name:18s} {level.value}")

# %% 2. A model-dependent linear system ------------------------------------
#
# Frequency-domain seismic in one dimension: a 25 Hz monochromatic source
# near the surface, a fast layer at depth, and a touch of attenuation to
# keep the boundary problem well posed. Everything is plain torch, so A
# carries the autograd graph of v.
n, dz = 120, 5.0
freq, alpha = 12.0, 0.05
omega = 2.0 * torch.pi * freq

v = torch.full((n,), 1800.0, dtype=torch.float64)
v[45:75] = 2400.0                               # a fast layer
v = v.requires_grad_(True)

off = torch.full((n - 1,), 1.0 / dz**2, dtype=torch.complex128)
A = (torch.diag((omega / v) ** 2 * (1 + 1j * alpha) - 2.0 / dz**2)
     + torch.diag(off, 1) + torch.diag(off, -1))
q = torch.zeros(n, dtype=torch.complex128)
q[4] = 1.0                                      # source near the surface

u = linear_solve_with_adjoint(A, q)             # forward: one solve
k_rcv = 100                                     # a deep receiver
loss = u.abs().pow(2)[k_rcv]                    # observed energy
loss.backward()                                 # backward: ONE adjoint solve
grad_ad = v.grad.clone()
print(f"[2] solved n={n} at {freq:.0f} Hz; the backward pass was one "
      f"solve of A^H, max |dL/dv| = {float(grad_ad.abs().max()):.3e}")

# %% 3. Trust, then verify: central differences ----------------------------
def loss_of(vel: torch.Tensor) -> float:
    a = (torch.diag((omega / vel) ** 2 * (1 + 1j * alpha) - 2.0 / dz**2)
         + torch.diag(off, 1) + torch.diag(off, -1))
    return float(torch.linalg.solve(a, q).abs().pow(2)[k_rcv])

probes = [10, 30, 50, 65, 90]
eps = 1e-4
grad_fd = []
with torch.no_grad():
    base = v.detach().clone()
    for k in probes:
        vp_, vm_ = base.clone(), base.clone()
        vp_[k] += eps
        vm_[k] -= eps
        grad_fd.append((loss_of(vp_) - loss_of(vm_)) / (2 * eps))
fd = torch.tensor(grad_fd, dtype=torch.float64)
ad = grad_ad[probes]
rel = float(((fd - ad).abs() / ad.abs().clamp(min=1e-30)).max())
print(f"[3] implicit VJP vs central differences on {len(probes)} probes: "
      f"max relative error = {rel:.2e}")

# %% 4. The shipped audit tool ---------------------------------------------
#
# gradient_check probes random entries of any operator with central
# differences and returns the raw numbers, an instrument for "WHICH cells
# have a wrong gradient", not a pass/fail flag.
report = gradient_check(
    GardnerOperator(),
    # a VARYING model, so the probes span a range of gradients
    ModelState(tensors={"vp": 2000.0 + 1400.0 * torch.rand(
        (12, 18), dtype=torch.float64)}),
    ForwardContext(),
    field_name="vp",
    scalar_fn=lambda out: out.tensors["rho"].pow(2).sum(),
    n_probes=16,
)
print(f"[4] gradient_check(GardnerOperator): {report['auto'].numel()} probes, "
      f"max relative error = {report['max_rel_err']:.2e}")

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(2, 2, scale=1.15)
zax = (torch.arange(n, dtype=torch.float64) * dz).numpy()

# (a) the ladder: which shipped operator sits on which rung
LADDER = [level.value for level in DifferentiabilityLevel]
occupants: dict[str, list[str]] = {name: [] for name in LADDER}
for op_name, level in declared.items():
    occupants[level.value].append(op_name)
for row, level_name in enumerate(LADDER):
    axes[0, 0].barh(row, 1.0, height=0.72, color=PALETTE[0],
                    alpha=0.10 + 0.16 * row)
    EMPTY = {"non_differentiable": "data loaders, pickers, format readers",
             "forward_only": "kernels that simulate but cannot be inverted"}
    who = ", ".join(occupants[level_name]) or EMPTY.get(level_name, "none")
    axes[0, 0].text(0.02, row, who, va="center", fontsize=9)
axes[0, 0].set_yticks(range(len(LADDER)),
                      [n.replace("_", " ") for n in LADDER])
axes[0, 0].set_xticks([])
axes[0, 0].set_xlim(0, 1)
axes[0, 0].grid(False)
axes[0, 0].set(title="Declared differentiability level")

# (b) the forward solve
axes[0, 1].plot(u.detach().abs().numpy(), zax, color=C_TRUTH, lw=1.8,
                label="|u|")
axes[0, 1].axhspan(45 * dz, 75 * dz, color="orange", alpha=0.15,
                   label="Fast layer")
axes[0, 1].axhline(k_rcv * dz, color=C_RECOVERED, ls="--", lw=1.4,
                   label="Receiver")
axes[0, 1].invert_yaxis()
axes[0, 1].set(title=f"Forward: one solve of A(v) u = q at {freq:.0f} Hz",
               xlabel="Wavefield amplitude", ylabel="Depth [m]")
axes[0, 1].legend(loc="lower right")

# (c) the backward: one adjoint solve, checked against finite differences
axes[1, 0].plot(grad_ad.numpy(), zax, color=C_SERIES, lw=2.0,
                label="Implicit VJP (one adjoint solve)")
axes[1, 0].plot(fd.numpy(), [zax[k] for k in probes], "o", ms=10, mfc="none",
                mew=2.0, color=C_OBSERVED, label="Central differences")
axes[1, 0].axhspan(45 * dz, 75 * dz, color="orange", alpha=0.15)
axes[1, 0].invert_yaxis()
axes[1, 0].set(title=f"Backward: dL/dv agrees to {rel:.0e} relative",
               xlabel="Gradient", ylabel="Depth [m]")
axes[1, 0].legend(loc="lower right")

# (d) the audit tool, on a different operator entirely
axes[1, 1].plot(report["fd"].numpy(), report["auto"].numpy(), "^", ms=9,
                color=PALETTE[2], label="gradient_check probes")
both = torch.cat([report["fd"], report["auto"]])
lo, hi = float(both.min()), float(both.max())
pad = 0.08 * (hi - lo)
axes[1, 1].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray",
                ls="--", lw=1.0, label="Perfect agreement")
axes[1, 1].set_xlim(lo - pad, hi + pad)
axes[1, 1].set_ylim(lo - pad, hi + pad)
axes[1, 1].set(title="FULL_AUTOGRAD audit (max rel. error "
                     f"{report['max_rel_err']:.0e})",
               xlabel="Finite difference", ylabel="Autograd")
axes[1, 1].legend(loc="upper left")

fig.savefig(OUT / "04_differentiability_levels.png")
print(f"saved {OUT / '04_differentiability_levels.png'}")
plt.show()
