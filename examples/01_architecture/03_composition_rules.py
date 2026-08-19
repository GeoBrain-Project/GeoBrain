"""Composition rules: what the platform derives, and what it refuses.

Operators compose two ways: in SERIES with ``@`` and in PARALLEL with an
:class:`OperatorBundle`. Both produce a new Operator whose contract is
DERIVED, never re-declared by you, and both refuse compositions that
cannot be honoured. This script is the rulebook, demonstrated:

RULE 1  ``B @ A`` reads like ``(B ∘ A)``: the RIGHTMOST link runs first.
RULE 2  A chain's trainable inputs come from the ENTRY link, its output
        keys from the TERMINAL link.
RULE 3  A chain's differentiability level is the WEAKEST of its members:
        one implicit-VJP link makes the whole chain implicit-VJP.
RULE 4  Only the LAST link may be a ForwardOperator. Everything before it
        must map state -> state, or the chain is refused at build time.
RULE 5  Mid-chain links may declare inputs that are produced upstream:
        Aki-Richards asks for vp, vs AND rho, but the chain ignores those
        declarations and parameterizes on the entry link alone.
RULE 6  A bundle's trainable inputs are the UNION of its channels', its
        output keys are the CHANNEL NAMES, and a chain drops into a
        bundle unchanged, because composition composes.

The physics is the standard elastic cascade: one velocity log drives
Gardner density and the Castagna mudrock line, and Aki–Richards turns the
triple into angle-dependent reflectivity.

APIs featured:
    - Operator.__matmul__ (chain) and the derived DifferentiabilitySpec
    - geobrain.core.OperatorBundle (parallel) and its derived spec
    - the structured refusal when a ForwardOperator sits mid-chain
    - torch.autograd through a three-link chain

Expected runtime: < 10 s.

Outputs:
    out/03_composition_rules.png: the cascade stage by stage, the
    derived contracts, and one gradient through all three links.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import C_SERIES, PALETTE, apply_style, figure
from geobrain.core import ForwardContext, GeoBrainError, ModelState, OperatorBundle
from geobrain.physics.rock import Gardner            # family-root FORWARD facade
from geobrain.physics.rock.models import CastagnaOperator, GardnerOperator
from geobrain.physics.wave import AkiRichards

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

ANGLES = [float(a) for a in range(0, 41, 5)]   # 0..40 deg

# %% 1. RULES 1-3, 5: build the cascade ------------------------------------
#
# Right-to-left: Gardner runs first, then Castagna, then Aki-Richards.
# Print each link's own declaration next to the chain's derived one: the
# mid-chain asks for vs and rho, the chain asks you for vp alone.
links = (GardnerOperator(), CastagnaOperator(), AkiRichards(angles_deg=ANGLES))
chain = links[2] @ links[1] @ links[0]
print(f"[1] {chain}")
for link in links:
    ls = link.differentiability
    print(f"    link {type(link).__name__:18s} declares "
          f"{str(ls.trainable_inputs):24s} -> {ls.output_keys}")

spec = chain.differentiability
print(f"    RULE 2  chain trainable = {spec.trainable_inputs}  (entry link only)")
print(f"    RULE 2  chain outputs   = {spec.output_keys}  (terminal link only)")
print(f"    RULE 3  chain level     = {spec.level.value}  (weakest of the three)")
print("    RULE 5  the mid-chain vs/rho declarations above are ignored: "
      "they are produced upstream, not asked of you")

# %% 2. RULE 4: the refusal ------------------------------------------------
#
# ``Gardner`` at the rock family root is a capability-reporting FORWARD
# operator, not the composable transform. Put it mid-chain and the chain
# refuses to build, with a structured error naming the offending slot.
print("[2] RULE 4  a ForwardOperator may not sit mid-chain:")
try:
    _refused = AkiRichards(angles_deg=ANGLES) @ Gardner()
except GeoBrainError as err:
    print(f"    refused: {err.args[0]}")
    print(f"    object={err.object_name!r} field={err.field!r} "
          f"expected={err.expected!r}")
    print("    fix: use geobrain.physics.rock.models.GardnerOperator, the "
          "PropertyTransform")

# %% 3. Run the cascade ----------------------------------------------------
nz, dz = 120, 5.0
z = torch.arange(nz, dtype=torch.float64) * dz
vp = (2300.0 + 1.4 * z).clone()
vp[30:46] += 260.0                                # a hard streak
vp[46:70] -= 340.0                                # a slow, gas-charged sand
vp[92:] += 180.0                                  # a deeper unit
vp = vp.requires_grad_(True)

pred = chain(ModelState(tensors={"vp": vp}), ForwardContext())
(refl,) = pred.fetch("reflectivity")              # (n_interfaces, n_angles)
print(f"[3] forward: vp({nz}) -> reflectivity {tuple(refl.shape)}")

# the intermediates are one call away when you want to see them
mid = CastagnaOperator()(GardnerOperator()(
    ModelState(tensors={"vp": vp.detach()})))
rho, vs = mid.fetch("rho", "vs")

# %% 4. RULE 6: a chain drops into a bundle unchanged ----------------------
bundle = OperatorBundle({
    "near": AkiRichards(angles_deg=[0.0, 10.0]) @ CastagnaOperator()
            @ GardnerOperator(),
    "far": AkiRichards(angles_deg=[30.0, 40.0]) @ CastagnaOperator()
           @ GardnerOperator(),
})
bspec = bundle.differentiability
print(f"[4] RULE 6  bundle of two chains: trainable={bspec.trainable_inputs} "
      f"(union) -> outputs={bspec.output_keys} (channel names), "
      f"level={bspec.level.value}")

# %% 5. One backward through all three links -------------------------------
loss = refl.pow(2).sum()
(grad,) = torch.autograd.grad(loss, vp)
print(f"[5] one backward crosses Aki-Richards, Castagna and Gardner: "
      f"||d loss/d vp|| = {float(grad.norm()):.3e}")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(1, 4, panel_w=3.5, panel_h=6.4)

axes[0].plot(vp.detach().numpy(), z.numpy(), color=C_SERIES, lw=1.8)
axes[0].invert_yaxis()
axes[0].set(title="Entry link input: vp", xlabel="vp [m/s]",
            ylabel="Depth [m]")

# The two mid-chain products are in different units, so they get two axes
# rather than one axis labelled vaguely enough to cover both. Each axis is
# tinted to its curve, which is what says which scale to read a line on.
axes[1].plot(vs.numpy(), z.numpy(), color=PALETTE[2], lw=1.8)
axes[1].invert_yaxis()
axes[1].set(title="Mid-chain products", xlabel="vs from Castagna [m/s]",
            ylabel="Depth [m]")
axes[1].xaxis.label.set_color(PALETTE[2])
axes[1].tick_params(axis="x", colors=PALETTE[2])
twin = axes[1].twiny()
twin.plot(rho.numpy(), z.numpy(), color=PALETTE[1], lw=1.8)
# The second scale goes BELOW the first, not above it, so that the panel
# title stays on the same line as every other title in the figure.
twin.xaxis.set_ticks_position("bottom")
twin.xaxis.set_label_position("bottom")
twin.spines["bottom"].set_position(("outward", 38))
twin.set_xlabel("ρ from Gardner [kg/m$^3$]", color=PALETTE[1])
twin.tick_params(axis="x", colors=PALETTE[1])
twin.grid(False)

# AVO is an ANGLE effect, so plot R against angle, one curve per interface
refl_np = refl.detach()
strongest = refl_np.abs().max(dim=1).values.argsort(descending=True)[:3]
for rank, idx in enumerate(sorted(int(i) for i in strongest)):
    axes[2].plot(ANGLES, refl_np[idx].numpy(), "o-", ms=4, lw=1.8,
                 color=PALETTE[rank],
                 label=f"Interface at {float(z[idx + 1]):.0f} m")
axes[2].axhline(0.0, color="gray", lw=0.8)
axes[2].set(title="Terminal link output: R(θ)",
            xlabel="Incidence angle [°]", ylabel="Reflectivity")
axes[2].legend(loc="best")

axes[3].plot(grad.numpy(), z.numpy(), color=PALETTE[0], lw=1.8)
axes[3].axvline(0.0, color="gray", lw=0.8)
axes[3].invert_yaxis()
axes[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
axes[3].set(title=r"One backward(): $\partial\,\|R\|^2/\partial v_p$",
            xlabel="Gradient", ylabel="Depth [m]")

fig.savefig(OUT / "03_composition_rules.png")
print(f"saved {OUT / '03_composition_rules.png'}")
plt.show()
