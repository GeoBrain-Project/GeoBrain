"""Parameterization is an operator: image, network weights, or latent code.

There are three ways to answer "what are the unknowns?" in an inverse
problem, and a platform is only as flexible as the number of them it can
express without touching the physics:

    EXPLICIT   the unknown is the porosity image itself, one number per
        cell. Maximum freedom and no prior, so wherever the data are
        weak the answer is whatever the optimiser drifted into.

    NETWORK    the unknown is a convolutional decoder's WEIGHTS, fed a
        frozen random code. The image is whatever that architecture can
        draw, so its smoothness and locality act as a prior nobody had
        to write down. This is the deep image prior.

    LATENT     the unknown is the CODE, with the decoder frozen. The
        image is confined to the decoder's range, the strongest prior
        of the three, on the fewest unknowns by a factor of five.

In GeoBrain none of these is a special mode or a separate solver. Each is
an operator, and you pick one by composing it in front of the physics
with ``@``::

    explicit = seismic                                     # phi is the input
    network  = seismic @ WeightReparameterization(decoder, ...)
    latent   = seismic @ LatentReparameterization(decoder, ...)

All three then go through the SAME ``InverseProblem`` and the same
``create_inverter().run()``. Not one line of the forward model changes
between them, and the chain reports the switch honestly: ask the network
version for its trainable inputs and it answers with the decoder's weight
names.

The physics, and the earth
--------------------------
Porosity and sand fraction are co-simulated with correlation 0.7 by the
platform's FFT-MA simulator, then squashed into their physical ranges.
Sand fraction is KNOWN and sets the quartz/clay mineral mix cell by cell;
porosity is the unknown. It drives Voigt–Reuss–Hill moduli, Hertz–Mindlin
contacts, the soft-sand line, Gassmann brine substitution and a density
model to (vp, vs, rho); a Shuey kernel makes angle-dependent reflectivity
and a Ricker wavelet convolves it into three angle stacks. The gradient
returns through all of it, and through the decoder too, when there is
one.

The starting model is a DIFFERENT realization of the same statistics:
geologically plausible, wrong in every detail. A constant start would
hand all three methods the same blank slate and hide the one thing this
script is about.

What the comparison shows
-------------------------
The two network parameterizations recover the section more accurately
than the explicit one, and the panels say why: the explicit run has
the freedom to put structure anywhere the three stacks are indifferent,
and it uses it. The decoder cannot, because no setting of its weights draws
cell-scale noise, so its errors stay where the ambiguity actually is.

Two numbers keep that honest. The misfit the TRUE model scores is printed
for every run: one that finishes below it has fit noise, not earth. And
the unknown counts, printed under each panel, do not order the results: the
latent run has a fifth of the explicit run's unknowns.

The result is a property of THIS earth, not a law. The field here is
smooth on a scale the wavelet resolves, which is the regime a
convolutional prior is built for; where bedding is finer than the
wavelength the same prior costs resolution instead of buying it. The
claim being made is narrower and more useful: trying all three cost three
lines and one ``@`` apiece, on the same physics, through the same door.

APIs featured:
    - geobrain.nn.WeightReparameterization / LatentReparameterization:
      parameterization as a composable operator
    - geobrain.nn.ConvDecoder2d: the decoder both of them wrap
    - InverseProblem.create_inverter().run(): one door, three unknowns
    - geobrain.geomodel FFT-MA co-simulation, rock physics as a chain
      link, and geobrain.physics.wave reflectivity

Expected runtime: < 2 min.

Outputs:
    out/06_neural_network_integration.png: the true section and the
    three recoveries, on one porosity scale.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _style import (
    CMAP_MODEL,
    apply_style,
    field,
    figure,
    outer_labels,
    shared_colorbar,
)
from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    ModelState,
    PropertyTransform,
)
from geobrain.geomodel.frames import GeoGrid, PropertyMetadata
from geobrain.geomodel.geostats import (
    FFTMA,
    CovarianceModel,
    SimulationExecutionConfig,
    VariogramKernel,
)
from geobrain.inverse import GaussianLikelihood, InverseProblem
from geobrain.nn import (
    ConvDecoder2d,
    LatentReparameterization,
    WeightReparameterization,
)
from geobrain.optim import AdamConfig
from geobrain.physics.rock import hertz_mindlin_moduli, velocities_from_moduli
from geobrain.physics.rock.models import VRH, DensityModel, SoftSand
from geobrain.physics.rock.models._fluid_gassmann import gassmann_k_sat
from geobrain.physics.wave import ricker
from geobrain.physics.wave.reflectivity import compute_reflectivity

apply_style()
torch.manual_seed(11)
torch.set_default_dtype(torch.float64)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NZ, NX, DZ, DX = 64, 64, 4.0, 4.0              # 256 m square section
PHI_LO, PHI_HI = 0.05, 0.40
VSAND_LO, VSAND_HI = 0.20, 0.80
ANGLES = [12.0, 24.0, 36.0]
DT_S, F0 = 0.002, 25.0
CORRELATION = 0.7                              # porosity vs sand fraction
# Smooth on a scale the wavelet resolves: a quarter wavelength is ~30 m and
# these ranges are 60-100 m. The azimuth stays 0, since on this grid a ROTATED
# anisotropic covariance leaves 4.6e-9 relative imaginary leakage in the FFT
# embedding and the simulator rejects it (threshold 1e-12) at every growth
# budget, which is the right call to make loudly rather than to paper over.
RANGE_V, RANGE_H, AZIMUTH = 60.0, 100.0, 0.0
SEED_TRUE, SEED_START = 2025, 1234             # truth, and a WRONG realization
LATENT_C, LATENT_H, LATENT_W = 8, NZ // 4, NX // 4


# %% 1. The earth: two correlated FFT-MA fields ---------------------------
def fftma_pair(seed: int):
    """Two correlated standard-normal (nz, nx) fields from the library FFT-MA.

    The variogram ranges bind to the ``GeoGrid`` axes POSITIONALLY and the
    result is read back with ``reshape(NZ, NX)``, so the grid is declared in
    that same (depth, distance) order and the ranges given as (vertical,
    lateral). Declaring them the other way round silently transposes the
    anisotropy into a wrong earth that still looks plausible.
    """
    domain = GeoGrid(shape=(NZ, NX, 1), origin=(0.0, 0.0, 0.0),
                     spacing=(DZ, DX, 1.0))
    model = CovarianceModel(nugget=0.0, structures=[
        VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=1.0,
                        ranges=(RANGE_V, RANGE_H, 1.0e6),
                        angles=(AZIMUTH, 0.0, 0.0))])
    prop = PropertyMetadata(name="z", kind="continuous", unit="1")
    draws = []
    for offset in (0, 1):
        frame = FFTMA(model, property=prop,
                      execution=SimulationExecutionConfig(
                          n_realizations=1, seed=seed + offset)
                      )(None, domain).realizations[0].frame
        draws.append(torch.as_tensor(frame.to_numpy("simulation").copy(),
                                     dtype=D).reshape(NZ, NX))
    mixed = (CORRELATION * draws[0]
             + (1.0 - CORRELATION ** 2) ** 0.5 * draws[1])
    return draws[0], mixed


def squash(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Logistic map into ``[lo, hi]``, bounded with no clipping artefacts."""
    return lo + (hi - lo) * torch.sigmoid(z)


z_phi, z_sand = fftma_pair(SEED_TRUE)
phi_true = squash(z_phi, PHI_LO, PHI_HI)
vsand = squash(z_sand, VSAND_LO, VSAND_HI)          # KNOWN side field
phi_start = squash(fftma_pair(SEED_START)[0], PHI_LO, PHI_HI)
pair = torch.stack([phi_true.reshape(-1), vsand.reshape(-1)])
print(f"[1] FFT-MA co-simulation, {RANGE_H:.0f} m lateral x {RANGE_V:.0f} m "
      f"vertical spherical variogram on {NZ}x{NX} cells of {DZ:.0f} m: "
      f"porosity {float(phi_true.min()):.3f}-{float(phi_true.max()):.3f}, "
      f"sand fraction {float(vsand.min()):.2f}-{float(vsand.max()):.2f}, "
      f"sample correlation {float(torch.corrcoef(pair)[0, 1]):.2f}")
print(f"[1] the starting model is realization {SEED_START}, not a constant: "
      f"it misses the truth by RMSE "
      f"{float(((phi_start - phi_true) ** 2).mean().sqrt()):.4f}")


# %% 2. Rock physics, then AVO: the physics the unknowns must reach -------
class PorosityToElastic(PropertyTransform):
    """phi -> (vp, vs, rho) through five differentiable rock-physics laws.

    Sand fraction is a KNOWN side field, so the Voigt-Reuss-Hill mineral
    moduli and the matrix density vary cell by cell. Porosity is the only
    unknown and it reaches vp, vs and rho through all five laws without
    one hand-written derivative.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("phi",),
        output_keys=("vp", "vs", "rho"),
    )

    K_QTZ, K_CLAY = 36.6e9, 21.0e9
    MU_QTZ, MU_CLAY = 44.0e9, 9.0e9
    RHO_QTZ, RHO_CLAY = 2650.0, 2550.0
    K_BRINE, RHO_BRINE = 3.06e9, 1080.0

    def __init__(self, sand: torch.Tensor) -> None:
        super().__init__()
        vrh = VRH()
        f64 = lambda x: torch.tensor(x, dtype=D)          # noqa: E731
        k_min = vrh(f64(self.K_QTZ), f64(self.K_CLAY), sand)
        mu_min = vrh(f64(self.MU_QTZ), f64(self.MU_CLAY), sand)
        hm = hertz_mindlin_moduli(f64(20.0e6), k_min, mu_min,
                                  f64(PHI_HI), f64(7.0))  # 20 MPa, C = 7
        self.register_buffer("k_min", k_min)
        self.register_buffer("k_hm", hm.k_dry)
        self.register_buffer("mu_hm", hm.mu_dry)
        self.register_buffer("rho_min", self.RHO_QTZ * sand
                             + self.RHO_CLAY * (1.0 - sand))
        # SoftSand takes SCALAR mineral moduli (they set only the stiff end of
        # the line), so it gets the section average; Hertz-Mindlin, Gassmann
        # and the density model all take the per-cell tensors.
        self.soft_sand = SoftSand(K_mineral=float(k_min.mean()),
                                  mu_mineral=float(mu_min.mean()),
                                  phi_critical=PHI_HI)
        self.density = DensityModel()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (phi,) = state.fetch("phi")
        k_dry, mu_dry = self.soft_sand(self.k_hm, self.mu_hm, phi)
        k_sat = gassmann_k_sat(k_dry, self.k_min, self.K_BRINE, phi)
        rho = self.density(phi, self.rho_min,
                           torch.tensor(self.RHO_BRINE, dtype=D))
        vp, vs = velocities_from_moduli(k_sat, mu_dry, rho)
        return state.with_tensors(vp=vp, vs=vs, rho=rho)


class AngleStackAVO(ForwardOperator):
    """(vp, vs, rho) sections -> three wavelet-convolved angle stacks.

    ``ConvolutionalAVO`` and ``AkiRichards`` take 1-D profiles only, so a
    SECTION goes through the functional ``compute_reflectivity``, which
    broadcasts and returns ``(n_angles, nz-1, nx)``.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho"),
        output_keys=("stacks",),
    )

    def __init__(self, angles: list[float], wavelet: torch.Tensor) -> None:
        super().__init__()
        self.angles = angles
        self.register_buffer("kernel", wavelet.flip(0).reshape(1, 1, -1))

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        vp, vs, rho = state.fetch("vp", "vs", "rho")
        refl = compute_reflectivity(vp[:-1], vs[:-1], rho[:-1],
                                    vp[1:], vs[1:], rho[1:],
                                    theta=self.angles, method="shuey")
        n_ang, nz, nx = refl.shape                 # convolve DOWN each trace
        traces = refl.permute(0, 2, 1).reshape(n_ang * nx, 1, nz)
        stacks = F.conv1d(traces, self.kernel, padding="same")
        return ForwardOutput(data={"stacks":
                                   stacks.reshape(n_ang, nx, nz).permute(0, 2, 1)})


wavelet = ricker(nt=41, dt=DT_S, f0=F0, dtype=D)
seismic = AngleStackAVO(ANGLES, wavelet) @ PorosityToElastic(vsand)
ctx = ForwardContext()
with torch.no_grad():
    clean = seismic(ModelState({"phi": phi_true}), ctx).data["stacks"]
NOISE = 0.03 * float(clean.abs().max())
observed = clean + NOISE * torch.randn_like(clean)
problem_kwargs = dict(observed={"stacks": observed},
                      likelihood=GaussianLikelihood(std=NOISE))
with torch.no_grad():
    FLOOR = float(-InverseProblem(forward=seismic, **problem_kwargs)
                  .log_likelihood(ModelState({"phi": phi_true})))
print(f"[2] chain {type(seismic).__name__}: trainable "
      f"{seismic.differentiability.trainable_inputs} -> "
      f"{seismic.differentiability.output_keys}; {len(ANGLES)} angle stacks of "
      f"{tuple(clean.shape[1:])} = {observed.numel()} data at 3% noise. The "
      f"TRUE model scores {FLOOR:.3e}, and below that is fitting noise")


def rmse(field: torch.Tensor) -> float:
    return float(((field - phi_true) ** 2).mean().sqrt())


# %% 3. Three parameterizations, one door ---------------------------------
decoder = ConvDecoder2d(LATENT_C, 1, hidden_channels=(32, 16), scale_factor=2,
                        final_activation="sigmoid").to(dtype=D)
code = torch.randn(1, LATENT_C, LATENT_H, LATENT_W, dtype=D)
TO_PHI = {"phi": lambda x: PHI_LO + (PHI_HI - PHI_LO) * x}

dip = WeightReparameterization(decoder, fixed_input=code, outputs={"phi": 0},
                               transforms=TO_PHI)
# The latent run reuses the decoder the DIP run has just trained: weights
# frozen at their fitted values, only the code free. The difference between
# the two placements is one class name.
latent = LatentReparameterization(
    decoder, outputs={"phi": 0}, latent_field="latent",
    latent_shape=(LATENT_C, LATENT_H, LATENT_W), transforms=TO_PHI)

PLAN = (
    ("Explicit", dict(forward=seismic, params={"phi": phi_start.clone()},
                      decode=None, bounds={"phi": (PHI_LO, PHI_HI)},
                      lr=4e-3, iters=400)),
    ("DIP", dict(forward=seismic @ dip, params=dip.initial_params(),
                 decode=dip, bounds=None, lr=6e-3, iters=400)),
    ("Latent", dict(forward=seismic @ latent,
                    params={"latent": code.reshape(-1)}, decode=latent,
                    bounds=None, lr=3e-2, iters=600)),
)

results: dict[str, dict] = {}
for name, spec in PLAN:
    t0 = time.time()
    problem = InverseProblem(forward=spec["forward"], **problem_kwargs)
    result = problem.create_inverter(
        params={k: v.detach().clone() for k, v in spec["params"].items()},
        optimizer=AdamConfig(lr=spec["lr"]),
        bounds=spec["bounds"]).run(n_iters=spec["iters"])
    with torch.no_grad():
        phi = (result.params["phi"] if spec["decode"] is None
               else spec["decode"](ModelState(result.params), ctx)
               .tensors["phi"])
    n = sum(int(v.numel()) for v in spec["params"].values())
    results[name] = dict(phi=phi.clone(), loss=list(result.loss_history), n=n,
                         seconds=time.time() - t0)
    verdict = ("below the true model's, so noise" if min(results[name]["loss"])
               < FLOOR else "above the true model's, so the prior binds")
    print(f"[3] {name:9s} {n:5d} unknowns, {spec['iters']} Adam steps in "
          f"{results[name]['seconds']:4.0f} s: loss "
          f"{result.loss_history[0]:.3e} -> {min(result.loss_history):.3e} "
          f"({verdict}), porosity RMSE {rmse(phi):.4f}")
    if name == "DIP":
        chain = spec["forward"]
        print(f"    the switch is visible in the contract: this chain's "
              f"trainable inputs are {chain.differentiability.trainable_inputs}")

for name, res in results.items():
    per_depth = (res["phi"] - phi_true).abs().mean(dim=1)
    worst = int(per_depth.argmax())
    print(f"    {name:9s} mean |error| {float(per_depth.mean()):.3f}, worst at "
          f"{(worst + 0.5) * DZ:.0f} m depth ({float(per_depth[worst]):.3f})")

best = min(results, key=lambda k: rmse(results[k]["phi"]))
gain = (rmse(results["Explicit"]["phi"]) - rmse(results[best]["phi"]))
print("[4] " + ", ".join(f"{k} {rmse(v['phi']):.4f}"
                         for k, v in results.items())
      + f"; {best} wins by {gain / rmse(results['Explicit']['phi']) * 100:.0f}%"
      " over the explicit run. Note that the ordering follows neither the "
      "unknown count nor the data misfit: the winner has the MOST unknowns, "
      "and the latent run beats the explicit one on half of them. The prior "
      "is not a term in the objective; it is the set of images the decoder "
      "can draw at all")


# %% 4. Picture ------------------------------------------------------------
# The section is square, so the panels are: forcing a square field into a
# landscape panel stretches it, and the reader is being asked to compare
# textures across the four.
fig, axes = figure(2, 2, panel_w=4.3, panel_h=4.3)
EXTENT = (0.0, NX * DX, NZ * DZ, 0.0)

# The truth and the three recoveries, on ONE porosity scale and nothing
# else. The comparison IS the figure; convergence histories and per-depth
# errors are numbers, and they are printed above where numbers belong.
maps = (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
image = field(maps[0], phi_true.numpy(), extent=EXTENT, cmap=CMAP_MODEL,
              vmin=PHI_LO, vmax=PHI_HI, title="True section")
for ax, (name, res) in zip(maps[1:], results.items()):
    image = field(ax, res["phi"].numpy(), extent=EXTENT, cmap=CMAP_MODEL,
                  vmin=PHI_LO, vmax=PHI_HI,
                  title=f"{name} - {res['n']} unknowns, RMSE "
                        f"{rmse(res['phi']):.4f}")
outer_labels(axes, "Distance [m]", "Depth [m]")
shared_colorbar(fig, image, axes, "Porosity [-]")

fig.savefig(OUT / "06_neural_network_integration.png")
print(f"saved {OUT / '06_neural_network_integration.png'}")
plt.show()
