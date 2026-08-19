"""Write your own operator: both extension points, and the gates you get.

Extending GeoBrain means subclassing one of the two arrows and filling in
one hook, ``_forward(state, ctx)``. This script writes both:

- ``LogSlowness`` is a ``PropertyTransform``: log-slowness in, slowness
  out. Twelve lines that buy positivity for free, because the physics
  never sees a negative slowness no matter what the optimizer tries.
- ``StraightRayTomography`` is a ``ForwardOperator``: cross-well travel
  times, ``t_i = Σ_c L[i, c] · slowness[c]``, with the ray matrix built
  once from the context mesh.

What you get for declaring rather than documenting:

    differentiability            the base class validates every call
    requires_mesh_capabilities   a mesh that cannot answer is refused,
                                 with a structured error, before any
                                 physics runs
    the whole downstream stack   composition, InverseProblem, Inverter,
                                 samplers, none of which were told your
                                 operator exists

The payoff is section 5: the custom pair is handed, untouched, to the
same ``create_inverter().run()`` used everywhere else in Part 1.

APIs featured:
    - subclassing geobrain.core.PropertyTransform and ForwardOperator
    - DifferentiabilitySpec, requires_mesh_capabilities, ctx.require_mesh
    - raising GeoBrainError from your own precondition checks
    - the unchanged InverseProblem + Inverter pipeline on custom physics

Expected runtime: < 60 s.

Outputs:
    out/07_custom_operator.png: rays over the true slowness, the
    recovery, the travel-time fit and convergence.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_VELOCITY,
    C_OBSERVED,
    C_PREDICTED,
    C_SERIES,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    GeoBrainError,
    ModelState,
    PropertyTransform,
)
from geobrain.inverse import GaussianLikelihood, InverseProblem
from geobrain.mesh import StructuredMesh, TensorMesh, UnstructuredMesh
from geobrain.optim import LBFGSConfig
from geobrain.optim.regularizers import smoothness

apply_style()
torch.manual_seed(3)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. Extension point one: a PropertyTransform ---------------------------
class LogSlowness(PropertyTransform):
    """log_s -> slowness. Positivity by construction, gradients included."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("log_s",),
        output_keys=("slowness",),
    )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (log_s,) = state.fetch("log_s")
        return state.with_tensors(slowness=log_s.exp())

# %% 2. Extension point two: a ForwardOperator -----------------------------
class StraightRayTomography(ForwardOperator):
    """Cross-well travel times: ``t_i = Σ_c L[i, c] · slowness[c]``.

    The ray matrix L (path length of ray i in cell c) is geometry, built
    once against the context mesh; the forward is one differentiable
    matmul. Declaring StructuredMesh is what lets the base class refuse a
    mesh with no index grid before this code ever runs.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("slowness",),
        output_keys=("travel_time",),
    )
    requires_mesh_capabilities = (StructuredMesh,)   # we read mesh.shape

    def __init__(self, src_depths: torch.Tensor, rcv_depths: torch.Tensor) -> None:
        super().__init__()
        self._src = src_depths
        self._rcv = rcv_depths
        self._L: torch.Tensor | None = None          # built lazily, once

    def _ray_matrix(self, mesh) -> torch.Tensor:
        nz, nx = mesh.shape
        dz, dx = mesh.spacing
        width = nx * dx
        n_samp = 4 * nx                              # dense sampling per ray
        ts = (torch.arange(n_samp, dtype=torch.float64) + 0.5) / n_samp
        rows = []
        for zs in self._src:                         # left borehole
            for zr in self._rcv:                     # right borehole
                zpath = zs + (zr - zs) * ts
                seg = (width**2 + (zr - zs) ** 2).sqrt() / n_samp
                iz = (zpath / dz).long().clamp(0, nz - 1)
                ix = (ts * width / dx).long().clamp(0, nx - 1)
                row = torch.zeros(nz * nx, dtype=torch.float64)
                row.index_add_(0, iz * nx + ix, seg.expand(n_samp).clone())
                rows.append(row)
        return torch.stack(rows)                     # (n_rays, n_cells)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        mesh = ctx.require_mesh()
        (slowness,) = state.fetch("slowness")
        # your own precondition, raised the platform's way so callers can
        # branch on it exactly like they branch on built-in failures
        if slowness.shape != tuple(mesh.shape):
            raise GeoBrainError(
                "slowness must match the context mesh",
                object_name=type(self).__name__, field="slowness",
                expected=tuple(mesh.shape), actual=tuple(slowness.shape),
                hint="project the model onto the physics mesh first",
            )
        if self._L is None:
            self._L = self._ray_matrix(mesh)
        return ForwardOutput(data={"travel_time": self._L @ slowness.reshape(-1)},
                             metadata={"units": {"travel_time": "s"}})

# %% 3. Both gates, demonstrated -------------------------------------------
mesh = TensorMesh(shape=(20, 16), spacing=(10.0, 10.0))    # 200 x 160 m panel
depths = torch.linspace(5.0, 195.0, 20, dtype=torch.float64)
tomo = StraightRayTomography(src_depths=depths, rcv_depths=depths)
forward = tomo @ LogSlowness()
ctx = ForwardContext.of(mesh=mesh)
print(f"[3] {forward}")
print(f"    contract: {forward.differentiability.trainable_inputs} -> "
      f"{forward.differentiability.output_keys}")

# gate one: the capability check, before any physics
tri_verts = torch.tensor([[0.0, 0.0], [0.0, 160.0], [200.0, 0.0],
                          [200.0, 160.0]], dtype=torch.float64)
triangulation = UnstructuredMesh.from_polygons(
    tri_verts, torch.tensor([[0, 1, 3], [0, 3, 2]], dtype=torch.long))
try:
    forward(ModelState(tensors={"log_s": torch.zeros(mesh.shape,
                                                     dtype=torch.float64)}),
            ForwardContext.of(mesh=triangulation))
except GeoBrainError as err:
    print(f"    capability gate: {err.args[0].splitlines()[0][:80]}")

# gate two: your own precondition
try:
    tomo(ModelState(tensors={"slowness": torch.ones(5, dtype=torch.float64)}),
         ctx)
except GeoBrainError as err:
    print(f"    your own check:  field={err.field!r} expected={err.expected} "
          f"actual={err.actual}")

# %% 4. Simulate a cross-well experiment -----------------------------------
slow_true = torch.full(mesh.shape, 1.0 / 2000.0, dtype=torch.float64)
slow_true[7:13, 5:11] = 1.0 / 3200.0                       # a fast inclusion
(t_clean,) = forward(ModelState(tensors={"log_s": slow_true.log()}),
                     ctx).fetch("travel_time")
noise = 0.002 * float(t_clean.mean())
t_obs = t_clean + noise * torch.randn_like(t_clean)
print(f"[4] {t_obs.numel()} rays ({len(depths)} sources x {len(depths)} "
      f"receivers), noise σ = {noise * 1e3:.3f} ms")

# %% 5. The stack does not care that you wrote this ------------------------
problem = InverseProblem(forward=forward, observed={"travel_time": t_obs},
                         likelihood=GaussianLikelihood(std=noise))
result = problem.create_inverter(
    params={"log_s": torch.full(mesh.shape, float(torch.tensor(1 / 2300.0).log()),
                                dtype=torch.float64)},
    optimizer=LBFGSConfig(lr=0.6, max_iter=20),
    regularizer=lambda p: smoothness(p["log_s"], dx=10.0, dz=10.0, weight=3e2),
    # bounds work on your operator exactly as on the shipped ones:
    # 1500-4000 m/s, expressed in the log-slowness the transform exposes
    bounds={"log_s": (float(torch.tensor(1 / 4000.0).log()),
                      float(torch.tensor(1 / 1500.0).log()))},
    ctx=ctx,
).run(n_iters=25)
v_rec = 1.0 / result.best_params["log_s"].detach().exp()
print(f"[5] recovered velocity range [{float(v_rec.min()):.0f}, "
      f"{float(v_rec.max()):.0f}] m/s  (truth 2000 / 3200)")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(2, 2, scale=1.05)
extent = (0.0, 160.0, 200.0, 0.0)
v_true = 1.0 / slow_true
vlim = (float(v_true.min()), float(v_true.max()))

im = field(axes[0, 0], v_true.numpy(), extent=extent, cmap=CMAP_VELOCITY,
           vmin=vlim[0], vmax=vlim[1],
           title="True velocity and ray coverage", xlabel="Distance [m]",
           ylabel="Depth [m]")
for zs in depths[::2]:
    for zr in depths[::3]:
        axes[0, 0].plot([0.0, 160.0], [float(zs), float(zr)], color="white",
                        lw=0.35, alpha=0.55)

field(axes[0, 1], v_rec.numpy(), extent=extent, cmap=CMAP_VELOCITY,
      vmin=vlim[0], vmax=vlim[1],
      title="Recovered",
      xlabel="Distance [m]")
shared_colorbar(fig, im, axes[0, :], "vp [m/s]")

(t_fit,) = forward(ModelState(tensors={"log_s": (1.0 / v_rec).log()}),
                   ctx).fetch("travel_time")
axes[1, 0].plot((t_obs * 1e3).numpy(), ".", ms=4, color=C_OBSERVED,
                label="Observed")
axes[1, 0].plot((t_fit.detach() * 1e3).numpy(), lw=1.4, color=C_PREDICTED,
                label="Predicted")
axes[1, 0].set(title="Travel-time fit", xlabel="Ray index", ylabel="Time [ms]")
axes[1, 0].legend()

axes[1, 1].semilogy(result.loss_history, color=C_SERIES, lw=2.0)
axes[1, 1].set(title="Convergence", xlabel="Evaluation", ylabel="Objective")
axes[1, 1].grid(which="both")

fig.savefig(OUT / "07_custom_operator.png")
print(f"saved {OUT / '07_custom_operator.png'}")
plt.show()
