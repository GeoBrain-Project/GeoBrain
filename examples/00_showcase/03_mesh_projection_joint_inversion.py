"""Differentiable mesh projection: the bridge that makes joint inversion work.

A joint inversion in which each physics uses the mesh KIND it actually
needs. The model lives on an :class:`UnstructuredMesh` of jittered
triangles refined along a DIPPING target band, built with
``from_polygons``. The wave equation insists on a regular grid, so the
seismic chain crosses over through a differentiable
:class:`MeshProjection`. Gravity insists on nothing: its kernel only
needs cell centres and areas, exactly what the GeometryMesh capability
provides, so a ~20-line operator evaluates it STRAIGHT on the
triangles, no projection, no structure::

    seismic = Helmholtz2D(survey) @ MeshProjection(triangles, 25 m grid, "vp")
    gravity = TriangleGravity2D(triangles, stations) @ Gardner()

One joint backward pass: the seismic residual flows through the
Helmholtz implicit adjoint and the projection; the gravity residual
flows through Gardner directly into the triangle values. The regularizer
never sees a grid either, since it is built from ``face_neighbors()``
connectivity records.

APIs featured:
    - geobrain.mesh.UnstructuredMesh.from_polygons (triangles as cells)
    - a capability-generic ForwardOperator: any mesh answering
      cell_centers()/cell_volumes() can carry gravity
    - MeshProjection bridging ONLY where the physics demands structure
    - Mesh.face_neighbors() records -> graph smoothness in three lines

Expected runtime: < 3 min.

Outputs:
    out/03_mesh_projection_joint_inversion.png: the truth on the
    triangles, its projection to the structured grid the wave equation
    needs, and the joint recovery back on the triangles.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection

from _style import (
    CMAP_VELOCITY,
    C_RECOVERED,
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
    ModelState,
)
from geobrain.mesh import MeshProjection, TensorMesh, UnstructuredMesh
from geobrain.physics.rock.models import GardnerOperator
from geobrain.physics.wave import (
    Helmholtz2D,
    Helmholtz2DReceiver,
    Helmholtz2DSource,
    Helmholtz2DSurvey,
)

apply_style()
torch.manual_seed(3)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. A triangular model mesh that hugs a dipping target ------------------
#
# Vertex lines are packed tight across the band the geology occupies and
# stretched elsewhere; interior vertices are jittered so the triangles
# are genuinely irregular.
def graded(fine_lo: float, fine_hi: float, hi: float, d_fine: float,
           d_coarse: float) -> torch.Tensor:
    pre = torch.arange(0.0, fine_lo, d_coarse)
    core = torch.arange(fine_lo, fine_hi, d_fine)
    post = torch.arange(fine_hi, hi - 1.0, d_coarse)
    return torch.cat([pre, core, post,
                      torch.tensor([hi])]).to(torch.float64)

z_lines = graded(80.0, 340.0, 600.0, 22.0, 80.0)
x_lines = graded(330.0, 880.0, 1200.0, 40.0, 110.0)
nz_v, nx_v = len(z_lines), len(x_lines)
verts = torch.stack(torch.meshgrid(z_lines, x_lines, indexing="ij"),
                    dim=-1).reshape(-1, 2).clone()
gen = torch.Generator().manual_seed(0)
for iz in range(1, nz_v - 1):                    # jitter interior vertices
    dz_loc = float(min(z_lines[iz] - z_lines[iz - 1], z_lines[iz + 1] - z_lines[iz]))
    for ix in range(1, nx_v - 1):
        dx_loc = float(min(x_lines[ix] - x_lines[ix - 1],
                           x_lines[ix + 1] - x_lines[ix]))
        j = (torch.rand(2, generator=gen, dtype=torch.float64) - 0.5)
        verts[iz * nx_v + ix] += j * 0.5 * torch.tensor([dz_loc, dx_loc])

tris = []
for iz in range(nz_v - 1):
    for ix in range(nx_v - 1):
        v00, v01 = iz * nx_v + ix, iz * nx_v + ix + 1
        v10, v11 = (iz + 1) * nx_v + ix, (iz + 1) * nx_v + ix + 1
        tris.append([v00, v01, v11])
        tris.append([v00, v11, v10])
triangles = torch.tensor(tris, dtype=torch.long)
model = UnstructuredMesh.from_polygons(verts, triangles)
centers = model.cell_centers()
print(f"unstructured model mesh: {model.n_cells} triangles "
      f"(declares no shape, no spacing)")

# %% 2. Truth: a dipping fast, dense slab ----------------------------------
def in_slab(c: torch.Tensor) -> torch.Tensor:
    s = (c[:, 1] - 420.0) / 360.0                # 0..1 along the slab
    z_mid = 130.0 + s * 160.0                    # dipping from 130 to 290 m
    return (s >= 0.0) & (s <= 1.0) & ((c[:, 0] - z_mid).abs() < 55.0)

vp_bg_cells = 1900.0 + 0.8 * centers[:, 0]
vp_true = vp_bg_cells.clone()
vp_true[in_slab(centers)] += 550.0

# %% 3. Seismic needs structure; gravity does not --------------------------
#
# Gravity on ANY mesh in ~20 lines: each cell contributes as a 2-D line
# mass at its centre, so the only geometry the kernel asks of the mesh is
# cell_centers() and cell_volumes(), which every GeoBrain mesh answers.
class TriangleGravity2D(ForwardOperator):
    """gz (elevation-up) at surface stations from per-cell rho."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("rho",),
        output_keys=("gz",),
    )

    def __init__(self, mesh, station_x: torch.Tensor,
                 station_elev: float = 1.0) -> None:
        super().__init__()
        c = mesh.cell_centers()                  # (n_cells, 2) as (z, x)
        area = mesh.cell_volumes()               # (n_cells,) m^2 in 2-D
        dz = c[:, 0][None, :] + station_elev     # station sits ABOVE ground
        dx = c[:, 1][None, :] - station_x[:, None]
        r2 = dz**2 + dx**2
        G = 6.674e-11
        self.register_buffer("kernel", -2.0 * G * area[None, :] * dz / r2,
                             persistent=False)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (rho,) = state.fetch("rho")
        return ForwardOutput(data={"gz": self.kernel @ rho},
                             metadata={"units": {"gz": "m/s^2"}})

seis_mesh = TensorMesh(shape=(24, 48), spacing=(25.0, 25.0))

N_SHOT, N_RCV = 4, 24
FREQS = (6.0, 10.0, 14.0)
src_x = torch.linspace(0.12, 0.88, N_SHOT) * 1200.0
rcv_x = torch.linspace(0.04, 0.96, N_RCV) * 1200.0
helm = Helmholtz2D(Helmholtz2DSurvey(
    sources=tuple(Helmholtz2DSource(position=(float(x), 30.0),
                                    amplitude=1.0 + 0.0j, shot_id=s)
                  for s, x in enumerate(src_x)),
    receivers=tuple(Helmholtz2DReceiver(position=(float(x), 30.0), shot_id=s)
                    for s in range(N_SHOT) for x in rcv_x),
    frequencies=FREQS, n_pml=10,
))
seismic = helm @ MeshProjection(model, seis_mesh, field_name="vp",
                                padding="border")

xs = torch.linspace(15.0, 1185.0, 32, dtype=torch.float64)
gravity = TriangleGravity2D(model, xs) @ GardnerOperator()
print(f"seismic chain: {seismic}   (structured, via projection)")
print(f"gravity chain: {gravity}   (STRAIGHT on the triangles)")

ctx_s = ForwardContext.of(mesh=seis_mesh)
ctx_g = ForwardContext()                        # gravity carries its geometry

with torch.no_grad():
    (p_obs,) = seismic(ModelState(tensors={"vp": vp_true}), ctx_s).fetch("p")
    (gz_obs,) = gravity(ModelState(tensors={"vp": vp_true}), ctx_g).fetch("gz")
p_obs = p_obs[..., 0].reshape(N_SHOT, N_RCV, len(FREQS))
p_obs = p_obs + 0.02 * p_obs.abs().mean() * (
    torch.randn_like(p_obs.real) + 1j * torch.randn_like(p_obs.real))
gz_obs = gz_obs + 0.02e-5 * torch.randn_like(gz_obs)   # 0.02 mGal instrument
print(f"observed: seismic {tuple(p_obs.shape)} on the 25 m grid, "
      f"gravity {tuple(gz_obs.shape)} on the triangles themselves")

# %% 4. Joint inversion ON THE TRIANGLES -----------------------------------
#
# The smoothness penalty comes straight from the mesh's connectivity:
# every internal face contributes (length / centre-distance) x the
# squared jump of the update across it.
faces = model.face_neighbors()
w_face = faces.area / (faces.dist_l + faces.dist_r)

def graph_smoothness(d: torch.Tensor) -> torch.Tensor:
    return (w_face * (d[faces.cell_i] - d[faces.cell_j]).pow(2)).sum()

log_v0 = vp_bg_cells.log()
log_v = log_v0.clone().requires_grad_(True)
hist_s, hist_g = [], []

def objective() -> torch.Tensor:
    vp_model = log_v.exp()
    (p,) = seismic(ModelState(tensors={"vp": vp_model}), ctx_s).fetch("p")
    p = p[..., 0].reshape(N_SHOT, N_RCV, len(FREQS))
    seis_term = ((p - p_obs).abs().pow(2).sum(dim=(1, 2))
                 / p_obs.abs().pow(2).sum(dim=(1, 2))).sum()
    (gz,) = gravity(ModelState(tensors={"vp": vp_model}), ctx_g).fetch("gz")
    grav_term = 100.0 * (gz - gz_obs).pow(2).sum() / gz_obs.pow(2).sum()
    hist_s.append(float(seis_term))
    hist_g.append(float(grav_term))
    return seis_term + grav_term + 5e-4 * graph_smoothness(log_v - log_v0)

adam = torch.optim.Adam([log_v], lr=0.01)
for _ in range(80):
    adam.zero_grad()
    objective().backward()
    adam.step()
stage2_at = len(hist_s)
lbfgs = torch.optim.LBFGS([log_v], lr=0.8, max_iter=90, history_size=25,
                          line_search_fn="strong_wolfe")

def closure() -> torch.Tensor:
    lbfgs.zero_grad()
    loss = objective()
    loss.backward()
    return loss

lbfgs.step(closure)
vp_rec = log_v.exp().detach()
print(f"joint inversion on {model.n_cells} triangles: misfit "
      f"{hist_s[0] + hist_g[0]:.3f} -> {hist_s[-1] + hist_g[-1]:.5f}; "
      f"max update {float((vp_rec - vp_bg_cells).max()):.0f} m/s (truth 550)")
print(f"  ONE backward, two mesh kinds: seismic (25 m grid, projected) "
      f"{hist_s[0]:.3f} -> {hist_s[-1]:.5f}, gravity (straight on the "
      f"triangles) {hist_g[0]:.3f} -> {hist_g[-1]:.5f}, over "
      f"{len(hist_s)} evaluations with L-BFGS taking over at {stage2_at}")

# %% 5. Picture ------------------------------------------------------------
tri_xy = verts[triangles][:, :, [1, 0]].numpy()   # (n_tri, 3, 2) as (x, z)

def tri_panel(ax, values, cmap, vmin=None, vmax=None, lw=0.0):
    pc = PolyCollection(tri_xy, cmap=cmap, edgecolors="0.35", linewidths=lw)
    pc.set_array(values)
    if vmin is not None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(0.0, 1200.0)
    ax.set_ylim(600.0, 0.0)
    ax.grid(False)
    return pc

slab_s = torch.linspace(0.0, 1.0, 40)
slab_x = (420.0 + slab_s * 360.0).numpy()
slab_top = (130.0 + slab_s * 160.0 - 55.0).numpy()
slab_bot = (130.0 + slab_s * 160.0 + 55.0).numpy()

def slab_outline(ax, color="white"):
    ax.plot(slab_x, slab_top, ls="--", lw=1.4, color=color)
    ax.plot(slab_x, slab_bot, ls="--", lw=1.4, color=color)

fig, axes = figure(3, 1, panel_w=6.4, panel_h=3.0, sharex=True,
                   sharey=True)
vmin, vmax = float(vp_true.min()), float(vp_true.max())

# The three velocity panels share one scale and therefore one colour bar:
# the whole point is that they are the SAME model seen through different
# discretisations, which three separate bars would quietly deny.
tp = tri_panel(axes[0], vp_true.numpy(), CMAP_VELOCITY, vmin, vmax, lw=0.25)
slab_outline(axes[0])
axes[0].set(title=f"Truth on {model.n_cells} triangles",
            ylabel="Depth [m]")

(vp_grid,) = MeshProjection(model, seis_mesh, field_name="vp",
                            padding="border")(
    ModelState(tensors={"vp": vp_true})).fetch("vp")
field(axes[1], vp_grid.numpy(), extent=(0.0, 1200.0, 600.0, 0.0),
      cmap=CMAP_VELOCITY, vmin=vmin, vmax=vmax, ylabel="Depth [m]",
      title="Projected to SEISMIC's structured 25 m grid")
axes[1].plot(src_x.numpy(), [30.0] * N_SHOT, "*", ms=10, color=C_RECOVERED,
             mec="k")
axes[1].plot(rcv_x.numpy(), [30.0] * N_RCV, "wv", ms=4, mec="k")

tri_panel(axes[2], vp_rec.numpy(), CMAP_VELOCITY, vmin, vmax, lw=0.1)
slab_outline(axes[2])
axes[2].set(title="Joint recovery, on the triangles",
            xlabel="Distance [m]", ylabel="Depth [m]")
shared_colorbar(fig, tp, axes, "vp [m/s]", location="bottom")

fig.savefig(OUT / "03_mesh_projection_joint_inversion.png")
print(f"saved {OUT / '03_mesh_projection_joint_inversion.png'}")
plt.show()
