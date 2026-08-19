"""The mesh taxonomy: four discretizations, one set of capabilities.

Geometry in GeoBrain is data, not a base class you inherit from. A mesh
is whatever can answer the questions a physics operator asks, and it
DECLARES which questions those are:

- ``UniformMesh``:         constant cell size (spectral methods, FDTD)
- ``StructuredMesh``:      an (nz, nx[, ny]) index grid exists
- ``PrismGeometryMesh``:   cells are axis-aligned boxes (Talwani kernels)
- ``GeometryMesh``:        cells have centres and volumes, nothing more
- ``ConnectivityMesh``:    cells know their face neighbours

An operator lists what it needs in ``requires_mesh_capabilities`` and the
base class refuses a mesh that cannot answer, before any physics runs.
That is why the same ``Gravity2D`` accepts a tensor grid and rejects a
triangulation, and why a hand-written line-mass kernel accepts both.

This script builds the same subsurface feature on four meshes (a uniform
tensor grid, a boundary-graded tensor grid, an adaptively refined octree
and an irregular triangulation), prints the capability matrix, and then
carries one field across all four with :class:`MeshProjection`, measuring
what each discretization costs in round-trip accuracy.

APIs featured:
    - geobrain.mesh.TensorMesh (spacing= and cell_widths= forms)
    - geobrain.mesh.OctreeMesh.from_tensor_mesh (+ refine_batch_fn)
    - geobrain.mesh.UnstructuredMesh.from_polygons
    - Mesh.declares(capability): the capability matrix
    - geobrain.mesh.MeshProjection (any mesh -> any mesh, differentiable)

Expected runtime: < 20 s.

Outputs:
    out/02_mesh_taxonomy.png: one field on four meshes, the capability
    matrix, and the round-trip error each discretization costs.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import ListedColormap

from _style import (
    CMAP_VELOCITY,
    PALETTE,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import ModelState
from geobrain.mesh import (
    ConnectivityMesh,
    GeometryMesh,
    MeshProjection,
    OctreeMesh,
    PrismGeometryMesh,
    StructuredMesh,
    TensorMesh,
    UniformMesh,
    UnstructuredMesh,
)

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

LZ, LX = 600.0, 1200.0                 # the physical domain, shared by all

# %% 1. The axis convention, checked with pixels ---------------------------
#
# shape = (nz, nx): the FIRST index walks depth (positive DOWN), the
# second walks x. Everything downstream (surveys, projections, plots)
# inherits this, so it is worth proving rather than asserting.
probe_mesh = TensorMesh(shape=(24, 48), spacing=(25.0, 25.0))
probe = torch.zeros(probe_mesh.shape)
probe[2, 40] = 1.0                     # shallow, far right
centres = probe_mesh.cell_centers()    # (n_cells, 2) as (z, x)
hit = centres[probe.reshape(-1).argmax()]
print(f"[1] axis convention: field[2, 40] sits at (z, x) = "
      f"({float(hit[0]):.1f}, {float(hit[1]):.1f}) m  -> shallow and far right")

# %% 2. Four meshes over the same domain -----------------------------------
uniform = TensorMesh(shape=(24, 48), spacing=(25.0, 25.0))

wz = torch.cat([                                  # 3 + 19 + 3 = 600 m
    torch.tensor([50.0, 35.0, 25.0], dtype=torch.float64),
    torch.full((19,), 20.0, dtype=torch.float64),
    torch.tensor([25.0, 35.0, 50.0], dtype=torch.float64),
])
wx = torch.cat([                                  # 3 + 28 + 3 = 1200 m
    torch.tensor([80.0, 60.0, 40.0], dtype=torch.float64),
    torch.full((28,), 30.0, dtype=torch.float64),
    torch.tensor([40.0, 60.0, 80.0], dtype=torch.float64),
])
graded = TensorMesh(shape=(len(wz), len(wx)), cell_widths=(wz, wx))

TARGET = torch.tensor([260.0, 620.0])          # (z, x) of the feature

def near_target(centers: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    return (centers - TARGET).norm(dim=1) < 220.0

octree_base = TensorMesh(shape=(12, 24), spacing=(50.0, 50.0))
octree = OctreeMesh.from_tensor_mesh(octree_base, refine_batch_fn=near_target,
                                     max_level=2)

zl = torch.cat([torch.arange(0.0, 160.0, 55.0),
                torch.arange(160.0, 380.0, 28.0),
                torch.arange(380.0, LZ, 55.0), torch.tensor([LZ])])
xl = torch.cat([torch.arange(0.0, 420.0, 90.0),
                torch.arange(420.0, 840.0, 45.0),
                torch.arange(840.0, LX, 90.0), torch.tensor([LX])])
verts = torch.stack(torch.meshgrid(zl.to(torch.float64), xl.to(torch.float64),
                                   indexing="ij"), dim=-1).reshape(-1, 2).clone()
nxv = len(xl)
gen = torch.Generator().manual_seed(1)
for iz in range(1, len(zl) - 1):
    for ix in range(1, nxv - 1):
        verts[iz * nxv + ix] += (torch.rand(2, generator=gen,
                                            dtype=torch.float64) - 0.5) * 14.0
tris = []
for iz in range(len(zl) - 1):
    for ix in range(nxv - 1):
        a, b = iz * nxv + ix, iz * nxv + ix + 1
        c, d = (iz + 1) * nxv + ix, (iz + 1) * nxv + ix + 1
        tris += [[a, b, d], [a, d, c]]
triangles = torch.tensor(tris, dtype=torch.long)
unstructured = UnstructuredMesh.from_polygons(verts, triangles)

meshes = {"Uniform tensor": uniform, "Graded tensor": graded,
          "Octree": octree, "Unstructured": unstructured}
for name, m in meshes.items():
    print(f"[2] {name:16s} {m.n_cells:5d} cells")

# %% 3. The capability matrix ----------------------------------------------
#
# This is the table an operator's requires_mesh_capabilities is checked
# against. Note what the octree and the triangulation give up: no shape,
# no spacing, no boxes, only geometry and connectivity.
CAPS = (("UniformMesh", UniformMesh), ("StructuredMesh", StructuredMesh),
        ("PrismGeometryMesh", PrismGeometryMesh), ("GeometryMesh", GeometryMesh),
        ("ConnectivityMesh", ConnectivityMesh))
matrix = torch.tensor([[float(m.declares(cap)) for _, cap in CAPS]
                       for m in meshes.values()])
print("[3] capability matrix")
for (name, _), row in zip(meshes.items(), matrix):
    yes = [cap for (cap, _), v in zip(CAPS, row) if v > 0]
    print(f"    {name:16s} declares {yes}")

# %% 4. One field, carried across all four ---------------------------------
#
# MeshProjection is geometry-first: give it any source and any target and
# it builds the differentiable transfer for that pair. Round-tripping a
# field back to the reference grid measures what each mesh costs.
def analytic(centers: torch.Tensor) -> torch.Tensor:
    zc, xc = centers[:, 0], centers[:, 1]
    trend = 1900.0 + 0.9 * zc
    blob = 480.0 * torch.exp(-(((zc - TARGET[0]) / 90.0) ** 2
                               + ((xc - TARGET[1]) / 150.0) ** 2))
    return trend + blob

ref = analytic(uniform.cell_centers()).reshape(uniform.shape)
carried, errors = {}, {}
for name, m in meshes.items():
    if m is uniform:
        carried[name], errors[name] = ref, 0.0
        continue
    # the projection reports when it has to extrapolate at the boundary or
    # when the target is a coarsening, worth reading rather than silencing
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (field_m,) = MeshProjection(uniform, m, field_name="vp",
                                    padding="border")(
            ModelState(tensors={"vp": ref})).fetch("vp")
        (back,) = MeshProjection(m, uniform, field_name="vp",
                                 padding="border")(
            ModelState(tensors={"vp": field_m})).fetch("vp")
    carried[name] = field_m
    errors[name] = float((back - ref).abs().mean())
    notes = {str(w.message).split(":")[1].strip().split("(")[0].strip()
             for w in caught}
    print(f"[4] {name:16s} round trip: mean |error| = {errors[name]:6.2f} m/s"
          + (f"   [projection noted: {'; '.join(sorted(notes))}]"
             if notes else ""))

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
extent = (0.0, LX, LZ, 0.0)
vmin, vmax = float(ref.min()), float(ref.max())

def grid_lines(ax, widths_z, widths_x):
    zn = torch.cat([torch.zeros(1), torch.cumsum(widths_z, 0)])
    xn = torch.cat([torch.zeros(1), torch.cumsum(widths_x, 0)])
    for v in zn:
        ax.axhline(float(v), color="0.35", lw=0.35, alpha=0.6)
    for v in xn:
        ax.axvline(float(v), color="0.35", lw=0.35, alpha=0.6)

# One field, four discretisations, ONE colour bar: the panels differ in
# their cells, not in what they hold, and four bars would say otherwise.
im = field(axes[0, 0], carried["Uniform tensor"].numpy(), extent=extent,
           cmap=CMAP_VELOCITY, vmin=vmin, vmax=vmax,
           title=f"Uniform tensor - {uniform.n_cells} cells",
           ylabel="Depth [m]")
grid_lines(axes[0, 0], torch.full((24,), 25.0), torch.full((48,), 25.0))

field(axes[0, 1], carried["Graded tensor"].numpy(), extent=extent,
      cmap=CMAP_VELOCITY, vmin=vmin, vmax=vmax,
      title=f"Graded tensor - {graded.n_cells} cells")
grid_lines(axes[0, 1], wz, wx)

oc, ow = octree.cell_centers(), octree.cell_widths()
rects = [plt.Rectangle((float(c[1] - w[1] / 2), float(c[0] - w[0] / 2)),
                       float(w[1]), float(w[0])) for c, w in zip(oc, ow)]
pc = PatchCollection(rects, cmap=CMAP_VELOCITY, edgecolor="0.35", lw=0.35)
pc.set_array(carried["Octree"].numpy())
pc.set_clim(vmin, vmax)
axes[0, 2].add_collection(pc)
axes[0, 2].set_xlim(0.0, LX)
axes[0, 2].set_ylim(LZ, 0.0)
axes[0, 2].grid(False)
axes[0, 2].set(title=f"Octree - {octree.n_cells} leaves from a "
                     f"{octree_base.n_cells}-cell base")

tri_xy = verts[triangles][:, :, [1, 0]].numpy()
tpc = PolyCollection(tri_xy, cmap=CMAP_VELOCITY, edgecolors="0.35",
                     linewidths=0.3)
tpc.set_array(carried["Unstructured"].numpy())
tpc.set_clim(vmin, vmax)
axes[1, 0].add_collection(tpc)
axes[1, 0].set_xlim(0.0, LX)
axes[1, 0].set_ylim(LZ, 0.0)
axes[1, 0].grid(False)
axes[1, 0].set(title=f"Unstructured - {unstructured.n_cells} triangles",
               xlabel="Distance [m]", ylabel="Depth [m]")
shared_colorbar(fig, im, (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]),
                "vp [m/s]")

# A capability table is yes/no, so it gets two flat colours rather than a
# ramp a reader might try to read a value off.
declared = ListedColormap(["0.93", PALETTE[0]])
field(axes[1, 1], matrix.numpy(), cmap=declared, vmin=0.0, vmax=1.0,
      aspect="auto")
axes[1, 1].set_xticks(range(len(CAPS)), [c for c, _ in CAPS], rotation=35,
                      ha="right")
axes[1, 1].set_yticks(range(len(meshes)), list(meshes))
for i in range(len(meshes)):
    for j in range(len(CAPS)):
        axes[1, 1].text(j, i, "✓" if matrix[i, j] > 0 else "·",
                        ha="center", va="center", fontsize=13,
                        color="white" if matrix[i, j] > 0 else "gray")
axes[1, 1].set(title="What each mesh declares")

names = [n for n in meshes if n != "Uniform tensor"]
axes[1, 2].barh(range(len(names)), [errors[n] for n in names],
                color=PALETTE[0], alpha=0.9)
axes[1, 2].set_yticks(range(len(names)), names)
axes[1, 2].invert_yaxis()
axes[1, 2].set(title="Round-trip cost through MeshProjection",
               xlabel="Mean |error| [m/s]")

fig.savefig(OUT / "02_mesh_taxonomy.png")
print(f"saved {OUT / '02_mesh_taxonomy.png'}")
plt.show()
