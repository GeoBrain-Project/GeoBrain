# Mesh (`geobrain.mesh`)

A mesh in GeoBrain carries more than an array shape. It **declares what it can
support**, so physics that needs structure cannot silently be run on something
that has none.

```{figure} /_figures/02_mesh_taxonomy.png
:class: full-width
:alt: One domain on four discretizations

Four discretizations of one domain, each declaring a different set of
capabilities. From `examples/01_architecture/02_mesh_taxonomy.py`.
```

## The four you will use

| Class | Declares | Used for |
|---|---|---|
| `TensorMesh` | uniform or per-axis graded cell widths | finite differences, most physics |
| `OctreeMesh` | refined leaves over a base grid | local refinement without a global cost |
| `UnstructuredMesh` | vertices, cells, face connectivity | geology that does not follow a grid |
| `CylindricalMesh` | r-z-θ geometry | borehole and loop-source problems |

```python
from geobrain.mesh import TensorMesh

mesh = TensorMesh(shape=(24, 48), spacing=(25.0, 25.0))
print(mesh.shape, mesh.n_cells)
```

```text
(24, 48) 1152
```

## Capabilities are protocols

Each mesh satisfies a set of named protocols, and each physics declares which
it requires:

| Protocol | A mesh that satisfies it can offer |
|---|---|
| `UniformMesh` | one cell size everywhere |
| `StructuredMesh` | an (i, j, k) index space |
| `GeometryMesh` | cell centres, volumes |
| `PrismGeometryMesh` | rectangular prism corners, what a potential-field kernel integrates over |
| `ConnectivityMesh` | which cells share a face |
| `EdgeConnectivityMesh` | edges, for curl operators |

Membership is **declared**, never inferred from which methods happen to exist,
and the platform refuses the structural shortcut outright:

```python
import torch
from geobrain.mesh import (PrismGeometryMesh, StructuredMesh, TensorMesh,
                           UniformMesh)

uniform = TensorMesh(shape=(8, 8), spacing=(10.0, 10.0))
graded = TensorMesh(
    shape=(4, 6),
    cell_widths=(torch.tensor([5.0, 10.0, 20.0, 40.0], dtype=torch.float64),
                 torch.full((6,), 10.0, dtype=torch.float64)))

for name, mesh in (("uniform", uniform), ("graded ", graded)):
    print(f"{name}  structured={mesh.declares(StructuredMesh)}"
          f"  prisms={mesh.declares(PrismGeometryMesh)}"
          f"  uniform={mesh.declares(UniformMesh)}")

try:
    isinstance(uniform, StructuredMesh)
except TypeError as err:
    print("isinstance:", err)
```

```text
uniform  structured=True  prisms=True  uniform=True
graded   structured=True  prisms=True  uniform=False
isinstance: capability membership is declaration-based: use
mesh.declares(StructuredMesh) or ensure_capable_mesh(...), not isinstance
```

Two things in that output. Capabilities are refined **per instance**. The
graded mesh is still structured and still made of prisms, but it is not
uniform, because its cells are not all the same size, and physics that needs a
single ``dx`` will be told so. And the structural shortcut is **refused**:
"it has a `cell_centers` attribute, so it must be a `GeometryMesh`" is how a
mesh ends up quietly accepted by physics it cannot actually serve.

`require_capable_mesh` and `ensure_capable_mesh` are how an operator states its
requirement, so the error arrives at build time and names what was missing
rather than producing a wrong answer.

## Discrete operators

`cell_gradient`, `face_divergence`, `edge_curl`, `boundary_divergence`,
`average_cell_to_face`, `average_face_to_cell`, `harmonic_face_values` and
`face_transmissibility` are the building blocks the PDE physics is written in.
They take a mesh and return sparse operators consistent with that mesh's
geometry.

## Projection between meshes

`MeshProjection` maps a field from one mesh to another **differentiably**, so a
gradient computed on one discretisation returns to the other:

```python
import torch
from geobrain.core import ModelState
from geobrain.mesh import MeshProjection, TensorMesh

coarse = TensorMesh(shape=(6, 12), spacing=(50.0, 50.0))
fine = TensorMesh(shape=(12, 24), spacing=(25.0, 25.0))
project = MeshProjection(coarse, fine, field_name="vp", padding="border")

vp = torch.full((6, 12), 2000.0, dtype=torch.float64, requires_grad=True)
vp_fine, = project(ModelState({"vp": vp})).fetch("vp")
vp_fine.sum().backward()
print(vp_fine.shape, "gradient returned:", vp.grad is not None)
```

```text
torch.Size([12, 24]) gradient returned: True
```

A joint inversion can therefore run the wave equation on a structured grid and
gravity straight on triangles, and still get one gradient from one backward
pass.

## Axis convention

Mesh axes are named in the order the arrays are indexed, and
`mesh_axes_to_xyz` / `xyz_to_mesh_axes` convert to and from a right-handed
spatial frame. Getting this wrong transposes the anisotropy into an earth that
still looks plausible, so the conversion is explicit rather than assumed.

## See also

- `examples/01_architecture/02_mesh_taxonomy.py`: four meshes, one domain, and
  the capability matrix that decides what may run on each.
- `examples/00_showcase/03_mesh_projection_joint_inversion.py`: the projection
  carrying a gradient between two mesh kinds.
