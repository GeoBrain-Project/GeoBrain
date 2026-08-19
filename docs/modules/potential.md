# Potential fields (`geobrain.physics.potential`)

Gravity and magnetics: the cheapest data in geophysics, and the least unique.

| Operator | Gives |
|---|---|
| `Gravity2D` | `gz` from a 2-D density-contrast section (Talwani) |
| `Gravity3D` | `gz` and the full gradient tensor, from prisms |
| `Magnetic3D` | total-field anomaly, with induced **and remanent** magnetisation |
| `MagneticVector3D` | the vector field, component by component |

Surveys are `PotentialSurvey2D` / `PotentialSurvey3D`. `EarthField` carries the
inducing field's strength and direction, which is what makes a magnetic anomaly
depend on where on the planet you are standing.

```python
import torch
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.physics.potential import (Gravity2D, PotentialSurvey2D,
                                        gravity_to_mgal)

mesh = TensorMesh(shape=(10, 20), spacing=(25.0, 25.0))
stations = torch.stack([torch.linspace(0.0, 500.0, 21, dtype=torch.float64),
                        torch.ones(21, dtype=torch.float64)], dim=1)
gravity = Gravity2D(PotentialSurvey2D(stations))

rho = torch.zeros(10, 20, dtype=torch.float64)
rho[3:6, 8:13] = -400.0                      # a light body
gz = gravity(ModelState({"rho": rho}), ForwardContext.of(mesh=mesh)).data["gz"]
print(f"peak anomaly {float(gravity_to_mgal(gz).abs().max()):.3f} mGal")
```

```text
peak anomaly 0.409 mGal
```

```{warning}
These operators are strict about `dtype=torch.float64`. A float32 survey is
rejected rather than quietly losing precision in a kernel that sums a
contribution from every cell in the model.
```

## Why gravity needs help

A gravity anomaly does not determine a density distribution: many earths fit
the same curve, and the smooth ones fit it best. Depth weighting
(`geobrain.optim.depth_weighting`) stops everything migrating to the surface;
an IRLS sparsity pass sharpens the blur; and a second physics, magnetics, with
a **different** sensitivity to the same rock, is what actually breaks the tie.
Hence the joint example.

```{figure} /_figures/06_potential_fields.png
:class: gb-tall
:alt: One body seen by gravity and by magnetics

One buried body, four fields: gravity, induced magnetisation at two
inclinations, and a remanent case. From
`examples/03_physics/06_potential_fields.py`.
```

## Processing and interpretation

`upward_continue`, `reduce_to_pole`, `vertical_derivative`,
`horizontal_gradient`, `tilt_derivative`, `analytic_signal_amplitude` and
`regional_residual_lowpass` are the standard filters. `bouguer_slab`,
`bouguer_spherical_cap`, `free_air_correction` and `terrain_correction_prism`
are the reductions. `euler_deconvolution` estimates source depth from a
structural index.

Units are SI inside and converted at the edge: `gravity_to_mgal`,
`magnetic_to_nt`, `gravity_gradient_to_eotvos`.

## See also

- `examples/00_showcase/01_gravity_inversion.py`: the whole platform in three
  objects, on a gravity problem.
- `examples/03_physics/06_potential_fields.py`: why magnetics answers
  differently at every latitude and for every remanence.
- `examples/03_physics/04_petrophysical_joint_inversion.py`: gravity and
  magnetics together, answering which rock is where rather than how dense it is.
