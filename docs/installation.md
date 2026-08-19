# Installation

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| PyTorch | 2.0 or newer |
| NumPy | 1.22 or newer |
| SciPy | 1.8 or newer |
| Matplotlib | 3.5 or newer |

GeoBrain runs on CPU. Everything in the [examples gallery](examples.md) is
sized to finish on one, and nothing in the platform requires a GPU. Where CUDA
is available, operators that support it will use it.

## From a checkout

```bash
git clone https://github.com/GeoBrain-Project/GeoBrain.git
cd GeoBrain
pip install -e ".[examples]"
```

## What the extras buy

`import geobrain` needs only `torch`, `numpy`, `scipy` and `jsonschema`. Every
other dependency is optional and named after the subpackage that cannot be
imported without it, so an install can stay as small as the work requires.

| Extra | Pulls in | Needed for |
|---|---|---|
| `vis` | matplotlib | `geobrain.vis`, which raises with an install hint without it |
| `io` | h5py, segyio, lasio, meshio | the SEG-Y, LAS, HDF5 and VTK paths in `geobrain.io` |
| `viewer` | pyvista, plotly, ipython | `Scene3D` and the `view_*` helpers |
| `mesh` | triangle | triangulating an `UnstructuredMesh` from points |
| `parallel` | threadpoolctl | thread pinning in the geostatistical simulators |
| `examples` | vis, io, mesh, parallel | running the gallery end to end |
| `all` | everything above | |

## Checking it works

```python
import geobrain
print(geobrain.__version__)
```

A better check is that a forward model builds and differentiates, which is one
import and four lines:

```python
import torch
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D

mesh = TensorMesh(shape=(8, 16), spacing=(25.0, 25.0))
stations = torch.stack([torch.linspace(0.0, 400.0, 12, dtype=torch.float64),
                        torch.ones(12, dtype=torch.float64)], dim=1)
gravity = Gravity2D(PotentialSurvey2D(stations))

rho = torch.full((8, 16), -200.0, dtype=torch.float64, requires_grad=True)
gz = gravity(ModelState({"rho": rho}), ForwardContext.of(mesh=mesh)).data["gz"]
gz.pow(2).sum().backward()
print(gz.shape, rho.grad.abs().max().item())
```

On a good install that prints `torch.Size([12]) 1.8012725035714458e-13`. The
gradient is tiny because `gz` is in SI (metres per second squared, not mGal),
so what matters is that it is finite and non-zero: the forward model ran and
the derivative came back through it.

## Example data

Everything in the gallery generates its own earth from a seed except the
seismic pair, which reads a window of the **Marmousi II** benchmark. Those
sections are 148 MB each, past what a git host accepts in one file, so they
are published as release assets and fetched on demand:

```bash
python examples/data/fetch_marmousi.py
```

Each section is checked against a recorded SHA-256 as it lands. That matters
more than it sounds: a truncated SEG-Y does not fail loudly, it reads back as
a shorter section and the inversion runs on the wrong earth. Pass `--all` for
the shear and density sections, which `marmousi(..., fields=("vs", "rho"))`
can return for work of your own, and `--url-base` to point at a copy you
already have.

## Building these docs

```bash
pip install sphinx sphinx-book-theme myst-parser sphinx-design sphinx-copybutton
sphinx-build -b html docs docs/_build/html
```
